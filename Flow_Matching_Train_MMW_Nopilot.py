import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Flow_Model_mmw import MiniUnet
from Fusion_model_mmw import VE_MultimodalityMMW
from mmw_dataset import (
    DEFAULT_CAVS,
    DEFAULT_CHANNEL_CONFIG,
    DEFAULT_CHANNEL_SCENARIO,
    DEFAULT_ROOT,
    DEFAULT_SENSOR_SCENARIO,
    MMVTjunctionDataset,
    collate_with_steering,
    parse_csv_names,
    split_dataset,
    write_json,
)
# -----------------------------------------------------------------------------
# PHYSICS PILOT ADDITION: the residual is kept in a helper module, while ConFIG
# is an optional dependency from the separate PBFM reference implementation.
# The baseline can still import/run when that package is absent; requesting the
# physics pilot will produce a clear dependency error.
# -----------------------------------------------------------------------------
from wireless_physics import steering_subspace_outside_fraction

try:
    from conflictfree.grad_operator import ConFIG_update
    from conflictfree.utils import apply_gradient_vector, get_gradient_vector
except ImportError:  # The baseline remains usable without the optional PBFM dependency.
    ConFIG_update = None
    apply_gradient_vector = None
    get_gradient_vector = None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the no-pilot Flow Matching model on MMW sunny Town03 Tjunction V2I data."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--weather", type=str, default="sunny")
    parser.add_argument("--town", type=str, default="Town03")
    parser.add_argument("--sensor_scenario", type=str, default=DEFAULT_SENSOR_SCENARIO)
    parser.add_argument("--channel_scenario", type=str, default=DEFAULT_CHANNEL_SCENARIO)
    parser.add_argument("--channel_config", type=str, default=DEFAULT_CHANNEL_CONFIG)
    parser.add_argument("--cavs", type=str, default=",".join(DEFAULT_CAVS))
    parser.add_argument(
        "--sensor_actor",
        type=str,
        default="rsu_1",
        help=(
            "Use RSU camera/LiDAR/GPS plus the communicating CAV GPS by default. "
            "Use 'same' for each target CAV, or 'both' for full RSU+CAV sensor inputs."
        ),
    )
    parser.add_argument("--camera", type=str, default="camera0")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--point_scale", type=float, default=50.0)
    parser.add_argument("--channel_domain", choices=["angular", "spatial"], default="angular")
    parser.add_argument("--csi_method", choices=["sum_paths", "steering"], default="sum_paths")
    parser.add_argument("--normalize_channel", type=parse_bool, default=True)
    # The physics residual operates on complex channels, so expose the same
    # target representation option as inference and validate it when enabled.
    parser.add_argument(
        "--target_mode",
        choices=["complex", "magnitude", "power"],
        default="complex",
    )
    parser.add_argument(
        "--use_link_geometry",
        type=parse_bool,
        default=True,
        help="Append explicit RSU-to-CAV relative geometry to the target GPS embedding.",
    )
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument(
        "--stage",
        choices=["joint", "vae", "fm"],
        default="joint",
        help="joint trains VE+FM together, vae trains only the multimodal prior, fm freezes a trained VE and trains only Flow.",
    )
    parser.add_argument(
        "--vae_checkpoint_dir",
        type=str,
        default="",
        help="Directory containing ve_multimodality_mmw_best.pth for --stage fm.",
    )
    parser.add_argument(
        "--vae_checkpoint_name",
        type=str,
        default="ve_multimodality_mmw_best.pth",
        help="VAE checkpoint filename used by --stage fm.",
    )
    parser.add_argument(
        "--use_vae_config",
        type=parse_bool,
        default=True,
        help="For --stage fm, reuse dataset/model options stored in the VAE config.json when present.",
    )
    parser.add_argument(
        "--fm_prior_mode",
        choices=["mu", "sample"],
        default="mu",
        help="Initial point used while training Flow from a frozen VE. Use mu for MSE-trained VAE checkpoints.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument(
        "--flow_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the flow matching velocity loss.",
    )
    parser.add_argument(
        "--ve_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the contrastive VAE loss.",
    )
    parser.add_argument(
        "--prior_mse_weight",
        type=float,
        default=0.0,
        help="Optional direct MSE weight between VE mean mu and the true CSI. Keep 0.0 for the clean no-pilot baseline.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # -------------------------------------------------------------------------
    # Optional PBFM-inspired pilot controls. Physics training is opt-in so a
    # normal baseline run retains its previous objective and behaviour.
    parser.add_argument(
        "--physics_loss",
        type=parse_bool,
        default=False,
        help="Enable simulator-angle steering-subspace loss with PBFM-style ConFIG updates.",
    )
    parser.add_argument(
        "--physics_unroll_max_steps",
        type=int,
        default=4,
        help="Maximum Euler steps used to unroll the training-time physics estimate.",
    )
    parser.add_argument(
        "--flow_checkpoint",
        type=str,
        default="",
        help="Optional flow checkpoint to fine-tune (defaults to outputs/flow_matching_mmw_best.pth for physics runs).",
    )
    # ------------------------- END PHYSICS CLI OPTIONS -----------------------
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=50,
        help="Stop after this many epochs without train-loss improvement. 0 disables early stopping.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    return parser


def option_was_provided(raw_args, option):
    return option in raw_args


def apply_stage_defaults(args, raw_args):
    if args.stage == "vae":
        if not option_was_provided(raw_args, "--flow_loss_weight"):
            args.flow_loss_weight = 0.0
        if not option_was_provided(raw_args, "--ve_loss_weight"):
            args.ve_loss_weight = 0.0
        if not option_was_provided(raw_args, "--prior_mse_weight"):
            args.prior_mse_weight = 1.0
    elif args.stage == "fm":
        if not option_was_provided(raw_args, "--flow_loss_weight"):
            args.flow_loss_weight = 1.0
        if not option_was_provided(raw_args, "--ve_loss_weight"):
            args.ve_loss_weight = 0.0
        if not option_was_provided(raw_args, "--prior_mse_weight"):
            args.prior_mse_weight = 0.0
    return args


def resolve_checkpoint_dir(path_text, script_dir):
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = script_dir / path
    return path


def load_config(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_vae_config_defaults(args, config, raw_args):
    if args.stage != "fm" or not args.use_vae_config:
        return args

    dataset_cfg = config.get("dataset", {})
    saved_args = config.get("args", {})
    dataset_keys = (
        "weather",
        "town",
        "sensor_scenario",
        "channel_scenario",
        "channel_config",
        "sensor_actor",
        "camera",
        "image_size",
        "num_points",
        "point_scale",
        "channel_domain",
        "csi_method",
        "normalize_channel",
        "target_mode",
    )
    for key in dataset_keys:
        option = f"--{key}"
        if key in dataset_cfg and not option_was_provided(raw_args, option):
            setattr(args, key, dataset_cfg[key])

    if "cavs" in dataset_cfg and not option_was_provided(raw_args, "--cavs"):
        args.cavs = ",".join(dataset_cfg["cavs"])
    if "base_channels" in saved_args and not option_was_provided(raw_args, "--base_channels"):
        args.base_channels = int(saved_args["base_channels"])
    if "max_samples" in saved_args and args.max_samples == 0 and not option_was_provided(raw_args, "--max_samples"):
        args.max_samples = int(saved_args["max_samples"])
    if "use_link_geometry" in saved_args and not option_was_provided(raw_args, "--use_link_geometry"):
        args.use_link_geometry = parse_bool(saved_args["use_link_geometry"])
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def contrastive_vae_loss(channel, prior, logit_scale, mu, logvar, kl_weight):
    channel_flat = F.normalize(channel.flatten(1), dim=-1)
    prior_flat = F.normalize(prior.flatten(1), dim=-1)
    logits = logit_scale * channel_flat @ prior_flat.t()
    targets = torch.arange(channel.shape[0], device=channel.device)
    clip_loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3)))
    return clip_loss + kl_weight * kl_loss


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def encode_prior(batch, ve_model, sample=True):
    target_gps = batch["target_gps"]
    link_geometry = batch.get("link_geometry")
    if not getattr(ve_model, "use_link_geometry", True):
        link_geometry = None
    if "cav_image" in batch:
        rsu_points = batch["rsu_points"].permute(0, 2, 1)
        cav_points = batch["cav_points"].permute(0, 2, 1)
        if sample:
            return ve_model.forward_dual(
                batch["rsu_image"],
                batch["rsu_gps"],
                rsu_points,
                batch["cav_image"],
                batch["cav_gps"],
                cav_points,
                target_gps,
                link_geometry,
            )
        mu, logvar = ve_model.encode_joint_dual(
            batch["rsu_image"],
            batch["rsu_gps"],
            rsu_points,
            batch["cav_image"],
            batch["cav_gps"],
            cav_points,
            target_gps,
            link_geometry,
        )
        return mu, mu, logvar

    points = batch["points"].permute(0, 2, 1)
    if sample:
        return ve_model(batch["image"], batch["sensor_gps"], target_gps, points, link_geometry)
    mu, logvar = ve_model.encode_joint(
        batch["image"],
        batch["sensor_gps"],
        target_gps,
        points,
        link_geometry,
    )
    return mu, mu, logvar


def train_one_epoch(
    dataset_loader,
    ve_model,
    flow_model,
    optimizer,
    loss_fn,
    args,
    device,
    epoch,
    gradient_model=None,
):
    ve_model.train(args.stage != "fm")
    flow_model.train(args.stage != "vae")
    total_loss = 0.0
    total_flow_loss = 0.0
    total_ve_loss = 0.0
    total_prior_mse = 0.0
    total_physics_loss = 0.0
    total_physics_score = 0.0
    total_count = 0

    for batch in dataset_loader:
        batch = move_batch_to_device(batch, device)
        x_1 = batch["channel"]

        if args.stage == "fm":
            sample_prior = args.fm_prior_mode == "sample"
            with torch.no_grad():
                x_0, mu, logvar = encode_prior(batch, ve_model, sample=sample_prior)
        elif args.stage == "vae":
            _, mu, logvar = encode_prior(batch, ve_model, sample=False)
            x_0 = mu
        else:
            x_0, mu, logvar = encode_prior(batch, ve_model, sample=True)

        logit_scale = torch.clamp(ve_model.logit_scale.exp(), max=100)

        optimizer.zero_grad(set_to_none=True)
        physics_loss = x_1.new_zeros(())
        physics_score = x_1.new_zeros(())
        if args.stage == "vae":
            flow_loss = x_1.new_zeros(())
        else:
            t = torch.rand(x_1.size(0), device=device)
            t_view = t[:, None, None, None]
            x_t = t_view * x_1 + (1.0 - t_view) * x_0
            velocity = flow_model(x_t, t)
            flow_loss = loss_fn(velocity, x_1 - x_0)
            # -----------------------------------------------------------------
            # PHYSICS PILOT ADDITION: unroll the learned flow toward t=1, then
            # evaluate the steering-subspace residual on that approximate final
            # sample. The step count ramps up during training to limit early cost.
            # This is the PBFM-style endpoint idea, not merely extra inference
            # sampling steps. Simulator path angles provide the atoms here.
            # -----------------------------------------------------------------
            if args.physics_loss:
                ramp = max(1, args.epochs // 4)
                unroll_steps = min(args.physics_unroll_max_steps, 1 + (epoch // ramp))
                dt = (1.0 - t) / unroll_steps
                x_phys = x_t + dt[:, None, None, None] * velocity
                for step in range(1, unroll_steps):
                    t_step = t + step * dt
                    velocity_step = flow_model(x_phys, t_step)
                    x_phys = x_phys + dt[:, None, None, None] * velocity_step
                physics_per_sample = steering_subspace_outside_fraction(
                    x_phys,
                    batch["steering_atoms"],
                    batch["steering_count"],
                )
                physics_score = physics_per_sample.mean()
                # PBFM weights the residual by t; ConFIG combines this objective's
                # gradient with flow matching without a hand-tuned scalar weight.
                physics_loss = torch.mean(t * physics_per_sample)
            # ---------------------- END UNROLLED PHYSICS LOSS -----------------
        if args.stage == "vae" and args.ve_loss_weight == 0.0:
            ve_loss = x_1.new_zeros(())
        else:
            ve_loss = contrastive_vae_loss(x_1, x_0, logit_scale, mu, logvar, args.kl_weight)
        prior_mse = loss_fn(mu, x_1)
        loss = (
            args.flow_loss_weight * flow_loss
            + args.ve_loss_weight * ve_loss
            + args.prior_mse_weight * prior_mse
        )
        if args.physics_loss:
            # Compute both objectives' gradients separately, then use ConFIG to
            # make one conflict-aware update for the trainable flow model. This is
            # gradient combination, not a scalar physics-loss weight. The safety
            # fallback keeps the ordinary objective gradient if physics gradients
            # become non-finite for a batch.
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            gradients = [get_gradient_vector(gradient_model)]
            optimizer.zero_grad(set_to_none=True)
            physics_loss.backward()
            gradients.append(get_gradient_vector(gradient_model))
            optimizer.zero_grad(set_to_none=True)
            if not gradients[1].isnan().any() and not gradients[1].isinf().any():
                apply_gradient_vector(gradient_model, ConFIG_update(gradients))
            else:
                print("Non-finite physics gradient; applying flow/objective gradient for this batch.", flush=True)
                apply_gradient_vector(gradient_model, gradients[0])
        else:
            loss.backward()
        optimizer.step()

        batch_count = x_1.size(0)
        total_loss += loss.item() * batch_count
        total_flow_loss += flow_loss.item() * batch_count
        total_ve_loss += ve_loss.item() * batch_count
        total_prior_mse += prior_mse.item() * batch_count
        total_physics_loss += physics_loss.item() * batch_count
        total_physics_score += physics_score.item() * batch_count
        total_count += batch_count

    total_count = max(total_count, 1)
    return {
        "loss": total_loss / total_count,
        "flow_loss": total_flow_loss / total_count,
        "ve_loss": total_ve_loss / total_count,
        "prior_mse": total_prior_mse / total_count,
        "physics_loss": total_physics_loss / total_count,
        "physics_score": total_physics_score / total_count,
    }


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    args = apply_stage_defaults(args, raw_args)
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    # -------------------------------------------------------------------------
    # Physics-pilot guardrails and defaults: keep the pretrained multimodal prior
    # frozen, require complex channels and ConFIG, and direct outputs to a separate
    # folder so baseline checkpoints are not overwritten.
    if args.physics_loss:
        if args.stage != "fm":
            raise ValueError("The physics pilot currently requires --stage fm to keep the pretrained multimodal prior frozen.")
        if args.target_mode != "complex":
            raise ValueError("The steering-subspace physics pilot requires --target_mode complex.")
        if args.physics_unroll_max_steps < 1:
            raise ValueError("--physics_unroll_max_steps must be at least 1.")
        if not args.vae_checkpoint_dir:
            args.vae_checkpoint_dir = "outputs"
        if args.flow_checkpoint == "":
            args.flow_checkpoint = str(Path("outputs") / "flow_matching_mmw_best.pth")
        if args.output_dir == Path("outputs"):
            args.output_dir = Path("outputs_pbfm_pilot")
        if ConFIG_update is None:
            raise ImportError(
                "Physics training requires the optional PBFM dependency 'conflictfree'. "
                "Install it in the same Python environment as PyTorch."
            )
    # ------------------------ END PHYSICS PILOT SETUP ------------------------
    vae_checkpoint_dir = resolve_checkpoint_dir(args.vae_checkpoint_dir, script_dir)
    vae_config = load_config(vae_checkpoint_dir / "config.json") if vae_checkpoint_dir else {}
    args = apply_vae_config_defaults(args, vae_config, raw_args)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MMVTjunctionDataset(
        root=args.data_root,
        weather=args.weather,
        town=args.town,
        sensor_scenario=args.sensor_scenario,
        channel_scenario=args.channel_scenario,
        channel_config=args.channel_config,
        cavs=parse_csv_names(args.cavs),
        sensor_actor=args.sensor_actor,
        camera=args.camera,
        image_size=args.image_size,
        num_points=args.num_points,
        point_scale=args.point_scale,
        channel_domain=args.channel_domain,
        csi_method=args.csi_method,
        normalize_channel=args.normalize_channel,
        max_samples=args.max_samples,
        random_lidar=True,
        include_steering_atoms=args.physics_loss,
    )
    train_set = split_dataset(dataset, 1.0, "all")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_steering if args.physics_loss else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ve_model = VE_MultimodalityMMW(channel_shape=dataset.channel_shape).to(device)
    ve_model.use_link_geometry = bool(args.use_link_geometry)
    flow_model = MiniUnet(base_channels=args.base_channels).to(device)

    flow_checkpoint_path = None
    if args.flow_checkpoint:
        flow_checkpoint_path = Path(args.flow_checkpoint)
        if not flow_checkpoint_path.is_absolute():
            flow_checkpoint_path = script_dir / flow_checkpoint_path
        if not flow_checkpoint_path.exists():
            raise FileNotFoundError(f"Missing flow checkpoint: {flow_checkpoint_path}")
        flow_model.load_state_dict(torch.load(flow_checkpoint_path, map_location=device))

    if args.stage == "fm":
        if vae_checkpoint_dir is None:
            raise ValueError("--vae_checkpoint_dir is required for --stage fm.")
        ve_path = vae_checkpoint_dir / args.vae_checkpoint_name
        if not ve_path.exists():
            raise FileNotFoundError(f"Missing VAE checkpoint for FM training: {ve_path}")
        ve_model.load_state_dict(torch.load(ve_path, map_location=device))
        ve_model.eval()
        for parameter in ve_model.parameters():
            parameter.requires_grad_(False)

    trainable_parameters = []
    if args.stage in {"joint", "vae"}:
        trainable_parameters.append({"params": ve_model.parameters(), "lr": args.lr})
    if args.stage in {"joint", "fm"}:
        trainable_parameters.append({"params": flow_model.parameters(), "lr": args.lr})
    optimizer = optim.AdamW(trainable_parameters, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    gradient_model = nn.ModuleList([flow_model]) if args.physics_loss else None

    config_payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset": dataset.config_dict(),
        "train_count": len(train_set),
        "validation": False,
        "device": str(device),
        "vae_checkpoint_dir": str(vae_checkpoint_dir) if vae_checkpoint_dir else "",
        "vae_checkpoint_name": args.vae_checkpoint_name if args.stage == "fm" else "",
    }
    write_json(output_dir / "config.json", config_payload)

    print(f"Using device: {device}", flush=True)
    print(f"Dataset samples: {len(dataset)} | train: {len(train_set)}", flush=True)
    print("Validation: disabled (all samples are used for training)", flush=True)
    print(
        f"CSI target: complex [2, {dataset.channel_shape[0]}, {dataset.channel_shape[1]}]",
        flush=True,
    )
    print(f"Training stage: {args.stage}", flush=True)
    print(f"Sensor actor: {args.sensor_actor} | target CAVs: {args.cavs}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    if args.stage == "fm":
        print(f"Frozen VAE checkpoint: {vae_checkpoint_dir / args.vae_checkpoint_name}", flush=True)
    if flow_checkpoint_path is not None:
        print(f"Flow initialization checkpoint: {flow_checkpoint_path}", flush=True)
    if args.physics_loss:
        print(
            f"Physics loss: simulator-angle steering subspace; PBFM ConFIG; "
            f"unroll steps ramp 1 to {args.physics_unroll_max_steps}",
            flush=True,
        )
    print(
        f"Link geometry: {'on' if args.use_link_geometry else 'off'} "
        f"(dim {dataset.link_geometry_dim})",
        flush=True,
    )

    best_train_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            train_loader,
            ve_model,
            flow_model,
            optimizer,
            loss_fn,
            args,
            device,
            epoch,
            gradient_model=gradient_model,
        )

        if args.stage == "vae":
            log_message = (
                f"====> Epoch: {epoch} "
                f"train_loss: {train_metrics['loss']:.8f} "
                f"(prior_mse {train_metrics['prior_mse']:.8f})"
            )
        else:
            log_message = (
                f"====> Epoch: {epoch} "
                f"train_loss: {train_metrics['loss']:.8f} "
                f"(flow {train_metrics['flow_loss']:.8f}, ve {train_metrics['ve_loss']:.8f}, "
                f"prior_mse {train_metrics['prior_mse']:.8f}, "
                f"physics {train_metrics['physics_loss']:.8f}, "
                f"outside {train_metrics['physics_score']:.8f})"
            )
        print(log_message, flush=True)

        latest_paths = []
        if args.stage in {"joint", "vae"}:
            latest_ve_path = output_dir / "ve_multimodality_mmw.pth"
            torch.save(ve_model.state_dict(), latest_ve_path)
            latest_paths.append(latest_ve_path.name)
        if args.stage in {"joint", "fm"}:
            latest_flow_path = output_dir / "flow_matching_mmw.pth"
            torch.save(flow_model.state_dict(), latest_flow_path)
            latest_paths.append(latest_flow_path.name)
        print(f"Saved latest checkpoints: {', '.join(latest_paths)}", flush=True)

        if train_metrics["loss"] < best_train_loss:
            previous_best = best_train_loss
            best_train_loss = train_metrics["loss"]
            epochs_without_improvement = 0
            best_paths = []
            if args.stage in {"joint", "vae"}:
                best_ve_path = output_dir / "ve_multimodality_mmw_best.pth"
                torch.save(ve_model.state_dict(), best_ve_path)
                best_paths.append(best_ve_path.name)
            if args.stage in {"joint", "fm"}:
                best_flow_path = output_dir / "flow_matching_mmw_best.pth"
                torch.save(flow_model.state_dict(), best_flow_path)
                best_paths.append(best_flow_path.name)
            if previous_best == float("inf"):
                print(
                    f"New best checkpoint at epoch {epoch}: train_loss={best_train_loss:.8f} "
                    f"-> {', '.join(best_paths)}",
                    flush=True,
                )
            else:
                print(
                    f"New best checkpoint at epoch {epoch}: train_loss "
                    f"{previous_best:.8f} -> {best_train_loss:.8f} "
                    f"-> {', '.join(best_paths)}",
                    flush=True,
                )
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}: no train_loss improvement "
                    f"for {args.early_stop_patience} epochs. Best: {best_train_loss:.8f}",
                    flush=True,
                )
                break

    print(f"Training complete. Best train_loss: {best_train_loss:.8f}", flush=True)


if __name__ == "__main__":
    main()
