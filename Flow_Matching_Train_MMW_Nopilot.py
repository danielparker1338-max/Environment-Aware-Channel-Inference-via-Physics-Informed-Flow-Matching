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
    parse_csv_names,
    split_dataset,
    write_json,
)


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


def train_one_epoch(dataset_loader, ve_model, flow_model, optimizer, loss_fn, args, device):
    ve_model.train(args.stage != "fm")
    flow_model.train(args.stage != "vae")
    total_loss = 0.0
    total_flow_loss = 0.0
    total_ve_loss = 0.0
    total_prior_mse = 0.0
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
        if args.stage == "vae":
            flow_loss = x_1.new_zeros(())
        else:
            t = torch.rand(x_1.size(0), device=device)
            t_view = t[:, None, None, None]
            x_t = t_view * x_1 + (1.0 - t_view) * x_0
            flow_loss = loss_fn(flow_model(x_t, t), x_1 - x_0)
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
        loss.backward()
        optimizer.step()

        batch_count = x_1.size(0)
        total_loss += loss.item() * batch_count
        total_flow_loss += flow_loss.item() * batch_count
        total_ve_loss += ve_loss.item() * batch_count
        total_prior_mse += prior_mse.item() * batch_count
        total_count += batch_count

    total_count = max(total_count, 1)
    return {
        "loss": total_loss / total_count,
        "flow_loss": total_flow_loss / total_count,
        "ve_loss": total_ve_loss / total_count,
        "prior_mse": total_prior_mse / total_count,
    }


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    args = apply_stage_defaults(args, raw_args)
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
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
    )
    train_set = split_dataset(dataset, 1.0, "all")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ve_model = VE_MultimodalityMMW(channel_shape=dataset.channel_shape).to(device)
    ve_model.use_link_geometry = bool(args.use_link_geometry)
    flow_model = MiniUnet(base_channels=args.base_channels).to(device)

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
    print(
        f"Link geometry: {'on' if args.use_link_geometry else 'off'} "
        f"(dim {dataset.link_geometry_dim})",
        flush=True,
    )

    best_train_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(train_loader, ve_model, flow_model, optimizer, loss_fn, args, device)

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
                f"prior_mse {train_metrics['prior_mse']:.8f})"
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
