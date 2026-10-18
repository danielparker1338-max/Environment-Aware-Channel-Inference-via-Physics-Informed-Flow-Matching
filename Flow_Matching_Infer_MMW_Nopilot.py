import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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
        description="No-pilot Flow Matching inference on MMW sunny Town03 Tjunction V2I data."
    )
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--vae_checkpoint_dir",
        type=str,
        default="",
        help="Optional separate directory containing the VAE checkpoint for two-stage training.",
    )
    parser.add_argument("--vae_checkpoint_name", type=str, default="ve_multimodality_mmw_best.pth")
    parser.add_argument("--use_best", type=parse_bool, default=True)
    parser.add_argument(
        "--use_checkpoint_config",
        type=parse_bool,
        default=True,
        help="Reuse dataset/model defaults stored by the training script.",
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--weather", type=str, default="sunny")
    parser.add_argument("--town", type=str, default="Town03")
    parser.add_argument("--sensor_scenario", type=str, default=DEFAULT_SENSOR_SCENARIO)
    parser.add_argument("--channel_scenario", type=str, default=DEFAULT_CHANNEL_SCENARIO)
    parser.add_argument("--channel_config", type=str, default=DEFAULT_CHANNEL_CONFIG)
    parser.add_argument("--cavs", type=str, default=",".join(DEFAULT_CAVS))
    parser.add_argument("--sensor_actor", type=str, default="rsu_1")
    parser.add_argument("--camera", type=str, default="camera0")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--point_scale", type=float, default=50.0)
    parser.add_argument("--channel_domain", choices=["angular", "spatial"], default="angular")
    parser.add_argument("--csi_method", choices=["sum_paths", "steering"], default="sum_paths")
    parser.add_argument("--normalize_channel", type=parse_bool, default=True)
    parser.add_argument("--target_mode", choices=["complex", "magnitude", "power"], default="complex")
    parser.add_argument(
        "--use_link_geometry",
        type=parse_bool,
        default=True,
        help="Append explicit RSU-to-CAV relative geometry to the target GPS embedding.",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--schedule", choices=["linear", "geometric"], default="linear")
    parser.add_argument("--schedule_power", type=float, default=1.5)
    parser.add_argument("--prior_mode", choices=["mu", "sample"], default="mu")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_training_config(checkpoint_dir):
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_saved_defaults(args, config):
    if not args.use_checkpoint_config:
        return args

    dataset_cfg = config.get("dataset", {})
    saved_args = config.get("args", {})
    if not args.vae_checkpoint_dir and config.get("vae_checkpoint_dir"):
        args.vae_checkpoint_dir = config["vae_checkpoint_dir"]
    if (
        args.vae_checkpoint_name == "ve_multimodality_mmw_best.pth"
        and config.get("vae_checkpoint_name")
    ):
        args.vae_checkpoint_name = config["vae_checkpoint_name"]
    for key in (
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
    ):
        if key in dataset_cfg:
            setattr(args, key, dataset_cfg[key])

    if "root" in dataset_cfg and args.data_root == DEFAULT_ROOT:
        args.data_root = Path(dataset_cfg["root"])
    if "cavs" in dataset_cfg:
        args.cavs = ",".join(dataset_cfg["cavs"])
    if "train_ratio" in saved_args:
        args.train_ratio = float(saved_args["train_ratio"])
    if "base_channels" in saved_args:
        args.base_channels = int(saved_args["base_channels"])
    if "max_samples" in saved_args and args.max_samples == 0:
        args.max_samples = int(saved_args["max_samples"])
    if "use_link_geometry" in saved_args:
        args.use_link_geometry = parse_bool(saved_args["use_link_geometry"])
    elif config:
        args.use_link_geometry = False
    return args


def build_time_schedule(num_steps, schedule, power, device):
    u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    if schedule == "linear":
        return u
    if schedule == "geometric":
        return u.pow(power)
    raise ValueError(f"Unsupported schedule: {schedule}")


def per_sample_nmse_db(pred, target):
    numerator = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    denominator = torch.sum(target ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return 10 * torch.log10(numerator / denominator)


def aggregate_nmse_db(pred, target):
    numerator = torch.sum((pred - target) ** 2)
    denominator = torch.sum(target ** 2).clamp_min(1e-12)
    return 10 * torch.log10(numerator / denominator)


def to_complex(channel):
    return torch.complex(channel[:, 0], channel[:, 1])


def aggregate_magnitude_nmse_db(pred, target):
    pred_mag = torch.abs(to_complex(pred))
    target_mag = torch.abs(to_complex(target))
    numerator = torch.sum((pred_mag - target_mag) ** 2)
    denominator = torch.sum(target_mag ** 2).clamp_min(1e-12)
    return 10 * torch.log10(numerator / denominator)


def aggregate_phase_aligned_nmse_db(pred, target):
    pred_complex = to_complex(pred)
    target_complex = to_complex(target)
    inner = torch.sum(target_complex * torch.conj(pred_complex), dim=(1, 2))
    phase = torch.exp(1j * torch.angle(inner))[:, None, None]
    aligned = pred_complex * phase
    numerator = torch.sum(torch.abs(aligned - target_complex) ** 2)
    denominator = torch.sum(torch.abs(target_complex) ** 2).clamp_min(1e-12)
    return 10 * torch.log10(numerator / denominator)


@torch.no_grad()
def compute_channel_mean(dataset, indices, device):
    channel_sum = None
    total_count = 0
    for index in indices:
        channel = dataset.channel_tensor_for_index(int(index)).to(device)
        if channel_sum is None:
            channel_sum = channel.clone()
        else:
            channel_sum += channel
        total_count += 1
    if channel_sum is None or total_count == 0:
        raise ValueError("Cannot compute channel mean from an empty split.")
    return channel_sum / total_count


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def encode_inference_prior(batch, ve_model):
    target_gps = batch["target_gps"]
    link_geometry = batch.get("link_geometry")
    if not getattr(ve_model, "use_link_geometry", True):
        link_geometry = None
    if "cav_image" in batch:
        return ve_model.encode_joint_dual(
            batch["rsu_image"],
            batch["rsu_gps"],
            batch["rsu_points"].permute(0, 2, 1),
            batch["cav_image"],
            batch["cav_gps"],
            batch["cav_points"].permute(0, 2, 1),
            target_gps,
            link_geometry,
        )
    return ve_model.encode_joint(
        batch["image"],
        batch["sensor_gps"],
        target_gps,
        batch["points"].permute(0, 2, 1),
        link_geometry,
    )


@torch.no_grad()
def run_inference(loader, ve_model, flow_model, args, device):
    ve_model.eval()
    flow_model.eval()

    predictions = []
    priors = []
    truths = []
    frames = []
    cav_indices = []
    dataset_indices = []

    t_points = build_time_schedule(args.n_steps, args.schedule, args.schedule_power, device)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        truth = batch["channel"]

        mu, logvar = encode_inference_prior(batch, ve_model)
        if args.prior_mode == "sample":
            x_t = ve_model.reparameterize(mu, logvar)
        else:
            x_t = mu
        prior = x_t.clone()

        for step in range(args.n_steps):
            t_current = t_points[step]
            t_next = t_points[step + 1]
            dt = (t_next - t_current).item()
            t = torch.full((truth.size(0),), t_current.item(), device=device)
            x_t = x_t + flow_model(x_t, t) * dt

        predictions.append(x_t.detach().cpu())
        priors.append(prior.detach().cpu())
        truths.append(truth.detach().cpu())
        frames.append(batch["frame"].detach().cpu())
        cav_indices.append(batch["cav_index"].detach().cpu())
        dataset_indices.append(batch["index"].detach().cpu())

    return {
        "prediction": torch.cat(predictions, dim=0),
        "prior": torch.cat(priors, dim=0),
        "truth": torch.cat(truths, dim=0),
        "frames": torch.cat(frames, dim=0),
        "cav_indices": torch.cat(cav_indices, dim=0),
        "dataset_indices": torch.cat(dataset_indices, dim=0),
    }


def save_outputs(path, outputs, metadata):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "csi_pred": outputs["prediction"].numpy(),
        "csi_true": outputs["truth"].numpy(),
        "prior_init": outputs["prior"].numpy(),
        "frames": outputs["frames"].numpy(),
        "cav_indices": outputs["cav_indices"].numpy(),
        "dataset_indices": outputs["dataset_indices"].numpy(),
    }

    if path.suffix == ".mat":
        try:
            import scipy.io as scio
        except ImportError as exc:
            raise ImportError("scipy is required for .mat output. Use a .npz save path instead.") from exc
        scio.savemat(path, arrays)
    else:
        np.savez_compressed(path, **arrays)

    write_json(path.with_suffix(path.suffix + ".metadata.json"), {"samples": metadata})
    print(f"Saved inference outputs to {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = script_dir / checkpoint_dir

    config = load_training_config(checkpoint_dir)
    args = apply_saved_defaults(args, config)
    vae_checkpoint_dir = Path(args.vae_checkpoint_dir) if args.vae_checkpoint_dir else None
    if vae_checkpoint_dir is None:
        vae_checkpoint_dir = checkpoint_dir
    elif not vae_checkpoint_dir.is_absolute():
        vae_checkpoint_dir = script_dir / vae_checkpoint_dir

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
        target_mode=args.target_mode,
        max_samples=args.max_samples,
        random_lidar=False,
    )

    split_set = split_dataset(dataset, args.train_ratio, args.split)
    split_indices = list(split_set.indices)
    train_indices = list(split_dataset(dataset, args.train_ratio, "train").indices)
    selected_indices = split_indices[args.sample_start :]
    if args.sample_num > 0:
        selected_indices = selected_indices[: args.sample_num]
    if not selected_indices:
        raise ValueError("Requested inference sample range is empty.")

    selected = Subset(dataset, selected_indices)
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ve_model = VE_MultimodalityMMW(
        channel_shape=dataset.channel_shape,
        output_channels=dataset.target_channels,
    ).to(device)
    ve_model.use_link_geometry = bool(args.use_link_geometry)
    flow_model = MiniUnet(
        in_channels=dataset.target_channels,
        out_channels=dataset.target_channels,
        base_channels=args.base_channels,
    ).to(device)

    suffix = "_best" if args.use_best else ""
    if vae_checkpoint_dir == checkpoint_dir:
        ve_path = vae_checkpoint_dir / f"ve_multimodality_mmw{suffix}.pth"
    else:
        ve_path = vae_checkpoint_dir / args.vae_checkpoint_name
    flow_path = checkpoint_dir / f"flow_matching_mmw{suffix}.pth"
    if not ve_path.exists() or not flow_path.exists():
        raise FileNotFoundError(f"Missing checkpoint files: {ve_path} / {flow_path}")

    ve_model.load_state_dict(torch.load(ve_path, map_location=device))
    flow_model.load_state_dict(torch.load(flow_path, map_location=device))

    print(f"Using device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"VAE checkpoint dir: {vae_checkpoint_dir}")
    print(f"Running no-pilot inference on {len(selected_indices)} samples from split '{args.split}'.")
    print(
        f"Target shape: [{dataset.target_channels}, {dataset.channel_shape[0]}, {dataset.channel_shape[1]}] "
        f"mode={dataset.target_mode}"
    )
    print(
        f"Link geometry: {'on' if args.use_link_geometry else 'off'} "
        f"(dim {dataset.link_geometry_dim})"
    )

    outputs = run_inference(loader, ve_model, flow_model, args, device)
    zero_baseline = torch.zeros_like(outputs["truth"])
    train_mean = compute_channel_mean(
        dataset,
        train_indices,
        device,
    ).cpu()
    train_mean_baseline = train_mean.unsqueeze(0).expand_as(outputs["truth"])
    zero_nmse = aggregate_nmse_db(zero_baseline, outputs["truth"])
    train_mean_nmse = aggregate_nmse_db(train_mean_baseline, outputs["truth"])
    prior_nmse = aggregate_nmse_db(outputs["prior"], outputs["truth"])
    final_nmse = aggregate_nmse_db(outputs["prediction"], outputs["truth"])
    if dataset.target_channels == 2:
        prior_mag_nmse = aggregate_magnitude_nmse_db(outputs["prior"], outputs["truth"])
        final_mag_nmse = aggregate_magnitude_nmse_db(outputs["prediction"], outputs["truth"])
        prior_phase_aligned_nmse = aggregate_phase_aligned_nmse_db(outputs["prior"], outputs["truth"])
        final_phase_aligned_nmse = aggregate_phase_aligned_nmse_db(outputs["prediction"], outputs["truth"])
    else:
        prior_mag_nmse = None
        final_mag_nmse = None
        prior_phase_aligned_nmse = None
        final_phase_aligned_nmse = None
    prior_per_sample = per_sample_nmse_db(outputs["prior"], outputs["truth"])
    final_per_sample = per_sample_nmse_db(outputs["prediction"], outputs["truth"])

    print(f"Zero baseline NMSE (aggregate): {zero_nmse.item():.4f} dB")
    print(f"Train-mean baseline NMSE (aggregate): {train_mean_nmse.item():.4f} dB")
    print(f"Prior NMSE (aggregate): {prior_nmse.item():.4f} dB")
    print(f"Final NMSE (aggregate): {final_nmse.item():.4f} dB")
    if prior_mag_nmse is not None:
        print(f"Prior magnitude NMSE (aggregate): {prior_mag_nmse.item():.4f} dB")
        print(f"Final magnitude NMSE (aggregate): {final_mag_nmse.item():.4f} dB")
        print(f"Prior phase-aligned NMSE (aggregate): {prior_phase_aligned_nmse.item():.4f} dB")
        print(f"Final phase-aligned NMSE (aggregate): {final_phase_aligned_nmse.item():.4f} dB")
    print(f"Mean per-sample NMSE: {final_per_sample.mean().item():.4f} dB")
    print(f"Median per-sample NMSE: {final_per_sample.median().item():.4f} dB")

    cav_names = list(parse_csv_names(args.cavs))
    for cav_index, cav_name in enumerate(cav_names):
        mask = outputs["cav_indices"] == cav_index
        if torch.any(mask):
            cav_values = final_per_sample[mask]
            print(
                f"  {cav_name} final NMSE: mean {cav_values.mean().item():.4f} dB, "
                f"median {cav_values.median().item():.4f} dB"
            )

    metadata = dataset.metadata_for_indices(selected_indices)
    for row, prior_value, final_value in zip(metadata, prior_per_sample, final_per_sample):
        cav_name = row["cav"]
        print(
            f"  frame {row['frame']} {cav_name}: "
            f"prior {prior_value.item():.4f} dB, final {final_value.item():.4f} dB"
        )

    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute():
            save_path = checkpoint_dir / save_path
        save_outputs(save_path, outputs, metadata)


if __name__ == "__main__":
    main()
