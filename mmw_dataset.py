import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in missing deps envs.
    raise ImportError(
        "PyYAML is required to read the MMW sensor metadata. Install pyyaml "
        "in the same Python environment as torch."
    ) from exc


DEFAULT_ROOT = Path("/home/largeDisk/guangming/Latent-Diffusion-Channel/Multimodal-Wireless/")
DEFAULT_SENSOR_SCENARIO = "Town03_Tjunction_wiz_slope_seed42"
DEFAULT_CHANNEL_SCENARIO = "Town03_Tjunction"
DEFAULT_CHANNEL_CONFIG = "Nt_1_64_Nr_1_16_fc_28GHz"
DEFAULT_CAVS = ("cav_1", "cav_2", "cav_3", "cav_4")


@dataclass(frozen=True)
class SampleRecord:
    frame: int
    cav: str
    sensor_actor: str
    image_path: Path
    point_path: Path
    channel_path: Path
    sensor_gps: np.ndarray
    target_gps: np.ndarray
    rsu_image_path: Optional[Path] = None
    rsu_point_path: Optional[Path] = None
    cav_image_path: Optional[Path] = None
    cav_point_path: Optional[Path] = None
    rsu_gps: Optional[np.ndarray] = None
    cav_gps: Optional[np.ndarray] = None


def parse_csv_names(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def location_dict_to_array(location: Dict) -> np.ndarray:
    return np.array(
        [location.get("x", 0.0), location.get("y", 0.0), location.get("z", 0.0)],
        dtype=np.float32,
    )


def extract_actor_location(meta: Dict, actor: str) -> np.ndarray:
    sensors = meta.get("sensors", {})

    if "GPS" in sensors and "location" in sensors["GPS"]:
        return location_dict_to_array(sensors["GPS"]["location"])

    pose_key_order = (
        "vehicle_pose",
        "rsu_pose",
        "lidar_pose",
        "imu_pose",
        "true_ego_pos",
        "predicted_ego_pos",
    )
    for key in pose_key_order:
        if key in sensors and "location" in sensors[key]:
            return location_dict_to_array(sensors[key]["location"])

    raise KeyError(f"Could not find a GPS/pose location for actor {actor}.")


def dft_matrix(size: int) -> np.ndarray:
    n = np.arange(size)
    k = n[:, None]
    return np.exp(-2j * np.pi * k * n / size).astype(np.complex64) / math.sqrt(size)


def normalize_complex_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    scale = max(float(np.max(np.abs(matrix.real))), float(np.max(np.abs(matrix.imag))))
    if scale <= 0:
        scale = 1.0
    return matrix / scale, scale


def cir_to_csi_sum_paths(npz_data: Dict[str, np.ndarray]) -> np.ndarray:
    path_matrix = np.squeeze(npz_data["a"])
    if path_matrix.ndim != 3:
        raise ValueError(f"Expected squeezed 'a' to have shape [Nr, Nt, L], got {path_matrix.shape}.")
    return np.sum(path_matrix, axis=-1).astype(np.complex64)


def cir_to_csi_steering(npz_data: Dict[str, np.ndarray], nr: int, nt: int) -> np.ndarray:
    # Reference implementation that mirrors generate_CSI.m. The MMW files already
    # contain per-antenna path matrices in 'a'; this mode uses one reference path
    # gain plus AoD/AoA angles to rebuild a narrowband matrix.
    a = np.asarray(npz_data["a"])
    alpha = np.squeeze(a)[0, 0, :].astype(np.complex64)
    phi_t = np.squeeze(npz_data["phi_t"]).astype(np.float32)
    phi_r = np.squeeze(npz_data["phi_r"]).astype(np.float32)
    d = 0.5

    tx_index = np.arange(nt, dtype=np.float32)
    rx_index = np.arange(nr, dtype=np.float32)
    channel = np.zeros((nr, nt), dtype=np.complex64)
    for path_index in range(alpha.shape[0]):
        a_tx = np.exp(1j * 2 * np.pi * d * tx_index * np.sin(phi_t[path_index]))
        a_rx = np.exp(1j * 2 * np.pi * d * rx_index * np.sin(phi_r[path_index]))
        channel += alpha[path_index] * np.outer(a_rx, np.conj(a_tx))
    return channel.astype(np.complex64)


def cir_file_to_csi(
    path: Path,
    method: str = "sum_paths",
    channel_domain: str = "angular",
    normalize: bool = True,
) -> Tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as npz:
        if method == "sum_paths":
            channel = cir_to_csi_sum_paths(npz)
        elif method == "steering":
            squeezed = np.squeeze(npz["a"])
            channel = cir_to_csi_steering(npz, nr=squeezed.shape[0], nt=squeezed.shape[1])
        else:
            raise ValueError(f"Unsupported CIR-to-CSI method: {method}")

    if channel_domain == "angular":
        nr, nt = channel.shape
        f_r = dft_matrix(nr)
        f_t = dft_matrix(nt)
        channel = f_r.conj().T @ channel @ f_t
    elif channel_domain != "spatial":
        raise ValueError(f"Unsupported channel domain: {channel_domain}")

    if normalize:
        channel, scale = normalize_complex_matrix(channel)
    else:
        scale = 1.0

    stacked = np.stack([channel.real, channel.imag], axis=0).astype(np.float32)
    return stacked, scale


def convert_channel_target(stacked_complex: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "complex":
        return stacked_complex.astype(np.float32)

    complex_channel = stacked_complex[0] + 1j * stacked_complex[1]
    if target_mode == "magnitude":
        return np.abs(complex_channel)[None, :, :].astype(np.float32)
    if target_mode == "power":
        return (np.abs(complex_channel) ** 2)[None, :, :].astype(np.float32)
    raise ValueError(f"Unsupported channel target mode: {target_mode}")


def parse_channel_shape_from_file(path: Path) -> Tuple[int, int]:
    with np.load(path, allow_pickle=False) as npz:
        squeezed = np.squeeze(npz["a"])
    if squeezed.ndim != 3:
        raise ValueError(f"Cannot infer channel shape from {path}: squeezed shape is {squeezed.shape}.")
    return int(squeezed.shape[0]), int(squeezed.shape[1])


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array.transpose(2, 0, 1)).float()


@lru_cache(maxsize=2048)
def cached_image_tensor(path_text: str, image_size: int) -> torch.Tensor:
    return load_image_tensor(Path(path_text), image_size)


def read_ascii_pcd_xyz(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip().lower().startswith("data"):
                break
        points = []
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not points:
        raise ValueError(f"No XYZ points found in {path}.")
    return np.asarray(points, dtype=np.float32)


@lru_cache(maxsize=2048)
def cached_ascii_pcd_xyz(path_text: str) -> np.ndarray:
    return read_ascii_pcd_xyz(Path(path_text))


def sample_points(points: np.ndarray, num_points: int, random_sample: bool) -> np.ndarray:
    point_count = points.shape[0]
    if point_count >= num_points:
        if random_sample:
            indices = np.random.choice(point_count, num_points, replace=False)
        else:
            indices = np.linspace(0, point_count - 1, num_points).astype(np.int64)
        return points[indices]

    pad_count = num_points - point_count
    pad = np.repeat(points[-1:, :], pad_count, axis=0)
    return np.concatenate([points, pad], axis=0)


@lru_cache(maxsize=8192)
def cached_cir_file_to_csi(
    path_text: str,
    method: str,
    channel_domain: str,
    normalize: bool,
) -> Tuple[np.ndarray, float]:
    return cir_file_to_csi(
        Path(path_text),
        method=method,
        channel_domain=channel_domain,
        normalize=normalize,
    )


class MMVTjunctionDataset(Dataset):
    def __init__(
        self,
        root: Path = DEFAULT_ROOT,
        weather: str = "sunny",
        town: str = "Town03",
        sensor_scenario: str = DEFAULT_SENSOR_SCENARIO,
        channel_scenario: str = DEFAULT_CHANNEL_SCENARIO,
        channel_config: str = DEFAULT_CHANNEL_CONFIG,
        cavs: Sequence[str] = DEFAULT_CAVS,
        sensor_actor: str = "rsu_1",
        camera: str = "camera0",
        image_size: int = 128,
        num_points: int = 1024,
        point_scale: float = 50.0,
        channel_domain: str = "angular",
        csi_method: str = "sum_paths",
        normalize_channel: bool = True,
        target_mode: str = "complex",
        max_samples: int = 0,
        random_lidar: bool = True,
    ) -> None:
        self.root = Path(root)
        self.weather = weather
        self.town = town
        self.sensor_scenario = sensor_scenario
        self.channel_scenario = channel_scenario
        self.channel_config = channel_config
        self.cavs = tuple(cavs)
        self.sensor_actor = sensor_actor
        self.use_dual_context = sensor_actor == "both"
        self.camera = camera
        self.image_size = image_size
        self.num_points = num_points
        self.point_scale = point_scale
        self.channel_domain = channel_domain
        self.csi_method = csi_method
        self.normalize_channel = normalize_channel
        if target_mode not in {"complex", "magnitude", "power"}:
            raise ValueError(f"Unsupported target_mode: {target_mode}")
        self.target_mode = target_mode
        self.target_channels = 2 if target_mode == "complex" else 1
        self.random_lidar = random_lidar
        self.link_geometry_dim = 9

        self.sensor_root = (
            self.root
            / weather
            / "Sensor Data"
            / town
            / sensor_scenario
            / sensor_scenario
        )
        self.channel_root = (
            self.root
            / weather
            / "Channel Data"
            / "V2I"
            / channel_config
            / town
            / town
            / channel_scenario
        )

        if not self.sensor_root.exists():
            raise FileNotFoundError(f"Sensor root does not exist: {self.sensor_root}")
        if not self.channel_root.exists():
            raise FileNotFoundError(f"Channel root does not exist: {self.channel_root}")

        self.records = self._build_records()
        if max_samples > 0:
            self.records = self.records[:max_samples]
        if not self.records:
            raise ValueError("No valid MMW Tjunction samples were found.")

        first_channel = self.records[0].channel_path
        self.channel_shape = parse_channel_shape_from_file(first_channel)
        self.gps_min, self.gps_max = self._compute_gps_bounds(self.records)
        self.link_distance_scale = self._compute_link_distance_scale(self.records)

    def _resolve_sensor_actor(self, cav: str) -> str:
        if self.sensor_actor == "both":
            return "rsu_1"
        return cav if self.sensor_actor == "same" else self.sensor_actor

    def _image_path(self, actor: str, frame: int) -> Path:
        return self.sensor_root / actor / f"{frame:06d}_{self.camera}.png"

    def _pcd_path(self, actor: str, frame: int) -> Path:
        return self.sensor_root / actor / f"{frame:06d}.pcd"

    def _meta_path(self, actor: str, frame: int) -> Path:
        return self.sensor_root / actor / f"{frame:06d}.yaml"

    def _channel_path(self, cav: str, frame: int) -> Path:
        return self.channel_root / cav / f"{frame:06d}_paths.npz"

    def _build_records(self) -> List[SampleRecord]:
        scene_dir = self.sensor_root / "scenes"
        frames = sorted(int(path.stem) for path in scene_dir.glob("*.yaml"))
        records: List[SampleRecord] = []
        rsu_actor = "rsu_1"

        for frame in frames:
            sensor_meta_cache: Dict[str, Dict] = {}
            for cav in self.cavs:
                sensor_actor = self._resolve_sensor_actor(cav)
                image_path = self._image_path(sensor_actor, frame)
                point_path = self._pcd_path(sensor_actor, frame)
                channel_path = self._channel_path(cav, frame)
                sensor_meta_path = self._meta_path(sensor_actor, frame)
                target_meta_path = self._meta_path(cav, frame)
                rsu_meta_path = self._meta_path(rsu_actor, frame)
                cav_image_path = self._image_path(cav, frame)
                cav_point_path = self._pcd_path(cav, frame)

                required = [
                    image_path,
                    point_path,
                    channel_path,
                    sensor_meta_path,
                    target_meta_path,
                    rsu_meta_path,
                ]
                if self.use_dual_context:
                    required.extend([cav_image_path, cav_point_path])
                if not all(path.exists() for path in required):
                    continue

                def get_actor_meta(actor: str, meta_path: Path) -> Dict:
                    if actor not in sensor_meta_cache:
                        sensor_meta_cache[actor] = load_yaml(meta_path)
                    return sensor_meta_cache[actor]

                sensor_meta = get_actor_meta(sensor_actor, sensor_meta_path)
                if cav == sensor_actor:
                    target_meta = sensor_meta
                else:
                    target_meta = get_actor_meta(cav, target_meta_path)
                rsu_meta = get_actor_meta(rsu_actor, rsu_meta_path)

                sensor_gps = extract_actor_location(sensor_meta, sensor_actor)
                target_gps = extract_actor_location(target_meta, cav)
                rsu_gps = extract_actor_location(rsu_meta, rsu_actor)
                cav_gps = target_gps

                records.append(
                    SampleRecord(
                        frame=frame,
                        cav=cav,
                        sensor_actor=sensor_actor,
                        image_path=image_path,
                        point_path=point_path,
                        channel_path=channel_path,
                        sensor_gps=sensor_gps,
                        target_gps=target_gps,
                        rsu_image_path=image_path if self.use_dual_context else None,
                        rsu_point_path=point_path if self.use_dual_context else None,
                        cav_image_path=cav_image_path if self.use_dual_context else None,
                        cav_point_path=cav_point_path if self.use_dual_context else None,
                        rsu_gps=rsu_gps,
                        cav_gps=cav_gps,
                    )
                )

        return records

    @staticmethod
    def _compute_gps_bounds(records: Sequence[SampleRecord]) -> Tuple[np.ndarray, np.ndarray]:
        values = []
        for record in records:
            values.extend([record.sensor_gps, record.target_gps])
            if record.rsu_gps is not None:
                values.append(record.rsu_gps)
            if record.cav_gps is not None:
                values.append(record.cav_gps)
        values = np.stack(values, axis=0)
        gps_min = values.min(axis=0).astype(np.float32)
        gps_max = values.max(axis=0).astype(np.float32)
        gps_max = np.maximum(gps_max, gps_min + 1e-6)
        return gps_min, gps_max

    @staticmethod
    def _compute_link_distance_scale(records: Sequence[SampleRecord]) -> float:
        distances = []
        for record in records:
            if record.rsu_gps is None or record.cav_gps is None:
                continue
            distances.append(float(np.linalg.norm(record.cav_gps - record.rsu_gps)))
        if not distances:
            return 1.0
        return max(max(distances), 1e-6)

    def _normalize_gps(self, gps: np.ndarray) -> np.ndarray:
        return ((gps - self.gps_min) / (self.gps_max - self.gps_min)).astype(np.float32)

    def _link_geometry(self, rsu_gps: np.ndarray, cav_gps: np.ndarray) -> np.ndarray:
        delta = (cav_gps - rsu_gps).astype(np.float32)
        distance = float(np.linalg.norm(delta))
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        azimuth = math.atan2(float(delta[1]), float(delta[0]))
        elevation = math.atan2(float(delta[2]), max(horizontal_distance, 1e-6))
        scale = float(self.link_distance_scale)
        return np.array(
            [
                delta[0] / scale,
                delta[1] / scale,
                delta[2] / scale,
                distance / scale,
                horizontal_distance / scale,
                math.sin(azimuth),
                math.cos(azimuth),
                math.sin(elevation),
                math.cos(elevation),
            ],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.records)

    def channel_tensor_for_index(self, index: int) -> torch.Tensor:
        record = self.records[int(index)]
        channel, _ = cached_cir_file_to_csi(
            str(record.channel_path),
            self.csi_method,
            self.channel_domain,
            bool(self.normalize_channel),
        )
        channel = convert_channel_target(channel, self.target_mode)
        return torch.from_numpy(channel)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]

        image = cached_image_tensor(str(record.image_path), self.image_size)

        points = cached_ascii_pcd_xyz(str(record.point_path))
        points = sample_points(points, self.num_points, self.random_lidar)
        points = points - points.mean(axis=0, keepdims=True)
        points = points / float(self.point_scale)
        points = torch.from_numpy(points.astype(np.float32))

        channel, channel_scale = cached_cir_file_to_csi(
            str(record.channel_path),
            self.csi_method,
            self.channel_domain,
            bool(self.normalize_channel),
        )
        channel = convert_channel_target(channel, self.target_mode)

        item = {
            "image": image,
            "points": points,
            "sensor_gps": torch.from_numpy(self._normalize_gps(record.sensor_gps)),
            "target_gps": torch.from_numpy(self._normalize_gps(record.target_gps)),
            "link_geometry": torch.from_numpy(self._link_geometry(record.rsu_gps, record.cav_gps)),
            "channel": torch.from_numpy(channel),
            "channel_scale": torch.tensor(channel_scale, dtype=torch.float32),
            "frame": torch.tensor(record.frame, dtype=torch.long),
            "cav_index": torch.tensor(self.cavs.index(record.cav), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.use_dual_context:
            rsu_points = points
            cav_points_raw = cached_ascii_pcd_xyz(str(record.cav_point_path))
            cav_points = sample_points(cav_points_raw, self.num_points, self.random_lidar)
            cav_points = cav_points - cav_points.mean(axis=0, keepdims=True)
            cav_points = cav_points / float(self.point_scale)
            item.update(
                {
                    "rsu_image": image,
                    "rsu_points": rsu_points,
                    "rsu_gps": torch.from_numpy(self._normalize_gps(record.rsu_gps)),
                    "cav_image": cached_image_tensor(str(record.cav_image_path), self.image_size),
                    "cav_points": torch.from_numpy(cav_points.astype(np.float32)),
                    "cav_gps": torch.from_numpy(self._normalize_gps(record.cav_gps)),
                }
            )
        return item

    def metadata_for_indices(self, indices: Iterable[int]) -> List[Dict]:
        payload = []
        for index in indices:
            record = self.records[int(index)]
            payload.append(
                {
                    "index": int(index),
                    "frame": int(record.frame),
                    "cav": record.cav,
                    "sensor_actor": record.sensor_actor,
                    "image_path": str(record.image_path),
                    "point_path": str(record.point_path),
                    "channel_path": str(record.channel_path),
                    "sensor_gps": record.sensor_gps.tolist(),
                    "target_gps": record.target_gps.tolist(),
                    "rsu_gps": record.rsu_gps.tolist() if record.rsu_gps is not None else [],
                    "cav_gps": record.cav_gps.tolist() if record.cav_gps is not None else [],
                    "link_geometry": self._link_geometry(record.rsu_gps, record.cav_gps).tolist(),
                    "rsu_image_path": str(record.rsu_image_path) if record.rsu_image_path else "",
                    "rsu_point_path": str(record.rsu_point_path) if record.rsu_point_path else "",
                    "cav_image_path": str(record.cav_image_path) if record.cav_image_path else "",
                    "cav_point_path": str(record.cav_point_path) if record.cav_point_path else "",
                }
            )
        return payload

    def config_dict(self) -> Dict:
        return {
            "root": str(self.root),
            "weather": self.weather,
            "town": self.town,
            "sensor_scenario": self.sensor_scenario,
            "channel_scenario": self.channel_scenario,
            "channel_config": self.channel_config,
            "cavs": list(self.cavs),
            "sensor_actor": self.sensor_actor,
            "use_dual_context": self.use_dual_context,
            "camera": self.camera,
            "image_size": self.image_size,
            "num_points": self.num_points,
            "point_scale": self.point_scale,
            "channel_domain": self.channel_domain,
            "csi_method": self.csi_method,
            "normalize_channel": self.normalize_channel,
            "target_mode": self.target_mode,
            "target_channels": self.target_channels,
            "channel_shape": list(self.channel_shape),
            "gps_min": self.gps_min.tolist(),
            "gps_max": self.gps_max.tolist(),
            "link_distance_scale": self.link_distance_scale,
            "link_geometry_dim": self.link_geometry_dim,
        }


def split_dataset(dataset: MMVTjunctionDataset, train_ratio: float, split: str) -> Subset:
    if split == "all":
        return Subset(dataset, list(range(len(dataset))))

    train_count = int(len(dataset) * train_ratio)
    train_count = min(max(train_count, 1), len(dataset) - 1)
    if split == "train":
        indices = list(range(train_count))
    elif split in {"val", "test"}:
        indices = list(range(train_count, len(dataset)))
    else:
        raise ValueError(f"Unsupported split: {split}")
    return Subset(dataset, indices)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
