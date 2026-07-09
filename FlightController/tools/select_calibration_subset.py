"""Select representative road calibration images and build ST Cloud .npz packs.

The selector favors diversity over visual niceness. It extracts lightweight
image statistics from all candidate frames, optionally appends ONNX model output
statistics when onnxruntime is available, then picks a spread-out subset using
extreme samples plus greedy farthest-point sampling.

Example:

    python FlightController/tools/select_calibration_subset.py ^
      --image-dir D:/drone2/adjustment/roads ^
      --model FlightController/Solutions/model/stcloud_upload/road_yolo11n_seg_fp32_for_stcloud.onnx ^
      --counts 64 96 128
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - only hit on broken local envs
    raise RuntimeError(
        "select_calibration_subset.py requires OpenCV. Install opencv-python "
        "or run it inside .venv_inspect."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = Path("D:/drone2/adjustment/roads")
DEFAULT_MODEL = (
    ROOT
    / "FlightController"
    / "Solutions"
    / "model"
    / "stcloud_upload"
    / "road_yolo11n_seg_fp32_for_stcloud.onnx"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "FlightController" / "Solutions" / "model" / "stcloud_upload" / "selected"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    feature: np.ndarray
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select representative road images for ST Cloud calibration."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--counts", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--input-size", type=int, default=416)
    parser.add_argument(
        "--use-model-features",
        action="store_true",
        help="Append ONNX output statistics if onnxruntime is installed.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy selected source images into per-count folders for visual review.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def letterbox_rgb(image_rgb: np.ndarray, input_size: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    scale = min(input_size / float(width), input_size / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    left = int(round((input_size - new_w) / 2.0))
    top = int(round((input_size - new_h) / 2.0))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def preprocess_for_model(path: Path, input_size: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = letterbox_rgb(rgb, input_size)
    arr = image.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, ...].astype(np.float32)


def image_feature(path: Path, index: int, total: int) -> tuple[np.ndarray, dict[str, float]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
    arr = small.astype(np.float32) / 255.0
    gray_u8 = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    edges = cv2.Canny(gray_u8, 64, 160).astype(np.float32) / 255.0

    channel_mean = arr.mean(axis=(0, 1))
    channel_std = arr.std(axis=(0, 1))
    brightness = float(gray.mean())
    contrast = float(gray.std())
    edge_mean = float(edges.mean())

    max_rgb = arr.max(axis=2)
    min_rgb = arr.min(axis=2)
    saturation = np.where(max_rgb > 1e-6, (max_rgb - min_rgb) / max_rgb, 0.0)
    saturation_mean = float(saturation.mean())
    saturation_std = float(saturation.std())

    # Coarse color histograms are intentionally low-dimensional and robust.
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=8, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(1.0, float(hist.sum()))
        hist_parts.append(hist)

    # Tiny spatial thumbnail keeps scene layout differences, not full pixels.
    thumb = cv2.resize(rgb, (12, 12), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    thumb = (thumb / 255.0).reshape(-1)

    temporal = float(index / max(1, total - 1))
    metrics = {
        "brightness": brightness,
        "contrast": contrast,
        "edge_mean": edge_mean,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "red_mean": float(channel_mean[0]),
        "green_mean": float(channel_mean[1]),
        "blue_mean": float(channel_mean[2]),
        "temporal": temporal,
    }
    scalar = np.array(
        [
            brightness,
            contrast,
            edge_mean,
            saturation_mean,
            saturation_std,
            *channel_mean.tolist(),
            *channel_std.tolist(),
            temporal,
        ],
        dtype=np.float32,
    )
    feature = np.concatenate([scalar, *hist_parts, thumb.astype(np.float32)])
    return feature, metrics


def load_onnx_session(model_path: Path):
    try:
        import onnxruntime as ort
    except Exception:
        return None, "onnxruntime is not installed"
    if not model_path.is_file():
        return None, f"model not found: {model_path}"
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return None, f"onnxruntime failed to load model: {exc}"
    return session, ""


def model_feature(session, path: Path, input_size: int) -> tuple[np.ndarray, dict[str, float]]:
    input_name = session.get_inputs()[0].name
    blob = preprocess_for_model(path, input_size)
    outputs = session.run(None, {input_name: blob})
    values = []
    metrics: dict[str, float] = {}

    for index, out in enumerate(outputs):
        arr = np.asarray(out, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            stats = np.zeros(7, dtype=np.float32)
        else:
            stats = np.array(
                [
                    float(finite.mean()),
                    float(finite.std()),
                    float(finite.min()),
                    float(finite.max()),
                    float(np.quantile(finite, 0.05)),
                    float(np.quantile(finite, 0.50)),
                    float(np.quantile(finite, 0.95)),
                ],
                dtype=np.float32,
            )
        values.append(stats)
        metrics[f"model_out{index}_mean"] = float(stats[0])
        metrics[f"model_out{index}_std"] = float(stats[1])
        metrics[f"model_out{index}_max"] = float(stats[3])

    return np.concatenate(values) if values else np.zeros(1, dtype=np.float32), metrics


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (features - mean) / std
    return np.nan_to_num(z, copy=False)


def add_extremes(records: list[ImageRecord], selected: list[int]) -> None:
    metric_names = [
        "brightness",
        "contrast",
        "edge_mean",
        "saturation_mean",
        "red_mean",
        "green_mean",
        "blue_mean",
    ]
    selected_set = set(selected)
    for name in metric_names:
        values = np.array([record.metrics[name] for record in records], dtype=np.float32)
        for idx in (int(values.argmin()), int(values.argmax())):
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)


def farthest_point_select(features: np.ndarray, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    n = len(features)
    if count >= n:
        return list(range(n))

    selected: list[int] = []
    # Start from a deterministic point near the global center plus metric extremes.
    center = features.mean(axis=0, keepdims=True)
    distances_to_center = np.linalg.norm(features - center, axis=1)
    selected.append(int(distances_to_center.argmin()))

    min_dist = np.linalg.norm(features - features[selected[0]], axis=1)
    min_dist[selected[0]] = -1.0
    while len(selected) < count:
        if len(selected) == 1:
            # Break ties in a stable but non-pathological way.
            jitter = rng.random(n) * 1e-6
            next_idx = int(np.argmax(min_dist + jitter))
        else:
            next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist = np.linalg.norm(features - features[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected] = -1.0
    return selected


def select_indices(records: list[ImageRecord], count: int, seed: int) -> list[int]:
    raw = np.stack([record.feature for record in records], axis=0)
    features = standardize(raw)

    selected: list[int] = []
    add_extremes(records, selected)
    if len(selected) >= count:
        return sorted(selected[:count])

    selected_set = set(selected)
    if not selected:
        selected = farthest_point_select(features, count, seed)
    else:
        min_dist = np.full(len(records), np.inf, dtype=np.float32)
        for idx in selected:
            min_dist = np.minimum(min_dist, np.linalg.norm(features - features[idx], axis=1))
        for idx in selected:
            min_dist[idx] = -1.0
        while len(selected) < count:
            next_idx = int(np.argmax(min_dist))
            if next_idx in selected_set:
                break
            selected.append(next_idx)
            selected_set.add(next_idx)
            dist = np.linalg.norm(features - features[next_idx], axis=1)
            min_dist = np.minimum(min_dist, dist)
            for idx in selected:
                min_dist[idx] = -1.0

    return sorted(selected)


def write_npz(output_path: Path, paths: Iterable[Path], input_size: int) -> None:
    arrays = [preprocess_for_model(path, input_size)[0] for path in paths]
    stacked = np.stack(arrays, axis=0).astype(np.float32)
    np.savez_compressed(output_path, images=stacked)


def write_csv(output_path: Path, selected: list[ImageRecord]) -> None:
    metric_keys = sorted({key for record in selected for key in record.metrics})
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "path", *metric_keys])
        writer.writeheader()
        for record in selected:
            row = {"filename": record.path.name, "path": str(record.path)}
            row.update({key: record.metrics.get(key, "") for key in metric_keys})
            writer.writerow(row)


def copy_images(output_dir: Path, selected: list[ImageRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(selected, start=1):
        dst = output_dir / f"{index:03d}_{record.path.name}"
        shutil.copy2(record.path, dst)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(args.image_dir)

    session = None
    model_status = "disabled"
    if args.use_model_features:
        session, model_status = load_onnx_session(args.model)
        if session is not None:
            model_status = "enabled"
        else:
            print(f"[WARN] model features disabled: {model_status}", flush=True)

    print(f"images: {len(image_paths)} from {args.image_dir}", flush=True)
    print(f"model features: {model_status}", flush=True)

    records: list[ImageRecord] = []
    t0 = time.perf_counter()
    for index, path in enumerate(image_paths):
        feat, metrics = image_feature(path, index, len(image_paths))
        if session is not None:
            m_feat, m_metrics = model_feature(session, path, args.input_size)
            feat = np.concatenate([feat, m_feat])
            metrics.update(m_metrics)
        records.append(ImageRecord(path=path, feature=feat, metrics=metrics))
        if (index + 1) % 25 == 0 or index + 1 == len(image_paths):
            print(f"processed {index + 1}/{len(image_paths)}", flush=True)

    manifest: dict[str, object] = {
        "created_at_epoch": int(time.time()),
        "image_dir": str(args.image_dir),
        "model": str(args.model),
        "model_features": model_status,
        "input_size": args.input_size,
        "source_count": len(records),
        "outputs": [],
    }

    for count in sorted(set(args.counts)):
        indices = select_indices(records, count, args.seed + count)
        selected = [records[index] for index in indices]
        stem = f"road_calib_selected_{count:03d}"
        npz_path = args.output_dir / f"{stem}.npz"
        csv_path = args.output_dir / f"{stem}.csv"
        write_npz(npz_path, [record.path for record in selected], args.input_size)
        write_csv(csv_path, selected)
        if args.copy_images:
            copy_images(args.output_dir / stem, selected)

        manifest["outputs"].append(
            {
                "count": count,
                "npz": str(npz_path),
                "csv": str(csv_path),
                "compressed_mb": round(npz_path.stat().st_size / 1024 / 1024, 2),
                "uncompressed_mb": round(
                    count * 3 * args.input_size * args.input_size * 4 / 1024 / 1024,
                    2,
                ),
                "filenames": [record.path.name for record in selected],
            }
        )
        print(
            f"wrote {npz_path.name}: count={count} "
            f"compressed={npz_path.stat().st_size / 1024 / 1024:.2f}MB",
            flush=True,
        )

    manifest_path = args.output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)
    print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
