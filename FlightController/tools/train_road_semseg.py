"""Train a lightweight road semantic segmentation model from CVAT masks.

The expected export layout is CVAT's Segmentation mask / Pascal VOC style:

    JPEGImages/
    SegmentationClass/
    labelmap.txt

Masks may be RGB color masks. Any non-black pixel is treated as road=1 and
black pixels are treated as background=0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _pil_resampling(name: str) -> int:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


BILINEAR = _pil_resampling("BILINEAR")
NEAREST = _pil_resampling("NEAREST")


@dataclass(frozen=True)
class Sample:
    stem: str
    group: str
    image_path: str
    mask_path: str


class RoadSegDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        size: int,
        train: bool,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        cache_images: bool,
    ) -> None:
        self.samples = samples
        self.size = size
        self.train = train
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if cache_images:
            self.cache = []
            for sample in self.samples:
                image = Image.open(sample.image_path).convert("RGB").resize(
                    (self.size, self.size), BILINEAR
                )
                mask_rgb = Image.open(sample.mask_path).convert("RGB").resize(
                    (self.size, self.size), NEAREST
                )
                self.cache.append((np.array(image, copy=True), np.array(mask_rgb, copy=True)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        if self.cache is None:
            image = Image.open(sample.image_path).convert("RGB").resize(
                (self.size, self.size), BILINEAR
            )
            mask_rgb = Image.open(sample.mask_path).convert("RGB").resize(
                (self.size, self.size), NEAREST
            )
        else:
            image_arr, mask_arr = self.cache[idx]
            image = Image.fromarray(image_arr)
            mask_rgb = Image.fromarray(mask_arr)

        if self.train:
            image, mask_rgb = self._augment(image, mask_rgb)

        img = np.asarray(image, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self.mean) / self.std

        mask_arr = np.asarray(mask_rgb, dtype=np.uint8)
        mask = (mask_arr.reshape(mask_arr.shape[0], mask_arr.shape[1], 3).any(axis=2)).astype(
            np.int64
        )

        return torch.from_numpy(img), torch.from_numpy(mask)

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.35))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.70, 1.35))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.25))

        if random.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))

        if random.random() < 0.35:
            angle = random.uniform(-4.0, 4.0)
            image = image.rotate(angle, resample=BILINEAR, fillcolor=(0, 0, 0))
            mask = mask.rotate(angle, resample=NEAREST, fillcolor=(0, 0, 0))

        return image, mask


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DSConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(
            ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch),
            ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1),
        )


class RoadFastSeg(nn.Module):
    """Small static-shape semantic segmentation network for 256/320 inputs."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = ConvBNReLU(3, 16, stride=2)
        self.enc1 = DSConv(16, 24, stride=2)
        self.enc2 = DSConv(24, 40, stride=2)
        self.enc3 = DSConv(40, 64, stride=2)
        self.context = nn.Sequential(
            DSConv(64, 96, stride=1),
            DSConv(96, 96, stride=1),
            ConvBNReLU(96, 64, kernel_size=1),
        )

        self.skip2 = ConvBNReLU(40, 64, kernel_size=1)
        self.skip1 = ConvBNReLU(24, 48, kernel_size=1)
        self.skip0 = ConvBNReLU(16, 32, kernel_size=1)

        self.dec2 = DSConv(64, 64)
        self.dec1_reduce = ConvBNReLU(64, 48, kernel_size=1)
        self.dec1 = DSConv(48, 48)
        self.dec0_reduce = ConvBNReLU(48, 32, kernel_size=1)
        self.dec0 = DSConv(32, 32)
        self.final = nn.Sequential(
            DSConv(32, 24),
            nn.Conv2d(24, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        x = self.enc3(s2)
        x = self.context(x)

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.dec2(x + self.skip2(s2))

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.dec1_reduce(x)
        x = self.dec1(x + self.skip1(s1))

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.dec0_reduce(x)
        x = self.dec0(x + self.skip0(s0))

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.final(x)


class InputNormalizeWrapper(nn.Module):
    """Embed RGB 0-1 to training normalization into the exported ONNX graph."""

    def __init__(
        self,
        model: nn.Module,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.model = model
        scale = torch.tensor([1.0 / value for value in std], dtype=torch.float32).view(1, 3, 1, 1)
        bias = torch.tensor([-mean[i] / std[i] for i in range(3)], dtype=torch.float32).view(
            1, 3, 1, 1
        )
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x * self.scale + self.bias)


def discover_samples(data_root: Path) -> list[Sample]:
    image_dir = data_root / "JPEGImages"
    mask_dir = data_root / "SegmentationClass"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing image directory: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"missing mask directory: {mask_dir}")

    samples: list[Sample] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        image_path = None
        for ext in IMAGE_EXTS:
            candidate = image_dir / f"{mask_path.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            raise FileNotFoundError(f"no matching image found for mask {mask_path.name}")

        group_match = re.match(r"(.+)_\d+$", mask_path.stem)
        group = group_match.group(1) if group_match else mask_path.stem
        samples.append(Sample(mask_path.stem, group, str(image_path), str(mask_path)))

    if not samples:
        raise RuntimeError(f"no .png masks found under {mask_dir}")
    return samples


def split_samples(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    by_group: dict[str, list[Sample]] = {}
    for sample in samples:
        by_group.setdefault(sample.group, []).append(sample)
    for group_samples in by_group.values():
        group_samples.sort(key=lambda s: s.stem)

    groups = sorted(by_group)
    if len(groups) >= 3:
        train_groups = groups[:-2]
        val_groups = [groups[-2]]
        test_groups = [groups[-1]]
        return (
            [s for g in train_groups for s in by_group[g]],
            [s for g in val_groups for s in by_group[g]],
            [s for g in test_groups for s in by_group[g]],
        )

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def compute_image_stats(samples: Iterable[Sample], size: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB").resize((size, size), BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        total += flat.sum(axis=0)
        total_sq += (flat * flat).sum(axis=0)
        count += flat.shape[0]
    mean = total / count
    var = np.maximum(total_sq / count - mean * mean, 1e-6)
    std = np.sqrt(var)
    return tuple(float(x) for x in mean), tuple(float(x) for x in std)


def compute_class_weights(samples: Iterable[Sample]) -> torch.Tensor:
    pixels = np.zeros(2, dtype=np.float64)
    for sample in samples:
        mask_rgb = Image.open(sample.mask_path).convert("RGB")
        arr = np.asarray(mask_rgb, dtype=np.uint8)
        road = arr.reshape(-1, 3).any(axis=1)
        pixels[1] += road.sum()
        pixels[0] += road.size - road.sum()
    freq = pixels / pixels.sum()
    weights = 1.0 / np.log(1.02 + freq)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    target_float = (target == 1).float()
    inter = (probs * target_float).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    confusion = torch.zeros((2, 2), dtype=torch.float64, device=device)
    bottom_confusion = torch.zeros((2, 2), dtype=torch.float64, device=device)
    ce = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = ce(logits, targets) + dice_loss(logits, targets)
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item())
        total_batches += 1

        for true_class in (0, 1):
            for pred_class in (0, 1):
                confusion[true_class, pred_class] += ((targets == true_class) & (preds == pred_class)).sum()

        bottom_start = targets.shape[-2] // 2
        bottom_targets = targets[:, bottom_start:, :]
        bottom_preds = preds[:, bottom_start:, :]
        for true_class in (0, 1):
            for pred_class in (0, 1):
                bottom_confusion[true_class, pred_class] += (
                    (bottom_targets == true_class) & (bottom_preds == pred_class)
                ).sum()

    def _iou(cm: torch.Tensor, class_idx: int) -> float:
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        return float((tp / torch.clamp(tp + fp + fn, min=1.0)).item())

    road_iou = _iou(confusion, 1)
    bg_iou = _iou(confusion, 0)
    bottom_road_iou = _iou(bottom_confusion, 1)
    accuracy = float((confusion.trace() / torch.clamp(confusion.sum(), min=1.0)).item())
    return {
        "loss": total_loss / max(total_batches, 1),
        "road_iou": road_iou,
        "mean_iou": (road_iou + bg_iou) / 2.0,
        "bottom_road_iou": bottom_road_iou,
        "pixel_accuracy": accuracy,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    class_weights: torch.Tensor,
    amp: bool,
) -> float:
    model.train()
    ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    running = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = ce(logits, targets) + dice_loss(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item())
    return running / max(len(loader), 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    metadata: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata,
        },
        path,
    )


def export_onnx(
    model: nn.Module,
    output_path: Path,
    size: int,
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> None:
    export_model = InputNormalizeWrapper(model, mean, std).to(device)
    export_model.eval()
    dummy = torch.zeros(1, 3, size, size, dtype=torch.float32, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    result = session.run(None, {"images": np.zeros((1, 3, size, size), dtype=np.float32)})
    if result[0].shape != (1, 2, size, size):
        raise RuntimeError(f"unexpected ONNX output shape: {result[0].shape}")


def write_split(path: Path, samples: list[Sample]) -> None:
    path.write_text("\n".join(sample.stem for sample in samples) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("FlightController/Solutions/model/road_semseg_runs/latest"),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--export-name", default="road_fastseg_256_fp32.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(args.data_root)
    train_samples, val_samples, test_samples = split_samples(
        samples, args.val_ratio, args.test_ratio, args.seed
    )
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("split produced an empty train/val/test set")

    mean, std = compute_image_stats(train_samples, args.size)
    class_weights = compute_class_weights(train_samples)

    metadata = {
        "data_root": str(args.data_root),
        "size": args.size,
        "num_classes": 2,
        "class_order": ["background", "road"],
        "mask_rule": "RGB non-black -> road=1, black -> background=0",
        "input_layout": "NCHW",
        "input_range_after_loader": "normalized float32",
        "onnx_input_range": "RGB float32 0.0-1.0; mean/std normalization is embedded in ONNX",
        "mean": mean,
        "std": std,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "train_groups": sorted({sample.group for sample in train_samples}),
        "val_groups": sorted({sample.group for sample in val_samples}),
        "test_groups": sorted({sample.group for sample in test_samples}),
        "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(output_dir)},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_split(output_dir / "train.txt", train_samples)
    write_split(output_dir / "val.txt", val_samples)
    write_split(output_dir / "test.txt", test_samples)

    cache_images = not args.no_cache
    train_ds = RoadSegDataset(
        train_samples, args.size, train=True, mean=mean, std=std, cache_images=cache_images
    )
    val_ds = RoadSegDataset(
        val_samples, args.size, train=False, mean=mean, std=std, cache_images=cache_images
    )
    test_ds = RoadSegDataset(
        test_samples, args.size, train=False, mean=mean, std=std, cache_images=cache_images
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadFastSeg(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = bool(device.type == "cuda" and not args.no_amp)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    total_params = sum(p.numel() for p in model.parameters())
    print(json.dumps(metadata, ensure_ascii=False))
    print(f"device={device} amp={amp} params={total_params:,} class_weights={class_weights.tolist()}")

    best_road_iou = -math.inf
    log_path = output_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "val_loss",
                "val_road_iou",
                "val_mean_iou",
                "val_bottom_road_iou",
                "val_pixel_accuracy",
            ],
        )
        writer.writeheader()

        start = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, device, class_weights, amp
            )
            scheduler.step()
            val_metrics = evaluate(model, val_loader, device)
            lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_road_iou": val_metrics["road_iou"],
                "val_mean_iou": val_metrics["mean_iou"],
                "val_bottom_road_iou": val_metrics["bottom_road_iou"],
                "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            }
            writer.writerow(row)
            fp.flush()

            print(
                "epoch={epoch:03d} lr={lr:.6f} train_loss={train_loss:.4f} "
                "val_loss={val_loss:.4f} road_iou={val_road_iou:.4f} "
                "mean_iou={val_mean_iou:.4f} bottom_iou={val_bottom_road_iou:.4f} "
                "acc={val_pixel_accuracy:.4f}".format(**row),
                flush=True,
            )

            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, val_metrics, metadata)
            if val_metrics["road_iou"] > best_road_iou:
                best_road_iou = val_metrics["road_iou"]
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer, epoch, val_metrics, metadata
                )

        elapsed = time.time() - start
        print(f"training_elapsed_sec={elapsed:.1f}")

    best_ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    print("test_metrics=" + json.dumps(test_metrics), flush=True)

    onnx_path = output_dir / args.export_name
    export_onnx(model, onnx_path, args.size, device, mean, std)
    print(f"exported_onnx={onnx_path}")


if __name__ == "__main__":
    main()
