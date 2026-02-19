from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from src.data.dataloaders import _load_from_npz, _load_from_tar_npz, _load_image, load_manifest
from src.data.config import SampleRecord
from src.model.hybrid_ngiml import HybridNGIML
from tools.colab_train_helpers import build_default_components
from tools.local_train_helpers import build_manifest_from_prepared


def find_latest_checkpoint(runs_root: Path) -> Path:
    runs_root = Path(runs_root)
    candidates = sorted(runs_root.rglob("checkpoints/checkpoint_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {runs_root}/**/checkpoints/checkpoint_epoch_*.pt")
    return candidates[-1]


def ensure_local_manifest(prepared_root: Path, manifest_path: Path | None = None) -> Path:
    prepared_root = Path(prepared_root)
    if manifest_path is not None and Path(manifest_path).exists():
        return Path(manifest_path)

    default_manifest = prepared_root / "manifest_local.json"
    if default_manifest.exists() and default_manifest.stat().st_size > 0:
        return default_manifest

    return build_manifest_from_prepared(prepared_root, manifest_out=default_manifest)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device | None = None) -> tuple[HybridNGIML, torch.device, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg, _, _, _ = build_default_components()
    model = HybridNGIML(model_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    info = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    return model, device, info


def select_manifest_sample(
    manifest_path: Path,
    split_priority: Sequence[str] = ("test", "val", "train"),
    fake_only: bool = True,
) -> SampleRecord:
    manifest = load_manifest(manifest_path)
    samples = manifest.samples

    if fake_only:
        fake_samples = [s for s in samples if int(getattr(s, "label", 0)) == 1 or s.mask_path is not None]
    else:
        fake_samples = samples

    for split_name in split_priority:
        split_samples = [s for s in fake_samples if s.split == split_name]
        if split_samples:
            return split_samples[0]

    if fake_samples:
        return fake_samples[0]

    raise RuntimeError(f"No samples available in manifest: {manifest_path}")


def _resolve_possible_local_path(path_str: str) -> str:
    path = Path(path_str)
    return path.as_posix()


def load_image_mask_from_record(record: SampleRecord) -> tuple[torch.Tensor, torch.Tensor]:
    image_path = str(record.image_path)
    if "::" in image_path and image_path.endswith(".npz"):
        image, mask, _ = _load_from_tar_npz(image_path)
    elif image_path.endswith(".npz"):
        image, mask, _ = _load_from_npz(_resolve_possible_local_path(image_path))
    else:
        image = _load_image(_resolve_possible_local_path(image_path))
        mask = None
        if record.mask_path is not None:
            loaded = _load_image(_resolve_possible_local_path(record.mask_path))
            mask = loaded[:1] if loaded.shape[0] > 1 else loaded

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0

    if mask is None:
        mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.float32)
    else:
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] > 1:
            mask = mask[:1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        if tuple(mask.shape[-2:]) != tuple(image.shape[-2:]):
            mask = F.interpolate(mask.unsqueeze(0), size=image.shape[-2:], mode="nearest").squeeze(0)

    return image, mask


def predict_probability_map(model: HybridNGIML, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x, target_size=image.shape[-2:])[-1]
        prob = torch.sigmoid(logits)[0, 0].detach().cpu()
    return prob


def infer_from_image_path(model: HybridNGIML, image_path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = _load_image(str(Path(image_path).as_posix())).float()
    if image.max() > 1.0:
        image = image / 255.0
    pred = predict_probability_map(model, image, device)
    return image, pred
