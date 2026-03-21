from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.infer_helpers import infer_from_image_path, load_model_from_checkpoint


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local image inference for all checkpoints in ./checkpoints over all images in ./samples. "
            "Model architecture and key inference settings are reconstructed from each checkpoint."
        )
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"), help="Directory containing .pt checkpoints")
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"), help="Directory containing input images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inference_outputs"),
        help="Directory to write per-checkpoint inference results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cuda, cuda:0, or cpu. Defaults to cuda when available.",
    )
    return parser.parse_args()


def discover_checkpoints(checkpoints_dir: Path) -> list[Path]:
    checkpoints = sorted([p for p in checkpoints_dir.rglob("*.pt") if p.is_file()])
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in {checkpoints_dir}")
    return checkpoints


def discover_sample_images(samples_dir: Path) -> list[Path]:
    images = sorted(
        [
            path
            for path in samples_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )
    if not images:
        raise FileNotFoundError(
            f"No supported images found in {samples_dir}. Supported suffixes: {sorted(IMAGE_SUFFIXES)}"
        )
    return images


def _read_train_config_field(checkpoint_path: Path, key: str, default):
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
        train_config = payload.get("train_config") if isinstance(payload, dict) else None
        if isinstance(train_config, dict) and key in train_config:
            return train_config.get(key, default)
    except Exception:
        pass
    return default


def _save_mask_png(mask_2d: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mask_2d, mode="L")
    image.save(output_path)


def _save_overlay_png(
    image_chw: torch.Tensor,
    binary_mask_2d: np.ndarray,
    output_path: Path,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = image_chw.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = np.clip(np.round(image_np * 255.0), 0, 255).astype(np.uint8)

    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    elif image_np.shape[-1] > 3:
        image_np = image_np[:, :, :3]

    overlay = image_np.copy().astype(np.float32)
    mask = binary_mask_2d.astype(bool)
    tint = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * tint

    overlay_uint8 = np.clip(np.round(overlay), 0, 255).astype(np.uint8)
    Image.fromarray(overlay_uint8, mode="RGB").save(output_path)


def run() -> None:
    args = parse_args()

    checkpoints_dir = args.checkpoints_dir.resolve()
    samples_dir = args.samples_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory does not exist: {checkpoints_dir}")
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory does not exist: {samples_dir}")

    checkpoints = discover_checkpoints(checkpoints_dir)
    sample_images = discover_sample_images(samples_dir)

    selected_device = torch.device(args.device) if args.device else None

    print(f"Found {len(checkpoints)} checkpoint(s) under: {checkpoints_dir}")
    print(f"Found {len(sample_images)} image(s) under: {samples_dir}")

    for ckpt_path in checkpoints:
        print(f"\n[Checkpoint] {ckpt_path.name}")
        model, device, info = load_model_from_checkpoint(ckpt_path, device=selected_device)

        normalization_mode = str(_read_train_config_field(ckpt_path, "normalization_mode", "zero_one")).strip().lower()
        if normalization_mode not in {"zero_one", "imagenet"}:
            normalization_mode = "zero_one"

        threshold = float(info.get("default_threshold", 0.5))
        max_short_side = int(info.get("max_short_side", 0) or 0)

        checkpoint_output_dir = output_dir / ckpt_path.stem
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "checkpoint": str(ckpt_path),
            "device": str(device),
            "config_source": str(info.get("config_source", "unknown")),
            "threshold": threshold,
            "threshold_source": str(info.get("threshold_source", "unknown")),
            "max_short_side": max_short_side,
            "final_pred_stage": int(info.get("final_pred_stage", -1)),
            "normalization_mode": normalization_mode,
            "images": [],
        }

        for image_path in sample_images:
            image_tensor, prob_map = infer_from_image_path(
                model,
                image_path=image_path,
                device=device,
                normalization_mode=normalization_mode,
                max_short_side=max_short_side,
            )
            prob_np = prob_map.numpy().astype(np.float32)
            bin_np = (prob_np >= threshold).astype(np.uint8)

            rel_parent = image_path.parent.relative_to(samples_dir)
            base_name = image_path.stem

            prob_out = checkpoint_output_dir / rel_parent / f"{base_name}_prob.png"
            bin_out = checkpoint_output_dir / rel_parent / f"{base_name}_bin.png"
            overlay_out = checkpoint_output_dir / rel_parent / f"{base_name}_overlay.png"

            prob_img = np.clip(np.round(prob_np * 255.0), 0, 255).astype(np.uint8)
            bin_img = (bin_np * 255).astype(np.uint8)

            _save_mask_png(prob_img, prob_out)
            _save_mask_png(bin_img, bin_out)
            _save_overlay_png(image_tensor, bin_np, overlay_out)

            metadata["images"].append(
                {
                    "input": str(image_path),
                    "probability_map": str(prob_out),
                    "binary_map": str(bin_out),
                    "overlay_map": str(overlay_out),
                }
            )
            print(f"  inferred {image_path.relative_to(samples_dir)}")

        metadata_path = checkpoint_output_dir / "inference_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"  wrote metadata: {metadata_path}")

    print(f"\nInference complete. Results written to: {output_dir}")


if __name__ == "__main__":
    run()
