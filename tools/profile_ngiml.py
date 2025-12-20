"""Profile the full NGIML hybrid model, including feature fusion."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, Tuple

import torch

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None

try:
    from torchinfo import summary as torchinfo_summary
except Exception:
    torchinfo_summary = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model.feature_fusion import FeatureFusionConfig
from src.model.hybrid_ngiml import HybridNGIML, HybridNGIMLConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile NGIML hybrid complexity")
    parser.add_argument("--height", type=int, default=256, help="Input image height")
    parser.add_argument("--width", type=int, default=256, help="Input image width")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for profiling")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained weights for deterministic benchmarking",
    )
    parser.add_argument(
        "--fusion-channels",
        type=str,
        default="128,192,256,320",
        help="Comma-separated channel counts for each fusion stage",
    )
    parser.add_argument("--noise-branch", type=str, default="residual", help="Noise branch name")
    parser.add_argument("--noise-decay", type=float, default=1.0, help="Geometric decay for noise gates")
    parser.add_argument(
        "--noise-skip-stage",
        type=int,
        default=-1,
        help="Disable noise input starting at this stage (use -1 to keep all stages)",
    )
    parser.add_argument("--fusion-norm", type=str, default="bn", help="Normalization for fusion blocks")
    parser.add_argument(
        "--fusion-activation",
        type=str,
        default="relu",
        help="Activation for fusion refinement convs",
    )
    parser.add_argument("--disable-low-level", action="store_true", help="Drop EfficientNet branch")
    parser.add_argument("--disable-context", action="store_true", help="Drop Swin branch")
    parser.add_argument("--disable-residual", action="store_true", help="Drop residual noise branch")
    return parser.parse_args()


def _parse_channels(arg: str) -> Tuple[int, ...]:
    values = [int(chunk.strip()) for chunk in arg.split(",") if chunk.strip()]
    if not values:
        raise ValueError("fusion-channels must provide at least one integer")
    return tuple(values)


def instantiate_model(args: argparse.Namespace) -> HybridNGIML:
    cfg = HybridNGIMLConfig()
    cfg.use_low_level = not args.disable_low_level
    cfg.use_context = not args.disable_context
    cfg.use_residual = not args.disable_residual

    fusion_channels = _parse_channels(args.fusion_channels)
    skip_stage = None if args.noise_skip_stage < 0 else args.noise_skip_stage
    cfg.fusion = FeatureFusionConfig(
        fusion_channels=fusion_channels,
        noise_branch=args.noise_branch,
        noise_skip_stage=skip_stage,
        noise_decay=args.noise_decay,
        norm=args.fusion_norm,
        activation=args.fusion_activation,
    )

    if args.no_pretrained:
        cfg.efficientnet.pretrained = False
        cfg.swin.pretrained = False

    model = HybridNGIML(cfg)
    return model


def profile_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def profile_flops(model: torch.nn.Module, sample: torch.Tensor) -> Tuple[float, str]:
    if FlopCountAnalysis is None:
        raise RuntimeError("fvcore is required for FLOP analysis. Install with pip install fvcore")
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, sample)
        total_flops = flops.total()
        return total_flops, flops.__str__()


def print_config_summary(model: HybridNGIML) -> None:
    summary = asdict(model.cfg)
    print("\nModel configuration:")
    print(json.dumps(summary, indent=2, default=str))


def describe_outputs(model: HybridNGIML, sample: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        outputs = model(sample)
    shapes = [tuple(out.shape) for out in outputs]
    print("\nDecoder output tensor shapes:")
    for idx, shape in enumerate(shapes):
        print(f"  Stage {idx}: {shape}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = instantiate_model(args).to(device)

    dummy = torch.randn(args.batch, 3, args.height, args.width, device=device)

    print_config_summary(model)

    total_params = profile_params(model)
    print(f"\nTotal parameters: {total_params:,}")

    if torchinfo_summary is not None:
        print("\nTorchinfo summary:")
        torchinfo_summary(
            model,
            input_size=(args.batch, 3, args.height, args.width),
            col_names=("output_size", "num_params"),
            verbose=0,
        )
    else:
        print("torchinfo is not installed; skipping detailed summary.")

    try:
        total_flops, detail = profile_flops(model, dummy)
        print(f"\nTotal FLOPs: {total_flops / 1e9:.3f} GFLOPs")
    except RuntimeError as err:
        print(f"FLOP profiling skipped: {err}")
    else:
        print("\nFLOP detail:")
        print(detail)

    describe_outputs(model, dummy)


if __name__ == "__main__":
    main()
