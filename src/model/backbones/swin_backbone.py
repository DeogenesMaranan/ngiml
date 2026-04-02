"""Swin-Tiny backbone for NGIML contextual feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import logging
import timm
import torch
import torch.nn.functional as NN_F
from torch import nn, Tensor

_LOG = logging.getLogger(__name__)
logging.getLogger("timm.models._builder").setLevel(logging.ERROR)


@dataclass
class SwinBackboneConfig:
    """Configuration for the Swin Transformer feature extractor."""

    model_name: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = True
    out_indices: Sequence[int] = (0, 1, 2, 3)
    input_size: Union[int, Tuple[int, int], None] = 448
    allow_variable_input: bool = True


class SwinBackbone(nn.Module):
    """Thin wrapper around timm Swin Transformer with multi-scale outputs."""

    def __init__(self, config: SwinBackboneConfig | None = None) -> None:
        super().__init__()
        cfg = config or SwinBackboneConfig()
        self.config = cfg
        model_kwargs = {"pretrained": cfg.pretrained, "features_only": True}
        if cfg.input_size is not None:
            if isinstance(cfg.input_size, int):
                model_kwargs["img_size"] = (int(cfg.input_size), int(cfg.input_size))
            else:
                model_kwargs["img_size"] = tuple(int(v) for v in cfg.input_size)
        self.model = timm.create_model(cfg.model_name, **model_kwargs)
        avail_n = len(self.model.feature_info)
        requested = tuple(cfg.out_indices) if cfg.out_indices is not None else tuple(range(avail_n))
        valid_indices = tuple(i for i in requested if 0 <= i < avail_n)
        if not valid_indices:
            valid_indices = tuple(range(avail_n))
        if valid_indices != tuple(requested):
            _LOG.warning(
                "requested swin out_indices %s adjusted to available indices %s for model %s",
                requested,
                valid_indices,
                cfg.model_name,
            )
        self.selected_indices = valid_indices
        self.out_channels: List[int] = [self.model.feature_info[i]["num_chs"] for i in self.selected_indices]
        patch = getattr(self.model, "patch_embed", None)
        if patch is None:
            raise ValueError("Swin backbone missing patch_embed; ensure model_name is a Swin variant")
        self.patch_embed = patch
        if isinstance(self.patch_embed.patch_size, tuple):
            self.patch_size: Tuple[int, int] = self.patch_embed.patch_size
        else:
            self.patch_size = (self.patch_embed.patch_size, self.patch_embed.patch_size)
        self.allow_variable_input = bool(cfg.allow_variable_input)
        if hasattr(self.patch_embed, "strict_img_size"):
            self.patch_embed.strict_img_size = not self.allow_variable_input
        self.stages: List[nn.Module] = [
            module
            for name, module in self.model.named_children()
            if name.startswith("layers_")
        ]
        if not self.stages:
            raise ValueError("Swin backbone structure unexpected; layers_* modules not found")
        self._last_spatial_size: Tuple[int, int] | None = None

        self._pad_multiple = self._compute_pad_multiple()

    @staticmethod
    def _normalize_spatial_size(value: object) -> Tuple[int, int] | None:
        if value is None:
            return None
        if isinstance(value, int):
            size = int(value)
            return (size, size)
        if isinstance(value, (tuple, list)):
            if len(value) == 2:
                return (int(value[0]), int(value[1]))
            if len(value) >= 3:
                return (int(value[-2]), int(value[-1]))
        return None

    def _compute_pad_multiple(self) -> int:
        win = None
        try:
            first_stage = self.stages[0]
            first_block = getattr(first_stage, "blocks", [None])[0]
            if first_block is not None and hasattr(first_block, "attn"):
                wsize = getattr(first_block.attn, "window_size", None)
                if isinstance(wsize, (tuple, list)) and len(wsize) >= 2:
                    win = int(max(wsize[-2], wsize[-1]))
                elif isinstance(wsize, int):
                    win = int(wsize)
        except Exception:
            win = None

        num_stages = max(1, len(self.stages))
        downsample_factor = 2 ** (num_stages - 1)

        patch_mult = max(1, self.patch_size[0], self.patch_size[1])
        window_mult = patch_mult if win is None else max(patch_mult, win * downsample_factor)
        return max(patch_mult, window_mult)

    def _expected_input_size(self) -> Tuple[int, int] | None:
        candidates = [
            getattr(self.model, "img_size", None),
            getattr(self.patch_embed, "img_size", None),
            self.config.input_size,
        ]
        default_cfg = getattr(self.model, "default_cfg", None) or {}
        if isinstance(default_cfg, dict):
            candidates.append(default_cfg.get("input_size"))

        for candidate in candidates:
            normalized = self._normalize_spatial_size(candidate)
            if normalized is not None:
                return normalized
        return None

    def _propagate_spatial_metadata(self, height: int, width: int) -> None:
        ph, pw = self.patch_size
        if height % ph != 0 or width % pw != 0:
            new_h = ((height + ph - 1) // ph) * ph
            new_w = ((width + pw - 1) // pw) * pw
            _LOG.warning(
                "Swin input spatial dims (%d,%d) are not multiples of patch size %s; adjusting to (%d,%d)",
                height,
                width,
                self.patch_size,
                new_h,
                new_w,
            )
            height, width = new_h, new_w

        if self._last_spatial_size == (height, width):
            return

        grid_h = height // self.patch_size[0]
        grid_w = width // self.patch_size[1]

        for stage_idx, stage in enumerate(self.stages):
            scale = 2 ** stage_idx
            stage_res = (
                (grid_h + scale - 1) // scale,
                (grid_w + scale - 1) // scale,
            )
            stage.input_resolution = stage_res
            blocks = getattr(stage, "blocks", [])
            for block in blocks:
                block.input_resolution = stage_res
                if hasattr(block, "attn_mask"):
                    device = None
                    dtype = None
                    if isinstance(block.attn_mask, torch.Tensor):
                        device = block.attn_mask.device
                        dtype = block.attn_mask.dtype
                    block.attn_mask = block.get_attn_mask(device=device, dtype=dtype)

        self._last_spatial_size = (height, width)

    def _ensure_channels_first(self, features: List[Tensor]) -> List[Tensor]:
        if len(features) != len(self.out_channels):
            raise ValueError(
                "Unexpected number of Swin feature maps; review out_indices configuration"
            )

        normalized: List[Tensor] = []
        for idx, (feat, expected_ch) in enumerate(zip(features, self.out_channels)):
            if feat.ndim != 4:
                raise ValueError(
                    f"Swin feature map {idx} must be 4D (NCHW), got shape {tuple(feat.shape)}"
                )

            if feat.shape[1] == expected_ch:
                normalized.append(feat)
                continue

            if feat.shape[-1] == expected_ch:
                normalized.append(feat.permute(0, 3, 1, 2).contiguous())
                continue

            raise ValueError(
                f"Swin feature map {idx} reports {feat.shape[1]} channels but expected {expected_ch}"
            )

        return normalized

    def forward(self, x: Tensor) -> List[Tensor]:
        try:
            first_param = next(self.model.parameters())
            model_dev = first_param.device
        except StopIteration:
            model_dev = None
        if model_dev is not None and model_dev != x.device:
            _LOG.warning(
                "SwinBackbone detected model device %s differs from input device %s. "
                "Avoid per-forward module moves; ensure model is moved to the input device once during setup.",
                model_dev,
                x.device,
            )

        _, _, h, w = x.shape
        ph, pw = self.patch_size
        multiple = self._pad_multiple if self.allow_variable_input else max(ph, pw)
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h or pad_w:
            _LOG.warning(
                "SwinBackbone internal pad to patch multiple: (%d,%d) -> (%d,%d)",
                h, w, h + pad_h, w + pad_w,
            )
            x = NN_F.pad(x, (0, pad_w, 0, pad_h), value=0)
            h, w = x.shape[-2:]

        if self.allow_variable_input:
            grid_h = h // ph
            grid_w = w // pw
            if hasattr(self.patch_embed, "img_size"):
                self.patch_embed.img_size = (h, w)
            if hasattr(self.patch_embed, "grid_size"):
                self.patch_embed.grid_size = (grid_h, grid_w)
            if hasattr(self.patch_embed, "num_patches"):
                self.patch_embed.num_patches = grid_h * grid_w

        expected_size = None if self.allow_variable_input else self._expected_input_size()
        strict_img_size = bool(getattr(self.patch_embed, "strict_img_size", False))
        if expected_size is not None:
            exp_h, exp_w = expected_size
            if strict_img_size and (x.shape[-2], x.shape[-1]) != (exp_h, exp_w):
                _LOG.warning(
                    "SwinBackbone strict-size resize to configured model input: (%d,%d) -> (%d,%d)",
                    x.shape[-2], x.shape[-1], exp_h, exp_w,
                )
                x = NN_F.interpolate(x, size=(exp_h, exp_w), mode="bilinear", align_corners=False)
            self._propagate_spatial_metadata(x.shape[-2], x.shape[-1])
            try:
                features = self.model(x)
            except AssertionError as err:
                _LOG.warning("Swin model assertion during forward at native padded size: %s", str(err))
                if (x.shape[-2], x.shape[-1]) != (exp_h, exp_w):
                    _LOG.warning(
                        "SwinBackbone fallback resize to model default: (%d,%d) -> (%d,%d)",
                        x.shape[-2], x.shape[-1], exp_h, exp_w,
                    )
                    x_resized = NN_F.interpolate(x, size=(exp_h, exp_w), mode="bilinear", align_corners=False)
                    self._propagate_spatial_metadata(exp_h, exp_w)
                    features = self.model(x_resized)
                else:
                    raise
        else:
            self._propagate_spatial_metadata(x.shape[-2], x.shape[-1])
            features = self.model(x)
        selected = [features[i] for i in self.selected_indices]
        return self._ensure_channels_first(selected)


__all__ = ["SwinBackbone", "SwinBackboneConfig"]
