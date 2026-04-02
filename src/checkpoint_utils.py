from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch


def select_highest_resolution_head(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    if not outputs:
        raise ValueError("Model returned empty predictions list")
    return outputs[0]


def parse_checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"checkpoint_epoch_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def disable_pretrained_backbones_for_checkpoint_load(model_cfg: object) -> object:
    """Best-effort disable of backbone pretrained flags for checkpoint-only init."""
    cfg_out = model_cfg
    for attr in ("efficientnet", "swin"):
        branch = getattr(cfg_out, attr, None)
        if branch is None or not hasattr(branch, "pretrained"):
            continue

        replaced = False
        try:
            cfg_out = replace(cfg_out, **{attr: replace(branch, pretrained=False)})
            replaced = True
        except Exception:
            replaced = False

        if not replaced:
            try:
                getattr(cfg_out, attr).pretrained = False
            except Exception:
                pass

    return cfg_out