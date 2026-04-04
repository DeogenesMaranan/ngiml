from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import torch

from src.training_types import Checkpoint, TrainConfig


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


def _checkpoint_epoch_sort_key(path: Path) -> int:
    parsed = parse_checkpoint_epoch(path)
    return int(parsed) if parsed is not None else -1


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    global_step: int,
    cfg: TrainConfig,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema_model: Optional[torch.nn.Module] = None,
    use_ema_for_model_state: bool = False,
    training_state: Optional[dict] = None,
) -> None:
    model_state = ema_model.state_dict() if (use_ema_for_model_state and ema_model is not None) else model.state_dict()
    ckpt = Checkpoint(
        epoch=epoch,
        global_step=global_step,
        model_state=model_state,
        raw_model_state=model.state_dict() if (use_ema_for_model_state and ema_model is not None) else None,
        ema_state=ema_model.state_dict() if ema_model is not None else None,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
        train_config=asdict(cfg),
        training_state=training_state,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt.__dict__, path)


def append_checkpoint_log(path: Path, record: dict) -> None:
    """Append checkpoint metrics using an atomic JSON rewrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def _backup_corrupt(existing: Path) -> None:
        try:
            ts = int(time.time())
            corrupt = existing.with_name(f"{existing.name}.corrupt.{ts}")
            existing.replace(corrupt)
            print(f"Backed up corrupt checkpoint log to {corrupt}")
        except Exception:
            pass

    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                records = [item for item in payload if isinstance(item, dict)]
            elif isinstance(payload, dict):
                records = [payload]
        except Exception as exc:
            print(f"Warning: failed to read existing checkpoint log {path}: {exc}")
            _backup_corrupt(path)
            records = []
    else:
        legacy_jsonl = path.with_suffix(".jsonl")
        if legacy_jsonl.exists() and legacy_jsonl.stat().st_size > 0:
            try:
                with open(legacy_jsonl, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if isinstance(item, dict):
                            records.append(item)
            except Exception as exc:
                print(f"Warning: failed to read legacy jsonl checkpoint log {legacy_jsonl}: {exc}")
                _backup_corrupt(legacy_jsonl)
                records = []

    records.append(record)

    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema_model: Optional[torch.nn.Module] = None,
) -> tuple[int, int, dict]:
    """Load checkpoint state and fall back to older checkpoints on corruption."""
    original_exc: Exception | None = None
    checkpoint_map_location: str | torch.device = "cpu"

    def _move_optimizer_state_to_device(opt: torch.optim.Optimizer, target_device: torch.device) -> None:
        for state in opt.state.values():
            if isinstance(state, dict):
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device=target_device, non_blocking=(target_device.type == "cuda"))

    def _attempt_load(p: Path):
        try:
            return torch.load(p, map_location=checkpoint_map_location), p
        except Exception as exc:
            return exc, p

    loaded_obj, loaded_path = _attempt_load(path)
    if isinstance(loaded_obj, Exception):
        original_exc = loaded_obj
        print(f"Failed to load checkpoint {path}: {loaded_obj}")
        cand_dir = path.parent
        try:
            candidates = sorted(cand_dir.glob("checkpoint_epoch_*.pt"), key=_checkpoint_epoch_sort_key)
        except Exception:
            candidates = []

        for cand in reversed(candidates):
            if cand == path:
                continue
            cand_obj, cand_path = _attempt_load(cand)
            if not isinstance(cand_obj, Exception):
                print(f"Loaded fallback checkpoint {cand}")
                loaded_obj, loaded_path = cand_obj, cand_path
                break
            else:
                print(f"Skipping unreadable checkpoint {cand}: {cand_obj}")

    if isinstance(loaded_obj, Exception):
        raise RuntimeError(f"Unable to load checkpoint {path} or any fallback checkpoints: {original_exc}") from original_exc

    data = loaded_obj
    model_state = data.get("raw_model_state") or data["model_state"]
    model.load_state_dict(model_state)
    if ema_model is not None:
        if data.get("ema_state") is not None:
            ema_model.load_state_dict(data["ema_state"])
        else:
            ema_model.load_state_dict(model.state_dict())
    optimizer.load_state_dict(data["optimizer_state"])
    _move_optimizer_state_to_device(optimizer, device)
    if scheduler is not None and data.get("scheduler_state") is not None:
        scheduler.load_state_dict(data["scheduler_state"])
    if data.get("scaler_state") and scaler.is_enabled():
        scaler.load_state_dict(data["scaler_state"])

    raw_epoch = data.get("epoch")
    if raw_epoch is None or int(raw_epoch) == 0:
        parsed = parse_checkpoint_epoch(loaded_path) if loaded_path is not None else None
        parsed_epoch = int(parsed) if parsed is not None else -1
        if parsed_epoch > 0:
            start_epoch = parsed_epoch
        else:
            start_epoch = int(raw_epoch or 0)
    else:
        start_epoch = int(raw_epoch)

    global_step = int(data.get("global_step", 0))
    training_state = data.get("training_state")
    if not isinstance(training_state, dict):
        training_state = {}
    return start_epoch, global_step, training_state


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=_checkpoint_epoch_sort_key)
    return candidates[-1] if candidates else None


__all__ = [
    "append_checkpoint_log",
    "disable_pretrained_backbones_for_checkpoint_load",
    "find_latest_checkpoint",
    "load_checkpoint",
    "parse_checkpoint_epoch",
    "save_checkpoint",
    "select_highest_resolution_head",
]