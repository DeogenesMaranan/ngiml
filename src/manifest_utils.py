from __future__ import annotations

from pathlib import Path
from typing import Any


def _norm(value: str) -> str:
    return str(value).replace("\\", "/")


def _suffix_score(a_parts, b_parts) -> int:
    score = 0
    for ax, bx in zip(reversed(a_parts), reversed(b_parts)):
        if ax != bx:
            break
        score += 1
    return score


def _candidate_paths(value: str, manifest_path: Path, data_root: Path) -> list[Path]:
    normalized = _norm(value)
    path_value = Path(normalized)
    candidates: list[Path] = []
    if path_value.is_absolute():
        candidates.append(path_value)
    else:
        candidates.extend(
            [
                manifest_path.parent / path_value,
                data_root / path_value,
                data_root / "ngiml" / path_value,
                Path("/content") / path_value,
                Path("/content/data") / path_value,
                Path("/content/ngiml") / path_value,
            ]
        )
    if "prepared/" in normalized:
        suffix = normalized.split("prepared/", 1)[1]
        candidates.extend(
            [
                data_root / "prepared" / suffix,
                data_root / "ngiml" / "prepared" / suffix,
                Path("/content") / "prepared" / suffix,
                Path("/content/ngiml") / "prepared" / suffix,
            ]
        )
    if "datasets/" in normalized:
        suffix = normalized.split("datasets/", 1)[1]
        candidates.extend(
            [
                data_root / "datasets" / suffix,
                data_root / "ngiml" / "datasets" / suffix,
                Path("/content") / "datasets" / suffix,
                Path("/content/ngiml") / "datasets" / suffix,
            ]
        )
    seen = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = candidate.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def build_tar_index(data_root: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    tar_files: list[Path] = []
    for pattern in ("*.tar", "*.tar.gz", "*.tgz"):
        tar_files.extend(data_root.rglob(pattern))
    tar_by_name: dict[str, list[Path]] = {}
    for tar_path in tar_files:
        tar_by_name.setdefault(tar_path.name, []).append(tar_path)
    return tar_files, tar_by_name


def _match_tar_by_basename(value: str, tar_by_name: dict[str, list[Path]]) -> Path | None:
    name = Path(_norm(value)).name
    matches = tar_by_name.get(name, [])
    if not matches:
        return None
    hint_parts = Path(_norm(value)).parts
    return max(matches, key=lambda path: _suffix_score(path.parts, hint_parts))


def _resolve_file(value: str, manifest_path: Path, data_root: Path, tar_by_name: dict[str, list[Path]]) -> Path:
    candidates = _candidate_paths(value, manifest_path, data_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if str(value).endswith((".tar", ".tar.gz", ".tgz")):
        tar_match = _match_tar_by_basename(value, tar_by_name)
        if tar_match is not None:
            return tar_match
    return candidates[0] if candidates else Path(_norm(value))


def resolve_path(path_str: str | None, manifest_path: Path, data_root: Path, tar_by_name: dict[str, list[Path]]) -> str | None:
    if path_str is None:
        return None
    normalized = _norm(path_str)
    if "::" in normalized:
        archive, member = normalized.split("::", 1)
        archive_path = _resolve_file(archive, manifest_path, data_root, tar_by_name).as_posix()
        member_path = _norm(member)
        return f"{archive_path}::{member_path}"
    return _resolve_file(normalized, manifest_path, data_root, tar_by_name).as_posix()


def sample_files_exist(sample: Any) -> bool:
    image_path = str(sample.image_path)
    if "::" in image_path:
        archive_path, _ = image_path.split("::", 1)
        if not Path(archive_path).exists():
            return False
    else:
        if not Path(image_path).exists():
            return False
    if sample.mask_path is not None and not Path(sample.mask_path).exists():
        return False
    return True


__all__ = [
    "build_tar_index",
    "resolve_path",
    "sample_files_exist",
]

# Backward-compatible aliases for older imports.
_build_tar_index = build_tar_index
_resolve_path = resolve_path
_sample_files_exist = sample_files_exist
