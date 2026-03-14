"""Local training helper utilities.

This module exists primarily for backwards-compatible imports used by tests and
older notebooks/scripts.
"""

from __future__ import annotations


def _infer_label_from_path(path: str) -> int:
    """Infer binary label from a prepared-sample path.

    Conventions:
    - any path segment named 'fake' => label 1
    - any path segment named 'real' => label 0

    Raises:
        ValueError: if both or neither of those segments are present.
    """

    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    parts = [p for p in path.replace("\\", "/").lower().split("/") if p]
    has_fake = "fake" in parts
    has_real = "real" in parts

    if has_fake and not has_real:
        return 1
    if has_real and not has_fake:
        return 0

    raise ValueError(f"Ambiguous or missing label in path: {path!r}")
