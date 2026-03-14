"""Pytest configuration.

This repo uses plain source folders (e.g. `src/`, `tools/`) without installing as a package.
Ensure the repository root is on `sys.path` so tests can import those modules reliably.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)
