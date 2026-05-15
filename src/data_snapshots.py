"""Backward-compatible module alias for ``src.artifacts.snapshots``.

The implementation was moved during the clean-architecture refactor. This file
keeps legacy imports working and, importantly, aliases the module object so
pytest monkeypatches applied to ``src.data_snapshots`` affect the real implementation.
"""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("src.artifacts.snapshots")
sys.modules[__name__] = _module
