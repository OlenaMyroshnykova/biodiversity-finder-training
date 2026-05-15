"""Backward-compatible module alias for ``src.artifacts.occurrence_points``.

The implementation was moved during the clean-architecture refactor. This file
keeps legacy imports working and, importantly, aliases the module object so
pytest monkeypatches applied to ``src.occurrence_points`` affect the real implementation.
"""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("src.artifacts.occurrence_points")
sys.modules[__name__] = _module
