"""Backward-compatible module alias for ``src.artifacts.offline_export``.

The implementation was moved during the clean-architecture refactor. This file
keeps legacy imports working and, importantly, aliases the module object so
pytest monkeypatches applied to ``src.offline_export`` affect the real implementation.
"""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("src.artifacts.offline_export")
sys.modules[__name__] = _module
