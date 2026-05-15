"""Backward-compatible module alias for ``src.acquisition.gbif``.

The implementation was moved during the clean-architecture refactor. This file
keeps legacy imports working and, importantly, aliases the module object so
pytest monkeypatches applied to ``src.data_acquisition`` affect the real implementation.
"""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("src.acquisition.gbif")
sys.modules[__name__] = _module
