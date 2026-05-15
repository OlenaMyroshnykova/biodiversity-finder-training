"""Backward-compatible module alias for ``src.cleaning.data_cleaning``.

The implementation was moved during the clean-architecture refactor. This file
keeps legacy imports working and, importantly, aliases the module object so
pytest monkeypatches applied to ``src.data_cleaning`` affect the real implementation.
"""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("src.cleaning.data_cleaning")
sys.modules[__name__] = _module
