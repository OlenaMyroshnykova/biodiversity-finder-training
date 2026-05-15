"""Backward-compatible module alias for ``src.artifacts.contract``.

The implementation was moved during the clean-architecture refactor. This file
keeps legacy imports working and, importantly, aliases the module object so
pytest monkeypatches applied to ``src.artifact_contract`` affect the real implementation.
"""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("src.artifacts.contract")
sys.modules[__name__] = _module
