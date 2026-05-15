"""Backward-compatible module alias for ``src.enrichment.climate``.

The implementation was moved during the clean-architecture refactor. This file
keeps legacy imports working and, importantly, aliases the module object so
pytest monkeypatches applied to ``src.climate_enrichment`` affect the real implementation.
"""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("src.enrichment.climate")
sys.modules[__name__] = _module
