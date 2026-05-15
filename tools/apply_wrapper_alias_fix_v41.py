"""Fix compatibility wrappers after clean-architecture move.

The stage-1 migration kept old modules such as ``src.climate_enrichment`` as
wrappers that did ``from src.enrichment.climate import *``. That preserves
imports, but it breaks tests and monkeypatching because functions keep their
real globals in the moved module.

This script rewrites wrappers so each legacy module becomes a true alias of the
new module in ``sys.modules``. Then monkeypatching ``src.climate_enrichment``
patches the actual module used by the functions.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

ALIASES = {
    "data_acquisition.py": "src.acquisition.gbif",
    "data_cleaning.py": "src.cleaning.data_cleaning",
    "feature_engineering.py": "src.cleaning.features",
    "data_snapshots.py": "src.artifacts.snapshots",
    "climate_enrichment.py": "src.enrichment.climate",
    "conservation_status.py": "src.enrichment.conservation",
    "image_enrichment.py": "src.enrichment.images",
    "media_validation.py": "src.enrichment.media_validation",
    "vernacular_names.py": "src.enrichment.vernacular_names",
    "iucn_candidates.py": "src.enrichment.iucn_candidates",
    "iucn_incremental_after_clean.py": "src.enrichment.iucn_incremental_after_clean",
    "artifact_contract.py": "src.artifacts.contract",
    "artifact_manifest.py": "src.artifacts.manifest",
    "hf_checkpoints.py": "src.artifacts.hf_checkpoints",
    "offline_export.py": "src.artifacts.offline_export",
    "model_training.py": "src.modeling.training",
    "eda_reporting.py": "src.reporting.eda",
    "dashboard_charts.py": "src.reporting.dashboard_charts",
    "dashboard_data_sources.py": "src.reporting.dashboard_data_sources",
    "dashboard_loader.py": "src.reporting.dashboard_loader",
    "dashboard_ui.py": "src.reporting.dashboard_ui",
    "search_tags.py": "src.search.tags",
    "encyclopedia.py": "src.artifacts.encyclopedia",
    "occurrence_points.py": "src.artifacts.occurrence_points",
}

TEMPLATE = '''"""Backward-compatible module alias for ``{target}``.

The implementation was moved during the clean-architecture refactor. This file
keeps legacy imports working and, importantly, aliases the module object so
pytest monkeypatches applied to ``src.{legacy}`` affect the real implementation.
"""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("{target}")
sys.modules[__name__] = _module
'''


def main() -> None:
    changed: list[str] = []
    for filename, target in ALIASES.items():
        path = SRC / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing expected wrapper: {path}")
        legacy = filename[:-3]
        content = TEMPLATE.format(target=target, legacy=legacy)
        if path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8", newline="\n")
            changed.append(str(path.relative_to(ROOT)))

    print("Wrapper alias fix v41 applied.")
    print(f"Wrappers updated: {len(changed)}")
    for item in changed:
        print(f"  - {item}")
    print("\nNext commands:")
    print("  python -m pytest")
    print("  python -m py_compile scripts/*.py src/*.py src/*/*.py")
    print("  git status")


if __name__ == "__main__":
    main()
