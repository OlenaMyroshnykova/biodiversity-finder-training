"""Apply a safe first-stage clean architecture refactor.

This script moves existing flat src/*.py modules into domain folders and leaves
backward-compatible wrapper modules in the old locations, so existing tests,
scripts and workflows can continue importing `src.<old_module>`.

Run from the repository root:
    python tools/apply_src_clean_architecture_v40.py
    python -m pytest
"""
from __future__ import annotations

from pathlib import Path
import shutil
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCS = ROOT / "docs"

# old module file -> new module file
MOVES: dict[str, str] = {
    # data acquisition / preparation
    "data_acquisition.py": "acquisition/gbif.py",
    "data_cleaning.py": "cleaning/data_cleaning.py",
    "feature_engineering.py": "cleaning/features.py",
    "data_snapshots.py": "artifacts/snapshots.py",

    # enrichment
    "climate_enrichment.py": "enrichment/climate.py",
    "conservation_status.py": "enrichment/conservation.py",
    "image_enrichment.py": "enrichment/images.py",
    "media_validation.py": "enrichment/media_validation.py",
    "vernacular_names.py": "enrichment/vernacular_names.py",
    "iucn_candidates.py": "enrichment/iucn_candidates.py",
    "iucn_incremental_after_clean.py": "enrichment/iucn_incremental_after_clean.py",

    # artifacts / publication helpers
    "artifact_contract.py": "artifacts/contract.py",
    "artifact_manifest.py": "artifacts/manifest.py",
    "hf_checkpoints.py": "artifacts/hf_checkpoints.py",
    "offline_export.py": "artifacts/offline_export.py",

    # modeling / reporting / search layer
    "model_training.py": "modeling/training.py",
    "eda_reporting.py": "reporting/eda.py",
    "dashboard_charts.py": "reporting/dashboard_charts.py",
    "dashboard_data_sources.py": "reporting/dashboard_data_sources.py",
    "dashboard_loader.py": "reporting/dashboard_loader.py",
    "dashboard_ui.py": "reporting/dashboard_ui.py",
    "search_tags.py": "search/tags.py",
    "encyclopedia.py": "artifacts/encyclopedia.py",
    "occurrence_points.py": "artifacts/occurrence_points.py",
}

PACKAGE_READMES = {
    "acquisition": "External data acquisition clients and GBIF download logic.",
    "cleaning": "Dataset cleaning, normalization and feature preparation.",
    "enrichment": "External enrichment layers: climate, images, common names and IUCN.",
    "artifacts": "Artifact contracts, manifests, checkpoints, exports and publication helpers.",
    "modeling": "Model training and evaluation logic.",
    "reporting": "EDA, dashboard loading and dashboard UI helpers.",
    "search": "Search document and tag construction for the app artifact.",
}


def module_path_from_file(file_path: str) -> str:
    return "src." + file_path[:-3].replace("/", ".").replace("\\", ".")


def write_package_files() -> None:
    for package, description in PACKAGE_READMES.items():
        package_dir = SRC / package
        package_dir.mkdir(parents=True, exist_ok=True)
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text(f'"""{description}"""\n', encoding="utf-8")
        readme = package_dir / "README.md"
        if not readme.exists():
            readme.write_text(f"# {package}\n\n{description}\n", encoding="utf-8")


def wrapper_content(old_file: str, new_file: str) -> str:
    old_module = module_path_from_file(old_file)
    new_module = module_path_from_file(new_file)
    return textwrap.dedent(
        f'''\
        """Backward-compatible import wrapper.

        The implementation moved to `{new_module}` during the clean architecture
        refactor. Keep this module so existing tests, scripts and notebooks using
        `{old_module}` continue to work.
        """
        from __future__ import annotations

        from {new_module} import *  # noqa: F401,F403
        '''
    )


def move_module(old_file: str, new_file: str) -> tuple[str, str, str]:
    old_path = SRC / old_file
    new_path = SRC / new_file
    new_path.parent.mkdir(parents=True, exist_ok=True)

    if new_path.exists():
        # Already moved in a previous run. Ensure wrapper exists.
        if old_path.exists():
            current = old_path.read_text(encoding="utf-8")
            if "Backward-compatible import wrapper" not in current:
                backup = old_path.with_suffix(old_path.suffix + ".before_v40")
                shutil.move(str(old_path), str(backup))
                old_path.write_text(wrapper_content(old_file, new_file), encoding="utf-8")
                return (old_file, new_file, f"new exists; backed up old to {backup.name}")
        else:
            old_path.write_text(wrapper_content(old_file, new_file), encoding="utf-8")
        return (old_file, new_file, "already moved")

    if not old_path.exists():
        return (old_file, new_file, "old file missing; skipped")

    shutil.move(str(old_path), str(new_path))
    old_path.write_text(wrapper_content(old_file, new_file), encoding="utf-8")
    return (old_file, new_file, "moved + wrapper created")


def write_architecture_doc(results: list[tuple[str, str, str]]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Clean Architecture Refactor — Stage 1\n",
        "\n",
        "This stage reorganizes the training code into domain folders while keeping backward-compatible wrappers in the old `src/*.py` locations.\n",
        "\n",
        "## Target structure\n",
        "\n",
        "```text\n",
        "src/\n",
        "  acquisition/   # GBIF and external raw data acquisition\n",
        "  cleaning/      # cleaning and feature preparation\n",
        "  enrichment/    # climate, images, common names, IUCN\n",
        "  artifacts/     # contracts, manifests, checkpoints, exports\n",
        "  modeling/      # model training and evaluation\n",
        "  reporting/     # EDA and dashboard helpers\n",
        "  search/        # search tags and search documents\n",
        "```\n",
        "\n",
        "## Compatibility rule\n",
        "\n",
        "Existing imports such as `from src.image_enrichment import ...` still work because `src/image_enrichment.py` now re-exports from `src.enrichment.images`.\n",
        "\n",
        "## Moves applied\n",
        "\n",
    ]
    for old_file, new_file, status in results:
        lines.append(f"- `src/{old_file}` → `src/{new_file}` — {status}\n")
    lines.extend([
        "\n",
        "## Next refactor stages\n",
        "\n",
        "1. Update internal imports gradually from wrapper modules to domain modules.\n",
        "2. Move pipeline orchestration logic from `scripts/` into `src/pipeline/`, leaving scripts as thin CLI wrappers.\n",
        "3. Add service-level tests around the new domain modules.\n",
        "4. Remove compatibility wrappers only after all imports are migrated.\n",
    ])
    (DOCS / "CLEAN_ARCHITECTURE_STAGE_1.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    if not SRC.exists():
        raise SystemExit("Run this script from the repository root; src/ was not found.")

    write_package_files()
    results = [move_module(old_file, new_file) for old_file, new_file in MOVES.items()]
    write_architecture_doc(results)

    print("Clean architecture stage 1 applied.")
    for old_file, new_file, status in results:
        print(f"- src/{old_file} -> src/{new_file}: {status}")
    print("\nNext commands:")
    print("  python -m pytest")
    print("  python -m py_compile scripts/*.py src/*.py src/*/*.py")
    print("  git status")


if __name__ == "__main__":
    main()
