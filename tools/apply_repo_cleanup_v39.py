"""Apply safe repository cleanup rules for the biodiversity training repo.

This script is intentionally conservative:
- it does not delete tracked source code;
- it only removes local cache folders such as __pycache__ and .pytest_cache;
- it updates .gitignore with generated artifact patterns;
- it keeps .gitkeep files so the folder structure remains visible.

Run from the repository root:

    python tools/apply_repo_cleanup_v39.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}

GITIGNORE_BLOCK = """
# === v39 repository hygiene: generated artifacts and local caches ===

# Local secrets
.env
.env.*
!.env.example

# Python cache
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/

# Virtual environments
.venv/
venv/
env/

# Local logs
*.log

# Generated data artifacts.
# Source of truth for generated parquet/csv/json artifacts is Hugging Face.
data/raw/*
data/interim/*
data/processed/*
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep

# Generated model artifacts
models/*
!models/.gitkeep

# Generated reports
reports/**/*.csv
reports/**/*.json
reports/**/*.parquet
reports/**/*.html
reports/**/*.png
reports/**/*.jpg
reports/**/*.jpeg
reports/**/*.webp
!reports/.gitkeep
!reports/**/.gitkeep

# Temporary downloads and local backups
downloads/
tmp/
temp/
hf_backups/
""".strip()


def remove_cache_dirs() -> list[Path]:
    removed: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in CACHE_DIR_NAMES:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.relative_to(ROOT))
    return removed


def ensure_gitkeep_files() -> list[Path]:
    created: list[Path] = []
    for folder in [
        ROOT / "data" / "raw",
        ROOT / "data" / "interim",
        ROOT / "data" / "processed",
        ROOT / "models",
        ROOT / "reports",
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        gitkeep = folder / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")
            created.append(gitkeep.relative_to(ROOT))
    return created


def update_gitignore() -> bool:
    gitignore = ROOT / ".gitignore"
    current = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""

    marker = "# === v39 repository hygiene: generated artifacts and local caches ==="
    if marker in current:
        return False

    suffix = "\n\n" if current and not current.endswith("\n") else "\n"
    gitignore.write_text(current + suffix + GITIGNORE_BLOCK + "\n", encoding="utf-8")
    return True


def main() -> None:
    removed = remove_cache_dirs()
    created = ensure_gitkeep_files()
    gitignore_changed = update_gitignore()

    print("Repository cleanup v39 complete.")
    print(f"Cache folders removed: {len(removed)}")
    for path in removed:
        print(f"  - {path}")

    print(f".gitkeep files created: {len(created)}")
    for path in created:
        print(f"  - {path}")

    print(f".gitignore updated: {gitignore_changed}")
    print()
    print("Next commands:")
    print("  python -m pytest")
    print("  git status")
    print("  git add .gitignore data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep models/.gitkeep reports/.gitkeep docs/REPOSITORY_STRUCTURE.md tools/apply_repo_cleanup_v39.py tools/check_repo_hygiene_v39.py")
    print('  git commit -m "Clean repository structure and ignore generated artifacts"')


if __name__ == "__main__":
    main()
