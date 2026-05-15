"""Check repository hygiene after cleanup.

The check is read-only. It reports:
- tracked generated files that should usually live on Hugging Face;
- local cache directories;
- missing .gitkeep placeholders.

Run from repository root:

    python tools/check_repo_hygiene_v39.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GENERATED_SUFFIXES = {
    ".parquet",
    ".csv",
    ".json",
    ".joblib",
    ".pkl",
    ".pickle",
    ".log",
}

ALLOWED_TRACKED_GENERATED = {
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
    "models/.gitkeep",
    "reports/.gitkeep",
}

CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}


def git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def find_tracked_generated(files: list[str]) -> list[str]:
    candidates: list[str] = []
    for file in files:
        if file in ALLOWED_TRACKED_GENERATED:
            continue
        path = Path(file)
        if path.suffix.lower() in GENERATED_SUFFIXES and (
            file.startswith("data/")
            or file.startswith("models/")
            or file.startswith("reports/")
        ):
            candidates.append(file)
    return candidates


def find_cache_dirs() -> list[str]:
    result: list[str] = []
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in CACHE_DIR_NAMES:
            result.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return sorted(result)


def missing_gitkeep() -> list[str]:
    required = [
        "data/raw/.gitkeep",
        "data/interim/.gitkeep",
        "data/processed/.gitkeep",
        "models/.gitkeep",
        "reports/.gitkeep",
    ]
    return [path for path in required if not (ROOT / path).exists()]


def main() -> None:
    files = git_ls_files()
    tracked_generated = find_tracked_generated(files)
    caches = find_cache_dirs()
    missing = missing_gitkeep()

    print("Repository hygiene report")
    print("=========================")
    print(f"Tracked files: {len(files)}")
    print(f"Tracked generated artifact candidates: {len(tracked_generated)}")
    for path in tracked_generated:
        print(f"  - {path}")

    print(f"Local cache directories: {len(caches)}")
    for path in caches:
        print(f"  - {path}")

    print(f"Missing .gitkeep placeholders: {len(missing)}")
    for path in missing:
        print(f"  - {path}")

    if not tracked_generated and not caches and not missing:
        print("OK: repository looks clean.")
    else:
        print("Review the items above before committing.")


if __name__ == "__main__":
    main()
