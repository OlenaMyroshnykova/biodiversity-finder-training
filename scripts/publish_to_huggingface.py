"""Publish generated Biodiversity Finder artifacts to Hugging Face Datasets.

This script is intentionally small and generic because the full pipeline can be
long-running. If the pipeline finishes and this publish step runs, it uploads the
latest local artifacts produced by the workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "selenamir/biodiversity-finder-artifacts"
DEFAULT_REPO_TYPE = "dataset"

# Local folder -> destination folder inside the HF dataset repository.
UPLOAD_FOLDERS: tuple[tuple[str, str], ...] = (
    ("data/processed", "processed"),
    ("data/interim", "interim"),
    ("data/checkpoints", "checkpoints"),
    ("checkpoints", "checkpoints"),
    ("reports", "reports"),
    ("samples", "samples"),
)

# Extensions we intentionally publish. This avoids uploading local caches,
# notebooks, logs, virtualenv files, etc.
ALLOWED_SUFFIXES = {
    ".parquet",
    ".csv",
    ".json",
    ".md",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def _has_publishable_files(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    return any(path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES for path in folder.rglob("*"))


def _ignore_patterns_for(folder: Path) -> list[str]:
    """Build ignore patterns for files we do not want to upload."""
    ignored: list[str] = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() not in ALLOWED_SUFFIXES:
            ignored.append(str(path.relative_to(folder)).replace("\\", "/"))
    return ignored


def publish_folders(
    *,
    repo_id: str,
    token: str,
    repo_type: str = DEFAULT_REPO_TYPE,
    folders: Iterable[tuple[str, str]] = UPLOAD_FOLDERS,
) -> list[str]:
    """Upload existing artifact folders and return their HF destination paths."""
    api = HfApi(token=token)
    uploaded: list[str] = []

    for local_folder, path_in_repo in folders:
        folder = Path(local_folder)
        if not _has_publishable_files(folder):
            print(f"[HF] Skip missing/empty folder: {local_folder}")
            continue

        print(f"[HF] Uploading {local_folder} -> {repo_id}/{path_in_repo}")
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            token=token,
            ignore_patterns=_ignore_patterns_for(folder),
            commit_message=f"Update {path_in_repo} artifacts from clean-first IUCN pipeline",
        )
        uploaded.append(path_in_repo)

    return uploaded


def main() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; cannot publish artifacts to Hugging Face.")

    repo_id = os.getenv("HF_REPO_ID", DEFAULT_REPO_ID)
    repo_type = os.getenv("HF_REPO_TYPE", DEFAULT_REPO_TYPE)

    uploaded = publish_folders(repo_id=repo_id, repo_type=repo_type, token=token)
    if not uploaded:
        raise SystemExit("No publishable artifact folders were found.")

    print("[HF] Published artifact folders:")
    for destination in uploaded:
        print(f"  - {destination}")


if __name__ == "__main__":
    main()
