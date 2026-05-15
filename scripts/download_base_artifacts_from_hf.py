from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

BASE_FILES = (
    "processed/species_encyclopedia.parquet",
    "processed/species_encyclopedia_light.parquet",
    "processed/species_occurrence_points.parquet",
    "processed/species_occurrence_points_light.parquet",
    "processed/gbif_occurrences_features.parquet",
    "interim/species_encyclopedia_clean_before_iucn.parquet",
    "interim/gbif_occurrences_clean.parquet",
    "interim/vernacular_names.csv",
    "interim/image_enrichment.csv",
    "interim/climate_reference.csv",
    "interim/iucn_status_cache.csv",
    "reports/metrics.json",
    "reports/classification_report.csv",
)


def _local_path(remote_path: str, root: Path) -> Path:
    if remote_path.startswith("processed/"):
        return root / "data" / remote_path
    if remote_path.startswith("interim/"):
        return root / "data" / remote_path
    return root / remote_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download reusable base artifacts from Hugging Face.")
    parser.add_argument("--repo-id", default=os.getenv("HF_REPO_ID", "selenamir/biodiversity-finder-artifacts"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--required", action="store_true", help="Fail if a tracked file cannot be downloaded.")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    from huggingface_hub import hf_hub_download

    root = Path(args.root)
    downloaded_count = 0
    missing: list[str] = []
    for remote_path in BASE_FILES:
        try:
            cached = hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=remote_path,
                token=token,
            )
        except Exception as exc:
            print(f"[HF BASE] missing {remote_path}: {exc}")
            missing.append(remote_path)
            continue

        target = _local_path(remote_path, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, target)
        downloaded_count += 1
        print(f"[HF BASE] {remote_path} -> {target}")

    print(f"[HF BASE] downloaded {downloaded_count}/{len(BASE_FILES)} files")
    if args.required and missing:
        raise SystemExit(f"Missing required remote base artifacts: {missing}")


if __name__ == "__main__":
    main()
