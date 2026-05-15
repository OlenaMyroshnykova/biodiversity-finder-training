"""Clean image URLs in existing biodiversity artifacts.

Use this script when artifacts already exist and you only want to remove audio,
video, documents or unverifiable media from image_url without rerunning the whole
pipeline.
"""
from __future__ import annotations

# Allow direct execution from GitHub Actions and local terminals.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import pandas as pd

from src.image_enrichment import ensure_image_columns
from src.media_validation import classify_media_url


def clean_dataframe_images(df: pd.DataFrame) -> pd.DataFrame:
    """Validate image_url and add media validation columns."""
    return ensure_image_columns(df)


def clean_image_enrichment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an image_enrichment.csv-like dataframe."""
    result = df.copy()
    if "image_url" not in result.columns:
        result["image_url"] = ""

    statuses = []
    media_types = []
    unverified = []
    cleaned = []
    for value in result["image_url"].fillna("").astype(str).tolist():
        decision = classify_media_url(value)
        statuses.append(decision.status)
        media_types.append(decision.media_type)
        if decision.is_valid_image:
            cleaned.append(decision.url)
            unverified.append("")
        elif decision.status in {"unknown_no_extension", "invalid_unknown_extension"}:
            cleaned.append("")
            unverified.append(decision.url)
        else:
            cleaned.append("")
            unverified.append("")

    result["image_url"] = cleaned
    result["media_type"] = media_types
    result["image_validation_status"] = statuses
    result["unverified_media_url"] = unverified
    result["has_image"] = result["image_url"].astype(str).str.len() > 0
    return result


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean media URLs in biodiversity artifacts")
    parser.add_argument("--input", required=True, help="Input parquet/csv path")
    parser.add_argument("--output", required=True, help="Output parquet/csv path")
    parser.add_argument(
        "--image-enrichment-table",
        action="store_true",
        help="Use image_enrichment.csv cleaning rules instead of encyclopedia rules",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = read_table(input_path)
    if args.image_enrichment_table:
        cleaned = clean_image_enrichment_table(df)
    else:
        cleaned = clean_dataframe_images(df)

    before = int(df.get("image_url", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum())
    after = int(cleaned.get("image_url", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum())
    write_table(cleaned, output_path)
    print(f"Cleaned media URLs: {before:,} -> {after:,} valid image_url values")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
