"""Print a compact IUCN summary for generated Biodiversity Finder artifacts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANDIDATE_PATHS = [
    Path("data/processed/species_encyclopedia_light.parquet"),
    Path("data/processed/species_encyclopedia.parquet"),
    Path("processed/species_encyclopedia_light.parquet"),
]


def main() -> None:
    path = next((candidate for candidate in CANDIDATE_PATHS if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("No generated encyclopedia parquet was found.")

    df = pd.read_parquet(path)
    print(f"[IUCN CHECK] File: {path}", flush=True)
    print(f"[IUCN CHECK] Rows: {len(df):,}", flush=True)

    iucn_columns = [
        column
        for column in df.columns
        if "iucn" in column.lower()
        or "conservation" in column.lower()
        or "threat" in column.lower()
    ]
    print(f"[IUCN CHECK] Columns: {iucn_columns}", flush=True)

    if "iucn_category" not in df.columns:
        print("[IUCN CHECK][WARNING] iucn_category column is missing.", flush=True)
        return

    print("[IUCN CHECK] Category distribution:", flush=True)
    print(df["iucn_category"].fillna("NULL").value_counts(dropna=False).to_string(), flush=True)

    if "conservation_source" in df.columns:
        print("[IUCN CHECK] Source distribution:", flush=True)
        print(df["conservation_source"].fillna("NULL").value_counts(dropna=False).to_string(), flush=True)

    threatened = df[df["iucn_category"].isin(["VU", "EN", "CR", "EW", "EX"])]
    official = df[df["iucn_category"].isin(["LC", "NT", "VU", "EN", "CR", "EW", "EX", "DD"])]

    print(f"[IUCN CHECK] Official IUCN rows: {len(official):,}", flush=True)
    print(f"[IUCN CHECK] Threatened rows: {len(threatened):,}", flush=True)

    cols = [
        "scientific_name",
        "canonical_scientific_name",
        "taxon_class",
        "family",
        "iucn_category",
        "iucn_status_label",
        "conservation_source",
        "is_threatened",
        "observations",
    ]
    cols = [column for column in cols if column in df.columns]

    if not threatened.empty:
        print("[IUCN CHECK] Threatened examples:", flush=True)
        print(threatened[cols].head(25).to_string(index=False), flush=True)
    elif not official.empty:
        print("[IUCN CHECK] Official non-threatened examples:", flush=True)
        print(official[cols].head(25).to_string(index=False), flush=True)
    else:
        print(
            "[IUCN CHECK][WARNING] No official IUCN categories found in the generated artifact.",
            flush=True,
        )


if __name__ == "__main__":
    main()
