"""Late incremental IUCN enrichment for a cleaned encyclopedia.

This module intentionally runs after the encyclopedia has been cleaned and
reduced to canonical species. It avoids querying Red List for duplicate or raw
GBIF names and can resume from a local/Hugging Face cache.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.conservation_status import (
    build_empty_conservation_table,
    build_no_data_record,
    fetch_iucn_record_by_scientific_name,
    fill_missing_conservation_columns,
    get_iucn_token,
    load_iucn_cache,
    record_to_cache_row,
    save_iucn_cache,
)
from src.iucn_candidates import extract_iucn_candidates

CheckpointCallback = Callable[[Path], None]


@dataclass(frozen=True)
class IncrementalIucnSummary:
    candidates_total: int
    cached_before: int
    pending_before: int
    requested_this_run: int
    official_found_this_run: int
    cache_rows_after: int


def _cached_names(cache_df: pd.DataFrame, *, recheck_no_data: bool) -> set[str]:
    if cache_df.empty or "canonical_scientific_name" not in cache_df.columns:
        return set()
    working = cache_df.copy()
    working["canonical_scientific_name"] = working["canonical_scientific_name"].fillna("").astype(str)
    if recheck_no_data and "iucn_category" in working.columns:
        working = working[working["iucn_category"].fillna("").astype(str).str.upper().ne("NO_DATA")]
    return set(working["canonical_scientific_name"].loc[lambda s: s.str.len() > 0].tolist())


def merge_iucn_cache_into_encyclopedia(encyclopedia_df: pd.DataFrame, cache_df: pd.DataFrame) -> pd.DataFrame:
    """Merge cached IUCN rows into the already cleaned encyclopedia."""
    working_df = encyclopedia_df.copy()
    if "canonical_scientific_name" not in working_df.columns:
        working_df["canonical_scientific_name"] = working_df.get("scientific_name", "").astype(str)
    if cache_df.empty:
        return fill_missing_conservation_columns(working_df)

    cache_clean = cache_df.drop_duplicates("canonical_scientific_name", keep="last").copy()
    enriched_df = working_df.merge(
        cache_clean,
        on="canonical_scientific_name",
        how="left",
        validate="many_to_one",
    )
    return fill_missing_conservation_columns(enriched_df)


def enrich_clean_encyclopedia_with_incremental_iucn(
    encyclopedia_df: pd.DataFrame,
    *,
    cache_path: str | Path,
    batch_size: int = 3000,
    request_delay_seconds: float = 0.5,
    checkpoint_every: int = 250,
    candidate_limit: int | None = None,
    recheck_no_data: bool = False,
    checkpoint_callback: CheckpointCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, IncrementalIucnSummary]:
    """Update IUCN cache by querying only pending clean candidates, then merge it.

    If there is no token, the function does not create NO_DATA rows for the whole
    batch. This avoids poisoning the cache with thousands of fake missing rows.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = extract_iucn_candidates(encyclopedia_df, limit=candidate_limit)
    cache_df = load_iucn_cache(cache_path)
    cached = _cached_names(cache_df, recheck_no_data=recheck_no_data)
    pending = [name for name in candidates if name not in cached]
    if batch_size and batch_size > 0:
        selected = pending[:batch_size]
    else:
        selected = pending

    token = get_iucn_token()
    print(f"[IUCN LATE] Clean candidates: {len(candidates):,}", flush=True)
    print(f"[IUCN LATE] Cached before: {len(cached):,}", flush=True)
    print(f"[IUCN LATE] Pending before: {len(pending):,}", flush=True)
    print(f"[IUCN LATE] Selected this run: {len(selected):,}", flush=True)
    print(f"[IUCN LATE] Token configured: {'yes' if token else 'no'}", flush=True)

    new_rows: list[dict] = []
    official_found = 0

    if token:
        for index, canonical_name in enumerate(selected, start=1):
            record = fetch_iucn_record_by_scientific_name(canonical_name, token)
            if record is None:
                record = build_no_data_record(canonical_name)
            elif record.iucn_is_official:
                official_found += 1
            new_rows.append(record_to_cache_row(record))

            if checkpoint_every > 0 and index % checkpoint_every == 0:
                save_iucn_cache(cache_path, cache_df, pd.DataFrame(new_rows))
                cache_df = load_iucn_cache(cache_path)
                if checkpoint_callback is not None:
                    checkpoint_callback(cache_path)
                print(f"[IUCN LATE] Checkpoint guardado tras {index:,} consultas.", flush=True)

            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
    elif selected:
        print("[IUCN LATE] Sin token: se omite live lookup y no se rellena cache con NO_DATA masivo.", flush=True)

    if new_rows:
        save_iucn_cache(cache_path, cache_df, pd.DataFrame(new_rows))
        cache_df = load_iucn_cache(cache_path)
        if checkpoint_callback is not None:
            checkpoint_callback(cache_path)

    enriched_df = merge_iucn_cache_into_encyclopedia(encyclopedia_df, cache_df)
    summary = IncrementalIucnSummary(
        candidates_total=len(candidates),
        cached_before=len(cached),
        pending_before=len(pending),
        requested_this_run=len(selected) if token else 0,
        official_found_this_run=official_found,
        cache_rows_after=len(cache_df),
    )
    return enriched_df, cache_df, summary
