"""Candidate selection for late IUCN enrichment.

The Red List API should be called only after the encyclopedia has already been
cleaned, deduplicated and enriched with the fields that decide whether a species
is useful for the final app. This module extracts that final, canonical candidate
list.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

PROJECT_SCOPE_KINGDOMS = {"animalia", "plantae"}
PRIORITY_CLASSES = {"mammalia", "aves", "reptilia", "amphibia"}


@dataclass(frozen=True)
class IucnCandidateSummary:
    total_rows: int
    unique_candidates: int
    with_images: int
    with_observations: int


def clean_canonical_scientific_name(value: object) -> str:
    """Return a stable genus + species name when possible.

    GBIF data may contain authors, parentheses, subspecies markers or extra text.
    IUCN lookup is more stable when we query a canonical binomial.
    """
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return ""

    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = [part.strip(" ,;:[]{}()") for part in text.split() if part.strip(" ,;:[]{}()")]
    if len(parts) < 2:
        return ""

    genus, species = parts[0], parts[1]
    if not genus[:1].isalpha() or not species[:1].isalpha():
        return ""
    if species.lower() in {"sp", "sp.", "spp", "spp.", "cf", "aff"}:
        return ""
    return f"{genus[:1].upper()}{genus[1:]} {species.lower()}"


def _truthy_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes", "si", "sí"})


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[column].fillna("").astype(str)


def prepare_clean_iucn_candidate_frame(encyclopedia_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a deduplicated candidate frame from the cleaned encyclopedia."""
    if encyclopedia_df.empty:
        return pd.DataFrame(columns=["canonical_scientific_name", "iucn_priority_score"])

    df = encyclopedia_df.copy()
    if "canonical_scientific_name" not in df.columns:
        df["canonical_scientific_name"] = _text_series(df, "scientific_name")
    df["canonical_scientific_name"] = df["canonical_scientific_name"].map(clean_canonical_scientific_name)
    df = df[df["canonical_scientific_name"].str.len() > 0].copy()

    if "kingdom" in df.columns:
        kingdom = _text_series(df, "kingdom").str.lower()
        scoped = df[kingdom.isin(PROJECT_SCOPE_KINGDOMS)].copy()
        if not scoped.empty:
            df = scoped

    has_image = _truthy_series(df, "has_image") | _text_series(df, "image_url").str.startswith("http")
    observations = _numeric_series(df, "observations")
    taxon_class = _text_series(df, "taxon_class").str.lower()
    vernacular = _text_series(df, "vernacular_names")

    df["_has_image_priority"] = has_image.astype(int)
    df["_priority_class"] = taxon_class.isin(PRIORITY_CLASSES).astype(int)
    df["_has_common_name"] = vernacular.str.strip().ne("").astype(int)
    df["_observations_priority"] = observations.clip(lower=0)
    df["iucn_priority_score"] = (
        df["_has_image_priority"] * 1_000_000
        + df["_priority_class"] * 100_000
        + df["_has_common_name"] * 10_000
        + df["_observations_priority"]
    )

    df = df.sort_values(
        ["iucn_priority_score", "canonical_scientific_name"],
        ascending=[False, True],
        kind="mergesort",
    )
    df = df.drop_duplicates("canonical_scientific_name", keep="first")
    return df.reset_index(drop=True)


def extract_iucn_candidates(
    encyclopedia_df: pd.DataFrame,
    *,
    limit: int | None = None,
) -> list[str]:
    """Return ordered canonical species names for late IUCN lookup."""
    candidates_df = prepare_clean_iucn_candidate_frame(encyclopedia_df)
    names = candidates_df["canonical_scientific_name"].dropna().astype(str).tolist()
    if limit is not None and limit > 0:
        return names[:limit]
    return names


def summarize_iucn_candidates(encyclopedia_df: pd.DataFrame) -> IucnCandidateSummary:
    candidates_df = prepare_clean_iucn_candidate_frame(encyclopedia_df)
    return IucnCandidateSummary(
        total_rows=len(encyclopedia_df),
        unique_candidates=len(candidates_df),
        with_images=int(candidates_df.get("_has_image_priority", pd.Series(dtype=int)).sum()),
        with_observations=int((candidates_df.get("_observations_priority", pd.Series(dtype=int)) > 0).sum()),
    )
