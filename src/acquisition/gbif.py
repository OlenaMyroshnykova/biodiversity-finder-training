"""GBIF data acquisition for Biodiversity Finder.

The acquisition plan is neutral by broad taxonomic groups and excludes fungi,
because the project scope is animals + plants.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from src.config import GBIF_OCCURRENCE_URL

GBIF_SPECIES_MATCH_URL = "https://api.gbif.org/v1/species/match"
GLOBAL_COUNTRY_VALUES = {"", "GLOBAL", "ALL", "WORLD", "NONE", "NULL"}


@dataclass(frozen=True)
class GBIFQuery:
    """Neutral GBIF query configuration."""

    source_query: str
    description: str
    params: dict[str, Any]
    share: float


def is_global_country(country: str | None) -> bool:
    if country is None:
        return True
    return str(country).strip().upper() in GLOBAL_COUNTRY_VALUES


def match_gbif_taxon_key(name: str, rank: str | None = None) -> int | None:
    params: dict[str, Any] = {"name": name}
    if rank:
        params["rank"] = rank
    try:
        response = requests.get(GBIF_SPECIES_MATCH_URL, params=params, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return None
    payload = response.json()
    usage_key = payload.get("usageKey")
    if isinstance(usage_key, int):
        return usage_key
    return None


def build_global_query_plan(country: str | None = "GLOBAL") -> list[GBIFQuery]:
    """Build neutral Animalia + Plantae GBIF query plan."""
    country_filter: dict[str, Any] = {}
    if not is_global_country(country):
        country_filter["country"] = str(country).upper()

    def with_common_params(params: dict[str, Any]) -> dict[str, Any]:
        merged_params = {"hasCoordinate": "true", **country_filter, **params}
        return {key: value for key, value in merged_params.items() if value is not None}

    taxon_specs = [
        ("class_mammalia", "Clase Mammalia", "Mammalia", "CLASS", 0.14),
        ("class_aves", "Clase Aves", "Aves", "CLASS", 0.14),
        ("class_reptilia", "Clase Reptilia", "Reptilia", "CLASS", 0.11),
        ("class_amphibia", "Clase Amphibia", "Amphibia", "CLASS", 0.10),
        ("class_insecta", "Clase Insecta", "Insecta", "CLASS", 0.14),
        ("class_arachnida", "Clase Arachnida", "Arachnida", "CLASS", 0.08),
        ("class_actinopterygii", "Clase Actinopterygii", "Actinopterygii", "CLASS", 0.10),
        ("class_magnoliopsida", "Clase Magnoliopsida", "Magnoliopsida", "CLASS", 0.08),
        ("class_liliopsida", "Clase Liliopsida", "Liliopsida", "CLASS", 0.05),
        ("class_pinopsida", "Clase Pinopsida", "Pinopsida", "CLASS", 0.03),
        ("class_polypodiopsida", "Clase Polypodiopsida", "Polypodiopsida", "CLASS", 0.03),
    ]

    query_plan: list[GBIFQuery] = []
    for source_query, description, scientific_name, rank, share in taxon_specs:
        taxon_key = match_gbif_taxon_key(scientific_name, rank)
        query_plan.append(
            GBIFQuery(
                source_query=source_query,
                description=description,
                params=with_common_params(
                    {
                        "taxonKey": taxon_key,
                        "scientificName": None if taxon_key else scientific_name,
                    }
                ),
                share=share,
            )
        )
    return query_plan


def download_biodiversity_training_dataset(
    country: str | None = "GLOBAL",
    max_records: int = 20_000,
    page_size: int = 300,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Download global training dataset by broad taxonomic groups."""
    query_plan = build_global_query_plan(country)
    frames = []
    print("Plan de descarga GBIF:", flush=True)
    for query in query_plan:
        query_records = max(1, math.floor(max_records * query.share))
        print(
            f"- {query.source_query}: objetivo {query_records:,} registros | {query.description}",
            flush=True,
        )
        query_df = download_gbif_occurrences_by_params(
            params=query.params,
            max_records=query_records,
            page_size=page_size,
            source_query=query.source_query,
            pause_seconds=pause_seconds,
        )
        frames.append(query_df)

    if not frames:
        return pd.DataFrame()
    combined_df = pd.concat(frames, ignore_index=True)
    if "key" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["key"])
    if "kingdom" in combined_df.columns:
        combined_df = combined_df[combined_df["kingdom"].isin(["Animalia", "Plantae"])].copy()
    combined_df = combined_df.head(max_records).reset_index(drop=True)
    print(
        f"Dataset combinado final: {len(combined_df):,} registros desde {len(frames)} consultas.",
        flush=True,
    )
    return combined_df


def download_gbif_occurrences(
    country: str | None = "GLOBAL",
    max_records: int = 20_000,
    page_size: int = 300,
) -> pd.DataFrame:
    """Backward-compatible wrapper."""
    if is_global_country(country):
        return download_biodiversity_training_dataset(
            country="GLOBAL",
            max_records=max_records,
            page_size=page_size,
        )
    params = {"country": str(country).upper(), "hasCoordinate": "true"}
    return download_gbif_occurrences_by_params(
        params=params,
        max_records=max_records,
        page_size=page_size,
        source_query=f"country_{str(country).upper()}",
    )


def download_gbif_occurrences_by_params(
    *,
    params: dict[str, Any],
    max_records: int,
    page_size: int,
    source_query: str,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Download GBIF occurrences using arbitrary parameters."""
    all_records: list[dict[str, Any]] = []
    offset = 0
    while len(all_records) < max_records:
        current_limit = min(page_size, max_records - len(all_records))
        request_params = {**params, "limit": current_limit, "offset": offset}
        try:
            response = requests.get(GBIF_OCCURRENCE_URL, params=request_params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"Error descargando {source_query} desde GBIF: {error}", flush=True)
            break

        payload = response.json()
        records = payload.get("results", [])
        if not records:
            break
        for record in records:
            record["source_query"] = source_query
        all_records.extend(records)
        print(f"  {source_query}: descargados {len(all_records):,} / {max_records:,}", flush=True)

        offset += current_limit
        if len(records) < current_limit:
            break
        time.sleep(pause_seconds)

    return pd.DataFrame(all_records[:max_records])
