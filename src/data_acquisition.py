"""Neutral GBIF acquisition for Biodiversity Finder.

This module intentionally downloads broad taxonomic groups, not hand-picked
"beautiful" demo species or hand-picked demo species.

Architecture goal:
- GBIF = observations, coordinates and taxonomy.
- Semantic tags are created later in feature engineering.
- The acquisition plan must not contain vibe words such as pink/savanna/large.
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
    """Configuration for one neutral GBIF download group."""

    source_query: str
    description: str
    params: dict[str, Any]
    share: float


def is_global_country(country: str | None) -> bool:
    """Return True if the country parameter means a global query."""
    if country is None:
        return True
    return str(country).strip().upper() in GLOBAL_COUNTRY_VALUES


def match_gbif_taxon_key(name: str, rank: str | None = None) -> int | None:
    """Resolve a GBIF Backbone taxonKey by scientific taxon name."""
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
    """Build a neutral broad-taxonomy GBIF acquisition plan.

    The plan avoids species-level demo targets. All queries are large taxonomic
    classes/kingdoms, so the dataset is not pre-shaped for specific natural
    language examples.
    """
    country_filter: dict[str, Any] = {}
    if not is_global_country(country):
        country_filter["country"] = str(country).upper()

    def with_common_params(params: dict[str, Any]) -> dict[str, Any]:
        merged = {"hasCoordinate": "true", **country_filter, **params}
        return {key: value for key, value in merged.items() if value is not None}

    taxon_keys = {
        "mammalia": match_gbif_taxon_key("Mammalia", "CLASS"),
        "aves": match_gbif_taxon_key("Aves", "CLASS"),
        "reptilia": match_gbif_taxon_key("Reptilia", "CLASS"),
        "amphibia": match_gbif_taxon_key("Amphibia", "CLASS"),
        "insecta": match_gbif_taxon_key("Insecta", "CLASS"),
        "actinopterygii": match_gbif_taxon_key("Actinopterygii", "CLASS"),
        "chondrichthyes": match_gbif_taxon_key("Chondrichthyes", "CLASS"),
        "arachnida": match_gbif_taxon_key("Arachnida", "CLASS"),
        "plantae": match_gbif_taxon_key("Plantae", "KINGDOM"),
        "fungi": match_gbif_taxon_key("Fungi", "KINGDOM"),
    }

    print(
        "  taxon_keys resueltos:",
        {key: value for key, value in taxon_keys.items() if value is not None},
        flush=True,
    )

    # Shares sum to exactly 1.00.
    query_specs = [
        ("global_background", "Muestra global neutral de biodiversidad", {}, 0.05),
        ("class_mammalia", "Clase Mammalia", {"taxonKey": taxon_keys["mammalia"], "scientificName": None if taxon_keys["mammalia"] else "Mammalia"}, 0.10),
        ("class_aves", "Clase Aves", {"taxonKey": taxon_keys["aves"], "scientificName": None if taxon_keys["aves"] else "Aves"}, 0.10),
        ("class_reptilia", "Clase Reptilia", {"taxonKey": taxon_keys["reptilia"], "scientificName": None if taxon_keys["reptilia"] else "Reptilia"}, 0.09),
        ("class_amphibia", "Clase Amphibia", {"taxonKey": taxon_keys["amphibia"], "scientificName": None if taxon_keys["amphibia"] else "Amphibia"}, 0.08),
        ("class_insecta", "Clase Insecta", {"taxonKey": taxon_keys["insecta"], "scientificName": None if taxon_keys["insecta"] else "Insecta"}, 0.12),
        ("class_actinopterygii", "Clase Actinopterygii", {"taxonKey": taxon_keys["actinopterygii"], "scientificName": None if taxon_keys["actinopterygii"] else "Actinopterygii"}, 0.10),
        ("class_chondrichthyes", "Clase Chondrichthyes", {"taxonKey": taxon_keys["chondrichthyes"], "scientificName": None if taxon_keys["chondrichthyes"] else "Chondrichthyes"}, 0.07),
        ("class_arachnida", "Clase Arachnida", {"taxonKey": taxon_keys["arachnida"], "scientificName": None if taxon_keys["arachnida"] else "Arachnida"}, 0.07),
        ("kingdom_plantae", "Reino Plantae", {"taxonKey": taxon_keys["plantae"], "scientificName": None if taxon_keys["plantae"] else "Plantae"}, 0.12),
        ("kingdom_fungi", "Reino Fungi", {"taxonKey": taxon_keys["fungi"], "scientificName": None if taxon_keys["fungi"] else "Fungi"}, 0.10),
    ]

    return [
        GBIFQuery(
            source_query=source_query,
            description=description,
            params=with_common_params(params),
            share=share,
        )
        for source_query, description, params, share in query_specs
    ]


def download_biodiversity_training_dataset(
    country: str | None = "GLOBAL",
    max_records: int = 20_000,
    page_size: int = 300,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Download a neutral broad-taxonomy dataset from GBIF."""
    query_plan = build_global_query_plan(country)
    frames = []

    print("Plan de descarga GBIF neutral por grupos taxonómicos:", flush=True)

    for query in query_plan:
        query_records = max(1, math.floor(max_records * query.share))
        print(
            f"- {query.source_query}: objetivo {query_records:,} registros | {query.description}",
            flush=True,
        )
        frame = download_gbif_occurrences(
            params=query.params,
            source_query=query.source_query,
            max_records=query_records,
            page_size=page_size,
            pause_seconds=pause_seconds,
        )
        frames.append(frame)

    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()

    return pd.concat(non_empty_frames, ignore_index=True).drop_duplicates()


def download_country_biodiversity_sample(
    country: str = "ES",
    max_records: int = 5_000,
    page_size: int = 300,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Download a simple country-level GBIF sample."""
    params = {"country": country.upper(), "hasCoordinate": "true"}
    return download_gbif_occurrences(
        params=params,
        source_query=f"country_{str(country).upper()}",
        max_records=max_records,
        page_size=page_size,
        pause_seconds=pause_seconds,
    )


def download_gbif_occurrences(
    params: dict[str, Any],
    source_query: str,
    max_records: int,
    page_size: int = 300,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Download GBIF occurrence records with pagination."""
    all_records: list[dict[str, Any]] = []
    offset = 0
    page_size = max(1, min(page_size, 300))

    while len(all_records) < max_records:
        limit = min(page_size, max_records - len(all_records))
        request_params = {**params, "limit": limit, "offset": offset}

        try:
            response = requests.get(GBIF_OCCURRENCE_URL, params=request_params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"Error descargando {source_query} desde GBIF: {error}", flush=True)
            break

        payload = response.json()
        results = payload.get("results", [])
        if not results:
            break

        for record in results:
            record["source_query"] = source_query
            all_records.append(record)

        print(
            f"  {source_query}: descargados {len(all_records):,} / {max_records:,}",
            flush=True,
        )

        if payload.get("endOfRecords") is True:
            break

        offset += limit
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    return pd.DataFrame(all_records)
