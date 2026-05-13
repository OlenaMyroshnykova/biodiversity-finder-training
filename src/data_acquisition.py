"""Descarga neutral de datos desde GBIF para Biodiversity Finder.

La estrategia de descarga no usa especies bonitas ni consultas preparadas para el
 demo. El scope del proyecto se limita a animales y plantas, tal como pide el
 entregable: especies animales o plantas.
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
    """Configuración de una consulta taxonómica neutral a GBIF."""

    source_query: str
    description: str
    params: dict[str, Any]
    share: float


def is_global_country(country: str | None) -> bool:
    """Comprueba si el parámetro country significa búsqueda global."""

    if country is None:
        return True
    return str(country).strip().upper() in GLOBAL_COUNTRY_VALUES


def match_gbif_taxon_key(name: str, rank: str | None = None) -> int | None:
    """Busca un taxonKey en GBIF Backbone usando un nombre científico."""

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
    """Construye un plan neutral de consultas para animales y plantas.

    No se usan especies demo como flamencos, osos polares o jaguares. Tampoco se
    incluye Fungi, porque el enunciado habla de especies animales o plantas.
    """

    country_filter: dict[str, Any] = {}
    if not is_global_country(country):
        country_filter["country"] = str(country).upper()

    def with_common_params(params: dict[str, Any]) -> dict[str, Any]:
        merged = {"hasCoordinate": "true", **country_filter, **params}
        return {key: value for key, value in merged.items() if value is not None}

    taxon_keys = {
        "animalia": match_gbif_taxon_key("Animalia", "KINGDOM"),
        "plantae": match_gbif_taxon_key("Plantae", "KINGDOM"),
        "mammalia": match_gbif_taxon_key("Mammalia", "CLASS"),
        "aves": match_gbif_taxon_key("Aves", "CLASS"),
        "reptilia": match_gbif_taxon_key("Reptilia", "CLASS"),
        "amphibia": match_gbif_taxon_key("Amphibia", "CLASS"),
        "insecta": match_gbif_taxon_key("Insecta", "CLASS"),
        "arachnida": match_gbif_taxon_key("Arachnida", "CLASS"),
        "actinopterygii": match_gbif_taxon_key("Actinopterygii", "CLASS"),
        "chondrichthyes": match_gbif_taxon_key("Chondrichthyes", "CLASS"),
        "magnoliopsida": match_gbif_taxon_key("Magnoliopsida", "CLASS"),
        "liliopsida": match_gbif_taxon_key("Liliopsida", "CLASS"),
        "pinopsida": match_gbif_taxon_key("Pinopsida", "CLASS"),
        "polypodiopsida": match_gbif_taxon_key("Polypodiopsida", "CLASS"),
    }

    print(
        " taxon_keys resueltos:",
        {key: value for key, value in taxon_keys.items() if value is not None},
        flush=True,
    )

    return [
        GBIFQuery(
            source_query="kingdom_animalia",
            description="Muestra global neutral de animales: Animalia",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["animalia"],
                    "scientificName": None if taxon_keys["animalia"] else "Animalia",
                }
            ),
            share=0.05,
        ),
        GBIFQuery(
            source_query="class_mammalia",
            description="Clase Mammalia",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["mammalia"],
                    "scientificName": None if taxon_keys["mammalia"] else "Mammalia",
                }
            ),
            share=0.10,
        ),
        GBIFQuery(
            source_query="class_aves",
            description="Clase Aves",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["aves"],
                    "scientificName": None if taxon_keys["aves"] else "Aves",
                }
            ),
            share=0.12,
        ),
        GBIFQuery(
            source_query="class_reptilia",
            description="Clase Reptilia",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["reptilia"],
                    "scientificName": None if taxon_keys["reptilia"] else "Reptilia",
                }
            ),
            share=0.09,
        ),
        GBIFQuery(
            source_query="class_amphibia",
            description="Clase Amphibia",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["amphibia"],
                    "scientificName": None if taxon_keys["amphibia"] else "Amphibia",
                }
            ),
            share=0.09,
        ),
        GBIFQuery(
            source_query="class_insecta",
            description="Clase Insecta",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["insecta"],
                    "scientificName": None if taxon_keys["insecta"] else "Insecta",
                }
            ),
            share=0.14,
        ),
        GBIFQuery(
            source_query="class_arachnida",
            description="Clase Arachnida",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["arachnida"],
                    "scientificName": None if taxon_keys["arachnida"] else "Arachnida",
                }
            ),
            share=0.07,
        ),
        GBIFQuery(
            source_query="class_actinopterygii",
            description="Clase Actinopterygii",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["actinopterygii"],
                    "scientificName": None if taxon_keys["actinopterygii"] else "Actinopterygii",
                }
            ),
            share=0.10,
        ),
        GBIFQuery(
            source_query="class_chondrichthyes",
            description="Clase Chondrichthyes",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["chondrichthyes"],
                    "scientificName": None if taxon_keys["chondrichthyes"] else "Chondrichthyes",
                }
            ),
            share=0.05,
        ),
        GBIFQuery(
            source_query="class_magnoliopsida",
            description="Plantas: clase Magnoliopsida",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["magnoliopsida"],
                    "scientificName": None if taxon_keys["magnoliopsida"] else "Magnoliopsida",
                }
            ),
            share=0.10,
        ),
        GBIFQuery(
            source_query="class_liliopsida",
            description="Plantas: clase Liliopsida",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["liliopsida"],
                    "scientificName": None if taxon_keys["liliopsida"] else "Liliopsida",
                }
            ),
            share=0.05,
        ),
        GBIFQuery(
            source_query="class_pinopsida",
            description="Plantas: clase Pinopsida",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["pinopsida"],
                    "scientificName": None if taxon_keys["pinopsida"] else "Pinopsida",
                }
            ),
            share=0.02,
        ),
        GBIFQuery(
            source_query="class_polypodiopsida",
            description="Plantas: clase Polypodiopsida",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["polypodiopsida"],
                    "scientificName": None if taxon_keys["polypodiopsida"] else "Polypodiopsida",
                }
            ),
            share=0.02,
        ),
    ]


def download_biodiversity_training_dataset(
    country: str | None = "GLOBAL",
    max_records: int = 20_000,
    page_size: int = 300,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Descarga un dataset global y taxonómico para entrenar Biodiversity Finder."""

    query_plan = build_global_query_plan(country)
    frames: list[pd.DataFrame] = []

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
    """Función compatible con versiones anteriores."""

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
    """Descarga ocurrencias de GBIF usando parámetros arbitrarios."""

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

        print(
            f"  {source_query}: descargados {len(all_records):,} / {max_records:,}",
            flush=True,
        )
        offset += current_limit
        if len(records) < current_limit:
            break
        time.sleep(pause_seconds)

    return pd.DataFrame(all_records[:max_records])
