"""Descarga de datos desde GBIF para Biodiversity Finder."""

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
    """Configuración de una consulta temática a GBIF."""

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
    """
    Busca un taxonKey en GBIF Backbone usando el nombre científico.
    """
    params: dict[str, Any] = {"name": name}

    if rank:
        params["rank"] = rank

    try:
        response = requests.get(
            GBIF_SPECIES_MATCH_URL,
            params=params,
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    payload = response.json()
    usage_key = payload.get("usageKey")

    if isinstance(usage_key, int):
        return usage_key

    return None


def build_global_query_plan(country: str | None = "GLOBAL") -> list[GBIFQuery]:
    """
    Construye el plan de consultas para un dataset global y temático.

    Si `country` no es GLOBAL, se añade filtro de país a todas las consultas.
    Para este proyecto se recomienda `country=GLOBAL`.
    """
    country_filter: dict[str, Any] = {}

    if not is_global_country(country):
        country_filter["country"] = str(country).upper()

    def with_common_params(params: dict[str, Any]) -> dict[str, Any]:
        merged_params = {
            "hasCoordinate": "true",
            **country_filter,
            **params,
        }
        return {key: value for key, value in merged_params.items() if value is not None}

    taxon_keys = {
        "flamingo": match_gbif_taxon_key("Phoenicopterus roseus", "SPECIES"),
        "polar_bear": match_gbif_taxon_key("Ursus maritimus", "SPECIES"),
        "lepidoptera": match_gbif_taxon_key("Lepidoptera", "ORDER"),
        "amphibia": match_gbif_taxon_key("Amphibia", "CLASS"),
        "accipitridae": match_gbif_taxon_key("Accipitridae", "FAMILY"),
        "magnoliopsida": match_gbif_taxon_key("Magnoliopsida", "CLASS"),
        "mammalia": match_gbif_taxon_key("Mammalia", "CLASS"),
        "jaguar": match_gbif_taxon_key("Panthera onca", "SPECIES"),
        "felidae": match_gbif_taxon_key("Felidae", "FAMILY"),
    }

    return [
        GBIFQuery(
            source_query="general_global",
            description="Muestra general global de biodiversidad",
            params=with_common_params({}),
            share=0.24,
        ),
        GBIFQuery(
            source_query="flamingo_pink_bird",
            description="Ave rosa: Phoenicopterus roseus",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["flamingo"],
                    "scientificName": None if taxon_keys["flamingo"] else "Phoenicopterus roseus",
                }
            ),
            share=0.07,
        ),
        GBIFQuery(
            source_query="polar_bear",
            description="Animal polar de hielo: Ursus maritimus",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["polar_bear"],
                    "scientificName": None if taxon_keys["polar_bear"] else "Ursus maritimus",
                }
            ),
            share=0.07,
        ),
        GBIFQuery(
            source_query="jaguar_panthera_onca",
            description="Jaguar / ягуар: Panthera onca",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["jaguar"],
                    "scientificName": None if taxon_keys["jaguar"] else "Panthera onca",
                }
            ),
            share=0.07,
        ),
        GBIFQuery(
            source_query="big_cats_felidae",
            description="Felinos y grandes gatos: familia Felidae",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["felidae"],
                    "scientificName": None if taxon_keys["felidae"] else "Felidae",
                }
            ),
            share=0.08,
        ),
        GBIFQuery(
            source_query="butterflies_lepidoptera",
            description="Mariposas y polillas: orden Lepidoptera",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["lepidoptera"],
                    "scientificName": None if taxon_keys["lepidoptera"] else "Lepidoptera",
                }
            ),
            share=0.12,
        ),
        GBIFQuery(
            source_query="amphibians",
            description="Ranas y anfibios: clase Amphibia",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["amphibia"],
                    "scientificName": None if taxon_keys["amphibia"] else "Amphibia",
                }
            ),
            share=0.09,
        ),
        GBIFQuery(
            source_query="raptors_accipitridae",
            description="Aves rapaces: familia Accipitridae",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["accipitridae"],
                    "scientificName": None if taxon_keys["accipitridae"] else "Accipitridae",
                }
            ),
            share=0.08,
        ),
        GBIFQuery(
            source_query="flowering_plants",
            description="Plantas con flor: Magnoliopsida",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["magnoliopsida"],
                    "scientificName": None if taxon_keys["magnoliopsida"] else "Magnoliopsida",
                }
            ),
            share=0.10,
        ),
        GBIFQuery(
            source_query="mammals",
            description="Mamíferos: Mammalia",
            params=with_common_params(
                {
                    "taxonKey": taxon_keys["mammalia"],
                    "scientificName": None if taxon_keys["mammalia"] else "Mammalia",
                }
            ),
            share=0.08,
        ),
    ]


def download_biodiversity_training_dataset(
    country: str | None = "GLOBAL",
    max_records: int = 20_000,
    page_size: int = 300,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """
    Descarga un dataset global y temático para entrenar Biodiversity Finder.
    """
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

    combined_df = combined_df.head(max_records).reset_index(drop=True)

    print(
        f"Dataset combinado final: {len(combined_df):,} registros "
        f"desde {len(frames)} consultas.",
        flush=True,
    )

    return combined_df


def download_gbif_occurrences(
    country: str | None = "GLOBAL",
    max_records: int = 20_000,
    page_size: int = 300,
) -> pd.DataFrame:
    """
    Función compatible con versiones anteriores.
    """
    if is_global_country(country):
        return download_biodiversity_training_dataset(
            country="GLOBAL",
            max_records=max_records,
            page_size=page_size,
        )

    params = {
        "country": str(country).upper(),
        "hasCoordinate": "true",
    }

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
    """
    Descarga ocurrencias de GBIF usando parámetros arbitrarios.
    """
    all_records: list[dict[str, Any]] = []
    offset = 0

    while len(all_records) < max_records:
        current_limit = min(page_size, max_records - len(all_records))

        request_params = {
            **params,
            "limit": current_limit,
            "offset": offset,
        }

        try:
            response = requests.get(
                GBIF_OCCURRENCE_URL,
                params=request_params,
                timeout=60,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            print(
                f"Error descargando {source_query} desde GBIF: {error}",
                flush=True,
            )
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
