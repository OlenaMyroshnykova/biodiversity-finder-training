"""Nombres comunes / vernacular names para especies.

Este módulo crea una tabla de nombres comunes desde GBIF Species API y la une
con la enciclopedia mediante `df.merge()`.

Objetivo del proyecto:
- Las personas buscan "leopardo", "jaguar", "frog", "mariposa".
- GBIF suele trabajar con "Panthera pardus", "Panthera onca", etc.
- Este módulo añade el puente entre nombres científicos y nombres humanos.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests


GBIF_SPECIES_URL = "https://api.gbif.org/v1/species"

SUPPORTED_LANGUAGE_PRIORITY = [
    "spa",
    "eng",
    "rus",
    "ukr",
    "por",
    "ita",
    "cat",
    "fra",
    "deu",
    "lat",
    "",
]


@dataclass(frozen=True)
class VernacularNameRecord:
    """Registro individual de nombre común."""

    scientific_name: str
    species_key: str
    language: str
    vernacular_name: str
    source: str


def add_vernacular_names_to_encyclopedia(
    encyclopedia_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    max_species: int = 800,
    use_api: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Añade nombres comunes a la enciclopedia usando df.merge().

    Returns:
        Tupla: enciclopedia enriquecida, tabla larga de nombres comunes.
    """
    if encyclopedia_df.empty:
        empty_names_df = build_empty_vernacular_names()
        return encyclopedia_df.copy(), empty_names_df

    species_lookup_df = build_species_lookup(
        encyclopedia_df=encyclopedia_df,
        features_df=features_df,
    )

    vernacular_names_df = build_vernacular_names_table(
        species_lookup_df=species_lookup_df,
        max_species=max_species,
        use_api=use_api,
    )

    vernacular_summary_df = summarize_vernacular_names(vernacular_names_df)

    enriched_df = encyclopedia_df.merge(
        vernacular_summary_df,
        on="scientific_name",
        how="left",
        validate="one_to_one",
    )

    enriched_df["vernacular_names"] = enriched_df["vernacular_names"].fillna("")
    enriched_df["vernacular_languages"] = enriched_df["vernacular_languages"].fillna("")

    enriched_df = enrich_search_document_with_vernacular_names(enriched_df)

    return enriched_df, vernacular_names_df


def build_species_lookup(
    encyclopedia_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Crea tabla species -> key para consultar GBIF Species API.

    Intenta usar speciesKey, acceptedTaxonKey, taxonKey o key si existen.
    """
    key_candidates = [
        "speciesKey",
        "acceptedTaxonKey",
        "taxonKey",
        "key",
        "species_key",
        "taxon_key",
    ]

    available_keys = [column for column in key_candidates if column in features_df.columns]

    columns = ["scientific_name"]

    if available_keys:
        columns.append(available_keys[0])

    lookup_df = encyclopedia_df[["scientific_name"]].drop_duplicates().copy()

    if available_keys:
        key_column = available_keys[0]

        feature_keys_df = (
            features_df[["scientific_name", key_column]]
            .dropna()
            .drop_duplicates(subset=["scientific_name"])
            .copy()
        )

        lookup_df = lookup_df.merge(
            feature_keys_df,
            on="scientific_name",
            how="left",
            validate="one_to_one",
        )

        lookup_df = lookup_df.rename(columns={key_column: "species_key"})
    else:
        lookup_df["species_key"] = ""

    lookup_df["species_key"] = lookup_df["species_key"].fillna("").astype(str)

    return lookup_df


def build_empty_vernacular_names() -> pd.DataFrame:
    """Devuelve tabla vacía de nombres comunes."""
    return pd.DataFrame(
        columns=[
            "scientific_name",
            "species_key",
            "language",
            "vernacular_name",
            "source",
        ]
    )


def build_vernacular_names_table(
    species_lookup_df: pd.DataFrame,
    *,
    max_species: int = 800,
    use_api: bool = True,
) -> pd.DataFrame:
    """Construye tabla larga de nombres comunes."""
    if species_lookup_df.empty:
        return build_empty_vernacular_names()

    records: list[VernacularNameRecord] = []

    limited_lookup_df = species_lookup_df.head(max_species).copy()

    for _, row in limited_lookup_df.iterrows():
        scientific_name = str(row.get("scientific_name", "")).strip()
        species_key = str(row.get("species_key", "")).strip()

        if not scientific_name:
            continue

        api_records: list[VernacularNameRecord] = []

        if use_api and species_key:
            api_records = fetch_gbif_vernacular_names(
                scientific_name=scientific_name,
                species_key=species_key,
            )

        if not api_records:
            api_records = build_fallback_scientific_name_records(
                scientific_name=scientific_name,
                species_key=species_key,
            )

        records.extend(api_records)

    if not records:
        return build_empty_vernacular_names()

    names_df = pd.DataFrame([record.__dict__ for record in records])
    names_df["vernacular_name"] = names_df["vernacular_name"].astype(str).str.strip()
    names_df = names_df[names_df["vernacular_name"] != ""]
    names_df = names_df.drop_duplicates(
        subset=["scientific_name", "language", "vernacular_name"]
    )

    return names_df.reset_index(drop=True)


def fetch_gbif_vernacular_names(
    *,
    scientific_name: str,
    species_key: str,
    timeout: int = 20,
) -> list[VernacularNameRecord]:
    """Descarga nombres comunes desde GBIF Species API."""
    url = f"{GBIF_SPECIES_URL}/{species_key}/vernacularNames"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    results = payload.get("results", [])

    records: list[VernacularNameRecord] = []

    for item in results:
        if not isinstance(item, dict):
            continue

        vernacular_name = str(item.get("vernacularName", "")).strip()
        language = str(item.get("language", "")).strip().lower()

        if not vernacular_name:
            continue

        records.append(
            VernacularNameRecord(
                scientific_name=scientific_name,
                species_key=species_key,
                language=language,
                vernacular_name=vernacular_name,
                source="GBIF Species API",
            )
        )

    return sort_vernacular_records(records)


def build_fallback_scientific_name_records(
    *,
    scientific_name: str,
    species_key: str,
) -> list[VernacularNameRecord]:
    """
    Fallback mínimo para que search_document siempre tenga el nombre científico.
    """
    return [
        VernacularNameRecord(
            scientific_name=scientific_name,
            species_key=species_key,
            language="lat",
            vernacular_name=scientific_name,
            source="scientific_name_fallback",
        )
    ]


def sort_vernacular_records(
    records: list[VernacularNameRecord],
) -> list[VernacularNameRecord]:
    """Ordena nombres comunes priorizando idiomas útiles para el proyecto."""
    priority = {
        language: index
        for index, language in enumerate(SUPPORTED_LANGUAGE_PRIORITY)
    }

    return sorted(
        records,
        key=lambda record: (
            priority.get(record.language, 999),
            record.vernacular_name.lower(),
        ),
    )


def summarize_vernacular_names(vernacular_names_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa nombres comunes por especie para poder hacer merge one_to_one."""
    if vernacular_names_df.empty:
        return pd.DataFrame(
            columns=[
                "scientific_name",
                "vernacular_names",
                "vernacular_languages",
            ]
        )

    grouped_df = (
        vernacular_names_df
        .groupby("scientific_name", as_index=False)
        .agg(
            vernacular_names=(
                "vernacular_name",
                lambda values: " | ".join(sorted(set(map(str, values)))),
            ),
            vernacular_languages=(
                "language",
                lambda values: " | ".join(sorted(set(map(str, values)))),
            ),
        )
    )

    return grouped_df


def enrich_search_document_with_vernacular_names(
    encyclopedia_df: pd.DataFrame,
) -> pd.DataFrame:
    """Añade nombres comunes a search_document y profile_text."""
    enriched_df = encyclopedia_df.copy()

    if "search_document" not in enriched_df.columns:
        enriched_df["search_document"] = ""

    enriched_df["search_document"] = (
        enriched_df["search_document"].fillna("").astype(str)
        + " "
        + enriched_df["vernacular_names"].fillna("").astype(str)
    )

    if "profile_text" in enriched_df.columns:
        enriched_df["profile_text"] = enriched_df.apply(
            add_common_names_to_profile_text,
            axis=1,
        )

    return enriched_df


def add_common_names_to_profile_text(row: pd.Series) -> str:
    """Añade una frase corta con nombres comunes al perfil de especie."""
    profile_text = str(row.get("profile_text", "") or "")
    vernacular_names = str(row.get("vernacular_names", "") or "").strip()

    if not vernacular_names:
        return profile_text

    short_names = " / ".join(
        name.strip()
        for name in vernacular_names.split("|")[:5]
        if name.strip()
    )

    if not short_names:
        return profile_text

    if "Nombres comunes:" in profile_text:
        return profile_text

    return f"{profile_text} Nombres comunes: {short_names}."
