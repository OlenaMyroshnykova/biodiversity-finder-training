"""Generación de puntos de avistamiento para mapas Folium.

Este módulo crea el artefacto:

    data/processed/species_occurrence_points.parquet

La app necesita este archivo para pintar puntos en el mapa Folium.
"""

from __future__ import annotations

import re

import pandas as pd


LATITUDE_CANDIDATES = [
    "decimalLatitude",
    "decimal_latitude",
    "latitude",
    "lat",
    "avg_latitude",
]

LONGITUDE_CANDIDATES = [
    "decimalLongitude",
    "decimal_longitude",
    "longitude",
    "lon",
    "lng",
    "avg_longitude",
]

COUNTRY_CANDIDATES = [
    "countryCode",
    "country_code",
    "country",
]

DATE_CANDIDATES = [
    "eventDate",
    "event_date",
    "year",
]

OUTPUT_COLUMNS = [
    "scientific_name",
    "canonical_scientific_name",
    "decimalLatitude",
    "decimalLongitude",
    "countryCode",
    "eventDate",
]


def build_species_occurrence_points(
    occurrences_df: pd.DataFrame,
    *,
    max_points_per_species: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Construye tabla ligera de coordenadas por especie.

    Debe recibir preferentemente `clean_df`, porque ahí se conservan
    las coordenadas originales de GBIF.
    """
    if occurrences_df.empty:
        return build_empty_occurrence_points()

    latitude_column = find_first_existing_column(occurrences_df, LATITUDE_CANDIDATES)
    longitude_column = find_first_existing_column(occurrences_df, LONGITUDE_CANDIDATES)

    if not latitude_column or not longitude_column:
        return build_empty_occurrence_points()

    if "scientific_name" not in occurrences_df.columns:
        return build_empty_occurrence_points()

    working_df = occurrences_df.copy()

    working_df[latitude_column] = pd.to_numeric(
        working_df[latitude_column],
        errors="coerce",
    )
    working_df[longitude_column] = pd.to_numeric(
        working_df[longitude_column],
        errors="coerce",
    )

    working_df = working_df.dropna(subset=[latitude_column, longitude_column])
    working_df = working_df[
        working_df[latitude_column].between(-90, 90)
        & working_df[longitude_column].between(-180, 180)
    ].copy()

    if working_df.empty:
        return build_empty_occurrence_points()

    country_column = find_first_existing_column(working_df, COUNTRY_CANDIDATES)
    date_column = find_first_existing_column(working_df, DATE_CANDIDATES)

    points_df = pd.DataFrame(
        {
            "scientific_name": working_df["scientific_name"].fillna("").astype(str),
            "canonical_scientific_name": (
                working_df["scientific_name"]
                .fillna("")
                .astype(str)
                .apply(canonicalize_scientific_name)
            ),
            "decimalLatitude": working_df[latitude_column].astype(float),
            "decimalLongitude": working_df[longitude_column].astype(float),
            "countryCode": build_optional_text_series(working_df, country_column),
            "eventDate": build_optional_text_series(working_df, date_column),
        },
        index=working_df.index,
    )

    for optional_column in ["taxon_class", "family", "source_query", "source_queries"]:
        if optional_column in working_df.columns:
            points_df[optional_column] = working_df[optional_column].fillna("").astype(str)

    points_df = points_df.drop_duplicates(
        subset=["canonical_scientific_name", "decimalLatitude", "decimalLongitude"]
    )

    points_df = limit_points_per_species(
        points_df,
        max_points_per_species=max_points_per_species,
        random_state=random_state,
    )

    return points_df.reset_index(drop=True)


def build_optional_text_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Devuelve una serie de texto para columnas opcionales."""
    if column and column in df.columns:
        return df[column].fillna("").astype(str)

    return pd.Series([""] * len(df), index=df.index, dtype=str)


def limit_points_per_species(
    points_df: pd.DataFrame,
    *,
    max_points_per_species: int,
    random_state: int,
) -> pd.DataFrame:
    """Limita el número de puntos por especie."""
    if points_df.empty:
        return points_df

    if max_points_per_species <= 0:
        return points_df

    return (
        points_df
        .groupby("canonical_scientific_name", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), max_points_per_species),
                random_state=random_state,
            )
        )
    )


def build_empty_occurrence_points() -> pd.DataFrame:
    """Devuelve una tabla vacía con el schema esperado por la app."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    """Encuentra la primera columna disponible."""
    for column in candidates:
        if column in df.columns:
            return column

    return ""


def canonicalize_scientific_name(scientific_name: object) -> str:
    """Quita autoría del nombre científico para facilitar filtros por mapa."""
    text = str(scientific_name or "").strip()
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text
