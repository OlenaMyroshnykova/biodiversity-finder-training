"""Generación de puntos de avistamiento para mapas Folium."""

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


def build_species_occurrence_points(
    occurrences_df: pd.DataFrame,
    *,
    max_points_per_species: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Construye tabla ligera de coordenadas por especie.

    Esta función debe recibir preferentemente `clean_df`, no `features_df`,
    porque los datos limpios conservan las coordenadas originales de GBIF.
    """
    empty_df = build_empty_occurrence_points()

    if occurrences_df.empty:
        return empty_df

    latitude_column = find_first_existing_column(occurrences_df, LATITUDE_CANDIDATES)
    longitude_column = find_first_existing_column(occurrences_df, LONGITUDE_CANDIDATES)

    if not latitude_column or not longitude_column:
        return empty_df

    if "scientific_name" not in occurrences_df.columns:
        return empty_df

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
        return empty_df

    working_df["canonical_scientific_name"] = working_df["scientific_name"].apply(
        canonicalize_scientific_name
    )

    country_column = find_first_existing_column(working_df, COUNTRY_CANDIDATES)
    date_column = find_first_existing_column(working_df, DATE_CANDIDATES)

    points_df = pd.DataFrame(
        {
            "scientific_name": working_df["scientific_name"].astype(str),
            "canonical_scientific_name": working_df["canonical_scientific_name"].astype(str),
            "decimalLatitude": working_df[latitude_column].astype(float),
            "decimalLongitude": working_df[longitude_column].astype(float),
            "countryCode": (
                working_df[country_column].fillna("").astype(str)
                if country_column
                else ""
            ),
            "eventDate": (
                working_df[date_column].fillna("").astype(str)
                if date_column
                else ""
            ),
        }
    )

    optional_columns = [
        "taxon_class",
        "family",
        "source_query",
        "source_queries",
    ]

    for column in optional_columns:
        if column in working_df.columns:
            points_df[column] = working_df[column].fillna("").astype(str)

    points_df = points_df.drop_duplicates(
        subset=["canonical_scientific_name", "decimalLatitude", "decimalLongitude"]
    )

    points_df = (
        points_df
        .groupby("canonical_scientific_name", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), max_points_per_species),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )

    return points_df


def build_empty_occurrence_points() -> pd.DataFrame:
    """Devuelve una tabla vacía con el schema esperado por la app."""
    return pd.DataFrame(
        columns=[
            "scientific_name",
            "canonical_scientific_name",
            "decimalLatitude",
            "decimalLongitude",
            "countryCode",
            "eventDate",
        ]
    )


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
