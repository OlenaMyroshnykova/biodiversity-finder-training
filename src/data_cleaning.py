"""Limpieza de datos de biodiversidad."""

from __future__ import annotations

import pandas as pd

PROJECT_SCOPE_KINGDOMS = {"animalia", "plantae"}


def clean_occurrences(raw_df: pd.DataFrame, min_class_records: int = 50) -> pd.DataFrame:
    """Limpia registros crudos de GBIF y limita el scope a animales y plantas."""

    cleaned_df = raw_df.copy()
    cleaned_df = normalize_column_names(cleaned_df)
    cleaned_df = keep_required_data(cleaned_df)
    cleaned_df = keep_project_scope(cleaned_df)
    cleaned_df = clean_coordinates(cleaned_df)
    cleaned_df = clean_dates(cleaned_df)
    cleaned_df = remove_duplicates(cleaned_df)
    cleaned_df = remove_rare_classes(cleaned_df, min_class_records)
    return cleaned_df.reset_index(drop=True)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas para trabajar de forma consistente."""

    return df.rename(
        columns={
            "scientificName": "scientific_name",
            "acceptedScientificName": "accepted_scientific_name",
            "countryCode": "country_code",
            "decimalLatitude": "decimal_latitude",
            "decimalLongitude": "decimal_longitude",
            "basisOfRecord": "basis_of_record",
            "individualCount": "individual_count",
            "coordinateUncertaintyInMeters": "coordinate_uncertainty_meters",
            "class": "taxon_class",
            "order": "taxon_order",
            "iucnRedListCategory": "iucn_red_list_category",
        }
    )


def keep_required_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas sin campos básicos."""

    required_columns = [
        "key",
        "scientific_name",
        "kingdom",
        "phylum",
        "taxon_class",
        "family",
        "country_code",
        "decimal_latitude",
        "decimal_longitude",
        "year",
        "month",
        "basis_of_record",
    ]
    existing = [column for column in required_columns if column in df.columns]
    return df.dropna(subset=existing).copy()


def keep_project_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Mantiene solo Animalia y Plantae.

    El enunciado del proyecto habla de especies animales o plantas. Fungi y otros
    reinos se eliminan aunque entren por una muestra global antigua o por un
    artefacto heredado.
    """

    if "kingdom" not in df.columns:
        return df.copy()

    kingdoms = df["kingdom"].fillna("").astype(str).str.strip().str.lower()
    return df[kingdoms.isin(PROJECT_SCOPE_KINGDOMS)].copy()


def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte coordenadas a números y elimina valores imposibles."""

    result_df = df.copy()
    result_df["decimal_latitude"] = pd.to_numeric(result_df["decimal_latitude"], errors="coerce")
    result_df["decimal_longitude"] = pd.to_numeric(result_df["decimal_longitude"], errors="coerce")
    return result_df[
        result_df["decimal_latitude"].between(-90, 90)
        & result_df["decimal_longitude"].between(-180, 180)
    ].copy()


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia año y mes de observación."""

    result_df = df.copy()
    result_df["year"] = pd.to_numeric(result_df["year"], errors="coerce")
    result_df["month"] = pd.to_numeric(result_df["month"], errors="coerce")
    result_df = result_df[
        result_df["year"].between(1900, 2100)
        & result_df["month"].between(1, 12)
    ].copy()
    result_df["year"] = result_df["year"].astype(int)
    result_df["month"] = result_df["month"].astype(int)
    return result_df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados básicos de ocurrencias."""

    subset_columns = [
        "scientific_name",
        "decimal_latitude",
        "decimal_longitude",
        "year",
        "month",
        "basis_of_record",
    ]
    return df.drop_duplicates(subset=subset_columns).copy()


def remove_rare_classes(df: pd.DataFrame, min_class_records: int) -> pd.DataFrame:
    """Elimina clases taxonómicas con muy pocos ejemplos."""

    if min_class_records <= 1 or "taxon_class" not in df.columns:
        return df.copy()

    class_counts = df["taxon_class"].value_counts()
    valid_classes = class_counts[class_counts >= min_class_records].index
    return df[df["taxon_class"].isin(valid_classes)].copy()
