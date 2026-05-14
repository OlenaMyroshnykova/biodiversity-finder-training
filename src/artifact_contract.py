"""Contrato de artifact generado por training y consumido por Streamlit."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import pandas as pd

PROJECT_SCOPE_KINGDOMS = {"animalia", "plantae"}
THREATENED_IUCN_CATEGORIES = {"VU", "EN", "CR", "EW", "EX"}

SEARCH_CONTRACT_COLUMNS = [
    "scientific_name",
    "canonical_scientific_name",
    "vernacular_names",
    "common_name_es",
    "common_name_en",
    "kingdom",
    "phylum",
    "taxon_class",
    "taxon_order",
    "family",
    "genus",
    "species",
    "countries",
    "source_queries",
    "profile_text",
    "color_tag",
    "habitat_tag",
    "size_tag",
    "tags_de_busqueda",
    "iucn_category",
    "iucn_status_label",
    "conservation_status",
    "conservation_category",
]


def normalize_text(value: object) -> str:
    text = str(value or "").lower().strip()
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(ascii_text.split())


def canonicalize_scientific_name(value: object) -> str:
    """Nombre canónico simple: género + especie cuando es posible."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("(", " ").replace(")", " ").replace(",", " ")
    parts = [part for part in text.split() if part]
    if len(parts) >= 2:
        return " ".join(parts[:2])
    return text


def ensure_artifact_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza columnas base antes de publicar el parquet."""
    result = df.copy()
    for column in SEARCH_CONTRACT_COLUMNS:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)

    if "canonical_scientific_name" not in result.columns or result["canonical_scientific_name"].eq("").all():
        result["canonical_scientific_name"] = result["scientific_name"].apply(canonicalize_scientific_name)

    if "iucn_category" not in result.columns:
        result["iucn_category"] = "NO_DATA"
    result["iucn_category"] = result["iucn_category"].replace({"": "NO_DATA"})

    if "is_threatened" not in result.columns:
        result["is_threatened"] = False
    result["is_threatened"] = (
        result["is_threatened"].fillna(False).astype(bool)
        | result["iucn_category"].str.upper().isin(THREATENED_IUCN_CATEGORIES)
    )

    result["search_document"] = build_search_document(result)
    return result


def build_search_document(df: pd.DataFrame) -> pd.Series:
    """Documento único para búsqueda: nombres originales + versión normalizada.

    La app necesita una versión normalizada para buscar rápido y sin problemas
    de mayúsculas/acentos. Pero el artifact también debe conservar los nombres
    científicos originales para fichas, depuración y tests de contrato.
    Por eso guardamos ambas capas en el mismo documento:
    1. `preserved_document`: texto original legible, sin lower().
    2. `normalized_document`: texto normalizado para búsqueda.
    """
    raw_document = pd.Series([""] * len(df), index=df.index, dtype=str)
    for column in SEARCH_CONTRACT_COLUMNS:
        if column in df.columns:
            raw_document = raw_document + " " + df[column].fillna("").astype(str)

    preserved_document = raw_document.apply(lambda value: " ".join(str(value or "").split()))
    normalized_document = raw_document.apply(normalize_text)

    return (preserved_document + " " + normalized_document).apply(
        lambda value: " ".join(str(value or "").split())
    )
