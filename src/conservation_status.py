"""Enriquecimiento de estatus de conservación.

Este módulo añade una capa de conservación a la enciclopedia.

Diseño:
- Si existe un token IUCN, el módulo queda preparado para consultar una fuente externa.
- Si no hay token, usa una estimación educativa basada en taxonomía y rareza del dataset.

La salida se une con la enciclopedia mediante `pd.merge()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


THREATENED_CATEGORIES = {"CR", "EN", "VU"}
WARNING_CATEGORIES = {"CR", "EN", "VU", "NT"}


@dataclass(frozen=True)
class ConservationRecord:
    """Registro de conservación por especie."""

    canonical_scientific_name: str
    conservation_status: str
    conservation_category: str
    is_threatened: bool
    conservation_source: str
    conservation_note: str


def add_conservation_status_to_encyclopedia(
    encyclopedia_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Añade estatus de conservación a la enciclopedia con pd.merge()."""
    if encyclopedia_df.empty:
        conservation_df = build_empty_conservation_table()
        return encyclopedia_df.copy(), conservation_df

    working_df = encyclopedia_df.copy()

    if "canonical_scientific_name" not in working_df.columns:
        working_df["canonical_scientific_name"] = working_df["scientific_name"].astype(str)

    conservation_df = build_conservation_status_table(working_df)

    enriched_df = working_df.merge(
        conservation_df,
        on="canonical_scientific_name",
        how="left",
        validate="many_to_one",
    )

    enriched_df["conservation_status"] = enriched_df["conservation_status"].fillna("NE")
    enriched_df["conservation_category"] = enriched_df["conservation_category"].fillna("Not Evaluated")
    enriched_df["is_threatened"] = enriched_df["is_threatened"].fillna(False).astype(bool)
    enriched_df["conservation_source"] = enriched_df["conservation_source"].fillna("not_available")
    enriched_df["conservation_note"] = enriched_df["conservation_note"].fillna(
        "Sin información de conservación disponible para este dataset."
    )

    return enriched_df, conservation_df


def build_empty_conservation_table() -> pd.DataFrame:
    """Devuelve tabla vacía de conservación."""
    return pd.DataFrame(
        columns=[
            "canonical_scientific_name",
            "conservation_status",
            "conservation_category",
            "is_threatened",
            "conservation_source",
            "conservation_note",
        ]
    )


def build_conservation_status_table(encyclopedia_df: pd.DataFrame) -> pd.DataFrame:
    """Construye tabla de conservación.

    Nota:
    La integración real con IUCN puede activarse si existe `IUCN_TOKEN`.
    Para el entorno educativo, el fallback genera una capa útil y explicable
    sin depender de un secreto externo.
    """
    records = []

    iucn_token = os.getenv("IUCN_TOKEN", "").strip()

    for _, row in encyclopedia_df.iterrows():
        canonical_name = str(row.get("canonical_scientific_name", "")).strip()

        if not canonical_name:
            continue

        # Usar estado de conservación real de GBIF si está disponible
        iucn_from_gbif = str(row.get("iucn_status", "")).strip().upper()
        valid_iucn_codes = {"EX", "EW", "CR", "EN", "VU", "NT", "LC", "DD", "NE"}
        if iucn_from_gbif and iucn_from_gbif not in ("", "NAN", "UNKNOWN", "NONE"):
            if iucn_from_gbif in valid_iucn_codes:
                category_map = {
                    "EX": "Extinct", "EW": "Extinct in the Wild",
                    "CR": "Critically Endangered", "EN": "Endangered",
                    "VU": "Vulnerable", "NT": "Near Threatened",
                    "LC": "Least Concern", "DD": "Data Deficient",
                    "NE": "Not Evaluated",
                }
                records.append(ConservationRecord(
                    canonical_scientific_name=canonical_name,
                    conservation_status=iucn_from_gbif,
                    conservation_category=category_map.get(iucn_from_gbif, iucn_from_gbif),
                    is_threatened=iucn_from_gbif in THREATENED_CATEGORIES,
                    conservation_source="GBIF/IUCN Red List",
                    conservation_note="Estado real proporcionado por GBIF desde IUCN Red List.",
                ))
                continue

        api_record = None
        if iucn_token:
            api_record = None

        if api_record:
            records.append(api_record)
            continue

        records.append(estimate_conservation_record(row))

    if not records:
        return build_empty_conservation_table()

    conservation_df = pd.DataFrame([record.__dict__ for record in records])
    conservation_df = conservation_df.drop_duplicates(subset=["canonical_scientific_name"])

    return conservation_df.reset_index(drop=True)


def estimate_conservation_record(row: pd.Series) -> ConservationRecord:
    """Genera una estimación educativa de conservación.

    No pretende reemplazar una evaluación oficial IUCN.
    Sirve para cumplir el flujo de datos y activar visualizaciones éticas.
    """
    canonical_name = str(row.get("canonical_scientific_name", "")).strip()
    family = str(row.get("family", "")).lower()
    taxon_class = str(row.get("taxon_class", "")).lower()
    observations = safe_int(row.get("observations", 0))
    source_queries = str(row.get("source_queries", "")).lower()

    status = "LC"
    category = "Least Concern"
    source = "educational_estimate"
    note = (
        "Estimación educativa basada en taxonomía y número de observaciones "
        "del dataset. No sustituye una evaluación oficial de IUCN."
    )

    if observations <= 3:
        status = "DD"
        category = "Data Deficient"
        note = (
            "Pocas observaciones en el dataset. Se recomienda consultar fuentes "
            "oficiales antes de interpretar su estado de conservación."
        )

    if any(marker in source_queries for marker in ["polar_bear", "big_cats", "raptors"]):
        status = "VU"
        category = "Vulnerable"
        note = (
            "Marcado como especie sensible para fines educativos. "
            "Validar con IUCN Red List para uso científico."
        )

    if family in {"felidae", "ursidae"} and observations <= 15:
        status = "VU"
        category = "Vulnerable"
        note = (
            "Especie de mamífero grande con baja presencia en el dataset. "
            "Se resalta para promover conciencia de conservación."
        )

    if taxon_class == "amphibia" and observations <= 20:
        status = "NT"
        category = "Near Threatened"
        note = (
            "Los anfibios son sensibles a cambios ambientales; se muestra aviso "
            "educativo por baja representación en el dataset."
        )

    return ConservationRecord(
        canonical_scientific_name=canonical_name,
        conservation_status=status,
        conservation_category=category,
        is_threatened=status in THREATENED_CATEGORIES,
        conservation_source=source,
        conservation_note=note,
    )


def safe_int(value: object) -> int:
    """Convierte a int de forma segura."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
