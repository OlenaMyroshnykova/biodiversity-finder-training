"""Generación de tags de búsqueda para consultas tipo vibe.

`tags_de_busqueda` cumple el requerimiento del enunciado: color + hábitat + tamaño.
`search_document` se construye aparte en `artifact_contract` con nombres, taxonomía,
tags e IUCN para que la app no necesite hacks.
"""
from __future__ import annotations

import pandas as pd

from src.artifact_contract import ensure_artifact_contract, normalize_text


def add_search_tags_to_encyclopedia(encyclopedia_df: pd.DataFrame) -> pd.DataFrame:
    """Añade color_tag, habitat_tag, size_tag, tags_de_busqueda y search_document."""
    tagged_df = encyclopedia_df.copy()
    tagged_df["color_tag"] = tagged_df.apply(infer_color_tag, axis=1)
    tagged_df["habitat_tag"] = tagged_df.apply(infer_habitat_tag, axis=1)
    tagged_df["size_tag"] = tagged_df.apply(infer_size_tag, axis=1)
    tagged_df["tags_de_busqueda"] = normalize_tag_series(
        tagged_df["color_tag"].fillna("").astype(str)
        + " "
        + tagged_df["habitat_tag"].fillna("").astype(str)
        + " "
        + tagged_df["size_tag"].fillna("").astype(str)
    )
    return ensure_artifact_contract(tagged_df)


def normalize_tag_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(normalize_text)


def infer_color_tag(row: pd.Series) -> str:
    text = build_row_text(row)
    if has_any(text, ["amphibia", "hylidae"]):
        return "green brown verde marron"
    if has_any(text, ["insecta", "lepidoptera", "papilionidae"]):
        return "colorful bright multicolor colorido"
    if has_any(text, ["felidae", "ursidae", "canidae", "carnivora", "mammalia"]):
        return "brown grey white black marron gris blanco negro"
    if has_any(text, ["plantae", "magnoliopsida", "tracheophyta"]):
        return "green colorful pink white verde colorido rosa blanco"
    if has_any(text, ["reptilia", "crocodylia", "serpentes", "lacertilia"]):
        return "green brown grey verde marron gris"
    if has_any(text, ["actinopterygii", "teleostei"]):
        return "silver blue colorful plateado azul colorido"
    if has_any(text, ["chondrichthyes", "selachimorpha"]):
        return "grey blue gris azul"
    if has_any(text, ["arachnida", "araneae", "scorpiones"]):
        return "brown black marron negro"
    if has_any(text, ["aves"]):
        return "brown white colorful pink blue marron blanco colorido rosa azul"
    return "unknown"


def infer_habitat_tag(row: pd.Series) -> str:
    text = build_row_text(row)
    latitude = safe_float(row.get("avg_latitude", row.get("decimal_latitude", None)))
    if latitude is not None and abs(latitude) >= 60:
        return "polar tundra cold ice hielo polar frio"
    if has_any(text, ["amphibia", "hylidae"]):
        return "wetland river water humedal rio agua"
    if has_any(text, ["aves", "anatidae", "laridae"]):
        return "wetland forest coast humedal bosque costa"
    if has_any(text, ["felidae", "ursidae", "canidae", "carnivora", "mammalia"]):
        return "forest mountain terrestrial bosque montana montaña terrestre"
    if has_any(text, ["insecta", "lepidoptera"]):
        return "meadow forest garden pradera bosque jardin jardín"
    if has_any(text, ["plantae", "magnoliopsida", "tracheophyta"]):
        return "terrestrial meadow garden forest pradera jardin jardín bosque"
    if has_any(text, ["crocodylia"]):
        return "wetland river tropical humedal rio tropical"
    if has_any(text, ["reptilia", "serpentes", "lacertilia"]):
        return "forest desert tropical bosque desierto tropical"
    if has_any(text, ["actinopterygii", "teleostei"]):
        return "ocean river lake aquatic oceano océano rio lago acuatico acuático"
    if has_any(text, ["chondrichthyes", "selachimorpha"]):
        return "ocean sea marine oceano océano mar marino"
    if has_any(text, ["arachnida", "araneae", "scorpiones"]):
        return "terrestrial forest desert bosque desierto terrestre"
    return "unknown"


def infer_size_tag(row: pd.Series) -> str:
    text = build_row_text(row)
    if has_any(text, ["ursidae", "felidae", "crocodylia", "chondrichthyes"]):
        return "large big grande gigante"
    if has_any(text, ["mammalia"]):
        return "medium large mediano grande"
    if has_any(text, ["aves"]):
        return "small medium mediano pequeno pequeño"
    if has_any(text, ["insecta", "lepidoptera", "arachnida", "araneae"]):
        return "small tiny pequeno pequeño mini"
    if has_any(text, ["amphibia", "plantae", "fungi", "reptilia", "serpentes", "lacertilia", "actinopterygii"]):
        return "small medium pequeno pequeño mediano"
    return "unknown"


def build_row_text(row: pd.Series) -> str:
    columns = [
        "kingdom",
        "phylum",
        "taxon_class",
        "taxon_order",
        "family",
        "genus",
        "species",
        "scientific_name",
        "vernacular_names",
        "source_queries",
        "profile_text",
    ]
    return " ".join(normalize_text(row.get(column, "")) for column in columns)


def has_any(text: str, terms: list[str]) -> bool:
    return any(normalize_text(term) in text for term in terms)


def safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
