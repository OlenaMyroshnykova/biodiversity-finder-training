"""Generación de tags de búsqueda para consultas tipo vibe.

Crea columnas:
- color_tag
- habitat_tag
- size_tag
- tags_de_busqueda

Estas columnas preparan la app para convertir lenguaje natural en máscaras
booleanas con `df.loc`.
"""

from __future__ import annotations

import pandas as pd


def add_search_tags_to_encyclopedia(encyclopedia_df: pd.DataFrame) -> pd.DataFrame:
    """Añade tags de color, hábitat y tamaño."""
    tagged_df = encyclopedia_df.copy()

    tagged_df["color_tag"] = tagged_df.apply(infer_color_tag, axis=1)
    tagged_df["habitat_tag"] = tagged_df.apply(infer_habitat_tag, axis=1)
    tagged_df["size_tag"] = tagged_df.apply(infer_size_tag, axis=1)

    tagged_df["tags_de_busqueda"] = (
        tagged_df["color_tag"].fillna("").astype(str)
        + " "
        + tagged_df["habitat_tag"].fillna("").astype(str)
        + " "
        + tagged_df["size_tag"].fillna("").astype(str)
        + " "
        + tagged_df.get("vernacular_names", "").fillna("").astype(str)
        + " "
        + tagged_df.get("taxon_class", "").fillna("").astype(str)
        + " "
        + tagged_df.get("family", "").fillna("").astype(str)
        + " "
        + tagged_df.get("source_queries", "").fillna("").astype(str)
    )

    tagged_df["tags_de_busqueda"] = (
        tagged_df["tags_de_busqueda"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    if "search_document" in tagged_df.columns:
        tagged_df["search_document"] = (
            tagged_df["search_document"].fillna("").astype(str)
            + " "
            + tagged_df["tags_de_busqueda"].fillna("").astype(str)
        )

    return tagged_df


def infer_color_tag(row: pd.Series) -> str:
    """Infiere color aproximado para búsqueda natural."""
    text = build_row_text(row)

    if any(term in text for term in ["flamingo", "phoenicopter", "rosa", "pink"]):
        return "pink rosa"

    if any(term in text for term in ["frog", "rana", "amphibia", "hylidae"]):
        return "green brown verde marron"

    if any(term in text for term in ["butterfly", "mariposa", "lepidoptera", "papilionidae"]):
        return "colorful bright multicolor"

    if any(term in text for term in ["felidae", "panthera", "puma", "lion", "leon"]):
        return "brown golden spotted marron dorado"

    if any(term in text for term in ["polar", "ursus maritimus", "ice", "hielo"]):
        return "white blanco"

    if any(term in text for term in ["plant", "planta", "flower", "flor", "magnoliopsida"]):
        return "green colorful verde colorido"

    if any(term in text for term in ["reptilia", "crocodylia", "crocodil", "caiman", "serpentes", "iguana"]):
        return "green brown grey verde marron gris"

    if any(term in text for term in ["actinopterygii", "pisces", "teleostei"]):
        return "colorful silver blue colorido plateado azul"

    if any(term in text for term in ["chondrichthyes", "selachimorpha"]):
        return "grey blue gris azul"

    if any(term in text for term in ["arachnida", "araneae", "scorpion"]):
        return "brown black marron negro"

    if any(term in text for term in ["fungi", "basidiomycota", "ascomycota"]):
        return "brown white red marron blanco rojo"

    if any(term in text for term in ["aves", "bird", "ave", "accipitridae"]):
        return "brown white colorful marron blanco colorido"

    if any(term in text for term in ["mammalia", "mammal"]):
        return "brown grey marron gris"

    return "unknown"


def infer_habitat_tag(row: pd.Series) -> str:
    """Infiere hábitat aproximado para búsqueda natural."""
    text = build_row_text(row)

    if any(term in text for term in ["polar", "ice", "hielo", "ursus maritimus"]):
        return "polar ice arctic hielo"

    if any(term in text for term in ["frog", "rana", "amphibia", "hylidae"]):
        return "wetland river water rio humedo"

    if any(term in text for term in ["flamingo", "phoenicopter", "anatidae", "laridae"]):
        return "wetland lake coast agua humedal"

    if any(term in text for term in ["felidae", "panthera", "puma"]):
        return "forest savanna mountain bosque sabana montaña"

    if any(term in text for term in ["lepidoptera", "butterfly", "mariposa"]):
        return "meadow forest garden pradera bosque jardin"

    if any(term in text for term in ["plant", "planta", "flower", "flor", "magnoliopsida"]):
        return "terrestrial meadow garden pradera jardin"

    if any(term in text for term in ["desert", "desierto", "arid"]):
        return "desert arid desierto"

    if any(term in text for term in ["crocodylia", "crocodil", "caiman"]):
        return "wetland river tropical rio humedo tropical"

    if any(term in text for term in ["reptilia", "serpentes", "iguana", "lacertilia"]):
        return "forest desert savanna tropical bosque desierto sabana"

    if any(term in text for term in ["actinopterygii", "teleostei", "pisces"]):
        return "ocean river lake aquatic oceano rio lago acuatico"

    if any(term in text for term in ["chondrichthyes", "selachimorpha"]):
        return "ocean sea marine oceano mar marino"

    if any(term in text for term in ["arachnida", "araneae", "scorpion"]):
        return "terrestrial forest desert bosque desierto"

    if any(term in text for term in ["fungi", "basidiomycota", "ascomycota"]):
        return "forest terrestrial bosque terrestre"

    if any(term in text for term in ["accipitridae"]):
        return "mountain forest savanna montaña bosque sabana"

    if any(term in text for term in ["mammalia", "mammal"]):
        return "terrestrial forest savanna bosque sabana"

    return "unknown"


def infer_size_tag(row: pd.Series) -> str:
    """Infiere tamaño aproximado para búsqueda natural."""
    text = build_row_text(row)

    if any(term in text for term in ["mammalia", "felidae", "ursidae", "panthera", "puma"]):
        return "large grande"

    if any(term in text for term in ["aves", "bird", "ave", "anatidae", "laridae", "accipitridae"]):
        return "medium large mediano grande"

    if any(term in text for term in ["insecta", "lepidoptera", "butterfly", "mariposa"]):
        return "small pequeño pequeno"

    if any(term in text for term in ["amphibia", "frog", "rana"]):
        return "small medium pequeño mediano"

    if any(term in text for term in ["plantae", "plant", "planta", "magnoliopsida"]):
        return "small medium pequeño mediano"

    if any(term in text for term in ["crocodylia", "crocodil", "caiman"]):
        return "large grande"

    if any(term in text for term in ["chondrichthyes", "selachimorpha"]):
        return "large grande"

    if any(term in text for term in ["reptilia", "serpentes"]):
        return "small medium large pequeño mediano grande"

    if any(term in text for term in ["actinopterygii", "teleostei"]):
        return "small medium large pequeño mediano grande"

    if any(term in text for term in ["arachnida", "araneae"]):
        return "small tiny pequeño mini"

    if any(term in text for term in ["scorpion", "scorpiones"]):
        return "small medium pequeño mediano"

    if any(term in text for term in ["fungi", "basidiomycota"]):
        return "small medium pequeño mediano"

    return "unknown"


def build_row_text(row: pd.Series) -> str:
    """Construye texto normalizado de una fila."""
    columns = [
        "scientific_name",
        "vernacular_names",
        "kingdom",
        "taxon_class",
        "taxon_order",
        "family",
        "genus",
        "source_queries",
        "profile_text",
    ]

    values = [
        str(row.get(column, "") or "").lower()
        for column in columns
    ]

    return " ".join(values)
