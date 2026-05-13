"""Generación de tags de búsqueda para consultas tipo vibe.

``tags_de_busqueda`` queda limitado a color, hábitat y tamaño, tal como pide
el entregable. Nombres comunes, taxonomía y textos multilingües se conservan
solo en ``search_document`` para búsqueda secundaria por nombre.
"""
from __future__ import annotations

import unicodedata

import pandas as pd


def add_search_tags_to_encyclopedia(encyclopedia_df: pd.DataFrame) -> pd.DataFrame:
    """Añade color_tag, habitat_tag, size_tag y tags_de_busqueda limpio."""
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
    )
    tagged_df["tags_de_busqueda"] = normalize_tag_series(tagged_df["tags_de_busqueda"])
    tagged_df["search_document"] = build_fallback_search_document(tagged_df)
    return tagged_df


def normalize_tag_series(series: pd.Series) -> pd.Series:
    """Normaliza espacios y minúsculas sin añadir nombres ni taxonomía."""
    return series.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def build_fallback_search_document(df: pd.DataFrame) -> pd.Series:
    """Construye texto para fallback por nombre, separado del vibe-search."""
    if "search_document" in df.columns:
        document = df["search_document"].fillna("").astype(str)
    else:
        document = pd.Series([""] * len(df), index=df.index, dtype=str)

    fallback_columns = [
        "scientific_name",
        "canonical_scientific_name",
        "vernacular_names",
        "kingdom",
        "phylum",
        "taxon_class",
        "taxon_order",
        "family",
        "genus",
        "species",
        "countries",
        "profile_text",
    ]
    for column in fallback_columns:
        if column in df.columns:
            document = document + " " + df[column].fillna("").astype(str)
    return document.str.replace(r"\s+", " ", regex=True).str.strip()


def infer_color_tag(row: pd.Series) -> str:
    """Infiere color educativo por grandes grupos taxonómicos."""
    text = build_row_text(row)
    if has_any(text, ["amphibia", "hylidae"]):
        return "green brown verde marron"
    if has_any(text, ["insecta", "lepidoptera", "papilionidae"]):
        return "colorful bright multicolor"
    if has_any(text, ["felidae", "carnivora"]):
        return "brown grey marron gris"
    if has_any(text, ["plantae", "magnoliopsida", "tracheophyta"]):
        return "green colorful verde colorido"
    if has_any(text, ["reptilia", "crocodylia", "serpentes", "lacertilia"]):
        return "green brown grey verde marron gris"
    if has_any(text, ["actinopterygii", "teleostei"]):
        return "silver blue colorful plateado azul colorido"
    if has_any(text, ["chondrichthyes", "selachimorpha"]):
        return "grey blue gris azul"
    if has_any(text, ["arachnida", "araneae", "scorpiones"]):
        return "brown black marron negro"
    if has_any(text, ["fungi", "basidiomycota", "ascomycota"]):
        return "brown white red marron blanco rojo"
    if has_any(text, ["aves"]):
        return "brown white colorful marron blanco colorido"
    if has_any(text, ["mammalia"]):
        return "brown grey marron gris"
    return "unknown"


def infer_habitat_tag(row: pd.Series) -> str:
    """Infiere hábitat educativo por taxonomía amplia y coordenadas."""
    text = build_row_text(row)
    latitude = safe_float(row.get("avg_latitude", row.get("decimal_latitude", None)))

    if latitude is not None and abs(latitude) >= 60:
        return "polar tundra cold hielo polar"
    if has_any(text, ["amphibia", "hylidae"]):
        return "wetland river water humedal rio agua"
    if has_any(text, ["aves", "anatidae", "laridae"]):
        return "wetland forest coast humedal bosque costa"
    if has_any(text, ["felidae", "carnivora", "mammalia"]):
        return "forest mountain terrestrial bosque montana terrestre"
    if has_any(text, ["insecta", "lepidoptera"]):
        return "meadow forest garden pradera bosque jardin"
    if has_any(text, ["plantae", "magnoliopsida", "tracheophyta"]):
        return "terrestrial meadow garden pradera jardin"
    if has_any(text, ["crocodylia"]):
        return "wetland river tropical humedal rio tropical"
    if has_any(text, ["reptilia", "serpentes", "lacertilia"]):
        return "forest desert tropical bosque desierto tropical"
    if has_any(text, ["actinopterygii", "teleostei"]):
        return "ocean river lake aquatic oceano rio lago acuatico"
    if has_any(text, ["chondrichthyes", "selachimorpha"]):
        return "ocean sea marine oceano mar marino"
    if has_any(text, ["arachnida", "araneae", "scorpiones"]):
        return "terrestrial forest desert bosque desierto terrestre"
    if has_any(text, ["fungi", "basidiomycota", "ascomycota"]):
        return "forest terrestrial bosque terrestre"
    return "unknown"


def infer_size_tag(row: pd.Series) -> str:
    """Infiere tamaño educativo por grupos taxonómicos amplios."""
    text = build_row_text(row)
    if has_any(text, ["mammalia", "crocodylia", "chondrichthyes"]):
        return "medium large mediano grande"
    if has_any(text, ["aves"]):
        return "small medium mediano pequeno pequeño"
    if has_any(text, ["insecta", "lepidoptera", "arachnida", "araneae"]):
        return "small tiny pequeno pequeño mini"
    if has_any(text, ["amphibia"]):
        return "small medium pequeno pequeño mediano"
    if has_any(text, ["plantae", "fungi"]):
        return "small medium pequeno pequeño mediano"
    if has_any(text, ["reptilia", "serpentes", "lacertilia", "actinopterygii"]):
        return "small medium pequeno pequeño mediano"
    return "unknown"


def build_row_text(row: pd.Series) -> str:
    """Construye texto normalizado solo con campos estructurales."""
    columns = [
        "kingdom",
        "phylum",
        "taxon_class",
        "taxon_order",
        "family",
        "genus",
        "species",
        "profile_text",
    ]
    values = [normalize_text(str(row.get(column, "") or "")) for column in columns]
    return " ".join(values)


def normalize_text(value: str) -> str:
    """Normaliza texto a ASCII básico para comparación estable."""
    normalized = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def has_any(text: str, terms: list[str]) -> bool:
    """Comprueba si algún término aparece en el texto normalizado."""
    return any(normalize_text(term) in text for term in terms)


def safe_float(value: object) -> float | None:
    """Convierte a float si es posible."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
