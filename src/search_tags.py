"""Generate clean educational vibe-search tags.

``tags_de_busqueda`` must contain only color + habitat + size.  Scientific
names, common names, taxonomy labels and source queries are kept for display or
fallback name search, not for the main natural-language vibe filter.

This file also removes the old demo-species logicand no source_query-driven tagging.
"""

from __future__ import annotations

import re

import pandas as pd


def add_search_tags_to_encyclopedia(encyclopedia_df: pd.DataFrame) -> pd.DataFrame:
    """Add color, habitat, size and clean ``tags_de_busqueda`` columns."""
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
    tagged_df["tags_de_busqueda"] = normalize_spaces(tagged_df["tags_de_busqueda"])

    # Fallback name/taxonomy search document.  It may contain names, but it is
    # not used for the primary structured df.loc search.
    name_search_document = build_name_search_document(tagged_df)
    if "search_document" in tagged_df.columns:
        existing_document = tagged_df["search_document"].fillna("").astype(str)
        tagged_df["search_document"] = existing_document + " " + name_search_document
    else:
        tagged_df["search_document"] = name_search_document
    tagged_df["search_document"] = normalize_spaces(tagged_df["search_document"])

    return tagged_df


def build_name_search_document(df: pd.DataFrame) -> pd.Series:
    """Build fallback text search document without vibe tags."""
    columns = [
        "scientific_name",
        "canonical_scientific_name",
        "vernacular_names",
        "common_name_en",
        "common_name_es",
        "kingdom",
        "taxon_class",
        "taxon_order",
        "family",
        "genus",
        "profile_text",
    ]
    base = pd.Series([""] * len(df), index=df.index, dtype=str)
    for column in columns:
        if column in df.columns:
            base = base + " " + df[column].fillna("").astype(str)
    return normalize_spaces(base)


def normalize_spaces(series: pd.Series) -> pd.Series:
    """Lowercase and collapse whitespace in a Series."""
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def has_any(text: str, terms: list[str]) -> bool:
    """Return True if any term appears as a token or phrase."""
    padded_text = f" {text} "
    for term in terms:
        normalized_term = str(term).lower().strip()
        if not normalized_term:
            continue
        if " " in normalized_term:
            if normalized_term in text:
                return True
        elif re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", padded_text):
            return True
    return False


def infer_color_tag(row: pd.Series) -> str:
    """Infer an educational color tag from descriptive words and broad taxonomy."""
    text = build_row_text(row)

    # Only real descriptive terms, not selected demo species.
    if has_any(text, ["pink", "rosa"]):
        return "pink rosa"
    if has_any(text, ["red", "rojo", "scarlet"]):
        return "red rojo"
    if has_any(text, ["yellow", "amarillo", "golden", "dorado"]):
        return "yellow golden amarillo dorado"
    if has_any(text, ["blue", "azul"]):
        return "blue azul"
    if has_any(text, ["green", "verde"]):
        return "green verde"
    if has_any(text, ["white", "blanco"]):
        return "white blanco"
    if has_any(text, ["black", "negro"]):
        return "black negro"
    if has_any(text, ["grey", "gray", "gris"]):
        return "grey gray gris"
    if has_any(text, ["brown", "marron", "marrón"]):
        return "brown marron"

    if has_any(text, ["amphibia", "hylidae"]):
        return "green brown verde marron"
    if has_any(text, ["lepidoptera", "papilionidae", "insecta"]):
        return "colorful bright multicolor"
    if has_any(text, ["felidae", "ursidae", "bovidae", "equidae", "cervidae"]):
        return "brown grey marron gris"
    if has_any(text, ["plantae", "magnoliopsida"]):
        return "green colorful verde colorido"
    if has_any(text, ["reptilia", "crocodylia", "serpentes", "lacertilia"]):
        return "green brown grey verde marron gris"
    if has_any(text, ["actinopterygii", "teleostei"]):
        return "colorful silver blue colorido plateado azul"
    if has_any(text, ["chondrichthyes", "selachimorpha"]):
        return "grey blue gris azul"
    if has_any(text, ["arachnida", "araneae", "scorpion"]):
        return "brown black marron negro"
    if has_any(text, ["fungi", "basidiomycota", "ascomycota"]):
        return "brown white red marron blanco rojo"
    if has_any(text, ["aves"]):
        return "brown white colorful marron blanco colorido"
    if has_any(text, ["mammalia"]):
        return "brown grey marron gris"
    return "unknown"


def infer_habitat_tag(row: pd.Series) -> str:
    """Infer an educational habitat tag from broad taxonomy and climate columns."""
    text = build_row_text(row)

    if has_any(text, ["desert", "desierto", "arid"]):
        return "desert arid desierto"
    if has_any(text, ["wetland", "humedal", "swamp", "pantano"]):
        return "wetland water humedal agua"
    if has_any(text, ["ocean", "marine", "sea", "oceano", "océano", "mar"]):
        return "ocean sea marine oceano mar marino"
    if has_any(text, ["forest", "bosque", "jungle", "selva"]):
        return "forest bosque selva"
    if has_any(text, ["mountain", "montana", "montaña"]):
        return "mountain montaña"
    if has_any(text, ["savanna", "sabana", "grassland", "pradera"]):
        return "savanna grassland sabana pradera"
    if has_any(text, ["polar", "arctic", "ice", "hielo"]):
        return "polar ice arctic hielo"

    # Broad ecological hints by taxonomic group/family. These are not species
    # shortcuts; they provide stable educational tags for df.loc filtering.
    if has_any(text, ["amphibia"]):
        return "wetland river water rio humedo"
    if has_any(text, ["phoenicopteridae", "anatidae", "laridae"]):
        return "wetland lake coast agua humedal"
    if has_any(text, ["elephantidae", "giraffidae", "bovidae", "equidae", "rhinocerotidae", "hippopotamidae"]):
        return "savanna grassland terrestrial sabana pradera"
    if has_any(text, ["felidae", "ursidae"]):
        return "forest mountain terrestrial bosque montaña"
    if has_any(text, ["lepidoptera", "insecta"]):
        return "meadow forest garden pradera bosque jardin"
    if has_any(text, ["plantae", "magnoliopsida"]):
        return "terrestrial meadow garden pradera jardin"
    if has_any(text, ["crocodylia"]):
        return "wetland river tropical rio humedo tropical"
    if has_any(text, ["reptilia", "serpentes", "lacertilia"]):
        return "forest desert tropical bosque desierto"
    if has_any(text, ["actinopterygii"]):
        return "ocean river lake aquatic oceano rio lago acuatico"
    if has_any(text, ["chondrichthyes"]):
        return "ocean sea marine oceano mar marino"
    if has_any(text, ["arachnida"]):
        return "terrestrial forest desert bosque desierto"
    if has_any(text, ["fungi"]):
        return "forest terrestrial bosque terrestre"
    if has_any(text, ["accipitridae"]):
        return "mountain forest montaña bosque"
    if has_any(text, ["mammalia"]):
        return "terrestrial forest grassland bosque pradera"
    if has_any(text, ["aves"]):
        return "terrestrial forest wetland bosque humedal"
    return "unknown"


def infer_size_tag(row: pd.Series) -> str:
    """Infer an educational size tag from broad taxonomy/family."""
    text = build_row_text(row)
    if has_any(text, ["elephantidae", "giraffidae", "rhinocerotidae", "hippopotamidae"]):
        return "large grande"
    if has_any(text, ["bovidae", "equidae", "ursidae"]):
        return "medium large mediano grande"
    if has_any(text, ["felidae"]):
        return "medium large mediano grande"
    if has_any(text, ["mammalia"]):
        return "medium mediano"
    if has_any(text, ["aves", "anatidae", "laridae", "accipitridae"]):
        return "medium mediano"
    if has_any(text, ["insecta", "lepidoptera"]):
        return "small pequeño pequeno"
    if has_any(text, ["amphibia"]):
        return "small medium pequeño mediano"
    if has_any(text, ["plantae", "magnoliopsida"]):
        return "small medium pequeño mediano"
    if has_any(text, ["crocodylia", "chondrichthyes"]):
        return "large grande"
    if has_any(text, ["reptilia", "serpentes"]):
        return "medium mediano"
    if has_any(text, ["actinopterygii"]):
        return "small medium pequeño mediano"
    if has_any(text, ["arachnida", "araneae"]):
        return "small tiny pequeño mini"
    if has_any(text, ["scorpion", "scorpiones", "fungi"]):
        return "small medium pequeño mediano"
    return "unknown"


def build_row_text(row: pd.Series) -> str:
    """Build normalized row text used only for tag inference."""
    columns = [
        "scientific_name",
        "canonical_scientific_name",
        "vernacular_names",
        "common_name_en",
        "common_name_es",
        "kingdom",
        "taxon_class",
        "taxon_order",
        "family",
        "genus",
        "profile_text",
        "climate_zone",
        "habitat_hint",
    ]
    values = [str(row.get(column, "") or "").lower() for column in columns]
    text = " ".join(values)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
