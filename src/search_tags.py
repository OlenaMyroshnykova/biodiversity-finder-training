"""Generate clean vibe-search tags.

The project requirement says that ``tags_de_busqueda`` must combine color,
habitat and size. Scientific names, vernacular names, taxonomy, Wikidata labels
and source queries stay available for display/fallback search, but they are not
allowed to pollute the main vibe-search tags.
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
    tagged_df["tags_de_busqueda"] = (
        tagged_df["tags_de_busqueda"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Keep a general text document for fallback name/taxonomy search only.
    # Do not append tags_de_busqueda here: structured tags are applied through
    # df.loc in the app before TF-IDF fallback search.
    if "search_document" not in tagged_df.columns:
        tagged_df["search_document"] = build_name_search_document(tagged_df)

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
    return base.str.replace(r"\s+", " ", regex=True).str.strip()


def has_any(text: str, terms: list[str]) -> bool:
    """Return True if any term appears as a safe token/phrase.

    This avoids false positives such as ``lion`` inside ``papilionidae``.
    """

    padded_text = f" {text} "
    for term in terms:
        normalized_term = str(term).lower().strip()
        if not normalized_term:
            continue
        if " " in normalized_term:
            if normalized_term in text:
                return True
        else:
            if re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", padded_text):
                return True
    return False


def infer_color_tag(row: pd.Series) -> str:
    """Infer an educational color tag for natural-language search."""

    text = build_row_text(row)
    if has_any(text, ["flamingo", "phoenicopter", "rosa", "pink"]):
        return "pink rosa"
    if has_any(text, ["frog", "rana", "amphibia", "hylidae"]):
        return "green brown verde marron"
    if has_any(text, ["butterfly", "mariposa", "lepidoptera", "papilionidae"]):
        return "colorful bright multicolor"
    if has_any(text, ["felidae", "panthera", "puma", "lion", "leon"]):
        return "brown golden spotted marron dorado"
    if has_any(text, ["polar", "ursus maritimus", "ice", "hielo"]):
        return "white blanco"
    if has_any(text, ["plant", "planta", "flower", "flor", "magnoliopsida"]):
        return "green colorful verde colorido"
    if has_any(text, ["reptilia", "crocodylia", "crocodil", "caiman", "serpentes", "iguana"]):
        return "green brown grey verde marron gris"
    if has_any(text, ["actinopterygii", "pisces", "teleostei"]):
        return "colorful silver blue colorido plateado azul"
    if has_any(text, ["chondrichthyes", "selachimorpha"]):
        return "grey blue gris azul"
    if has_any(text, ["arachnida", "araneae", "scorpion"]):
        return "brown black marron negro"
    if has_any(text, ["fungi", "basidiomycota", "ascomycota"]):
        return "brown white red marron blanco rojo"
    if has_any(text, ["aves", "bird", "ave", "accipitridae"]):
        return "brown white colorful marron blanco colorido"
    if has_any(text, ["mammalia", "mammal"]):
        return "brown grey marron gris"
    return "unknown"


def infer_habitat_tag(row: pd.Series) -> str:
    """Infer an educational habitat tag for natural-language search."""

    text = build_row_text(row)
    if has_any(text, ["polar", "ice", "hielo", "ursus maritimus"]):
        return "polar ice arctic hielo"
    if has_any(text, ["frog", "rana", "amphibia", "hylidae"]):
        return "wetland river water rio humedo"
    if has_any(text, ["flamingo", "phoenicopter", "anatidae", "laridae"]):
        return "wetland lake coast agua humedal"
    if has_any(text, ["panthera leo", "lion", "leon", "loxodonta", "giraffa", "equus quagga"]):
        return "savanna grassland sabana pradera"
    if has_any(text, ["felidae", "panthera", "puma"]):
        return "forest mountain bosque montaña"
    if has_any(text, ["lepidoptera", "butterfly", "mariposa"]):
        return "meadow forest garden pradera bosque jardin"
    if has_any(text, ["plant", "planta", "flower", "flor", "magnoliopsida"]):
        return "terrestrial meadow garden pradera jardin"
    if has_any(text, ["desert", "desierto", "arid"]):
        return "desert arid desierto"
    if has_any(text, ["crocodylia", "crocodil", "caiman"]):
        return "wetland river tropical rio humedo tropical"
    if has_any(text, ["reptilia", "serpentes", "iguana", "lacertilia"]):
        return "forest desert tropical bosque desierto"
    if has_any(text, ["actinopterygii", "teleostei", "pisces"]):
        return "ocean river lake aquatic oceano rio lago acuatico"
    if has_any(text, ["chondrichthyes", "selachimorpha"]):
        return "ocean sea marine oceano mar marino"
    if has_any(text, ["arachnida", "araneae", "scorpion"]):
        return "terrestrial forest desert bosque desierto"
    if has_any(text, ["fungi", "basidiomycota", "ascomycota"]):
        return "forest terrestrial bosque terrestre"
    if has_any(text, ["accipitridae"]):
        return "mountain forest montaña bosque"
    if has_any(text, ["mammalia", "mammal"]):
        return "terrestrial forest bosque"
    return "unknown"


def infer_size_tag(row: pd.Series) -> str:
    """Infer an educational size tag for natural-language search."""

    text = build_row_text(row)
    if has_any(text, ["elephantidae", "loxodonta", "giraffa", "rhinocerotidae", "hippopotamus"]):
        return "large grande"
    if has_any(text, ["panthera", "lion", "leon", "puma", "ursus"]):
        return "large grande"
    if has_any(text, ["mammalia", "felidae"]):
        return "medium large mediano grande"
    if has_any(text, ["aves", "bird", "ave", "anatidae", "laridae", "accipitridae"]):
        return "medium large mediano grande"
    if has_any(text, ["insecta", "lepidoptera", "butterfly", "mariposa"]):
        return "small pequeño pequeno"
    if has_any(text, ["amphibia", "frog", "rana"]):
        return "small medium pequeño mediano"
    if has_any(text, ["plantae", "plant", "planta", "magnoliopsida"]):
        return "small medium pequeño mediano"
    if has_any(text, ["crocodylia", "crocodil", "caiman"]):
        return "large grande"
    if has_any(text, ["chondrichthyes", "selachimorpha"]):
        return "large grande"
    if has_any(text, ["reptilia", "serpentes"]):
        return "medium mediano"
    if has_any(text, ["actinopterygii", "teleostei"]):
        return "small pequeño"
    if has_any(text, ["arachnida", "araneae"]):
        return "small tiny pequeño mini"
    if has_any(text, ["scorpion", "scorpiones"]):
        return "small medium pequeño mediano"
    if has_any(text, ["fungi", "basidiomycota"]):
        return "small medium pequeño mediano"
    return "unknown"


def build_row_text(row: pd.Series) -> str:
    """Build normalized row text used only for tag inference."""

    columns = [
        "scientific_name",
        "canonical_scientific_name",
        "vernacular_names",
        "kingdom",
        "taxon_class",
        "taxon_order",
        "family",
        "genus",
        "profile_text",
    ]
    values = [str(row.get(column, "") or "").lower() for column in columns]
    text = " ".join(values)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
