"""Build the intelligent species encyclopedia."""
from __future__ import annotations

from typing import Any

import pandas as pd


def build_species_encyclopedia(features_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per species and preserve image URLs from GBIF media."""
    if features_df.empty:
        return pd.DataFrame()

    df = features_df.copy()
    df["scientific_name"] = get_first_existing_column(
        df, ["scientific_name", "scientificName"], default="Unknown species"
    )
    df["taxon_class"] = get_first_existing_column(
        df, ["taxon_class", "class"], default="Unknown class"
    )
    df["taxon_order"] = get_first_existing_column(
        df, ["taxon_order", "order"], default="Unknown order"
    )
    df["country_code"] = get_first_existing_column(
        df, ["country_code", "countryCode", "country"], default="Unknown country"
    )
    df["basis_of_record"] = get_first_existing_column(
        df, ["basis_of_record", "basisOfRecord"], default="Unknown basis"
    )
    df["decimal_latitude"] = get_first_existing_numeric_column(
        df, ["decimal_latitude", "decimalLatitude", "latitude", "lat"]
    )
    df["decimal_longitude"] = get_first_existing_numeric_column(
        df, ["decimal_longitude", "decimalLongitude", "longitude", "lon", "lng"]
    )

    required_columns = {
        "kingdom": "Unknown kingdom",
        "phylum": "Unknown phylum",
        "family": "Unknown family",
        "genus": "Unknown genus",
        "species": "Unknown species",
        "season": "desconocida",
        "source_query": "unknown_source",
    }
    for column, default_value in required_columns.items():
        if column not in df.columns:
            df[column] = default_value
    if "year" not in df.columns:
        df["year"] = pd.NA

    df["image_url_candidate"] = df.apply(extract_image_url_from_occurrence_row, axis=1)
    df["image_source_candidate"] = df["image_url_candidate"].apply(
        lambda value: "GBIF occurrence media" if str(value or "").strip() else "No image"
    )

    encyclopedia_df = (
        df.groupby("scientific_name", as_index=False)
        .agg(
            kingdom=("kingdom", most_common_value),
            phylum=("phylum", most_common_value),
            taxon_class=("taxon_class", most_common_value),
            taxon_order=("taxon_order", most_common_value),
            family=("family", most_common_value),
            genus=("genus", most_common_value),
            species=("species", most_common_value),
            observations=("scientific_name", "count"),
            countries=("country_code", join_unique_values),
            first_year=("year", "min"),
            last_year=("year", "max"),
            avg_latitude=("decimal_latitude", "mean"),
            avg_longitude=("decimal_longitude", "mean"),
            most_common_basis=("basis_of_record", most_common_value),
            most_common_season=("season", most_common_value),
            source_queries=("source_query", join_unique_values),
            image_url=("image_url_candidate", first_valid_url),
            image_source=("image_source_candidate", image_source_for_group),
        )
    )
    encyclopedia_df["has_image"] = encyclopedia_df["image_url"].fillna("").astype(str).str.startswith(("http://", "https://"))
    encyclopedia_df["profile_text"] = encyclopedia_df.apply(build_profile_text, axis=1)
    encyclopedia_df["search_document"] = encyclopedia_df.apply(build_search_document, axis=1)

    return (
        encyclopedia_df.sort_values(["observations", "scientific_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def extract_image_url_from_occurrence_row(row: pd.Series) -> str:
    """Extract a first valid image URL from GBIF occurrence media fields."""
    for column in ["media", "multimedia", "images"]:
        value = row.get(column)
        url = extract_image_url_from_media_value(value)
        if url:
            return url

    for column in ["associatedMedia", "identifier", "references"]:
        value = row.get(column)
        if isinstance(value, str):
            url = first_url_from_text(value)
            if url:
                return url
    return ""


def extract_image_url_from_media_value(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for key in ["identifier", "references", "url", "file", "source"]:
                    url = first_url_from_text(item.get(key))
                    if url:
                        return url
            elif isinstance(item, str):
                url = first_url_from_text(item)
                if url:
                    return url
    if isinstance(value, dict):
        for key in ["identifier", "references", "url", "file", "source"]:
            url = first_url_from_text(value.get(key))
            if url:
                return url
    if isinstance(value, str):
        return first_url_from_text(value)
    return ""


def first_url_from_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for separator in ["|", ";", ",", " "]:
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        for part in parts:
            if is_valid_image_url(part):
                return part
    if is_valid_image_url(text):
        return text
    return ""


def is_valid_image_url(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized.startswith(("http://", "https://")):
        return False
    if any(marker in normalized for marker in ["placeholder", "no_image", "noimage", "missing"]):
        return False
    return True


def get_first_existing_column(df: pd.DataFrame, candidates: list[str], default: str) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def get_first_existing_numeric_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def most_common_value(values: pd.Series) -> str:
    clean_values = values.dropna().astype(str)
    clean_values = clean_values[clean_values.str.strip() != ""]
    if clean_values.empty:
        return "Unknown"
    mode_values = clean_values.mode()
    if mode_values.empty:
        return clean_values.iloc[0]
    return mode_values.iloc[0]


def join_unique_values(values: pd.Series) -> str:
    clean_values = values.dropna().astype(str)
    clean_values = clean_values[clean_values.str.strip() != ""]
    unique_values = sorted(set(clean_values))
    if not unique_values:
        return "Unknown"
    return ", ".join(unique_values)


def first_valid_url(values: pd.Series) -> str:
    for value in values.dropna().astype(str):
        if is_valid_image_url(value):
            return value
    return ""


def image_source_for_group(values: pd.Series) -> str:
    return "GBIF occurrence media" if first_valid_url(values) else "No image"


def build_profile_text(row: pd.Series) -> str:
    return (
        f"{row['scientific_name']} pertenece a la clase {row['taxon_class']}, "
        f"al orden {row['taxon_order']} y a la familia {row['family']}. "
        f"En el dataset tiene {row['observations']} observaciones "
        f"entre {row['first_year']} y {row['last_year']}."
    )


def build_search_document(row: pd.Series) -> str:
    """Build fallback text-search document, without download-query shortcuts."""
    scientific_text = " ".join(
        [
            str(row.get("scientific_name", "")),
            str(row.get("kingdom", "")),
            str(row.get("phylum", "")),
            str(row.get("taxon_class", "")),
            str(row.get("taxon_order", "")),
            str(row.get("family", "")),
            str(row.get("genus", "")),
            str(row.get("species", "")),
            str(row.get("countries", "")),
            str(row.get("most_common_basis", "")),
            str(row.get("profile_text", "")),
        ]
    )
    human_terms = build_human_search_terms(row)
    return f"{scientific_text} {human_terms}".strip()


def build_human_search_terms(row: pd.Series) -> str:
    """Add broad human vocabulary by taxonomic group."""
    combined_text = " ".join(
        [
            str(row.get("kingdom", "")),
            str(row.get("taxon_class", "")),
            str(row.get("taxon_order", "")),
            str(row.get("family", "")),
        ]
    ).lower()
    terms: list[str] = []
    if "animalia" in combined_text:
        terms.extend(["animal", "animales", "fauna", "organismo", "especie"])
    if "plantae" in combined_text or "magnoliopsida" in combined_text:
        terms.extend(["planta", "plantas", "vegetal", "flor", "flores", "plant"])
    if "aves" in combined_text:
        terms.extend(["ave", "aves", "pajaro", "pájaro", "bird", "birds", "plumas"])
    if "mammalia" in combined_text:
        terms.extend(["mamifero", "mamífero", "mamiferos", "mamíferos", "mammal", "mammals"])
    if "insecta" in combined_text:
        terms.extend(["insecto", "insectos", "insect", "insects", "bicho", "bichos"])
    if "lepidoptera" in combined_text:
        terms.extend(["mariposa", "mariposas", "butterfly", "butterflies", "polilla", "moth"])
    if "amphibia" in combined_text:
        terms.extend(["rana", "ranas", "anfibio", "anfibios", "frog", "frogs", "agua"])
    if "accipitridae" in combined_text:
        terms.extend(["ave rapaz", "rapaz", "rapaces", "aguila", "águila", "eagle", "hawk"])
    if "actinopterygii" in combined_text or "chondrichthyes" in combined_text:
        terms.extend(["pez", "peces", "fish", "agua", "acuatico", "acuático"])
    if "arachnida" in combined_text:
        terms.extend(["aracnido", "arácnido", "araña", "spider", "escorpion", "scorpion"])
    return " ".join(sorted(set(terms)))
