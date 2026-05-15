"""Image enrichment for the final species encyclopedia.

The training repository prepares stable image fields in the artifact. The app should
only consume image_url/image_source and should not need fragile per-card live lookups
for the normal demo flow.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import quote

import pandas as pd
import requests

from src.media_validation import clean_media_url, is_valid_image_url, validate_media_url

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
GBIF_OCCURRENCE_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
REQUEST_HEADERS = {
    "User-Agent": "BiodiversityFinder/1.0 educational project image enrichment"
}


@dataclass(frozen=True)
class ImageRecord:
    scientific_name: str
    image_url: str
    image_source: str
    unverified_media_url: str = ""
    media_type: str = ""
    image_validation_status: str = ""
    has_image: bool = False


def add_images_to_encyclopedia(
    encyclopedia_df: pd.DataFrame,
    *,
    max_species: int = 5000,
    max_gbif_fallback_species: int = 1500,
    use_api: bool = True,
    use_wikidata: bool = True,
    pause_seconds: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add stable, validated image_url/image_source columns to the artifact."""
    if encyclopedia_df.empty:
        empty = encyclopedia_df.copy()
        return empty, pd.DataFrame(
            columns=[
                "scientific_name", "image_url", "image_source", "unverified_media_url",
                "media_type", "image_validation_status", "has_image",
            ]
        )

    result_df = encyclopedia_df.copy()
    if "image_url" not in result_df.columns:
        result_df["image_url"] = ""
    if "image_source" not in result_df.columns:
        result_df["image_source"] = "No image"

    result_df = validate_image_columns(result_df)

    if use_api:
        missing_df = (
            result_df.loc[~result_df["has_image"], ["scientific_name"]]
            .dropna()
            .drop_duplicates()
            .head(max_species)
        )
        image_map: dict[str, tuple[str, str]] = {}
        if use_wikidata:
            for scientific_name in missing_df["scientific_name"].astype(str):
                url = find_wikidata_commons_image(scientific_name)
                if url:
                    image_map[scientific_name] = (url, "Wikidata Commons P18")
                time.sleep(pause_seconds)
        result_df = apply_image_map(result_df, image_map)
        result_df = validate_image_columns(result_df)

        still_missing_df = (
            result_df.loc[~result_df["has_image"], ["scientific_name"]]
            .dropna()
            .drop_duplicates()
            .head(max_gbif_fallback_species)
        )
        gbif_map: dict[str, tuple[str, str]] = {}
        for scientific_name in still_missing_df["scientific_name"].astype(str):
            url = find_gbif_occurrence_image(scientific_name)
            if url:
                gbif_map[scientific_name] = (url, "GBIF occurrence media fallback")
            time.sleep(pause_seconds)
        result_df = apply_image_map(result_df, gbif_map)
        result_df = validate_image_columns(result_df)

    records = [
        ImageRecord(
            scientific_name=str(row.get("scientific_name", "")),
            image_url=clean_url(row.get("image_url", "")),
            image_source=str(row.get("image_source", "No image")),
            unverified_media_url=clean_url(row.get("unverified_media_url", "")),
            media_type=str(row.get("media_type", "")),
            image_validation_status=str(row.get("image_validation_status", "")),
            has_image=bool(row.get("has_image", False)),
        )
        for _, row in result_df.iterrows()
    ]
    return result_df, pd.DataFrame([record.__dict__ for record in records])


def validate_image_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize image diagnostics and keep only renderable images in image_url."""
    result_df = df.copy()
    diagnostics = result_df["image_url"].apply(validate_media_url)
    result_df["image_url"] = diagnostics.apply(lambda item: item.image_url)
    result_df["unverified_media_url"] = diagnostics.apply(lambda item: item.unverified_media_url)
    result_df["media_type"] = diagnostics.apply(lambda item: item.media_type)
    result_df["image_validation_status"] = diagnostics.apply(lambda item: item.image_validation_status)
    result_df["has_image"] = diagnostics.apply(lambda item: item.has_image).astype(bool)
    result_df.loc[~result_df["has_image"], "image_source"] = "No valid image"
    return result_df


def apply_image_map(df: pd.DataFrame, image_map: dict[str, tuple[str, str]]) -> pd.DataFrame:
    if not image_map:
        return df
    result_df = df.copy()
    for scientific_name, (url, source) in image_map.items():
        validation = validate_media_url(url)
        mask = result_df["scientific_name"].astype(str) == scientific_name
        if validation.has_image:
            result_df.loc[mask, "image_url"] = validation.image_url
            result_df.loc[mask, "image_source"] = source
            result_df.loc[mask, "unverified_media_url"] = ""
            result_df.loc[mask, "media_type"] = validation.media_type
            result_df.loc[mask, "image_validation_status"] = validation.image_validation_status
            result_df.loc[mask, "has_image"] = True
        else:
            result_df.loc[mask, "image_url"] = ""
            result_df.loc[mask, "image_source"] = "No valid image"
            result_df.loc[mask, "unverified_media_url"] = validation.unverified_media_url
            result_df.loc[mask, "media_type"] = validation.media_type
            result_df.loc[mask, "image_validation_status"] = validation.image_validation_status
            result_df.loc[mask, "has_image"] = False
    return result_df


def find_wikidata_commons_image(scientific_name: str) -> str:
    qid = find_wikidata_qid(scientific_name)
    if not qid:
        return ""
    try:
        response = requests.get(
            WIKIDATA_ENTITY_URL.format(qid=qid),
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        entity = response.json().get("entities", {}).get(qid, {})
        claims = entity.get("claims", {})
        p18_claims = claims.get("P18", [])
        for claim in p18_claims:
            filename = (
                claim.get("mainsnak", {})
                .get("datavalue", {})
                .get("value", "")
            )
            url = commons_file_url(filename)
            if is_valid_image_url(url):
                return url
    except Exception:
        return ""
    return ""


def find_wikidata_qid(scientific_name: str) -> str:
    try:
        response = requests.get(
            WIKIDATA_SEARCH_URL,
            params={
                "action": "wbsearchentities",
                "search": scientific_name,
                "language": "en",
                "format": "json",
                "limit": 3,
            },
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        for item in response.json().get("search", []):
            label = str(item.get("label", ""))
            description = str(item.get("description", ""))
            if label.lower() == scientific_name.lower() or "species" in description.lower() or "taxon" in description.lower():
                return str(item.get("id", ""))
    except Exception:
        return ""
    return ""


def commons_file_url(filename: str) -> str:
    clean_filename = str(filename or "").strip().replace(" ", "_")
    if not clean_filename:
        return ""
    return "https://commons.wikimedia.org/wiki/Special:FilePath/" + quote(clean_filename)


def find_gbif_occurrence_image(scientific_name: str) -> str:
    try:
        response = requests.get(
            GBIF_OCCURRENCE_SEARCH_URL,
            params={"scientificName": scientific_name, "mediaType": "StillImage", "limit": 10},
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        for result in response.json().get("results", []):
            for media_item in result.get("media", []) or []:
                for key in ["identifier", "references", "source", "file", "url"]:
                    url = clean_url(media_item.get(key, ""))
                    if is_valid_image_url(url):
                        return url
    except Exception:
        return ""
    return ""


def clean_url(value: object) -> str:
    return clean_media_url(value)


def is_valid_image_url_compat(value: object) -> bool:
    return is_valid_image_url(value)
