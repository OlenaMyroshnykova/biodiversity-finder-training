"""Image enrichment for the Biodiversity Finder encyclopedia.

The Streamlit app should not perform many live image lookups while rendering
cards. This module moves image lookup into the training pipeline and stores only
stable, validated still-image URLs in the parquet artifacts.

Important production rule: ``image_url`` must contain only confirmed still images.
Audio/video/documents or URLs without a verifiable image extension are moved to
technical metadata and are not rendered by the app.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from src.media_validation import (
    classify_media_url,
    is_unverified_media_url,
    is_valid_image_url,
)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
COMMONS_FILEPATH_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
REQUEST_HEADERS = {
    "User-Agent": (
        "BiodiversityFinderTraining/1.0 "
        "(educational project; image enrichment for demo artifacts)"
    )
}

IMAGE_URL_COLUMNS = [
    "image_url",
    "thumbnail_url",
    "media_url",
    "gbif_image_url",
    "wikidata_image_url",
]


@dataclass(frozen=True)
class ImageRecord:
    """One image enrichment result per canonical species name."""

    scientific_name: str
    canonical_scientific_name: str
    image_url: str
    image_source: str
    image_search_name: str
    media_type: str = "image"
    image_validation_status: str = "valid_image_extension"
    unverified_media_url: str = ""


def add_images_to_encyclopedia(
    encyclopedia_df: pd.DataFrame,
    *,
    max_species: int = 2000,
    max_gbif_fallback_species: int = 500,
    use_api: bool = True,
    pause_seconds: float = 0.05,
    timeout_seconds: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add prepared image metadata to an encyclopedia dataframe."""
    if encyclopedia_df.empty:
        empty = build_empty_image_table()
        return ensure_image_columns(encyclopedia_df.copy()), empty

    working_df = encyclopedia_df.copy()
    if "canonical_scientific_name" not in working_df.columns:
        working_df["canonical_scientific_name"] = working_df.get(
            "scientific_name",
            pd.Series([""] * len(working_df), index=working_df.index),
        ).apply(canonicalize_scientific_name)

    working_df["canonical_scientific_name"] = working_df[
        "canonical_scientific_name"
    ].apply(canonicalize_scientific_name)

    working_df = ensure_image_columns(working_df)
    lookup_df = build_image_lookup_table(working_df, max_species=max_species)

    print(
        "[Images] Species selected for prepared image lookup: "
        f"{len(lookup_df):,}",
        flush=True,
    )
    print(
        "[Images] image_url will keep only validated still images; "
        "audio/video/no-extension media is excluded from cards.",
        flush=True,
    )

    image_records: list[ImageRecord] = []
    if use_api and not lookup_df.empty:
        for index, (_, row) in enumerate(lookup_df.iterrows(), start=1):
            scientific_name = str(row.get("scientific_name", "") or "").strip()
            canonical_name = str(row.get("canonical_scientific_name", "") or "").strip()

            existing_url = first_valid_existing_image_url(row)
            if existing_url:
                decision = classify_media_url(existing_url)
                image_records.append(
                    ImageRecord(
                        scientific_name=scientific_name,
                        canonical_scientific_name=canonical_name,
                        image_url=existing_url,
                        image_source="artifact_existing_image",
                        image_search_name=canonical_name or scientific_name,
                        media_type=decision.media_type,
                        image_validation_status=decision.status,
                    )
                )
                continue

            candidates = build_image_search_names(row)
            record = find_image_for_species(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_name,
                candidate_names=candidates,
                allow_gbif_fallback=index <= max_gbif_fallback_species,
                timeout_seconds=timeout_seconds,
            )
            if record is not None:
                image_records.append(record)

            if index % 100 == 0:
                print(
                    f"[Images] {index:,}/{len(lookup_df):,} species checked; "
                    f"valid images found: {len(image_records):,}",
                    flush=True,
                )
            sleep_if_needed(pause_seconds)

    image_df = records_to_image_dataframe(image_records)
    enriched_df = merge_image_metadata(working_df, image_df)
    found_count = int(enriched_df["has_image"].sum()) if "has_image" in enriched_df else 0
    print(
        f"[Images] Valid card images in encyclopedia: {found_count:,} / {len(enriched_df):,}",
        flush=True,
    )
    if not image_df.empty and "image_source" in image_df.columns:
        print(
            "[Images] Sources: "
            f"{image_df['image_source'].value_counts().to_dict()}",
            flush=True,
        )
    return enriched_df, image_df


def build_empty_image_table() -> pd.DataFrame:
    """Return an empty image metadata table with stable columns."""
    return pd.DataFrame(
        columns=[
            "scientific_name",
            "canonical_scientific_name",
            "image_url",
            "image_source",
            "image_search_name",
            "media_type",
            "image_validation_status",
            "unverified_media_url",
        ]
    )


def ensure_image_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure image columns exist and contain only render-safe images."""
    result_df = df.copy()
    for column in ["image_url", "image_source", "unverified_media_url", "media_type", "image_validation_status"]:
        if column not in result_df.columns:
            result_df[column] = ""
        result_df[column] = result_df[column].fillna("").astype(str)

    cleaned_urls: list[str] = []
    unverified_urls: list[str] = []
    media_types: list[str] = []
    statuses: list[str] = []

    for value in result_df["image_url"].tolist():
        decision = classify_media_url(value)
        if decision.is_valid_image:
            cleaned_urls.append(decision.url)
            unverified_urls.append("")
        elif is_unverified_media_url(value):
            cleaned_urls.append("")
            unverified_urls.append(decision.url)
        else:
            cleaned_urls.append("")
            unverified_urls.append("")
        media_types.append(decision.media_type)
        statuses.append(decision.status)

    result_df["image_url"] = cleaned_urls
    result_df["unverified_media_url"] = result_df["unverified_media_url"].where(
        result_df["unverified_media_url"].astype(str).str.len() > 0,
        pd.Series(unverified_urls, index=result_df.index),
    )
    result_df["media_type"] = media_types
    result_df["image_validation_status"] = statuses
    result_df["has_image"] = result_df["image_url"].astype(str).str.len() > 0
    result_df.loc[result_df["has_image"] & (result_df["image_source"] == ""), "image_source"] = (
        "artifact_existing_image"
    )
    return result_df


def build_image_lookup_table(
    encyclopedia_df: pd.DataFrame,
    *,
    max_species: int = 2000,
) -> pd.DataFrame:
    """Select species that should be enriched with images."""
    if encyclopedia_df.empty:
        return pd.DataFrame()

    lookup_df = encyclopedia_df.copy()
    if "observations" in lookup_df.columns:
        lookup_df["_observations_sort"] = pd.to_numeric(
            lookup_df["observations"], errors="coerce"
        ).fillna(0)
        lookup_df = lookup_df.sort_values(
            ["_observations_sort", "scientific_name"], ascending=[False, True]
        )
    else:
        lookup_df = lookup_df.sort_values("scientific_name")

    if max_species and max_species > 0:
        lookup_df = lookup_df.head(max_species)

    keep_columns = [
        column
        for column in [
            "scientific_name",
            "canonical_scientific_name",
            "vernacular_names",
            "image_url",
            "thumbnail_url",
            "media_url",
            "gbif_image_url",
            "wikidata_image_url",
            "observations",
        ]
        if column in lookup_df.columns
    ]
    return lookup_df[keep_columns].drop_duplicates("canonical_scientific_name")


def first_valid_existing_image_url(row: pd.Series) -> str:
    """Return the first valid still-image URL already present in a row."""
    for column in IMAGE_URL_COLUMNS:
        if column not in row.index:
            continue
        value = str(row.get(column, "") or "").strip()
        if is_valid_image_url(value):
            return value
    return ""


def build_image_search_names(row: pd.Series) -> list[str]:
    """Build specific-to-general candidate names for image lookup."""
    names: list[str] = []
    for column in ["scientific_name", "canonical_scientific_name"]:
        value = str(row.get(column, "") or "").strip()
        names.extend(build_scientific_name_candidates(value))

    vernacular_text = str(row.get("vernacular_names", "") or "")
    for raw_name in re.split(r"\s*\|\s*|,", vernacular_text):
        name = raw_name.strip()
        if name and is_probable_public_name(name):
            names.append(name)

    return unique_preserve_order(names)


def build_scientific_name_candidates(value: object) -> list[str]:
    """Return full, author-free and binomial scientific name candidates."""
    text = str(value or "").strip()
    if not text:
        return []
    author_free = remove_authorship(text)
    binomial = to_binomial(author_free)
    return unique_preserve_order([text, author_free, binomial])


def find_image_for_species(
    *,
    scientific_name: str,
    canonical_scientific_name: str,
    candidate_names: list[str],
    allow_gbif_fallback: bool,
    timeout_seconds: int = 15,
) -> ImageRecord | None:
    """Find one representative still image for a species."""
    for candidate_name in candidate_names:
        image_url = fetch_wikidata_p18_image(candidate_name, timeout_seconds=timeout_seconds)
        if image_url:
            decision = classify_media_url(image_url)
            return ImageRecord(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_scientific_name,
                image_url=image_url,
                image_source="Wikidata P18 / Wikimedia Commons",
                image_search_name=candidate_name,
                media_type=decision.media_type,
                image_validation_status=decision.status,
            )

    if allow_gbif_fallback:
        for candidate_name in candidate_names:
            image_url = fetch_gbif_occurrence_image(candidate_name, timeout_seconds=timeout_seconds)
            if image_url:
                decision = classify_media_url(image_url)
                return ImageRecord(
                    scientific_name=scientific_name,
                    canonical_scientific_name=canonical_scientific_name,
                    image_url=image_url,
                    image_source="GBIF occurrence media",
                    image_search_name=candidate_name,
                    media_type=decision.media_type,
                    image_validation_status=decision.status,
                )
    return None


_WIKIDATA_CACHE: dict[str, str] = {}
_GBIF_IMAGE_CACHE: dict[str, str] = {}


def fetch_wikidata_p18_image(search_name: str, *, timeout_seconds: int = 15) -> str:
    """Fetch a Wikimedia Commons image URL from Wikidata P18."""
    clean_name = remove_authorship(search_name)
    if not clean_name:
        return ""

    cache_key = clean_name.lower()
    if cache_key in _WIKIDATA_CACHE:
        return _WIKIDATA_CACHE[cache_key]

    escaped_name = clean_name.replace("\\", "\\\\").replace('"', '\\"')
    query = f'''
    SELECT ?image WHERE {{
      ?item wdt:P225 "{escaped_name}".
      ?item wdt:P18 ?image.
    }} LIMIT 3
    '''
    try:
        response = requests.get(
            WIKIDATA_SPARQL_URL,
            params={"query": query, "format": "json"},
            headers=REQUEST_HEADERS,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        _WIKIDATA_CACHE[cache_key] = ""
        return ""

    bindings = payload.get("results", {}).get("bindings", [])
    for binding in bindings:
        image_value = binding.get("image", {}).get("value", "")
        filename = image_value.rsplit("/", 1)[-1]
        if not filename:
            continue
        image_url = COMMONS_FILEPATH_URL.format(filename=quote(filename.replace(" ", "_")))
        if is_valid_image_url(image_url):
            _WIKIDATA_CACHE[cache_key] = image_url
            return image_url

    _WIKIDATA_CACHE[cache_key] = ""
    return ""


def fetch_gbif_occurrence_image(search_name: str, *, timeout_seconds: int = 15) -> str:
    """Fetch a representative still image from GBIF occurrence media."""
    clean_name = remove_authorship(search_name)
    if not clean_name:
        return ""

    cache_key = clean_name.lower()
    if cache_key in _GBIF_IMAGE_CACHE:
        return _GBIF_IMAGE_CACHE[cache_key]

    try:
        response = requests.get(
            GBIF_OCCURRENCE_URL,
            params={
                "scientificName": clean_name,
                "mediaType": "StillImage",
                "hasCoordinate": "true",
                "limit": 10,
            },
            headers=REQUEST_HEADERS,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        _GBIF_IMAGE_CACHE[cache_key] = ""
        return ""

    for result in payload.get("results", []):
        media_items = result.get("media", [])
        if not isinstance(media_items, list):
            continue
        for media in media_items:
            if not isinstance(media, dict):
                continue
            image_url = str(
                media.get("identifier") or media.get("references") or media.get("source") or ""
            ).strip()
            if is_valid_image_url(image_url):
                _GBIF_IMAGE_CACHE[cache_key] = image_url
                return image_url

    _GBIF_IMAGE_CACHE[cache_key] = ""
    return ""


def records_to_image_dataframe(records: list[ImageRecord]) -> pd.DataFrame:
    """Convert image records to dataframe."""
    if not records:
        return build_empty_image_table()

    image_df = pd.DataFrame([record.__dict__ for record in records])
    image_df = image_df[image_df["image_url"].apply(is_valid_image_url)]
    image_df = image_df.drop_duplicates("canonical_scientific_name")
    return image_df.reset_index(drop=True)


def merge_image_metadata(
    encyclopedia_df: pd.DataFrame,
    image_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge prepared image metadata back into encyclopedia."""
    working_df = ensure_image_columns(encyclopedia_df)
    if image_df.empty:
        return working_df

    merge_columns = [
        column
        for column in [
            "canonical_scientific_name",
            "image_url",
            "image_source",
            "media_type",
            "image_validation_status",
            "unverified_media_url",
        ]
        if column in image_df.columns
    ]
    merge_df = image_df[merge_columns].rename(
        columns={
            "image_url": "_new_image_url",
            "image_source": "_new_image_source",
            "media_type": "_new_media_type",
            "image_validation_status": "_new_image_validation_status",
            "unverified_media_url": "_new_unverified_media_url",
        }
    )

    enriched_df = working_df.merge(
        merge_df,
        on="canonical_scientific_name",
        how="left",
        validate="many_to_one",
    )

    missing_image_mask = enriched_df["image_url"].fillna("").astype(str).str.len() == 0
    has_new_image_mask = enriched_df.get("_new_image_url", pd.Series("", index=enriched_df.index)).fillna("").astype(str).str.len() > 0
    use_new_mask = missing_image_mask & has_new_image_mask

    if "_new_image_url" in enriched_df.columns:
        enriched_df.loc[use_new_mask, "image_url"] = enriched_df.loc[use_new_mask, "_new_image_url"]
    if "_new_image_source" in enriched_df.columns:
        enriched_df.loc[use_new_mask, "image_source"] = enriched_df.loc[use_new_mask, "_new_image_source"]
    if "_new_media_type" in enriched_df.columns:
        enriched_df.loc[use_new_mask, "media_type"] = enriched_df.loc[use_new_mask, "_new_media_type"]
    if "_new_image_validation_status" in enriched_df.columns:
        enriched_df.loc[use_new_mask, "image_validation_status"] = enriched_df.loc[
            use_new_mask, "_new_image_validation_status"
        ]
    if "_new_unverified_media_url" in enriched_df.columns:
        missing_unverified = enriched_df["unverified_media_url"].fillna("").astype(str).str.len() == 0
        has_new_unverified = enriched_df["_new_unverified_media_url"].fillna("").astype(str).str.len() > 0
        enriched_df.loc[missing_unverified & has_new_unverified, "unverified_media_url"] = enriched_df.loc[
            missing_unverified & has_new_unverified, "_new_unverified_media_url"
        ]

    drop_columns = [column for column in enriched_df.columns if column.startswith("_new_")]
    enriched_df = enriched_df.drop(columns=drop_columns)
    enriched_df = ensure_image_columns(enriched_df)
    return enriched_df


def canonicalize_scientific_name(value: object) -> str:
    """Convert a scientific name with authorship to a stable canonical name."""
    return remove_authorship(value)


def remove_authorship(value: object) -> str:
    """Remove author/year fragments while preserving trinomial names."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(?:subsp|ssp|var|forma|f)\.\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{3,4}\b", " ", text)
    text = re.sub(r"[,;:]", " ", text)
    parts = [part for part in re.split(r"\s+", text) if part]
    clean_parts: list[str] = []
    for part in parts:
        if not clean_parts:
            clean_parts.append(part)
            continue
        if re.fullmatch(r"[a-z][a-z-]+", part):
            clean_parts.append(part)
            if len(clean_parts) >= 3:
                break
        elif len(clean_parts) < 2:
            clean_parts.append(part)
        else:
            break
    return " ".join(clean_parts).strip()


def to_binomial(value: object) -> str:
    """Return genus + species when possible."""
    parts = str(value or "").strip().split()
    if len(parts) >= 2:
        return " ".join(parts[:2])
    return str(value or "").strip()


def is_probable_public_name(value: str) -> bool:
    """Filter empty/technical names before image lookup."""
    text = str(value or "").strip()
    if len(text) < 2:
        return False
    if text.startswith("http"):
        return False
    if re.search(r"[\u0400-\u04FF]", text):
        return False
    return True


def unique_preserve_order(values: list[str]) -> list[str]:
    """Return unique non-empty strings preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def sleep_if_needed(seconds: float) -> None:
    """Sleep a little to avoid hammering public APIs."""
    if seconds > 0:
        time.sleep(seconds)
