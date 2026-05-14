"""Image enrichment for the Biodiversity Finder encyclopedia.

The Streamlit app should not perform many live image lookups while rendering
cards. This module moves that work into the training pipeline and stores stable
``image_url`` metadata in the parquet artifacts.

Order of sources:
1. Existing image columns already present in the encyclopedia, if any.
2. Wikidata P18 image by scientific name.
3. GBIF occurrence media as a limited fallback.

The module is deliberately taxonomy-neutral: it does not inject hand-picked demo
species. It enriches the top species selected for the light encyclopedia so the
app can stay fast and simple.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

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

BAD_IMAGE_MARKERS = [
    "placeholder",
    "no_image",
    "noimage",
    "no-photo",
    "no_photo",
    "logo",
    "icon",
    "map",
    "range_map",
    "distribution_map",
    "blank",
    "transparent",
]

BAD_IMAGE_EXTENSIONS = [".svg"]


@dataclass(frozen=True)
class ImageRecord:
    """One image enrichment result per canonical species name."""

    scientific_name: str
    canonical_scientific_name: str
    image_url: str
    image_source: str
    image_search_name: str


def add_images_to_encyclopedia(
    encyclopedia_df: pd.DataFrame,
    *,
    max_species: int = 2000,
    max_gbif_fallback_species: int = 500,
    use_api: bool = True,
    pause_seconds: float = 0.05,
    timeout_seconds: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``image_url`` metadata to an encyclopedia dataframe.

    ``max_species`` should be at least the light artifact size. The pipeline
    passes the same value as ``--offline-max-species`` by default, so every
    species shown by the app can receive a prepared image when public sources
    provide one.
    """

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

    # Keep existing valid artifact images, but normalize all image columns first.
    working_df = ensure_image_columns(working_df)

    lookup_df = build_image_lookup_table(working_df, max_species=max_species)
    print(
        "[Images] Species selected for prepared image lookup: "
        f"{len(lookup_df):,}",
        flush=True,
    )
    print(
        "[Images] Runtime app lookups should stay disabled/limited; "
        "this step prepares image_url in artifacts.",
        flush=True,
    )

    image_records: list[ImageRecord] = []
    if use_api and not lookup_df.empty:
        for index, (_, row) in enumerate(lookup_df.iterrows(), start=1):
            scientific_name = str(row.get("scientific_name", "")).strip()
            canonical_name = str(row.get("canonical_scientific_name", "")).strip()
            existing_url = first_valid_existing_image_url(row)
            if existing_url:
                image_records.append(
                    ImageRecord(
                        scientific_name=scientific_name,
                        canonical_scientific_name=canonical_name,
                        image_url=existing_url,
                        image_source="artifact_existing_image",
                        image_search_name=canonical_name or scientific_name,
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
                    f"images found: {len(image_records):,}",
                    flush=True,
                )

            sleep_if_needed(pause_seconds)

    image_df = records_to_image_dataframe(image_records)
    enriched_df = merge_image_metadata(working_df, image_df)

    found_count = int(enriched_df["has_image"].sum()) if "has_image" in enriched_df else 0
    print(
        f"[Images] Prepared images in encyclopedia: {found_count:,} / {len(enriched_df):,}",
        flush=True,
    )
    if not image_df.empty:
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
        ]
    )


def ensure_image_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure image columns exist and have safe values."""

    result_df = df.copy()
    for column in ["image_url", "image_source"]:
        if column not in result_df.columns:
            result_df[column] = ""
        result_df[column] = result_df[column].fillna("").astype(str)

    result_df["image_url"] = result_df["image_url"].apply(
        lambda value: value if is_valid_image_url(value) else ""
    )
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
    """Select the species that should be enriched with images.

    The light artifact is sorted by observations, so we use the same rule here.
    """

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
    """Return a valid image URL already present in the input row."""

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
    """Find one representative image for a species from public sources."""

    for candidate_name in candidate_names:
        image_url = fetch_wikidata_p18_image(candidate_name, timeout_seconds=timeout_seconds)
        if image_url:
            return ImageRecord(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_scientific_name,
                image_url=image_url,
                image_source="Wikidata P18 / Wikimedia Commons",
                image_search_name=candidate_name,
            )

    if allow_gbif_fallback:
        for candidate_name in candidate_names:
            image_url = fetch_gbif_occurrence_image(candidate_name, timeout_seconds=timeout_seconds)
            if image_url:
                return ImageRecord(
                    scientific_name=scientific_name,
                    canonical_scientific_name=canonical_scientific_name,
                    image_url=image_url,
                    image_source="GBIF occurrence media",
                    image_search_name=candidate_name,
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
                media.get("identifier")
                or media.get("references")
                or media.get("source")
                or ""
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

    merge_df = image_df[
        ["canonical_scientific_name", "image_url", "image_source"]
    ].rename(
        columns={
            "image_url": "_new_image_url",
            "image_source": "_new_image_source",
        }
    )
    enriched_df = working_df.merge(
        merge_df,
        on="canonical_scientific_name",
        how="left",
        validate="many_to_one",
    )

    missing_image_mask = enriched_df["image_url"].fillna("").astype(str).str.len() == 0
    has_new_image_mask = enriched_df["_new_image_url"].fillna("").astype(str).str.len() > 0
    use_new_mask = missing_image_mask & has_new_image_mask

    enriched_df.loc[use_new_mask, "image_url"] = enriched_df.loc[
        use_new_mask, "_new_image_url"
    ]
    enriched_df.loc[use_new_mask, "image_source"] = enriched_df.loc[
        use_new_mask, "_new_image_source"
    ]
    enriched_df = enriched_df.drop(columns=["_new_image_url", "_new_image_source"])
    enriched_df["has_image"] = enriched_df["image_url"].fillna("").astype(str).str.len() > 0
    return enriched_df


def canonicalize_scientific_name(value: object) -> str:
    """Convert a scientific name with authorship to a stable canonical name."""

    text = remove_authorship(value)
    return text


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
        # Keep lower-case epithets, stop when authority words start.
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


def is_valid_image_url(value: object) -> bool:
    """Return True for URLs that look like real representative images."""

    url = str(value or "").strip()
    if not url.startswith(("http://", "https://")):
        return False
    lower = url.lower()
    if any(marker in lower for marker in BAD_IMAGE_MARKERS):
        return False
    if any(lower.split("?", 1)[0].endswith(ext) for ext in BAD_IMAGE_EXTENSIONS):
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
