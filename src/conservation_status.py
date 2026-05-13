"""Official IUCN Red List conservation enrichment for the species encyclopedia.

This module keeps the project architecture simple and honest:

- GBIF is used for observations and taxonomy.
- IUCN Red List is used for conservation status.
- If IUCN data is not configured or not found, the code returns ``NO_DATA``.
- It never invents ``LC`` as a fallback.

The output is joined back to the encyclopedia with ``pd.merge()``.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

IUCN_API_BASE_URL = "https://api.iucnredlist.org/api/v4"
IUCN_TOKEN_ENV_NAMES = ("IUCN_API_TOKEN", "IUCN_TOKEN")

IUCN_CATEGORY_LABELS = {
    "EX": "Extinct",
    "EW": "Extinct in the Wild",
    "CR": "Critically Endangered",
    "EN": "Endangered",
    "VU": "Vulnerable",
    "NT": "Near Threatened",
    "LC": "Least Concern",
    "DD": "Data Deficient",
    "NE": "Not Evaluated",
    "NO_DATA": "Sin datos IUCN",
}

THREATENED_CATEGORIES = {"VU", "EN", "CR", "EW", "EX"}
WARNING_CATEGORIES = THREATENED_CATEGORIES | {"NT"}
VALID_IUCN_CODES = set(IUCN_CATEGORY_LABELS) - {"NO_DATA"}


@dataclass(frozen=True)
class ConservationRecord:
    """One conservation status record per canonical species name."""

    canonical_scientific_name: str
    iucn_category: str
    iucn_status_label: str
    iucn_source: str
    iucn_is_official: bool
    is_threatened: bool
    conservation_status: str
    conservation_category: str
    conservation_source: str
    conservation_note: str


def add_conservation_status_to_encyclopedia(
    encyclopedia_df: pd.DataFrame,
    *,
    cache_path: str | Path | None = None,
    max_api_species: int | None = None,
    request_delay_seconds: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add official IUCN status fields to the encyclopedia using ``pd.merge()``.

    If the API token is unavailable, the function still returns a conservation
    table with ``NO_DATA`` rows. This keeps the pipeline and tests stable while
    avoiding fake conservation claims.
    """

    if encyclopedia_df.empty:
        conservation_df = build_empty_conservation_table()
        return encyclopedia_df.copy(), conservation_df

    working_df = encyclopedia_df.copy()
    if "canonical_scientific_name" not in working_df.columns:
        working_df["canonical_scientific_name"] = working_df.get(
            "scientific_name", pd.Series([""] * len(working_df), index=working_df.index)
        ).astype(str)

    working_df["canonical_scientific_name"] = working_df[
        "canonical_scientific_name"
    ].apply(clean_scientific_name)

    conservation_df = build_conservation_status_table(
        working_df,
        cache_path=cache_path,
        max_api_species=max_api_species,
        request_delay_seconds=request_delay_seconds,
    )

    enriched_df = working_df.merge(
        conservation_df,
        on="canonical_scientific_name",
        how="left",
        validate="many_to_one",
    )

    enriched_df = fill_missing_conservation_columns(enriched_df)
    return enriched_df, conservation_df


def build_empty_conservation_table() -> pd.DataFrame:
    """Return an empty conservation table with all expected columns."""

    return pd.DataFrame(
        columns=[
            "canonical_scientific_name",
            "iucn_category",
            "iucn_status_label",
            "iucn_source",
            "iucn_is_official",
            "is_threatened",
            "conservation_status",
            "conservation_category",
            "conservation_source",
            "conservation_note",
        ]
    )


def build_conservation_status_table(
    encyclopedia_df: pd.DataFrame,
    *,
    cache_path: str | Path | None = None,
    max_api_species: int | None = None,
    request_delay_seconds: float = 0.05,
) -> pd.DataFrame:
    """Build a one-row-per-species IUCN status table.

    Lookup priority:
    1. Already available IUCN/GBIF status in the input row.
    2. Cached IUCN lookup file.
    3. IUCN API v4, if ``IUCN_API_TOKEN`` or ``IUCN_TOKEN`` exists.
    4. Honest ``NO_DATA`` fallback.
    """

    if encyclopedia_df.empty:
        return build_empty_conservation_table()

    token = get_iucn_token()
    cache_df = load_iucn_cache(cache_path)
    cache_records = {
        str(row["canonical_scientific_name"]): row.to_dict()
        for _, row in cache_df.iterrows()
        if str(row.get("canonical_scientific_name", "")).strip()
    }

    unique_names = (
        encyclopedia_df["canonical_scientific_name"]
        .dropna()
        .astype(str)
        .map(clean_scientific_name)
        .loc[lambda series: series.str.len() > 0]
        .drop_duplicates()
        .tolist()
    )

    if max_api_species is not None and max_api_species > 0:
        lookup_names = set(unique_names[:max_api_species])
    else:
        lookup_names = set(unique_names)

    records: list[ConservationRecord] = []
    new_cache_rows: list[dict[str, Any]] = []

    indexed_rows = (
        encyclopedia_df.drop_duplicates("canonical_scientific_name")
        .set_index("canonical_scientific_name")
        .to_dict("index")
    )

    for canonical_name in unique_names:
        row = indexed_rows.get(canonical_name, {})

        existing_status = extract_existing_iucn_status(row)
        if existing_status:
            record = build_official_record(
                canonical_name,
                existing_status,
                source="GBIF/IUCN Red List",
                note="Estado de conservación proporcionado por los datos de entrada y normalizado en el pipeline.",
            )
            records.append(record)
            new_cache_rows.append(record_to_cache_row(record))
            continue

        cached = cache_records.get(canonical_name)
        if cached:
            records.append(record_from_cache_row(cached))
            continue

        if token and canonical_name in lookup_names:
            api_record = fetch_iucn_record_by_scientific_name(canonical_name, token)
            if api_record is not None:
                records.append(api_record)
                new_cache_rows.append(record_to_cache_row(api_record))
                if request_delay_seconds > 0:
                    time.sleep(request_delay_seconds)
                continue

        no_data_record = build_no_data_record(canonical_name)
        records.append(no_data_record)
        new_cache_rows.append(record_to_cache_row(no_data_record))

    conservation_df = pd.DataFrame([record.__dict__ for record in records])
    if conservation_df.empty:
        conservation_df = build_empty_conservation_table()
    else:
        conservation_df = conservation_df.drop_duplicates("canonical_scientific_name")
        conservation_df = fill_missing_conservation_columns(conservation_df)

    save_iucn_cache(cache_path, cache_df, pd.DataFrame(new_cache_rows))
    return conservation_df.reset_index(drop=True)


def get_iucn_token() -> str:
    """Read the IUCN token from supported environment variable names."""

    for env_name in IUCN_TOKEN_ENV_NAMES:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def clean_scientific_name(value: object) -> str:
    """Normalize a scientific name to a binomial when possible.

    Examples:
    ``Panthera leo (Linnaeus, 1758)`` -> ``Panthera leo``.
    ``Felis chaus Schreber, 1777`` -> ``Felis chaus``.
    """

    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = text.split()
    if len(parts) >= 2 and parts[0][0:1].isupper():
        return f"{parts[0]} {parts[1]}"
    return text


def extract_existing_iucn_status(row: dict[str, Any] | pd.Series) -> str:
    """Extract an official-looking IUCN code already present in a row."""

    for column in [
        "iucn_category",
        "iucn_status",
        "redlist_category",
        "red_list_category",
        "category",
        "conservation_status",
    ]:
        value = str(row.get(column, "") or "").strip().upper()
        if value in VALID_IUCN_CODES:
            return value
    return ""


def fetch_iucn_record_by_scientific_name(
    scientific_name: str,
    token: str,
    *,
    timeout_seconds: int = 20,
) -> ConservationRecord | None:
    """Fetch the latest IUCN assessment for a scientific name via API v4.

    The parser is intentionally defensive because the API response can expose
    the category either at taxon level or inside an ``assessments`` list.
    """

    encoded_name = quote(scientific_name, safe="")
    url = f"{IUCN_API_BASE_URL}/taxa/scientific_name/{encoded_name}"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers, timeout=timeout_seconds)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None
    except ValueError:
        return None

    category = extract_iucn_category_from_payload(payload)
    if not category:
        return None

    return build_official_record(
        scientific_name,
        category,
        source="IUCN Red List",
        note="Estado de conservación obtenido desde IUCN Red List API v4.",
    )


def extract_iucn_category_from_payload(payload: Any) -> str:
    """Extract a Red List category code from several possible JSON shapes."""

    if not isinstance(payload, dict):
        return ""

    direct_category = normalize_iucn_category(payload.get("red_list_category"))
    if direct_category:
        return direct_category

    direct_category = normalize_iucn_category(payload.get("category"))
    if direct_category:
        return direct_category

    assessments = payload.get("assessments")
    if isinstance(assessments, dict):
        assessments = assessments.get("data") or assessments.get("results") or []

    if isinstance(assessments, list):
        for assessment in assessments:
            if not isinstance(assessment, dict):
                continue
            for key in ["red_list_category", "category", "redlist_category"]:
                category = normalize_iucn_category(assessment.get(key))
                if category:
                    return category

    return ""


def normalize_iucn_category(value: Any) -> str:
    """Normalize a raw API category object/string to an IUCN code."""

    if isinstance(value, dict):
        for key in ["code", "category", "red_list_category", "name", "description"]:
            category = normalize_iucn_category(value.get(key))
            if category:
                return category
        return ""

    text = str(value or "").strip().upper()
    if text in VALID_IUCN_CODES:
        return text

    label_to_code = {label.upper(): code for code, label in IUCN_CATEGORY_LABELS.items()}
    return label_to_code.get(text, "")


def build_official_record(
    canonical_name: str,
    category: str,
    *,
    source: str,
    note: str,
) -> ConservationRecord:
    """Create a normalized official IUCN conservation record."""

    category = normalize_iucn_category(category) or "NO_DATA"
    label = IUCN_CATEGORY_LABELS.get(category, category)
    return ConservationRecord(
        canonical_scientific_name=canonical_name,
        iucn_category=category,
        iucn_status_label=label,
        iucn_source=source,
        iucn_is_official=category != "NO_DATA",
        is_threatened=category in THREATENED_CATEGORIES,
        conservation_status=category,
        conservation_category=label,
        conservation_source=source,
        conservation_note=note,
    )


def build_no_data_record(canonical_name: str) -> ConservationRecord:
    """Create an honest fallback record when IUCN data is unavailable."""

    return ConservationRecord(
        canonical_scientific_name=canonical_name,
        iucn_category="NO_DATA",
        iucn_status_label=IUCN_CATEGORY_LABELS["NO_DATA"],
        iucn_source="No IUCN data",
        iucn_is_official=False,
        is_threatened=False,
        conservation_status="NO_DATA",
        conservation_category=IUCN_CATEGORY_LABELS["NO_DATA"],
        conservation_source="No IUCN data",
        conservation_note=(
            "No se encontró estado oficial IUCN para esta especie en esta ejecución "
            "del pipeline. No se usa una categoría LC inventada como fallback."
        ),
    )


def fill_missing_conservation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure conservation columns exist and have safe values."""

    filled_df = df.copy()
    defaults: dict[str, Any] = {
        "iucn_category": "NO_DATA",
        "iucn_status_label": IUCN_CATEGORY_LABELS["NO_DATA"],
        "iucn_source": "No IUCN data",
        "iucn_is_official": False,
        "is_threatened": False,
        "conservation_status": "NO_DATA",
        "conservation_category": IUCN_CATEGORY_LABELS["NO_DATA"],
        "conservation_source": "No IUCN data",
        "conservation_note": "Sin datos IUCN disponibles para esta especie.",
    }

    for column, default_value in defaults.items():
        if column not in filled_df.columns:
            filled_df[column] = default_value
        else:
            filled_df[column] = filled_df[column].fillna(default_value)

    filled_df["iucn_is_official"] = filled_df["iucn_is_official"].astype(bool)
    filled_df["is_threatened"] = filled_df["is_threatened"].astype(bool)
    return filled_df


def load_iucn_cache(cache_path: str | Path | None) -> pd.DataFrame:
    """Load cached IUCN records if a cache path exists."""

    if cache_path is None:
        return build_empty_conservation_table()

    path = Path(cache_path)
    if not path.exists():
        return build_empty_conservation_table()

    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return build_empty_conservation_table()


def save_iucn_cache(
    cache_path: str | Path | None,
    existing_cache_df: pd.DataFrame,
    new_cache_df: pd.DataFrame,
) -> None:
    """Persist IUCN cache rows if a cache path was provided."""

    if cache_path is None or new_cache_df.empty:
        return

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    combined_df = pd.concat([existing_cache_df, new_cache_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates("canonical_scientific_name", keep="last")
    combined_df = fill_missing_conservation_columns(combined_df)

    if path.suffix.lower() == ".parquet":
        combined_df.to_parquet(path, index=False)
    else:
        combined_df.to_csv(path, index=False)


def record_to_cache_row(record: ConservationRecord) -> dict[str, Any]:
    """Serialize a conservation record for cache storage."""

    return record.__dict__.copy()


def record_from_cache_row(row: dict[str, Any]) -> ConservationRecord:
    """Build a ConservationRecord from a cache row."""

    canonical_name = str(row.get("canonical_scientific_name", "")).strip()
    category = normalize_iucn_category(row.get("iucn_category")) or "NO_DATA"
    label = str(row.get("iucn_status_label", "") or IUCN_CATEGORY_LABELS.get(category, category))
    source = str(row.get("iucn_source", "") or row.get("conservation_source", "") or "No IUCN data")
    is_official = bool(row.get("iucn_is_official", category != "NO_DATA"))
    note = str(row.get("conservation_note", "") or "Estado leído desde cache IUCN.")

    return ConservationRecord(
        canonical_scientific_name=canonical_name,
        iucn_category=category,
        iucn_status_label=label,
        iucn_source=source,
        iucn_is_official=is_official,
        is_threatened=category in THREATENED_CATEGORIES,
        conservation_status=category,
        conservation_category=label,
        conservation_source=source,
        conservation_note=note,
    )


def safe_int(value: object) -> int:
    """Backward-compatible helper used by older tests/imports."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
