"""Official IUCN Red List conservation enrichment for Biodiversity Finder.

Architecture:
- GBIF remains the source for observations, coordinates and taxonomy.
- IUCN Red List is the source for conservation status.
- If a token is missing, an API call fails, or a species is not found, the pipeline
  returns ``NO_DATA``. It never invents ``LC`` as a fallback.
- The resulting conservation table is joined back with ``pd.merge()``.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

IUCN_API_BASE_URL = "https://api.iucnredlist.org/api/v4"
IUCN_API_V3_BASE_URL = "https://apiv3.iucnredlist.org/api/v3"
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
    request_delay_seconds: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add official IUCN status fields to the encyclopedia using ``pd.merge()``."""

    if encyclopedia_df.empty:
        conservation_df = build_empty_conservation_table()
        return encyclopedia_df.copy(), conservation_df

    working_df = encyclopedia_df.copy()
    if "canonical_scientific_name" not in working_df.columns:
        working_df["canonical_scientific_name"] = working_df.get(
            "scientific_name",
            pd.Series([""] * len(working_df), index=working_df.index),
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
    request_delay_seconds: float = 0.2,
) -> pd.DataFrame:
    """Build a one-row-per-species IUCN status table.

    Lookup priority:
    1. Existing official-looking IUCN code in the input row.
    2. Cached official IUCN records.
    3. Live IUCN API lookup if ``IUCN_API_TOKEN`` or ``IUCN_TOKEN`` exists.
    4. Honest ``NO_DATA`` fallback.

    Important: cached ``NO_DATA`` rows do not block a new API lookup when a token is
    available. This prevents old no-token runs from poisoning future runs.
    """

    if encyclopedia_df.empty:
        return build_empty_conservation_table()

    token = get_iucn_token()
    print(f"[IUCN] Token configured: {'yes' if token else 'no'}", flush=True)

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

    print(f"[IUCN] Unique species: {len(unique_names):,}", flush=True)
    print(f"[IUCN] Species allowed for API lookup: {len(lookup_names):,}", flush=True)
    if cache_path:
        print(f"[IUCN] Cache path: {cache_path}", flush=True)

    records: list[ConservationRecord] = []
    new_cache_rows: list[dict[str, Any]] = []
    official_count = 0
    no_data_count = 0
    api_attempts = 0

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
            official_count += 1
            continue

        cached = cache_records.get(canonical_name)
        cached_category = normalize_iucn_category(cached.get("iucn_category")) if cached else ""
        if cached and cached_category and cached_category != "NO_DATA":
            record = record_from_cache_row(cached)
            records.append(record)
            official_count += int(record.iucn_is_official)
            no_data_count += int(not record.iucn_is_official)
            continue

        if token and canonical_name in lookup_names:
            api_attempts += 1
            api_record = fetch_iucn_record_by_scientific_name(canonical_name, token)
            if api_record is not None:
                records.append(api_record)
                new_cache_rows.append(record_to_cache_row(api_record))
                official_count += int(api_record.iucn_is_official)
                if request_delay_seconds > 0:
                    time.sleep(request_delay_seconds)
                continue
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)

        no_data_record = build_no_data_record(canonical_name)
        records.append(no_data_record)
        # Cache NO_DATA only when a token existed and lookup was actually attempted.
        # If no token exists, caching NO_DATA would poison a future official run.
        if token and canonical_name in lookup_names:
            new_cache_rows.append(record_to_cache_row(no_data_record))
        no_data_count += 1

    conservation_df = pd.DataFrame([record.__dict__ for record in records])
    if conservation_df.empty:
        conservation_df = build_empty_conservation_table()
    else:
        conservation_df = conservation_df.drop_duplicates("canonical_scientific_name")

    conservation_df = fill_missing_conservation_columns(conservation_df)
    save_iucn_cache(cache_path, cache_df, pd.DataFrame(new_cache_rows))

    print(f"[IUCN] API attempts: {api_attempts:,}", flush=True)
    print(f"[IUCN] Official statuses found: {official_count:,}", flush=True)
    print(f"[IUCN] NO_DATA rows: {no_data_count:,}", flush=True)
    if token and api_attempts > 0 and official_count == 0:
        print(
            "[IUCN][WARNING] Token is configured and API was attempted, but 0 official statuses were found. "
            "Check endpoint/authentication and sample scientific names.",
            flush=True,
        )

    return conservation_df.reset_index(drop=True)


def get_iucn_token() -> str:
    """Read the IUCN token from supported environment variable names."""

    for env_name in IUCN_TOKEN_ENV_NAMES:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def clean_scientific_name(value: object) -> str:
    """Normalize a scientific name to a binomial when possible."""

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
        value = normalize_iucn_category(row.get(column, ""))
        if value in VALID_IUCN_CODES:
            return value
    return ""


def fetch_iucn_record_by_scientific_name(
    scientific_name: str,
    token: str,
    *,
    timeout_seconds: int = 30,
) -> ConservationRecord | None:
    """Fetch the latest IUCN assessment for a scientific name using API v4.

    Correct v4 endpoint according to IUCN Swagger:
    ``GET /api/v4/taxa/scientific_name`` with query parameters
    ``genus_name`` and ``species_name``. The previous path-style endpoint
    returned HTML 404 and produced all ``NO_DATA`` rows.
    """

    payload = fetch_iucn_payload(scientific_name, token, timeout_seconds=timeout_seconds)
    category = extract_iucn_category_from_payload(payload) if payload else ""
    if not category:
        return None

    source = "IUCN Red List"
    note = "Estado de conservación obtenido desde IUCN Red List API v4."
    return build_official_record(scientific_name, category, source=source, note=note)


def split_scientific_name_for_iucn(scientific_name: str) -> dict[str, str]:
    """Split a scientific name into IUCN v4 query parameters.

    IUCN v4 expects genus/species parameters, not a single URL path with the
    full binomial. Authority text is removed by ``clean_scientific_name`` before
    this function is normally called.
    """

    clean_name = clean_scientific_name(scientific_name)
    parts = [part for part in clean_name.split() if part]
    if len(parts) < 2:
        return {}

    params = {
        "genus_name": parts[0],
        "species_name": parts[1],
    }

    # Infra/subspecies names are optional. We only include a simple third token
    # when it looks like a taxonomic epithet, not an authority/year.
    if len(parts) >= 3 and re.fullmatch(r"[a-z][a-z-]+", parts[2]):
        params["infra_name"] = parts[2]

    return params


def fetch_iucn_payload(
    scientific_name: str,
    token: str,
    *,
    timeout_seconds: int = 30,
) -> Any | None:
    """Call the official IUCN v4 scientific-name endpoint defensively."""

    params = split_scientific_name_for_iucn(scientific_name)
    if not params:
        return None

    url = f"{IUCN_API_BASE_URL}/taxa/scientific_name"
    base_headers = {
        "Accept": "application/json",
        "User-Agent": "biodiversity-finder-training/1.0",
    }

    # Primary auth is Bearer token. The second form is a conservative fallback
    # for API/client differences; both keep the token out of the URL when possible.
    requests_to_try: list[tuple[dict[str, str], dict[str, str]]] = [
        ({**base_headers, "Authorization": f"Bearer {token}"}, params),
        ({**base_headers, "Authorization": token}, params),
    ]

    last_status: int | None = None
    last_preview = ""
    for request_headers, request_params in requests_to_try:
        try:
            response = requests.get(
                url,
                headers=request_headers,
                params=request_params,
                timeout=timeout_seconds,
            )
            last_status = response.status_code
            last_preview = response.text[:200].replace("\n", " ")

            if response.status_code in {401, 403, 404, 429}:
                print(
                    f"[IUCN] lookup {scientific_name}: HTTP {response.status_code} "
                    f"at {response.url} preview={last_preview!r}",
                    flush=True,
                )
                if response.status_code == 429:
                    # Do not hammer the API after a rate-limit response.
                    return None
                continue

            response.raise_for_status()
            payload = response.json()
            if extract_iucn_category_from_payload(payload):
                return payload
            print(
                f"[IUCN] lookup {scientific_name}: HTTP 200 but no category found. "
                f"Payload preview={str(payload)[:300]!r}",
                flush=True,
            )
        except requests.RequestException as exc:
            print(f"[IUCN] lookup {scientific_name}: request error {exc}", flush=True)
            continue
        except ValueError as exc:
            print(
                f"[IUCN] lookup {scientific_name}: non-JSON response "
                f"status={last_status} preview={last_preview!r} error={exc}",
                flush=True,
            )
            continue

    return None


def extract_iucn_category_from_payload(payload: Any) -> str:
    """Extract a Red List category code from IUCN API v4 JSON shapes."""

    if payload is None:
        return ""

    direct = _extract_category_from_known_keys(payload)
    if direct:
        return direct

    if isinstance(payload, dict):
        # IUCN v4 scientific_name endpoint can return assessments in one of
        # these collection fields. Prefer records that look latest/global when
        # such metadata is available; otherwise fall back to the first valid code.
        for collection_key in ["assessments", "result", "results", "data", "items"]:
            category = _extract_best_category_from_collection(payload.get(collection_key))
            if category:
                return category

    return ""


def _extract_best_category_from_collection(value: Any) -> str:
    """Prefer latest/global assessment category from a list of assessments."""

    if isinstance(value, dict):
        nested_category = _extract_category_from_known_keys(value)
        if nested_category:
            return nested_category
        for nested_key in ["data", "results", "items", "assessments", "result"]:
            category = _extract_best_category_from_collection(value.get(nested_key))
            if category:
                return category
        return ""

    if not isinstance(value, list):
        return _extract_category_from_known_keys(value)

    candidates: list[tuple[int, str]] = []
    for item in value:
        category = _extract_category_from_known_keys(item)
        if not category:
            category = _extract_best_category_from_collection(item)
        if not category:
            continue
        score = 0
        if isinstance(item, dict):
            text = str(item).lower()
            if "latest" in item and bool(item.get("latest")):
                score += 20
            if "is_latest" in item and bool(item.get("is_latest")):
                score += 20
            if "current" in item and bool(item.get("current")):
                score += 10
            if "global" in text:
                score += 5
            year = safe_int(item.get("year") or item.get("assessment_year"))
            score += min(max(year - 1900, 0), 200)
        candidates.append((score, category))

    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_category_from_collection(value: Any) -> str:
    """Extract category from a list/dict collection."""

    if isinstance(value, dict):
        for nested_key in ["data", "results", "items", "assessments", "result"]:
            category = _extract_category_from_collection(value.get(nested_key))
            if category:
                return category
        return _extract_category_from_known_keys(value)

    if isinstance(value, list):
        for item in value:
            category = _extract_category_from_known_keys(item)
            if category:
                return category
            category = _extract_category_from_collection(item)
            if category:
                return category

    return ""


def _extract_category_from_known_keys(value: Any) -> str:
    """Extract category from common IUCN key names."""

    if not isinstance(value, dict):
        return normalize_iucn_category(value)

    for key in [
        "red_list_category",
        "redlist_category",
        "redListCategory",
        "category",
        "code",
    ]:
        category = normalize_iucn_category(value.get(key))
        if category:
            return category

    # Some v4 payloads keep the object under taxon/latest_assessment fields.
    for key in ["latest_assessment", "assessment", "taxon"]:
        category = _extract_category_from_known_keys(value.get(key))
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
    if not text:
        return ""
    text = text.replace("CRITICALLY ENDANGERED", "CR")
    text = text.replace("ENDANGERED", "EN") if text == "ENDANGERED" else text
    text = text.replace("VULNERABLE", "VU")
    text = text.replace("NEAR THREATENED", "NT")
    text = text.replace("LEAST CONCERN", "LC")
    text = text.replace("DATA DEFICIENT", "DD")
    text = text.replace("EXTINCT IN THE WILD", "EW")
    text = text.replace("EXTINCT", "EX") if text == "EXTINCT" else text

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
            "No se encontró estado oficial IUCN para esta especie en esta ejecución del pipeline. "
            "No se usa una categoría LC inventada como fallback."
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
    is_official = bool(row.get("iucn_is_official", category != "NO_DATA")) and category != "NO_DATA"
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
