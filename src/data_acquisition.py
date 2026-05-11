"""Descarga de datos reales desde GBIF."""
from __future__ import annotations
import time
from typing import Any
import pandas as pd
import requests
from src.config import GBIF_OCCURRENCE_URL

GBIF_COLUMNS = [
    'key','scientificName','acceptedScientificName','kingdom','phylum','class','order','family','genus','species',
    'countryCode','decimalLatitude','decimalLongitude','year','month','basisOfRecord','individualCount',
    'coordinateUncertaintyInMeters','issue'
]

def download_gbif_occurrences(country: str, max_records: int, page_size: int, sleep_seconds: float = 0.15) -> pd.DataFrame:
    """Descarga ocurrencias biológicas desde la API pública de GBIF."""
    records: list[dict[str, Any]] = []
    offset = 0
    while len(records) < max_records:
        limit = min(page_size, max_records - len(records))
        params = {'country': country, 'hasCoordinate': 'true', 'hasGeospatialIssue': 'false', 'limit': limit, 'offset': offset}
        response = requests.get(GBIF_OCCURRENCE_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        page_results = payload.get('results', [])
        if not page_results: break
        records.extend(page_results)
        offset += limit
        if payload.get('endOfRecords', False): break
        time.sleep(sleep_seconds)
    return keep_expected_columns(pd.DataFrame(records))

def keep_expected_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo las columnas necesarias para el proyecto."""
    for column in GBIF_COLUMNS:
        if column not in raw_df.columns:
            raw_df[column] = None
    return raw_df[GBIF_COLUMNS].copy()
