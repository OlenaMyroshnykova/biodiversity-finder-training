"""Configuración central del proyecto."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
GBIF_OCCURRENCE_URL = 'https://api.gbif.org/v1/occurrence/search'
DEFAULT_COUNTRY = 'ES'
DEFAULT_MAX_RECORDS = 50_000
DEFAULT_PAGE_SIZE = 300
RANDOM_STATE = 42
RAW_OCCURRENCES_PATH = RAW_DIR / 'gbif_occurrences_raw.parquet'
CLEAN_OCCURRENCES_PATH = INTERIM_DIR / 'gbif_occurrences_clean.parquet'
FEATURES_PATH = PROCESSED_DIR / 'gbif_occurrences_features.parquet'
ENCYCLOPEDIA_PATH = PROCESSED_DIR / 'species_encyclopedia.parquet'
MODEL_PATH = MODELS_DIR / 'taxon_classifier.joblib'
METRICS_PATH = REPORTS_DIR / 'metrics.json'
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / 'classification_report.csv'
HF_REPO_ID = 'selenamir/biodiversity-finder-artifacts'
HF_REPO_TYPE = 'dataset'
