"""Ejecuta el pipeline completo del proyecto."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.config import CLEAN_OCCURRENCES_PATH, DEFAULT_COUNTRY, DEFAULT_MAX_RECORDS, DEFAULT_PAGE_SIZE, ENCYCLOPEDIA_PATH, FEATURES_PATH, MODEL_PATH, RAW_OCCURRENCES_PATH, METRICS_PATH, CLASSIFICATION_REPORT_PATH  # noqa: E402
from src.data_acquisition import download_gbif_occurrences  # noqa: E402
from src.data_cleaning import clean_occurrences  # noqa: E402
from src.encyclopedia import build_species_encyclopedia  # noqa: E402
from src.feature_engineering import create_features  # noqa: E402
from src.model_training import train_model  # noqa: E402

def parse_args() -> argparse.Namespace:
    """Lee argumentos de consola."""
    parser = argparse.ArgumentParser(description='Pipeline de Biodiversity Finder Training')
    parser.add_argument('--country', default=DEFAULT_COUNTRY)
    parser.add_argument('--max-records', type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument('--page-size', type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument('--min-class-records', type=int, default=50)
    return parser.parse_args()

def main() -> None:
    """Ejecuta descarga, limpieza, features, entrenamiento y enciclopedia."""
    args = parse_args()
    print('1/5 Descargando datos reales desde GBIF...')
    raw_df = download_gbif_occurrences(country=args.country, max_records=args.max_records, page_size=args.page_size)
    RAW_OCCURRENCES_PATH.parent.mkdir(parents=True, exist_ok=True); raw_df.to_parquet(RAW_OCCURRENCES_PATH, index=False)
    print(f'   Registros crudos: {len(raw_df):,}')
    print('2/5 Limpiando datos...')
    clean_df = clean_occurrences(raw_df, min_class_records=args.min_class_records)
    CLEAN_OCCURRENCES_PATH.parent.mkdir(parents=True, exist_ok=True); clean_df.to_parquet(CLEAN_OCCURRENCES_PATH, index=False)
    print(f'   Registros limpios: {len(clean_df):,}')
    print('3/5 Creando features...')
    features_df = create_features(clean_df)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True); features_df.to_parquet(FEATURES_PATH, index=False)
    print(f'   Registros con features: {len(features_df):,}')
    print('4/5 Entrenando modelo ML...')
    metrics = train_model(features_df=features_df, model_path=MODEL_PATH, metrics_path=METRICS_PATH, report_path=CLASSIFICATION_REPORT_PATH)
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print('5/5 Construyendo enciclopedia...')
    encyclopedia_df = build_species_encyclopedia(features_df)
    ENCYCLOPEDIA_PATH.parent.mkdir(parents=True, exist_ok=True); encyclopedia_df.to_parquet(ENCYCLOPEDIA_PATH, index=False)
    print(f'   Especies en enciclopedia: {len(encyclopedia_df):,}')
    print('Pipeline terminado correctamente.')
if __name__ == '__main__': main()
