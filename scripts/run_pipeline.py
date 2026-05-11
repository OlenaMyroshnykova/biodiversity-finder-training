"""Ejecuta el pipeline completo del proyecto."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CLEAN_OCCURRENCES_PATH,
    CLASSIFICATION_REPORT_PATH,
    DEFAULT_MAX_RECORDS,
    DEFAULT_PAGE_SIZE,
    ENCYCLOPEDIA_PATH,
    FEATURES_PATH,
    MODEL_PATH,
    RAW_OCCURRENCES_PATH,
    REPORTS_DIR,
    METRICS_PATH,
)
from src.data_acquisition import download_biodiversity_training_dataset  # noqa: E402
from src.data_cleaning import clean_occurrences  # noqa: E402
from src.data_snapshots import save_pipeline_snapshots  # noqa: E402
from src.encyclopedia import build_species_encyclopedia  # noqa: E402
from src.feature_engineering import create_features  # noqa: E402
from src.model_training import train_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Lee argumentos de consola."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Biodiversity Finder Training"
    )
    parser.add_argument(
        "--country",
        default="GLOBAL",
        help="Usa GLOBAL para dataset global o un código de país como ES.",
    )
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--min-class-records", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    """Ejecuta descarga, limpieza, features, entrenamiento y enciclopedia."""
    args = parse_args()

    parameters = {
        "country": args.country,
        "max_records": args.max_records,
        "page_size": args.page_size,
        "min_class_records": args.min_class_records,
        "download_strategy": "global_multi_source_gbif",
    }

    print("Pipeline parameters:", parameters, flush=True)

    print("1/6 Descargando dataset global multi-source desde GBIF...", flush=True)
    raw_df = download_biodiversity_training_dataset(
        country=args.country,
        max_records=args.max_records,
        page_size=args.page_size,
    )
    RAW_OCCURRENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(RAW_OCCURRENCES_PATH, index=False)
    print(f"   Registros crudos: {len(raw_df):,}", flush=True)
    print(f"   Source queries: {raw_df['source_query'].nunique() if 'source_query' in raw_df.columns else 0}", flush=True)
    print(f"   Guardado raw parquet: {RAW_OCCURRENCES_PATH}", flush=True)

    print("2/6 Limpiando datos...", flush=True)
    clean_df = clean_occurrences(
        raw_df,
        min_class_records=args.min_class_records,
    )
    CLEAN_OCCURRENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(CLEAN_OCCURRENCES_PATH, index=False)
    print(f"   Registros limpios: {len(clean_df):,}", flush=True)
    print(f"   Guardado clean parquet: {CLEAN_OCCURRENCES_PATH}", flush=True)

    print("3/6 Creando features...", flush=True)
    features_df = create_features(clean_df)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(FEATURES_PATH, index=False)
    print(f"   Registros con features: {len(features_df):,}", flush=True)
    print(f"   Guardado features parquet: {FEATURES_PATH}", flush=True)

    print("4/6 Entrenando modelo ML...", flush=True)
    metrics = train_model(
        features_df=features_df,
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        report_path=CLASSIFICATION_REPORT_PATH,
    )
    print(f"   Accuracy: {metrics['accuracy']:.3f}", flush=True)
    print(f"   Guardado modelo: {MODEL_PATH}", flush=True)
    print(f"   Guardado metrics: {METRICS_PATH}", flush=True)
    print(f"   Guardado classification report: {CLASSIFICATION_REPORT_PATH}", flush=True)

    print("5/6 Construyendo enciclopedia...", flush=True)
    encyclopedia_df = build_species_encyclopedia(features_df)
    ENCYCLOPEDIA_PATH.parent.mkdir(parents=True, exist_ok=True)
    encyclopedia_df.to_parquet(ENCYCLOPEDIA_PATH, index=False)
    print(f"   Especies en enciclopedia: {len(encyclopedia_df):,}", flush=True)
    print(f"   Guardado encyclopedia parquet: {ENCYCLOPEDIA_PATH}", flush=True)

    print("6/6 Guardando muestras inspeccionables...", flush=True)
    snapshot_files = save_pipeline_snapshots(
        raw_df=raw_df,
        clean_df=clean_df,
        features_df=features_df,
        encyclopedia_df=encyclopedia_df,
        output_dir=REPORTS_DIR / "data_samples",
        parameters=parameters,
        sample_size=100,
    )

    for name, path in snapshot_files.items():
        print(f"   Sample {name}: {path}", flush=True)

    print("Pipeline terminado correctamente.", flush=True)


if __name__ == "__main__":
    main()
