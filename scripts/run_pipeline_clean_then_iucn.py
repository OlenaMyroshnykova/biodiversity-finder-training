"""Pipeline largo: limpiar enciclopedia primero, IUCN después.

Este workflow está pensado para construir una enciclopedia grande sin consultar
IUCN sobre datos crudos o duplicados. El estado Red List se consulta al final,
solo sobre candidatos canónicos extraídos de la enciclopedia limpia.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.climate_enrichment import add_climate_features  # noqa: E402
from src.config import (  # noqa: E402
    CLEAN_OCCURRENCES_PATH,
    CLASSIFICATION_REPORT_PATH,
    DEFAULT_MAX_RECORDS,
    DEFAULT_PAGE_SIZE,
    ENCYCLOPEDIA_PATH,
    FEATURES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RAW_OCCURRENCES_PATH,
    REPORTS_DIR,
)
from src.data_acquisition import download_biodiversity_training_dataset  # noqa: E402
from src.data_cleaning import clean_occurrences  # noqa: E402
from src.data_snapshots import save_pipeline_snapshots  # noqa: E402
from src.eda_reporting import generate_eda_reports  # noqa: E402
from src.encyclopedia import build_species_encyclopedia  # noqa: E402
from src.feature_engineering import create_features  # noqa: E402
from src.hf_checkpoints import (  # noqa: E402
    DEFAULT_HF_REPO_ID,
    DEFAULT_IUCN_CACHE_REPO_PATH,
    download_checkpoint_file,
    upload_checkpoint_file,
)
from src.image_enrichment import add_images_to_encyclopedia  # noqa: E402
from src.iucn_candidates import summarize_iucn_candidates  # noqa: E402
from src.iucn_incremental_after_clean import enrich_clean_encyclopedia_with_incremental_iucn  # noqa: E402
from src.model_training import train_model  # noqa: E402
from src.occurrence_points import build_species_occurrence_points  # noqa: E402
from src.offline_export import build_offline_encyclopedia, build_offline_occurrence_points  # noqa: E402
from src.search_tags import add_search_tags_to_encyclopedia  # noqa: E402
from src.vernacular_names import add_vernacular_names_to_encyclopedia  # noqa: E402

CLIMATE_REFERENCE_PATH = PROJECT_ROOT / "data" / "interim" / "climate_reference.csv"
VERNACULAR_NAMES_PATH = PROJECT_ROOT / "data" / "interim" / "vernacular_names.csv"
IMAGE_ENRICHMENT_PATH = PROJECT_ROOT / "data" / "interim" / "image_enrichment.csv"
IUCN_STATUS_CACHE_PATH = PROJECT_ROOT / "data" / "interim" / "iucn_status_cache.csv"
CONSERVATION_STATUS_PATH = PROJECT_ROOT / "data" / "interim" / "conservation_status.csv"
ENCYCLOPEDIA_CLEAN_BEFORE_IUCN_PATH = PROJECT_ROOT / "data" / "interim" / "species_encyclopedia_clean_before_iucn.parquet"
OCCURRENCE_POINTS_PATH = PROJECT_ROOT / "data" / "processed" / "species_occurrence_points.parquet"
OFFLINE_ENCYCLOPEDIA_PATH = PROJECT_ROOT / "data" / "processed" / "species_encyclopedia_light.parquet"
OFFLINE_POINTS_PATH = PROJECT_ROOT / "data" / "processed" / "species_occurrence_points_light.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline clean-first + incremental IUCN")
    parser.add_argument("--country", default="GLOBAL")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--min-class-records", type=int, default=3)
    parser.add_argument("--max-climate-points", type=int, default=250)
    parser.add_argument("--skip-climate-api", action="store_true")
    parser.add_argument("--max-vernacular-species", type=int, default=15000)
    parser.add_argument("--skip-vernacular-api", action="store_true")
    parser.add_argument("--skip-wikidata", action="store_true")
    parser.add_argument("--max-image-species", type=int, default=15000)
    parser.add_argument("--max-gbif-image-fallback-species", type=int, default=3000)
    parser.add_argument("--skip-image-api", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-random-state", type=int, default=42)
    parser.add_argument("--offline-max-species", type=int, default=10000)
    parser.add_argument("--iucn-batch-size", type=int, default=3000)
    parser.add_argument("--iucn-delay-seconds", type=float, default=0.5)
    parser.add_argument("--iucn-checkpoint-every", type=int, default=250)
    parser.add_argument("--iucn-candidate-limit", type=int, default=0)
    parser.add_argument("--recheck-iucn-no-data", action="store_true")
    parser.add_argument("--resume-iucn-cache-from-hf", action="store_true")
    parser.add_argument("--publish-iucn-cache-to-hf", action="store_true")
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-iucn-cache-path", default=DEFAULT_IUCN_CACHE_REPO_PATH)
    return parser.parse_args()


def api_limit(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return int(value)


def maybe_upload_iucn_cache(args: argparse.Namespace, cache_path: Path) -> None:
    if not args.publish_iucn_cache_to_hf:
        return
    upload_checkpoint_file(
        repo_id=args.hf_repo_id,
        repo_path=args.hf_iucn_cache_path,
        local_path=cache_path,
        commit_message="Update IUCN checkpoint cache",
    )


def main() -> None:
    args = parse_args()
    if args.resume_iucn_cache_from_hf:
        download_checkpoint_file(
            repo_id=args.hf_repo_id,
            repo_path=args.hf_iucn_cache_path,
            local_path=IUCN_STATUS_CACHE_PATH,
        )

    parameters = {
        "country": args.country,
        "max_records": args.max_records,
        "page_size": args.page_size,
        "min_class_records": args.min_class_records,
        "iucn_architecture": "clean encyclopedia first, then incremental IUCN cache",
        "iucn_batch_size": args.iucn_batch_size,
        "iucn_delay_seconds": args.iucn_delay_seconds,
        "iucn_checkpoint_every": args.iucn_checkpoint_every,
        "resume_iucn_cache_from_hf": args.resume_iucn_cache_from_hf,
        "publish_iucn_cache_to_hf": args.publish_iucn_cache_to_hf,
        "offline_max_species": args.offline_max_species,
    }
    print("Pipeline parameters:", parameters, flush=True)

    print("1/14 Descargando dataset global multi-source desde GBIF...", flush=True)
    raw_df = download_biodiversity_training_dataset(
        country=args.country,
        max_records=args.max_records,
        page_size=args.page_size,
    )
    RAW_OCCURRENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(RAW_OCCURRENCES_PATH, index=False)
    print(f" Registros crudos: {len(raw_df):,}", flush=True)

    print("2/14 Limpiando datos crudos...", flush=True)
    clean_df = clean_occurrences(raw_df, min_class_records=args.min_class_records)
    CLEAN_OCCURRENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(CLEAN_OCCURRENCES_PATH, index=False)
    print(f" Registros limpios: {len(clean_df):,}", flush=True)

    print("3/14 Creando features taxonómicas...", flush=True)
    features_df = create_features(clean_df)
    if args.sample_size and args.sample_size > 0 and len(features_df) > args.sample_size:
        print(" Aplicando df.sample() para desarrollo inicial...", flush=True)
        features_df = features_df.sample(n=args.sample_size, random_state=args.sample_random_state).reset_index(drop=True)

    print("4/14 Enriqueciendo clima antes de la enciclopedia...", flush=True)
    features_df, climate_reference_df = add_climate_features(
        features_df,
        coordinate_precision=0,
        max_api_points=args.max_climate_points,
        use_api=not args.skip_climate_api,
    )
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(FEATURES_PATH, index=False)
    CLIMATE_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    climate_reference_df.to_csv(CLIMATE_REFERENCE_PATH, index=False)

    print("5/14 Construyendo enciclopedia base limpia...", flush=True)
    encyclopedia_df = build_species_encyclopedia(features_df)
    print(f" Especies base: {len(encyclopedia_df):,}", flush=True)

    print("6/14 Añadiendo nombres comunes...", flush=True)
    encyclopedia_df, vernacular_names_df = add_vernacular_names_to_encyclopedia(
        encyclopedia_df=encyclopedia_df,
        features_df=features_df,
        max_species=api_limit(args.max_vernacular_species),
        use_api=not args.skip_vernacular_api,
        use_wikidata=not args.skip_wikidata,
    )
    VERNACULAR_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    vernacular_names_df.to_csv(VERNACULAR_NAMES_PATH, index=False)

    print("7/14 Añadiendo imágenes antes de seleccionar candidatos IUCN...", flush=True)
    encyclopedia_df, image_enrichment_df = add_images_to_encyclopedia(
        encyclopedia_df=encyclopedia_df,
        max_species=api_limit(args.max_image_species),
        max_gbif_fallback_species=api_limit(args.max_gbif_image_fallback_species),
        use_api=not args.skip_image_api,
        use_wikidata=not args.skip_wikidata,
    )
    IMAGE_ENRICHMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image_enrichment_df.to_csv(IMAGE_ENRICHMENT_PATH, index=False)

    print("8/14 Guardando enciclopedia limpia antes de IUCN...", flush=True)
    encyclopedia_df = encyclopedia_df.drop_duplicates("canonical_scientific_name" if "canonical_scientific_name" in encyclopedia_df.columns else "scientific_name")
    ENCYCLOPEDIA_CLEAN_BEFORE_IUCN_PATH.parent.mkdir(parents=True, exist_ok=True)
    encyclopedia_df.to_parquet(ENCYCLOPEDIA_CLEAN_BEFORE_IUCN_PATH, index=False)
    candidate_summary = summarize_iucn_candidates(encyclopedia_df)
    print(f" Candidatos IUCN limpios: {candidate_summary.unique_candidates:,}", flush=True)
    print(f" Candidatos con imagen: {candidate_summary.with_images:,}", flush=True)

    print("9/14 IUCN Red List incremental sobre candidatos limpios...", flush=True)
    encyclopedia_df, conservation_df, iucn_summary = enrich_clean_encyclopedia_with_incremental_iucn(
        encyclopedia_df,
        cache_path=IUCN_STATUS_CACHE_PATH,
        batch_size=args.iucn_batch_size,
        request_delay_seconds=args.iucn_delay_seconds,
        checkpoint_every=args.iucn_checkpoint_every,
        candidate_limit=api_limit(args.iucn_candidate_limit),
        recheck_no_data=args.recheck_iucn_no_data,
        checkpoint_callback=(lambda path: maybe_upload_iucn_cache(args, path)),
    )
    CONSERVATION_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    conservation_df.to_csv(CONSERVATION_STATUS_PATH, index=False)
    maybe_upload_iucn_cache(args, IUCN_STATUS_CACHE_PATH)
    print(f" IUCN requested this run: {iucn_summary.requested_this_run:,}", flush=True)
    print(f" IUCN cache rows after: {iucn_summary.cache_rows_after:,}", flush=True)

    print("10/14 Añadiendo tags_de_busqueda y search_document al artifact final...", flush=True)
    encyclopedia_df = add_search_tags_to_encyclopedia(encyclopedia_df)
    ENCYCLOPEDIA_PATH.parent.mkdir(parents=True, exist_ok=True)
    encyclopedia_df.to_parquet(ENCYCLOPEDIA_PATH, index=False)
    print(f" Enciclopedia final: {len(encyclopedia_df):,}", flush=True)

    print("11/14 Entrenando modelo después del enrichment/cleaning final...", flush=True)
    metrics = train_model(
        features_df=features_df,
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        report_path=CLASSIFICATION_REPORT_PATH,
    )
    print(f" Accuracy: {metrics['accuracy']:.3f}", flush=True)

    print("12/14 Construyendo puntos Folium...", flush=True)
    occurrence_points_df = build_species_occurrence_points(clean_df)
    OCCURRENCE_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    occurrence_points_df.to_parquet(OCCURRENCE_POINTS_PATH, index=False)

    print("13/14 Exportando versión ligera offline...", flush=True)
    offline_encyclopedia_df = build_offline_encyclopedia(encyclopedia_df, max_species=args.offline_max_species)
    offline_points_df = build_offline_occurrence_points(occurrence_points_df)
    OFFLINE_ENCYCLOPEDIA_PATH.parent.mkdir(parents=True, exist_ok=True)
    OFFLINE_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    offline_encyclopedia_df.to_parquet(OFFLINE_ENCYCLOPEDIA_PATH, index=False)
    offline_points_df.to_parquet(OFFLINE_POINTS_PATH, index=False)

    print("14/14 Guardando muestras y EDA...", flush=True)
    snapshot_files = save_pipeline_snapshots(
        raw_df=raw_df,
        clean_df=clean_df,
        features_df=features_df,
        encyclopedia_df=encyclopedia_df,
        output_dir=REPORTS_DIR / "data_samples",
        parameters=parameters,
        sample_size=100,
    )
    extra_sample_files = {
        "climate_reference_sample": (climate_reference_df, "climate_reference_sample.csv"),
        "vernacular_names_sample": (vernacular_names_df, "vernacular_names_sample.csv"),
        "image_enrichment_sample": (image_enrichment_df, "image_enrichment_sample.csv"),
        "conservation_status_sample": (conservation_df, "conservation_status_sample.csv"),
        "occurrence_points_sample": (occurrence_points_df, "occurrence_points_sample.csv"),
        "offline_encyclopedia_sample": (offline_encyclopedia_df, "offline_encyclopedia_sample.csv"),
    }
    for name, (dataframe, filename) in extra_sample_files.items():
        sample_path = REPORTS_DIR / "data_samples" / filename
        dataframe.head(100).to_csv(sample_path, index=False)
        snapshot_files[name] = sample_path
    snapshot_files.update(
        generate_eda_reports(
            encyclopedia_df=encyclopedia_df,
            features_df=features_df,
            output_dir=REPORTS_DIR / "eda",
        )
    )
    for name, path in snapshot_files.items():
        print(f" Sample {name}: {path}", flush=True)
    print("Pipeline clean-first IUCN terminado correctamente.", flush=True)


if __name__ == "__main__":
    main()
