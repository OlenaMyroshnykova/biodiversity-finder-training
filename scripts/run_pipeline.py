"""Ejecuta el pipeline completo del proyecto."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

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
from src.climate_enrichment import add_climate_features  # noqa: E402
from src.conservation_status import add_conservation_status_to_encyclopedia  # noqa: E402
from src.data_acquisition import download_biodiversity_training_dataset  # noqa: E402
from src.data_cleaning import clean_occurrences  # noqa: E402
from src.data_snapshots import save_pipeline_snapshots  # noqa: E402
from src.eda_reporting import generate_eda_reports  # noqa: E402
from src.encyclopedia import build_species_encyclopedia  # noqa: E402
from src.feature_engineering import create_features  # noqa: E402
from src.image_enrichment import add_images_to_encyclopedia  # noqa: E402
from src.model_training import train_model  # noqa: E402
from src.occurrence_points import build_species_occurrence_points  # noqa: E402
from src.offline_export import build_offline_encyclopedia, build_offline_occurrence_points  # noqa: E402
from src.search_tags import add_search_tags_to_encyclopedia  # noqa: E402
from src.vernacular_names import add_vernacular_names_to_encyclopedia  # noqa: E402

CLIMATE_REFERENCE_PATH = PROJECT_ROOT / "data" / "interim" / "climate_reference.csv"
VERNACULAR_NAMES_PATH = PROJECT_ROOT / "data" / "interim" / "vernacular_names.csv"
IMAGE_ENRICHMENT_PATH = PROJECT_ROOT / "data" / "interim" / "image_enrichment.csv"
CONSERVATION_STATUS_PATH = PROJECT_ROOT / "data" / "interim" / "conservation_status.csv"
IUCN_STATUS_CACHE_PATH = PROJECT_ROOT / "data" / "interim" / "iucn_status_cache.csv"
OCCURRENCE_POINTS_PATH = PROJECT_ROOT / "data" / "processed" / "species_occurrence_points.parquet"
OFFLINE_ENCYCLOPEDIA_PATH = PROJECT_ROOT / "data" / "processed" / "species_encyclopedia_light.parquet"
OFFLINE_POINTS_PATH = PROJECT_ROOT / "data" / "processed" / "species_occurrence_points_light.parquet"


def parse_args() -> argparse.Namespace:
    """Lee argumentos de consola."""
    parser = argparse.ArgumentParser(description="Pipeline de Biodiversity Finder Training")
    parser.add_argument(
        "--country",
        default="GLOBAL",
        help="Usa GLOBAL para dataset global o un código de país como ES.",
    )
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--min-class-records", type=int, default=3)
    parser.add_argument(
        "--max-climate-points",
        type=int,
        default=250,
        help="Máximo de coordenadas únicas consultadas a NASA POWER.",
    )
    parser.add_argument(
        "--skip-climate-api",
        action="store_true",
        help="Usa solo estimación climática por latitud, sin llamar a NASA POWER.",
    )
    parser.add_argument(
        "--max-vernacular-species",
        type=int,
        default=5000,
        help="Máximo de especies consultadas para nombres comunes.",
    )
    parser.add_argument(
        "--skip-vernacular-api",
        action="store_true",
        help="Usa solo fallback de nombre científico, sin consultar fuentes externas.",
    )
    parser.add_argument(
        "--skip-wikidata",
        action="store_true",
        help="No consulta Wikidata SPARQL/Search.",
    )
    parser.add_argument(
        "--max-image-species",
        type=int,
        default=5000,
        help="Máximo de especies sin imagen enriquecidas por Wikidata/Wikipedia.",
    )
    parser.add_argument(
        "--max-gbif-image-fallback-species",
        type=int,
        default=1500,
        help="Máximo de especies sin imagen consultadas como fallback en GBIF occurrence media.",
    )
    parser.add_argument(
        "--skip-image-api",
        action="store_true",
        help="No consulta APIs externas para enriquecer imágenes.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Muestreo aleatorio con df.sample() para desarrollo inicial.",
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=42,
        help="Semilla para df.sample().",
    )
    parser.add_argument(
        "--offline-max-species",
        type=int,
        default=5000,
        help="Máximo de especies para la enciclopedia offline ligera.",
    )
    parser.add_argument(
        "--max-iucn-species",
        type=int,
        default=0,
        help="Máximo de especies únicas consultadas en IUCN. 0 = todas.",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta descarga, limpieza, features, clima, modelo y enciclopedia."""
    args = parse_args()
    parameters = {
        "country": args.country,
        "max_records": args.max_records,
        "page_size": args.page_size,
        "min_class_records": args.min_class_records,
        "download_strategy": "global_multi_source_gbif",
        "climate_source": "NASA POWER API + latitude fallback",
        "max_climate_points": args.max_climate_points,
        "vernacular_names_source": "GBIF Species API + Wikidata + scientific_name fallback",
        "max_vernacular_species": args.max_vernacular_species,
        "image_source": "GBIF occurrence media + Wikidata Commons + GBIF fallback",
        "max_image_species": args.max_image_species,
        "max_gbif_image_fallback_species": args.max_gbif_image_fallback_species,
        "iucn_source": "IUCN Red List API v4 + cache + NO_DATA fallback",
        "max_iucn_species": args.max_iucn_species,
        "sample_size": args.sample_size,
        "sample_random_state": args.sample_random_state,
        "offline_max_species": args.offline_max_species,
    }

    print("Pipeline parameters:", parameters, flush=True)

    print("1/13 Descargando dataset global multi-source desde GBIF...", flush=True)
    raw_df = download_biodiversity_training_dataset(
        country=args.country,
        max_records=args.max_records,
        page_size=args.page_size,
    )
    RAW_OCCURRENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(RAW_OCCURRENCES_PATH, index=False)
    print(f" Registros crudos: {len(raw_df):,}", flush=True)
    print(
        f" Source queries: {raw_df['source_query'].nunique() if 'source_query' in raw_df.columns else 0}",
        flush=True,
    )
    print(f" Guardado raw parquet: {RAW_OCCURRENCES_PATH}", flush=True)

    print("2/13 Limpiando datos...", flush=True)
    clean_df = clean_occurrences(raw_df, min_class_records=args.min_class_records)
    CLEAN_OCCURRENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(CLEAN_OCCURRENCES_PATH, index=False)
    print(f" Registros limpios: {len(clean_df):,}", flush=True)
    print(f" Guardado clean parquet: {CLEAN_OCCURRENCES_PATH}", flush=True)

    print("3/13 Creando features taxonómicas...", flush=True)
    features_df = create_features(clean_df)
    print(f" Registros con features base: {len(features_df):,}", flush=True)
    if args.sample_size and args.sample_size > 0 and len(features_df) > args.sample_size:
        print(" Aplicando df.sample() para desarrollo inicial...", flush=True)
        features_df = features_df.sample(
            n=args.sample_size,
            random_state=args.sample_random_state,
        ).reset_index(drop=True)
        print(f" Registros tras sample: {len(features_df):,}", flush=True)

    print("4/13 Enriqueciendo con clima usando df.merge()...", flush=True)
    features_df, climate_reference_df = add_climate_features(
        features_df,
        coordinate_precision=0,
        max_api_points=args.max_climate_points,
        use_api=not args.skip_climate_api,
    )
    CLIMATE_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    climate_reference_df.to_csv(CLIMATE_REFERENCE_PATH, index=False)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(FEATURES_PATH, index=False)
    print(f" Puntos climáticos: {len(climate_reference_df):,}", flush=True)
    print(f" Guardado climate reference: {CLIMATE_REFERENCE_PATH}", flush=True)
    print(f" Guardado features enriquecidas: {FEATURES_PATH}", flush=True)

    print(" [DEMO JOIN] Demostrando pd.concat() y df.merge() con 3 fuentes...", flush=True)
    demonstrate_big_join(features_df, climate_reference_df)

    print("5/13 Entrenando modelo ML...", flush=True)
    metrics = train_model(
        features_df=features_df,
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        report_path=CLASSIFICATION_REPORT_PATH,
    )
    print(f" Accuracy: {metrics['accuracy']:.3f}", flush=True)
    print(f" Guardado modelo: {MODEL_PATH}", flush=True)
    print(f" Guardado metrics: {METRICS_PATH}", flush=True)
    print(f" Guardado classification report: {CLASSIFICATION_REPORT_PATH}", flush=True)

    print("6/13 Construyendo enciclopedia base...", flush=True)
    encyclopedia_df = build_species_encyclopedia(features_df)
    print(f" Especies base en enciclopedia: {len(encyclopedia_df):,}", flush=True)

    print("7/13 Añadiendo nombres comunes usando GBIF + Wikidata + df.merge()...", flush=True)
    encyclopedia_df, vernacular_names_df = add_vernacular_names_to_encyclopedia(
        encyclopedia_df=encyclopedia_df,
        features_df=features_df,
        max_species=args.max_vernacular_species,
        use_api=not args.skip_vernacular_api,
        use_wikidata=not args.skip_wikidata,
    )
    VERNACULAR_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    vernacular_names_df.to_csv(VERNACULAR_NAMES_PATH, index=False)
    print(f" Nombres comunes reunidos: {len(vernacular_names_df):,}", flush=True)
    print(f" Guardado vernacular names: {VERNACULAR_NAMES_PATH}", flush=True)

    print("8/13 Añadiendo imágenes estables al artifact...", flush=True)
    encyclopedia_df, image_enrichment_df = add_images_to_encyclopedia(
        encyclopedia_df=encyclopedia_df,
        max_species=args.max_image_species,
        max_gbif_fallback_species=args.max_gbif_image_fallback_species,
        use_api=not args.skip_image_api,
        use_wikidata=not args.skip_wikidata,
    )
    IMAGE_ENRICHMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image_enrichment_df.to_csv(IMAGE_ENRICHMENT_PATH, index=False)
    image_count = int(encyclopedia_df.get("has_image", pd.Series(dtype=bool)).sum())
    print(f" Especies con imagen: {image_count:,}", flush=True)
    print(f" Guardado image enrichment: {IMAGE_ENRICHMENT_PATH}", flush=True)

    print("9/13 Añadiendo estatus oficial IUCN Red List con pd.merge()...", flush=True)
    encyclopedia_df, conservation_df = add_conservation_status_to_encyclopedia(
        encyclopedia_df,
        cache_path=IUCN_STATUS_CACHE_PATH,
        max_api_species=args.max_iucn_species if args.max_iucn_species > 0 else None,
    )
    CONSERVATION_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    conservation_df.to_csv(CONSERVATION_STATUS_PATH, index=False)
    print(f" Registros de conservación: {len(conservation_df):,}", flush=True)
    official_count = int(conservation_df.get("iucn_is_official", pd.Series(dtype=bool)).sum())
    print(f" Registros oficiales IUCN/GBIF: {official_count:,}", flush=True)
    print(f" Guardado conservation status: {CONSERVATION_STATUS_PATH}", flush=True)
    print(f" Guardado IUCN cache: {IUCN_STATUS_CACHE_PATH}", flush=True)

    print("10/13 Añadiendo tags_de_busqueda y search_document para df.loc...", flush=True)
    encyclopedia_df = add_search_tags_to_encyclopedia(encyclopedia_df)
    ENCYCLOPEDIA_PATH.parent.mkdir(parents=True, exist_ok=True)
    encyclopedia_df.to_parquet(ENCYCLOPEDIA_PATH, index=False)
    print(f" Especies enriquecidas: {len(encyclopedia_df):,}", flush=True)
    print(f" Guardado encyclopedia parquet: {ENCYCLOPEDIA_PATH}", flush=True)

    print("11/13 Construyendo puntos para mapas Folium desde clean_df...", flush=True)
    occurrence_points_df = build_species_occurrence_points(clean_df)
    OCCURRENCE_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    occurrence_points_df.to_parquet(OCCURRENCE_POINTS_PATH, index=False)
    print(f" Puntos de avistamiento: {len(occurrence_points_df):,}", flush=True)
    print(f" Guardado occurrence points: {OCCURRENCE_POINTS_PATH}", flush=True)

    print("12/13 Exportando versión ligera offline...", flush=True)
    offline_encyclopedia_df = build_offline_encyclopedia(
        encyclopedia_df,
        max_species=args.offline_max_species,
    )
    offline_points_df = build_offline_occurrence_points(occurrence_points_df)
    OFFLINE_ENCYCLOPEDIA_PATH.parent.mkdir(parents=True, exist_ok=True)
    OFFLINE_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    offline_encyclopedia_df.to_parquet(OFFLINE_ENCYCLOPEDIA_PATH, index=False)
    offline_points_df.to_parquet(OFFLINE_POINTS_PATH, index=False)
    print(f" Especies offline: {len(offline_encyclopedia_df):,}", flush=True)
    print(f" Puntos offline: {len(offline_points_df):,}", flush=True)

    print("13/13 Guardando muestras, EDA y hallazgos...", flush=True)
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

    eda_files = generate_eda_reports(
        encyclopedia_df=encyclopedia_df,
        features_df=features_df,
        output_dir=REPORTS_DIR / "eda",
    )
    snapshot_files.update(eda_files)
    for name, path in snapshot_files.items():
        print(f" Sample {name}: {path}", flush=True)
    print("Pipeline terminado correctamente.", flush=True)


def demonstrate_big_join(features_df: pd.DataFrame, climate_reference_df: pd.DataFrame) -> None:
    """Small explicit demo for the assignment: pd.concat() + df.merge() with 3 sources."""
    required_cols = ["scientific_name", "taxon_class", "kingdom", "decimal_latitude", "decimal_longitude", "year"]
    missing = [column for column in required_cols if column not in features_df.columns]
    if missing:
        print(f" [DEMO JOIN] Saltado: faltan columnas {missing}", flush=True)
        return

    source_gbif = features_df[required_cols].copy()
    source_climate = climate_reference_df.copy()
    if "source_query" in features_df.columns:
        source_origin = features_df[["scientific_name", "source_query"]].drop_duplicates()
    else:
        source_origin = features_df[["scientific_name"]].drop_duplicates()
        source_origin["source_query"] = "unknown_source"

    classes = source_gbif["taxon_class"].dropna().unique()[:3]
    subsets = [
        source_gbif[source_gbif["taxon_class"] == taxon_class]
        for taxon_class in classes
        if len(source_gbif[source_gbif["taxon_class"] == taxon_class]) > 0
    ]
    if subsets:
        demo_concat = pd.concat(subsets, ignore_index=True)
        print(f" [DEMO JOIN] pd.concat(): {len(subsets)} subsets → {len(demo_concat):,} filas", flush=True)

    coord_cols = [column for column in ["decimal_latitude", "decimal_longitude"] if column in source_climate.columns]
    if len(coord_cols) == 2:
        demo_merge = source_gbif.merge(source_climate, on=coord_cols, how="left")
        demo_merge_2 = demo_merge.merge(source_origin, on="scientific_name", how="left")
        print(f" [DEMO JOIN] df.merge() + fuente3: {len(demo_merge_2):,} filas", flush=True)
        print(" [DEMO JOIN] ✓ pd.concat() + df.merge() con 3 fuentes completado.", flush=True)


if __name__ == "__main__":
    main()
