"""Sube artefactos generados a Hugging Face Datasets."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "selenamir/biodiversity-finder-artifacts"
REPO_TYPE = "dataset"

FILES_TO_UPLOAD = [
    # Datos completos para práctica
    ("data/raw/gbif_occurrences_raw.parquet", "raw/gbif_occurrences_raw.parquet"),
    ("data/interim/gbif_occurrences_clean.parquet", "interim/gbif_occurrences_clean.parquet"),
    ("data/interim/climate_reference.csv", "interim/climate_reference.csv"),
    ("data/interim/vernacular_names.csv", "interim/vernacular_names.csv"),
    ("data/interim/conservation_status.csv", "interim/conservation_status.csv"),
    ("data/processed/gbif_occurrences_features.parquet", "processed/gbif_occurrences_features.parquet"),
    ("data/processed/species_encyclopedia.parquet", "processed/species_encyclopedia.parquet"),
    ("data/processed/species_occurrence_points.parquet", "processed/species_occurrence_points.parquet"),
    ("data/processed/species_encyclopedia_light.parquet", "processed/species_encyclopedia_light.parquet"),
    ("data/processed/species_occurrence_points_light.parquet", "processed/species_occurrence_points_light.parquet"),

    # Modelo y reportes
    ("models/taxon_classifier.joblib", "models/taxon_classifier.joblib"),
    ("reports/metrics.json", "reports/metrics.json"),
    ("reports/classification_report.csv", "reports/classification_report.csv"),

    # Muestras pequeñas inspeccionables
    ("reports/data_samples/raw_sample.csv", "samples/raw_sample.csv"),
    ("reports/data_samples/clean_sample.csv", "samples/clean_sample.csv"),
    ("reports/data_samples/features_sample.csv", "samples/features_sample.csv"),
    ("reports/data_samples/encyclopedia_sample.csv", "samples/encyclopedia_sample.csv"),
    ("reports/data_samples/climate_reference_sample.csv", "samples/climate_reference_sample.csv"),
    ("reports/data_samples/vernacular_names_sample.csv", "samples/vernacular_names_sample.csv"),
    ("reports/data_samples/conservation_status_sample.csv", "samples/conservation_status_sample.csv"),
    ("reports/data_samples/occurrence_points_sample.csv", "samples/occurrence_points_sample.csv"),
    ("reports/data_samples/offline_encyclopedia_sample.csv", "samples/offline_encyclopedia_sample.csv"),
    ("reports/data_samples/data_dictionary.csv", "samples/data_dictionary.csv"),
    ("reports/data_samples/pipeline_summary.json", "samples/pipeline_summary.json"),

    # EDA y documentación automática
    ("reports/eda/eda_class_distribution.csv", "eda/eda_class_distribution.csv"),
    ("reports/eda/eda_conservation_summary.csv", "eda/eda_conservation_summary.csv"),
    ("reports/eda/eda_habitat_summary.csv", "eda/eda_habitat_summary.csv"),
    ("reports/eda/eda_findings.json", "eda/eda_findings.json"),
]


def main() -> None:
    """Sube archivos generados por el pipeline al repositorio de Hugging Face."""
    api = HfApi()

    for local_path, repo_path in FILES_TO_UPLOAD:
        full_local_path = PROJECT_ROOT / local_path

        if not full_local_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {full_local_path}")

        api.upload_file(
            path_or_fileobj=str(full_local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

        file_size_mb = full_local_path.stat().st_size / (1024 * 1024)
        print(
            f"Subido: {local_path} → {repo_path} ({file_size_mb:.2f} MB)",
            flush=True,
        )


if __name__ == "__main__":
    main()
