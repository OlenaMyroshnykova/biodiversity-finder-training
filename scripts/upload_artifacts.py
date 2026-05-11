"""Sube artefactos generados a Hugging Face Datasets."""
from __future__ import annotations
from pathlib import Path
from huggingface_hub import HfApi
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = 'selenamir/biodiversity-finder-artifacts'
REPO_TYPE = 'dataset'
FILES_TO_UPLOAD = [
    ('data/processed/species_encyclopedia.parquet','processed/species_encyclopedia.parquet'),
    ('data/processed/gbif_occurrences_features.parquet','processed/gbif_occurrences_features.parquet'),
    ('models/taxon_classifier.joblib','models/taxon_classifier.joblib'),
    ('reports/metrics.json','reports/metrics.json'),
    ('reports/classification_report.csv','reports/classification_report.csv'),
]
def main() -> None:
    """Sube archivos generados por el pipeline al repositorio de Hugging Face."""
    api = HfApi()
    for local_path, repo_path in FILES_TO_UPLOAD:
        full_local_path = PROJECT_ROOT / local_path
        if not full_local_path.exists(): raise FileNotFoundError(f'No existe el archivo: {full_local_path}')
        api.upload_file(path_or_fileobj=str(full_local_path), path_in_repo=repo_path, repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f'Subido: {local_path} → {repo_path}')
if __name__ == '__main__': main()
