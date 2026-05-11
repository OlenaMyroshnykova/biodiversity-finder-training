# 🐾 Biodiversity Finder Training

Este repositorio contiene el pipeline de entrenamiento para el proyecto **Biodiversity Finder**.

El pipeline descarga datos reales desde GBIF, limpia registros, crea variables, entrena una modelo de Machine Learning y publica los artefactos en Hugging Face Datasets.

## Hugging Face Dataset

Los artefactos se publican en:

```text
selenamir/biodiversity-finder-artifacts
```

## Artefactos generados

```text
processed/species_encyclopedia.parquet
processed/gbif_occurrences_features.parquet
models/taxon_classifier.joblib
reports/metrics.json
reports/classification_report.csv
```

## Ejecución local rápida

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
pytest
python scripts/run_pipeline.py --country ES --max-records 5000 --min-class-records 10
```

## Ejecución local grande

```bash
python scripts/run_pipeline.py --country ES --max-records 50000 --min-class-records 30
```

## Publicar artefactos manualmente

Solo para pruebas locales:

```bash
huggingface-cli login
python scripts/upload_artifacts.py
```

En GitHub Actions se usa el secret `HF_TOKEN`.

## Convención

- Variables, funciones y archivos: inglés.
- Comentarios, docstrings y documentación: español.
- Datos grandes, modelos y reportes no se guardan en GitHub.
