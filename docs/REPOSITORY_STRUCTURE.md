# Repository structure

This repository contains the **training and artifact-generation side** of Biodiversity Finder.

The application repository should not rebuild data. It should only consume the generated artifacts published to Hugging Face.

## Source of truth

Generated datasets and model outputs are published to Hugging Face:

- `processed/species_encyclopedia.parquet`
- `processed/species_encyclopedia_light.parquet`
- `processed/species_occurrence_points.parquet`
- `processed/species_occurrence_points_light.parquet`
- `interim/iucn_status_cache.csv`
- `reports/metrics.json`
- `reports/classification_report.csv`

The GitHub repository should keep **code, tests, workflows, documentation and empty folder placeholders**, not generated parquet/csv/model artifacts.

## Current clean structure

```text
.github/workflows/
  train_and_publish.yml
  train_full_clean_then_iucn.yml
  refresh_artifacts_production.yml

scripts/
  Thin command-line entry points for workflows and manual maintenance.

src/
  Reusable Python modules:
  - acquisition and cleaning
  - enrichment: climate, images, IUCN, vernacular names
  - artifact validation and manifest
  - feature engineering and model training
  - dashboard helpers

tests/
  Unit and contract tests.

data/
  Local generated artifacts only. Ignored by git except `.gitkeep`.

models/
  Local generated model files only. Ignored by git except `.gitkeep`.

reports/
  Local generated reports only. Ignored by git except `.gitkeep`.
```

## Clean-code direction

The current `src/` structure is intentionally kept stable because all tests are green and workflows are working.

A future refactor can gradually move modules into a package layout:

```text
src/biodiversity_training/
  acquisition/
  cleaning/
  enrichment/
  artifacts/
  modeling/
  reporting/
  pipeline/
```

Do this gradually, with compatibility imports, not as a big-bang rename.

## Safe cleanup commands

```bash
python tools/apply_repo_cleanup_v39.py
python -m pytest
python tools/check_repo_hygiene_v39.py
git status
```

If everything is green:

```bash
git add .gitignore docs/REPOSITORY_STRUCTURE.md tools/apply_repo_cleanup_v39.py tools/check_repo_hygiene_v39.py
git commit -m "Clean repository structure and ignore generated artifacts"
git push
```

## Do not commit

```text
.env
.venv/
__pycache__/
.pytest_cache/
data/raw/*
data/interim/*
data/processed/*
models/*
reports/**/*.csv
reports/**/*.json
reports/**/*.parquet
```

The only tracked files inside `data/`, `models/` and `reports/` should normally be `.gitkeep` placeholders.
