# Clean Architecture Plan

The repository cleanup is complete: generated datasets, reports and models are ignored and Hugging Face is the source of truth for artifacts. The next goal is to make the Python code easier to navigate without breaking workflows.

## Refactor rule

Do not delete old imports immediately. Move implementation files into domain packages and leave compatibility wrappers in the original `src/*.py` files.

This keeps existing imports working:

```python
from src.image_enrichment import add_images_to_encyclopedia
```

while enabling new clean imports:

```python
from src.enrichment.images import add_images_to_encyclopedia
```

## Target packages

```text
src/acquisition/   # GBIF and raw data download
src/cleaning/      # cleaning and feature preparation
src/enrichment/    # climate, images, vernacular names, IUCN
src/artifacts/     # contracts, manifests, checkpoints, exports
src/modeling/      # model training
src/reporting/     # EDA and dashboard helpers
src/search/        # search tags and documents
```

## Stage 1

Run:

```bash
python tools/apply_src_clean_architecture_v40.py
python -m pytest
python -m py_compile scripts/*.py src/*.py src/*/*.py
```

If tests pass, commit the refactor in the `src-clean-architecture` branch.
