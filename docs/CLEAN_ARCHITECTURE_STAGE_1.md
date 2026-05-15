# Clean Architecture Refactor — Stage 1

This stage reorganizes the training code into domain folders while keeping backward-compatible wrappers in the old `src/*.py` locations.

## Target structure

```text
src/
  acquisition/   # GBIF and external raw data acquisition
  cleaning/      # cleaning and feature preparation
  enrichment/    # climate, images, common names, IUCN
  artifacts/     # contracts, manifests, checkpoints, exports
  modeling/      # model training and evaluation
  reporting/     # EDA and dashboard helpers
  search/        # search tags and search documents
```

## Compatibility rule

Existing imports such as `from src.image_enrichment import ...` still work because `src/image_enrichment.py` now re-exports from `src.enrichment.images`.

## Moves applied

- `src/data_acquisition.py` → `src/acquisition/gbif.py` — moved + wrapper created
- `src/data_cleaning.py` → `src/cleaning/data_cleaning.py` — moved + wrapper created
- `src/feature_engineering.py` → `src/cleaning/features.py` — moved + wrapper created
- `src/data_snapshots.py` → `src/artifacts/snapshots.py` — moved + wrapper created
- `src/climate_enrichment.py` → `src/enrichment/climate.py` — moved + wrapper created
- `src/conservation_status.py` → `src/enrichment/conservation.py` — moved + wrapper created
- `src/image_enrichment.py` → `src/enrichment/images.py` — moved + wrapper created
- `src/media_validation.py` → `src/enrichment/media_validation.py` — moved + wrapper created
- `src/vernacular_names.py` → `src/enrichment/vernacular_names.py` — moved + wrapper created
- `src/iucn_candidates.py` → `src/enrichment/iucn_candidates.py` — moved + wrapper created
- `src/iucn_incremental_after_clean.py` → `src/enrichment/iucn_incremental_after_clean.py` — moved + wrapper created
- `src/artifact_contract.py` → `src/artifacts/contract.py` — moved + wrapper created
- `src/artifact_manifest.py` → `src/artifacts/manifest.py` — moved + wrapper created
- `src/hf_checkpoints.py` → `src/artifacts/hf_checkpoints.py` — moved + wrapper created
- `src/offline_export.py` → `src/artifacts/offline_export.py` — moved + wrapper created
- `src/model_training.py` → `src/modeling/training.py` — moved + wrapper created
- `src/eda_reporting.py` → `src/reporting/eda.py` — moved + wrapper created
- `src/dashboard_charts.py` → `src/reporting/dashboard_charts.py` — moved + wrapper created
- `src/dashboard_data_sources.py` → `src/reporting/dashboard_data_sources.py` — moved + wrapper created
- `src/dashboard_loader.py` → `src/reporting/dashboard_loader.py` — moved + wrapper created
- `src/dashboard_ui.py` → `src/reporting/dashboard_ui.py` — moved + wrapper created
- `src/search_tags.py` → `src/search/tags.py` — moved + wrapper created
- `src/encyclopedia.py` → `src/artifacts/encyclopedia.py` — moved + wrapper created
- `src/occurrence_points.py` → `src/artifacts/occurrence_points.py` — moved + wrapper created

## Next refactor stages

1. Update internal imports gradually from wrapper modules to domain modules.
2. Move pipeline orchestration logic from `scripts/` into `src/pipeline/`, leaving scripts as thin CLI wrappers.
3. Add service-level tests around the new domain modules.
4. Remove compatibility wrappers only after all imports are migrated.
