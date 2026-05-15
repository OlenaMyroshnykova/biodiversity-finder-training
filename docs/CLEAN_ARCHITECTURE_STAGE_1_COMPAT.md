# Clean Architecture Stage 1 Compatibility

Stage 1 moves implementation modules into domain packages while keeping the old
`src/*.py` import paths alive.

Legacy modules are module aliases, not star-import wrappers. This matters because
pytest monkeypatching and workflow code may patch names such as
`src.climate_enrichment.fetch_nasa_power_climate`. With module aliases, the patch
is applied to the real implementation module.

Example:

```python
import src.climate_enrichment

# This object is the real src.enrichment.climate module.
```

This keeps the refactor safe while scripts/tests can gradually migrate to the new
paths.
