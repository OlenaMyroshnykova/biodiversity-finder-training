# Global GBIF Multi-source Patch

Este parche corrige el problema de que el proyecto estaba demasiado limitado a `country=ES`.

El proyecto del enunciado no es "biodiversidad de España", sino un **Buscador Inteligente de Especies** general.  
Por eso ahora el pipeline puede construir un dataset global y temático.

## Qué cambia

Se reemplazan o añaden estos archivos:

```text
src/data_acquisition.py
src/data_snapshots.py
scripts/run_pipeline.py
scripts/upload_artifacts.py
.github/workflows/train_and_publish.yml
tests/test_data_acquisition.py
tests/test_data_snapshots.py
```

## Nueva lógica de descarga

En vez de descargar solo:

```text
country=ES
```

ahora se descargan varios bloques desde GBIF:

```text
general_global
flamingo_pink_bird
polar_bear
butterflies_lepidoptera
amphibians
raptors_accipitridae
flowering_plants
mammals
```

Luego se unen con:

```python
pd.concat(...)
```

Esto encaja mejor con el enunciado del proyecto, que menciona explícitamente `pd.concat()`.

## Consultas que deberían mejorar

Después de reentrenar con este patch, la biblioteca debería funcionar mejor con ejemplos como:

```text
pajaro rosa
animal polar hielo
insecto mariposa
rana verde rio
ave rapaz montaña
planta flor
```

## Columna nueva

El dataset raw tendrá una columna nueva:

```text
source_query
```

Sirve para saber de qué bloque vino cada registro:

```text
general_global
polar_bear
butterflies_lepidoptera
...
```

## Datos online para practicar

El patch también publica en Hugging Face los datasets completos:

```text
raw/gbif_occurrences_raw.parquet
interim/gbif_occurrences_clean.parquet
processed/gbif_occurrences_features.parquet
processed/species_encyclopedia.parquet
```

y muestras pequeñas:

```text
samples/raw_sample.csv
samples/clean_sample.csv
samples/features_sample.csv
samples/encyclopedia_sample.csv
samples/data_dictionary.csv
samples/pipeline_summary.json
```

## Cómo aplicar

Copia los archivos encima del repositorio:

```text
C:/Users/Olena/Documents/biodiversity-finder-training
```

Luego:

```bash
pytest
git add .
git commit -m "Add global multi-source GBIF dataset"
git push
```

Después ejecuta GitHub Actions con:

```text
country: GLOBAL
max_records: 20000
min_class_records: 20
```

Cuando funcione, puedes probar:

```text
country: GLOBAL
max_records: 50000
min_class_records: 20
```
