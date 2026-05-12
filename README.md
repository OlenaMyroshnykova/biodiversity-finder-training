# 🌿 Biodiversity Finder Training

Pipeline de datos y Machine Learning para el proyecto **Biodiversity Finder / Buscador Inteligente de Especies**.

Este repositorio prepara los datos que después usa la app interactiva:

🔎 App pública:  
https://github.com/OlenaMyroshnykova/biodiversity-finder-app

📊 Dashboard del modelo:  
https://biodiversity-finder-training.streamlit.app/

📦 Artefactos publicados en Hugging Face:  
https://huggingface.co/datasets/selenamir/biodiversity-finder-artifacts

---

## 👩‍💻 Equipo

Proyecto realizado por:

- **Olena Myroshnykova**
- **Nana Torres Gonzalez**
- **Camila Rabelo**
- **Neema Nelly**

---

## 🎯 Objetivo del proyecto

El objetivo es construir una **enciclopedia inteligente interactiva de biodiversidad**.

La idea principal es que una persona pueda escribir una descripción o un nombre común de una especie, por ejemplo:

```text
león
lion
leão
jaguar
mariposa
ave rosa
oso polar
```

y la app pueda buscar especies reales usando datos enriquecidos.

---

## 📌 Relación con el enunciado del proyecto

El enunciado pedía trabajar con bases de datos masivas sobre especies animales o plantas y aplicar:

| Requisito | Dónde está en el proyecto |
|---|---|
| `pd.concat()` | En la descarga multi-source desde GBIF y en la unión de fuentes de nombres comunes |
| `df.merge()` | En la unión de datos taxonómicos con datos climáticos y nombres comunes |
| `df[condición]` | En los filtros de la app interactiva |
| Enciclopedia inteligente | En `species_encyclopedia.parquet`, consumida por la app |

---

## 🧠 Qué hace este repositorio

Este repositorio no es la app final.  
Este repositorio es la parte de **Data Science / Training**.

Hace estas tareas:

1. Descarga datos reales desde **GBIF**.
2. Combina varias consultas de especies con `pd.concat()`.
3. Limpia registros incompletos.
4. Crea variables taxonómicas, geográficas y temporales.
5. Enriquece los datos con clima usando **NASA POWER API**.
6. Une datos climáticos con `df.merge()`.
7. Entrena un modelo de Machine Learning.
8. Construye una enciclopedia de especies.
9. Enriquece la enciclopedia con nombres comunes usando:
   - GBIF Species API
   - Wikidata P1843 / taxon common name
   - Wikidata labels
   - Wikidata aliases
10. Publica artefactos en Hugging Face para que la app pueda usarlos.

---

## 🔄 Evolución de la lógica del proyecto

### 1. Primera idea: buscar solo por datos científicos

Al principio la app buscaba principalmente por:

```text
scientific_name
kingdom
family
genus
taxon_class
```

Esto funcionaba para nombres científicos como:

```text
Panthera leo
Panthera onca
Ursus maritimus
```

Pero no funcionaba bien para nombres humanos como:

```text
león
lion
лев
jaguar
oso polar
```

---

### 2. Segunda idea: añadir reglas manuales en la app

Probamos reglas como:

```text
jaguar → Panthera
leopardo → Panthera
oso → Ursus
mariposa → Lepidoptera
```

Esta idea parecía funcionar rápido, pero era incorrecta como solución general.

¿Por qué?

Porque la app empezaba a convertirse en un diccionario manual de animales.

Eso no escala:

```text
hoy añadimos león
mañana tigre
después zorro
después búho
después miles de especies más
```

Conclusión:

> Los nombres comunes no deben vivir como reglas manuales en la app.  
> Deben vivir en los datos.

---

### 3. Tercera idea: enriquecer los datos con nombres comunes

La solución correcta fue mover esa inteligencia al pipeline de datos.

Ahora el training repo genera una columna:

```text
vernacular_names
```

con nombres comunes en varios idiomas cuando las fuentes externas los proporcionan.

Ejemplo esperado:

```text
Panthera leo
↓
Lion | León | Лев | Leão | Leone
```

Así la app no necesita saber manualmente qué es un león.

La app solo busca en la enciclopedia enriquecida.

---

## 🌍 Fuentes de datos

### GBIF Occurrence API

Fuente principal de observaciones de biodiversidad.

Se usa para obtener registros reales de especies, con datos como:

```text
scientificName
kingdom
phylum
class
order
family
genus
species
countryCode
decimalLatitude
decimalLongitude
eventDate
media
```

---

### GBIF Species API

Se usa para buscar nombres comunes oficiales cuando GBIF los tiene:

```text
vernacularNames
```

---

### Wikidata

Se usa como fuente complementaria de nombres humanos.

El pipeline consulta:

```text
P1843 - taxon common name
labels
aliases
```

Esto mejora la búsqueda multilingüe.

---

### NASA POWER API

Se usa para enriquecer las coordenadas con datos climáticos aproximados:

```text
temperature
precipitation
humidity
```

---

## 🧩 Arquitectura del pipeline

```text
GBIF Occurrence API
        ↓
data/raw/gbif_occurrences_raw.parquet
        ↓
clean_occurrences()
        ↓
data/interim/gbif_occurrences_clean.parquet
        ↓
create_features()
        ↓
add_climate_features()
        ↓
data/processed/gbif_occurrences_features.parquet
        ↓
train_model()
        ↓
models/taxon_classifier.joblib
reports/metrics.json
        ↓
build_species_encyclopedia()
        ↓
add_vernacular_names_to_encyclopedia()
        ↓
data/processed/species_encyclopedia.parquet
        ↓
Hugging Face Dataset
        ↓
Streamlit App
```

---

## 📁 Estructura del repositorio

```text
.github/workflows/
  train_and_publish.yml

data/
  raw/
  interim/
  processed/

models/
  taxon_classifier.joblib

reports/
  metrics.json
  classification_report.csv
  data_samples/

scripts/
  run_pipeline.py
  upload_artifacts.py

src/
  climate_enrichment.py
  config.py
  dashboard_loader.py
  data_acquisition.py
  data_cleaning.py
  data_snapshots.py
  encyclopedia.py
  feature_engineering.py
  model_training.py
  vernacular_names.py

tests/
  test_climate_enrichment.py
  test_data_acquisition.py
  test_data_cleaning.py
  test_feature_engineering.py
  test_model_training.py
  test_vernacular_names.py

model_dashboard.py
pytest.ini
requirements.txt
README.md
```

---

## 🔎 Dónde mirar el código según el enunciado

### `pd.concat()`

Archivo:

```text
src/data_acquisition.py
```

Se usa para unir varias consultas de GBIF en un solo dataset.

También se usa en:

```text
src/vernacular_names.py
```

para combinar nombres comunes desde varias fuentes:

```text
GBIF Species API
Wikidata
scientific_name fallback
```

---

### `df.merge()`

Archivo:

```text
src/climate_enrichment.py
```

Se usa para unir los datos de especies con los datos climáticos.

Archivo:

```text
src/vernacular_names.py
```

Se usa para unir la enciclopedia con los nombres comunes.

---

### `df[condición]`

Está principalmente en la app:

```text
biodiversity-finder-app
```

Se usa para filtrar especies según:

```text
número mínimo de observaciones
clase taxonómica
texto de búsqueda
```

---

### Machine Learning

Archivo:

```text
src/model_training.py
```

Entrena un modelo para clasificar registros por clase taxonómica.

---

### Enciclopedia inteligente

Archivo:

```text
src/encyclopedia.py
```

Construye una fila por especie.

Archivo:

```text
src/vernacular_names.py
```

Añade nombres comunes para que la búsqueda sea más humana y multilingüe.

---

## 📦 Artefactos generados

El pipeline genera:

```text
data/raw/gbif_occurrences_raw.parquet
data/interim/gbif_occurrences_clean.parquet
data/interim/climate_reference.csv
data/interim/vernacular_names.csv
data/processed/gbif_occurrences_features.parquet
data/processed/species_encyclopedia.parquet
models/taxon_classifier.joblib
reports/metrics.json
reports/classification_report.csv
reports/data_samples/*.csv
```

Los artefactos importantes para la app son:

```text
processed/species_encyclopedia.parquet
reports/metrics.json
```

---

## 🚀 Ejecución local

Crear entorno virtual:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar tests:

```bash
pytest
```

Ejecutar pipeline pequeño:

```bash
python scripts/run_pipeline.py \
  --country GLOBAL \
  --max-records 5000 \
  --min-class-records 10 \
  --max-climate-points 50 \
  --max-vernacular-species 300
```

Ejecutar pipeline grande recomendado:

```bash
python scripts/run_pipeline.py \
  --country GLOBAL \
  --max-records 50000 \
  --min-class-records 20 \
  --max-climate-points 250 \
  --max-vernacular-species 3000
```

---

## ⚙️ GitHub Actions

El workflow principal es:

```text
.github/workflows/train_and_publish.yml
```

Se ejecuta manualmente desde:

```text
Actions → Train and publish biodiversity artifacts → Run workflow
```

Parámetros recomendados:

```text
country: GLOBAL
max_records: 50000
min_class_records: 20
max_climate_points: 250
max_vernacular_species: 3000
skip_climate_api: false
skip_vernacular_api: false
skip_wikidata: false
```

---

## 🔐 Secret necesario

Para subir artefactos a Hugging Face, GitHub Actions necesita el secret:

```text
HF_TOKEN
```

---

## 📊 Dashboard del modelo

El dashboard permite revisar:

```text
accuracy
classification report
distribución de clases
muestras de datos
artefactos generados
```

Archivo:

```text
model_dashboard.py
```

App publicada:

```text
https://biodiversity-finder-training.streamlit.app/
```

---

## 🧪 Tests

Ejecutar:

```bash
pytest
```

Los tests verifican:

```text
descarga y planificación de consultas GBIF
limpieza de datos
feature engineering
enriquecimiento climático
nombres comunes desde GBIF + Wikidata
documento de búsqueda inteligente
entrenamiento del modelo
dashboard
```

---

## 🧠 Explicación simple para la presentación

Este proyecto tiene dos partes:

1. **Training repo**  
   Prepara los datos, entrena el modelo y genera la enciclopedia.

2. **App repo**  
   Usa la enciclopedia para que el usuario pueda buscar especies.

La parte más importante fue entender que la app no debe tener reglas manuales como:

```text
si escriben "león", busca "Panthera leo"
```

Eso sería un parche.

La solución correcta es enriquecer los datos con nombres comunes desde fuentes externas.  
Así la app puede buscar de forma más natural y multilingüe.

---

## 🧭 Convenciones del proyecto

- Variables, funciones y archivos: inglés.
- Comentarios, docstrings y documentación: español.
- README y explicación para el equipo: español.
- Datos grandes y modelos: se publican en Hugging Face, no se guardan manualmente en GitHub.
- La app debe ser genérica.
- Los nombres de especies deben venir de los datos, no de reglas manuales en la app.

---

## ✅ Estado actual

El proyecto actualmente incluye:

```text
pipeline global multi-source desde GBIF
climate enrichment con NASA POWER API
vernacular names desde GBIF Species API + Wikidata
modelo ML
enciclopedia inteligente
publicación en Hugging Face
dashboard de métricas
GitHub Actions manual con parámetros
```
