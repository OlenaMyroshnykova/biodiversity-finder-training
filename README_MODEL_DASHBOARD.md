# Biodiversity Finder — Model Evaluation Dashboard

Esta aplicación Streamlit forma parte del repositorio técnico `biodiversity-finder-training`. Su objetivo es mostrar de forma visual la calidad del modelo entrenado en el pipeline de datos y explicar de dónde salen los datos usados por el proyecto.

## Arquitectura

```text
GBIF / Wikidata / Wikipedia / NASA POWER / IUCN
        ↓
GitHub Actions en biodiversity-finder-training
        ↓
limpieza + feature engineering + enriquecimiento + entrenamiento
        ↓
reports/metrics.json + reports/classification_report.csv
        ↓
Hugging Face Datasets
        ↓
model_dashboard.py
```

El dashboard no entrena el modelo. Solo visualiza y explica los resultados ya publicados.

## Fuentes de datos documentadas

- **GBIF Occurrence API:** observaciones de biodiversidad, coordenadas, fechas, país y taxonomía.
- **GBIF Species API:** nombres comunes y apoyo taxonómico.
- **Wikidata / Wikipedia / Wikimedia Commons:** imágenes, etiquetas y enlaces abiertos.
- **NASA POWER API:** variables climáticas aproximadas para enriquecer el hábitat.
- **IUCN Red List API v4:** estado de conservación oficial cuando hay token y datos disponibles.
- **Pipeline propio:** `tags_de_busqueda`, `search_document`, exports light/offline y contrato de artefactos.
- **Hugging Face Datasets:** publicación central de parquet, CSV, métricas y reports.

## Métricas mostradas

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Precision por clase
- Recall por clase
- F1-score por clase
- Support por clase

## Cómo interpretar las métricas

- **Accuracy** resume el acierto global, pero puede engañar si hay muchas muestras de una sola clase.
- **Precision** responde: cuando el modelo predice una clase, ¿cuántas veces acierta?
- **Recall** responde: de todos los ejemplos reales de una clase, ¿cuántos encuentra?
- **F1-score** equilibra precision y recall.
- **Macro F1** da el mismo peso a todas las clases y es clave para detectar si las clases pequeñas están mal aprendidas.
- **Weighted F1** pondera por support y refleja el rendimiento global, aunque puede ocultar problemas en clases minoritarias.
- **Support** indica cuántos ejemplos reales hay por clase en test.

## Artefactos usados

```text
selenamir/biodiversity-finder-artifacts
├── reports/metrics.json
└── reports/classification_report.csv
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run model_dashboard.py
```

## Deploy en Streamlit Cloud

```text
Repository: OlenaMyroshnykova/biodiversity-finder-training
Branch: main
Main file path: model_dashboard.py
```

## Relación con la app de enciclopedia

La enciclopedia pública vive en otro repositorio:

```text
biodiversity-finder-app
```

La app sirve para buscar especies. Este dashboard sirve para defender la parte técnica: fuentes de datos, arquitectura del pipeline, métricas del modelo y limitaciones éticas.
