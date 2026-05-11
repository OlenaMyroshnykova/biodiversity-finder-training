# 🤖 Biodiversity Finder — Model Evaluation Dashboard

Esta aplicación Streamlit forma parte del repositorio técnico `biodiversity-finder-training`.

Su objetivo es mostrar de forma visual la calidad de la modelo entrenada en el pipeline de datos.

## Arquitectura

```text
GBIF
↓
GitHub Actions en biodiversity-finder-training
↓
limpieza + feature engineering + entrenamiento
↓
reports/metrics.json + reports/classification_report.csv
↓
Hugging Face Datasets
↓
model_dashboard.py
```

El dashboard no entrena la modelo. Solo visualiza los resultados ya publicados.

## Artefactos usados

```text
selenamir/biodiversity-finder-artifacts
├── reports/metrics.json
└── reports/classification_report.csv
```

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

Y puede enlazar a este dashboard técnico para que el usuario o el profesorado vea la evaluación de la modelo.
