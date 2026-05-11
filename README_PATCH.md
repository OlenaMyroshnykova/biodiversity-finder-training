# Model Dashboard Patch para `biodiversity-finder-training`

Este parche añade al repositorio de entrenamiento una aplicación Streamlit separada para visualizar la calidad de la modelo.

## Qué añade

```text
model_dashboard.py
src/dashboard_loader.py
src/dashboard_charts.py
src/dashboard_ui.py
tests/test_dashboard_ui.py
```

También reemplaza:

```text
requirements.txt
README_MODEL_DASHBOARD.md
```

## Qué muestra el dashboard

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Gráfico de precision por clase
- Gráfico de recall por clase
- Gráfico de F1-score por clase
- Gráfico de support por clase
- Tabla completa `classification_report.csv`
- Interpretación simple de adecuación de la modelo

## Fuente de datos

El dashboard NO lee archivos locales de `reports/`, porque esos archivos están ignorados por Git.

Lee los artefactos publicados en Hugging Face:

```text
selenamir/biodiversity-finder-artifacts
```

Archivos usados:

```text
reports/metrics.json
reports/classification_report.csv
```

## Cómo ejecutar localmente

```bash
pip install -r requirements.txt
pytest
streamlit run model_dashboard.py
```

## Cómo desplegar en Streamlit Cloud

```text
Repository: OlenaMyroshnykova/biodiversity-finder-training
Branch: main
Main file path: model_dashboard.py
```

## Después de aplicar el parche

```bash
git add .
git commit -m "Add model evaluation dashboard"
git push
```
