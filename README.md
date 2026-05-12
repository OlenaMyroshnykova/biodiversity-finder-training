# 🌿 Biodiversity Finder Training

Training pipeline for **Biodiversity Finder**, an intelligent species search and biodiversity encyclopedia project.

This repository contains the data science and machine learning pipeline used to collect biodiversity data, clean and enrich it, train a classification model, build a species encyclopedia, and publish the generated artifacts for the Streamlit application.

---

## 🔗 Related Links

- **Interactive app repository:**  
  https://github.com/OlenaMyroshnykova/biodiversity-finder-app

- **Model dashboard:**  
  https://biodiversity-finder-training.streamlit.app/

- **Published artifacts on Hugging Face:**  
  https://huggingface.co/datasets/selenamir/biodiversity-finder-artifacts

---

## 👩‍💻 Contributors

This project was developed by:

- **Olena Myroshnykova**
- **Nana Torres Gonzalez**
- **Camila Rabelo**
- **Neema Nelly**

---

## 📌 Project Overview

**Biodiversity Finder Training** prepares the datasets and machine learning artifacts used by the public Biodiversity Finder app.

The pipeline works with real biodiversity data and produces an enriched species encyclopedia that can be searched by scientific names, taxonomy, common names, and multilingual descriptions.

The project uses:

- biodiversity occurrence data from **GBIF**;
- taxonomic and geographic data cleaning;
- climate enrichment using **NASA POWER API**;
- common name enrichment using **GBIF Species API** and **Wikidata**;
- a machine learning model for taxonomic classification;
- Hugging Face as a public artifact storage.

---

## ✨ Main Features

- Download biodiversity records from GBIF.
- Combine multiple biodiversity queries into a single dataset.
- Clean and validate occurrence records.
- Generate taxonomic, geographic, temporal, and climate features.
- Enrich records with climate data.
- Train a machine learning classifier.
- Build an intelligent species encyclopedia.
- Enrich species with common names from external sources.
- Publish datasets, reports, and model artifacts to Hugging Face.
- Provide a Streamlit dashboard for model and data inspection.

---

## 🧠 What This Repository Produces

The pipeline generates the following main artifacts:

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
reports/data_samples/
```

The most important artifact for the public app is:

```text
data/processed/species_encyclopedia.parquet
```

It contains one row per species with taxonomy, observations, geographic information, profile text, search text, and common names.

---

## 🏗️ Repository Structure

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
  test_dashboard_ui.py
  test_data_acquisition.py
  test_data_cleaning.py
  test_data_snapshots.py
  test_feature_engineering.py
  test_intelligent_search_document.py
  test_jaguar_query_plan.py
  test_model_training.py
  test_vernacular_names.py

model_dashboard.py
pytest.ini
requirements.txt
README.md
```

---

## ⚙️ Tech Stack

- **Python**
- **Pandas**
- **Scikit-learn**
- **Requests**
- **Joblib**
- **PyArrow**
- **Streamlit**
- **Hugging Face Hub**
- **GitHub Actions**

---

## 🌍 Data Sources

### GBIF Occurrence API

Used to download real biodiversity occurrence records.

Typical fields include:

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

### GBIF Species API

Used to collect available common names for species.

### Wikidata

Used as an additional source of multilingual common names, labels, and aliases.

### NASA POWER API

Used to enrich geographic coordinates with climate-related variables.

---

## 🔄 Pipeline Workflow

```text
GBIF occurrence data
        ↓
Raw dataset
        ↓
Data cleaning
        ↓
Feature engineering
        ↓
Climate enrichment
        ↓
Model training
        ↓
Species encyclopedia generation
        ↓
Common name enrichment
        ↓
Artifact publication on Hugging Face
        ↓
Streamlit app consumption
```

---

## 🚀 Local Setup

Clone the repository:

```bash
git clone https://github.com/OlenaMyroshnykova/biodiversity-finder-training.git
cd biodiversity-finder-training
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

---

## ▶️ Running the Pipeline Locally

Small local run:

```bash
python scripts/run_pipeline.py \
  --country GLOBAL \
  --max-records 5000 \
  --min-class-records 10 \
  --max-climate-points 50 \
  --max-vernacular-species 300
```

Recommended larger run:

```bash
python scripts/run_pipeline.py \
  --country GLOBAL \
  --max-records 50000 \
  --min-class-records 20 \
  --max-climate-points 250 \
  --max-vernacular-species 3000
```

Optional flags:

```bash
--skip-climate-api
--skip-vernacular-api
--skip-wikidata
```

---

## 📊 Model Dashboard

The repository includes a Streamlit dashboard for inspecting model results and generated artifacts.

Run locally:

```bash
streamlit run model_dashboard.py
```

Public dashboard:

```text
https://biodiversity-finder-training.streamlit.app/
```

The dashboard includes:

- model accuracy;
- classification report;
- generated artifact overview;
- sample datasets;
- data quality information.

---

## 🤖 Machine Learning

The model is trained in:

```text
src/model_training.py
```

It uses engineered features to classify biodiversity records by taxonomic class.

The generated model is saved as:

```text
models/taxon_classifier.joblib
```

Model metrics are saved in:

```text
reports/metrics.json
reports/classification_report.csv
```

---

## 🔎 Species Encyclopedia

The species encyclopedia is built in:

```text
src/encyclopedia.py
```

and enriched with common names in:

```text
src/vernacular_names.py
```

The final encyclopedia is saved as:

```text
data/processed/species_encyclopedia.parquet
```

This file is used by the public Streamlit app to search and display species.

---

## 🧪 Testing

Run all tests:

```bash
pytest
```

The test suite checks:

- data acquisition logic;
- data cleaning;
- feature engineering;
- climate enrichment;
- common name enrichment;
- search document generation;
- model training;
- dashboard helpers.

---

## 🚢 GitHub Actions

The training workflow is located at:

```text
.github/workflows/train_and_publish.yml
```

It is designed to be executed manually from GitHub Actions:

```text
Actions → Train and publish biodiversity artifacts → Run workflow
```

Recommended parameters:

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

## 🔐 Required Secrets

To publish artifacts to Hugging Face, the repository requires the following GitHub secret:

```text
HF_TOKEN
```

---

## 📦 Artifact Publication

Artifacts are uploaded to the Hugging Face dataset repository:

```text
selenamir/biodiversity-finder-artifacts
```

The upload script is:

```text
scripts/upload_artifacts.py
```

---

## 📝 License

This project is intended for educational purposes.

---

## 🌱 Project Status

The repository currently supports:

- global GBIF data collection;
- multi-source biodiversity queries;
- climate enrichment;
- multilingual common name enrichment;
- machine learning training;
- species encyclopedia generation;
- Hugging Face artifact publishing;
- Streamlit model dashboard.
