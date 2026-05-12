# 🌿 Biodiversity Finder Training

Training pipeline for **Biodiversity Finder**, an intelligent biodiversity search and species encyclopedia project.

This repository prepares the data science layer of the project: data acquisition, cleaning, enrichment, machine learning, EDA reports, conservation awareness, offline exports, and artifact publication.

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

**Biodiversity Finder Training** builds the enriched dataset used by the public Streamlit app.

The pipeline combines biodiversity observations, climate features, common names, conservation signals, search tags, and map coordinates into a species encyclopedia.

The final app can search species by scientific names, common names, taxonomy, and natural-language-style tags.

---

## ✨ Main Features

- GBIF occurrence data collection.
- Multi-source dataset creation with `pd.concat()`.
- Data cleaning and feature engineering.
- Climate enrichment with `pd.merge()`.
- Common name enrichment using GBIF Species API and Wikidata.
- Conservation status enrichment layer.
- `tags_de_busqueda` for vibe-style natural search.
- Random sampling with `df.sample()` for development with large datasets.
- Occurrence points for future Folium habitat maps.
- Offline light dataset export.
- EDA reports and data discovery findings.
- Machine learning model training.
- Hugging Face artifact publication.

---

## 🧠 Data Science Requirements Covered

### Big Join

The project joins multiple sources:

```text
GBIF occurrences
+ climate reference data
+ common names from GBIF/Wikidata
+ conservation status layer
+ search tags
```

The pipeline demonstrates:

```text
pd.concat()
pd.merge()
df.sample()
```

### Search Tags

The column:

```text
tags_de_busqueda
```

combines estimated color, habitat, size, common names, taxonomy, and source query information.

This prepares the app for fast `df.loc` filters such as:

```python
df.loc[
    (df["size_tag"].str.contains("small")) &
    (df["habitat_tag"].str.contains("desert"))
]
```

### Conservation Awareness

The encyclopedia includes:

```text
conservation_status
conservation_category
is_threatened
conservation_source
conservation_note
```

These fields allow the app to display threatened species with a different visual style.

---

## 🌍 Data Sources

- GBIF Occurrence API
- GBIF Species API
- Wikidata
- NASA POWER API
- Educational conservation estimates / optional official validation

---

## 🔄 Pipeline Workflow

```text
GBIF occurrence data
        ↓
Raw dataset
        ↓
Cleaning
        ↓
Feature engineering
        ↓
Climate enrichment
        ↓
Model training
        ↓
Species encyclopedia
        ↓
Common names
        ↓
Conservation status
        ↓
Search tags
        ↓
Occurrence points
        ↓
Offline light exports
        ↓
EDA reports
        ↓
Hugging Face publication
```

---

## 🏗️ Repository Structure

```text
.github/workflows/
  train_and_publish.yml

src/
  climate_enrichment.py
  conservation_status.py
  data_acquisition.py
  data_cleaning.py
  eda_reporting.py
  encyclopedia.py
  feature_engineering.py
  model_training.py
  occurrence_points.py
  offline_export.py
  search_tags.py
  vernacular_names.py

scripts/
  run_pipeline.py
  upload_artifacts.py

tests/
  test_pro_requirements.py
  test_vernacular_names.py
  ...
```

---

## 📦 Main Artifacts

```text
data/processed/species_encyclopedia.parquet
data/processed/species_occurrence_points.parquet
data/processed/species_encyclopedia_light.parquet
data/processed/species_occurrence_points_light.parquet
data/interim/conservation_status.csv
data/interim/vernacular_names.csv
reports/eda/eda_findings.json
reports/eda/eda_class_distribution.csv
reports/eda/eda_conservation_summary.csv
reports/eda/eda_habitat_summary.csv
```

---

## 🚀 Local Setup

```bash
git clone https://github.com/OlenaMyroshnykova/biodiversity-finder-training.git
cd biodiversity-finder-training
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
pytest
```

---

## ▶️ Running the Pipeline

Small development run:

```bash
python scripts/run_pipeline.py \
  --country GLOBAL \
  --max-records 5000 \
  --min-class-records 10 \
  --max-climate-points 50 \
  --max-vernacular-species 300 \
  --sample-size 1000
```

Recommended run:

```bash
python scripts/run_pipeline.py \
  --country GLOBAL \
  --max-records 50000 \
  --min-class-records 20 \
  --max-climate-points 250 \
  --max-vernacular-species 3000
```

---

## 📊 EDA and Data Discovery

The pipeline generates EDA outputs in:

```text
reports/eda/
```

These reports help discover patterns such as:

- dominant taxonomic classes;
- distribution of conservation categories;
- common habitat tags;
- possible data imbalance;
- species with low observation counts.

---

## 🌱 Ethical Impact

This project promotes awareness of biodiversity and conservation.

However:

- open biodiversity data can be incomplete or geographically biased;
- conservation status must be validated with official scientific sources;
- estimated tags are educational and should not be treated as biological facts;
- the app is a learning tool, not an official ecological risk assessment system.

---

## 📴 Offline Mode

The pipeline creates lightweight compressed artifacts:

```text
species_encyclopedia_light.parquet
species_occurrence_points_light.parquet
```

These files allow the app to work as a field-friendly biodiversity encyclopedia with fewer network dependencies.

Offline mode means:

```text
the app uses a prepared local/light dataset
```

It does not mean that the system can download new GBIF, Wikidata, or climate data without Internet.

---

## 🧪 Testing

```bash
pytest
```

The tests cover:

- data cleaning;
- feature engineering;
- climate enrichment;
- common names;
- conservation status;
- search tags;
- occurrence points;
- offline export;
- EDA findings;
- model training.

---

## 🚢 GitHub Actions

The workflow is:

```text
.github/workflows/train_and_publish.yml
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

## 🔐 Required Secret

```text
HF_TOKEN
```

Used to publish artifacts to Hugging Face.

---

## 📝 License

This project is intended for educational purposes.
