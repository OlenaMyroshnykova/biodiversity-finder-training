"""Entrenamiento y evaluación de la modelo ML."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ['decimal_latitude','decimal_longitude','year','month','observation_decade']
CATEGORICAL_FEATURES = ['kingdom','phylum','family','country_code','basis_of_record','season','coordinate_precision']
TEXT_FEATURE = 'scientific_text'
TARGET = 'taxon_label'

def train_model(features_df: pd.DataFrame, model_path: Path, metrics_path: Path, report_path: Path, random_state: int = 42) -> dict[str, float]:
    """Entrena y evalúa una modelo de clasificación taxonómica."""
    model_df = prepare_model_dataframe(features_df)
    x_data = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]]
    y_data = model_df[TARGET]
    stratify_target = y_data if y_data.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=random_state, stratify=stratify_target)
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    model_path.parent.mkdir(parents=True, exist_ok=True); metrics_path.parent.mkdir(parents=True, exist_ok=True); report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics = {'accuracy': float(accuracy), 'train_rows': int(len(x_train)), 'test_rows': int(len(x_test)), 'classes': int(y_data.nunique())}
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame(report).transpose().to_csv(report_path)
    return metrics

def build_model() -> Pipeline:
    """Construye el pipeline de Machine Learning."""
    preprocessor = ColumnTransformer(transformers=[
        ('numeric', StandardScaler(), NUMERIC_FEATURES),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ('text', TfidfVectorizer(lowercase=True, strip_accents='unicode', min_df=2, max_features=8000, ngram_range=(1,2)), TEXT_FEATURE),
    ], remainder='drop')
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def prepare_model_dataframe(features_df: pd.DataFrame) -> pd.DataFrame:
    """Prepara columnas antes de entrenar."""
    model_df = features_df.copy()
    for column in NUMERIC_FEATURES:
        model_df[column] = pd.to_numeric(model_df[column], errors='coerce')
    for column in CATEGORICAL_FEATURES + [TEXT_FEATURE, TARGET]:
        model_df[column] = model_df[column].fillna('desconocido').astype(str)
    return model_df.dropna(subset=NUMERIC_FEATURES + [TARGET])
