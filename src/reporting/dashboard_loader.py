"""Carga de artefactos de evaluación desde Hugging Face."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


REPO_ID = "selenamir/biodiversity-finder-artifacts"
REPO_TYPE = "dataset"


@st.cache_data(show_spinner="Cargando métricas desde Hugging Face...")
def load_metrics() -> dict:
    """
    Carga `metrics.json` desde Hugging Face.

    Returns:
        Diccionario con métricas principales de entrenamiento.
    """
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename="reports/metrics.json",
    )

    with open(file_path, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@st.cache_data(show_spinner="Cargando classification report desde Hugging Face...")
def load_classification_report() -> pd.DataFrame:
    """
    Carga `classification_report.csv` desde Hugging Face.

    Returns:
        Dataframe con precision, recall, f1-score y support.
    """
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename="reports/classification_report.csv",
    )

    report_df = pd.read_csv(file_path)

    return normalize_report_columns(report_df)


def normalize_report_columns(report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas del classification report.

    `classification_report` de sklearn puede guardarse con la etiqueta como índice.
    Al leerlo desde CSV, esa etiqueta puede venir como `Unnamed: 0`.
    """
    normalized_df = report_df.copy()

    if "label" not in normalized_df.columns:
        if "Unnamed: 0" in normalized_df.columns:
            normalized_df = normalized_df.rename(columns={"Unnamed: 0": "label"})
        else:
            first_column = normalized_df.columns[0]
            normalized_df = normalized_df.rename(columns={first_column: "label"})

    return normalized_df
