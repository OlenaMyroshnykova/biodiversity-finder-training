"""Tests para el dashboard de evaluación de la modelo."""

import pandas as pd

from src.dashboard_loader import normalize_report_columns
from src.dashboard_ui import extract_model_summary, format_metric, get_class_rows


def build_report() -> pd.DataFrame:
    """
    Crea un classification report mínimo para tests.
    """
    return pd.DataFrame(
        [
            {
                "label": "Aves",
                "precision": 0.90,
                "recall": 0.80,
                "f1-score": 0.85,
                "support": 20,
            },
            {
                "label": "Mammalia",
                "precision": 1.00,
                "recall": 1.00,
                "f1-score": 1.00,
                "support": 15,
            },
            {
                "label": "macro avg",
                "precision": 0.95,
                "recall": 0.90,
                "f1-score": 0.925,
                "support": 35,
            },
            {
                "label": "weighted avg",
                "precision": 0.94,
                "recall": 0.91,
                "f1-score": 0.92,
                "support": 35,
            },
        ]
    )


def test_format_metric() -> None:
    """
    Debe formatear números como porcentajes.
    """
    assert format_metric(0.923) == "92.3%"
    assert format_metric(None) == "N/A"


def test_extract_model_summary() -> None:
    """
    Debe extraer métricas principales.
    """
    summary = extract_model_summary(
        metrics={"accuracy": 0.91},
        classification_report_df=build_report(),
    )

    assert summary["accuracy"] == 0.91
    assert summary["macro_precision"] == 0.95
    assert summary["macro_recall"] == 0.90
    assert summary["macro_f1"] == 0.925
    assert summary["weighted_f1"] == 0.92


def test_get_class_rows_removes_summary_rows() -> None:
    """
    Debe quitar macro avg y weighted avg.
    """
    class_rows_df = get_class_rows(build_report())

    assert len(class_rows_df) == 2
    assert set(class_rows_df["label"]) == {"Aves", "Mammalia"}


def test_normalize_report_columns_renames_unnamed_column() -> None:
    """
    Debe convertir Unnamed: 0 en label.
    """
    report_df = pd.DataFrame(
        {
            "Unnamed: 0": ["Aves"],
            "precision": [1.0],
            "recall": [1.0],
            "f1-score": [1.0],
            "support": [10],
        }
    )

    normalized_df = normalize_report_columns(report_df)

    assert "label" in normalized_df.columns
    assert normalized_df.iloc[0]["label"] == "Aves"
