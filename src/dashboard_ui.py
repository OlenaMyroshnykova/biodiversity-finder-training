"""Componentes visuales y lógica de resumen del dashboard de la modelo."""

from __future__ import annotations

import pandas as pd
import streamlit as st


SUMMARY_LABELS = {"accuracy", "macro avg", "weighted avg"}


def apply_dashboard_styles() -> None:
    """
    Aplica estilos CSS suaves al dashboard.
    """
    css = """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(46, 125, 50, 0.12), transparent 32%),
            radial-gradient(circle at bottom right, rgba(25, 118, 210, 0.10), transparent 36%),
            linear-gradient(135deg, #f7fbf7 0%, #ffffff 48%, #eef6ff 100%);
    }

    h1, h2, h3 {
        color: #1b5e20;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(46, 125, 50, 0.16);
        border-radius: 18px;
        padding: 0.85rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header() -> None:
    """
    Renderiza cabecera principal.
    """
    st.title("🤖 Biodiversity Finder — Model Evaluation Dashboard")

    st.info(
        "Dashboard técnico para revisar la calidad de la modelo entrenada. "
        "Los datos se leen desde Hugging Face Datasets, no desde archivos locales."
    )

    st.markdown(
        """
        **Fuente de artefactos:** `selenamir/biodiversity-finder-artifacts`

        Este dashboard complementa la enciclopedia pública. La enciclopedia sirve para
        buscar especies; esta página sirve para justificar la calidad de la parte ML.
        """
    )


def extract_model_summary(
    metrics: dict,
    classification_report_df: pd.DataFrame,
) -> dict[str, float | None]:
    """
    Extrae métricas resumen desde `metrics.json` y `classification_report.csv`.
    """
    summary = {
        "accuracy": get_metric_from_dict(metrics, "accuracy"),
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "weighted_f1": None,
    }

    if classification_report_df.empty or "label" not in classification_report_df.columns:
        return summary

    macro_row = get_row_by_label(classification_report_df, "macro avg")
    weighted_row = get_row_by_label(classification_report_df, "weighted avg")

    if macro_row is not None:
        summary["macro_precision"] = get_numeric_value(macro_row, "precision")
        summary["macro_recall"] = get_numeric_value(macro_row, "recall")
        summary["macro_f1"] = get_numeric_value(macro_row, "f1-score")

    if weighted_row is not None:
        summary["weighted_f1"] = get_numeric_value(weighted_row, "f1-score")

    return summary


def render_summary_metrics(summary: dict[str, float | None]) -> None:
    """
    Renderiza tarjetas de métricas generales.
    """
    st.header("📈 Métricas principales")

    columns = st.columns(5)

    metric_items = [
        ("Accuracy", summary.get("accuracy")),
        ("Macro precision", summary.get("macro_precision")),
        ("Macro recall", summary.get("macro_recall")),
        ("Macro F1", summary.get("macro_f1")),
        ("Weighted F1", summary.get("weighted_f1")),
    ]

    for column, (label, value) in zip(columns, metric_items):
        with column:
            st.metric(label, format_metric(value))


def render_adequacy_notes(
    summary: dict[str, float | None],
    classification_report_df: pd.DataFrame,
) -> None:
    """
    Muestra interpretación simple de la adecuación de la modelo.
    """
    accuracy = summary.get("accuracy")
    macro_f1 = summary.get("macro_f1")
    class_rows_df = get_class_rows(classification_report_df)

    if accuracy is not None and accuracy >= 0.90:
        st.success(
            "La accuracy es alta. Para este dataset, la modelo separa bien las "
            "clases taxonómicas aprendidas."
        )
    elif accuracy is not None and accuracy >= 0.70:
        st.warning(
            "La accuracy es aceptable, pero conviene revisar clases con pocos ejemplos."
        )
    else:
        st.error(
            "La accuracy es baja o no está disponible. Haría falta revisar datos, "
            "features o entrenamiento."
        )

    if macro_f1 is not None:
        if macro_f1 >= 0.80:
            st.success(
                "El Macro F1 es alto. Esto sugiere que el rendimiento no depende "
                "solo de las clases mayoritarias."
            )
        else:
            st.warning(
                "El Macro F1 no es muy alto. Puede haber clases minoritarias mal "
                "representadas."
            )

    if not class_rows_df.empty and "support" in class_rows_df.columns:
        min_support = int(class_rows_df["support"].min())
        max_support = int(class_rows_df["support"].max())

        if min_support < 10:
            st.warning(
                f"Hay clases con muy pocos ejemplos de test. Support mínimo = {min_support}. "
                "Las métricas de esas clases pueden ser poco estables."
            )

        st.info(
            f"Rango de support por clase: mínimo {min_support}, máximo {max_support}. "
            "Si una clase tiene muchos más ejemplos que otra, conviene mirar Macro F1, "
            "no solo Accuracy."
        )


def get_class_rows(classification_report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve solo filas que representan clases reales.
    """
    if classification_report_df.empty or "label" not in classification_report_df.columns:
        return pd.DataFrame()

    class_rows_df = classification_report_df[
        ~classification_report_df["label"].astype(str).isin(SUMMARY_LABELS)
    ].copy()

    required_columns = ["precision", "recall", "f1-score", "support"]

    for column in required_columns:
        if column not in class_rows_df.columns:
            return pd.DataFrame()

        class_rows_df[column] = pd.to_numeric(class_rows_df[column], errors="coerce")

    return class_rows_df.dropna(subset=required_columns)


def get_row_by_label(
    classification_report_df: pd.DataFrame,
    label: str,
) -> pd.Series | None:
    """
    Devuelve fila por nombre de label.
    """
    rows = classification_report_df[
        classification_report_df["label"].astype(str).eq(label)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_numeric_value(row: pd.Series, column: str) -> float | None:
    """
    Extrae valor numérico de una fila.
    """
    if column not in row:
        return None

    value = pd.to_numeric(row[column], errors="coerce")

    if pd.isna(value):
        return None

    return float(value)


def get_metric_from_dict(metrics: dict, key: str) -> float | None:
    """
    Extrae una métrica numérica desde un diccionario.
    """
    value = metrics.get(key)

    if isinstance(value, (int, float)):
        return float(value)

    return None


def format_metric(value: float | None) -> str:
    """
    Formatea métrica como porcentaje.
    """
    if value is None:
        return "N/A"

    return f"{value * 100:.1f}%"
