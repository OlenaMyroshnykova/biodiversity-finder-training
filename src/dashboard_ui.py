"""Componentes visuales y lógica de resumen del dashboard de evaluación."""
from __future__ import annotations

import pandas as pd
import streamlit as st

SUMMARY_LABELS = {"accuracy", "macro avg", "weighted avg"}


def apply_dashboard_styles() -> None:
    """Aplica estilos CSS suaves al dashboard."""
    css = """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(67, 56, 202, 0.06);
        border: 1px solid rgba(67, 56, 202, 0.14);
        padding: 1rem;
        border-radius: 1rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header() -> None:
    """Renderiza cabecera principal."""
    st.title("Biodiversity Finder — Model Evaluation Dashboard")
    st.info(
        "Dashboard técnico para revisar la calidad del modelo de clasificación "
        "taxonómica y explicar de forma transparente los datos usados por el pipeline."
    )
    st.markdown(
        """
        **Fuente de artefactos:** `selenamir/biodiversity-finder-artifacts`  
        Este dashboard complementa la enciclopedia pública: la app busca especies;
        esta página justifica la parte de Data Science, el modelo ML, las fuentes de datos
        y las limitaciones del proyecto.
        """
    )
    render_theory_explainer()


def render_theory_explainer() -> None:
    """Explica qué hace el modelo y qué significa cada métrica."""
    with st.expander("Teoría: qué hace el modelo y cómo leer las métricas", expanded=False):
        st.markdown("### 1. ¿Qué hace el modelo?")
        st.write(
            "El modelo es un **clasificador taxonómico**. A partir de un registro de observación "
            "de biodiversidad —por ejemplo coordenadas, fecha, país, familia, nombre científico "
            "y otros metadatos— predice la **clase taxonómica** de la especie, como `Aves`, "
            "`Mammalia`, `Insecta`, `Reptilia` o `Amphibia`."
        )
        st.write(
            "No intenta adivinar la especie exacta. Ese sería un problema mucho más difícil, "
            "porque requeriría muchas más muestras por especie y una validación biológica más estricta."
        )

        st.markdown("### 2. ¿Qué aprende el modelo?")
        st.markdown(
            """
            El pipeline de scikit-learn combina tres familias de señales:

            - **Features numéricas:** latitud, longitud, año, mes u otras variables derivadas. Se escalan con `StandardScaler`.
            - **Features categóricas:** reino, filo, familia, país, estación o tipo de registro. Se codifican con `OneHotEncoder`.
            - **Features textuales:** nombre científico y taxonomía. Se transforman con `TfidfVectorizer` para que el modelo pueda usar palabras y bigramas.

            Para el entregable, el modelo demuestra que el dataset no es solo una tabla: tiene estructura, patrones y calidad suficiente para entrenar una solución ML básica.
            """
        )

        st.divider()
        st.markdown("### 3. Matriz mental: predicción correcta vs. errores")
        st.markdown(
            """
            Para entender precision y recall, imagina que evaluamos la clase **Mammalia**:

            | Caso | Significado |
            |---|---|
            | **True Positive** | Era Mammalia y el modelo dijo Mammalia. |
            | **False Positive** | No era Mammalia, pero el modelo dijo Mammalia. |
            | **False Negative** | Era Mammalia, pero el modelo dijo otra clase. |
            | **True Negative** | No era Mammalia y el modelo no dijo Mammalia. |
            """
        )

        st.divider()
        st.markdown("### 4. Métricas principales")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Accuracy")
            st.write(
                "Mide qué porcentaje total de registros del test se clasificó correctamente. "
                "Es fácil de entender, pero puede engañar si el dataset está desbalanceado. "
                "Ejemplo: si casi todo son aves, un modelo que predice siempre aves podría tener "
                "accuracy alta y aun así ser malo para anfibios o reptiles."
            )

            st.markdown("#### Precision")
            st.write(
                "Responde: de todo lo que el modelo predijo como una clase, ¿cuánto era realmente de esa clase? "
                "Precision alta significa pocas falsas alarmas. Es importante cuando no queremos etiquetar "
                "incorrectamente especies de otras clases."
            )

            st.markdown("#### Recall")
            st.write(
                "Responde: de todos los ejemplos reales de una clase, ¿cuántos encontró el modelo? "
                "Recall alto significa que el modelo pierde pocos casos. Es importante para no invisibilizar "
                "clases pequeñas o amenazadas."
            )

        with col2:
            st.markdown("#### F1-score")
            st.write(
                "Combina precision y recall en una sola medida: `F1 = 2 × (precision × recall) / (precision + recall)`. "
                "Es útil cuando queremos equilibrio: no basta con acertar mucho una clase grande, también hay que reducir errores."
            )

            st.markdown("#### Macro average")
            st.write(
                "Calcula la media dando el mismo peso a cada clase. Para este proyecto es muy importante, "
                "porque nos dice si el modelo funciona también en clases minoritarias, no solo en la clase dominante."
            )

            st.markdown("#### Weighted average")
            st.write(
                "Calcula la media ponderando por el número de ejemplos de cada clase. Refleja mejor el rendimiento global, "
                "pero puede ocultar problemas en clases con poco support."
            )

        st.divider()
        st.markdown("### 5. ¿Qué es support y por qué importa?")
        st.write(
            "El **support** es el número de ejemplos reales de cada clase en el conjunto de test. "
            "Si una clase tiene support muy bajo, sus métricas no son estables: con 5 ejemplos, un solo error cambia mucho el porcentaje."
        )

        st.markdown("### 6. Cómo leer el dashboard en la defensa")
        st.markdown(
            """
            - Primero mira **accuracy** para una idea global rápida.
            - Luego mira **Macro F1** para comprobar si el modelo trata bien a las clases pequeñas.
            - Después revisa los gráficos por clase: una clase con bajo F1 y bajo support probablemente necesita más datos.
            - Si **Weighted F1** es alto pero **Macro F1** es bajo, el modelo funciona bien en clases grandes pero mal en clases pequeñas.
            """
        )

        st.markdown("### 7. Umbrales orientativos para este proyecto")
        st.markdown(
            """
            | Métrica | Aceptable | Buena | Muy buena |
            |---|---:|---:|---:|
            | Accuracy | ≥ 70% | ≥ 85% | ≥ 92% |
            | Macro F1 | ≥ 0.60 | ≥ 0.75 | ≥ 0.85 |
            | Weighted F1 | ≥ 0.70 | ≥ 0.85 | ≥ 0.92 |
            """
        )
        st.caption(
            "Son umbrales educativos para un proyecto de aprendizaje con datos abiertos. "
            "Una solución científica real necesitaría validación cruzada, auditoría de errores y revisión experta."
        )

        st.divider()
        st.markdown("### 8. Limitaciones del modelo")
        st.warning(
            "El modelo clasifica clases taxonómicas, no especies exactas. Además, depende de la calidad del dataset: "
            "si GBIF contiene sesgo geográfico, duplicados o clases poco representadas, las métricas también reflejan esas limitaciones."
        )


def extract_model_summary(
    metrics: dict,
    classification_report_df: pd.DataFrame,
) -> dict[str, float | None]:
    """Extrae métricas resumen desde metrics.json y classification_report.csv."""
    summary = {
        "accuracy": get_metric_from_dict(metrics, "accuracy"),
        "macro_f1": get_metric_from_dict(metrics, "macro_f1"),
        "macro_precision": get_metric_from_dict(metrics, "macro_precision"),
        "macro_recall": get_metric_from_dict(metrics, "macro_recall"),
        "weighted_f1": get_metric_from_dict(metrics, "weighted_f1"),
        "train_rows": get_metric_from_dict(metrics, "train_rows"),
        "test_rows": get_metric_from_dict(metrics, "test_rows"),
        "classes": get_metric_from_dict(metrics, "classes"),
    }

    if classification_report_df.empty or "label" not in classification_report_df.columns:
        return summary

    macro_row = get_row_by_label(classification_report_df, "macro avg")
    weighted_row = get_row_by_label(classification_report_df, "weighted avg")

    if macro_row is not None:
        if summary["macro_precision"] is None:
            summary["macro_precision"] = get_numeric_value(macro_row, "precision")
        if summary["macro_recall"] is None:
            summary["macro_recall"] = get_numeric_value(macro_row, "recall")
        if summary["macro_f1"] is None:
            summary["macro_f1"] = get_numeric_value(macro_row, "f1-score")

    if weighted_row is not None and summary["weighted_f1"] is None:
        summary["weighted_f1"] = get_numeric_value(weighted_row, "f1-score")

    return summary


def render_summary_metrics(summary: dict[str, float | None]) -> None:
    """Renderiza tarjetas de métricas generales con contexto."""
    st.header("Métricas principales")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "Accuracy",
            format_metric(summary.get("accuracy")),
            help="Porcentaje total de registros clasificados correctamente. Puede engañar si el dataset está desbalanceado.",
        )
    with col2:
        st.metric(
            "Macro Precision",
            format_metric(summary.get("macro_precision")),
            help="Precision media dando el mismo peso a cada clase. Alta = menos falsos positivos por clase.",
        )
    with col3:
        st.metric(
            "Macro Recall",
            format_metric(summary.get("macro_recall")),
            help="Recall medio dando el mismo peso a cada clase. Alto = se pierden menos ejemplos de clases pequeñas.",
        )
    with col4:
        st.metric(
            "Macro F1",
            format_metric(summary.get("macro_f1")),
            help="Balance entre precision y recall con igual peso por clase. Muy útil para datasets desbalanceados.",
        )
    with col5:
        st.metric(
            "Weighted F1",
            format_metric(summary.get("weighted_f1")),
            help="F1 ponderado por support. Refleja rendimiento global, pero puede ocultar clases pequeñas.",
        )

    if any(summary.get(k) is not None for k in ["train_rows", "test_rows", "classes"]):
        st.markdown("---")
        c1, c2, c3, _ = st.columns([1, 1, 1, 2])
        with c1:
            v = summary.get("train_rows")
            st.metric(
                "Registros train",
                f"{int(v):,}" if v else "N/A",
                help="Observaciones usadas para ajustar el modelo.",
            )
        with c2:
            v = summary.get("test_rows")
            st.metric(
                "Registros test",
                f"{int(v):,}" if v else "N/A",
                help="Observaciones reservadas para evaluar el modelo. No se usan durante el entrenamiento.",
            )
        with c3:
            v = summary.get("classes")
            st.metric(
                "Clases taxonómicas",
                f"{int(v)}" if v else "N/A",
                help="Número de clases que el modelo aprendió a distinguir.",
            )


def render_adequacy_notes(
    summary: dict[str, float | None],
    classification_report_df: pd.DataFrame,
) -> None:
    """Muestra interpretación contextualizada de las métricas."""
    accuracy = summary.get("accuracy")
    macro_f1 = summary.get("macro_f1")
    class_rows_df = get_class_rows(classification_report_df)

    if accuracy is not None:
        if accuracy >= 0.90:
            st.success(
                f"Accuracy = {accuracy*100:.1f}% — el modelo clasifica bien la mayoría de los registros. "
                "Para datos abiertos de biodiversidad, es una señal sólida."
            )
        elif accuracy >= 0.70:
            st.warning(
                f"Accuracy = {accuracy*100:.1f}% — aceptable, pero conviene revisar las clases con menos support."
            )
        else:
            st.error(
                f"Accuracy = {accuracy*100:.1f}% — baja. Posibles causas: dataset desbalanceado, poco volumen o features insuficientes."
            )

    if macro_f1 is not None:
        if macro_f1 >= 0.80:
            st.success(
                f"Macro F1 = {macro_f1*100:.1f}% — el modelo mantiene buen equilibrio entre clases grandes y pequeñas."
            )
        elif macro_f1 >= 0.60:
            st.warning(
                f"Macro F1 = {macro_f1*100:.1f}% — algunas clases minoritarias pueden necesitar más datos."
            )
        else:
            st.error(
                f"Macro F1 = {macro_f1*100:.1f}% — las clases minoritarias probablemente están mal representadas."
            )

    if not class_rows_df.empty and "support" in class_rows_df.columns:
        min_support = int(class_rows_df["support"].min())
        max_support = int(class_rows_df["support"].max())
        min_class = class_rows_df.loc[class_rows_df["support"].idxmin(), "label"]
        ratio = max_support / max(min_support, 1)

        if min_support < 10:
            st.warning(
                f"La clase '{min_class}' tiene solo {min_support} ejemplos en test. "
                "Sus métricas son poco estables estadísticamente."
            )
        if ratio > 20:
            st.info(
                f"Dataset muy desbalanceado: la clase más grande tiene {max_support} ejemplos y la más pequeña {min_support} "
                f"(ratio {ratio:.0f}x). En este caso Macro F1 es más informativo que Accuracy."
            )
        else:
            st.info(
                f"Rango de support: mínimo {min_support} ('{min_class}'), máximo {max_support}. "
                "El dataset parece razonablemente balanceado para una demo educativa."
            )


def get_class_rows(classification_report_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo filas que representan clases reales."""
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
    """Devuelve fila por nombre de label."""
    rows = classification_report_df[
        classification_report_df["label"].astype(str).eq(label)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def get_numeric_value(row: pd.Series, column: str) -> float | None:
    """Extrae valor numérico de una fila."""
    if column not in row:
        return None
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def get_metric_from_dict(metrics: dict, key: str) -> float | None:
    """Extrae una métrica numérica desde un diccionario."""
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def format_metric(value: float | None) -> str:
    """Formatea métrica como porcentaje."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"
