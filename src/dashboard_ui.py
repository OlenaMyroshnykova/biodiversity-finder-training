"""Componentes visuales y lógica de resumen del dashboard de evaluación."""

from __future__ import annotations

import pandas as pd
import streamlit as st


SUMMARY_LABELS = {"accuracy", "macro avg", "weighted avg"}


def apply_dashboard_styles() -> None:
    """Aplica estilos CSS suaves al dashboard."""
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
    """Renderiza cabecera principal."""
    st.title("🤖 Biodiversity Finder — Model Evaluation Dashboard")

    st.info(
        "Dashboard técnico para revisar la calidad del modelo de clasificación taxonómica. "
        "Los datos se leen desde Hugging Face Datasets tras cada ejecución del pipeline."
    )

    st.markdown(
        """
        **Fuente de artefactos:** `selenamir/biodiversity-finder-artifacts`

        Este dashboard complementa la enciclopedia pública. La enciclopedia sirve para
        buscar especies; esta página sirve para justificar y entender la calidad de la parte ML.
        """
    )

    render_theory_explainer()


def render_theory_explainer() -> None:
    """Explica qué hace el modelo y qué significa cada métrica."""
    with st.expander("📖 ¿Qué hace el modelo y qué significan las métricas?", expanded=False):

        st.markdown("### 🧠 ¿Qué hace el modelo?")
        st.write(
            "El modelo es un **clasificador taxonómico**: dado un registro de observación "
            "de GBIF (coordenadas, fecha, nombre científico, familia, país...), "
            "predice a qué **clase taxonómica** pertenece la especie "
            "(Mammalia, Aves, Insecta, Reptilia, Fungi, etc.)."
        )
        st.write(
            "Es una **Regresión Logística** entrenada con un pipeline de scikit-learn que combina "
            "tres tipos de features:"
        )
        st.markdown(
            """
            - **Numéricas** (latitud, longitud, año, mes, decade): escala con `StandardScaler`
            - **Categóricas** (reino, filo, familia, país, basis_of_record, estación): codifica con `OneHotEncoder`
            - **Texto** (nombre científico + taxonomía completa): vectoriza con `TfidfVectorizer` bigramas
            """
        )
        st.caption(
            "El modelo no se usa para búsqueda en la app — esa parte usa TF-IDF directamente "
            "sobre la enciclopedia. El clasificador demuestra que el dataset es lo suficientemente "
            "rico y limpio como para aprender patrones taxonómicos."
        )

        st.divider()
        st.markdown("### 📏 ¿Qué significa cada métrica?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Accuracy")
            st.write(
                "De todos los registros del test, ¿qué fracción clasificó correctamente? "
                "**Alta accuracy no siempre es buena señal** si el dataset está muy desbalanceado "
                "(si el 90% de registros son Insecta, predecir siempre Insecta da 90% de accuracy)."
            )

            st.markdown("#### Precision")
            st.write(
                "De todas las veces que el modelo predijo 'Mammalia', ¿qué fracción "
                "era realmente Mammalia? "
                "**Precision alta** = pocas falsas alarmas. "
                "Importante cuando los falsos positivos tienen coste."
            )

            st.markdown("#### Recall")
            st.write(
                "De todos los registros que realmente son Mammalia, ¿cuántos identificó "
                "correctamente el modelo? "
                "**Recall alto** = pocas pérdidas. "
                "Importante cuando los falsos negativos tienen coste."
            )

        with col2:
            st.markdown("#### F1-score")
            st.write(
                "Media harmónica entre Precision y Recall: "
                "`F1 = 2 × (P × R) / (P + R)`. "
                "Equilibra ambas métricas. **Es la métrica más útil** para este problema "
                "porque el dataset puede estar desbalanceado entre clases."
            )

            st.markdown("#### Macro F1")
            st.write(
                "F1 promediado dando **el mismo peso a cada clase**, "
                "independientemente de cuántos ejemplos tenga. "
                "Un Macro F1 alto significa que el modelo funciona bien incluso "
                "en clases minoritarias como Reptilia o Fungi."
            )

            st.markdown("#### Weighted F1")
            st.write(
                "F1 promediado **ponderando por el número de ejemplos de cada clase** (support). "
                "Refleja mejor el rendimiento global real, pero puede ocultar "
                "problemas en clases pequeñas."
            )

        st.divider()
        st.markdown("### 📊 ¿Qué es el Support?")
        st.write(
            "El **support** es el número de ejemplos de test que pertenecen a cada clase. "
            "Si una clase tiene support = 5, sus métricas de precision/recall/f1 "
            "son estadísticamente poco fiables — un solo error cambia mucho el resultado."
        )

        st.divider()
        st.markdown("### 🎯 ¿Qué valores son buenos para este proyecto?")
        st.info(
            "Para un clasificador taxonómico con 10-15 clases y datos de ciencia ciudadana "
            "(ruidosos y desbalanceados), se considera:"
        )
        st.markdown(
            """
            | Métrica | Aceptable | Bueno | Muy bueno |
            |---------|-----------|-------|-----------|
            | Accuracy | ≥ 70% | ≥ 85% | ≥ 92% |
            | Macro F1 | ≥ 0.60 | ≥ 0.75 | ≥ 0.85 |
            | Weighted F1 | ≥ 0.70 | ≥ 0.85 | ≥ 0.92 |
            """
        )
        st.caption(
            "Estos umbrales son orientativos para datos de biodiversidad. "
            "Un modelo de producción real requeriría validación cruzada, "
            "análisis de errores por especie y evaluación por biólogos expertos."
        )

        st.divider()
        st.markdown("### ⚠️ Limitaciones del modelo")
        st.warning(
            "Este modelo clasifica **clases taxonómicas** (Mammalia, Aves...), "
            "no especies individuales. Predecir la especie exacta requeriría "
            "muchos más datos y un modelo mucho más complejo. "
            "Además, el rendimiento depende directamente de la calidad y diversidad "
            "del dataset descargado de GBIF — si una clase tiene pocos registros "
            "en el pipeline, sus métricas serán bajas."
        )


def extract_model_summary(
    metrics: dict,
    classification_report_df: pd.DataFrame,
) -> dict[str, float | None]:
    """Extrae métricas resumen desde metrics.json y classification_report.csv."""
    # Preferimos leer macro_f1/weighted_f1 directamente de metrics.json
    # (guardados por model_training.py). El classification_report es fallback.
    summary = {
        "accuracy":        get_metric_from_dict(metrics, "accuracy"),
        "macro_f1":        get_metric_from_dict(metrics, "macro_f1"),
        "macro_precision": get_metric_from_dict(metrics, "macro_precision"),
        "macro_recall":    get_metric_from_dict(metrics, "macro_recall"),
        "weighted_f1":     get_metric_from_dict(metrics, "weighted_f1"),
        "train_rows":      get_metric_from_dict(metrics, "train_rows"),
        "test_rows":       get_metric_from_dict(metrics, "test_rows"),
        "classes":         get_metric_from_dict(metrics, "classes"),
    }

    # Fallback: si no están en metrics.json, leer del classification_report
    if classification_report_df.empty or "label" not in classification_report_df.columns:
        return summary

    macro_row    = get_row_by_label(classification_report_df, "macro avg")
    weighted_row = get_row_by_label(classification_report_df, "weighted avg")

    if macro_row is not None:
        if summary["macro_precision"] is None:
            summary["macro_precision"] = get_numeric_value(macro_row, "precision")
        if summary["macro_recall"] is None:
            summary["macro_recall"] = get_numeric_value(macro_row, "recall")
        if summary["macro_f1"] is None:
            summary["macro_f1"] = get_numeric_value(macro_row, "f1-score")

    if weighted_row is not None:
        if summary["weighted_f1"] is None:
            summary["weighted_f1"] = get_numeric_value(weighted_row, "f1-score")

    return summary


def render_summary_metrics(summary: dict[str, float | None]) -> None:
    """Renderiza tarjetas de métricas generales con contexto."""
    st.header("📈 Métricas principales")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            format_metric(summary.get("accuracy")),
            help="Fracción de registros clasificados correctamente en test. "
                 "Ojo: puede ser alta aunque el modelo falle en clases minoritarias.",
        )
    with col2:
        st.metric(
            "Macro Precision",
            format_metric(summary.get("macro_precision")),
            help="Precision promediada con igual peso por clase. "
                 "Alta = pocas falsas alarmas en todas las clases.",
        )
    with col3:
        st.metric(
            "Macro Recall",
            format_metric(summary.get("macro_recall")),
            help="Recall promediado con igual peso por clase. "
                 "Alto = el modelo no pierde ejemplos de clases minoritarias.",
        )
    with col4:
        st.metric(
            "Macro F1",
            format_metric(summary.get("macro_f1")),
            help="Balance entre Precision y Recall, igual peso por clase. "
                 "La métrica más representativa para datasets desbalanceados.",
        )
    with col5:
        st.metric(
            "Weighted F1",
            format_metric(summary.get("weighted_f1")),
            help="F1 ponderado por número de ejemplos. "
                 "Refleja el rendimiento global real.",
        )

    # Fila secundaria: contexto del entrenamiento
    if any(summary.get(k) is not None for k in ["train_rows", "test_rows", "classes"]):
        st.markdown("---")
        c1, c2, c3, _ = st.columns([1, 1, 1, 2])
        with c1:
            v = summary.get("train_rows")
            st.metric("Registros train", f"{int(v):,}" if v else "N/A",
                      help="Número de observaciones usadas para entrenar el modelo.")
        with c2:
            v = summary.get("test_rows")
            st.metric("Registros test", f"{int(v):,}" if v else "N/A",
                      help="Número de observaciones reservadas para evaluar (nunca vistas en training).")
        with c3:
            v = summary.get("classes")
            st.metric("Clases taxonómicas", f"{int(v)}" if v else "N/A",
                      help="Número de clases taxonómicas distintas que el modelo aprendió a identificar.")


def render_adequacy_notes(
    summary: dict[str, float | None],
    classification_report_df: pd.DataFrame,
) -> None:
    """Muestra interpretación contextualizada de las métricas."""
    accuracy = summary.get("accuracy")
    macro_f1 = summary.get("macro_f1")
    class_rows_df = get_class_rows(classification_report_df)

    # Accuracy
    if accuracy is not None:
        if accuracy >= 0.90:
            st.success(
                f"✅ **Accuracy = {accuracy*100:.1f}%** — El modelo clasifica bien "
                "la mayoría de los registros. Para datos de ciencia ciudadana con "
                "ruido taxonómico, esto es un resultado sólido."
            )
        elif accuracy >= 0.70:
            st.warning(
                f"⚠️ **Accuracy = {accuracy*100:.1f}%** — Aceptable, pero hay margen "
                "de mejora. Revisa las clases con support bajo en los gráficos."
            )
        else:
            st.error(
                f"❌ **Accuracy = {accuracy*100:.1f}%** — Baja. Posibles causas: "
                "dataset muy desbalanceado, pocas clases con suficientes ejemplos, "
                "o features insuficientes. Revisa min_class_records en el pipeline."
            )

    # Macro F1
    if macro_f1 is not None:
        if macro_f1 >= 0.80:
            st.success(
                f"✅ **Macro F1 = {macro_f1*100:.1f}%** — El modelo funciona bien "
                "incluso en clases minoritarias. Las nuevas clases (Reptilia, Fungi, "
                "Chondrichthyes) están bien representadas en el dataset."
            )
        elif macro_f1 >= 0.60:
            st.warning(
                f"⚠️ **Macro F1 = {macro_f1*100:.1f}%** — Algunas clases minoritarias "
                "tienen rendimiento bajo. Considera aumentar max_records o bajar "
                "min_class_records para clases con pocos registros."
            )
        else:
            st.error(
                f"❌ **Macro F1 = {macro_f1*100:.1f}%** — Las clases minoritarias "
                "están muy mal clasificadas. El dataset necesita más diversidad "
                "o hay clases con demasiado pocos ejemplos."
            )

    # Support
    if not class_rows_df.empty and "support" in class_rows_df.columns:
        min_support = int(class_rows_df["support"].min())
        max_support = int(class_rows_df["support"].max())
        min_class = class_rows_df.loc[class_rows_df["support"].idxmin(), "label"]

        if min_support < 10:
            st.warning(
                f"⚠️ La clase **'{min_class}'** tiene solo **{min_support} ejemplos** "
                "en el conjunto de test. Sus métricas de precision/recall/F1 "
                "son estadísticamente poco fiables con tan pocos datos."
            )

        ratio = max_support / max(min_support, 1)
        if ratio > 20:
            st.info(
                f"ℹ️ **Dataset muy desbalanceado**: la clase más grande tiene "
                f"{max_support} ejemplos de test y la más pequeña {min_support} "
                f"(ratio {ratio:.0f}x). En este caso el **Macro F1 es más informativo "
                "que la Accuracy** para evaluar el rendimiento real del modelo."
            )
        else:
            st.info(
                f"ℹ️ Rango de support: mínimo {min_support} ('{min_class}'), "
                f"máximo {max_support}. Dataset razonablemente balanceado."
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
