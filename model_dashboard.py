"""Dashboard Streamlit para evaluar la modelo de Biodiversity Finder."""

from __future__ import annotations

import streamlit as st

from src.dashboard_charts import (
    build_metric_by_class_chart,
    build_support_chart,
)
from src.dashboard_loader import load_classification_report, load_metrics
from src.dashboard_ui import (
    apply_dashboard_styles,
    extract_model_summary,
    get_class_rows,
    render_adequacy_notes,
    render_header,
    render_summary_metrics,
)


def main() -> None:
    """Ejecuta el dashboard de evaluación de la modelo."""
    st.set_page_config(
        page_title="Biodiversity Finder — Model Dashboard",
        page_icon="🤖",
        layout="wide",
    )

    apply_dashboard_styles()
    render_header()

    metrics = load_metrics()
    classification_report_df = load_classification_report()

    summary = extract_model_summary(metrics, classification_report_df)

    render_summary_metrics(summary)

    st.divider()

    st.header("📌 Interpretación de adecuación")
    render_adequacy_notes(summary, classification_report_df)

    st.divider()

    st.header("📊 Métricas por clase")

    class_rows_df = get_class_rows(classification_report_df)

    if class_rows_df.empty:
        st.warning(
            "No se encontraron filas de clases reales en classification_report.csv."
        )
    else:
        chart_column_1, chart_column_2 = st.columns(2)

        with chart_column_1:
            st.altair_chart(
                build_metric_by_class_chart(class_rows_df, "precision"),
                width="stretch",
            )

        with chart_column_2:
            st.altair_chart(
                build_metric_by_class_chart(class_rows_df, "recall"),
                width="stretch",
            )

        chart_column_3, chart_column_4 = st.columns(2)

        with chart_column_3:
            st.altair_chart(
                build_metric_by_class_chart(class_rows_df, "f1-score"),
                width="stretch",
            )

        with chart_column_4:
            st.altair_chart(
                build_support_chart(class_rows_df),
                width="stretch",
            )

    st.divider()

    st.header("🧾 Classification report completo")
    st.dataframe(
        classification_report_df,
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()
