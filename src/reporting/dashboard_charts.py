"""Gráficos para el dashboard de evaluación de la modelo."""

from __future__ import annotations

import altair as alt
import pandas as pd


def build_metric_by_class_chart(
    class_report_df: pd.DataFrame,
    metric_name: str,
) -> alt.Chart:
    """
    Construye un gráfico de barras para precision, recall o f1-score por clase.
    """
    chart_data = class_report_df.sort_values(metric_name, ascending=False).copy()

    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                f"{metric_name}:Q",
                title=metric_name,
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("label:N", title="Clase", sort="-x"),
            tooltip=[
                "label",
                alt.Tooltip(f"{metric_name}:Q", format=".3f"),
                alt.Tooltip("support:Q", format=".0f"),
            ],
        )
        .properties(
            height=320,
            title=f"{metric_name} por clase",
        )
    )


def build_support_chart(class_report_df: pd.DataFrame) -> alt.Chart:
    """
    Construye un gráfico de cantidad de ejemplos por clase.
    """
    chart_data = class_report_df.sort_values("support", ascending=False).copy()

    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("support:Q", title="Support"),
            y=alt.Y("label:N", title="Clase", sort="-x"),
            tooltip=[
                "label",
                alt.Tooltip("support:Q", format=".0f"),
                alt.Tooltip("f1-score:Q", format=".3f"),
            ],
        )
        .properties(
            height=320,
            title="Support por clase",
        )
    )
