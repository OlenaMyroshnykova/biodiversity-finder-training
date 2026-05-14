"""Tests para la sección de fuentes de datos del dashboard."""
from src.dashboard_data_sources import build_data_sources_table


def test_data_sources_table_contains_core_sources() -> None:
    """La tabla debe documentar las fuentes externas principales."""
    sources_df = build_data_sources_table()

    source_names = set(sources_df["Fuente"].astype(str))

    assert "GBIF Occurrence API" in source_names
    assert "GBIF Species API" in source_names
    assert "NASA POWER API" in source_names
    assert "IUCN Red List API v4" in source_names
    assert "Hugging Face Datasets" in source_names


def test_data_sources_table_documents_limitations() -> None:
    """Cada fuente debe explicar su limitación para defender el proyecto."""
    sources_df = build_data_sources_table()

    assert not sources_df.empty
    assert sources_df["Riesgo / limitación"].astype(str).str.len().min() > 10
    assert sources_df["Uso en el proyecto"].astype(str).str.len().min() > 10
