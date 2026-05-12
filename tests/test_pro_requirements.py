"""Tests para requisitos profesionales nuevos."""

import pandas as pd

from src.conservation_status import add_conservation_status_to_encyclopedia
from src.eda_reporting import build_eda_findings
from src.occurrence_points import build_species_occurrence_points
from src.offline_export import build_offline_encyclopedia
from src.search_tags import add_search_tags_to_encyclopedia


def build_test_encyclopedia() -> pd.DataFrame:
    """Crea enciclopedia mínima."""
    return pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo (Linnaeus, 1758)",
                "canonical_scientific_name": "Panthera leo",
                "vernacular_names": "Lion | León",
                "kingdom": "Animalia",
                "taxon_class": "Mammalia",
                "taxon_order": "Carnivora",
                "family": "Felidae",
                "genus": "Panthera",
                "observations": 9,
                "countries": "KE | TZ",
                "source_queries": "big_cats_felidae",
                "profile_text": "Large cat.",
                "search_document": "Panthera leo Lion León Felidae",
            },
            {
                "scientific_name": "Papilio machaon Linnaeus, 1758",
                "canonical_scientific_name": "Papilio machaon",
                "vernacular_names": "Swallowtail butterfly",
                "kingdom": "Animalia",
                "taxon_class": "Insecta",
                "taxon_order": "Lepidoptera",
                "family": "Papilionidae",
                "genus": "Papilio",
                "observations": 50,
                "countries": "ES",
                "source_queries": "butterflies_lepidoptera",
                "profile_text": "Butterfly.",
                "search_document": "Papilio butterfly mariposa",
            },
        ]
    )


def build_test_features() -> pd.DataFrame:
    """Crea features mínimas con coordenadas."""
    return pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo (Linnaeus, 1758)",
                "canonical_scientific_name": "Panthera leo",
                "decimalLatitude": -1.0,
                "decimalLongitude": 36.0,
                "countryCode": "KE",
                "eventDate": "2026-01-01",
                "taxon_class": "Mammalia",
                "family": "Felidae",
            },
            {
                "scientific_name": "Panthera leo (Linnaeus, 1758)",
                "canonical_scientific_name": "Panthera leo",
                "decimalLatitude": -2.0,
                "decimalLongitude": 35.0,
                "countryCode": "TZ",
                "eventDate": "2026-01-02",
                "taxon_class": "Mammalia",
                "family": "Felidae",
            },
        ]
    )


def test_conservation_status_adds_required_columns_with_merge() -> None:
    """Debe añadir columnas de conservación."""
    enriched_df, conservation_df = add_conservation_status_to_encyclopedia(
        build_test_encyclopedia()
    )

    assert not conservation_df.empty
    assert "conservation_status" in enriched_df.columns
    assert "is_threatened" in enriched_df.columns
    assert enriched_df["is_threatened"].dtype == bool


def test_search_tags_adds_tags_de_busqueda() -> None:
    """Debe crear tags_de_busqueda."""
    tagged_df = add_search_tags_to_encyclopedia(build_test_encyclopedia())

    assert "color_tag" in tagged_df.columns
    assert "habitat_tag" in tagged_df.columns
    assert "size_tag" in tagged_df.columns
    assert "tags_de_busqueda" in tagged_df.columns
    assert "large" in tagged_df.iloc[0]["tags_de_busqueda"]
    assert "butterfly" in tagged_df.iloc[1]["tags_de_busqueda"]


def test_occurrence_points_for_folium_map() -> None:
    """Debe crear puntos de avistamiento para mapa."""
    points_df = build_species_occurrence_points(build_test_features())

    assert not points_df.empty
    assert "decimalLatitude" in points_df.columns
    assert "decimalLongitude" in points_df.columns


def test_offline_export_keeps_essential_columns() -> None:
    """Debe crear enciclopedia ligera."""
    enriched_df, _ = add_conservation_status_to_encyclopedia(build_test_encyclopedia())
    tagged_df = add_search_tags_to_encyclopedia(enriched_df)

    offline_df = build_offline_encyclopedia(tagged_df, max_species=1)

    assert len(offline_df) == 1
    assert "tags_de_busqueda" in offline_df.columns
    assert "conservation_status" in offline_df.columns


def test_eda_findings_include_ethics_and_limitations() -> None:
    """Debe generar hallazgos EDA con impacto ético."""
    enriched_df, _ = add_conservation_status_to_encyclopedia(build_test_encyclopedia())
    findings = build_eda_findings(enriched_df, build_test_features())

    assert "ethical_impact" in findings
    assert "limitations" in findings
    assert findings["total_species"] == 2
