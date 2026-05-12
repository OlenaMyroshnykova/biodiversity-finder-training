"""Tests para puntos de avistamiento."""

import pandas as pd

from src.occurrence_points import build_species_occurrence_points


def test_build_species_occurrence_points_from_gbif_columns() -> None:
    """Debe construir puntos desde decimalLatitude/decimalLongitude."""
    df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo (Linnaeus, 1758)",
                "decimalLatitude": -1.2,
                "decimalLongitude": 36.8,
                "countryCode": "KE",
                "eventDate": "2026-01-01",
            },
            {
                "scientific_name": "Panthera leo (Linnaeus, 1758)",
                "decimalLatitude": -2.0,
                "decimalLongitude": 35.0,
                "countryCode": "TZ",
                "eventDate": "2026-01-02",
            },
        ]
    )

    points_df = build_species_occurrence_points(df)

    assert len(points_df) == 2
    assert "canonical_scientific_name" in points_df.columns
    assert "decimalLatitude" in points_df.columns
    assert "decimalLongitude" in points_df.columns
    assert points_df.iloc[0]["canonical_scientific_name"] == "Panthera leo"


def test_build_species_occurrence_points_from_snake_case_columns() -> None:
    """Debe aceptar nombres snake_case."""
    df = pd.DataFrame(
        [
            {
                "scientific_name": "Rana temporaria Linnaeus, 1758",
                "decimal_latitude": 40.0,
                "decimal_longitude": -3.0,
                "country_code": "ES",
                "event_date": "2026-01-01",
            }
        ]
    )

    points_df = build_species_occurrence_points(df)

    assert len(points_df) == 1
    assert points_df.iloc[0]["canonical_scientific_name"] == "Rana temporaria Linnaeus, 1758"
    assert points_df.iloc[0]["decimalLatitude"] == 40.0
    assert points_df.iloc[0]["decimalLongitude"] == -3.0


def test_build_species_occurrence_points_returns_empty_without_coordinates() -> None:
    """Debe devolver schema vacío si no hay coordenadas."""
    df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo",
                "family": "Felidae",
            }
        ]
    )

    points_df = build_species_occurrence_points(df)

    assert points_df.empty
    assert points_df.columns.tolist() == [
        "scientific_name",
        "canonical_scientific_name",
        "decimalLatitude",
        "decimalLongitude",
        "countryCode",
        "eventDate",
    ]
