"""Tests para enriquecimiento climático y df.merge()."""

import pandas as pd

from src.climate_enrichment import (
    ClimateRecord,
    add_climate_features,
    classify_humidity_zone,
    classify_temperature_zone,
)


def test_add_climate_features_uses_merge_with_coordinates() -> None:
    """Debe añadir columnas climáticas mediante coordenadas redondeadas."""
    features_df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera onca",
                "decimalLatitude": -3.12,
                "decimalLongitude": -60.02,
            },
            {
                "scientific_name": "Ursus maritimus",
                "decimalLatitude": 70.4,
                "decimalLongitude": -42.2,
            },
        ]
    )

    enriched_df, climate_reference_df = add_climate_features(
        features_df,
        coordinate_precision=0,
        use_api=False,
    )

    assert len(enriched_df) == 2
    assert not climate_reference_df.empty
    assert "climate_temperature_mean" in enriched_df.columns
    assert "climate_precipitation_mean" in enriched_df.columns
    assert "climate_humidity_zone" in enriched_df.columns


def test_add_climate_features_can_use_mocked_api(monkeypatch) -> None:
    """Debe poder usar datos de API simulada."""
    def fake_fetch(latitude: float, longitude: float, timeout: int = 20):
        return ClimateRecord(
            climate_lat=latitude,
            climate_lon=longitude,
            climate_temperature_mean=22.0,
            climate_precipitation_mean=3.0,
            climate_humidity_mean=65.0,
            climate_temperature_zone="calido",
            climate_humidity_zone="intermedio",
            climate_source="NASA POWER API",
        )

    monkeypatch.setattr(
        "src.climate_enrichment.fetch_nasa_power_climate",
        fake_fetch,
    )

    features_df = pd.DataFrame(
        [
            {
                "scientific_name": "Test species",
                "decimalLatitude": 10.2,
                "decimalLongitude": 20.4,
            }
        ]
    )

    enriched_df, climate_reference_df = add_climate_features(
        features_df,
        coordinate_precision=0,
        max_api_points=10,
        use_api=True,
    )

    assert enriched_df.iloc[0]["climate_source"] == "NASA POWER API"
    assert climate_reference_df.iloc[0]["climate_temperature_mean"] == 22.0


def test_temperature_zone_classifier() -> None:
    """Debe clasificar temperatura."""
    assert classify_temperature_zone(2) == "frio"
    assert classify_temperature_zone(12) == "templado"
    assert classify_temperature_zone(22) == "calido"


def test_humidity_zone_classifier() -> None:
    """Debe clasificar humedad."""
    assert classify_humidity_zone(30) == "seco"
    assert classify_humidity_zone(55) == "intermedio"
    assert classify_humidity_zone(80) == "humedo"
