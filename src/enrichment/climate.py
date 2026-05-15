"""Enriquecimiento climático para cumplir df.merge() del proyecto.

Este módulo crea una tabla climática por coordenadas redondeadas y la une
con los datos de especies usando `df.merge()`.

La fuente prioritaria es NASA POWER API. Si la API falla, se usa una
estimación educativa por latitud para que el pipeline siga siendo estable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


NASA_POWER_CLIMATOLOGY_URL = (
    "https://power.larc.nasa.gov/api/temporal/climatology/point"
)

CLIMATE_PARAMETERS = "T2M,PRECTOTCORR,RH2M"


@dataclass(frozen=True)
class ClimateRecord:
    """Registro climático por punto redondeado."""

    climate_lat: float
    climate_lon: float
    climate_temperature_mean: float
    climate_precipitation_mean: float
    climate_humidity_mean: float
    climate_temperature_zone: str
    climate_humidity_zone: str
    climate_source: str


def add_climate_features(
    features_df: pd.DataFrame,
    *,
    coordinate_precision: int = 0,
    max_api_points: int = 250,
    use_api: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Añade variables climáticas a un DataFrame de especies.

    Esta función contiene el `df.merge()` explícito que pide el enunciado.

    Args:
        features_df: DataFrame con columnas decimalLatitude y decimalLongitude.
        coordinate_precision: decimales para agrupar coordenadas.
        max_api_points: máximo de puntos que se consultan a NASA POWER.
        use_api: permite desactivar API en tests o ejecución rápida.

    Returns:
        Tupla: DataFrame enriquecido, tabla climática.
    """
    if features_df.empty:
        return features_df.copy(), build_empty_climate_reference()

    working_df = features_df.copy()

    if "decimalLatitude" not in working_df.columns or "decimalLongitude" not in working_df.columns:
        working_df["climate_lat"] = pd.NA
        working_df["climate_lon"] = pd.NA
        climate_reference_df = build_empty_climate_reference()

        return working_df, climate_reference_df

    working_df["climate_lat"] = pd.to_numeric(
        working_df["decimalLatitude"],
        errors="coerce",
    ).round(coordinate_precision)

    working_df["climate_lon"] = pd.to_numeric(
        working_df["decimalLongitude"],
        errors="coerce",
    ).round(coordinate_precision)

    unique_points_df = (
        working_df[["climate_lat", "climate_lon"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    climate_reference_df = build_climate_reference(
        unique_points_df=unique_points_df,
        max_api_points=max_api_points,
        use_api=use_api,
    )

    # df.merge() requerido por el enunciado:
    enriched_df = working_df.merge(
        climate_reference_df,
        on=["climate_lat", "climate_lon"],
        how="left",
        validate="many_to_one",
    )

    return enriched_df, climate_reference_df


def build_empty_climate_reference() -> pd.DataFrame:
    """Devuelve una tabla climática vacía con el esquema esperado."""
    return pd.DataFrame(
        columns=[
            "climate_lat",
            "climate_lon",
            "climate_temperature_mean",
            "climate_precipitation_mean",
            "climate_humidity_mean",
            "climate_temperature_zone",
            "climate_humidity_zone",
            "climate_source",
        ]
    )


def build_climate_reference(
    unique_points_df: pd.DataFrame,
    *,
    max_api_points: int = 250,
    use_api: bool = True,
) -> pd.DataFrame:
    """Construye una tabla climática para coordenadas únicas."""
    if unique_points_df.empty:
        return build_empty_climate_reference()

    climate_records: list[ClimateRecord] = []

    for position, row in unique_points_df.iterrows():
        latitude = float(row["climate_lat"])
        longitude = float(row["climate_lon"])

        record: ClimateRecord | None = None

        if use_api and position < max_api_points:
            record = fetch_nasa_power_climate(
                latitude=latitude,
                longitude=longitude,
            )

        if record is None:
            record = estimate_climate_by_latitude(
                latitude=latitude,
                longitude=longitude,
            )

        climate_records.append(record)

    return pd.DataFrame([record.__dict__ for record in climate_records])


def fetch_nasa_power_climate(
    *,
    latitude: float,
    longitude: float,
    timeout: int = 20,
) -> ClimateRecord | None:
    """
    Descarga climatología anual desde NASA POWER API.

    Variables:
    - T2M: temperatura media a 2 metros.
    - PRECTOTCORR: precipitación corregida.
    - RH2M: humedad relativa a 2 metros.
    """
    params = {
        "parameters": CLIMATE_PARAMETERS,
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "format": "JSON",
    }

    try:
        response = requests.get(
            NASA_POWER_CLIMATOLOGY_URL,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    parameters = payload.get("properties", {}).get("parameter", {})

    temperature = extract_annual_value(parameters, "T2M")
    precipitation = extract_annual_value(parameters, "PRECTOTCORR")
    humidity = extract_annual_value(parameters, "RH2M")

    if temperature is None or precipitation is None or humidity is None:
        return None

    return ClimateRecord(
        climate_lat=latitude,
        climate_lon=longitude,
        climate_temperature_mean=round(float(temperature), 3),
        climate_precipitation_mean=round(float(precipitation), 3),
        climate_humidity_mean=round(float(humidity), 3),
        climate_temperature_zone=classify_temperature_zone(float(temperature)),
        climate_humidity_zone=classify_humidity_zone(float(humidity)),
        climate_source="NASA POWER API",
    )


def extract_annual_value(parameters: dict[str, Any], parameter_name: str) -> float | None:
    """Extrae valor anual de respuesta NASA POWER."""
    values = parameters.get(parameter_name, {})

    if not isinstance(values, dict):
        return None

    for annual_key in ("ANN", "Annual", "annual"):
        value = values.get(annual_key)

        if value is not None:
            return float(value)

    numeric_values = [
        float(value)
        for value in values.values()
        if isinstance(value, (int, float))
    ]

    if not numeric_values:
        return None

    return sum(numeric_values) / len(numeric_values)


def estimate_climate_by_latitude(
    *,
    latitude: float,
    longitude: float,
) -> ClimateRecord:
    """
    Estimación educativa por latitud.

    No sustituye una base climática real, pero permite que el pipeline funcione
    si la API externa no está disponible.
    """
    abs_latitude = abs(latitude)

    temperature = 28.0 - (abs_latitude * 0.35)

    if abs_latitude < 15:
        precipitation = 5.5
        humidity = 78.0
    elif abs_latitude < 35:
        precipitation = 2.5
        humidity = 58.0
    elif abs_latitude < 55:
        precipitation = 1.8
        humidity = 64.0
    else:
        precipitation = 0.8
        humidity = 70.0

    return ClimateRecord(
        climate_lat=latitude,
        climate_lon=longitude,
        climate_temperature_mean=round(temperature, 3),
        climate_precipitation_mean=round(precipitation, 3),
        climate_humidity_mean=round(humidity, 3),
        climate_temperature_zone=classify_temperature_zone(temperature),
        climate_humidity_zone=classify_humidity_zone(humidity),
        climate_source="latitude_estimate_fallback",
    )


def classify_temperature_zone(temperature: float) -> str:
    """Clasifica temperatura media anual en zona simple."""
    if temperature < 5:
        return "frio"

    if temperature < 18:
        return "templado"

    return "calido"


def classify_humidity_zone(humidity: float) -> str:
    """Clasifica humedad relativa en zona simple."""
    if humidity < 45:
        return "seco"

    if humidity < 70:
        return "intermedio"

    return "humedo"
