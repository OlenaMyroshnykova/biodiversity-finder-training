"""Generación de muestras inspeccionables del pipeline de datos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SAMPLE_SIZE = 100


def save_pipeline_snapshots(
    *,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    features_df: pd.DataFrame,
    encyclopedia_df: pd.DataFrame,
    output_dir: Path,
    parameters: dict[str, Any],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Path]:
    """
    Guarda muestras pequeñas y documentación automática del pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = {
        "raw_sample": output_dir / "raw_sample.csv",
        "clean_sample": output_dir / "clean_sample.csv",
        "features_sample": output_dir / "features_sample.csv",
        "encyclopedia_sample": output_dir / "encyclopedia_sample.csv",
        "data_dictionary": output_dir / "data_dictionary.csv",
        "pipeline_summary": output_dir / "pipeline_summary.json",
    }

    save_sample_csv(raw_df, generated_files["raw_sample"], sample_size)
    save_sample_csv(clean_df, generated_files["clean_sample"], sample_size)
    save_sample_csv(features_df, generated_files["features_sample"], sample_size)
    save_sample_csv(encyclopedia_df, generated_files["encyclopedia_sample"], sample_size)

    data_dictionary_df = build_data_dictionary(
        {
            "raw": raw_df,
            "clean": clean_df,
            "features": features_df,
            "encyclopedia": encyclopedia_df,
        }
    )
    data_dictionary_df.to_csv(generated_files["data_dictionary"], index=False)

    summary = build_pipeline_summary(
        raw_df=raw_df,
        clean_df=clean_df,
        features_df=features_df,
        encyclopedia_df=encyclopedia_df,
        parameters=parameters,
        generated_files=generated_files,
    )

    with open(generated_files["pipeline_summary"], "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    return generated_files


def save_sample_csv(
    df: pd.DataFrame,
    output_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> None:
    """
    Guarda una muestra pequeña de un DataFrame en CSV.
    """
    sample_df = get_stable_sample(df, sample_size)
    export_df = make_csv_friendly(sample_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False, encoding="utf-8")


def get_stable_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Obtiene una muestra estable y reproducible.
    """
    if df.empty:
        return df.copy()

    if len(df) <= sample_size:
        return df.copy()

    return df.sample(n=sample_size, random_state=42).reset_index(drop=True)


def make_csv_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte valores complejos en texto para exportación CSV.
    """
    export_df = df.copy()

    for column in export_df.columns:
        export_df[column] = export_df[column].apply(convert_value_for_csv)

    return export_df


def convert_value_for_csv(value: Any) -> Any:
    """
    Convierte listas, diccionarios y sets en JSON string.
    """
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    return value


def build_data_dictionary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye un diccionario de datos automático para comparar etapas.
    """
    rows = []

    for stage_name, df in datasets.items():
        for column in df.columns:
            series = df[column]
            rows.append(
                {
                    "stage": stage_name,
                    "column": column,
                    "dtype": str(series.dtype),
                    "rows": int(len(series)),
                    "non_null_count": int(series.notna().sum()),
                    "null_count": int(series.isna().sum()),
                    "unique_count_sample": int(series.astype(str).nunique()),
                    "example_value": get_example_value(series),
                }
            )

    return pd.DataFrame(rows)


def get_example_value(series: pd.Series) -> str | None:
    """
    Devuelve un ejemplo no vacío para una columna.
    """
    non_null_values = series.dropna()

    if non_null_values.empty:
        return None

    return str(convert_value_for_csv(non_null_values.iloc[0]))[:500]


def build_pipeline_summary(
    *,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    features_df: pd.DataFrame,
    encyclopedia_df: pd.DataFrame,
    parameters: dict[str, Any],
    generated_files: dict[str, Path],
) -> dict[str, Any]:
    """
    Construye un resumen JSON del pipeline.
    """
    raw_rows = len(raw_df)
    clean_rows = len(clean_df)
    features_rows = len(features_df)

    return {
        "parameters": parameters,
        "stages": {
            "raw": describe_dataframe(raw_df),
            "clean": describe_dataframe(clean_df),
            "features": describe_dataframe(features_df),
            "encyclopedia": describe_dataframe(encyclopedia_df),
        },
        "retention": {
            "clean_vs_raw": calculate_ratio(clean_rows, raw_rows),
            "features_vs_clean": calculate_ratio(features_rows, clean_rows),
            "species_vs_features_rows": calculate_ratio(len(encyclopedia_df), features_rows),
        },
        "generated_sample_files": {
            name: str(path.as_posix()) for name, path in generated_files.items()
        },
        "explanation": {
            "raw": "Datos originales devueltos por la API de GBIF.",
            "clean": "Datos después de eliminar registros incompletos, coordenadas inválidas y clases demasiado pequeñas.",
            "features": "Datos con columnas nuevas preparadas para la modelo.",
            "encyclopedia": "Tabla final agregada: una fila por especie.",
        },
    }


def describe_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Describe tamaño y columnas de un DataFrame.
    """
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
    }


def calculate_ratio(numerator: int, denominator: int) -> float | None:
    """
    Calcula ratio seguro.
    """
    if denominator == 0:
        return None

    return round(numerator / denominator, 4)
