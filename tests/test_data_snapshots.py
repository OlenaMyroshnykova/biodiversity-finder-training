"""Tests para muestras inspeccionables del pipeline."""

from pathlib import Path

import pandas as pd

from src.data_snapshots import (
    build_data_dictionary,
    build_pipeline_summary,
    save_pipeline_snapshots,
)


def build_small_dataframe() -> pd.DataFrame:
    """Crea un DataFrame mínimo para tests."""
    return pd.DataFrame(
        [
            {
                "scientificName": "Phoenicopterus roseus",
                "kingdom": "Animalia",
                "class": "Aves",
                "family": "Phoenicopteridae",
                "decimalLatitude": 38.0,
                "decimalLongitude": -0.5,
                "source_query": "flamingo_pink_bird",
            },
            {
                "scientificName": "Ursus maritimus",
                "kingdom": "Animalia",
                "class": "Mammalia",
                "family": "Ursidae",
                "decimalLatitude": 70.0,
                "decimalLongitude": -40.0,
                "source_query": "polar_bear",
            },
        ]
    )


def test_build_data_dictionary_contains_stages() -> None:
    """Debe crear diccionario de datos con nombre de etapa."""
    df = build_small_dataframe()

    dictionary_df = build_data_dictionary({"raw": df, "clean": df})

    assert set(dictionary_df["stage"]) == {"raw", "clean"}
    assert "scientificName" in set(dictionary_df["column"])


def test_build_pipeline_summary_contains_retention() -> None:
    """Debe calcular ratios de retención."""
    df = build_small_dataframe()

    summary = build_pipeline_summary(
        raw_df=df,
        clean_df=df.head(1),
        features_df=df.head(1),
        encyclopedia_df=df.head(1),
        parameters={"country": "GLOBAL"},
        generated_files={"raw_sample": Path("raw_sample.csv")},
    )

    assert summary["parameters"]["country"] == "GLOBAL"
    assert summary["retention"]["clean_vs_raw"] == 0.5


def test_save_pipeline_snapshots_creates_files(tmp_path: Path) -> None:
    """Debe crear CSV y JSON de muestras."""
    df = build_small_dataframe()

    generated_files = save_pipeline_snapshots(
        raw_df=df,
        clean_df=df,
        features_df=df,
        encyclopedia_df=df,
        output_dir=tmp_path,
        parameters={"country": "GLOBAL"},
        sample_size=1,
    )

    for path in generated_files.values():
        assert path.exists()

    raw_sample_df = pd.read_csv(generated_files["raw_sample"])

    assert len(raw_sample_df) == 1
    assert "scientificName" in raw_sample_df.columns
