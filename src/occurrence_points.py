"""Generación de puntos de avistamiento para mapas Folium."""

from __future__ import annotations

import pandas as pd


def build_species_occurrence_points(
    features_df: pd.DataFrame,
    *,
    max_points_per_species: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Construye tabla ligera de coordenadas por especie."""
    required_columns = ["scientific_name", "decimalLatitude", "decimalLongitude"]
    missing_columns = [column for column in required_columns if column not in features_df.columns]

    if missing_columns:
        return pd.DataFrame(
            columns=[
                "scientific_name",
                "canonical_scientific_name",
                "decimalLatitude",
                "decimalLongitude",
                "countryCode",
                "eventDate",
            ]
        )

    points_df = features_df.dropna(subset=["decimalLatitude", "decimalLongitude"]).copy()

    if points_df.empty:
        return pd.DataFrame(
            columns=[
                "scientific_name",
                "canonical_scientific_name",
                "decimalLatitude",
                "decimalLongitude",
                "countryCode",
                "eventDate",
            ]
        )

    if "canonical_scientific_name" not in points_df.columns:
        points_df["canonical_scientific_name"] = points_df["scientific_name"].astype(str)

    optional_columns = [
        "countryCode",
        "eventDate",
        "taxon_class",
        "family",
    ]
    selected_columns = [
        "scientific_name",
        "canonical_scientific_name",
        "decimalLatitude",
        "decimalLongitude",
    ] + [column for column in optional_columns if column in points_df.columns]

    points_df = points_df[selected_columns].copy()
    points_df = points_df.drop_duplicates(
        subset=["canonical_scientific_name", "decimalLatitude", "decimalLongitude"]
    )

    points_df = (
        points_df
        .groupby("canonical_scientific_name", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), max_points_per_species),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )

    return points_df
