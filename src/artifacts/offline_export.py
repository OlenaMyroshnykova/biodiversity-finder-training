"""Light export for offline/demo mode."""
from __future__ import annotations

import pandas as pd

OFFLINE_ENCYCLOPEDIA_COLUMNS = [
    "scientific_name",
    "canonical_scientific_name",
    "vernacular_names",
    "vernacular_sources",
    "kingdom",
    "taxon_class",
    "taxon_order",
    "family",
    "genus",
    "observations",
    "countries",
    "avg_latitude",
    "avg_longitude",
    "image_url",
    "image_source",
    "has_image",
    "iucn_category",
    "iucn_status_label",
    "iucn_source",
    "iucn_is_official",
    "conservation_status",
    "conservation_category",
    "conservation_source",
    "is_threatened",
    "conservation_note",
    "color_tag",
    "habitat_tag",
    "size_tag",
    "tags_de_busqueda",
    "profile_text",
    "search_document",
]


def build_offline_encyclopedia(
    encyclopedia_df: pd.DataFrame,
    *,
    max_species: int = 2000,
) -> pd.DataFrame:
    """Create a compressed light encyclopedia for offline/demo use."""
    if encyclopedia_df.empty:
        return encyclopedia_df.copy()
    sorted_df = encyclopedia_df.sort_values("observations", ascending=False).copy()
    light_df = sorted_df.head(max_species).copy()
    existing_columns = [column for column in OFFLINE_ENCYCLOPEDIA_COLUMNS if column in light_df.columns]
    return light_df[existing_columns].reset_index(drop=True)


def build_offline_occurrence_points(
    occurrence_points_df: pd.DataFrame,
    *,
    max_total_points: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a light version of occurrence points for Folium maps."""
    if occurrence_points_df.empty:
        return occurrence_points_df.copy()
    if len(occurrence_points_df) <= max_total_points:
        return occurrence_points_df.reset_index(drop=True)
    return occurrence_points_df.sample(n=max_total_points, random_state=random_state).reset_index(drop=True)
