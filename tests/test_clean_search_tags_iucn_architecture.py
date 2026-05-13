"""Tests for clean vibe tags and IUCN fields in the final dataset."""

from __future__ import annotations

import pandas as pd

from src.search_tags import add_search_tags_to_encyclopedia


def test_tags_de_busqueda_contains_only_vibe_tags() -> None:
    """Search tags must not include names, taxonomy, source queries or Wikidata noise."""

    df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo (Linnaeus, 1758)",
                "canonical_scientific_name": "Panthera leo",
                "vernacular_names": "Lion | León | Löwe | лев",
                "kingdom": "Animalia",
                "taxon_class": "Mammalia",
                "taxon_order": "Carnivora",
                "family": "Felidae",
                "genus": "Panthera",
                "source_queries": "big_cats_felidae",
                "profile_text": "Large cat of the savanna.",
                "search_document": "Panthera leo Lion León Felidae",
            },
            {
                "scientific_name": "Papilio machaon Linnaeus, 1758",
                "canonical_scientific_name": "Papilio machaon",
                "vernacular_names": "Swallowtail butterfly | mariposa",
                "kingdom": "Animalia",
                "taxon_class": "Insecta",
                "taxon_order": "Lepidoptera",
                "family": "Papilionidae",
                "genus": "Papilio",
                "source_queries": "butterflies_lepidoptera",
                "profile_text": "Butterfly.",
                "search_document": "Papilio butterfly mariposa",
            },
        ]
    )

    tagged_df = add_search_tags_to_encyclopedia(df)

    lion_tags = tagged_df.iloc[0]["tags_de_busqueda"]
    butterfly_tags = tagged_df.iloc[1]["tags_de_busqueda"]

    assert "large" in lion_tags
    assert "savanna" in lion_tags
    assert "panthera" not in lion_tags
    assert "lion" not in lion_tags
    assert "felidae" not in lion_tags
    assert "big_cats" not in lion_tags

    assert "small" in butterfly_tags
    assert "colorful" in butterfly_tags
    assert "butterfly" not in butterfly_tags
    assert "papilio" not in butterfly_tags
    assert "lepidoptera" not in butterfly_tags


def test_search_document_is_kept_for_fallback_without_polluting_tags() -> None:
    """Common names can stay in search_document but not tags_de_busqueda."""

    df = pd.DataFrame(
        [
            {
                "scientific_name": "Papilio machaon",
                "vernacular_names": "Swallowtail butterfly",
                "taxon_class": "Insecta",
                "family": "Papilionidae",
                "profile_text": "Butterfly.",
            }
        ]
    )

    tagged_df = add_search_tags_to_encyclopedia(df)

    assert "butterfly" in tagged_df.iloc[0]["search_document"].lower()
    assert "butterfly" not in tagged_df.iloc[0]["tags_de_busqueda"].lower()
