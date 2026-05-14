import pandas as pd

from src.image_enrichment import (
    add_images_to_encyclopedia,
    build_image_search_names,
    build_image_lookup_table,
    is_valid_image_url,
    remove_authorship,
)


def test_remove_authorship_keeps_trinomial_and_binomial_fallback():
    assert remove_authorship("Panthera leo melanochaita (C.E.H.Smith, 1858)") == "Panthera leo melanochaita"
    row = pd.Series({"scientific_name": "Panthera leo melanochaita (C.E.H.Smith, 1858)"})
    names = build_image_search_names(row)
    assert names[0] == "Panthera leo melanochaita (C.E.H.Smith, 1858)"
    assert "Panthera leo melanochaita" in names
    assert "Panthera leo" in names


def test_image_lookup_selects_top_offline_species_by_observations():
    df = pd.DataFrame(
        [
            {"scientific_name": "Low", "canonical_scientific_name": "Low", "observations": 1},
            {"scientific_name": "High", "canonical_scientific_name": "High", "observations": 100},
        ]
    )
    lookup = build_image_lookup_table(df, max_species=1)
    assert lookup.iloc[0]["scientific_name"] == "High"


def test_add_images_preserves_existing_valid_image_without_api():
    df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo",
                "canonical_scientific_name": "Panthera leo",
                "observations": 10,
                "image_url": "https://example.org/lion.jpg",
            }
        ]
    )
    enriched, image_metadata = add_images_to_encyclopedia(df, use_api=False)
    assert enriched.loc[0, "has_image"] is True or bool(enriched.loc[0, "has_image"])
    assert enriched.loc[0, "image_url"] == "https://example.org/lion.jpg"
    assert image_metadata.empty


def test_bad_placeholder_images_are_rejected():
    assert not is_valid_image_url("https://example.org/placeholder.svg")
    assert not is_valid_image_url("https://example.org/range_map.jpg")
    assert is_valid_image_url("https://example.org/species_photo.jpg")
