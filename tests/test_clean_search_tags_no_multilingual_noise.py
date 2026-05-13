from __future__ import annotations

import pandas as pd

from src.search_tags import add_search_tags_to_encyclopedia


def test_tags_de_busqueda_contains_only_vibe_tags():
    df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo",
                "vernacular_names": "lion|león|лев|лев африканський|leone",
                "kingdom": "Animalia",
                "taxon_class": "Mammalia",
                "taxon_order": "Carnivora",
                "family": "Felidae",
                "source_queries": "general global flamingo polar bear",
            }
        ]
    )

    result = add_search_tags_to_encyclopedia(df)
    tags = result.loc[0, "tags_de_busqueda"]

    assert "brown" in tags
    assert "savanna" in tags
    assert "large" in tags

    forbidden_fragments = [
        "panthera",
        "leo",
        "lion",
        "león",
        "лев",
        "leone",
        "animalia",
        "mammalia",
        "felidae",
        "flamingo",
        "polar",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in tags


def test_search_document_can_keep_names_but_tags_stay_clean():
    df = pd.DataFrame(
        [
            {
                "scientific_name": "Phoenicopterus roseus",
                "vernacular_names": "flamingo|flamenco|фламинго",
                "kingdom": "Animalia",
                "taxon_class": "Aves",
                "family": "Phoenicopteridae",
                "search_document": "existing document",
            }
        ]
    )

    result = add_search_tags_to_encyclopedia(df)

    assert "flamingo" not in result.loc[0, "tags_de_busqueda"]
    assert "фламинго" not in result.loc[0, "tags_de_busqueda"]
    assert "phoenicopterus" not in result.loc[0, "tags_de_busqueda"]

    assert "Phoenicopterus roseus" in result.loc[0, "search_document"]
    assert "flamingo" in result.loc[0, "search_document"]
