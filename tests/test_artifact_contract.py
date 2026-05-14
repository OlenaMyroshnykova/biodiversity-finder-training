import pandas as pd

from src.artifact_contract import ensure_artifact_contract
from src.search_tags import add_search_tags_to_encyclopedia


def test_artifact_contract_builds_search_document_from_names_taxonomy_and_tags():
    df = pd.DataFrame(
        [
            {
                "scientific_name": "Ursus arctos (Linnaeus, 1758)",
                "canonical_scientific_name": "Ursus arctos",
                "vernacular_names": "oso pardo | brown bear",
                "kingdom": "Animalia",
                "taxon_class": "Mammalia",
                "family": "Ursidae",
                "tags_de_busqueda": "brown forest large marron bosque grande",
                "iucn_category": "LC",
            }
        ]
    )

    result = ensure_artifact_contract(df)
    document = result.iloc[0]["search_document"]

    assert "oso pardo" in document
    assert "brown bear" in document
    assert "ursidae" in document
    assert "bosque" in document


def test_search_tags_keep_required_vibe_column_and_contract_document():
    df = pd.DataFrame(
        [
            {
                "scientific_name": "Aquila test",
                "canonical_scientific_name": "Aquila test",
                "vernacular_names": "águila | eagle",
                "kingdom": "Animalia",
                "taxon_class": "Aves",
                "family": "Accipitridae",
            }
        ]
    )

    result = add_search_tags_to_encyclopedia(df)

    assert "tags_de_busqueda" in result.columns
    assert "search_document" in result.columns
    assert "ave" in result.iloc[0]["search_document"] or "aves" in result.iloc[0]["search_document"]
