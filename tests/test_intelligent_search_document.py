"""Tests para documentos de búsqueda enriquecidos."""

import pandas as pd

from src.encyclopedia import build_species_encyclopedia


def test_search_document_adds_human_terms_for_polar_bear() -> None:
    """Debe añadir términos humanos para oso polar."""
    features_df = pd.DataFrame(
        [
            {
                "scientific_name": "Ursus maritimus",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Mammalia",
                "order": "Carnivora",
                "family": "Ursidae",
                "genus": "Ursus",
                "species": "Ursus maritimus",
                "countryCode": "CA",
                "basisOfRecord": "HUMAN_OBSERVATION",
                "season": "invierno",
                "year": 2024,
                "decimalLatitude": 70.0,
                "decimalLongitude": -40.0,
                "source_query": "polar_bear",
            }
        ]
    )

    encyclopedia_df = build_species_encyclopedia(features_df)
    document = encyclopedia_df.iloc[0]["search_document"].lower()

    assert "oso polar" in document
    assert "animal polar" in document
    assert "hielo" in document


def test_search_document_adds_human_terms_for_butterfly() -> None:
    """Debe añadir términos humanos para mariposas."""
    features_df = pd.DataFrame(
        [
            {
                "scientific_name": "Vanessa atalanta",
                "kingdom": "Animalia",
                "phylum": "Arthropoda",
                "class": "Insecta",
                "order": "Lepidoptera",
                "family": "Nymphalidae",
                "genus": "Vanessa",
                "species": "Vanessa atalanta",
                "countryCode": "ES",
                "basisOfRecord": "HUMAN_OBSERVATION",
                "season": "primavera",
                "year": 2024,
                "decimalLatitude": 40.0,
                "decimalLongitude": -3.0,
                "source_query": "butterflies_lepidoptera",
            }
        ]
    )

    encyclopedia_df = build_species_encyclopedia(features_df)
    document = encyclopedia_df.iloc[0]["search_document"].lower()

    assert "taxon_order" not in document
    assert "lepidoptera" in document
    assert "mariposa" in document
    assert "butterfly" in document


def test_search_document_adds_human_terms_for_raptor() -> None:
    """Debe añadir términos humanos para aves rapaces."""
    features_df = pd.DataFrame(
        [
            {
                "scientific_name": "Aquila chrysaetos",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Aves",
                "order": "Accipitriformes",
                "family": "Accipitridae",
                "genus": "Aquila",
                "species": "Aquila chrysaetos",
                "countryCode": "ES",
                "basisOfRecord": "HUMAN_OBSERVATION",
                "season": "primavera",
                "year": 2024,
                "decimalLatitude": 40.0,
                "decimalLongitude": -3.0,
                "source_query": "raptors_accipitridae",
            }
        ]
    )

    encyclopedia_df = build_species_encyclopedia(features_df)
    document = encyclopedia_df.iloc[0]["search_document"].lower()

    assert "ave rapaz" in document
    assert "aguila" in document
    assert "montaña" in document
