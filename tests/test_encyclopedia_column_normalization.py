"""Tests para columnas normalizadas de enciclopedia."""

import pandas as pd

from src.encyclopedia import build_species_encyclopedia


def test_encyclopedia_uses_snake_case_columns_from_clean_pipeline() -> None:
    """Debe usar country_code, basis_of_record y coordenadas normalizadas."""
    features_df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "taxon_class": "Mammalia",
                "taxon_order": "Carnivora",
                "family": "Felidae",
                "genus": "Panthera",
                "species": "Panthera leo",
                "country_code": "KE",
                "basis_of_record": "HUMAN_OBSERVATION",
                "decimal_latitude": -1.2,
                "decimal_longitude": 36.8,
                "year": 2026,
                "season": "invierno",
                "source_query": "big_cats_felidae",
            },
            {
                "scientific_name": "Panthera leo",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "taxon_class": "Mammalia",
                "taxon_order": "Carnivora",
                "family": "Felidae",
                "genus": "Panthera",
                "species": "Panthera leo",
                "country_code": "TZ",
                "basis_of_record": "HUMAN_OBSERVATION",
                "decimal_latitude": -2.0,
                "decimal_longitude": 35.0,
                "year": 2026,
                "season": "verano",
                "source_query": "big_cats_felidae",
            },
        ]
    )

    encyclopedia_df = build_species_encyclopedia(features_df)
    row = encyclopedia_df.iloc[0]

    assert row["countries"] == "KE, TZ"
    assert row["most_common_basis"] == "HUMAN_OBSERVATION"
    assert row["avg_latitude"] == (-1.2 + -2.0) / 2
    assert row["avg_longitude"] == (36.8 + 35.0) / 2


def test_encyclopedia_still_accepts_camel_case_gbif_columns() -> None:
    """Debe seguir aceptando nombres originales de GBIF."""
    features_df = pd.DataFrame(
        [
            {
                "scientificName": "Rana temporaria",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Amphibia",
                "order": "Anura",
                "family": "Ranidae",
                "genus": "Rana",
                "species": "Rana temporaria",
                "countryCode": "ES",
                "basisOfRecord": "PRESERVED_SPECIMEN",
                "decimalLatitude": 40.0,
                "decimalLongitude": -3.0,
                "year": 2025,
                "season": "primavera",
                "source_query": "amphibians",
            }
        ]
    )

    encyclopedia_df = build_species_encyclopedia(features_df)
    row = encyclopedia_df.iloc[0]

    assert row["scientific_name"] == "Rana temporaria"
    assert row["countries"] == "ES"
    assert row["most_common_basis"] == "PRESERVED_SPECIMEN"
    assert row["avg_latitude"] == 40.0
    assert row["avg_longitude"] == -3.0
