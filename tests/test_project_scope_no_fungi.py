"""Tests del scope final: animales y plantas, sin Fungi."""

import pandas as pd

from src.data_cleaning import clean_occurrences, keep_project_scope
from src.encyclopedia import build_human_search_terms, build_species_encyclopedia


def build_raw_scope_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": "1",
                "scientificName": "Amanita muscaria",
                "kingdom": "Fungi",
                "phylum": "Basidiomycota",
                "class": "Agaricomycetes",
                "order": "Agaricales",
                "family": "Amanitaceae",
                "genus": "Amanita",
                "species": "Amanita muscaria",
                "countryCode": "ES",
                "decimalLatitude": 40.0,
                "decimalLongitude": -3.0,
                "year": 2024,
                "month": 5,
                "basisOfRecord": "HUMAN_OBSERVATION",
                "source_query": "legacy_fungi",
            },
            {
                "key": "2",
                "scientificName": "Equus quagga",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Mammalia",
                "order": "Perissodactyla",
                "family": "Equidae",
                "genus": "Equus",
                "species": "Equus quagga",
                "countryCode": "ZA",
                "decimalLatitude": -25.0,
                "decimalLongitude": 28.0,
                "year": 2024,
                "month": 7,
                "basisOfRecord": "HUMAN_OBSERVATION",
                "source_query": "class_mammalia",
            },
            {
                "key": "3",
                "scientificName": "Quercus robur",
                "kingdom": "Plantae",
                "phylum": "Tracheophyta",
                "class": "Magnoliopsida",
                "order": "Fagales",
                "family": "Fagaceae",
                "genus": "Quercus",
                "species": "Quercus robur",
                "countryCode": "ES",
                "decimalLatitude": 42.0,
                "decimalLongitude": -6.0,
                "year": 2024,
                "month": 4,
                "basisOfRecord": "HUMAN_OBSERVATION",
                "source_query": "class_magnoliopsida",
            },
        ]
    )


def test_keep_project_scope_removes_fungi() -> None:
    df = pd.DataFrame(
        {
            "kingdom": ["Animalia", "Plantae", "Fungi", "Bacteria", None],
            "scientific_name": ["animal", "plant", "fungus", "bacteria", "unknown"],
        }
    )

    scoped_df = keep_project_scope(df)

    assert set(scoped_df["kingdom"]) == {"Animalia", "Plantae"}


def test_clean_occurrences_removes_fungi_even_if_gbif_returns_them() -> None:
    cleaned_df = clean_occurrences(build_raw_scope_df(), min_class_records=1)

    assert set(cleaned_df["kingdom"]) == {"Animalia", "Plantae"}
    assert "Agaricomycetes" not in set(cleaned_df["taxon_class"])
    assert "Amanita muscaria" not in set(cleaned_df["scientific_name"])


def test_encyclopedia_scope_removes_fungi_rows() -> None:
    cleaned_df = clean_occurrences(build_raw_scope_df(), min_class_records=1)
    encyclopedia_df = build_species_encyclopedia(cleaned_df)

    assert set(encyclopedia_df["kingdom"]) == {"Animalia", "Plantae"}
    assert "Agaricomycetes" not in set(encyclopedia_df["taxon_class"])


def test_human_search_terms_do_not_add_fungi_vocabulary() -> None:
    terms = build_human_search_terms(
        pd.Series(
            {
                "kingdom": "Fungi",
                "taxon_class": "Agaricomycetes",
                "taxon_order": "Agaricales",
                "family": "Amanitaceae",
            }
        )
    )

    assert terms == ""
