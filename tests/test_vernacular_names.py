"""Tests para nombres comunes y df.merge()."""

import pandas as pd

from src.vernacular_names import (
    VernacularNameRecord,
    add_vernacular_names_to_encyclopedia,
    build_species_lookup,
    summarize_vernacular_names,
)


def test_build_species_lookup_uses_species_key() -> None:
    """Debe crear lookup species -> species_key."""
    encyclopedia_df = pd.DataFrame(
        [{"scientific_name": "Panthera pardus"}]
    )
    features_df = pd.DataFrame(
        [{"scientific_name": "Panthera pardus", "speciesKey": 5219404}]
    )

    lookup_df = build_species_lookup(encyclopedia_df, features_df)

    assert lookup_df.iloc[0]["scientific_name"] == "Panthera pardus"
    assert str(lookup_df.iloc[0]["species_key"]) == "5219404"


def test_summarize_vernacular_names_groups_by_species() -> None:
    """Debe agrupar nombres comunes por especie."""
    names_df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera pardus",
                "species_key": "5219404",
                "language": "eng",
                "vernacular_name": "Leopard",
                "source": "test",
            },
            {
                "scientific_name": "Panthera pardus",
                "species_key": "5219404",
                "language": "spa",
                "vernacular_name": "Leopardo",
                "source": "test",
            },
        ]
    )

    summary_df = summarize_vernacular_names(names_df)

    assert len(summary_df) == 1
    assert "Leopard" in summary_df.iloc[0]["vernacular_names"]
    assert "Leopardo" in summary_df.iloc[0]["vernacular_names"]


def test_add_vernacular_names_to_encyclopedia_uses_merge(monkeypatch) -> None:
    """Debe añadir vernacular_names a la enciclopedia y search_document."""
    encyclopedia_df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera pardus",
                "search_document": "Panthera pardus Felidae",
                "profile_text": "Especie de prueba.",
            }
        ]
    )
    features_df = pd.DataFrame(
        [{"scientific_name": "Panthera pardus", "speciesKey": 5219404}]
    )

    def fake_fetch(scientific_name: str, species_key: str, timeout: int = 20):
        return [
            VernacularNameRecord(
                scientific_name=scientific_name,
                species_key=species_key,
                language="eng",
                vernacular_name="Leopard",
                source="GBIF Species API",
            ),
            VernacularNameRecord(
                scientific_name=scientific_name,
                species_key=species_key,
                language="spa",
                vernacular_name="Leopardo",
                source="GBIF Species API",
            ),
        ]

    monkeypatch.setattr(
        "src.vernacular_names.fetch_gbif_vernacular_names",
        fake_fetch,
    )

    enriched_df, names_df = add_vernacular_names_to_encyclopedia(
        encyclopedia_df=encyclopedia_df,
        features_df=features_df,
        max_species=10,
        use_api=True,
    )

    assert not names_df.empty
    assert "vernacular_names" in enriched_df.columns
    assert "Leopard" in enriched_df.iloc[0]["vernacular_names"]
    assert "Leopardo" in enriched_df.iloc[0]["search_document"]
    assert "Nombres comunes:" in enriched_df.iloc[0]["profile_text"]


def test_add_vernacular_names_fallback_without_api() -> None:
    """Debe funcionar sin API usando nombre científico."""
    encyclopedia_df = pd.DataFrame(
        [
            {
                "scientific_name": "Rosa canina",
                "search_document": "Rosa canina Plantae",
                "profile_text": "Planta de prueba.",
            }
        ]
    )
    features_df = pd.DataFrame(
        [{"scientific_name": "Rosa canina"}]
    )

    enriched_df, names_df = add_vernacular_names_to_encyclopedia(
        encyclopedia_df=encyclopedia_df,
        features_df=features_df,
        max_species=10,
        use_api=False,
    )

    assert not names_df.empty
    assert "Rosa canina" in enriched_df.iloc[0]["vernacular_names"]
    assert "Rosa canina" in enriched_df.iloc[0]["search_document"]
