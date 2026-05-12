"""Tests para nombres comunes desde GBIF + Wikidata."""

import pandas as pd

from src.vernacular_names import (
    VernacularNameRecord,
    add_vernacular_names_to_encyclopedia,
    build_wikidata_query,
    canonicalize_scientific_name,
    combine_vernacular_sources,
    fetch_wikidata_vernacular_names,
    normalize_language_code,
    summarize_vernacular_names,
)


class FakeResponse:
    """Respuesta falsa para requests."""

    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        """No hace nada."""

    def json(self) -> dict:
        """Devuelve payload."""
        return self.payload


def test_canonicalize_scientific_name_removes_authorship() -> None:
    """Debe quitar autoría del nombre científico."""
    assert canonicalize_scientific_name("Panthera leo (Linnaeus, 1758)") == "Panthera leo"


def test_wikidata_query_contains_scientific_name() -> None:
    """La query debe buscar por P225."""
    query = build_wikidata_query("Panthera leo")

    assert 'wdt:P225 "Panthera leo"' in query
    assert "wdt:P1843" in query
    assert "skos:altLabel" in query


def test_normalize_language_code() -> None:
    """Debe convertir ISO 639-1 a ISO 639-3."""
    assert normalize_language_code("es") == "spa"
    assert normalize_language_code("en") == "eng"
    assert normalize_language_code("ru") == "rus"


def test_fetch_wikidata_vernacular_names_parses_common_names(monkeypatch) -> None:
    """Debe leer nombres comunes, labels y aliases de Wikidata."""
    def fake_get(url, params=None, headers=None, timeout=None):
        return FakeResponse(
            {
                "results": {
                    "bindings": [
                        {
                            "commonName": {
                                "type": "literal",
                                "value": "León",
                                "xml:lang": "es",
                            },
                            "label": {
                                "type": "literal",
                                "value": "lion",
                                "xml:lang": "en",
                            },
                            "alias": {
                                "type": "literal",
                                "value": "Лев",
                                "xml:lang": "ru",
                            },
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr("src.vernacular_names.requests.get", fake_get)

    records = fetch_wikidata_vernacular_names(
        scientific_name="Panthera leo (Linnaeus, 1758)",
        canonical_scientific_name="Panthera leo",
        species_key="",
    )

    names = {record.vernacular_name for record in records}
    sources = {record.source for record in records}

    assert "León" in names
    assert "lion" in names
    assert "Лев" in names
    assert "Wikidata P1843" in sources
    assert "Wikidata label" in sources
    assert "Wikidata alias" in sources


def test_combine_vernacular_sources_uses_concat() -> None:
    """Debe combinar fuentes de nombres comunes."""
    gbif_df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo",
                "canonical_scientific_name": "Panthera leo",
                "species_key": "1",
                "language": "eng",
                "vernacular_name": "Lion",
                "source": "GBIF Species API",
            }
        ]
    )
    wikidata_df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo",
                "canonical_scientific_name": "Panthera leo",
                "species_key": "1",
                "language": "spa",
                "vernacular_name": "León",
                "source": "Wikidata P1843",
            }
        ]
    )
    fallback_df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo",
                "canonical_scientific_name": "Panthera leo",
                "species_key": "1",
                "language": "lat",
                "vernacular_name": "Panthera leo",
                "source": "scientific_name_fallback",
            }
        ]
    )

    combined_df = combine_vernacular_sources(
        gbif_names_df=gbif_df,
        wikidata_names_df=wikidata_df,
        fallback_names_df=fallback_df,
    )

    assert set(combined_df["vernacular_name"]) == {"Lion", "León", "Panthera leo"}


def test_summarize_vernacular_names_groups_by_canonical_species() -> None:
    """Debe agrupar nombres comunes por especie canónica."""
    names_df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo",
                "canonical_scientific_name": "Panthera leo",
                "species_key": "1",
                "language": "eng",
                "vernacular_name": "Lion",
                "source": "GBIF Species API",
            },
            {
                "scientific_name": "Panthera leo",
                "canonical_scientific_name": "Panthera leo",
                "species_key": "1",
                "language": "spa",
                "vernacular_name": "León",
                "source": "Wikidata P1843",
            },
        ]
    )

    summary_df = summarize_vernacular_names(names_df)

    assert len(summary_df) == 1
    assert "Lion" in summary_df.iloc[0]["vernacular_names"]
    assert "León" in summary_df.iloc[0]["vernacular_names"]


def test_add_vernacular_names_to_encyclopedia_uses_wikidata(monkeypatch) -> None:
    """Debe añadir nombres de Wikidata a la enciclopedia mediante merge."""
    encyclopedia_df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo (Linnaeus, 1758)",
                "search_document": "Panthera leo Felidae",
                "profile_text": "Especie de prueba.",
            }
        ]
    )
    features_df = pd.DataFrame(
        [{"scientific_name": "Panthera leo (Linnaeus, 1758)", "speciesKey": 123}]
    )

    def fake_gbif(*args, **kwargs):
        return []

    def fake_wikidata(scientific_name: str, canonical_scientific_name: str, species_key: str, timeout: int = 30):
        return [
            VernacularNameRecord(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_scientific_name,
                species_key=species_key,
                language="spa",
                vernacular_name="León",
                source="Wikidata P1843",
            ),
            VernacularNameRecord(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_scientific_name,
                species_key=species_key,
                language="eng",
                vernacular_name="Lion",
                source="Wikidata label",
            ),
        ]

    monkeypatch.setattr(
        "src.vernacular_names.fetch_gbif_vernacular_names",
        fake_gbif,
    )
    monkeypatch.setattr(
        "src.vernacular_names.fetch_wikidata_vernacular_names",
        fake_wikidata,
    )

    enriched_df, names_df = add_vernacular_names_to_encyclopedia(
        encyclopedia_df=encyclopedia_df,
        features_df=features_df,
        max_species=10,
        use_api=True,
        use_wikidata=True,
        pause_seconds=0,
    )

    assert not names_df.empty
    assert "León" in enriched_df.iloc[0]["vernacular_names"]
    assert "Lion" in enriched_df.iloc[0]["search_document"]
    assert "Nombres comunes:" in enriched_df.iloc[0]["profile_text"]
