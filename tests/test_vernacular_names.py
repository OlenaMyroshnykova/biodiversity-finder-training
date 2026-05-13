"""Tests para nombres comunes desde GBIF + Wikidata, limitados a ES/EN."""

import pandas as pd

from src.vernacular_names import (
    VernacularNameRecord,
    add_vernacular_names_to_encyclopedia,
    build_wikidata_query,
    build_wikidata_query_by_gbif_key,
    canonicalize_scientific_name,
    clean_species_key,
    combine_vernacular_sources,
    fetch_wikidata_search_names,
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


def test_clean_species_key_removes_float_suffix() -> None:
    """Debe limpiar claves que vienen como float string."""
    assert clean_species_key("5219404.0") == "5219404"
    assert clean_species_key("5219404") == "5219404"


def test_wikidata_query_contains_scientific_name() -> None:
    """La query debe buscar por P225 y limitar idiomas a ES/EN."""
    query = build_wikidata_query("Panthera leo")
    assert 'wdt:P225 "Panthera leo"' in query
    assert "wdt:P1843" in query
    assert "skos:altLabel" in query
    assert '"es"' in query
    assert '"en"' in query
    assert '"ru"' not in query
    assert '"uk"' not in query


def test_wikidata_query_by_gbif_key_contains_p846() -> None:
    """La query debe buscar por GBIF taxon ID P846."""
    query = build_wikidata_query_by_gbif_key("5219404")
    assert 'wdt:P846 "5219404"' in query
    assert "rdfs:label" in query
    assert '"es"' in query
    assert '"en"' in query


def test_normalize_language_code() -> None:
    """Debe convertir ISO 639-1 a ISO 639-3 para ES/EN."""
    assert normalize_language_code("es") == "spa"
    assert normalize_language_code("en") == "eng"
    # Old multilingual values are no longer part of supported common-name collection.
    assert normalize_language_code("ru") == "ru"


def test_fetch_wikidata_vernacular_names_uses_sparql_and_filters_languages(monkeypatch) -> None:
    """Debe leer nombres ES/EN y descartar otros idiomas de Wikidata SPARQL."""

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
        species_key="5219404",
    )

    names = {record.vernacular_name for record in records}
    languages = {record.language for record in records}

    assert "León" in names
    assert "lion" in names
    assert "Лев" not in names
    assert languages <= {"spa", "eng"}


def test_fetch_wikidata_vernacular_names_falls_back_to_search_and_filters_alias_noise(monkeypatch) -> None:
    """Debe usar Search API sin recuperar aliases multilingües antiguos."""

    def fake_get(url, params=None, headers=None, timeout=None):
        if "query.wikidata.org" in url:
            return FakeResponse({"results": {"bindings": []}})
        return FakeResponse(
            {
                "search": [
                    {
                        "label": "lion",
                        "description": "species of mammal",
                        "aliases": ["León", "Лев"],
                    }
                ]
            }
        )

    monkeypatch.setattr("src.vernacular_names.requests.get", fake_get)

    records = fetch_wikidata_vernacular_names(
        scientific_name="Panthera leo (Linnaeus, 1758)",
        canonical_scientific_name="Panthera leo",
        species_key="5219404",
    )

    names = {record.vernacular_name for record in records}
    assert "lion" in names
    assert "León" in names
    assert "Лев" not in names
    assert {record.language for record in records} <= {"spa", "eng"}


def test_fetch_wikidata_search_names_filters_taxon_results(monkeypatch) -> None:
    """Debe leer nombres desde Search API en ES/EN."""

    def fake_get(url, params=None, headers=None, timeout=None):
        return FakeResponse(
            {
                "search": [
                    {
                        "label": "lion",
                        "description": "species of mammal",
                        "aliases": ["León"],
                    }
                ]
            }
        )

    monkeypatch.setattr("src.vernacular_names.requests.get", fake_get)

    records = fetch_wikidata_search_names(
        scientific_name="Panthera leo (Linnaeus, 1758)",
        canonical_scientific_name="Panthera leo",
        species_key="5219404",
    )
    names = {record.vernacular_name for record in records}

    assert "lion" in names
    assert "León" in names


def test_combine_vernacular_sources_uses_concat_and_drops_unsupported_languages() -> None:
    """Debe combinar fuentes de nombres comunes sin ruido multilingüe."""
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
            },
            {
                "scientific_name": "Panthera leo",
                "canonical_scientific_name": "Panthera leo",
                "species_key": "1",
                "language": "rus",
                "vernacular_name": "Лев",
                "source": "Wikidata alias",
            },
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
    assert "rus" not in set(combined_df["language"])


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
            {
                "scientific_name": "Panthera leo",
                "canonical_scientific_name": "Panthera leo",
                "species_key": "1",
                "language": "rus",
                "vernacular_name": "Лев",
                "source": "Wikidata alias",
            },
        ]
    )

    summary_df = summarize_vernacular_names(names_df)

    assert len(summary_df) == 1
    assert "Lion" in summary_df.iloc[0]["vernacular_names"]
    assert "León" in summary_df.iloc[0]["vernacular_names"]
    assert "Лев" not in summary_df.iloc[0]["vernacular_names"]


def test_add_vernacular_names_to_encyclopedia_uses_wikidata(monkeypatch) -> None:
    """Debe añadir nombres ES/EN a la enciclopedia mediante merge."""
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
        [{"scientific_name": "Panthera leo (Linnaeus, 1758)", "speciesKey": 5219404}]
    )

    def fake_gbif(*args, **kwargs):
        return []

    def fake_wikidata(
        scientific_name: str,
        canonical_scientific_name: str,
        species_key: str,
        timeout: int = 30,
    ):
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
            VernacularNameRecord(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_scientific_name,
                species_key=species_key,
                language="rus",
                vernacular_name="Лев",
                source="Wikidata alias",
            ),
        ]

    monkeypatch.setattr("src.vernacular_names.fetch_gbif_vernacular_names", fake_gbif)
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
    assert "Лев" not in enriched_df.iloc[0]["vernacular_names"]
    assert "Nombres comunes:" in enriched_df.iloc[0]["profile_text"]
