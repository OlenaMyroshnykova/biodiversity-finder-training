"""Tests para descarga multi-source desde GBIF."""

from src.data_acquisition import (
    GBIFQuery,
    build_global_query_plan,
    is_global_country,
)


def test_is_global_country_accepts_global_values() -> None:
    """Debe detectar valores globales."""
    assert is_global_country("GLOBAL")
    assert is_global_country("ALL")
    assert is_global_country("")
    assert is_global_country(None)


def test_is_global_country_rejects_country_code() -> None:
    """Debe detectar códigos de país como no globales."""
    assert not is_global_country("ES")
    assert not is_global_country("FR")


def test_build_global_query_plan_contains_required_sources(monkeypatch) -> None:
    """El plan debe contener fuentes temáticas importantes."""
    monkeypatch.setattr(
        "src.data_acquisition.match_gbif_taxon_key",
        lambda name, rank=None: 123,
    )

    query_plan = build_global_query_plan("GLOBAL")
    source_queries = {query.source_query for query in query_plan}

    assert "general_global" in source_queries
    assert "flamingo_pink_bird" in source_queries
    assert "polar_bear" in source_queries
    assert "butterflies_lepidoptera" in source_queries
    assert "amphibians" in source_queries
    assert all(isinstance(query, GBIFQuery) for query in query_plan)
