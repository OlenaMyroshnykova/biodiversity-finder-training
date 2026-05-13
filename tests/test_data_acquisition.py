"""Tests for neutral GBIF acquisition."""

from src.data_acquisition import GBIFQuery, build_global_query_plan, is_global_country


def test_is_global_country_accepts_global_values() -> None:
    assert is_global_country("GLOBAL")
    assert is_global_country("ALL")
    assert is_global_country("")
    assert is_global_country(None)


def test_is_global_country_rejects_country_code() -> None:
    assert not is_global_country("ES")
    assert not is_global_country("FR")


def test_build_global_query_plan_uses_neutral_taxonomic_sources(monkeypatch) -> None:
    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)

    query_plan = build_global_query_plan("GLOBAL")
    source_queries = {query.source_query for query in query_plan}

    assert "global_background" in source_queries
    assert "class_mammalia" in source_queries
    assert "class_aves" in source_queries
    assert "class_reptilia" in source_queries
    assert "class_amphibia" in source_queries
    assert "class_insecta" in source_queries
    assert "class_actinopterygii" in source_queries
    assert "class_chondrichthyes" in source_queries
    assert "class_arachnida" in source_queries
    assert "kingdom_plantae" in source_queries
    assert "kingdom_fungi" in source_queries
    assert all(isinstance(query, GBIFQuery) for query in query_plan)


def test_query_plan_has_no_demo_species_or_vibe_queries(monkeypatch) -> None:
    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)

    query_plan = build_global_query_plan("GLOBAL")
    plan_text = " ".join(
        f"{query.source_query} {query.description} {query.params}".lower()
        for query in query_plan
    )

    forbidden_terms = [
        "flamingo",
        "phoenicopterus",
        "polar_bear",
        "ursus maritimus",
        "jaguar",
        "panthera onca",
        "big_cats",
        "pink_bird",
        "ave rosa",
        "animal polar",
        "large_savanna",
        "small_desert",
    ]
    for term in forbidden_terms:
        assert term not in plan_text


def test_query_plan_shares_sum_to_one(monkeypatch) -> None:
    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)

    query_plan = build_global_query_plan("GLOBAL")
    total_share = sum(query.share for query in query_plan)

    assert round(total_share, 2) == 1.00
