"""Legacy guard: the acquisition plan must not contain hand-picked demo species."""

from src.data_acquisition import build_global_query_plan


def test_global_query_plan_does_not_contain_jaguar_or_felidae_demo_queries(monkeypatch) -> None:
    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)

    query_plan = build_global_query_plan("GLOBAL")
    source_queries = {query.source_query for query in query_plan}

    assert "jaguar_panthera_onca" not in source_queries
    assert "big_cats_felidae" not in source_queries
    assert "class_mammalia" in source_queries


def test_query_plan_shares_sum_to_one(monkeypatch) -> None:
    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)

    query_plan = build_global_query_plan("GLOBAL")
    total_share = sum(query.share for query in query_plan)

    assert round(total_share, 2) == 1.00


def test_query_plan_contains_broad_taxonomic_groups(monkeypatch) -> None:
    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)

    query_plan = build_global_query_plan("GLOBAL")
    source_queries = {query.source_query for query in query_plan}

    assert "class_reptilia" in source_queries
    assert "class_actinopterygii" in source_queries
    assert "class_chondrichthyes" in source_queries
    assert "class_arachnida" in source_queries
    assert "kingdom_fungi" in source_queries
