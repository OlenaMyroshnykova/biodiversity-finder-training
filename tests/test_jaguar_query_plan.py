"""Tests para consultas específicas de jaguar y felinos."""

from src.data_acquisition import build_global_query_plan


def test_global_query_plan_contains_jaguar_and_felidae(monkeypatch) -> None:
    """El plan debe incluir consultas para jaguar y felinos."""
    monkeypatch.setattr(
        "src.data_acquisition.match_gbif_taxon_key",
        lambda name, rank=None: 123,
    )

    query_plan = build_global_query_plan("GLOBAL")
    source_queries = {query.source_query for query in query_plan}

    assert "jaguar_panthera_onca" in source_queries
    assert "big_cats_felidae" in source_queries


def test_query_plan_shares_sum_to_one(monkeypatch) -> None:
    """Las proporciones deben sumar aproximadamente 1."""
    monkeypatch.setattr(
        "src.data_acquisition.match_gbif_taxon_key",
        lambda name, rank=None: 123,
    )

    query_plan = build_global_query_plan("GLOBAL")
    total_share = sum(query.share for query in query_plan)

    assert round(total_share, 2) == 1.00
