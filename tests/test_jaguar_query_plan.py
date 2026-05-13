"""Tests de compatibilidad: el plan ya no debe estar anclado a jaguar/felinos."""

from src.data_acquisition import build_global_query_plan


def test_global_query_plan_does_not_use_curated_jaguar_or_big_cat_queries(monkeypatch) -> None:
    """El plan debe evitar consultas preparadas para especies bonitas de demo."""
    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)
    query_plan = build_global_query_plan("GLOBAL")
    source_queries = {query.source_query for query in query_plan}

    assert "jaguar_panthera_onca" not in source_queries
    assert "big_cats_felidae" not in source_queries
    assert "class_mammalia" in source_queries


def test_query_plan_shares_sum_to_one(monkeypatch) -> None:
    """Las proporciones deben sumar aproximadamente 1."""
    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)
    query_plan = build_global_query_plan("GLOBAL")
    total_share = sum(query.share for query in query_plan)
    assert round(total_share, 2) == 1.00
