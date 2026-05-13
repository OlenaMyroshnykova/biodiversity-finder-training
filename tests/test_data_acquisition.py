"""Tests para descarga neutral multi-source desde GBIF."""

from src.data_acquisition import GBIFQuery, build_global_query_plan, is_global_country


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


def test_build_global_query_plan_uses_neutral_animal_and_plant_sources(monkeypatch) -> None:
    """El plan debe usar grupos amplios de animales y plantas, no especies demo."""

    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)

    query_plan = build_global_query_plan("GLOBAL")
    source_queries = {query.source_query for query in query_plan}

    assert "class_mammalia" in source_queries
    assert "class_aves" in source_queries
    assert "class_reptilia" in source_queries
    assert "class_amphibia" in source_queries
    assert "class_insecta" in source_queries
    assert "class_arachnida" in source_queries
    assert "class_actinopterygii" in source_queries
    assert "class_magnoliopsida" in source_queries
    assert "class_liliopsida" in source_queries
    assert all(isinstance(query, GBIFQuery) for query in query_plan)

    forbidden = {
        "flamingo_pink_bird",
        "polar_bear",
        "jaguar_panthera_onca",
        "big_cats_felidae",
        "butterflies_lepidoptera",
        "raptors_accipitridae",
        "kingdom_fungi",
        "fungi_mushrooms",
        "class_agaricomycetes",
    }
    assert source_queries.isdisjoint(forbidden)


def test_query_plan_has_no_fungi_or_demo_descriptions(monkeypatch) -> None:
    """El scope del proyecto debe ser animales y plantas, no hongos."""

    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)

    query_plan = build_global_query_plan("GLOBAL")
    plan_text = "\n".join(
        f"{query.source_query} {query.description} {query.params}".lower()
        for query in query_plan
    )

    forbidden_terms = [
        "fungi",
        "hongo",
        "hongos",
        "seta",
        "mushroom",
        "flamingo",
        "phoenicopterus",
        "polar_bear",
        "ursus maritimus",
        "jaguar",
        "panthera onca",
    ]
    for term in forbidden_terms:
        assert term not in plan_text


def test_query_plan_shares_sum_to_one(monkeypatch) -> None:
    """Las proporciones deben sumar aproximadamente 1."""

    monkeypatch.setattr("src.data_acquisition.match_gbif_taxon_key", lambda name, rank=None: 123)

    query_plan = build_global_query_plan("GLOBAL")
    total_share = sum(query.share for query in query_plan)

    assert round(total_share, 2) == 1.00
