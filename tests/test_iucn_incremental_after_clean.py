from pathlib import Path

import pandas as pd

from src import iucn_incremental_after_clean as module
from src.conservation_status import build_official_record


def test_incremental_iucn_queries_only_pending_clean_candidates(monkeypatch, tmp_path: Path):
    df = pd.DataFrame(
        [
            {"scientific_name": "Panthera leo", "kingdom": "Animalia", "taxon_class": "Mammalia"},
            {"scientific_name": "Ursus arctos", "kingdom": "Animalia", "taxon_class": "Mammalia"},
            {"scientific_name": "Panthera leo", "kingdom": "Animalia", "taxon_class": "Mammalia"},
        ]
    )
    cache_path = tmp_path / "iucn_status_cache.csv"
    pd.DataFrame(
        [
            {
                "canonical_scientific_name": "Panthera leo",
                "iucn_category": "VU",
                "iucn_status_label": "Vulnerable",
                "iucn_source": "IUCN Red List",
                "iucn_is_official": True,
                "is_threatened": True,
                "conservation_status": "VU",
                "conservation_category": "Vulnerable",
                "conservation_source": "IUCN Red List",
                "conservation_note": "cached",
            }
        ]
    ).to_csv(cache_path, index=False)

    calls = []

    def fake_fetch(name, token):
        calls.append(name)
        return build_official_record(name, "LC", source="IUCN Red List", note="test")

    monkeypatch.setattr(module, "get_iucn_token", lambda: "token")
    monkeypatch.setattr(module, "fetch_iucn_record_by_scientific_name", fake_fetch)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    enriched_df, cache_df, summary = module.enrich_clean_encyclopedia_with_incremental_iucn(
        df,
        cache_path=cache_path,
        batch_size=10,
        request_delay_seconds=0,
    )

    assert calls == ["Ursus arctos"]
    assert set(cache_df["canonical_scientific_name"]) == {"Panthera leo", "Ursus arctos"}
    assert summary.requested_this_run == 1
    assert "iucn_category" in enriched_df.columns


def test_incremental_iucn_without_token_does_not_poison_cache_with_no_data(monkeypatch, tmp_path: Path):
    df = pd.DataFrame(
        [{"scientific_name": "Ursus arctos", "kingdom": "Animalia", "taxon_class": "Mammalia"}]
    )
    cache_path = tmp_path / "iucn_status_cache.csv"
    monkeypatch.setattr(module, "get_iucn_token", lambda: "")

    _enriched_df, cache_df, summary = module.enrich_clean_encyclopedia_with_incremental_iucn(
        df,
        cache_path=cache_path,
        batch_size=10,
        request_delay_seconds=0,
    )

    assert cache_df.empty
    assert summary.requested_this_run == 0
