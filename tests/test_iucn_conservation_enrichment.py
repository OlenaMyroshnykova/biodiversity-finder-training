"""Tests for real IUCN Red List conservation enrichment architecture."""

from __future__ import annotations

import pandas as pd

from src.conservation_status import (
    add_conservation_status_to_encyclopedia,
    clean_scientific_name,
    extract_iucn_category_from_payload,
)


def build_test_encyclopedia() -> pd.DataFrame:
    """Create a minimal species encyclopedia."""

    return pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo (Linnaeus, 1758)",
                "canonical_scientific_name": "Panthera leo",
                "kingdom": "Animalia",
                "taxon_class": "Mammalia",
                "family": "Felidae",
                "observations": 9,
            },
            {
                "scientific_name": "Papilio machaon Linnaeus, 1758",
                "canonical_scientific_name": "Papilio machaon",
                "kingdom": "Animalia",
                "taxon_class": "Insecta",
                "family": "Papilionidae",
                "observations": 50,
            },
        ]
    )


def test_clean_scientific_name_removes_authority() -> None:
    """IUCN lookup must receive clean binomial names."""

    assert clean_scientific_name("Panthera leo (Linnaeus, 1758)") == "Panthera leo"
    assert clean_scientific_name("Felis chaus Schreber, 1777") == "Felis chaus"


def test_no_token_returns_no_data_not_fake_lc(monkeypatch) -> None:
    """Without token the pipeline must not invent LC as fallback."""

    monkeypatch.delenv("IUCN_API_TOKEN", raising=False)
    monkeypatch.delenv("IUCN_TOKEN", raising=False)

    enriched_df, conservation_df = add_conservation_status_to_encyclopedia(
        build_test_encyclopedia()
    )

    assert not conservation_df.empty
    assert set(conservation_df["iucn_category"]) == {"NO_DATA"}
    assert set(enriched_df["conservation_status"]) == {"NO_DATA"}
    assert not enriched_df["iucn_is_official"].any()
    assert not enriched_df["is_threatened"].any()


def test_existing_official_iucn_status_is_preserved() -> None:
    """If a row already contains an official category, preserve it."""

    df = build_test_encyclopedia()
    df.loc[0, "iucn_status"] = "EN"

    enriched_df, _ = add_conservation_status_to_encyclopedia(df)

    lion = enriched_df[enriched_df["canonical_scientific_name"] == "Panthera leo"].iloc[0]
    assert lion["iucn_category"] == "EN"
    assert lion["conservation_category"] == "Endangered"
    assert bool(lion["iucn_is_official"]) is True
    assert bool(lion["is_threatened"]) is True


def test_iucn_api_payload_parser_accepts_assessments_list() -> None:
    """The parser should accept the common v4 shape with assessments list."""

    payload = {
        "scientific_name": "Panthera leo",
        "assessments": [
            {"red_list_category": {"code": "VU", "name": "Vulnerable"}}
        ],
    }

    assert extract_iucn_category_from_payload(payload) == "VU"
