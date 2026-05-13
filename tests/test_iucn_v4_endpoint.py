"""Tests for the IUCN Red List API v4 scientific-name lookup."""
from __future__ import annotations

from src.conservation_status import (
    extract_iucn_category_from_payload,
    split_scientific_name_for_iucn,
    fetch_iucn_payload,
)


def test_split_scientific_name_for_iucn_uses_genus_and_species_params() -> None:
    """The v4 endpoint needs query params, not a path with full scientific name."""

    assert split_scientific_name_for_iucn("Panthera leo (Linnaeus, 1758)") == {
        "genus_name": "Panthera",
        "species_name": "leo",
    }


def test_split_scientific_name_for_iucn_keeps_simple_infra_name() -> None:
    """Subspecies/infra names are sent only when they are simple epithets."""

    assert split_scientific_name_for_iucn("Canis lupus familiaris") == {
        "genus_name": "Canis",
        "species_name": "lupus",
        "infra_name": "familiaris",
    }


def test_iucn_v4_payload_parser_accepts_nested_category() -> None:
    """Parser accepts the expected v4 assessment-list shape."""

    payload = {
        "name": "Panthera leo",
        "assessments": [
            {
                "latest": True,
                "scope": "Global",
                "red_list_category": {"code": "VU", "name": "Vulnerable"},
            }
        ],
    }

    assert extract_iucn_category_from_payload(payload) == "VU"


def test_fetch_iucn_payload_calls_correct_v4_endpoint(monkeypatch) -> None:
    """The API call must use /taxa/scientific_name with genus/species params."""

    calls = []

    class FakeResponse:
        status_code = 200
        url = "https://api.iucnredlist.org/api/v4/taxa/scientific_name?genus_name=Panthera&species_name=leo"
        text = '{"assessments": [{"red_list_category": {"code": "VU"}}]}'

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"assessments": [{"red_list_category": {"code": "VU"}}]}

    def fake_get(url, headers, params, timeout):
        calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("src.conservation_status.requests.get", fake_get)

    payload = fetch_iucn_payload("Panthera leo", "fake-token")

    assert payload is not None
    assert calls[0]["url"] == "https://api.iucnredlist.org/api/v4/taxa/scientific_name"
    assert calls[0]["params"] == {"genus_name": "Panthera", "species_name": "leo"}
    assert calls[0]["headers"]["Authorization"] == "Bearer fake-token"
