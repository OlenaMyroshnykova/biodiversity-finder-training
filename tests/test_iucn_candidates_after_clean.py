import pandas as pd

from src.iucn_candidates import extract_iucn_candidates, prepare_clean_iucn_candidate_frame


def test_iucn_candidates_are_clean_unique_and_prioritized_after_encyclopedia_cleaning():
    df = pd.DataFrame(
        [
            {
                "scientific_name": "Panthera leo (Linnaeus, 1758)",
                "kingdom": "Animalia",
                "taxon_class": "Mammalia",
                "observations": 10,
                "has_image": True,
                "vernacular_names": "Lion | León",
            },
            {
                "scientific_name": "Panthera leo",
                "kingdom": "Animalia",
                "taxon_class": "Mammalia",
                "observations": 5,
                "has_image": False,
                "vernacular_names": "Lion",
            },
            {
                "scientific_name": "Unknown sp.",
                "kingdom": "Animalia",
                "taxon_class": "Mammalia",
            },
            {
                "scientific_name": "Agaricus bisporus",
                "kingdom": "Fungi",
                "taxon_class": "Agaricomycetes",
            },
            {
                "scientific_name": "Phoenicopterus roseus",
                "kingdom": "Animalia",
                "taxon_class": "Aves",
                "observations": 100,
                "has_image": False,
                "vernacular_names": "Flamenco",
            },
        ]
    )

    candidates = extract_iucn_candidates(df)

    assert candidates == ["Panthera leo", "Phoenicopterus roseus"]


def test_prepare_candidate_frame_contains_priority_score():
    df = pd.DataFrame(
        [{"scientific_name": "Ursus arctos", "kingdom": "Animalia", "taxon_class": "Mammalia"}]
    )
    result = prepare_clean_iucn_candidate_frame(df)
    assert "iucn_priority_score" in result.columns
    assert result.loc[0, "canonical_scientific_name"] == "Ursus arctos"
