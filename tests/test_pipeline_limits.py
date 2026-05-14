from argparse import Namespace

from scripts.run_pipeline import normalize_pipeline_limits


def test_zero_api_limits_mean_all_species():
    args = Namespace(
        max_iucn_species=0,
        max_vernacular_species=0,
        max_image_species=0,
        max_gbif_image_fallback_species=0,
        sample_size=0,
        offline_max_species=5000,
    )

    limits = normalize_pipeline_limits(args)

    assert limits["max_iucn_species_api"] is None
    assert limits["max_vernacular_species_api"] is None
    assert limits["max_image_species_api"] is None
    assert limits["max_gbif_image_fallback_species_api"] is None
    assert limits["sample_size"] == 0
    assert limits["offline_max_species"] == 5000


def test_positive_api_limits_are_kept():
    args = Namespace(
        max_iucn_species=1000,
        max_vernacular_species=2000,
        max_image_species=3000,
        max_gbif_image_fallback_species=400,
        sample_size=500,
        offline_max_species=800,
    )

    limits = normalize_pipeline_limits(args)

    assert limits["max_iucn_species_api"] == 1000
    assert limits["max_vernacular_species_api"] == 2000
    assert limits["max_image_species_api"] == 3000
    assert limits["max_gbif_image_fallback_species_api"] == 400
    assert limits["sample_size"] == 500
    assert limits["offline_max_species"] == 800
