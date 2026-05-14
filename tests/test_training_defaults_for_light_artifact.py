from scripts.run_pipeline import parse_args, normalize_pipeline_limits


def test_default_limits_cover_light_artifact(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_pipeline.py"])
    args = normalize_pipeline_limits(parse_args())
    assert args.offline_max_species == 2000
    assert args.max_vernacular_species >= args.offline_max_species
    assert args.max_image_species >= args.offline_max_species
    assert args.max_iucn_species == 500


def test_limits_are_raised_when_user_sets_too_low_values(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pipeline.py",
            "--offline-max-species",
            "2000",
            "--max-vernacular-species",
            "10",
            "--max-image-species",
            "10",
        ],
    )
    args = normalize_pipeline_limits(parse_args())
    assert args.max_vernacular_species == 2000
    assert args.max_image_species == 2000
