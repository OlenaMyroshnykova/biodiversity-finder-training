"""Checks that old curated demo-species and fungi architecture are gone."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_code_has_no_old_demo_species_or_fungi_download_markers() -> None:
    checked_files = [
        PROJECT_ROOT / "src" / "data_acquisition.py",
        PROJECT_ROOT / "src" / "data_cleaning.py",
        PROJECT_ROOT / "src" / "encyclopedia.py",
        PROJECT_ROOT / "src" / "search_tags.py",
    ]
    combined_text = "\n".join(path.read_text(encoding="utf-8").lower() for path in checked_files)

    forbidden_markers = [
        "flamingo_pink_bird",
        "polar_bear",
        "jaguar_panthera_onca",
        "big_cats_felidae",
        "pink_bird",
        "ave rosa",
        "animal polar",
        "panthera onca",
        "ursus maritimus",
        "phoenicopterus roseus",
        "fungi_mushrooms",
        "kingdom_fungi",
        "class_agaricomycetes",
    ]
    for marker in forbidden_markers:
        assert marker not in combined_text
