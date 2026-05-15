from pathlib import Path

from scripts import publish_to_huggingface


def test_publish_script_has_expected_default_repo() -> None:
    assert publish_to_huggingface.DEFAULT_REPO_ID == "selenamir/biodiversity-finder-artifacts"
    assert publish_to_huggingface.DEFAULT_REPO_TYPE == "dataset"


def test_publish_script_detects_publishable_files(tmp_path: Path) -> None:
    folder = tmp_path / "processed"
    folder.mkdir()
    (folder / "species_encyclopedia.parquet").write_text("fake", encoding="utf-8")
    (folder / "debug.tmp").write_text("ignore", encoding="utf-8")

    assert publish_to_huggingface._has_publishable_files(folder)
    assert "debug.tmp" in publish_to_huggingface._ignore_patterns_for(folder)
