from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.artifact_manifest import (
    BASE_ARTIFACTS,
    build_manifest,
    decide_refresh_mode,
    save_manifest,
    load_manifest,
)


def test_manifest_records_missing_and_existing_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "data/processed/species_encyclopedia.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not really parquet, but enough for hash")

    manifest = build_manifest(root_dir=tmp_path, artifact_paths=["data/processed/species_encyclopedia.parquet", "missing.csv"])

    existing = manifest.get("data/processed/species_encyclopedia.parquet")
    missing = manifest.get("missing.csv")
    assert existing is not None and existing.exists
    assert existing.sha256
    assert missing is not None and not missing.exists


def test_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = build_manifest(root_dir=tmp_path, artifact_paths=[])
    path = save_manifest(manifest, tmp_path / "manifest.json")
    loaded = load_manifest(path)
    assert loaded.schema_version == manifest.schema_version
    assert loaded.generated_at_utc == manifest.generated_at_utc


def test_refresh_decision_requires_full_rebuild_when_base_missing(tmp_path: Path) -> None:
    manifest = build_manifest(root_dir=tmp_path, artifact_paths=BASE_ARTIFACTS)
    decision = decide_refresh_mode(manifest, max_age_days=7)
    assert decision.refresh_mode == "full_base"
    assert decision.missing_artifacts


def test_refresh_decision_iucn_only_when_base_exists_but_cache_missing(tmp_path: Path) -> None:
    for relative in BASE_ARTIFACTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    manifest = build_manifest(root_dir=tmp_path, artifact_paths=[*BASE_ARTIFACTS, "data/interim/iucn_status_cache.csv"])
    decision = decide_refresh_mode(manifest, max_age_days=7)
    assert decision.refresh_mode == "iucn_only"


def test_refresh_decision_none_when_base_and_iucn_are_fresh(tmp_path: Path) -> None:
    for relative in [*BASE_ARTIFACTS, "data/interim/iucn_status_cache.csv"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".csv"):
            path.write_text("canonical_scientific_name,iucn_category\nPanthera leo,VU\n", encoding="utf-8")
        else:
            path.write_text("x", encoding="utf-8")

    manifest = build_manifest(root_dir=tmp_path, artifact_paths=[*BASE_ARTIFACTS, "data/interim/iucn_status_cache.csv"])
    decision = decide_refresh_mode(manifest, max_age_days=7)
    assert decision.refresh_mode == "none"
