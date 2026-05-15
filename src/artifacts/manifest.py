"""Artifact manifest and freshness policy for production training runs.

The production pipeline should not blindly rebuild everything.  Each expensive
artifact stores enough metadata to decide whether it can be reused, must be
refreshed, or can be incrementally enriched.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = "1.0"
DEFAULT_MANIFEST_PATH = Path("reports/artifact_manifest.json")

BASE_ARTIFACTS: tuple[str, ...] = (
    "data/processed/species_encyclopedia.parquet",
    "data/processed/species_encyclopedia_light.parquet",
    "data/processed/species_occurrence_points.parquet",
    "data/processed/species_occurrence_points_light.parquet",
    "data/interim/species_encyclopedia_clean_before_iucn.parquet",
    "data/interim/gbif_occurrences_clean.parquet",
    "data/interim/vernacular_names.csv",
    "data/interim/image_enrichment.csv",
    "data/interim/climate_reference.csv",
)

INCREMENTAL_ARTIFACTS: tuple[str, ...] = (
    "data/interim/iucn_status_cache.csv",
    "reports/metrics.json",
    "reports/classification_report.csv",
)

CRITICAL_ARTIFACTS: tuple[str, ...] = (
    "data/processed/species_encyclopedia.parquet",
    "data/interim/species_encyclopedia_clean_before_iucn.parquet",
    "data/interim/iucn_status_cache.csv",
    "reports/metrics.json",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(timezone.utc)
    except Exception:
        return None


def compute_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_csv_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.reader(file_obj)
            row_count = sum(1 for _ in reader)
        return max(row_count - 1, 0)
    except Exception:
        return None


def infer_row_count(path: str | Path) -> int | None:
    """Best-effort row count without making manifest creation fragile."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _count_csv_rows(path)
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                for key in ("rows", "row_count", "train_rows", "test_rows"):
                    if key in data and isinstance(data[key], int):
                        return int(data[key])
        except Exception:
            return None
    if suffix == ".parquet":
        try:
            import pandas as pd

            return int(len(pd.read_parquet(path, columns=[])))
        except Exception:
            try:
                import pandas as pd

                return int(len(pd.read_parquet(path)))
            except Exception:
                return None
    return None


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    exists: bool
    size_bytes: int = 0
    sha256: str | None = None
    row_count: int | None = None
    generated_at_utc: str | None = None
    source: str = "pipeline"
    stage: str = "unknown"


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    generated_at_utc: str
    pipeline_name: str
    git_sha: str | None
    run_id: str | None
    parameters: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifacts"] = [asdict(item) for item in self.artifacts]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactManifest":
        artifacts = [ArtifactRecord(**item) for item in data.get("artifacts", [])]
        return cls(
            schema_version=str(data.get("schema_version", MANIFEST_SCHEMA_VERSION)),
            generated_at_utc=str(data.get("generated_at_utc", "")),
            pipeline_name=str(data.get("pipeline_name", "unknown")),
            git_sha=data.get("git_sha"),
            run_id=data.get("run_id"),
            parameters=dict(data.get("parameters", {})),
            artifacts=artifacts,
        )

    def get(self, path: str) -> ArtifactRecord | None:
        normalized = path.replace("\\", "/")
        for artifact in self.artifacts:
            if artifact.path == normalized:
                return artifact
        return None


def infer_stage(path: str) -> str:
    if "/processed/" in path:
        return "processed"
    if "/interim/" in path:
        return "interim"
    if "/reports/" in path or path.startswith("reports/"):
        return "reports"
    if "/raw/" in path:
        return "raw"
    return "unknown"


def build_manifest(
    *,
    root_dir: str | Path = ".",
    artifact_paths: Iterable[str] | None = None,
    parameters: dict[str, Any] | None = None,
    pipeline_name: str = "biodiversity-training",
) -> ArtifactManifest:
    root = Path(root_dir)
    paths = list(artifact_paths or (*BASE_ARTIFACTS, *INCREMENTAL_ARTIFACTS))
    generated_at = utc_now_iso()
    records: list[ArtifactRecord] = []

    for relative in paths:
        normalized = relative.replace("\\", "/")
        absolute = root / normalized
        if not absolute.exists():
            records.append(ArtifactRecord(path=normalized, exists=False, generated_at_utc=generated_at, stage=infer_stage(normalized)))
            continue
        records.append(
            ArtifactRecord(
                path=normalized,
                exists=True,
                size_bytes=absolute.stat().st_size,
                sha256=compute_sha256(absolute),
                row_count=infer_row_count(absolute),
                generated_at_utc=generated_at,
                stage=infer_stage(normalized),
            )
        )

    return ArtifactManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        generated_at_utc=generated_at,
        pipeline_name=pipeline_name,
        git_sha=os.getenv("GITHUB_SHA"),
        run_id=os.getenv("GITHUB_RUN_ID"),
        parameters=parameters or {},
        artifacts=records,
    )


def save_manifest(manifest: ArtifactManifest, path: str | Path = DEFAULT_MANIFEST_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def load_manifest(path: str | Path) -> ArtifactManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ArtifactManifest.from_dict(data)


@dataclass(frozen=True)
class FreshnessDecision:
    refresh_mode: str
    reasons: list[str]
    missing_artifacts: list[str] = field(default_factory=list)
    stale_artifacts: list[str] = field(default_factory=list)

    @property
    def should_run(self) -> bool:
        return self.refresh_mode != "none"


def decide_refresh_mode(
    manifest: ArtifactManifest | None,
    *,
    max_age_days: int = 7,
    require_iucn_cache: bool = True,
    now: datetime | None = None,
) -> FreshnessDecision:
    """Decide whether to rebuild base data or run only incremental enrichment.

    Modes:
    - ``full_base``: expensive GBIF/NASA/images/common-name rebuild is needed.
    - ``iucn_only``: base artifacts are usable; run incremental IUCN/checkpoint merge.
    - ``none``: everything is fresh enough.
    """
    if manifest is None:
        return FreshnessDecision(refresh_mode="full_base", reasons=["remote manifest missing"])

    now = now or datetime.now(timezone.utc)
    missing_base: list[str] = []
    stale_base: list[str] = []

    for path in BASE_ARTIFACTS:
        record = manifest.get(path)
        if record is None or not record.exists:
            missing_base.append(path)
            continue
        generated = parse_utc(record.generated_at_utc or manifest.generated_at_utc)
        if generated is None:
            stale_base.append(path)
        else:
            age_days = (now - generated).total_seconds() / 86400
            if max_age_days >= 0 and age_days > max_age_days:
                stale_base.append(path)

    if missing_base or stale_base:
        reasons = []
        if missing_base:
            reasons.append(f"missing base artifacts: {len(missing_base)}")
        if stale_base:
            reasons.append(f"stale base artifacts: {len(stale_base)}")
        return FreshnessDecision("full_base", reasons, missing_base, stale_base)

    missing_incremental: list[str] = []
    if require_iucn_cache:
        record = manifest.get("data/interim/iucn_status_cache.csv")
        if record is None or not record.exists:
            missing_incremental.append("data/interim/iucn_status_cache.csv")
        elif record.row_count is not None and record.row_count <= 0:
            missing_incremental.append("data/interim/iucn_status_cache.csv: empty")

    if missing_incremental:
        return FreshnessDecision(
            refresh_mode="iucn_only",
            reasons=["base artifacts are reusable, but IUCN cache is missing or empty"],
            missing_artifacts=missing_incremental,
        )

    return FreshnessDecision("none", ["all tracked artifacts are fresh"])
