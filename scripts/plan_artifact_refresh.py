from __future__ import annotations

# Allow direct execution from GitHub Actions and local terminals.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
from pathlib import Path

from src.artifact_manifest import FreshnessDecision, decide_refresh_mode, load_manifest


def _download_remote_manifest(repo_id: str, repo_path: str, token: str | None) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        print(f"[PLAN] huggingface_hub unavailable: {exc}")
        return None
    try:
        downloaded = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_path, token=token)
    except Exception as exc:
        print(f"[PLAN] remote manifest not available: {exc}")
        return None
    return Path(downloaded)


def _write_github_outputs(decision: FreshnessDecision) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as file_obj:
        file_obj.write(f"refresh_mode={decision.refresh_mode}\n")
        file_obj.write(f"should_run={'true' if decision.should_run else 'false'}\n")
        file_obj.write(f"reasons={'; '.join(decision.reasons)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan whether production artifacts need refresh.")
    parser.add_argument("--repo-id", default=os.getenv("HF_REPO_ID", "selenamir/biodiversity-finder-artifacts"))
    parser.add_argument("--manifest-path", default="reports/artifact_manifest.json")
    parser.add_argument("--max-age-days", type=int, default=7)
    parser.add_argument("--force-mode", choices=["auto", "none", "iucn_only", "full_base"], default="auto")
    parser.add_argument("--output-json", default="reports/artifact_refresh_plan.json")
    args = parser.parse_args()

    if args.force_mode != "auto":
        decision = FreshnessDecision(refresh_mode=args.force_mode, reasons=[f"forced mode: {args.force_mode}"])
    else:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        local_manifest = _download_remote_manifest(args.repo_id, args.manifest_path, token)
        manifest = load_manifest(local_manifest) if local_manifest and local_manifest.exists() else None
        decision = decide_refresh_mode(manifest, max_age_days=args.max_age_days)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(
            {
                "refresh_mode": decision.refresh_mode,
                "should_run": decision.should_run,
                "reasons": decision.reasons,
                "missing_artifacts": decision.missing_artifacts,
                "stale_artifacts": decision.stale_artifacts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_github_outputs(decision)
    print(f"[PLAN] refresh_mode={decision.refresh_mode}")
    for reason in decision.reasons:
        print(f"[PLAN] reason: {reason}")


if __name__ == "__main__":
    main()
