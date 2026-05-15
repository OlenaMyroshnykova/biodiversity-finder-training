from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.iucn_incremental_after_clean import enrich_clean_encyclopedia_with_incremental_iucn
from src.hf_checkpoints import DEFAULT_HF_REPO_ID, upload_checkpoint_file


def _copy_processed_outputs(enriched_df: pd.DataFrame, *, light_size: int) -> None:
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    enriched_df.to_parquet(processed_dir / "species_encyclopedia.parquet", index=False)

    if light_size and light_size > 0:
        sort_columns = [col for col in ("has_image", "observations") if col in enriched_df.columns]
        if sort_columns:
            ascending = [False for _ in sort_columns]
            light_df = enriched_df.sort_values(sort_columns, ascending=ascending).head(light_size).copy()
        else:
            light_df = enriched_df.head(light_size).copy()
        light_df.to_parquet(processed_dir / "species_encyclopedia_light.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only late incremental IUCN on already-clean HF artifacts.")
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--light-size", type=int, default=10000)
    parser.add_argument("--publish-cache", action="store_true")
    args = parser.parse_args()

    clean_path = Path("data/interim/species_encyclopedia_clean_before_iucn.parquet")
    if not clean_path.exists():
        raise FileNotFoundError(
            "Missing data/interim/species_encyclopedia_clean_before_iucn.parquet. "
            "Run scripts/download_base_artifacts_from_hf.py first."
        )

    encyclopedia_df = pd.read_parquet(clean_path)
    cache_path = Path("data/interim/iucn_status_cache.csv")
    candidate_limit = args.candidate_limit if args.candidate_limit and args.candidate_limit > 0 else None

    def checkpoint_callback(path: Path) -> None:
        if args.publish_cache:
            upload_checkpoint_file(
                repo_id=args.repo_id,
                repo_path="interim/iucn_status_cache.csv",
                local_path=path,
                commit_message="Update incremental IUCN cache checkpoint",
            )

    enriched_df, cache_df, summary = enrich_clean_encyclopedia_with_incremental_iucn(
        encyclopedia_df,
        cache_path=cache_path,
        batch_size=args.batch_size,
        request_delay_seconds=args.delay_seconds,
        checkpoint_every=args.checkpoint_every,
        candidate_limit=candidate_limit,
        checkpoint_callback=checkpoint_callback,
    )

    _copy_processed_outputs(enriched_df, light_size=args.light_size)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/iucn_incremental_summary.json").write_text(
        json.dumps(summary.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[IUCN ONLY] summary:", summary)
    print(f"[IUCN ONLY] cache rows after: {len(cache_df):,}")


if __name__ == "__main__":
    main()
