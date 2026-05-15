from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.artifact_manifest import build_manifest, save_manifest


def _collect_parameters_from_env() -> dict[str, str]:
    prefixes = ("INPUT_", "PIPELINE_", "IUCN_", "GBIF_", "HF_")
    safe: dict[str, str] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefixes):
            continue
        if "TOKEN" in key or "SECRET" in key or "PASSWORD" in key:
            safe[key] = "***"
        else:
            safe[key] = str(value)
    return safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build artifact manifest for freshness-aware production runs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/artifact_manifest.json")
    parser.add_argument("--copy-to-processed", action="store_true")
    parser.add_argument("--pipeline-name", default="biodiversity-training")
    parser.add_argument("--parameters-json", default=None)
    args = parser.parse_args()

    parameters = _collect_parameters_from_env()
    if args.parameters_json:
        parameters.update(json.loads(args.parameters_json))

    manifest = build_manifest(root_dir=args.root, parameters=parameters, pipeline_name=args.pipeline_name)
    output = save_manifest(manifest, args.output)
    print(f"[MANIFEST] saved {output}")

    if args.copy_to_processed:
        processed_output = Path("data/processed/artifact_manifest.json")
        save_manifest(manifest, processed_output)
        print(f"[MANIFEST] copied {processed_output}")


if __name__ == "__main__":
    main()
