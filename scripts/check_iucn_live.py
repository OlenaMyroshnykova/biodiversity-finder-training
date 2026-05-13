"""Small live smoke test for IUCN Red List API v4.

Usage:
    python scripts/check_iucn_live.py
    python scripts/check_iucn_live.py "Panthera leo" "Equus quagga"

This script performs only a few requests. Do not use it for bulk scraping.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from src.conservation_status import (  # noqa: E402
    extract_iucn_category_from_payload,
    fetch_iucn_payload,
)


def main() -> None:
    token = os.getenv("IUCN_API_TOKEN") or os.getenv("IUCN_TOKEN")
    if not token:
        raise SystemExit("IUCN_API_TOKEN is empty. Set it in .env or environment variables.")

    species = sys.argv[1:] or ["Panthera leo", "Equus quagga", "Ursus maritimus"]
    print("[IUCN LIVE] Token configured: yes")
    print("[IUCN LIVE] Endpoint: https://api.iucnredlist.org/api/v4/taxa/scientific_name")

    official_count = 0
    for scientific_name in species:
        payload = fetch_iucn_payload(scientific_name, token, timeout_seconds=30)
        category = extract_iucn_category_from_payload(payload) if payload else ""
        print(f"[IUCN LIVE] {scientific_name}: {category or 'NO_DATA'}")
        official_count += int(bool(category))
        time.sleep(2)

    if official_count == 0:
        raise SystemExit(
            "[IUCN LIVE] 0 official statuses found. Check token, endpoint, or API access."
        )


if __name__ == "__main__":
    main()
