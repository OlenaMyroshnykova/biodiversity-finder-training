"""Small Hugging Face checkpoint helpers for long-running training pipelines."""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_HF_REPO_ID = "selenamir/biodiversity-finder-artifacts"
DEFAULT_IUCN_CACHE_REPO_PATH = "checkpoints/iucn_status_cache.csv"


def get_hf_token() -> str:
    return os.getenv("HF_TOKEN", "").strip()


def download_checkpoint_file(
    *,
    repo_id: str,
    repo_path: str,
    local_path: str | Path,
    repo_type: str = "dataset",
) -> bool:
    """Download one checkpoint file from HF if it exists.

    Returns True if a file was downloaded, False otherwise. Missing token or missing
    remote file is not fatal because the first run starts without a checkpoint.
    """
    token = get_hf_token()
    if not token:
        print("[HF CHECKPOINT] HF_TOKEN no configurado; no se descarga checkpoint.", flush=True)
        return False
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        print("[HF CHECKPOINT] huggingface_hub no disponible.", flush=True)
        return False

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=repo_path,
            token=token,
        )
    except Exception as exc:
        print(f"[HF CHECKPOINT] No se pudo descargar {repo_path}: {exc}", flush=True)
        return False

    source = Path(downloaded)
    local_path.write_bytes(source.read_bytes())
    print(f"[HF CHECKPOINT] Descargado {repo_path} → {local_path}", flush=True)
    return True


def upload_checkpoint_file(
    *,
    repo_id: str,
    repo_path: str,
    local_path: str | Path,
    commit_message: str,
    repo_type: str = "dataset",
) -> bool:
    """Upload one checkpoint file to HF.

    Returns True if upload succeeded. Failures are logged but not fatal for the
    data-building step.
    """
    token = get_hf_token()
    local_path = Path(local_path)
    if not token:
        print("[HF CHECKPOINT] HF_TOKEN no configurado; no se sube checkpoint.", flush=True)
        return False
    if not local_path.exists():
        print(f"[HF CHECKPOINT] No existe local_path: {local_path}", flush=True)
        return False
    try:
        from huggingface_hub import HfApi
    except Exception:
        print("[HF CHECKPOINT] huggingface_hub no disponible.", flush=True)
        return False

    try:
        HfApi().upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            token=token,
            commit_message=commit_message,
        )
    except Exception as exc:
        print(f"[HF CHECKPOINT] No se pudo subir {repo_path}: {exc}", flush=True)
        return False

    print(f"[HF CHECKPOINT] Subido {local_path} → {repo_id}/{repo_path}", flush=True)
    return True
