"""Apply v38 fix: make GitHub Actions and CLI scripts able to import local src package.

Run from the root of biodiversity-finder-training:
    python tools/apply_pythonpath_fix_v38.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()

WORKFLOW = ROOT / ".github" / "workflows" / "refresh_artifacts_production.yml"
SCRIPTS = [
    ROOT / "scripts" / "plan_artifact_refresh.py",
    ROOT / "scripts" / "download_base_artifacts_from_hf.py",
    ROOT / "scripts" / "run_iucn_only_from_hf.py",
    ROOT / "scripts" / "build_artifact_manifest.py",
    ROOT / "scripts" / "publish_to_huggingface.py",
    ROOT / "scripts" / "clean_image_enrichment.py",
]

BOOTSTRAP = '''\n# Allow direct execution from GitHub Actions and local terminals.\nfrom pathlib import Path\nimport sys\n\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\nif str(PROJECT_ROOT) not in sys.path:\n    sys.path.insert(0, str(PROJECT_ROOT))\n'''


def ensure_workflow_pythonpath(path: Path) -> bool:
    if not path.exists():
        print(f"SKIP missing workflow: {path}")
        return False
    text = path.read_text(encoding="utf-8")
    if "PYTHONPATH: ${{ github.workspace }}" in text:
        print(f"OK workflow already has PYTHONPATH: {path}")
        return False

    lines = text.splitlines()
    insert_at = 1
    # Keep `name:` first; insert top-level env before `on:` if possible.
    for idx, line in enumerate(lines):
        if line.startswith("on:"):
            insert_at = idx
            break

    env_block = ["", "env:", "  PYTHONPATH: ${{ github.workspace }}", ""]
    lines[insert_at:insert_at] = env_block
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"UPDATED workflow PYTHONPATH: {path}")
    return True


def find_bootstrap_insert_index(lines: list[str]) -> int:
    idx = 0

    # Shebang / encoding comments.
    while idx < len(lines) and (lines[idx].startswith("#!") or "coding" in lines[idx][:40]):
        idx += 1

    # Module docstring.
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx < len(lines) and lines[idx].lstrip().startswith(('"""', "'''")):
        quote = lines[idx].lstrip()[:3]
        # One-line docstring.
        if lines[idx].count(quote) >= 2 and len(lines[idx].strip()) > 3:
            idx += 1
        else:
            idx += 1
            while idx < len(lines):
                current = lines[idx]
                idx += 1
                if quote in current:
                    break

    # Blank lines after docstring.
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1

    # Future imports must stay before normal imports.
    while idx < len(lines) and lines[idx].startswith("from __future__ import"):
        idx += 1
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1

    return idx


def ensure_script_bootstrap(path: Path) -> bool:
    if not path.exists():
        print(f"SKIP missing script: {path}")
        return False
    text = path.read_text(encoding="utf-8")
    if "PROJECT_ROOT = Path(__file__).resolve().parents[1]" in text:
        print(f"OK bootstrap already exists: {path}")
        return False
    if "from src." not in text and "import src" not in text:
        print(f"SKIP no src import: {path}")
        return False

    lines = text.splitlines()
    idx = find_bootstrap_insert_index(lines)
    block_lines = BOOTSTRAP.strip("\n").splitlines()
    lines[idx:idx] = block_lines + [""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"UPDATED script bootstrap: {path}")
    return True


def main() -> None:
    if not (ROOT / "scripts").exists() or not (ROOT / "src").exists():
        raise SystemExit("Run this script from the root of biodiversity-finder-training")

    changed = False
    changed |= ensure_workflow_pythonpath(WORKFLOW)
    for script in SCRIPTS:
        changed |= ensure_script_bootstrap(script)

    print("Done." if changed else "No changes needed.")


if __name__ == "__main__":
    main()
