"""Canonical filesystem paths for CORE4D scripts."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
CORE4D = REPO / "workspace/core4d"
SCRIPTS = CORE4D / "scripts"
RESULTS = CORE4D / "results"


def repo_path(path: str | Path) -> Path:
    """Return an absolute path under the repo unless `path` is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def rel_repo(path: str | Path) -> str:
    """Return a repo-relative path string when possible."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)
