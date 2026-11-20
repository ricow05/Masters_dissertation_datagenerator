"""Which code made a dataset: the package version and, when run from a git checkout, the commit."""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parent.parent     # the repository root of a source checkout


def _git(*args) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


@lru_cache(maxsize=1)
def generator_info() -> dict:
    """``{"name", "version", "commit", "dirty"}``.

    ``commit`` is the git commit of the datagen checkout and ``dirty`` says whether it had
    uncommitted changes. Both are None when datagen is not run from its own git checkout
    (an installed wheel, or no git), so the commit of some enclosing repository is never
    recorded by mistake.
    """
    info = {"name": "datagen", "version": __version__, "commit": None, "dirty": None}
    top = _git("rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != ROOT:
        return info
    commit = _git("rev-parse", "HEAD")
    if commit is None:                          # a repository without commits
        return info
    status = _git("status", "--porcelain")
    info.update(commit=commit, dirty=bool(status) if status is not None else None)
    return info
