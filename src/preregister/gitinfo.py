"""Best-effort git state capture for ledger entries."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitState:
    commit: str | None
    dirty: bool | None

    def to_dict(self) -> dict[str, str | bool | None]:
        return {"commit": self.commit, "dirty": self.dirty}


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def git_state(root: str | Path) -> GitState:
    """Return HEAD commit and whether tracked files differ from it.

    Changes under ``.prereg/`` are ignored so that appending to the ledger does not
    itself mark the working tree dirty.
    """
    root = Path(root)
    head = _git(root, "rev-parse", "HEAD")
    if head is None:
        return GitState(commit=None, dirty=None)
    status = _git(
        root, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude).prereg"
    )
    dirty = None if status is None else bool(status.strip())
    return GitState(commit=head.strip(), dirty=dirty)


def git_toplevel(root: str | Path) -> Path | None:
    out = _git(Path(root), "rev-parse", "--show-toplevel")
    return Path(out.strip()) if out else None


def git_commit_paths(root: str | Path, paths: list[Path], message: str) -> bool:
    root = Path(root)
    if _git(root, "add", "--", *[str(p) for p in paths]) is None:
        return False
    return _git(root, "commit", "-m", message, "--", *[str(p) for p in paths]) is not None
