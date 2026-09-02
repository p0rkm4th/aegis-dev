#!/usr/bin/env python3
"""Validate CURRENT_STATE release/evidence pointers against the checked-out Git state."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from aegis.release_truth import validate_state_pointers


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def main() -> int:
    path = Path("CURRENT_STATE.json")
    state = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_state_pointers(state)
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    recorded = state.get("repository_head_sha")
    if branch != state.get("active_branch"):
        errors.append(f"active_branch={state.get('active_branch')!r} does not match {branch!r}")
    if not isinstance(recorded, str) or not _is_ancestor(recorded, head):
        errors.append(f"repository_head_sha={recorded!r} is not an ancestor of checked-out HEAD")
    pushed = state.get("last_pushed_sha")
    if isinstance(pushed, str) and not _is_ancestor(pushed, head):
        errors.append("last_pushed_sha is not an ancestor of checked-out HEAD")
    if errors:
        for error in errors:
            print(f"CURRENT_STATE invalid: {error}")
        return 1
    print(f"CURRENT_STATE valid: branch={branch} head={head[:12]}")
    return 0


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", ancestor, descendant),
            check=False,
        ).returncode
        == 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
