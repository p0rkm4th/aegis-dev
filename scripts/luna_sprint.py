#!/usr/bin/env python3
"""Inspect or select the next task from the durable Luna development sprint."""

from __future__ import annotations

import argparse
from pathlib import Path

from development.luna_sprint import load, save


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--next", action="store_true", help="select and persist the next task")
    args = parser.parse_args()
    state = load(args.state)
    task = state.select_next_task() if args.next else None
    if args.next:
        save(args.state, state)
    print(task.task_id if task else "no-ready-task")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
