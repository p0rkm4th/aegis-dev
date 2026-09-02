#!/usr/bin/env python3
"""Audit semantic evaluation split integrity without invoking a model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED = {
    "id",
    "family",
    "utterance",
    "kind",
    "semantic_mode",
    "phenomena",
    "provenance",
    "expected_mutation",
}
ALLOWED_PROVENANCE = {"manual", "owner_harvest", "transformed", "oss_harvest"}
ALLOWED_KINDS = {"ANSWER", "ACTION", "CLARIFY"}
ALLOWED_SEMANTIC_MODES = {"READ", "ACTION", "GENERATION", "CLARIFY"}


def _load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path}: corpus must be a JSON array")
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}]: case must be an object")
        missing = REQUIRED - item.keys()
        if missing:
            raise ValueError(f"{path}[{index}]: missing {', '.join(sorted(missing))}")
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise ValueError(f"{path}[{index}]: id must be a non-empty string")
        if not isinstance(item["family"], str) or not item["family"].strip():
            raise ValueError(f"{path}[{index}]: family must be a non-empty string")
        if not isinstance(item["utterance"], str) or not item["utterance"].strip():
            raise ValueError(f"{path}[{index}]: utterance must be a non-empty string")
        if item["kind"] not in ALLOWED_KINDS:
            raise ValueError(f"{path}[{index}]: unsupported kind {item['kind']!r}")
        if item["semantic_mode"] not in ALLOWED_SEMANTIC_MODES:
            raise ValueError(
                f"{path}[{index}]: unsupported semantic_mode {item['semantic_mode']!r}"
            )
        if not isinstance(item["phenomena"], list) or not item["phenomena"]:
            raise ValueError(f"{path}[{index}]: phenomena must be a non-empty array")
        if item["provenance"] not in ALLOWED_PROVENANCE:
            raise ValueError(f"{path}[{index}]: unsupported provenance {item['provenance']!r}")
        if not isinstance(item["expected_mutation"], bool):
            raise ValueError(f"{path}[{index}]: expected_mutation must be boolean")
        cases.append(item)
    return cases


def audit(development: Path, held_out: Path) -> dict[str, Any]:
    dev = _load(development)
    held = _load(held_out)
    all_cases = dev + held
    ids = [str(item["id"]) for item in all_cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs must be unique across development and held-out splits")
    normalized = [str(item["utterance"]).strip().casefold() for item in all_cases]
    if len(normalized) != len(set(normalized)):
        raise ValueError("utterances must be unique across development and held-out splits")
    return {
        "development_cases": len(dev),
        "held_out_cases": len(held),
        "total_cases": len(all_cases),
        "development_families": dict(sorted(Counter(str(x["family"]) for x in dev).items())),
        "held_out_families": dict(sorted(Counter(str(x["family"]) for x in held).items())),
        "provenance": dict(sorted(Counter(str(x["provenance"]) for x in all_cases).items())),
        "held_out_utterance_overlap": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("held_out", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.development, args.held_out), indent=2, sort_keys=True))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
