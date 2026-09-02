import json
from pathlib import Path


def test_frozen_semantic_corpora_assign_every_case_to_a_family() -> None:
    for filename in ("semantic_dev.json", "semantic_heldout.json"):
        cases = json.loads((Path("evaluation") / filename).read_text(encoding="utf-8"))
        assert cases
        assert all(isinstance(case.get("family"), str) and case["family"] for case in cases)


def test_development_and_heldout_corpora_keep_distinct_case_ids() -> None:
    development = {
        case["id"]
        for case in json.loads(Path("evaluation/semantic_dev.json").read_text(encoding="utf-8"))
    }
    heldout = {
        case["id"]
        for case in json.loads(Path("evaluation/semantic_heldout.json").read_text(encoding="utf-8"))
    }

    assert development.isdisjoint(heldout)
