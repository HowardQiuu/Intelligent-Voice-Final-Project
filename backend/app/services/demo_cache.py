from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_demo_cases() -> list[dict]:
    with (DATA_DIR / "demo_cases.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_demo_results() -> dict:
    with (DATA_DIR / "demo_results.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def get_case(case_id: str) -> dict:
    cases = load_demo_cases()
    for case in cases:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def get_result(case_id: str) -> dict:
    results = load_demo_results()
    if case_id not in results:
        raise KeyError(case_id)
    return results[case_id]
