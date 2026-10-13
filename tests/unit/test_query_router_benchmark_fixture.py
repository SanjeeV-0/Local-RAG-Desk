"""Compares the deterministic router's decisions against the
`expected_subqueries` labels in the existing V2 retrieval benchmark fixture.

This is a decision-agreement check only -- it calls `should_decompose` as a
pure function against the 102 fixture queries and does not touch the
database, the embedding model, the reranker, or the decomposition LLM. It is
deliberately NOT a retrieval-quality benchmark run (see Stage 2 report for
the full breakdown and the rationale for the known, accepted disagreements
asserted below).
"""

import json
from pathlib import Path

from app.retrieval.query_router import should_decompose

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "fixtures"
    / "benchmark_v2"
    / "questions_v2.json"
)

# Cases where the router disagrees with the fixture's `expected_subqueries`
# label. Every one of these is a *conservative* disagreement -- the router
# decomposes a query the fixture labeled as not needing it -- never the
# reverse (a query the fixture says needed decomposition getting bypassed).
# See the Stage 2 report for the per-case rationale; this set is intentional
# and is asserted here so a future rule change that regresses it (in either
# direction) is caught.
EXPECTED_FALSE_DECOMPOSE_IDS = {
    "q009",
    "q024",
    "q035",
    "q036",
    "q040",
    "q045",
    "q046",
    "q048",
    "q050",
    "q051",
    "q052",
    "q074",
    "q076",
    "q077",
    "q079",
    "q080",
}


def _load_cases() -> list[dict]:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_router_has_zero_false_bypasses_against_benchmark_fixture() -> None:
    """No fixture case labeled as needing decomposition
    (`expected_subqueries` non-empty) may be bypassed by the router. A false
    bypass -- skipping decomposition a query actually needed -- is the risk
    direction the router design must never produce."""

    cases = _load_cases()

    false_bypasses = [
        case["id"]
        for case in cases
        if case.get("expected_subqueries") and not should_decompose(case["query"])
    ]

    assert false_bypasses == []


def test_router_disagreements_match_known_conservative_set() -> None:
    """Pins the exact set of (benign, over-cautious) disagreements so this
    test fails loudly -- for review, not silent drift -- if a future rule
    change shrinks or grows that set."""

    cases = _load_cases()

    disagreements = {
        case["id"]
        for case in cases
        if bool(case.get("expected_subqueries")) != should_decompose(case["query"])
    }

    assert disagreements == EXPECTED_FALSE_DECOMPOSE_IDS


def test_router_bypasses_majority_of_benchmark_fixture() -> None:
    """Sanity check that the router is actually doing useful work -- not a
    strict quality bar, just a guard against a future change making it
    decompose nearly everything."""

    cases = _load_cases()

    bypass_count = sum(1 for case in cases if not should_decompose(case["query"]))

    assert bypass_count / len(cases) > 0.7
