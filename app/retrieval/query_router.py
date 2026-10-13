"""Deterministic, rule-based gate on whether to run query decomposition.

`should_decompose` is a pure function with no I/O and no model calls -- it is
a cheap syntactic pre-filter in front of `QueryDecomposer`, not a replacement
for it. It only decides *whether* `QueryDecomposer.decompose` is called;
`QueryDecomposer`'s own normalization and failure-fallback behavior
(app.retrieval.query_decomposer) is unchanged and still the sole authority on
what retrieval queries actually run once decomposition is invoked.

Design goal (see Stage 1 audit): bypass decomposition only for confidently
single-intent queries, and default to decomposition for anything else,
including ambiguous cases. A false "decompose" costs one extra LLM call; a
false "bypass" costs retrieval quality on a query that needed splitting --
so every rule below is biased toward the former, never the latter. Query
length is deliberately not used as a signal (see MILESTONE_BRIEF.md / audit
sec. 2): it does not reliably distinguish simple from compound queries in
this domain's query set.

Rules are validated against the 102-case fixture in
`tests/evaluation/fixtures/benchmark_v2/questions_v2.json` -- see
`tests/unit/test_query_router_benchmark_fixture.py` and the Stage 2 report
for the full agreement/disagreement breakdown.
"""

import re

# A second, distinct question embedded in the same query ("What is X? How
# does Y work?") is an unambiguous multi-intent signal.
_MULTIPLE_QUESTION_MARKS_THRESHOLD = 1

# Explicit requests to enumerate or aggregate over multiple items -- these
# describe an aggregation request regardless of sentence shape.
_ENUMERATION_PHRASES = (
    "list all",
    "list the",
    "enumerate",
    "each of the following",
)

# A compound question: a clause introduced by ", and" / ", or" immediately
# followed by its own interrogative/auxiliary word, e.g. "..., and what
# mitigation does the system apply?" or "..., and how does that compare?".
# This is the single most reliable syntactic signal in the benchmark fixture
# for "this query actually contains two separate information requests".
_WH_WORDS = (
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
    "does",
    "is",
    "can",
    "are",
    "will",
    "should",
)
_COMPOUND_CLAUSE_RE = re.compile(
    r",\s*(?:and|or)\s+(?:" + "|".join(_WH_WORDS) + r")\b",
    re.IGNORECASE,
)

# An Oxford-style list of three or more distinct items ("A, B, and C") is an
# aggregation/comparison request spanning multiple named things, as opposed
# to a single "X and Y" pairing (e.g. "difference between X and Y" is a
# single relationship question, not three).
_TRAILING_LIST_RE = re.compile(r",\s*(?:and|or)\s", re.IGNORECASE)
_MIN_COMMAS_FOR_LIST = 2

# Three or more parallel "<noun phrase> from <noun phrase> from ..."
# fragments enumerate distinct items without using a comma-separated list
# (e.g. "an embedding provider failure from an index-related failure from a
# reranker failure"). Two or more occurrences of " from " is a narrow but,
# per the benchmark fixture, zero-false-positive signal for this shape.
_MIN_FROM_OCCURRENCES_FOR_ENUMERATION = 2


def should_decompose(query: str) -> bool:
    """Return True if `query` should be routed through `QueryDecomposer`.

    Returns True (run decomposition) for empty/whitespace-only input too --
    the router must never change `QueryDecomposer.decompose`'s existing
    `ValueError` on an empty query, so it defers to that behavior rather
    than short-circuiting it.
    """

    text = query.strip()

    if not text:
        return True

    lower = text.lower()

    if text.count("?") > _MULTIPLE_QUESTION_MARKS_THRESHOLD:
        return True

    if any(phrase in lower for phrase in _ENUMERATION_PHRASES):
        return True

    if _COMPOUND_CLAUSE_RE.search(text):
        return True

    if text.count(",") >= _MIN_COMMAS_FOR_LIST and _TRAILING_LIST_RE.search(text):
        return True

    if lower.count(" from ") >= _MIN_FROM_OCCURRENCES_FOR_ENUMERATION:
        return True

    return False
