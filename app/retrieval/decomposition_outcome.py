"""Always-on, trace-independent summary of what query decomposition
actually did for one retrieval execution.

Built and read on every `RetrievalService.search()` call regardless of
whether diagnostic tracing (`trace=True`) is enabled -- unlike
`app.retrieval.trace.RetrievalTrace` (full candidate/score diagnostics,
only built when tracing is on), this exists specifically so a user can see
whether their query was decomposed, bypassed, or fell back, without ever
needing Trace Mode. Kept as a separate, smaller object rather than folded
into `RetrievalTrace` so the two can be read/served independently: HTTP
responses include this on every request, and only additionally include the
full trace when tracing is both requested and server-enabled.
"""

from dataclasses import dataclass, field
from typing import Literal

DecompositionStatus = Literal["bypassed", "decomposed", "not_needed", "fallback"]

# Shown to users when decomposition fell back to the original query after a
# provider/parsing failure. Deliberately generic -- the real exception text
# (`QueryDecompositionResult.fallback_reason`, see
# `app.retrieval.query_decomposer.QueryDecomposer.decompose`) is never
# surfaced in this user-facing field. Note: that raw detail is not
# currently captured by any logging/observability mechanism either (a
# pre-existing gap, unrelated to and not addressed by this field) -- it is
# simply discarded once this safe constant is chosen in its place.
FALLBACK_REASON = (
    "Automatic query decomposition was unavailable for this request; "
    "the original query was used directly."
)


@dataclass
class DecompositionOutcome:
    """What actually happened for ONE retrieval execution's query planning.

    `status` meanings:
      - "bypassed": the deterministic query router decided this query was
        confidently single-intent and skipped the decomposer entirely.
      - "decomposed": the decomposer ran and produced more than one
        retrieval sub-query.
      - "not_needed": the decomposer ran (the router did not bypass it,
        either because it's disabled or because it didn't bypass this
        query) and decided a single sub-query was sufficient.
      - "fallback": the decomposer ran, the underlying provider/parsing
        failed, and retrieval fell back to the original query alone.

    `sub_queries` is always the actual, real, order-preserved list of
    queries used for retrieval -- never reconstructed or guessed; it is
    always `[original_query]` for every status except "decomposed".
    """

    original_query: str
    sub_queries: list[str] = field(default_factory=list)
    status: DecompositionStatus = "not_needed"
    reason: str | None = None
