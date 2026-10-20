"""Internal (non-Pydantic) data shape for one retrieval execution's debug
trace. Built by `app.retrieval.service.RetrievalService` only when
`trace=True` is passed to `search()`/`_search()`; converted to the API
response shape (`RetrievalTraceResponse` et al.) by `app.api.retrieval` and
`app.api.research`. Kept separate from the Pydantic schemas so the service
layer has no FastAPI/Pydantic dependency.
"""

from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class RetrievalTraceCandidate:
    """Trace information for a retrieved candidate.

    `section_id`/`section_path` are `None` for a headingless document's
    chunks -- see `app.retrieval.models.RetrievedChunk`.
    """

    chunk_id: UUID
    document_id: UUID
    section_id: UUID | None
    section_path: str | None
    page_numbers: list[int]
    content: str
    distance: float
    rerank_score: float | None = None
    matched_sub_queries: list[str] = field(default_factory=list)


@dataclass
class RetrievalTraceContext:
    """Trace information for context selected for generation."""

    text: str
    sources: list[RetrievalTraceCandidate] = field(default_factory=list)


@dataclass
class RetrievalTrace:
    """Debug trace for a complete retrieval operation."""

    query: str

    candidate_limit: int

    candidates: list[RetrievalTraceCandidate] = field(default_factory=list)

    final_results: list[RetrievalTraceCandidate] = field(default_factory=list)

    context: RetrievalTraceContext | None = None

    original_query: str = ""
    sub_queries: list[str] = field(default_factory=list)
    raw_candidate_count: int = 0
    deduplicated_candidate_count: int = 0

    # True when the deterministic query router (app.retrieval.query_router)
    # decided this query was confidently single-intent and skipped the
    # QueryDecomposer call entirely. False whenever the router is disabled,
    # or decomposition ran (whether or not it actually produced more than
    # one sub-query).
    router_bypassed: bool = False
