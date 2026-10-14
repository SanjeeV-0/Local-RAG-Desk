from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalSearchRequest(BaseModel):
    """Request payload for vector retrieval.

    `trace` is a per-request ask for trace capture -- it is always combined
    with the server-side `Settings.trace_enabled` capability ceiling as
    `request.trace AND settings.trace_enabled` (see `app.api.retrieval`).
    Neither this flag nor the server setting ever affects whether query
    decomposition or retrieval itself runs; tracing is observability only.

    `router_enabled` is an optional per-request override of the server-side
    `Settings.query_router_enabled` default (OFF). `None` (the default, and
    what every existing caller that omits this field gets) means "use the
    configured application default" -- omitting it is fully backward
    compatible. `True`/`False` controls the deterministic query router for
    THIS request only; it never mutates the application-wide setting and
    has no effect on any other request.
    """

    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    document_id: UUID | None = None
    section_id: UUID | None = None
    trace: bool = False
    router_enabled: bool | None = None


class RetrievedChunkResponse(BaseModel):
    """API representation of a retrieved chunk.

    `section_id`/`section_path` are nullable: a headingless document's
    chunks have no section (the repository's outer join to DocumentSection
    keeps them retrievable rather than excluding them), so this must accept
    and serialize that as JSON null rather than rejecting the response.
    """

    document_id: UUID
    chunk_id: UUID
    section_id: UUID | None
    section_path: str | None
    page_numbers: list[int]
    content: str
    distance: float
    similarity: float
    rerank_score: float | None


class RetrievalTraceCandidateResponse(BaseModel):
    """API representation of a traced retrieval candidate.

    See `RetrievedChunkResponse` -- `section_id`/`section_path` are
    nullable for the same headingless-document reason.
    """

    document_id: UUID
    chunk_id: UUID
    section_id: UUID | None
    section_path: str | None
    page_numbers: list[int]
    content: str
    distance: float
    rerank_score: float | None


class RetrievalTraceContextResponse(BaseModel):
    """API representation of context prepared for generation."""

    text: str
    sources: list[RetrievalTraceCandidateResponse]


class RetrievalTraceResponse(BaseModel):
    """Debug trace for a retrieval operation.

    `router_bypassed` is `True` exactly when the deterministic query router
    was enabled (via the server default or this request's `router_enabled`
    override) AND it actually decided to skip the decomposition call for
    `original_query`. It is `False` whenever the router was disabled, or
    enabled but decomposition still ran (including when the router merely
    found nothing to bypass). This reflects the actual executed decision
    (`RetrievalService.last_router_bypassed`/`RetrievalTrace.router_bypassed`)
    -- it is never inferred from `sub_queries` length or query text.
    """

    query: str
    original_query: str
    sub_queries: list[str]
    candidate_limit: int
    raw_candidate_count: int
    deduplicated_candidate_count: int
    candidates: list[RetrievalTraceCandidateResponse]
    final_results: list[RetrievalTraceCandidateResponse]
    context: RetrievalTraceContextResponse | None = None
    router_bypassed: bool


class RetrievalSearchResponse(BaseModel):
    """Response payload for vector retrieval.

    `trace_requested`/`trace_available`/`trace_unavailable_reason` let the
    caller distinguish "trace wasn't asked for" from "trace was asked for
    but the server has tracing disabled" without inferring server capability
    from `trace == null` alone. `trace_available` is true exactly when
    `trace` is populated. `trace_unavailable_reason` is `"server_disabled"`
    when trace was requested but `Settings.trace_enabled` is false, and
    `None` otherwise (including when trace wasn't requested at all).
    """

    results: list[RetrievedChunkResponse]
    trace_requested: bool
    trace_available: bool
    trace_unavailable_reason: Literal["server_disabled"] | None = None
    trace: RetrievalTraceResponse | None = None
