"""HTTP boundary for the Retrieval Explorer debugging tool: a thin
translation layer over `RetrievalService.search`. Contains no retrieval
logic itself -- its only job is computing `effective_trace` (the
request-level `trace` flag ANDed with the server-side `Settings.trace_enabled`
ceiling) and converting between `RetrievalService`'s dataclasses and this
module's Pydantic response schemas. This endpoint never calls generation --
see `app.api.research` for the RAG endpoint.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.config.settings import Settings
from app.core.dependencies import (
    get_app_settings,
    get_retrieval_service,
)
from app.privacy.pii_detector import redact
from app.privacy.schemas import RedactionOutcome
from app.retrieval.schemas import (
    DecompositionOutcomeResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievalTraceCandidateResponse,
    RetrievalTraceContextResponse,
    RetrievalTraceResponse,
    RetrievedChunkResponse,
)
from app.retrieval.service import RetrievalService

router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
)


def _trace_candidate_response(
    candidate,
) -> RetrievalTraceCandidateResponse:
    """Convert a trace candidate into an API response."""

    return RetrievalTraceCandidateResponse(
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        section_id=candidate.section_id,
        section_path=candidate.section_path,
        page_numbers=candidate.page_numbers,
        content=candidate.content,
        distance=candidate.distance,
        rerank_score=candidate.rerank_score,
    )


@router.post(
    "/search",
    response_model=RetrievalSearchResponse,
)
async def search(
    request: RetrievalSearchRequest,
    retrieval_service: Annotated[
        RetrievalService,
        Depends(get_retrieval_service),
    ],
    settings: Annotated[
        Settings,
        Depends(get_app_settings),
    ],
) -> RetrievalSearchResponse:
    """Search for relevant document chunks.

    `Settings.trace_enabled` is a server-side capability ceiling, and
    `request.trace` is the per-request ask -- trace is only captured when
    both are true (`effective_trace`). This never gates retrieval or query
    decomposition themselves, which always run: `effective_trace` only
    controls whether `RetrievalService` records a trace of that execution.
    """

    effective_trace = request.trace and settings.trace_enabled

    results = await retrieval_service.search(
        query=request.query,
        limit=request.limit,
        document_id=request.document_id,
        section_id=request.section_id,
        trace=effective_trace,
        router_enabled=request.router_enabled,
        candidate_limit=request.candidate_limit,
    )

    # Always set by `RetrievalService._search` on every call that reaches
    # the decomposition decision point -- independent of `trace`/
    # `effective_trace`, so this is present on every successful response.
    decomposition_outcome = retrieval_service.last_decomposition_outcome
    assert decomposition_outcome is not None, (
        "RetrievalService.search() must always set last_decomposition_outcome "
        "before returning successfully."
    )

    trace_response = None

    trace = retrieval_service.last_trace

    if trace is not None:
        context_response = None

        if trace.context is not None:
            context_response = RetrievalTraceContextResponse(
                text=trace.context.text,
                sources=[_trace_candidate_response(source) for source in trace.context.sources],
            )

        trace_response = RetrievalTraceResponse(
            query=trace.query,
            original_query=trace.original_query,
            sub_queries=trace.sub_queries,
            candidate_limit=trace.candidate_limit,
            raw_candidate_count=trace.raw_candidate_count,
            deduplicated_candidate_count=trace.deduplicated_candidate_count,
            candidates=[_trace_candidate_response(candidate) for candidate in trace.candidates],
            final_results=[_trace_candidate_response(result) for result in trace.final_results],
            context=context_response,
            router_bypassed=trace.router_bypassed,
        )

    trace_unavailable_reason = (
        "server_disabled" if request.trace and not settings.trace_enabled else None
    )

    redacted_source_count = 0
    response_results: list[RetrievedChunkResponse] = []

    for result in results:
        content_redaction = redact(result.content)

        if content_redaction.redacted:
            redacted_source_count += 1

        response_results.append(
            RetrievedChunkResponse(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                section_id=result.section_id,
                section_path=result.section_path,
                page_numbers=result.page_numbers,
                content=content_redaction.text,
                distance=result.distance,
                similarity=result.similarity,
                rerank_score=result.rerank_score,
            )
        )

    return RetrievalSearchResponse(
        results=response_results,
        decomposition=DecompositionOutcomeResponse(
            original_query=decomposition_outcome.original_query,
            sub_queries=decomposition_outcome.sub_queries,
            status=decomposition_outcome.status,
            reason=decomposition_outcome.reason,
        ),
        redaction=RedactionOutcome(
            redaction_applied=redacted_source_count > 0,
            affected_source_count=redacted_source_count,
            answer_redacted=False,
            possibly_incomplete=redacted_source_count > 0,
        ),
        trace_requested=request.trace,
        trace_available=trace_response is not None,
        trace_unavailable_reason=trace_unavailable_reason,
        trace=trace_response,
    )
