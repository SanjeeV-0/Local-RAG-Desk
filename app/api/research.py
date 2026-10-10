"""HTTP boundary for RAG question-answering: retrieval -> context assembly
-> generation, orchestrated here but implemented entirely by
`RetrievalService`, `ContextAssembler`, and `GenerationService` -- this
module owns only the HTTP contract and the trace-availability metadata
(same `effective_trace` rule as `app.api.retrieval`), never any retrieval,
decomposition, or generation logic itself. Response is a plain
`dict[str, object]` rather than a declared Pydantic `response_model`, kept
that way to preserve exact backward-compatible key presence/absence
(e.g. `"trace"` is omitted, not `null`, when unavailable).
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config.settings import Settings
from app.core.dependencies import (
    get_app_settings,
    get_generation_service,
    get_retrieval_service,
)
from app.generation.context import ContextAssembler
from app.generation.generation_service import GenerationService
from app.observability.langfuse import get_langfuse
from app.retrieval.service import RetrievalService

router = APIRouter(
    prefix="/research",
    tags=["research"],
)


class ResearchRequest(BaseModel):
    """`trace` is a per-request ask for trace capture -- combined with the
    server-side `Settings.trace_enabled` capability ceiling as
    `request.trace AND settings.trace_enabled` (see `ask` below), the same
    rule `POST /retrieval/search` uses. It never affects whether query
    decomposition or retrieval runs."""

    query: str = Field(min_length=1)
    trace: bool = False


@router.post("/ask")
async def ask(
    request: ResearchRequest,
    retrieval_service: Annotated[
        RetrievalService,
        Depends(get_retrieval_service),
    ],
    generation_service: Annotated[
        GenerationService,
        Depends(get_generation_service),
    ],
    settings: Annotated[
        Settings,
        Depends(get_app_settings),
    ],
) -> dict[str, object]:
    """Answer a research question using retrieved context.

    `Settings.trace_enabled` is a server-side capability ceiling, and
    `request.trace` is the per-request ask -- trace is only captured when
    both are true (`effective_trace`), following the same rule as
    `POST /retrieval/search`. This never gates retrieval or query
    decomposition themselves, which always run regardless of tracing.
    """

    query = request.query
    trace_requested = request.trace
    effective_trace = trace_requested and settings.trace_enabled
    langfuse = get_langfuse()

    if langfuse is None:
        return await _execute_research(
            query=query,
            retrieval_service=retrieval_service,
            generation_service=generation_service,
            trace_requested=trace_requested,
            effective_trace=effective_trace,
            result_limit=settings.research_ask_result_limit,
        )

    with langfuse.start_as_current_observation(
        as_type="span",
        name="research-request",
        input={
            "query": query,
            "trace": trace_requested,
        },
    ) as observation:
        try:
            response = await _execute_research(
                query=query,
                retrieval_service=retrieval_service,
                generation_service=generation_service,
                trace_requested=trace_requested,
                effective_trace=effective_trace,
                result_limit=settings.research_ask_result_limit,
            )

            observation.update(
                output={
                    "answer": response["answer"],
                    "model": response["model"],
                    "source_count": len(response["sources"]),
                },
            )

            return response

        except Exception as exc:
            observation.update(
                level="ERROR",
                status_message=str(exc),
            )
            raise


async def _execute_research(
    query: str,
    retrieval_service: RetrievalService,
    generation_service: GenerationService,
    trace_requested: bool = False,
    effective_trace: bool = False,
    result_limit: int = 10,
) -> dict[str, object]:
    """Execute retrieval, context assembly, and generation.

    `result_limit` defaults to the previously hard-coded literal (10), so
    any direct call that omits it (e.g. existing tests) preserves today's
    exact behavior -- the production route always passes
    `Settings.research_ask_result_limit` explicitly.
    """

    results = await retrieval_service.search(
        query=query,
        limit=result_limit,
        trace=effective_trace,
    )

    context = ContextAssembler().assemble(results)

    generation_result = await generation_service.generate(
        query=query,
        context=context,
    )

    response: dict[str, object] = {
        "answer": generation_result.text,
        "model": generation_result.model,
        "sources": [
            {
                "document_id": source.document_id,
                "chunk_id": source.chunk_id,
                "section_id": source.section_id,
                "section_path": source.section_path,
                "page_numbers": source.page_numbers,
            }
            for source in context.sources
        ],
        "trace_requested": trace_requested,
        "trace_available": False,
        "trace_unavailable_reason": (
            "server_disabled" if trace_requested and not effective_trace else None
        ),
    }

    if effective_trace and retrieval_service.last_trace is not None:
        response["trace_available"] = True
        response["trace_unavailable_reason"] = None
        response["trace"] = {
            "query": retrieval_service.last_trace.query,
            "original_query": retrieval_service.last_trace.original_query,
            "sub_queries": retrieval_service.last_trace.sub_queries,
            "candidate_limit": retrieval_service.last_trace.candidate_limit,
            "raw_candidate_count": (retrieval_service.last_trace.raw_candidate_count),
            "deduplicated_candidate_count": (
                retrieval_service.last_trace.deduplicated_candidate_count
            ),
            "candidates": [
                {
                    "document_id": candidate.document_id,
                    "chunk_id": candidate.chunk_id,
                    "section_id": candidate.section_id,
                    "section_path": candidate.section_path,
                    "page_numbers": candidate.page_numbers,
                    "content": candidate.content,
                    "distance": candidate.distance,
                    "rerank_score": candidate.rerank_score,
                }
                for candidate in retrieval_service.last_trace.candidates
            ],
            "final_results": [
                {
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "section_id": result.section_id,
                    "section_path": result.section_path,
                    "page_numbers": result.page_numbers,
                    "content": result.content,
                    "distance": result.distance,
                    "rerank_score": result.rerank_score,
                }
                for result in retrieval_service.last_trace.final_results
            ],
            "context": (
                {
                    "text": retrieval_service.last_trace.context.text,
                    "sources": [
                        {
                            "document_id": source.document_id,
                            "chunk_id": source.chunk_id,
                            "section_id": source.section_id,
                            "section_path": source.section_path,
                            "page_numbers": source.page_numbers,
                            "content": source.content,
                            "distance": source.distance,
                            "rerank_score": source.rerank_score,
                        }
                        for source in retrieval_service.last_trace.context.sources
                    ],
                }
                if retrieval_service.last_trace.context is not None
                else None
            ),
        }

    return response
