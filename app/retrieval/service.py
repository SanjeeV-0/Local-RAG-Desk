"""The retrieval pipeline: decomposition -> per-query vector search -> merge
-> global rerank -> Top K (-> optional trace capture).

This is the one and only retrieval implementation in the application --
both `POST /retrieval/search` (the debugging "Retrieval Explorer") and
`POST /research/ask` (RAG) call `RetrievalService.search` and never
duplicate any of this logic themselves. It owns query decomposition
orchestration, candidate merging/deduplication, and reranking; it does
NOT own context assembly or LLM generation (see `app.generation`) and does
NOT own the HTTP contract (see `app.api.retrieval` / `app.api.research`).

Key invariant verified by `tests/unit/test_retrieval_decomposition.py`:
decomposition and retrieval run identically regardless of `trace` -- the
`trace` flag only controls whether `self.last_trace` is populated.
"""

from uuid import UUID

from app.core.repositories.document import DocumentRepository
from app.embeddings.provider import EmbeddingProvider
from app.generation.context import ContextAssembler
from app.observability.langfuse import get_langfuse
from app.retrieval.decomposition_outcome import (
    FALLBACK_REASON,
    DecompositionOutcome,
    DecompositionStatus,
)
from app.retrieval.models import RetrievedChunk
from app.retrieval.query_decomposer import QueryDecomposer
from app.retrieval.query_router import should_decompose
from app.retrieval.reranker import Reranker
from app.retrieval.trace import (
    RetrievalTrace,
    RetrievalTraceCandidate,
    RetrievalTraceContext,
)


class RetrievalService:
    """Retrieves and reranks document chunks."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker | None = None,
        context_assembler: ContextAssembler | None = None,
        query_decomposer: QueryDecomposer | None = None,
        candidate_limit_floor: int = 50,
        router_enabled: bool = False,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.context_assembler = context_assembler or ContextAssembler()
        self.query_decomposer = query_decomposer
        # Defaults to False, so constructing a RetrievalService without this
        # argument (every existing call site, including unit tests)
        # preserves today's exact behavior: decomposition always runs when a
        # decomposer is configured. Settings-backed via
        # `Settings.query_router_enabled` in production (see
        # app.core.dependencies.get_retrieval_service).
        self.router_enabled = router_enabled
        # Defaults to the existing hard-coded literal (50), so constructing
        # a RetrievalService without this argument (every existing call
        # site, including unit tests) preserves today's exact behavior.
        # Settings-backed via `Settings.retrieval_candidate_limit_floor` in
        # production (see app.core.dependencies.get_retrieval_service).
        self.candidate_limit_floor = candidate_limit_floor
        self.last_trace: RetrievalTrace | None = None
        # Set on every `search()` call (unlike `last_trace`, which is only
        # populated when `trace=True`) so the router's decision can be
        # reported in the Langfuse observation below regardless of the
        # trace flag.
        self.last_router_bypassed: bool = False
        # Same "always set, trace-independent" rule as `last_router_bypassed`
        # -- read by both HTTP endpoints on every successful request so the
        # decomposition outcome is visible without Trace Mode. See
        # `app.retrieval.decomposition_outcome.DecompositionOutcome`.
        self.last_decomposition_outcome: DecompositionOutcome | None = None

    async def search(
        self,
        query: str,
        limit: int = 10,
        max_distance: float | None = None,
        document_id: UUID | None = None,
        section_id: UUID | None = None,
        candidate_limit: int | None = None,
        trace: bool = False,
        router_enabled: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and optionally rerank document chunks.

        `router_enabled` is an optional PER-CALL override of `self.router_enabled`
        (itself resolved once, at construction time, from `Settings.query_router_enabled`
        -- see `app.core.dependencies.get_retrieval_service`). `None` (the default)
        means "use whatever this service instance was constructed with"; `True`/`False`
        overrides it for this call only. This is a plain local value threaded through
        to `_search` -- it never mutates `self.router_enabled` or any other shared
        state, so concurrent calls on the same (or different) `RetrievalService`
        instances with different overrides cannot affect one another.
        """

        self.last_trace = None
        self.last_decomposition_outcome = None

        if candidate_limit is None:
            candidate_limit = max(limit, self.candidate_limit_floor)

        if candidate_limit < limit:
            raise ValueError("candidate_limit must be greater than or equal to limit.")

        langfuse = get_langfuse()

        if langfuse is None:
            return await self._search(
                query=query,
                limit=limit,
                max_distance=max_distance,
                document_id=document_id,
                section_id=section_id,
                candidate_limit=candidate_limit,
                trace=trace,
                router_enabled=router_enabled,
            )

        with langfuse.start_as_current_observation(
            as_type="retriever",
            name="document-retrieval",
            input={
                "query": query,
                "limit": limit,
                "candidate_limit": candidate_limit,
                "max_distance": max_distance,
                "document_id": (str(document_id) if document_id is not None else None),
                "section_id": (str(section_id) if section_id is not None else None),
            },
        ) as observation:
            try:
                results = await self._search(
                    query=query,
                    limit=limit,
                    max_distance=max_distance,
                    document_id=document_id,
                    section_id=section_id,
                    candidate_limit=candidate_limit,
                    trace=trace,
                    router_enabled=router_enabled,
                )

                observation.update(
                    output={
                        "result_count": len(results),
                        "router_bypassed": self.last_router_bypassed,
                        "results": [
                            {
                                "chunk_id": str(result.chunk_id),
                                "document_id": str(result.document_id),
                                "section_id": str(result.section_id),
                                "section_path": (result.section_path),
                                "page_numbers": (result.page_numbers),
                                "distance": result.distance,
                                "rerank_score": (result.rerank_score),
                            }
                            for result in results
                        ],
                    }
                )

                return results

            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=str(exc),
                )
                raise

    async def _search(
        self,
        query: str,
        limit: int,
        max_distance: float | None,
        document_id: UUID | None,
        section_id: UUID | None,
        candidate_limit: int,
        trace: bool,
        router_enabled: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Execute retrieval without observability concerns.

        `trace` is read ONLY in the final `if trace:` block below -- it must
        never gate decomposition or the retrieval loop above it. Query
        planning (decomposition) and candidate retrieval are the actual
        execution; tracing is a side channel that observes that execution
        after the fact.
        """

        # Per-call override (`None` falls back to the instance default set at
        # construction time). A plain local variable, not an assignment to
        # `self.router_enabled` -- nothing here is shared across calls or
        # mutated, so concurrent requests (even sharing a `RetrievalService`
        # instance, which they don't in production -- see
        # `app.core.dependencies.get_retrieval_service`) can never observe or
        # influence each other's effective router setting.
        effective_router_enabled = (
            self.router_enabled if router_enabled is None else router_enabled
        )

        router_bypassed = False

        if self.query_decomposer is None:
            # Untriggered in production -- `get_retrieval_service` always
            # wires a decomposer; this branch only exists for unit tests
            # that construct `RetrievalService` without one. No split ever
            # happens here, so "not_needed" is the accurate status.
            retrieval_queries = (query,)
            decomposition_status: DecompositionStatus = "not_needed"
            decomposition_reason: str | None = None
        elif effective_router_enabled and not should_decompose(query):
            # The deterministic router decided this query is confidently
            # single-intent -- skip the QueryDecomposer call (and therefore
            # the decomposition LLM call) entirely. `retrieval_queries` ends
            # up identical to what QueryDecomposer would have returned for a
            # query it did not decompose: `(query,)`.
            retrieval_queries = (query,)
            router_bypassed = True
            decomposition_status = "bypassed"
            decomposition_reason = None
        else:
            # Decomposition always runs here, unconditionally, whenever a
            # decomposer is configured and the router is disabled (or did
            # not bypass) -- this is the one call site that decides whether
            # retrieval is single-query or multi-query, and it has no
            # dependency on `trace` whatsoever.
            decomposition = await self.query_decomposer.decompose(query)
            retrieval_queries = decomposition.sub_queries

            if decomposition.fallback_reason is not None:
                # The provider/parsing failed inside QueryDecomposer, which
                # already fell back to the original query alone. The real
                # exception text (`decomposition.fallback_reason`) is
                # deliberately never surfaced here -- only the safe, generic
                # `FALLBACK_REASON` constant is user-facing.
                decomposition_status = "fallback"
                decomposition_reason = FALLBACK_REASON
            elif decomposition.was_decomposed:
                decomposition_status = "decomposed"
                decomposition_reason = None
            else:
                decomposition_status = "not_needed"
                decomposition_reason = None

        self.last_router_bypassed = router_bypassed
        # Always overwritten here, on every call that reaches this point --
        # never conditional on `trace`, and never left over from a previous
        # call, so no stale outcome can leak into this request's response.
        self.last_decomposition_outcome = DecompositionOutcome(
            original_query=query,
            sub_queries=list(retrieval_queries),
            status=decomposition_status,
            reason=decomposition_reason,
        )

        merged_candidates: dict[UUID, RetrievedChunk] = {}

        raw_candidate_count = 0
        merged_candidates = {}

        for retrieval_query in retrieval_queries:
            query_embedding = self.embedding_provider.embed_text(retrieval_query)

            # candidate_limit applies PER sub-query, not split across them --
            # N sub-queries can yield up to N * candidate_limit raw results
            # before merging.
            candidates = await self.repository.search_similar_chunks(
                query_embedding=query_embedding,
                limit=candidate_limit,
                max_distance=max_distance,
                document_id=document_id,
                section_id=section_id,
            )
            raw_candidate_count += len(candidates)

            for candidate in candidates:
                existing = merged_candidates.get(candidate.chunk_id)

                # Merge by chunk_id across sub-queries, keeping each chunk's
                # best (lowest-distance) occurrence rather than the first one
                # seen -- a chunk surfaced by two sub-queries should be
                # ranked by its strongest match, not an arbitrary one.
                if existing is None or candidate.distance < existing.distance:
                    merged_candidates[candidate.chunk_id] = candidate

        candidates = sorted(
            merged_candidates.values(),
            key=lambda chunk: chunk.distance,
        )
        deduplicated_candidate_count = len(merged_candidates)

        if self.reranker is None:
            final_results = candidates[:limit]
        else:
            # One global rerank pass over the WHOLE deduplicated pool, using
            # the caller's original query (never an individual sub-query) --
            # this is what makes cross-sub-query ranking comparable. Reranking
            # per sub-query would score each pool against a different query
            # and produce incomparable scores.
            reranked = self.reranker.rerank(
                query,
                candidates,
            )

            final_results = reranked[:limit]

        if trace:
            # `candidates` here is the deduplicated pool BEFORE reranking, so
            # its trace entries carry a cosine `distance` but a null
            # `rerank_score` (RetrievedChunk.rerank_score defaults to None
            # until the reranker sets it). `final_results` is the post-rerank
            # list, so those entries DO carry a populated `rerank_score`.
            # This asymmetry is intentional, not a bug: the candidate pool is
            # "what was available", final_results is "what the reranker
            # chose and how it scored it".
            self.last_trace = RetrievalTrace(
                query=query,
                original_query=query,
                sub_queries=list(retrieval_queries),
                raw_candidate_count=raw_candidate_count,
                deduplicated_candidate_count=deduplicated_candidate_count,
                candidate_limit=candidate_limit,
                candidates=[self._to_trace_candidate(chunk) for chunk in candidates],
                final_results=[self._to_trace_candidate(chunk) for chunk in final_results],
                router_bypassed=router_bypassed,
            )

            generation_context = self.context_assembler.assemble(final_results)

            self.last_trace.context = RetrievalTraceContext(
                text=generation_context.text,
                sources=[self._to_trace_candidate(chunk) for chunk in final_results],
            )

        return final_results

    @staticmethod
    def _to_trace_candidate(
        chunk: RetrievedChunk,
    ) -> RetrievalTraceCandidate:
        """Convert a retrieved chunk into trace metadata."""

        return RetrievalTraceCandidate(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            page_numbers=chunk.page_numbers,
            content=chunk.content,
            distance=chunk.distance,
            rerank_score=chunk.rerank_score,
        )
