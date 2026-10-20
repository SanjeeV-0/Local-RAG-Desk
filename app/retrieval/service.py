"""The retrieval pipeline: decomposition -> per-query vector search -> merge
-> global rerank (and/or RRF fusion) -> Top K (-> optional trace capture).

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

`ranking_strategy` selects how the final, limit-truncated result order is
produced from the reranked candidate pool:
  - `"rerank"` (default -- today's exact, unchanged behavior): the
    cross-encoder reranker's own score order, truncated to `limit`.
  - `"rrf"`: Reciprocal Rank Fusion (`app.retrieval.rrf.fuse_rankings`)
    combining the pre-rerank vector-distance ranking with the post-rerank
    cross-encoder ranking, truncated to `limit` only AFTER fusion. See
    `app.retrieval.rrf`'s module docstring for the formula and the
    deterministic tie-break rule. Vector rank, for both single-query and
    decomposed-query retrieval, is defined identically either way: a
    chunk's 1-indexed position in `candidates` -- the ALREADY-MERGED,
    already-distance-sorted pool produced by the existing per-sub-query
    merge step below (best/lowest distance wins across sub-queries,
    unchanged). RRF never alters that merge policy; it only consumes its
    output as one of its two input rankings.
"""

import asyncio
import dataclasses
from typing import Literal
from uuid import UUID

from app.core.repositories.document import DocumentRepository
from app.embeddings.provider import EmbeddingProvider
from app.generation.context import ContextAssembler
from app.observability.langfuse import get_langfuse
from app.privacy.pii_detector import redact
from app.retrieval.decomposition_outcome import (
    FALLBACK_REASON,
    DecompositionOutcome,
    DecompositionStatus,
)
from app.retrieval.models import RetrievedChunk
from app.retrieval.query_decomposer import QueryDecomposer
from app.retrieval.query_router import should_decompose
from app.retrieval.reranker import Reranker
from app.retrieval.rrf import DEFAULT_RRF_K, fuse_rankings
from app.retrieval.trace import (
    RetrievalTrace,
    RetrievalTraceCandidate,
    RetrievalTraceContext,
)

RankingStrategy = Literal["rerank", "rrf"]


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
        ranking_strategy: RankingStrategy = "rerank",
        rrf_k: int = DEFAULT_RRF_K,
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
        # Defaults to "rerank" -- today's exact, unchanged behavior for
        # every existing caller/test that doesn't pass this argument.
        # Settings-backed via `Settings.retrieval_ranking_strategy` in
        # production (see app.core.dependencies.get_retrieval_service and
        # this module's own docstring for what "rrf" changes).
        self.ranking_strategy = ranking_strategy
        # Defaults to the standard RRF smoothing constant (60); irrelevant
        # when `ranking_strategy="rerank"`. Settings-backed via
        # `Settings.retrieval_rrf_k`.
        self.rrf_k = rrf_k
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

        # Redact PII-shaped spans in the query before it reaches anything
        # external (the decomposition OpenRouter call) or observational
        # (Langfuse, trace, decomposition outcome). Retrieval/embedding runs
        # entirely locally (no network call -- see
        # `app.embeddings.sentence_transformer`), so using the redacted
        # query for embedding too only affects the rare case where the query
        # itself contained a PII-shaped span; it never changes behavior for
        # the overwhelming majority of ordinary queries.
        query = redact(query).text

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
                # Never send the raw exception text to Langfuse -- only a
                # safe category. `ProviderUnavailableError` (raised by the
                # decomposition provider on an OpenRouter failure, though it
                # is normally already caught and absorbed by
                # `QueryDecomposer.decompose`) carries a safe category name;
                # any other exception type reports only its class name,
                # never its message.
                category = getattr(exc, "category", None)
                safe_status = category.value if category is not None else exc.__class__.__name__
                observation.update(
                    level="ERROR",
                    status_message=safe_status,
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
            # The input `query` was already redacted above, so the
            # decomposer itself never sees a raw PII-shaped span -- this
            # second pass guards against the (expected to be rare) case
            # where the decomposer's own output reconstructs or echoes a
            # PII-shaped pattern that wasn't present in its (already
            # redacted) input.
            retrieval_queries = tuple(redact(q).text for q in decomposition.sub_queries)

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
            # Run off the event loop: `embed_text` is a synchronous,
            # CPU-bound sentence-transformers call -- calling it directly
            # here would block every other in-flight request (including
            # unrelated health checks) for the duration of inference.
            # `asyncio.to_thread` preserves the exact return value and
            # exception behavior, just off the main thread.
            query_embedding = await asyncio.to_thread(
                self.embedding_provider.embed_text,
                retrieval_query,
            )

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

                # Accumulate every sub-query whose search surfaced this
                # chunk -- NOT just the one tied to whichever occurrence
                # ends up kept below. A chunk retrieved by two sub-queries
                # should report both, even though only its best-distance
                # occurrence determines ranking.
                matched_sub_queries = (
                    existing.matched_sub_queries if existing is not None else ()
                )
                if retrieval_query not in matched_sub_queries:
                    matched_sub_queries = (*matched_sub_queries, retrieval_query)

                # Merge by chunk_id across sub-queries, keeping each chunk's
                # best (lowest-distance) occurrence rather than the first one
                # seen -- a chunk surfaced by two sub-queries should be
                # ranked by its strongest match, not an arbitrary one.
                if existing is None or candidate.distance < existing.distance:
                    merged_candidates[candidate.chunk_id] = dataclasses.replace(
                        candidate, matched_sub_queries=matched_sub_queries
                    )
                else:
                    merged_candidates[candidate.chunk_id] = dataclasses.replace(
                        existing, matched_sub_queries=matched_sub_queries
                    )

        candidates = sorted(
            merged_candidates.values(),
            key=lambda chunk: chunk.distance,
        )
        # Note for `ranking_strategy="rrf"` below: this sort is already
        # fully deterministic on a distance tie -- `sorted()` is stable, and
        # `merged_candidates.values()`'s iteration order is itself
        # deterministic (insertion order, driven by the fixed order
        # `retrieval_queries` is processed in and the repository's own
        # deterministic result order), so no additional tie-break key is
        # needed here. `fuse_rankings` assigns vector rank from this list's
        # position, so it inherits that same determinism unchanged.
        deduplicated_candidate_count = len(merged_candidates)

        if self.reranker is None:
            final_results = candidates[:limit]
        else:
            # One global rerank pass over the WHOLE deduplicated pool, using
            # the caller's original query (never an individual sub-query) --
            # this is what makes cross-sub-query ranking comparable. Reranking
            # per sub-query would score each pool against a different query
            # and produce incomparable scores.
            # Same rationale as the embed_text call above: CrossEncoder.predict
            # is a synchronous, CPU-bound call over the whole candidate pool
            # and must not block the event loop.
            reranked = await asyncio.to_thread(
                self.reranker.rerank,
                query,
                candidates,
            )

            if self.ranking_strategy == "rrf":
                # Fuse the pre-rerank vector ranking (`candidates`) with the
                # post-rerank cross-encoder ranking (`reranked`) -- both are
                # orderings of the exact same deduplicated pool, so
                # `fuse_rankings` can always find a matching rank for every
                # chunk in both (see its own ValueError guards for what
                # happens if that invariant is ever violated by a future
                # reranker implementation). Truncation to `limit` happens
                # only AFTER fusion, never before.
                final_results = fuse_rankings(candidates, reranked, k=self.rrf_k)[:limit]
            else:
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
        """Convert a retrieved chunk into trace metadata.

        `content` is redacted here -- this is the single shared conversion
        point behind `last_trace.candidates`, `last_trace.final_results`,
        and `last_trace.context.sources`, so none of them can ever diverge
        from the redaction already applied to the generation context itself.
        """

        return RetrievalTraceCandidate(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            page_numbers=chunk.page_numbers,
            content=redact(chunk.content).text,
            distance=chunk.distance,
            rerank_score=chunk.rerank_score,
            matched_sub_queries=list(chunk.matched_sub_queries),
        )
