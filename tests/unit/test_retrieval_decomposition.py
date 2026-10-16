import asyncio
from uuid import uuid4

import pytest

from app.retrieval.decomposition_outcome import FALLBACK_REASON
from app.retrieval.models import RetrievedChunk
from app.retrieval.query_decomposer import QueryDecomposer
from app.retrieval.service import RetrievalService


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeRepository:
    def __init__(self, responses: dict[str, list[RetrievedChunk]]) -> None:
        self.responses = responses
        self.queries: list[tuple[float, float]] = []

    async def search_similar_chunks(
        self,
        query_embedding,
        limit,
        max_distance=None,
        document_id=None,
        section_id=None,
    ):
        self.queries.append(tuple(query_embedding))

        if tuple(query_embedding) == (1.0, 0.0):
            # The fake embedding is intentionally simple; responses
            # are supplied through the test-specific repository below.
            return []

        return []


class FakeDecomposer:
    def __init__(
        self,
        sub_queries: tuple[str, ...],
        fallback_reason: str | None = None,
    ) -> None:
        self.sub_queries = sub_queries
        # Mirrors `QueryDecompositionResult.fallback_reason` -- set this to
        # simulate a provider/parsing failure that fell back to the
        # original query, exactly like the real `QueryDecomposer` would
        # produce.
        self.fallback_reason = fallback_reason
        self.received_query: str | None = None

    async def decompose(self, query: str):
        self.received_query = query

        return type(
            "Decomposition",
            (),
            {
                "sub_queries": self.sub_queries,
                "was_decomposed": len(self.sub_queries) > 1,
                "fallback_reason": self.fallback_reason,
            },
        )()


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def rerank(self, query, chunks):
        self.calls.append((query, list(chunks)))
        return list(reversed(chunks))


def _chunk(content: str, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[1],
        content=content,
        distance=distance,
    )


class QueryRepository:
    def __init__(self, responses: dict[str, list[RetrievedChunk]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def search_similar_chunks(
        self,
        query_embedding,
        limit,
        max_distance=None,
        document_id=None,
        section_id=None,
    ):
        query = query_embedding[0]
        self.calls.append(query)
        return self.responses[query]


class QueryEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.queries.append(text)
        return [text, 0.0]


@pytest.mark.asyncio
async def test_retrieval_runs_once_per_sub_query() -> None:
    first = _chunk("first", 0.2)
    second = _chunk("second", 0.1)

    repository = QueryRepository(
        {
            "query one": [first],
            "query two": [second],
        }
    )
    embedding_provider = QueryEmbeddingProvider()
    decomposer = FakeDecomposer(("query one", "query two"))
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
        query_decomposer=decomposer,
    )

    await service.search(
        "original complex query",
        limit=2,
        candidate_limit=5,
    )

    assert embedding_provider.queries == [
        "query one",
        "query two",
    ]

    assert repository.calls == [
        "query one",
        "query two",
    ]


@pytest.mark.asyncio
async def test_decomposition_and_merge_happen_even_when_trace_is_disabled() -> None:
    """Tracing is pure observability about a retrieval execution -- it must
    never be a condition for that execution. With `trace=False` (the
    default), decomposition must still run, retrieval must still run once
    per sub-query, and candidates must still be merged/deduplicated and
    globally reranked exactly as when tracing is on. Only trace *capture*
    (`service.last_trace`) may be affected by the flag."""

    first = _chunk("first", 0.2)
    second = _chunk("second", 0.1)

    repository = QueryRepository(
        {
            "query one": [first],
            "query two": [second],
        }
    )
    embedding_provider = QueryEmbeddingProvider()
    decomposer = FakeDecomposer(("query one", "query two"))
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
        query_decomposer=decomposer,
    )

    results = await service.search(
        "original complex query",
        limit=2,
        candidate_limit=5,
        trace=False,
    )

    # Decomposition and per-sub-query retrieval both still ran.
    assert embedding_provider.queries == ["query one", "query two"]
    assert repository.calls == ["query one", "query two"]

    # Candidates from both sub-queries were merged and globally reranked.
    assert len(reranker.calls) == 1
    _, reranked_candidates = reranker.calls[0]
    assert {chunk.chunk_id for chunk in reranked_candidates} == {
        first.chunk_id,
        second.chunk_id,
    }
    assert len(results) == 2

    # The only thing `trace=False` is allowed to affect: no trace captured.
    assert service.last_trace is None


@pytest.mark.asyncio
async def test_duplicate_chunks_are_deduplicated_and_best_distance_is_kept() -> None:
    shared_chunk_id = uuid4()

    first = RetrievedChunk(
        document_id=uuid4(),
        chunk_id=shared_chunk_id,
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[1],
        content="shared",
        distance=0.4,
    )

    better = RetrievedChunk(
        document_id=first.document_id,
        chunk_id=shared_chunk_id,
        section_id=first.section_id,
        section_path="Results",
        page_numbers=[1],
        content="shared",
        distance=0.1,
    )

    other = _chunk("other", 0.2)

    repository = QueryRepository(
        {
            "query one": [first],
            "query two": [better, other],
        }
    )

    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        reranker=reranker,
        query_decomposer=FakeDecomposer(("query one", "query two")),
    )

    await service.search(
        "original query",
        limit=10,
        candidate_limit=10,
        trace=True,
    )

    assert len(reranker.calls) == 1

    _, reranked_candidates = reranker.calls[0]

    assert len(reranked_candidates) == 2

    shared = next(chunk for chunk in reranked_candidates if chunk.chunk_id == shared_chunk_id)

    assert shared.distance == 0.1
    assert service.last_trace is not None
    assert service.last_trace.original_query == "original query"
    assert service.last_trace.sub_queries == ["query one", "query two"]
    assert service.last_trace.raw_candidate_count == 3
    assert service.last_trace.deduplicated_candidate_count == 2


@pytest.mark.asyncio
async def test_global_reranker_runs_once_with_original_query() -> None:
    first = _chunk("first", 0.2)
    second = _chunk("second", 0.1)

    repository = QueryRepository(
        {
            "query one": [first],
            "query two": [second],
        }
    )

    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        reranker=reranker,
        query_decomposer=FakeDecomposer(("query one", "query two")),
    )

    results = await service.search(
        "original query",
        limit=1,
        candidate_limit=10,
    )

    assert len(reranker.calls) == 1

    rerank_query, rerank_candidates = reranker.calls[0]

    assert rerank_query == "original query"
    assert len(rerank_candidates) == 2
    assert {chunk.chunk_id for chunk in rerank_candidates} == {
        first.chunk_id,
        second.chunk_id,
    }

    assert len(results) == 1
    assert results[0].chunk_id in {
        first.chunk_id,
        second.chunk_id,
    }


@pytest.mark.asyncio
async def test_decomposed_retrieval_merges_candidates_and_reranks_globally() -> None:
    first = _chunk("first", 0.2)
    shared = _chunk("shared", 0.4)
    better_shared = RetrievedChunk(
        document_id=shared.document_id,
        chunk_id=shared.chunk_id,
        section_id=shared.section_id,
        section_path=shared.section_path,
        page_numbers=shared.page_numbers,
        content=shared.content,
        distance=0.1,
    )
    third = _chunk("third", 0.3)

    repository = QueryRepository(
        {
            "query one": [first, shared],
            "query two": [better_shared, third],
        }
    )

    embedding_provider = QueryEmbeddingProvider()
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=embedding_provider,
        reranker=reranker,
        query_decomposer=FakeDecomposer(("query one", "query two")),
    )

    results = await service.search(
        "original complex query",
        limit=2,
        candidate_limit=10,
        trace=True,
    )

    # Each sub-query gets its own embedding and retrieval call.
    assert embedding_provider.queries == [
        "query one",
        "query two",
    ]
    assert repository.calls == [
        "query one",
        "query two",
    ]

    # Both retrieval results are merged and duplicate chunks are removed.
    assert len(reranker.calls) == 1

    rerank_query, rerank_candidates = reranker.calls[0]

    assert rerank_query == "original complex query"
    assert len(rerank_candidates) == 3

    shared_candidate = next(
        chunk for chunk in rerank_candidates if chunk.chunk_id == shared.chunk_id
    )
    assert shared_candidate.distance == 0.1

    # The single global reranker determines the final Top K.
    assert len(results) == 2
    assert results == list(reversed(rerank_candidates))[:2]

    # Retrieval trace exposes the decomposition and candidate accounting.
    assert service.last_trace is not None
    assert service.last_trace.original_query == "original complex query"
    assert service.last_trace.sub_queries == [
        "query one",
        "query two",
    ]
    assert service.last_trace.raw_candidate_count == 4
    assert service.last_trace.deduplicated_candidate_count == 3


@pytest.mark.asyncio
async def test_router_disabled_always_calls_decomposer() -> None:
    """Regression guard for the default (`router_enabled=False`): behavior
    must be byte-identical to before the router existed -- the decomposer is
    always called, even for a query the router would bypass if enabled."""

    chunk = _chunk("only", 0.1)
    repository = QueryRepository({"a bypassable simple query": [chunk]})
    decomposer = FakeDecomposer(("a bypassable simple query",))

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=False,
    )

    await service.search("a bypassable simple query", limit=1, candidate_limit=5)

    assert decomposer.received_query == "a bypassable simple query"
    assert service.last_router_bypassed is False


@pytest.mark.asyncio
async def test_router_enabled_bypasses_decomposer_for_simple_query() -> None:
    chunk = _chunk("only", 0.1)
    repository = QueryRepository(
        {"What token budget does chunking target?": [chunk]}
    )
    decomposer = FakeDecomposer(("should not be used",))

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=True,
    )

    results = await service.search(
        "What token budget does chunking target?",
        limit=1,
        candidate_limit=5,
    )

    # The decomposition provider was never consulted.
    assert decomposer.received_query is None
    assert service.last_router_bypassed is True
    assert len(results) == 1
    assert results[0].chunk_id == chunk.chunk_id


@pytest.mark.asyncio
async def test_router_enabled_still_decomposes_complex_query() -> None:
    first = _chunk("first", 0.2)
    second = _chunk("second", 0.1)

    repository = QueryRepository(
        {
            "query one": [first],
            "query two": [second],
        }
    )
    decomposer = FakeDecomposer(("query one", "query two"))

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=True,
    )

    compound_query = (
        "What are the main failure modes of query decomposition, and what "
        "mitigation does the system apply for each one?"
    )

    await service.search(compound_query, limit=2, candidate_limit=5)

    assert decomposer.received_query == compound_query
    assert service.last_router_bypassed is False
    assert repository.calls == ["query one", "query two"]


@pytest.mark.asyncio
async def test_router_bypass_still_reranks_with_original_query() -> None:
    chunk = _chunk("only", 0.1)
    simple_query = "How many dimensions does the embedding model output?"
    repository = QueryRepository({simple_query: [chunk]})
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        reranker=reranker,
        query_decomposer=FakeDecomposer(("should not be used",)),
        router_enabled=True,
    )

    await service.search(simple_query, limit=1, candidate_limit=5)

    assert len(reranker.calls) == 1
    rerank_query, _ = reranker.calls[0]
    assert rerank_query == simple_query


class _UnreachableProvider:
    """A QueryDecompositionProvider that fails the test if it is ever
    called -- used to prove the empty-query ValueError is raised before any
    provider call, with or without the router enabled."""

    async def decompose(self, query: str) -> list[str]:
        raise AssertionError("provider.decompose must not be called for an empty query")


@pytest.mark.asyncio
async def test_router_enabled_preserves_empty_query_valueerror() -> None:
    """The router must never swallow QueryDecomposer's hard failure on an
    empty/whitespace query -- `should_decompose("")` returns True precisely
    so this case always falls through to the real decomposer, which raises,
    regardless of whether the router is enabled."""

    service = RetrievalService(
        repository=QueryRepository({}),
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=QueryDecomposer(provider=_UnreachableProvider()),
        router_enabled=True,
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        await service.search("   ", limit=1, candidate_limit=5)


SIMPLE_QUERY = "What distance operators does pgvector provide for comparing vectors?"
COMPLEX_QUERY = (
    "What are the main failure modes of query decomposition, and what "
    "mitigation does the system apply for each one?"
)


@pytest.mark.asyncio
async def test_per_call_router_enabled_true_overrides_instance_default_false() -> None:
    """A per-call `router_enabled=True` must bypass decomposition for a
    simple query even though the service instance itself was constructed
    with the application default (`router_enabled=False`)."""

    chunk = _chunk("only", 0.1)
    repository = QueryRepository({SIMPLE_QUERY: [chunk]})
    decomposer = FakeDecomposer(("should not be used",))

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=False,
    )

    await service.search(SIMPLE_QUERY, limit=1, candidate_limit=5, router_enabled=True)

    assert decomposer.received_query is None
    assert service.last_router_bypassed is True


@pytest.mark.asyncio
async def test_per_call_router_enabled_false_overrides_instance_default_true() -> None:
    """A per-call `router_enabled=False` must force decomposition to run
    even though the service instance itself was constructed with the
    router enabled."""

    chunk = _chunk("only", 0.1)
    repository = QueryRepository({SIMPLE_QUERY: [chunk]})
    decomposer = FakeDecomposer((SIMPLE_QUERY,))

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=True,
    )

    await service.search(SIMPLE_QUERY, limit=1, candidate_limit=5, router_enabled=False)

    assert decomposer.received_query == SIMPLE_QUERY
    assert service.last_router_bypassed is False


@pytest.mark.asyncio
async def test_per_call_router_enabled_none_falls_back_to_instance_default() -> None:
    """Omitting the per-call override (the default, `None`) must preserve
    whatever the service instance was constructed with -- this is the
    exact case every existing call site (including every other test in
    this file) already relies on."""

    chunk = _chunk("only", 0.1)
    repository = QueryRepository({SIMPLE_QUERY: [chunk]})
    decomposer = FakeDecomposer(("should not be used",))

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=True,
    )

    # No `router_enabled` kwarg passed at all.
    await service.search(SIMPLE_QUERY, limit=1, candidate_limit=5)

    assert decomposer.received_query is None
    assert service.last_router_bypassed is True


@pytest.mark.asyncio
async def test_per_call_router_override_still_decomposes_a_complex_query() -> None:
    """A per-call override to `True` must still decompose a query the
    deterministic router itself classifies as needing it -- the override
    only changes whether the router runs at all, never its rules."""

    first = _chunk("first", 0.2)
    second = _chunk("second", 0.1)
    repository = QueryRepository({"query one": [first], "query two": [second]})
    decomposer = FakeDecomposer(("query one", "query two"))

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=False,
    )

    await service.search(COMPLEX_QUERY, limit=2, candidate_limit=5, router_enabled=True)

    assert decomposer.received_query == COMPLEX_QUERY
    assert service.last_router_bypassed is False
    assert repository.calls == ["query one", "query two"]


@pytest.mark.asyncio
async def test_concurrent_calls_with_opposite_router_overrides_do_not_leak() -> None:
    """Two concurrent `search()` calls on the SAME service instance, with
    opposite per-call router overrides, must not influence each other --
    `router_enabled` is a local variable threaded through `_search`, never
    an assignment to `self.router_enabled` or any other shared state."""

    chunk_simple = _chunk("simple-result", 0.1)
    chunk_complex_a = _chunk("complex-a", 0.2)
    chunk_complex_b = _chunk("complex-b", 0.15)

    repository = QueryRepository(
        {
            SIMPLE_QUERY: [chunk_simple],
            "query one": [chunk_complex_a],
            "query two": [chunk_complex_b],
        }
    )
    decomposer = FakeDecomposer(("query one", "query two"))

    # Constructed with the application default OFF, matching production's
    # actual default -- both calls below rely entirely on their own
    # per-call override, not on the instance's own setting.
    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=False,
    )

    bypass_call = service.search(SIMPLE_QUERY, limit=1, candidate_limit=5, router_enabled=True)
    decompose_call = service.search(
        COMPLEX_QUERY, limit=2, candidate_limit=5, router_enabled=False
    )

    bypass_results, decompose_results = await asyncio.gather(bypass_call, decompose_call)

    # The simple-query call bypassed decomposition entirely...
    assert [r.chunk_id for r in bypass_results] == [chunk_simple.chunk_id]
    # ...while the complex-query call still decomposed, regardless of
    # interleaving -- neither call's override leaked into the other's
    # execution or into the shared `decomposer`/`repository` fakes.
    assert {r.chunk_id for r in decompose_results} == {
        chunk_complex_a.chunk_id,
        chunk_complex_b.chunk_id,
    }


@pytest.mark.asyncio
async def test_decomposition_outcome_status_bypassed() -> None:
    """`last_decomposition_outcome.status` is "bypassed" exactly when the
    router actually skipped the decomposer -- never inferred from
    sub_queries length or query text."""

    chunk = _chunk("only", 0.1)
    repository = QueryRepository({SIMPLE_QUERY: [chunk]})
    decomposer = FakeDecomposer(("should not be used",))
    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=True,
    )

    await service.search(SIMPLE_QUERY, limit=1, candidate_limit=5)

    outcome = service.last_decomposition_outcome
    assert outcome is not None
    assert outcome.status == "bypassed"
    assert outcome.original_query == SIMPLE_QUERY
    assert outcome.sub_queries == [SIMPLE_QUERY]
    assert outcome.reason is None


@pytest.mark.asyncio
async def test_decomposition_outcome_status_decomposed() -> None:
    """`status` is "decomposed" exactly when the decomposer actually
    produced more than one sub-query -- the sub_queries list is the real,
    order-preserved output, never reconstructed."""

    first = _chunk("first", 0.2)
    second = _chunk("second", 0.1)
    repository = QueryRepository({"query one": [first], "query two": [second]})
    decomposer = FakeDecomposer(("query one", "query two"))
    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=False,
    )

    await service.search(COMPLEX_QUERY, limit=2, candidate_limit=5)

    outcome = service.last_decomposition_outcome
    assert outcome is not None
    assert outcome.status == "decomposed"
    assert outcome.original_query == COMPLEX_QUERY
    assert outcome.sub_queries == ["query one", "query two"]
    assert outcome.reason is None


@pytest.mark.asyncio
async def test_decomposition_outcome_status_not_needed() -> None:
    """`status` is "not_needed" when the decomposer ran (the router didn't
    bypass it) and itself decided a single sub-query was sufficient --
    distinct from "bypassed", even though `sub_queries` looks identical."""

    chunk = _chunk("only", 0.1)
    repository = QueryRepository({SIMPLE_QUERY: [chunk]})
    decomposer = FakeDecomposer((SIMPLE_QUERY,))
    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=False,
    )

    await service.search(SIMPLE_QUERY, limit=1, candidate_limit=5)

    outcome = service.last_decomposition_outcome
    assert outcome is not None
    assert outcome.status == "not_needed"
    assert outcome.sub_queries == [SIMPLE_QUERY]
    assert outcome.reason is None


@pytest.mark.asyncio
async def test_decomposition_outcome_status_fallback_uses_safe_reason() -> None:
    """`status` is "fallback" when the decomposer's provider failed --
    `reason` must be the safe, generic constant, NEVER the raw exception
    text the provider raised."""

    chunk = _chunk("only", 0.1)
    repository = QueryRepository({SIMPLE_QUERY: [chunk]})
    decomposer = FakeDecomposer(
        (SIMPLE_QUERY,),
        fallback_reason="connection reset by peer at 10.0.0.5:443",
    )
    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=False,
    )

    await service.search(SIMPLE_QUERY, limit=1, candidate_limit=5)

    outcome = service.last_decomposition_outcome
    assert outcome is not None
    assert outcome.status == "fallback"
    assert outcome.sub_queries == [SIMPLE_QUERY]
    assert outcome.reason == FALLBACK_REASON
    # The raw exception text must never leak into the user-facing reason.
    assert "10.0.0.5" not in (outcome.reason or "")


@pytest.mark.asyncio
async def test_decomposition_outcome_present_even_without_trace() -> None:
    """`last_decomposition_outcome` must be set regardless of `trace` --
    the entire point of this milestone."""

    chunk = _chunk("only", 0.1)
    repository = QueryRepository({SIMPLE_QUERY: [chunk]})
    decomposer = FakeDecomposer((SIMPLE_QUERY,))
    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=False,
    )

    await service.search(SIMPLE_QUERY, limit=1, candidate_limit=5, trace=False)

    assert service.last_trace is None
    assert service.last_decomposition_outcome is not None


@pytest.mark.asyncio
async def test_decomposition_outcome_does_not_leak_between_requests() -> None:
    """A second call's outcome must fully replace the first's -- no stale
    status/sub_queries/reason can survive across requests on the same
    instance."""

    chunk_a = _chunk("a", 0.1)
    chunk_b = _chunk("b", 0.1)
    repository = QueryRepository({"query a": [chunk_a], SIMPLE_QUERY: [chunk_b]})
    decomposer = FakeDecomposer(("query a",))
    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        query_decomposer=decomposer,
        router_enabled=False,
    )

    await service.search("first query", limit=1, candidate_limit=5)
    first_outcome = service.last_decomposition_outcome
    assert first_outcome is not None
    assert first_outcome.original_query == "first query"

    decomposer.sub_queries = (SIMPLE_QUERY,)
    await service.search(SIMPLE_QUERY, limit=1, candidate_limit=5)
    second_outcome = service.last_decomposition_outcome

    assert second_outcome is not None
    assert second_outcome.original_query == SIMPLE_QUERY
    assert second_outcome is not first_outcome
