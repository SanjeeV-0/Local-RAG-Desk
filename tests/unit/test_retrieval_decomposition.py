from uuid import uuid4

import pytest

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
    def __init__(self, sub_queries: tuple[str, ...]) -> None:
        self.sub_queries = sub_queries
        self.received_query: str | None = None

    async def decompose(self, query: str):
        self.received_query = query

        return type(
            "Decomposition",
            (),
            {"sub_queries": self.sub_queries},
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
