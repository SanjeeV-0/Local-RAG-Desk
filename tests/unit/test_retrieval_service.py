from uuid import uuid4

import pytest

from app.generation.context import ContextAssembler
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import RetrievalService


class FakeEmbeddingProvider:
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeRepository:
    def __init__(
        self,
        chunks: list[RetrievedChunk],
    ) -> None:
        self.chunks = chunks
        self.received_limit = None

    async def search_similar_chunks(
        self,
        query_embedding,
        limit,
        max_distance=None,
        document_id=None,
        section_id=None,
    ):
        self.received_limit = limit
        return self.chunks


class FakeReranker:
    def __init__(self) -> None:
        self.received_query = None
        self.received_chunks = None

    def rerank(self, query, chunks):
        self.received_query = query
        self.received_chunks = list(chunks)

        return list(reversed(chunks))


def _chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[1],
        content=f"chunk {index}",
        distance=0.1,
    )


@pytest.mark.asyncio
async def test_retrieval_service_reranks_candidates() -> None:
    chunks = [
        _chunk(1),
        _chunk(2),
        _chunk(3),
    ]

    repository = FakeRepository(chunks)
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        reranker=reranker,
    )

    results = await service.search(
        "research query",
        limit=2,
        candidate_limit=3,
    )

    assert repository.received_limit == 3
    assert reranker.received_query == "research query"
    assert reranker.received_chunks == chunks

    assert [result.content for result in results] == [
        "chunk 3",
        "chunk 2",
    ]


@pytest.mark.asyncio
async def test_candidate_limit_cannot_be_smaller_than_limit() -> None:
    service = RetrievalService(
        repository=FakeRepository([]),
        embedding_provider=FakeEmbeddingProvider(),
        reranker=FakeReranker(),
    )

    with pytest.raises(ValueError):
        await service.search(
            "research query",
            limit=10,
            candidate_limit=5,
        )


@pytest.mark.asyncio
async def test_retrieval_service_trace_contains_generation_context() -> None:
    chunks = [
        _chunk(1),
        _chunk(2),
        _chunk(3),
    ]

    repository = FakeRepository(chunks)
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        reranker=reranker,
        context_assembler=ContextAssembler(),
    )

    results = await service.search(
        "research query",
        limit=2,
        candidate_limit=3,
        trace=True,
    )

    assert [result.content for result in results] == [
        "chunk 3",
        "chunk 2",
    ]

    assert service.last_trace is not None

    trace = service.last_trace

    assert len(trace.candidates) == 3
    assert len(trace.final_results) == 2

    assert trace.context is not None

    assert trace.context.text == ("[Source 1]\nchunk 3\n\n[Source 2]\nchunk 2")

    assert len(trace.context.sources) == 2

    assert [source.chunk_id for source in trace.context.sources] == [
        chunks[2].chunk_id,
        chunks[1].chunk_id,
    ]


@pytest.mark.asyncio
async def test_retrieval_service_clears_previous_trace() -> None:
    chunks = [_chunk(1)]

    repository = FakeRepository(chunks)
    reranker = FakeReranker()

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        reranker=reranker,
    )

    await service.search(
        "research query",
        trace=True,
    )

    assert service.last_trace is not None

    await service.search(
        "another query",
        trace=False,
    )

    assert service.last_trace is None


@pytest.mark.asyncio
async def test_default_candidate_limit_floor_matches_previous_hard_coded_value() -> None:
    """Constructing RetrievalService without `candidate_limit_floor` (every
    existing call site before this milestone) must preserve the exact
    previous hard-coded behavior: candidate_limit defaults to
    max(limit, 50)."""

    repository = FakeRepository([_chunk(1)])

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
    )

    await service.search("query", limit=5)

    assert repository.received_limit == 50


@pytest.mark.asyncio
async def test_configured_candidate_limit_floor_is_actually_used() -> None:
    repository = FakeRepository([_chunk(1)])

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        candidate_limit_floor=200,
    )

    await service.search("query", limit=5)

    assert repository.received_limit == 200


@pytest.mark.asyncio
async def test_configured_candidate_limit_floor_does_not_override_a_larger_explicit_limit() -> None:
    """The floor is a MINIMUM, not a cap -- an explicit `limit` larger than
    the configured floor must still win (matches `max(limit, floor)`)."""

    repository = FakeRepository([_chunk(1)])

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        candidate_limit_floor=10,
    )

    await service.search("query", limit=30)

    assert repository.received_limit == 30
