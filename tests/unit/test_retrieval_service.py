from uuid import uuid4

import pytest

from app.generation.context import ContextAssembler
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import RetrievalService


class FakeEmbeddingProvider:
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeRepository:
    """Mirrors `DocumentRepository.search_similar_chunks`'s actual filter
    semantics closely enough to catch a service-layer regression without a
    real database: `document_id`/`section_id`, when provided, are compared
    directly against each chunk's own `document_id`/`section_id` -- exactly
    like the real repository's `DocumentChunk.document_id ==
    document_id` / `DocumentChunk.section_id == section_id` WHERE clauses."""

    def __init__(
        self,
        chunks: list[RetrievedChunk],
    ) -> None:
        self.chunks = chunks
        self.received_limit = None
        self.received_document_id = None
        self.received_section_id = None

    async def search_similar_chunks(
        self,
        query_embedding,
        limit,
        max_distance=None,
        document_id=None,
        section_id=None,
    ):
        self.received_limit = limit
        self.received_document_id = document_id
        self.received_section_id = section_id

        results = self.chunks

        if document_id is not None:
            results = [c for c in results if c.document_id == document_id]

        if section_id is not None:
            results = [c for c in results if c.section_id == section_id]

        return results


class FakeReranker:
    def __init__(self) -> None:
        self.received_query = None
        self.received_chunks = None

    def rerank(self, query, chunks):
        self.received_query = query
        self.received_chunks = list(chunks)

        return list(reversed(chunks))


def _chunk(index: int, document_id=None, section_id=None) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id if document_id is not None else uuid4(),
        chunk_id=uuid4(),
        section_id=section_id if section_id is not None else uuid4(),
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


@pytest.mark.asyncio
async def test_unfiltered_retrieval_returns_results_across_all_documents() -> None:
    """Baseline: with no `document_id`, retrieval must continue to work
    exactly as before -- chunks from any document are eligible."""

    doc_a, doc_b = uuid4(), uuid4()
    chunks = [_chunk(1, document_id=doc_a), _chunk(2, document_id=doc_b)]

    repository = FakeRepository(chunks)
    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = await service.search("query", limit=10, candidate_limit=10)

    assert repository.received_document_id is None
    assert {r.content for r in results} == {"chunk 1", "chunk 2"}


@pytest.mark.asyncio
async def test_document_filter_returns_only_that_documents_chunks() -> None:
    """Selecting a document must return its matching chunks -- and only
    its chunks, never another document's."""

    target_document = uuid4()
    other_document = uuid4()
    chunks = [
        _chunk(1, document_id=target_document),
        _chunk(2, document_id=other_document),
    ]

    repository = FakeRepository(chunks)
    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = await service.search(
        "query",
        limit=10,
        candidate_limit=10,
        document_id=target_document,
    )

    assert repository.received_document_id == target_document
    assert len(results) == 1
    assert results[0].content == "chunk 1"
    assert results[0].document_id == target_document


@pytest.mark.asyncio
async def test_document_filter_with_no_matching_chunks_returns_empty() -> None:
    """A document with no relevant (or no) chunks must correctly return no
    results -- not an error, not another document's chunks."""

    chunks = [_chunk(1, document_id=uuid4())]  # belongs to a different document
    unrelated_document = uuid4()

    repository = FakeRepository(chunks)
    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = await service.search(
        "query",
        limit=10,
        candidate_limit=10,
        document_id=unrelated_document,
    )

    assert results == []


@pytest.mark.asyncio
async def test_document_filter_does_not_exclude_the_selected_documents_valid_chunks() -> None:
    """Multiple valid chunks belonging to the selected document must all
    survive the filter -- it must never accidentally over-exclude."""

    target_document = uuid4()
    chunks = [
        _chunk(1, document_id=target_document),
        _chunk(2, document_id=target_document),
        _chunk(3, document_id=uuid4()),
    ]

    repository = FakeRepository(chunks)
    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = await service.search(
        "query",
        limit=10,
        candidate_limit=10,
        document_id=target_document,
    )

    assert {r.content for r in results} == {"chunk 1", "chunk 2"}
    assert all(r.document_id == target_document for r in results)


@pytest.mark.asyncio
async def test_section_filter_still_behaves_correctly() -> None:
    """Section filtering must continue to work independently of (and
    unaffected by) the document-filter fix."""

    target_section = uuid4()
    chunks = [
        _chunk(1, section_id=target_section),
        _chunk(2, section_id=uuid4()),
    ]

    repository = FakeRepository(chunks)
    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = await service.search(
        "query",
        limit=10,
        candidate_limit=10,
        section_id=target_section,
    )

    assert repository.received_section_id == target_section
    assert len(results) == 1
    assert results[0].content == "chunk 1"
    assert results[0].section_id == target_section


@pytest.mark.asyncio
async def test_combined_document_and_section_filter_both_apply() -> None:
    target_document = uuid4()
    target_section = uuid4()
    chunks = [
        _chunk(1, document_id=target_document, section_id=target_section),
        _chunk(2, document_id=target_document, section_id=uuid4()),
        _chunk(3, document_id=uuid4(), section_id=target_section),
    ]

    repository = FakeRepository(chunks)
    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
    )

    results = await service.search(
        "query",
        limit=10,
        candidate_limit=10,
        document_id=target_document,
        section_id=target_section,
    )

    assert len(results) == 1
    assert results[0].content == "chunk 1"
