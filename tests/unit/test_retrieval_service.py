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


# ---------------------------------------------------------------------------
# ranking_strategy="rrf"
# ---------------------------------------------------------------------------


class ScoredReranker:
    """Reorders by a caller-supplied chunk_id -> score mapping (descending),
    exactly matching `app.retrieval.reranker.Reranker`'s contract."""

    def __init__(self, score_by_chunk_id: dict) -> None:
        self.score_by_chunk_id = score_by_chunk_id
        self.received_chunks: list[RetrievedChunk] | None = None

    def rerank(self, query, chunks):
        self.received_chunks = list(chunks)
        scored = sorted(
            chunks,
            key=lambda c: self.score_by_chunk_id[c.chunk_id],
            reverse=True,
        )
        return [
            RetrievedChunk(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                section_id=c.section_id,
                section_path=c.section_path,
                page_numbers=c.page_numbers,
                content=c.content,
                distance=c.distance,
                rerank_score=self.score_by_chunk_id[c.chunk_id],
            )
            for c in scored
        ]


def _chunk_with_distance(content: str, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[1],
        content=content,
        distance=distance,
    )


@pytest.mark.asyncio
async def test_rerank_is_the_default_strategy_when_unspecified() -> None:
    """Constructing a RetrievalService without `ranking_strategy` must keep
    today's exact behavior -- the regression this whole test file already
    enforces, re-asserted explicitly here for clarity."""
    service = RetrievalService(
        repository=FakeRepository([]),
        embedding_provider=FakeEmbeddingProvider(),
        reranker=FakeReranker(),
    )
    assert service.ranking_strategy == "rerank"


@pytest.mark.asyncio
async def test_rrf_strategy_fuses_vector_and_reranker_rankings() -> None:
    # Three candidates: `vector_first` is the closest vector match but the
    # reranker scores it worst; `reranker_first` is the reranker's top pick
    # but a weak vector match; `middling` is mediocre on both. RRF should
    # favor whichever balances both rankings best -- here, `vector_first`
    # (vector rank 1, reranker rank 3) vs `reranker_first` (vector rank 3,
    # reranker rank 1) are symmetric and must tie exactly; `middling`
    # (vector rank 2, reranker rank 2) must win outright under k=60, since
    # 1/61+1/62 (two middling ranks) is NOT guaranteed to beat a symmetric
    # 1/61+1/63 pair in general, so this test only asserts the two
    # structural facts that must hold regardless of exact arithmetic.
    vector_first = _chunk_with_distance("vector_first", distance=0.1)
    middling = _chunk_with_distance("middling", distance=0.2)
    reranker_first = _chunk_with_distance("reranker_first", distance=0.3)

    chunks = [vector_first, middling, reranker_first]
    repository = FakeRepository(chunks)
    reranker = ScoredReranker(
        {
            reranker_first.chunk_id: 3.0,
            middling.chunk_id: 2.0,
            vector_first.chunk_id: 1.0,
        }
    )

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        reranker=reranker,
        ranking_strategy="rrf",
    )

    results = await service.search("query", limit=3, candidate_limit=3)

    result_ids = {r.chunk_id for r in results}
    assert result_ids == {vector_first.chunk_id, middling.chunk_id, reranker_first.chunk_id}
    # The purely-reranked order would be [reranker_first, middling, vector_first];
    # RRF must NOT simply reproduce that order unchanged -- vector_first's
    # strong vector rank must pull it up from last place.
    reranked_only_order = [reranker_first.chunk_id, middling.chunk_id, vector_first.chunk_id]
    assert [r.chunk_id for r in results] != reranked_only_order


@pytest.mark.asyncio
async def test_rrf_applies_final_limit_only_after_fusion() -> None:
    chunks = [_chunk_with_distance(f"c{i}", distance=0.1 * i) for i in range(5)]
    repository = FakeRepository(chunks)
    # Reranker reverses vector order entirely.
    reranker = ScoredReranker({c.chunk_id: float(i) for i, c in enumerate(chunks)})

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        reranker=reranker,
        ranking_strategy="rrf",
    )

    results = await service.search("query", limit=2, candidate_limit=5)

    assert len(results) == 2
    # Both candidate lists `ScoredReranker` saw must be the FULL pool (5),
    # confirming fusion ran over everything before the limit=2 truncation.
    assert len(reranker.received_chunks) == 5


@pytest.mark.asyncio
async def test_rrf_candidate_set_matches_vector_and_reranker_pools_exactly() -> None:
    """RRF must never be handed mismatched candidate sets -- both inputs to
    `fuse_rankings` come from the same deduplicated pool by construction;
    this test pins that invariant at the service-integration level."""
    chunks = [_chunk_with_distance(f"c{i}", distance=0.1 * i) for i in range(4)]
    repository = FakeRepository(chunks)
    reranker = ScoredReranker({c.chunk_id: float(i) for i, c in enumerate(chunks)})

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        reranker=reranker,
        ranking_strategy="rrf",
    )

    results = await service.search("query", limit=4, candidate_limit=4)

    assert {r.chunk_id for r in results} == {c.chunk_id for c in chunks}


@pytest.mark.asyncio
async def test_rrf_trace_candidates_remain_the_pre_rerank_vector_pool() -> None:
    """Trace semantics must not change under RRF: `trace.candidates` is
    always the deduplicated, distance-sorted pool BEFORE any reranking or
    fusion -- only `final_results`'s composition/order is strategy-
    dependent."""
    chunks = [_chunk_with_distance(f"c{i}", distance=0.1 * i) for i in range(3)]
    repository = FakeRepository(chunks)
    reranker = ScoredReranker({c.chunk_id: float(i) for i, c in enumerate(chunks)})

    service = RetrievalService(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(),
        reranker=reranker,
        ranking_strategy="rrf",
        context_assembler=ContextAssembler(),
    )

    await service.search("query", limit=2, candidate_limit=3, trace=True)

    trace = service.last_trace
    assert trace is not None
    # Pre-rerank pool is vector-distance order, unaffected by RRF.
    assert [c.chunk_id for c in trace.candidates] == [c.chunk_id for c in chunks]
    assert len(trace.final_results) == 2


@pytest.mark.asyncio
async def test_rrf_with_decomposed_query_fuses_over_the_existing_merged_pool() -> None:
    """RRF must consume the EXISTING per-sub-query merge policy's output
    (best-distance-wins across sub-queries) unchanged -- never a different
    merge strategy of its own."""

    class QueryRepository:
        def __init__(self, responses: dict) -> None:
            self.responses = responses

        async def search_similar_chunks(
            self, query_embedding, limit, max_distance=None, document_id=None, section_id=None
        ):
            return self.responses[query_embedding[0]]

    class QueryEmbeddingProvider:
        def embed_text(self, text: str) -> list[float]:
            return [text, 0.0]

    class FixedSubqueryDecomposer:
        def __init__(self, sub_queries):
            self.sub_queries = sub_queries

        async def decompose(self, query: str):
            return type(
                "Decomposition",
                (),
                {
                    "sub_queries": self.sub_queries,
                    "was_decomposed": len(self.sub_queries) > 1,
                    "fallback_reason": None,
                },
            )()

    shared = _chunk_with_distance("shared", distance=0.5)  # will appear in both sub-query results
    only_q1 = _chunk_with_distance("only_q1", distance=0.1)
    only_q2 = _chunk_with_distance("only_q2", distance=0.2)

    # `shared` has a better (lower) distance under "query two" -- the
    # existing merge rule must keep that better distance.
    repository = QueryRepository(
        {
            "query one": [only_q1, RetrievedChunk(**{**shared.__dict__, "distance": 0.9})],
            "query two": [only_q2, shared],
        }
    )
    reranker = ScoredReranker(
        {
            only_q1.chunk_id: 1.0,
            only_q2.chunk_id: 2.0,
            shared.chunk_id: 3.0,
        }
    )

    service = RetrievalService(
        repository=repository,
        embedding_provider=QueryEmbeddingProvider(),
        reranker=reranker,
        query_decomposer=FixedSubqueryDecomposer(("query one", "query two")),
        ranking_strategy="rrf",
    )

    results = await service.search("original query", limit=3, candidate_limit=5, trace=True)

    assert {r.chunk_id for r in results} == {only_q1.chunk_id, only_q2.chunk_id, shared.chunk_id}

    # The merged pool's vector order must reflect the BEST distance for
    # `shared` (0.5, from "query two"), not the worse one from "query one".
    trace = service.last_trace
    shared_trace_candidate = next(c for c in trace.candidates if c.chunk_id == shared.chunk_id)
    assert shared_trace_candidate.distance == 0.5


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
