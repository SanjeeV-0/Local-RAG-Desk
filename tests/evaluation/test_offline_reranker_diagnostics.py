"""Focused, fast tests for `tests.evaluation.offline_reranker_diagnostics`.

No real embedding/reranker model is loaded here -- every test uses small,
deterministic fakes so these tests run in milliseconds and never make a
local model inference call, let alone a database or provider call. A
separate, explicit validation run (not part of this test file, not run in
CI) exercised the module against real fixture documents with the real
production embedding provider and reranker; see the task report for that
evidence. These tests check the module's logic (ranking construction,
metric calculation, gold-reference resolution, report aggregation), not
embedding/reranking quality.
"""

import json
from uuid import uuid4

import pytest

from tests.evaluation.offline_reranker_diagnostics import (
    DiagnosticQuestion,
    InMemoryChunk,
    QueryDiagnostic,
    aggregate_by_category,
    aggregate_diagnostic,
    cosine_distance,
    diagnose_query,
    documents_needed_for,
    load_questions_from_json,
    resolve_gold_references,
    run_offline_reranker_diagnostic,
)
from tests.evaluation.retrieval_dataset import RelevantChunkSpec

# --------------------------------------------------------------------------
# Fakes -- no real model inference.
# --------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """Returns a fixed, caller-supplied vector for the query; chunk
    embeddings come directly from the `InMemoryChunk` objects under test,
    never from this fake."""

    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = query_vector

    def embed_text(self, text: str) -> list[float]:
        return self.query_vector

    def embed_batch(self, texts):
        return [self.query_vector for _ in texts]


class FakeReranker:
    """Reorders candidates by a caller-supplied chunk_id -> score mapping,
    exactly matching `app.retrieval.reranker.Reranker`'s contract without
    any real cross-encoder inference."""

    def __init__(self, score_by_chunk_id: dict) -> None:
        self.score_by_chunk_id = score_by_chunk_id

    def rerank(self, query, chunks):
        scored = [(chunk, self.score_by_chunk_id[chunk.chunk_id]) for chunk in chunks]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            type(chunk)(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                section_id=chunk.section_id,
                section_path=chunk.section_path,
                page_numbers=chunk.page_numbers,
                content=chunk.content,
                distance=chunk.distance,
                rerank_score=float(score),
            )
            for chunk, score in scored
        ]


def _chunk(title: str, content: str, embedding: list[float]) -> InMemoryChunk:
    return InMemoryChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title=title,
        content=content,
        section_id=None,
        section_path=None,
        page_numbers=[1],
        embedding=embedding,
    )


# --------------------------------------------------------------------------
# cosine_distance
# --------------------------------------------------------------------------


def test_cosine_distance_identical_vectors_is_zero() -> None:
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_cosine_distance_orthogonal_vectors_is_one() -> None:
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_cosine_distance_zero_vector_returns_max_distance_not_an_error() -> None:
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


# --------------------------------------------------------------------------
# resolve_gold_references
# --------------------------------------------------------------------------


def _question(qid: str, relevant_chunks) -> DiagnosticQuestion:
    return DiagnosticQuestion(
        id=qid,
        query=f"query for {qid}",
        category="semantic_retrieval",
        relevant_chunks=tuple(relevant_chunks),
    )


def test_resolve_gold_references_exact_single_match_is_resolved() -> None:
    corpus = [_chunk("Doc A", "the quick brown fox", [1.0, 0.0])]
    question = _question(
        "q1",
        [RelevantChunkSpec(document_title="Doc A", content_fragment="quick brown", relevance=3)],
    )

    resolutions = resolve_gold_references([question], corpus)

    assert len(resolutions) == 1
    assert resolutions[0].status == "resolved"
    assert resolutions[0].chunk_id == corpus[0].chunk_id


def test_resolve_gold_references_no_match_is_unresolved() -> None:
    corpus = [_chunk("Doc A", "the quick brown fox", [1.0, 0.0])]
    question = _question(
        "q1",
        [
            RelevantChunkSpec(
                document_title="Doc A", content_fragment="not present here", relevance=3
            )
        ],
    )

    resolutions = resolve_gold_references([question], corpus)

    assert resolutions[0].status == "unresolved"
    assert resolutions[0].chunk_id is None
    assert resolutions[0].match_count == 0


def test_resolve_gold_references_multiple_matches_is_ambiguous() -> None:
    corpus = [
        _chunk("Doc A", "repeated phrase one", [1.0, 0.0]),
        _chunk("Doc A", "repeated phrase two", [0.9, 0.1]),
    ]
    question = _question(
        "q1",
        [
            RelevantChunkSpec(
                document_title="Doc A", content_fragment="repeated phrase", relevance=3
            )
        ],
    )

    resolutions = resolve_gold_references([question], corpus)

    assert resolutions[0].status == "ambiguous"
    assert resolutions[0].match_count == 2


def test_resolve_gold_references_only_matches_within_the_named_document() -> None:
    """A fragment that matches content in a DIFFERENT document's chunk must
    not resolve -- document_title scoping is part of the matching rule."""
    corpus = [
        _chunk("Doc A", "unique content alpha", [1.0, 0.0]),
        _chunk("Doc B", "unique content alpha", [0.0, 1.0]),
    ]
    question = _question(
        "q1",
        [RelevantChunkSpec(document_title="Doc A", content_fragment="content alpha", relevance=3)],
    )

    resolutions = resolve_gold_references([question], corpus)

    assert resolutions[0].status == "resolved"
    assert resolutions[0].chunk_id == corpus[0].chunk_id


# --------------------------------------------------------------------------
# diagnose_query -- rank alignment, missing gold candidates, metrics
# --------------------------------------------------------------------------


def test_diagnose_query_vector_rank_matches_distance_order() -> None:
    gold = _chunk("Doc A", "gold content", [1.0, 0.0])  # distance 0 to query
    other = _chunk("Doc A", "other content", [0.0, 1.0])  # distance 1 to query
    corpus = [other, gold]

    question = _question(
        "q1",
        [RelevantChunkSpec(document_title="Doc A", content_fragment="gold content", relevance=3)],
    )
    embedding_provider = FakeEmbeddingProvider(query_vector=[1.0, 0.0])
    reranker = FakeReranker({gold.chunk_id: 1.0, other.chunk_id: 0.0})

    result = diagnose_query(
        question=question,
        corpus=corpus,
        gold_chunk_ids=frozenset({gold.chunk_id}),
        embedding_provider=embedding_provider,
        reranker=reranker,
        candidate_limit=10,
        final_limit=10,
    )

    gold_record = next(c for c in result.candidates if c.chunk_id == gold.chunk_id)
    assert gold_record.vector_rank == 1
    assert gold_record.vector_distance == pytest.approx(0.0)
    assert gold_record.is_gold is True
    assert result.vector_recall_at_3 == 1.0
    assert result.vector_mrr == 1.0


def test_diagnose_query_reranked_rank_reflects_rerank_score_not_distance() -> None:
    """The reranker demotes the vector-best candidate and promotes the
    vector-worst one -- reranked_rank must follow the rerank score, not the
    original distance order."""
    gold = _chunk("Doc A", "gold content", [0.0, 1.0])  # worst vector match
    other = _chunk("Doc A", "other content", [1.0, 0.0])  # best vector match
    corpus = [gold, other]

    question = _question(
        "q1",
        [RelevantChunkSpec(document_title="Doc A", content_fragment="gold content", relevance=3)],
    )
    embedding_provider = FakeEmbeddingProvider(query_vector=[1.0, 0.0])
    # Reranker strongly prefers `gold` despite its worse vector distance.
    reranker = FakeReranker({gold.chunk_id: 5.0, other.chunk_id: -5.0})

    result = diagnose_query(
        question=question,
        corpus=corpus,
        gold_chunk_ids=frozenset({gold.chunk_id}),
        embedding_provider=embedding_provider,
        reranker=reranker,
        candidate_limit=10,
        final_limit=10,
    )

    gold_record = next(c for c in result.candidates if c.chunk_id == gold.chunk_id)
    assert gold_record.vector_rank == 2
    assert gold_record.reranked_rank == 1
    assert result.vector_recall_at_3 == 1.0  # still recalled at this small pool size
    assert result.reranked_mrr == 1.0
    assert result.vector_mrr == 0.5


def test_diagnose_query_gold_chunk_outside_candidate_limit_is_absent_from_pool() -> None:
    """A gold chunk that doesn't make the candidate pool (candidate_limit
    smaller than its rank) must be reflected as a recall/MRR miss, not an
    error, and must not appear in `candidates`."""
    gold = _chunk("Doc A", "gold content", [0.0, 1.0])  # far from query
    decoys = [_chunk("Doc A", f"decoy {i}", [1.0, 0.0]) for i in range(5)]
    corpus = [gold, *decoys]

    question = _question(
        "q1",
        [RelevantChunkSpec(document_title="Doc A", content_fragment="gold content", relevance=3)],
    )
    embedding_provider = FakeEmbeddingProvider(query_vector=[1.0, 0.0])
    reranker = FakeReranker({c.chunk_id: 0.0 for c in corpus})

    result = diagnose_query(
        question=question,
        corpus=corpus,
        gold_chunk_ids=frozenset({gold.chunk_id}),
        embedding_provider=embedding_provider,
        reranker=reranker,
        candidate_limit=3,  # smaller than the corpus; gold is ranked last
        final_limit=3,
    )

    assert gold.chunk_id not in {c.chunk_id for c in result.candidates}
    assert result.gold_refs_in_pool == 0
    assert result.num_gold_refs == 1
    assert result.vector_recall_at_3 == 0.0
    assert result.vector_mrr == 0.0
    assert result.reranked_recall_at_3 == 0.0
    assert result.reranked_mrr == 0.0


def test_diagnose_query_multiple_gold_refs_tracked_independently() -> None:
    gold_a = _chunk("Doc A", "gold a", [1.0, 0.0])
    gold_b = _chunk("Doc A", "gold b", [0.9, 0.1])
    corpus = [gold_a, gold_b]

    question = _question(
        "q1",
        [
            RelevantChunkSpec(document_title="Doc A", content_fragment="gold a", relevance=3),
            RelevantChunkSpec(document_title="Doc A", content_fragment="gold b", relevance=2),
        ],
    )
    embedding_provider = FakeEmbeddingProvider(query_vector=[1.0, 0.0])
    reranker = FakeReranker({gold_a.chunk_id: 1.0, gold_b.chunk_id: 0.5})

    result = diagnose_query(
        question=question,
        corpus=corpus,
        gold_chunk_ids=frozenset({gold_a.chunk_id, gold_b.chunk_id}),
        embedding_provider=embedding_provider,
        reranker=reranker,
        candidate_limit=10,
        final_limit=10,
    )

    assert result.num_gold_refs == 2
    assert result.gold_refs_in_pool == 2
    assert result.vector_recall_at_5 == 1.0


# --------------------------------------------------------------------------
# aggregate_diagnostic / aggregate_by_category
# --------------------------------------------------------------------------


def _query_diagnostic(
    qid: str, category: str, vector_mrr: float, reranked_mrr: float
) -> QueryDiagnostic:
    return QueryDiagnostic(
        question_id=qid,
        category=category,
        query=f"query {qid}",
        candidates=(),
        num_gold_refs=1,
        gold_refs_in_pool=1,
        vector_recall_at_3=1.0,
        vector_recall_at_5=1.0,
        vector_mrr=vector_mrr,
        reranked_recall_at_3=1.0,
        reranked_recall_at_5=1.0,
        reranked_mrr=reranked_mrr,
    )


def test_aggregate_diagnostic_macro_averages_each_metric() -> None:
    rows = [
        _query_diagnostic("q1", "semantic_retrieval", vector_mrr=1.0, reranked_mrr=0.5),
        _query_diagnostic("q2", "semantic_retrieval", vector_mrr=0.0, reranked_mrr=1.0),
    ]

    aggregate = aggregate_diagnostic(rows)

    assert aggregate.vector_mrr == pytest.approx(0.5)
    assert aggregate.reranked_mrr == pytest.approx(0.75)
    assert aggregate.question_count == 2


def test_aggregate_diagnostic_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        aggregate_diagnostic([])


def test_aggregate_by_category_groups_correctly() -> None:
    rows = [
        _query_diagnostic("q1", "semantic_retrieval", vector_mrr=1.0, reranked_mrr=1.0),
        _query_diagnostic("q2", "distractor_heavy", vector_mrr=0.0, reranked_mrr=1.0),
        _query_diagnostic("q3", "distractor_heavy", vector_mrr=0.0, reranked_mrr=0.0),
    ]

    by_category = aggregate_by_category(rows)

    assert set(by_category) == {"semantic_retrieval", "distractor_heavy"}
    assert by_category["semantic_retrieval"].question_count == 1
    assert by_category["distractor_heavy"].question_count == 2
    assert by_category["distractor_heavy"].reranked_mrr == pytest.approx(0.5)


# --------------------------------------------------------------------------
# load_questions_from_json -- configurable subset selection
# --------------------------------------------------------------------------


def _write_fixture_questions(path) -> None:
    raw = [
        {
            "id": "q001",
            "query": "first question",
            "category": "semantic_retrieval",
            "relevant_chunks": [
                {"document_title": "Doc A", "content_fragment": "fragment one", "relevance": 3}
            ],
        },
        {
            "id": "q002",
            "query": "second question",
            "category": "distractor_heavy",
            "relevant_chunks": [
                {"document_title": "Doc B", "content_fragment": "fragment two", "relevance": 2}
            ],
        },
        {
            "id": "q003",
            "query": "third question",
            "category": "semantic_retrieval",
            "relevant_chunks": [
                {"document_title": "Doc A", "content_fragment": "fragment three", "relevance": 3}
            ],
        },
    ]
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_load_questions_from_json_preserves_stable_id(tmp_path) -> None:
    """The specific gap this loader exists to close: the existing
    `tests.evaluation.retrieval_cases_v2` loader discards each question's
    id entirely."""
    fixture_path = tmp_path / "questions.json"
    _write_fixture_questions(fixture_path)

    questions = load_questions_from_json(fixture_path)

    assert [q.id for q in questions] == ["q001", "q002", "q003"]


def test_load_questions_from_json_filters_by_category(tmp_path) -> None:
    fixture_path = tmp_path / "questions.json"
    _write_fixture_questions(fixture_path)

    questions = load_questions_from_json(fixture_path, categories=("distractor_heavy",))

    assert [q.id for q in questions] == ["q002"]


def test_load_questions_from_json_excludes_specific_ids(tmp_path) -> None:
    fixture_path = tmp_path / "questions.json"
    _write_fixture_questions(fixture_path)

    questions = load_questions_from_json(fixture_path, exclude_ids=("q002",))

    assert [q.id for q in questions] == ["q001", "q003"]


def test_diagnostic_question_rejects_empty_relevant_chunks() -> None:
    with pytest.raises(ValueError, match="relevant_chunks"):
        DiagnosticQuestion(id="q1", query="query", category="cat", relevant_chunks=())


# --------------------------------------------------------------------------
# documents_needed_for
# --------------------------------------------------------------------------


def test_documents_needed_for_returns_only_referenced_documents() -> None:
    available = [
        ("a.md", "Doc A", uuid4(), "markdown"),
        ("b.md", "Doc B", uuid4(), "markdown"),
        ("c.md", "Doc C", uuid4(), "markdown"),
    ]
    question = _question(
        "q1", [RelevantChunkSpec(document_title="Doc B", content_fragment="x", relevance=3)]
    )

    needed = documents_needed_for([question], available)

    assert [doc[1] for doc in needed] == ["Doc B"]


# --------------------------------------------------------------------------
# run_offline_reranker_diagnostic -- orchestration, with the corpus-building
# step replaced by a fake (no real I/O, no real model inference).
# --------------------------------------------------------------------------


def test_run_offline_reranker_diagnostic_excludes_fully_unresolved_question(monkeypatch) -> None:
    resolved_chunk = _chunk("Doc A", "resolvable gold content", [1.0, 0.0])
    corpus = [resolved_chunk]

    def fake_build_corpus(documents, embedding_provider, fixture_dir):
        return corpus

    monkeypatch.setattr(
        "tests.evaluation.offline_reranker_diagnostics.build_in_memory_corpus",
        fake_build_corpus,
    )

    resolvable_question = _question(
        "q_ok",
        [
            RelevantChunkSpec(
                document_title="Doc A", content_fragment="resolvable gold", relevance=3
            )
        ],
    )
    unresolvable_question = _question(
        "q_bad",
        [
            RelevantChunkSpec(
                document_title="Doc A", content_fragment="never present text", relevance=3
            )
        ],
    )

    embedding_provider = FakeEmbeddingProvider(query_vector=[1.0, 0.0])
    reranker = FakeReranker({resolved_chunk.chunk_id: 1.0})

    report = run_offline_reranker_diagnostic(
        questions=[resolvable_question, unresolvable_question],
        embedding_provider=embedding_provider,
        reranker=reranker,
        documents=[],
        candidate_limit=10,
        final_limit=10,
    )

    assert [q.question_id for q in report.per_query] == ["q_ok"]
    assert len(report.unresolved_references) == 1
    assert report.unresolved_references[0].question_id == "q_bad"
    assert report.ambiguous_references == ()


def test_run_offline_reranker_diagnostic_reports_ambiguous_reference(monkeypatch) -> None:
    chunk_one = _chunk("Doc A", "shared phrase here", [1.0, 0.0])
    chunk_two = _chunk("Doc A", "shared phrase there", [0.9, 0.1])
    corpus = [chunk_one, chunk_two]

    def fake_build_corpus(documents, embedding_provider, fixture_dir):
        return corpus

    monkeypatch.setattr(
        "tests.evaluation.offline_reranker_diagnostics.build_in_memory_corpus",
        fake_build_corpus,
    )

    question = _question(
        "q1",
        [RelevantChunkSpec(document_title="Doc A", content_fragment="shared phrase", relevance=3)],
    )
    embedding_provider = FakeEmbeddingProvider(query_vector=[1.0, 0.0])
    reranker = FakeReranker({chunk_one.chunk_id: 1.0, chunk_two.chunk_id: 0.5})

    report = run_offline_reranker_diagnostic(
        questions=[question],
        embedding_provider=embedding_provider,
        reranker=reranker,
        documents=[],
    )

    assert report.per_query == ()
    assert report.ambiguous_references[0].question_id == "q1"
    assert report.ambiguous_references[0].match_count == 2
