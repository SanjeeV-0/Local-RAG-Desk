"""Reusable, database-free diagnostic for comparing vector-only retrieval
against reranked retrieval over an in-memory copy of the V2 benchmark
fixture documents.

This module exists because `tests.evaluation.test_benchmark_v2_retrieval`
(the committed, database-backed V2 benchmark) has two structural
limitations that make it unsuitable for reranker regression diagnosis:
it always applies the configured reranker inside `RetrievalService.search`
with no vector-only comparison path, and it discards every per-candidate
detail (chunk id, vector distance, rerank score) the moment the four
scalar metrics are computed (see `RetrievalCaseMetrics` in
`tests.evaluation.retrieval_metrics`).

Everything here reuses the EXACT production ingestion/ranking components,
never a reimplementation:
`app.ingestion.structure_extractor.StructureExtractor`,
`app.ingestion.size_guard.fragment_table_units`/`apply_size_guard` (wired
tokenizer-aware exactly as `app.ingestion.service.IngestionService` wires
them -- same `getattr(embedding_provider, "count_tokens"/"max_embedding_tokens",
None)` duck-typed fallback, so a provider without that capability falls
back to the same 500-word-estimated behavior production falls back to),
`app.ingestion.section_builder.SectionBuilder`,
`app.ingestion.semantic_shredder.shred_semantically`,
`app.ingestion.embedding_text.build_embedding_text`, and
`app.retrieval.models.RetrievedChunk` (the same type
`app.retrieval.reranker.Reranker.rerank` already consumes). The only
substitutions versus the full application are: (1) section/chunk IDs are
generated locally (`uuid4()`) instead of persisted, since
`app.ingestion.section_map.SectionMap` and `app.ingestion.size_guard.ChildChunk`
are already pure, in-memory, ID-agnostic data shapes with no database
coupling of their own; and (2) vector search is exact, brute-force cosine
distance in Python rather than pgvector/HNSW -- see `cosine_distance`'s
docstring for why this is a disclosed, not hidden, limitation.

No database connection is ever opened by this module. `embedding_provider`/
`reranker` are passed in by the caller (dependency injection, matching
`IngestionService`'s/`RetrievalService`'s own constructor style) -- this
module never constructs or downloads a model itself, so importing it has
no side effects; local embedding/reranker inference only happens when
`run_offline_reranker_diagnostic` (or a lower-level function) is actually
called with a real provider.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from app.ingestion.embedding_text import build_embedding_text
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.ingestion.section_builder import SectionBuilder
from app.ingestion.section_map import SectionMap
from app.ingestion.semantic_shredder import shred_semantically
from app.ingestion.size_guard import apply_size_guard, fragment_table_units
from app.ingestion.structure_extractor import StructureExtractor
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import Reranker
from tests.evaluation.benchmark_corpus_v2 import (
    BENCHMARK_V2_DOCUMENTS,
    BENCHMARK_V2_FIXTURE_DIR,
)
from tests.evaluation.retrieval_dataset import RelevantChunkSpec
from tests.evaluation.retrieval_metrics import mrr_at_k, recall_at_k

_LOADERS_BY_FORMAT: dict[str, DocumentLoader] = {
    "markdown": MarkdownLoader(),
    "docx": DocxLoader(),
    "pdf": PDFLoader(),
}


# --------------------------------------------------------------------------
# Question input -- a thin, additive wrapper around the EXISTING
# `RelevantChunkSpec` gold-reference shape (reused unchanged from
# `tests.evaluation.retrieval_dataset`), adding only the one field that
# shape and `RetrievalEvaluationCaseSpec` both lack: a stable question id.
# `RETRIEVAL_BENCHMARK_V2_CASE_SPECS` (tests.evaluation.retrieval_cases_v2)
# discards each question's "qNNN" id while loading -- see that module's
# `_load_case_specs`, which never reads `question["id"]`. A per-query
# diagnostic report is meaningless without a stable identifier, so this
# module loads directly from the same JSON fixture instead of reusing that
# loader, while reusing its identical field mapping.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticQuestion:
    """One evaluation question: a stable id, its text, category, and gold
    evidence (reusing `RelevantChunkSpec` unchanged)."""

    id: str
    query: str
    category: str
    relevant_chunks: tuple[RelevantChunkSpec, ...]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id must not be empty")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.relevant_chunks:
            raise ValueError("relevant_chunks must not be empty")


def load_questions_from_json(
    path: Path,
    categories: Sequence[str] | None = None,
    exclude_ids: Sequence[str] = (),
) -> list[DiagnosticQuestion]:
    """Load `DiagnosticQuestion`s directly from a `questions_v2.json`-shaped
    fixture file, optionally filtered to specific categories and/or
    excluding specific ids.

    This is the configurable-subset entry point: pass `categories` to
    restrict to e.g. `("semantic_retrieval", "cross_section",
    "distractor_heavy")`, and/or `exclude_ids` for a question whose gold
    reference is known not to resolve under the current chunker (as
    established in this project's own prior offline evaluation). Never
    mutates the fixture file; purely a read + filter.
    """

    raw_questions = json.loads(Path(path).read_text(encoding="utf-8"))

    category_set = set(categories) if categories is not None else None
    exclude_set = set(exclude_ids)

    questions: list[DiagnosticQuestion] = []

    for raw in raw_questions:
        if raw["id"] in exclude_set:
            continue
        if category_set is not None and raw["category"] not in category_set:
            continue

        relevant_chunks = tuple(
            RelevantChunkSpec(
                document_title=relevant_chunk["document_title"],
                content_fragment=relevant_chunk["content_fragment"],
                relevance=relevant_chunk["relevance"],
            )
            for relevant_chunk in raw["relevant_chunks"]
        )

        questions.append(
            DiagnosticQuestion(
                id=raw["id"],
                query=raw["query"],
                category=raw["category"],
                relevant_chunks=relevant_chunks,
            )
        )

    return questions


# --------------------------------------------------------------------------
# In-memory corpus construction -- the exact production chunking pipeline,
# reused unchanged.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InMemoryChunk:
    """One chunk produced by the in-memory pipeline, with a locally
    generated (never persisted) identity."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    content: str
    section_id: UUID | None
    section_path: str | None
    page_numbers: list[int]
    embedding: list[float]


def _process_document_in_memory(
    path: Path,
    loader: DocumentLoader,
    embedding_provider: object,
) -> list:
    """Run one fixture document through the exact production chunking
    pipeline (`StructureExtractor` -> `fragment_table_units` ->
    `SectionBuilder` -> `shred_semantically` -> `apply_size_guard`),
    returning `ChildChunk`s. Mirrors
    `app.ingestion.service.IngestionService._process_document`'s chunking
    stages exactly, including the tokenizer-aware budget/section-path
    reservation wiring when the embedding provider exposes
    `count_tokens`/`max_embedding_tokens` -- see that method and
    `app.embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider`.
    """

    pages = loader.load(path)

    structure_extractor = StructureExtractor()
    structural_units = structure_extractor.extract(pages)

    token_counter = getattr(embedding_provider, "count_tokens", None)
    max_embedding_tokens = getattr(embedding_provider, "max_embedding_tokens", None)

    if token_counter is not None and max_embedding_tokens is not None:
        max_chunk_tokens = max_embedding_tokens
        size_guard_kwargs: dict[str, object] = {
            "count_tokens": token_counter,
            "reserve_section_path_tokens": True,
        }
    else:
        max_chunk_tokens = 500
        size_guard_kwargs = {}

    structural_units = fragment_table_units(
        structural_units,
        max_tokens=max_chunk_tokens,
        **size_guard_kwargs,
    )

    section_builder = SectionBuilder()
    section_nodes = section_builder.build(structural_units)

    section_map = SectionMap()
    for node in section_nodes:
        section_map.add(node.path, uuid4())

    semantic_units = shred_semantically(
        structural_units,
        embedding_provider=embedding_provider,
        threshold=0.7,
    )

    return apply_size_guard(
        semantic_units,
        section_map=section_map,
        max_tokens=max_chunk_tokens,
        **size_guard_kwargs,
    )


def build_in_memory_corpus(
    documents: Sequence[tuple[str, str, object, str]],
    embedding_provider: object,
    fixture_dir: Path = BENCHMARK_V2_FIXTURE_DIR,
) -> list[InMemoryChunk]:
    """Build the full in-memory chunk corpus for `documents`.

    `documents` uses the same 4-tuple shape as
    `tests.evaluation.benchmark_corpus_v2.BENCHMARK_V2_DOCUMENTS`
    (`filename, title, logical_document_id, source_format`) so a caller can
    pass a filtered subset of that exact constant directly -- the
    `logical_document_id` element is accepted for shape compatibility but
    unused here (no persistence occurs, so no stable document id is
    needed). Never writes to a database, never persists an embedding.
    """

    corpus: list[InMemoryChunk] = []

    for filename, title, _logical_document_id, source_format in documents:
        loader = _LOADERS_BY_FORMAT[source_format]
        chunks = _process_document_in_memory(
            fixture_dir / filename,
            loader,
            embedding_provider,
        )

        embedding_texts = [
            build_embedding_text(chunk.section_path, chunk.content) for chunk in chunks
        ]
        embeddings = embedding_provider.embed_batch(embedding_texts)

        document_id = uuid4()

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            corpus.append(
                InMemoryChunk(
                    chunk_id=uuid4(),
                    document_id=document_id,
                    document_title=title,
                    content=chunk.content,
                    section_id=chunk.section_id,
                    section_path=chunk.section_path,
                    page_numbers=chunk.page_numbers,
                    embedding=embedding,
                )
            )

    return corpus


def documents_needed_for(
    questions: Sequence[DiagnosticQuestion],
    available_documents: Sequence[tuple[str, str, object, str]] = BENCHMARK_V2_DOCUMENTS,
) -> list[tuple[str, str, object, str]]:
    """Return only the entries of `available_documents` referenced by at
    least one of `questions`' gold references -- avoids building chunks for
    documents no selected question needs."""

    needed_titles = {
        relevant_chunk.document_title
        for question in questions
        for relevant_chunk in question.relevant_chunks
    }

    return [doc for doc in available_documents if doc[1] in needed_titles]


# --------------------------------------------------------------------------
# Gold-reference resolution -- the identical content-fragment matching rule
# used by `tests.evaluation.benchmark_corpus_v2.resolve_benchmark_v2_evaluation_cases`
# and `tests.evaluation.retrieval_dataset.resolve_evaluation_cases`, applied
# against in-memory chunks instead of persisted ones. Matching is purely
# `fragment in chunk.content` -- content-based, never ID-based -- so this is
# not a reimplementation of a different algorithm, only the same algorithm
# run against a different (in-memory, not database) chunk source.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldReferenceResolution:
    """The outcome of resolving one gold reference against the in-memory
    corpus."""

    question_id: str
    reference_index: int
    document_title: str
    status: str  # "resolved" | "unresolved" | "ambiguous"
    chunk_id: UUID | None
    match_count: int


def resolve_gold_references(
    questions: Sequence[DiagnosticQuestion],
    corpus: Sequence[InMemoryChunk],
) -> list[GoldReferenceResolution]:
    """Resolve every gold reference in `questions` against `corpus`.

    Exactly one matching chunk -> `"resolved"` (with its `chunk_id`); zero
    matches -> `"unresolved"`; more than one match -> `"ambiguous"`. Never
    raises -- every reference, resolved or not, is reported, so a caller
    can decide how to handle unresolved/ambiguous references (as this
    project's own prior offline evaluations did) rather than this function
    silently dropping or guessing.
    """

    resolutions: list[GoldReferenceResolution] = []

    for question in questions:
        for index, relevant_chunk in enumerate(question.relevant_chunks):
            matches = [
                chunk
                for chunk in corpus
                if chunk.document_title == relevant_chunk.document_title
                and relevant_chunk.content_fragment in chunk.content
            ]

            if len(matches) == 1:
                resolutions.append(
                    GoldReferenceResolution(
                        question_id=question.id,
                        reference_index=index,
                        document_title=relevant_chunk.document_title,
                        status="resolved",
                        chunk_id=matches[0].chunk_id,
                        match_count=1,
                    )
                )
            elif len(matches) == 0:
                resolutions.append(
                    GoldReferenceResolution(
                        question_id=question.id,
                        reference_index=index,
                        document_title=relevant_chunk.document_title,
                        status="unresolved",
                        chunk_id=None,
                        match_count=0,
                    )
                )
            else:
                resolutions.append(
                    GoldReferenceResolution(
                        question_id=question.id,
                        reference_index=index,
                        document_title=relevant_chunk.document_title,
                        status="ambiguous",
                        chunk_id=None,
                        match_count=len(matches),
                    )
                )

    return resolutions


# --------------------------------------------------------------------------
# Vector-only vs. reranked comparison.
# --------------------------------------------------------------------------


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact cosine distance between two vectors.

    This is NOT pgvector/HNSW -- production retrieval uses an approximate
    nearest-neighbor index with a configured `ef_search`, while this
    computes exact distance against every corpus chunk. For the small,
    single-diagnostic-run corpus sizes this module targets, exact search is
    strictly more accurate than production's approximation, never less --
    but it is also not a byte-for-byte reproduction of production ranking
    behavior, and callers/reports must not claim it is.
    """

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 1.0

    return 1.0 - (dot / (norm_a * norm_b))


@dataclass(frozen=True)
class CandidateRecord:
    """Per-candidate diagnostic detail -- the exact information
    `tests.evaluation.test_benchmark_v2_retrieval` discards after computing
    its four scalar metrics (see this module's docstring)."""

    chunk_id: UUID
    vector_rank: int | None
    vector_distance: float | None
    reranked_rank: int | None
    reranked_score: float | None
    is_gold: bool


@dataclass(frozen=True)
class QueryDiagnostic:
    """Full diagnostic result for one question: per-candidate records plus
    vector-only and reranked metrics computed from the identical candidate
    pool and limits."""

    question_id: str
    category: str
    query: str
    candidates: tuple[CandidateRecord, ...]
    num_gold_refs: int
    gold_refs_in_pool: int
    vector_recall_at_3: float
    vector_recall_at_5: float
    vector_mrr: float
    reranked_recall_at_3: float
    reranked_recall_at_5: float
    reranked_mrr: float


def _mrr(retrieved_ids: Sequence[UUID], relevant_ids: frozenset[UUID]) -> float:
    """MRR with no K cutoff (ranks over the full `retrieved_ids` list) --
    `mrr_at_k` from `tests.evaluation.retrieval_metrics` is reused with
    `k=len(retrieved_ids)` so the underlying, already-tested implementation
    is never duplicated."""
    return mrr_at_k(retrieved_ids, relevant_ids, k=len(retrieved_ids))


def diagnose_query(
    question: DiagnosticQuestion,
    corpus: Sequence[InMemoryChunk],
    gold_chunk_ids: frozenset[UUID],
    embedding_provider: object,
    reranker: Reranker,
    candidate_limit: int,
    final_limit: int,
) -> QueryDiagnostic:
    """Run one question through vector-only AND reranked retrieval over the
    identical candidate pool and limits, retaining full per-candidate
    detail.

    `reranker` is anything satisfying `app.retrieval.reranker.Reranker`
    (the same Protocol `RetrievalService` depends on) -- the real
    `CrossEncoderReranker` in production use, or a fake in tests.
    """

    query_embedding = embedding_provider.embed_text(question.query)

    distances = [(chunk, cosine_distance(query_embedding, chunk.embedding)) for chunk in corpus]
    distances.sort(key=lambda item: item[1])
    candidate_entries = distances[:candidate_limit]

    vector_candidates = [
        RetrievedChunk(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            page_numbers=chunk.page_numbers,
            content=chunk.content,
            distance=distance,
        )
        for chunk, distance in candidate_entries
    ]

    reranked_candidates = reranker.rerank(question.query, vector_candidates)

    vector_rank_by_id = {c.chunk_id: i + 1 for i, c in enumerate(vector_candidates)}
    vector_distance_by_id = {c.chunk_id: c.distance for c in vector_candidates}
    reranked_rank_by_id = {c.chunk_id: i + 1 for i, c in enumerate(reranked_candidates)}
    reranked_score_by_id = {c.chunk_id: c.rerank_score for c in reranked_candidates}

    candidate_ids = {c.chunk_id for c in vector_candidates}

    candidates = tuple(
        CandidateRecord(
            chunk_id=chunk_id,
            vector_rank=vector_rank_by_id.get(chunk_id),
            vector_distance=vector_distance_by_id.get(chunk_id),
            reranked_rank=reranked_rank_by_id.get(chunk_id),
            reranked_score=reranked_score_by_id.get(chunk_id),
            is_gold=chunk_id in gold_chunk_ids,
        )
        for chunk_id in candidate_ids
    )

    vector_final = [c.chunk_id for c in vector_candidates[:final_limit]]
    reranked_final = [c.chunk_id for c in reranked_candidates[:final_limit]]

    gold_refs_in_pool = len(gold_chunk_ids & candidate_ids)

    return QueryDiagnostic(
        question_id=question.id,
        category=question.category,
        query=question.query,
        candidates=candidates,
        num_gold_refs=len(gold_chunk_ids),
        gold_refs_in_pool=gold_refs_in_pool,
        vector_recall_at_3=recall_at_k(vector_final, gold_chunk_ids, 3),
        vector_recall_at_5=recall_at_k(vector_final, gold_chunk_ids, 5),
        vector_mrr=_mrr(vector_final, gold_chunk_ids),
        reranked_recall_at_3=recall_at_k(reranked_final, gold_chunk_ids, 3),
        reranked_recall_at_5=recall_at_k(reranked_final, gold_chunk_ids, 5),
        reranked_mrr=_mrr(reranked_final, gold_chunk_ids),
    )


# --------------------------------------------------------------------------
# Report assembly.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticReport:
    """Full output of one diagnostic run."""

    per_query: tuple[QueryDiagnostic, ...]
    unresolved_references: tuple[GoldReferenceResolution, ...]
    ambiguous_references: tuple[GoldReferenceResolution, ...]


@dataclass(frozen=True)
class AggregateMetrics:
    vector_recall_at_3: float
    vector_recall_at_5: float
    vector_mrr: float
    reranked_recall_at_3: float
    reranked_recall_at_5: float
    reranked_mrr: float
    question_count: int


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def aggregate_diagnostic(per_query: Sequence[QueryDiagnostic]) -> AggregateMetrics:
    """Macro-average every per-query metric. Raises if `per_query` is
    empty -- an empty aggregate is not a meaningful zero, it is a caller
    error (e.g. an empty or fully-excluded question set)."""

    if not per_query:
        raise ValueError("per_query must not be empty")

    return AggregateMetrics(
        vector_recall_at_3=_mean([q.vector_recall_at_3 for q in per_query]),
        vector_recall_at_5=_mean([q.vector_recall_at_5 for q in per_query]),
        vector_mrr=_mean([q.vector_mrr for q in per_query]),
        reranked_recall_at_3=_mean([q.reranked_recall_at_3 for q in per_query]),
        reranked_recall_at_5=_mean([q.reranked_recall_at_5 for q in per_query]),
        reranked_mrr=_mean([q.reranked_mrr for q in per_query]),
        question_count=len(per_query),
    )


def aggregate_by_category(
    per_query: Sequence[QueryDiagnostic],
) -> dict[str, AggregateMetrics]:
    """`aggregate_diagnostic`, grouped by `QueryDiagnostic.category`."""

    by_category: dict[str, list[QueryDiagnostic]] = {}
    for q in per_query:
        by_category.setdefault(q.category, []).append(q)

    return {category: aggregate_diagnostic(rows) for category, rows in sorted(by_category.items())}


def run_offline_reranker_diagnostic(
    questions: Sequence[DiagnosticQuestion],
    embedding_provider: object,
    reranker: Reranker,
    documents: Sequence[tuple[str, str, object, str]] | None = None,
    fixture_dir: Path = BENCHMARK_V2_FIXTURE_DIR,
    candidate_limit: int = 50,
    final_limit: int = 10,
) -> DiagnosticReport:
    """Run the full offline diagnostic for `questions`.

    `documents` defaults to only the fixture documents `questions` actually
    reference (via `documents_needed_for`), not the full nine-document
    corpus. This default is appropriate for a quick smoke test of a small
    question subset, but IS NOT equivalent to a realistic retrieval
    evaluation: narrowing the candidate pool to only documents that contain
    a correct answer makes distractor rejection artificially easy and
    changes vector/reranked rankings relative to a run against the full
    campaign's document set (confirmed empirically: re-running the same 5
    questions with the full 8-document set the original 25-question
    evaluation used, versus the question-subset default, produced different
    MRR values for the same questions -- a smaller candidate pool changes
    which non-relevant chunks compete for the final ranking). To reproduce
    or compare against a prior run's exact numbers, pass that run's exact
    `documents` list explicitly rather than relying on this default.

    No database access, no embedding/reranker persistence, no provider
    calls: `embedding_provider` and `reranker` are used purely as local,
    in-process objects.

    Unresolved/ambiguous gold references are reported in the returned
    `DiagnosticReport`, never silently dropped or treated as resolved --
    the corresponding question's `gold_chunk_ids` simply has fewer entries
    than its raw `relevant_chunks` count, matching `num_gold_refs`/
    `gold_refs_in_pool` reported per query.
    """

    if documents is None:
        documents = documents_needed_for(questions)

    corpus = build_in_memory_corpus(documents, embedding_provider, fixture_dir)

    resolutions = resolve_gold_references(questions, corpus)

    gold_ids_by_question: dict[str, set[UUID]] = {q.id: set() for q in questions}
    unresolved: list[GoldReferenceResolution] = []
    ambiguous: list[GoldReferenceResolution] = []

    for resolution in resolutions:
        if resolution.status == "resolved":
            gold_ids_by_question[resolution.question_id].add(resolution.chunk_id)
        elif resolution.status == "unresolved":
            unresolved.append(resolution)
        else:
            ambiguous.append(resolution)

    per_query = tuple(
        diagnose_query(
            question=question,
            corpus=corpus,
            gold_chunk_ids=frozenset(gold_ids_by_question[question.id]),
            embedding_provider=embedding_provider,
            reranker=reranker,
            candidate_limit=candidate_limit,
            final_limit=final_limit,
        )
        for question in questions
        if gold_ids_by_question[question.id]
    )

    return DiagnosticReport(
        per_query=per_query,
        unresolved_references=tuple(unresolved),
        ambiguous_references=tuple(ambiguous),
    )
