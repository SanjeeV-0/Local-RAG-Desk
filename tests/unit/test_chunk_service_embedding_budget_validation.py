"""Unit tests for `app.ingestion.chunk_service.validate_embedding_inputs_within_budget`
-- the final, tokenizer-aware safety net before any embedding call. Pure and
DB-independent (no `AsyncSession`/`ChunkService` construction needed), so
these run without a database, unlike `tests/integration/test_chunk_service.py`.
"""

import pytest

from app.ingestion.chunk_service import validate_embedding_inputs_within_budget
from app.ingestion.size_guard import ChildChunk


def _chunk(index: int, content: str = "content") -> ChildChunk:
    return ChildChunk(
        index=index,
        content=content,
        page_numbers=[1],
        section_id=None,
        section_path=None,
        section_level=0,
        source_units=(),
    )


def test_no_op_when_provider_exposes_no_token_counter() -> None:
    """A provider without real token counting (e.g. test fakes) must never
    be checked -- this must not raise regardless of embedding_text length."""
    chunks = [_chunk(0)]
    embedding_texts = ["x" * 10_000]

    validate_embedding_inputs_within_budget(
        chunks,
        embedding_texts,
        token_counter=None,
        max_embedding_tokens=None,
    )


def test_no_op_when_only_one_of_the_two_values_is_supplied() -> None:
    chunks = [_chunk(0)]
    embedding_texts = ["anything"]

    validate_embedding_inputs_within_budget(
        chunks,
        embedding_texts,
        token_counter=len,
        max_embedding_tokens=None,
    )

    validate_embedding_inputs_within_budget(
        chunks,
        embedding_texts,
        token_counter=None,
        max_embedding_tokens=5,
    )


def test_passes_silently_when_every_input_is_within_budget() -> None:
    chunks = [_chunk(0), _chunk(1)]
    embedding_texts = ["short", "also short"]

    validate_embedding_inputs_within_budget(
        chunks,
        embedding_texts,
        token_counter=len,
        max_embedding_tokens=20,
    )


def test_raises_when_any_input_exceeds_budget() -> None:
    chunks = [_chunk(0), _chunk(1, content="the oversized one")]
    embedding_texts = ["short", "x" * 50]

    with pytest.raises(ValueError, match="chunk index 1"):
        validate_embedding_inputs_within_budget(
            chunks,
            embedding_texts,
            token_counter=len,
            max_embedding_tokens=20,
        )


def test_never_truncates_or_mutates_embedding_texts() -> None:
    """The function only measures and raises -- it must never modify the
    list it was given, even for an oversized entry."""
    chunks = [_chunk(0)]
    embedding_texts = ["x" * 50]
    original = list(embedding_texts)

    with pytest.raises(ValueError):
        validate_embedding_inputs_within_budget(
            chunks,
            embedding_texts,
            token_counter=len,
            max_embedding_tokens=10,
        )

    assert embedding_texts == original


def test_boundary_exactly_at_budget_does_not_raise() -> None:
    chunks = [_chunk(0)]
    embedding_texts = ["x" * 10]

    validate_embedding_inputs_within_budget(
        chunks,
        embedding_texts,
        token_counter=len,
        max_embedding_tokens=10,
    )


def test_boundary_one_over_budget_raises() -> None:
    chunks = [_chunk(0)]
    embedding_texts = ["x" * 11]

    with pytest.raises(ValueError):
        validate_embedding_inputs_within_budget(
            chunks,
            embedding_texts,
            token_counter=len,
            max_embedding_tokens=10,
        )
