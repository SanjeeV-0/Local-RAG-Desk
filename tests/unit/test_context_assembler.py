from uuid import uuid4

import pytest

from app.generation.context import ContextAssembler
from app.retrieval.models import RetrievedChunk


def _chunk(
    content: str,
    page_number: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[page_number],
        content=content,
        distance=0.1,
    )


def test_context_assembler_preserves_chunk_order() -> None:
    first = _chunk("First retrieved chunk.", 1)
    second = _chunk("Second retrieved chunk.", 2)

    assembler = ContextAssembler()

    context = assembler.assemble([first, second])

    assert context.text == (
        "[Source 1]\nFirst retrieved chunk.\n\n[Source 2]\nSecond retrieved chunk."
    )

    assert len(context.sources) == 2

    assert context.sources[0].chunk_id == first.chunk_id
    assert context.sources[0].page_numbers == [1]

    assert context.sources[1].chunk_id == second.chunk_id
    assert context.sources[1].page_numbers == [2]


def test_context_assembler_handles_empty_chunks() -> None:
    assembler = ContextAssembler()

    context = assembler.assemble([])

    assert context.text == ""
    assert context.sources == []


def test_context_assembler_respects_character_budget() -> None:
    first = _chunk("First chunk.", 1)
    second = _chunk("Second chunk.", 2)
    third = _chunk("Third chunk.", 3)

    assembler = ContextAssembler(
        max_characters=len("[Source 1]\nFirst chunk.\n\n[Source 2]\nSecond chunk.")
    )

    context = assembler.assemble([first, second, third])

    assert context.text == ("[Source 1]\nFirst chunk.\n\n[Source 2]\nSecond chunk.")

    assert [source.chunk_id for source in context.sources] == [
        first.chunk_id,
        second.chunk_id,
    ]


def test_context_assembler_never_partially_includes_chunk() -> None:
    first = _chunk("A" * 20, 1)
    second = _chunk("B" * 100, 2)

    assembler = ContextAssembler(max_characters=len("[Source 1]\n" + ("A" * 20)))

    context = assembler.assemble([first, second])

    assert context.text == ("[Source 1]\n" + ("A" * 20))

    assert len(context.sources) == 1
    assert context.sources[0].chunk_id == first.chunk_id


def test_context_assembler_rejects_negative_budget() -> None:
    with pytest.raises(ValueError):
        ContextAssembler(max_characters=-1)


def test_context_assembler_redacts_pii_in_chunk_content() -> None:
    chunk = _chunk("Contact jdoe@example.com for the on-call rotation.", 1)

    context = ContextAssembler().assemble([chunk])

    assert "[REDACTED:EMAIL]" in context.text
    assert "jdoe@example.com" not in context.text
    assert "Contact" in context.text
    assert "for the on-call rotation." in context.text
    assert context.redacted_source_count == 1


def test_context_assembler_counts_only_chunks_with_redactions() -> None:
    clean = _chunk("No sensitive content here.", 1)
    dirty = _chunk("Call 555-123-4567 for support.", 2)

    context = ContextAssembler().assemble([clean, dirty])

    assert context.redacted_source_count == 1
    assert "555-123-4567" not in context.text
    assert "[REDACTED:PHONE]" in context.text


def test_context_assembler_clean_chunks_report_zero_redactions() -> None:
    chunk = _chunk("Entirely safe research content.", 1)

    context = ContextAssembler().assemble([chunk])

    assert context.redacted_source_count == 0


def test_context_assembler_passes_table_markdown_through_unchanged() -> None:
    """A table chunk's content is just text to ContextAssembler -- no
    special-cased table generation path exists or is needed."""

    table_chunk = _chunk(
        "| Model | Score |\n| --- | --- |\n| A | 0.90 |\n| B | 0.85 |",
        1,
    )

    context = ContextAssembler().assemble([table_chunk])

    assert context.text == (
        "[Source 1]\n| Model | Score |\n| --- | --- |\n| A | 0.90 |\n| B | 0.85 |"
    )
    assert context.sources[0].chunk_id == table_chunk.chunk_id
