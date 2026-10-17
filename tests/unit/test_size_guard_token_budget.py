"""Tokenizer-aware chunk-sizing tests for `app.ingestion.size_guard` and
`app.ingestion.table_chunker`.

Uses a synthetic `count_tokens` callable (2 tokens per word) rather than the
real sentence-transformers tokenizer, so these tests are fast, deterministic,
and prove the INJECTED counter is actually used -- not merely that the
default word-count estimator still works (already covered by
`test_size_guard.py`/`test_table_chunker.py`, which this file never
modifies or duplicates).
"""

from uuid import UUID, uuid4

import pytest

from app.ingestion.section_map import SectionMap
from app.ingestion.semantic_shredder import SemanticUnit
from app.ingestion.size_guard import apply_size_guard, estimate_tokens, fragment_table_units
from app.ingestion.structure import StructuralUnit, TableData, UnitType


def _synthetic_count_tokens(text: str) -> int:
    """2 tokens per whitespace word -- a stand-in for a real tokenizer's
    typical word-to-wordpiece inflation, distinct enough from plain word
    count that a test can prove this callable (not the default) was used."""
    return len(text.split()) * 2


def _paragraph(
    content: str,
    section_path: str = "Results",
    index: int = 0,
) -> StructuralUnit:
    return StructuralUnit(
        unit_type=UnitType.PARAGRAPH,
        content=content,
        page_numbers=[1],
        section_path=section_path,
        section_level=1,
        section_index=index,
    )


def _section_map(section_path: str = "Results") -> tuple[SectionMap, UUID]:
    section_map = SectionMap()
    section_id = uuid4()
    section_map.add(section_path, section_id)
    return section_map, section_id


# --- Budget boundaries: below / at / above, using the injected counter ---


def test_content_below_injected_budget_is_not_split() -> None:
    """3 words * 2 tokens/word = 6 tokens, budget 6 -- fits exactly."""
    semantic_unit = SemanticUnit((_paragraph("one two three"),))
    section_map, _ = _section_map()

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=6,
        count_tokens=_synthetic_count_tokens,
    )

    assert len(result) == 1
    assert result[0].content == "one two three"


def test_content_exactly_at_injected_budget_is_not_split() -> None:
    semantic_unit = SemanticUnit((_paragraph("one two three four"),))
    section_map, _ = _section_map()

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=8,  # 4 words * 2 == 8, exactly the budget
        count_tokens=_synthetic_count_tokens,
    )

    assert len(result) == 1


def test_content_above_injected_budget_is_split_using_the_injected_counter() -> None:
    """With the default word-count estimator, 10 words would fit a
    max_tokens=10 budget untouched (as `test_size_guard.py` already proves).
    With the synthetic 2x counter, 10 words == 20 tokens, which must split --
    directly proving the injected counter (not the default) drives the
    decision."""
    semantic_unit = SemanticUnit((_paragraph("one two three four five six seven eight nine ten"),))
    section_map, _ = _section_map()

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=10,
        count_tokens=_synthetic_count_tokens,
    )

    assert len(result) > 1
    assert all(_synthetic_count_tokens(child.content) <= 10 for child in result)


# --- Long section path consuming part of the budget ---


_LONG_SECTION_PATH = "Architecture > Retrieval > Reranking"  # 5 words -> 10 synthetic tokens


def test_long_section_path_reserves_budget_from_content_when_opted_in() -> None:
    """A long section path combined with content that alone would fit must
    still trigger a split once the prefix's reserved tokens are accounted
    for -- proving `reserve_section_path_tokens=True` actually shrinks the
    usable content budget rather than merely being accepted as a no-op
    flag."""
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "one two three four five six",
                section_path=_LONG_SECTION_PATH,
            ),
        )
    )
    section_map, _ = _section_map(_LONG_SECTION_PATH)

    # Content alone is 6 words == 12 tokens; budget is 20, so it WOULD fit
    # if the section path prefix were not reserved.
    result_without_reservation = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=20,
        count_tokens=_synthetic_count_tokens,
        reserve_section_path_tokens=False,
    )

    assert len(result_without_reservation) == 1

    # With reservation: 10 tokens reserved for the prefix leaves only a
    # 10-token content budget, below the 12-token content -- must split.
    result_with_reservation = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=20,
        count_tokens=_synthetic_count_tokens,
        reserve_section_path_tokens=True,
    )

    assert len(result_with_reservation) > 1
    for child in result_with_reservation:
        assert _synthetic_count_tokens(child.content) <= 10


def test_default_behavior_is_unchanged_when_reservation_not_requested() -> None:
    """Every existing caller (all of `test_size_guard.py`) omits
    `count_tokens`/`reserve_section_path_tokens` entirely -- confirms the
    defaults reproduce the pre-existing, word-count-estimated, unreserved
    behavior exactly."""
    semantic_unit = SemanticUnit((_paragraph("one two three four five six seven eight nine ten"),))
    section_map, section_id = _section_map()

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=5,
    )

    assert len(result) == 2
    assert all(estimate_tokens(child.content) <= 5 for child in result)
    assert all(child.section_id == section_id for child in result)


def test_section_path_alone_exceeding_budget_fails_explicitly() -> None:
    """If the section-path prefix alone would consume the whole budget (or
    more), there is no room left for any content -- this must fail loudly,
    not silently produce a chunk with no content budget."""
    huge_section_path = "A " * 50  # 50 words -> 100 synthetic tokens
    semantic_unit = SemanticUnit(
        (_paragraph("short content here", section_path=huge_section_path.strip()),)
    )
    section_map, _ = _section_map(huge_section_path.strip())

    with pytest.raises(ValueError, match="Section path prefix alone"):
        apply_size_guard(
            [semantic_unit],
            section_map=section_map,
            max_tokens=10,
            count_tokens=_synthetic_count_tokens,
            reserve_section_path_tokens=True,
        )


# --- Multi-row table splitting with the injected counter ---


def _table_unit(
    rows: list[list[str]],
    section_path: str = "Results",
) -> StructuralUnit:
    table = TableData(
        table_id="table-1",
        title="Retrieval Results",
        headers=["Model", "Score"],
        rows=rows,
        page_numbers=[10],
    )
    return StructuralUnit(
        unit_type=UnitType.TABLE,
        content="placeholder -- fragment_table_units always re-serializes",
        page_numbers=[10],
        section_path=section_path,
        section_level=1,
        section_index=0,
        table=table,
    )


def test_fragment_table_units_uses_injected_counter_for_row_splitting() -> None:
    rows = [[f"Model{i}", f"0.{90 - i}"] for i in range(10)]
    unit = _table_unit(rows=rows)

    result = fragment_table_units(
        [unit],
        max_tokens=40,
        count_tokens=_synthetic_count_tokens,
    )

    assert len(result) > 1
    for fragment in result:
        assert _synthetic_count_tokens(fragment.content) <= 40
        # Every fragment still a valid, self-contained markdown table.
        assert "| Model | Score |" in fragment.content


def test_fragment_table_units_reserves_section_path_budget_for_tables() -> None:
    """Tables get the same section-path reservation treatment as prose --
    the embedding input for a table chunk also carries the heading prefix,
    so it must also be budgeted for when opted in."""
    long_section_path = "Benchmarks > Retrieval Quality > Per-Model Comparison"
    rows = [[f"Model{i}", f"0.{90 - i}"] for i in range(6)]
    unit = _table_unit(rows=rows, section_path=long_section_path)

    without_reservation = fragment_table_units(
        [unit],
        max_tokens=60,
        count_tokens=_synthetic_count_tokens,
        reserve_section_path_tokens=False,
    )

    with_reservation = fragment_table_units(
        [unit],
        max_tokens=60,
        count_tokens=_synthetic_count_tokens,
        reserve_section_path_tokens=True,
    )

    # Reserving room for a non-trivial section path leaves less room for
    # table rows per fragment, so it should never produce FEWER fragments.
    assert len(with_reservation) >= len(without_reservation)


# --- Unsplittable single table row: explicit failure, not corruption ---


def test_single_oversized_table_row_still_fails_explicitly_with_injected_counter() -> None:
    """A single row (with headers) that alone exceeds even the
    section-path-reserved budget cannot be shrunk further by row-boundary
    splitting -- `apply_size_guard` must raise, never silently truncate or
    corrupt the table."""
    unit = _table_unit(rows=[["A", "0.90"]])

    fragments = fragment_table_units(
        [unit],
        max_tokens=5,
        count_tokens=_synthetic_count_tokens,
    )

    section_map, _ = _section_map("Results")

    with pytest.raises(ValueError, match="Table exceeds size guard"):
        apply_size_guard(
            [SemanticUnit((fragment,)) for fragment in fragments],
            section_map=section_map,
            max_tokens=5,
            count_tokens=_synthetic_count_tokens,
        )


# --- Provenance preservation across a reservation-triggered split ---


def test_provenance_preserved_across_a_section_path_triggered_split() -> None:
    """Splitting caused purely by section-path budget reservation (not by
    oversized content alone) must still preserve section_id, section_path,
    and page_numbers on every resulting piece -- the hierarchy and source
    attribution a reader relies on must not degrade just because a chunk
    was split for budget reasons."""
    unit = StructuralUnit(
        unit_type=UnitType.PARAGRAPH,
        content="one two three four five six",
        page_numbers=[7, 8],
        section_path=_LONG_SECTION_PATH,
        section_level=4,
        section_index=0,
    )
    semantic_unit = SemanticUnit((unit,))
    section_map, section_id = _section_map(_LONG_SECTION_PATH)

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=20,
        count_tokens=_synthetic_count_tokens,
        reserve_section_path_tokens=True,
    )

    assert len(result) > 1
    for child in result:
        assert child.section_id == section_id
        assert child.section_path == _LONG_SECTION_PATH
        assert child.page_numbers == [7, 8]
