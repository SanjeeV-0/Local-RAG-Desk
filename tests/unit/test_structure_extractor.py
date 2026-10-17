from app.ingestion.loaders.base import LoadedPage
from app.ingestion.structure import UnitType
from app.ingestion.structure_extractor import StructureExtractor
from app.ingestion.table_serializer import table_to_markdown


def test_extracts_markdown_headings_and_paragraphs() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=(
                "# Introduction\n\n"
                "This is the introduction.\n\n"
                "## Retrieval\n\n"
                "Retrieval improves search."
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.HEADING,
        UnitType.PARAGRAPH,
        UnitType.HEADING,
        UnitType.PARAGRAPH,
    ]

    assert units[0].section_path == "Introduction"
    assert units[2].section_path == ("Introduction > Retrieval")


def test_preserves_page_numbers() -> None:
    pages = [
        LoadedPage(
            page_number=3,
            content="Some content.",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert len(units) == 1
    assert units[0].page_numbers == [3]


def test_extracts_lists() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=("- First item\n- Second item\n1. Third item"),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert all(unit.unit_type == UnitType.LIST for unit in units)

    assert len(units) == 3


def test_extracts_markdown_table_under_a_heading() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=("# Results\n\n| Model | Score |\n| --- | --- |\n| A | 0.90 |\n| B | 0.85 |\n"),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.HEADING,
        UnitType.TABLE,
    ]

    table_unit = units[1]
    assert table_unit.section_path == "Results"
    assert table_unit.page_numbers == [1]
    assert table_unit.table is not None
    assert table_unit.table.headers == ["Model", "Score"]
    assert table_unit.table.rows == [["A", "0.90"], ["B", "0.85"]]

    # content is re-serialized through the canonical table_serializer, not
    # kept as the raw source lines verbatim.
    assert table_unit.content == table_to_markdown(table_unit.table)


def test_extracts_text_and_table_together() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=(
                "Intro paragraph before the table.\n\n"
                "| A | B |\n"
                "| --- | --- |\n"
                "| 1 | 2 |\n\n"
                "Paragraph after the table."
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.PARAGRAPH,
        UnitType.TABLE,
        UnitType.PARAGRAPH,
    ]

    assert units[0].content == "Intro paragraph before the table."
    assert units[2].content == "Paragraph after the table."


def test_extracts_multiple_tables_with_distinct_table_ids() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=(
                "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n| C | D |\n| --- | --- |\n| 3 | 4 |\n"
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    table_units = [unit for unit in units if unit.unit_type == UnitType.TABLE]

    assert len(table_units) == 2
    assert table_units[0].table.table_id != table_units[1].table.table_id
    assert table_units[0].table.headers == ["A", "B"]
    assert table_units[1].table.headers == ["C", "D"]


def test_headingless_table_has_empty_section_path() -> None:
    """A table with no preceding heading must not be assigned a fabricated
    section -- it gets the same empty section_path as any other headingless
    unit, which downstream (SemanticUnit/apply_size_guard) becomes `None`."""

    pages = [
        LoadedPage(
            page_number=1,
            content=("| A | B |\n| --- | --- |\n| 1 | 2 |\n"),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert len(units) == 1
    assert units[0].unit_type == UnitType.TABLE
    assert units[0].section_path == ""


def test_table_row_without_separator_is_not_detected_as_a_table() -> None:
    """A line that merely contains pipe characters is not a table unless
    immediately followed by a valid GFM separator row -- otherwise it (and
    the ordinary text line after it) is just paragraph text."""

    pages = [
        LoadedPage(
            page_number=1,
            content="Cost | Benefit analysis is not a table.\nIt is followed by plain text.",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert len(units) == 1
    assert units[0].unit_type == UnitType.PARAGRAPH


def test_horizontal_rule_is_not_mistaken_for_a_table_separator() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content="Some paragraph line.\n\n---\n\nAnother paragraph.",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert all(unit.unit_type == UnitType.PARAGRAPH for unit in units)


def test_isolated_numbered_line_is_a_heading_not_a_list_item() -> None:
    """Regression: a numbered section header with no blank-line separation
    from its body text (e.g. "1. Objective\\nSome body text.") was
    previously misparsed as a LIST item -- producing a near-content-free
    standalone chunk at ingestion time. An isolated numbered line (no
    numbered/bulleted neighbor immediately before or after) must now become
    a HEADING, and the following prose must be attributed to it via
    section_path -- exactly like a Markdown `#` heading."""

    pages = [
        LoadedPage(
            page_number=1,
            content="1. Objective\nThis section explains the goal of the test.",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.HEADING,
        UnitType.PARAGRAPH,
    ]
    assert units[0].content == "Objective"
    assert units[0].section_path == "Objective"
    assert units[1].content == "This section explains the goal of the test."
    assert units[1].section_path == "Objective"


def test_nested_numbered_heading_is_always_a_heading() -> None:
    """A dot-separated, nested numbered line ("2.1 Title") is never treated
    as a list item -- ad-hoc lists are not numbered this way -- and its
    section_level reflects the nesting depth."""

    pages = [
        LoadedPage(
            page_number=1,
            content=(
                "1. Overview\n"
                "Top-level introduction text.\n\n"
                "2.1 Personal Identity\n"
                "Synthetic personal information appears in this subsection."
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.HEADING,
        UnitType.PARAGRAPH,
        UnitType.HEADING,
        UnitType.PARAGRAPH,
    ]

    assert units[0].content == "Overview"
    assert units[0].section_level == 1

    assert units[2].content == "Personal Identity"
    assert units[2].section_level == 2
    # Nests under the preceding level-1 heading, same as Markdown `#`/`##`.
    assert units[2].section_path == "Overview > Personal Identity"

    assert units[3].content == (
        "Synthetic personal information appears in this subsection."
    )


def test_deeply_nested_numbered_heading_level_reflects_dot_depth() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content="2.1.3 Deeply Nested\nBody text under the deeply nested heading.",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert units[0].unit_type == UnitType.HEADING
    assert units[0].content == "Deeply Nested"
    assert units[0].section_level == 3


def test_genuine_numbered_list_is_still_a_list_not_a_heading() -> None:
    """A run of two or more numbered lines is still treated as a numbered
    list -- the fix for isolated numbered headings must not reclassify
    genuine lists."""

    pages = [
        LoadedPage(
            page_number=1,
            content="1. Buy milk\n2. Buy eggs\n3. Buy bread",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.LIST,
        UnitType.LIST,
        UnitType.LIST,
    ]
    assert [unit.content for unit in units] == ["Buy milk", "Buy eggs", "Buy bread"]


def test_numbered_item_adjacent_to_a_bullet_item_is_still_a_list() -> None:
    """Regression guard for the pre-existing mixed-marker test
    (`test_extracts_lists`): a numbered line immediately following a bullet
    line is part of that list, not an isolated heading."""

    pages = [
        LoadedPage(
            page_number=1,
            content="- First item\n- Second item\n1. Third item",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert all(unit.unit_type == UnitType.LIST for unit in units)
    assert len(units) == 3
    assert units[2].content == "Third item"


def test_numbered_heading_introducing_a_bullet_list_is_a_heading_not_a_list_item() -> None:
    """Regression (previously misclassified, now fixed): a numbered heading
    immediately followed by an unrelated bulleted list it introduces must be
    recognized as a HEADING, not swallowed into that list -- the "1." marker
    never recurs later, which is what distinguishes this from a genuine
    numbered list item. The bullets themselves are correctly attributed to
    the heading's section_path."""

    pages = [
        LoadedPage(
            page_number=1,
            content="1. Shopping\n- Milk\n- Eggs",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.HEADING,
        UnitType.LIST,
        UnitType.LIST,
    ]
    assert units[0].content == "Shopping"
    assert units[0].section_path == "Shopping"
    assert units[1].content == "Milk"
    assert units[1].section_path == "Shopping"
    assert units[2].content == "Eggs"
    assert units[2].section_path == "Shopping"


def test_numbered_heading_with_trailing_colon_introducing_a_bullet_list() -> None:
    """Same fix, with the extremely common "Requirements:" phrasing -- the
    heading's content legitimately keeps its trailing colon (no cosmetic
    stripping is attempted); only its classification is corrected."""

    pages = [
        LoadedPage(
            page_number=1,
            content="1. Requirements:\n- Must be 18 or older\n- Must have valid ID",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert units[0].unit_type == UnitType.HEADING
    assert units[0].content == "Requirements:"
    assert units[1].unit_type == UnitType.LIST
    assert units[1].section_path == "Requirements:"
    assert units[2].unit_type == UnitType.LIST
    assert units[2].section_path == "Requirements:"


def test_numbered_list_item_with_nested_bullet_sub_item_stays_a_list() -> None:
    """The trade-off case from the prior review: a numbered list item whose
    own body has a nested bullet sub-item, followed by the next numbered
    item, must still be recognized as a genuine numbered list -- the
    forward scan must look PAST the nested bullet and find the recurring
    "2." marker, rather than stopping at the first list-marker line."""

    pages = [
        LoadedPage(
            page_number=1,
            content="1. First numbered step\n- Sub-bullet detail\n2. Second numbered step",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.LIST,
        UnitType.LIST,
        UnitType.LIST,
    ]
    assert [unit.content for unit in units] == [
        "First numbered step",
        "Sub-bullet detail",
        "Second numbered step",
    ]


def test_isolated_single_item_numbered_instruction_is_an_unresolved_ambiguity() -> None:
    """Documents, rather than hides, a genuine, unresolvable ambiguity: a
    single numbered instruction with no second list item and no neighboring
    list marker at all is textually indistinguishable from a numbered
    heading -- no text-only heuristic can tell these apart reliably, so this
    is treated as a heading. This test exists so a future change to the
    heuristic cannot silently alter this known trade-off without a failing
    test calling attention to it."""

    pages = [
        LoadedPage(
            page_number=1,
            content=(
                "1. Submit the form.\nThis step must be completed before proceeding."
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert units[0].unit_type == UnitType.HEADING
    assert units[0].content == "Submit the form."


def test_substantive_body_text_is_not_lost_or_duplicated_after_heading_fix() -> None:
    """End-to-end sanity check across several numbered headings: every body
    paragraph must appear exactly once, verbatim, and no heading's title
    text leaks into a PARAGRAPH unit's content."""

    pages = [
        LoadedPage(
            page_number=1,
            content=(
                "1. Objective\n"
                "First body paragraph.\n\n"
                "2. Direct Prompt Injection\n"
                "Second body paragraph.\n\n"
                "2.1 Personal Identity\n"
                "Third body paragraph."
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    paragraph_contents = [u.content for u in units if u.unit_type == UnitType.PARAGRAPH]
    heading_contents = [u.content for u in units if u.unit_type == UnitType.HEADING]

    assert paragraph_contents == [
        "First body paragraph.",
        "Second body paragraph.",
        "Third body paragraph.",
    ]
    assert heading_contents == ["Objective", "Direct Prompt Injection", "Personal Identity"]
    # No cross-contamination: no paragraph contains a heading's title text
    # and vice versa.
    for heading_text in heading_contents:
        assert all(heading_text not in p for p in paragraph_contents)
