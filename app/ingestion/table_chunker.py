"""Splits an oversized `TableData` into row-group `TableFragment`s that each
fit a token budget, keeping every fragment a valid, self-contained Markdown
table (headers repeated on each fragment via `table_serializer`).

Part of the active ingestion pipeline: `app.ingestion.size_guard.
fragment_table_units` calls `split_table` on every TABLE unit produced by
`StructureExtractor`, before `shred_semantically`/`apply_size_guard` ever
see it (see that function's docstring for why the ordering matters). Also
still reachable from the separate, unused `structure_chunker.
group_structural_units` (see that module's docstring) -- both call sites
share this one splitting implementation rather than duplicating it.
"""

from collections.abc import Callable
from dataclasses import dataclass

from app.ingestion.structure import TableData
from app.ingestion.table_serializer import table_to_markdown


@dataclass(frozen=True)
class TableFragment:
    """A retrieval-sized fragment of a table."""

    table_id: str
    title: str | None
    fragment_index: int
    fragment_count: int
    content: str
    page_numbers: list[int]


def _estimate_tokens(text: str) -> int:
    """Estimate token count for deterministic chunk sizing.

    Default `count_tokens` for `split_table` -- preserves this exact,
    word-count-estimated behavior for every existing caller. See
    `app.ingestion.size_guard`'s module docstring for why a real tokenizer
    is only used when a caller explicitly opts in by passing one.
    """
    return max(1, len(text.split()))


def split_table(
    table: TableData,
    max_tokens: int = 600,
    count_tokens: Callable[[str], int] = _estimate_tokens,
) -> list[TableFragment]:
    """Split a large table into self-contained Markdown fragments.

    `count_tokens` measures each candidate fragment's full rendered
    Markdown (never an additive approximation), so it is exact regardless
    of which counter is supplied.
    """
    if not table.rows:
        content = table_to_markdown(table)

        return [
            TableFragment(
                table_id=table.table_id,
                title=table.title,
                fragment_index=0,
                fragment_count=1,
                content=content,
                page_numbers=table.page_numbers,
            )
        ]

    fragments: list[list[list[str]]] = []
    current_rows: list[list[str]] = []

    for row in table.rows:
        candidate_rows = current_rows + [row]

        candidate = TableData(
            table_id=table.table_id,
            title=table.title,
            headers=table.headers,
            rows=candidate_rows,
            page_numbers=table.page_numbers,
        )

        if current_rows and count_tokens(table_to_markdown(candidate)) > max_tokens:
            fragments.append(current_rows)
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        fragments.append(current_rows)

    fragment_count = len(fragments)

    return [
        TableFragment(
            table_id=table.table_id,
            title=table.title,
            fragment_index=index,
            fragment_count=fragment_count,
            content=table_to_markdown(
                TableData(
                    table_id=table.table_id,
                    title=table.title,
                    headers=table.headers,
                    rows=rows,
                    page_numbers=table.page_numbers,
                )
            ),
            page_numbers=table.page_numbers,
        )
        for index, rows in enumerate(fragments)
    ]
