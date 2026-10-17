"""Final step before persistence in the active ingestion pipeline: converts
`SemanticUnit`s into `ChildChunk`s (the shape `ChunkService.persist_chunks`
embeds and stores), enforcing a maximum per-chunk token budget. The caller
(`IngestionService._process_document`) passes either a tokenizer-derived
budget (when the configured embedding provider exposes one -- see
`app.embeddings.sentence_transformer`) or the previous hard-coded 500
word-estimated fallback.

Also owns `fragment_table_units`, the pre-fragmentation pass that MUST run
on `StructureExtractor`'s output before `shred_semantically`/`apply_size_guard`
ever see a TABLE unit (see `apply_size_guard`'s ValueError below) -- it is
co-located here because both functions enforce the same `max_tokens` budget,
just for different content shapes (prose vs. table rows).

`estimate_tokens` is explicitly a whitespace-word-count approximation, not a
real tokenizer -- kept as the default `count_tokens` callable so every
existing caller (including all unit tests) is completely unaffected unless
it explicitly opts into a real tokenizer. Both `apply_size_guard` and
`fragment_table_units` accept an optional `count_tokens` callable to measure
length with a real tokenizer instead, and an optional
`reserve_section_path_tokens` flag (default `False`, preserving today's
exact behavior) that, when enabled, additionally reserves enough of
`max_tokens` for a unit's own `section_path` prefix (see
`app.ingestion.embedding_text.section_path_prefix_text`) so the complete
`build_embedding_text` output -- prefix plus content -- is guaranteed to fit,
not just the content alone. This is opt-in rather than automatic because
unconditionally reserving that overhead would shrink the effective budget
for every existing caller, including unit tests that exercise tiny
`max_tokens` values where a few reserved tokens would change their
documented chunk-count assertions; `IngestionService` is the only
production call site and explicitly opts in.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from app.ingestion.embedding_text import section_path_prefix_text
from app.ingestion.section_map import SectionMap
from app.ingestion.semantic_shredder import SemanticUnit
from app.ingestion.structure import StructuralUnit, UnitType
from app.ingestion.table_chunker import split_table


@dataclass(frozen=True)
class ChildChunk:
    """Final retrieval unit produced by the size guard."""

    index: int
    content: str
    page_numbers: list[int]
    section_id: UUID | None
    section_path: str | None
    section_level: int
    source_units: tuple[StructuralUnit, ...]


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using whitespace-separated words.

    This is deterministic and dependency-free. A production tokenizer
    can replace this implementation later.
    """
    return len(text.split())


def _content_budget_for(
    section_path: str | None,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    reserve_section_path_tokens: bool,
) -> int:
    """Return how many tokens of `max_tokens` are actually available for a
    unit's CONTENT, after optionally reserving room for its section-path
    prefix.

    When `reserve_section_path_tokens` is `False` (the default everywhere
    except `IngestionService`'s production call site), this returns
    `max_tokens` unchanged -- exactly today's behavior. When `True`, a
    non-empty `section_path` reserves `count_tokens(section_path_prefix)`
    tokens, so splitting content to fit the remainder guarantees the final
    `build_embedding_text` output (prefix + content) fits `max_tokens`
    overall, not just the content in isolation.

    Raises `ValueError` if the section path alone would consume the entire
    budget (or more) -- an explicit failure, matching this module's existing
    policy for an unshrinkable table row, rather than silently producing
    content with zero or negative room to exist in.
    """

    if not reserve_section_path_tokens:
        return max_tokens

    prefix = section_path_prefix_text(section_path)

    if not prefix:
        return max_tokens

    reserved = count_tokens(prefix)
    budget = max_tokens - reserved

    if budget < 1:
        raise ValueError(
            "Section path prefix alone consumes the entire embedding token "
            f"budget ({reserved} tokens reserved from a {max_tokens}-token "
            "budget); no room remains for any content."
        )

    return budget


def fragment_table_units(
    units: list[StructuralUnit],
    max_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
    reserve_section_path_tokens: bool = False,
) -> list[StructuralUnit]:
    """Pre-fragment every TABLE unit so its content already fits `max_tokens`
    before `shred_semantically`/`apply_size_guard` ever see it.

    `shred_semantically` always isolates a TABLE unit into its own
    `SemanticUnit` (never merges it with surrounding prose), and
    `apply_size_guard` cannot itself split an oversized table (splitting a
    markdown table by sentence/word boundaries the way prose is split would
    destroy its row structure) -- it raises instead. So fragmentation MUST
    happen here, upstream, using the existing row-group splitting algorithm
    (`app.ingestion.table_chunker.split_table`), which keeps each fragment a
    valid, self-contained markdown table (headers repeated on every
    fragment) and is called unconditionally for every table, even one
    already under `max_tokens` (it simply returns a single fragment in that
    case).

    `count_tokens`/`reserve_section_path_tokens` behave exactly as described
    in this module's docstring -- defaults preserve today's exact,
    word-count-estimated, unreserved behavior.

    Non-TABLE units pass through unchanged. A table whose single largest row
    (with headers) still exceeds the (possibly section-path-reserved) budget
    on its own cannot be split further by `split_table` (it only splits at
    row boundaries, not within a row) -- that remains a known, documented
    limitation: such a fragment would still trip `apply_size_guard`'s
    ValueError below, by design -- never silently truncated or corrupted.
    """

    fragmented: list[StructuralUnit] = []

    for unit in units:
        if unit.unit_type != UnitType.TABLE or unit.table is None:
            fragmented.append(unit)
            continue

        unit_budget = _content_budget_for(
            unit.section_path,
            max_tokens,
            count_tokens,
            reserve_section_path_tokens,
        )

        for fragment in split_table(
            unit.table,
            max_tokens=unit_budget,
            count_tokens=count_tokens,
        ):
            fragmented.append(
                StructuralUnit(
                    unit_type=UnitType.TABLE,
                    content=fragment.content,
                    page_numbers=fragment.page_numbers,
                    section_path=unit.section_path,
                    section_level=unit.section_level,
                    section_index=unit.section_index,
                    table=unit.table,
                    metadata={
                        "content_type": "table",
                        "table_id": fragment.table_id,
                        "table_title": fragment.title,
                        "table_fragment_index": fragment.fragment_index,
                        "table_fragment_count": fragment.fragment_count,
                    },
                )
            )

    return fragmented


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentence-like units."""
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )
        if sentence.strip()
    ]


def _split_hard(
    text: str,
    max_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> list[str]:
    """Hard-split text into word-boundary groups that each fit `max_tokens`
    as measured by `count_tokens`.

    Iterative rather than a fixed `words[i:i+max_tokens]` slice so this
    stays correct for a real tokenizer, where word count and wordpiece
    count diverge -- for the default `estimate_tokens` (plain word count),
    this produces IDENTICAL groupings to a fixed-size slice, since adding
    one word at a time and flushing exactly when the group would exceed
    `max_tokens` words is equivalent to slicing every `max_tokens` words.

    A single word whose own token count already exceeds `max_tokens` cannot
    be split further (no smaller boundary than a word exists here) and is
    emitted as its own, still-oversized group -- a pre-existing, documented
    limitation, not new behavior.
    """
    words = text.split()

    if not words:
        return []

    groups: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = current + [word]

        if current and count_tokens(" ".join(candidate)) > max_tokens:
            groups.append(" ".join(current))
            current = [word]
        else:
            current = candidate

    if current:
        groups.append(" ".join(current))

    return groups


def _split_paragraph(
    text: str,
    max_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> list[str]:
    """
    Split an oversized paragraph using sentence boundaries first.

    Falls back to hard token splitting for oversized sentences.
    """
    if count_tokens(text) <= max_tokens:
        return [text.strip()]

    sentences = _split_sentences(text)

    if not sentences:
        return _split_hard(text, max_tokens, count_tokens=count_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if sentence_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0

            chunks.extend(
                _split_hard(
                    sentence,
                    max_tokens,
                    count_tokens=count_tokens,
                )
            )
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_prose_unit(
    unit: StructuralUnit,
    max_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> list[str]:
    """Split an oversized prose structural unit."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            unit.content.strip(),
        )
        if paragraph.strip()
    ]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)

        if paragraph_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0

            chunks.extend(
                _split_paragraph(
                    paragraph,
                    max_tokens,
                    count_tokens=count_tokens,
                )
            )
            continue

        if current and current_tokens + paragraph_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

        current.append(paragraph)
        current_tokens += paragraph_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def apply_size_guard(
    semantic_units: list[SemanticUnit],
    section_map: SectionMap,
    max_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
    reserve_section_path_tokens: bool = False,
) -> list[ChildChunk]:
    """
    Convert semantic units into final child chunks.

    Semantic boundaries are preserved whenever possible.

    Oversized prose is split in this order:

    1. paragraph boundaries
    2. sentence boundaries
    3. hard token boundaries

    `count_tokens`/`reserve_section_path_tokens` behave exactly as described
    in this module's docstring -- defaults preserve today's exact,
    word-count-estimated, unreserved behavior.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    children: list[ChildChunk] = []

    for semantic_unit in semantic_units:
        content = semantic_unit.content

        if not content:
            continue
        section_id = (
            section_map.get(semantic_unit.section_path)
            if semantic_unit.section_path is not None
            else None
        )

        unit_budget = _content_budget_for(
            semantic_unit.section_path,
            max_tokens,
            count_tokens,
            reserve_section_path_tokens,
        )

        if count_tokens(content) <= unit_budget:
            children.append(
                ChildChunk(
                    index=len(children),
                    content=content,
                    page_numbers=semantic_unit.page_numbers,
                    section_id=section_id,
                    section_path=semantic_unit.section_path,
                    section_level=semantic_unit.section_level,
                    source_units=semantic_unit.units,
                )
            )
            continue

        for unit in semantic_unit.units:
            if unit.unit_type == UnitType.TABLE:
                # Reaching here means a TABLE unit arrived without having
                # been pre-fragmented by `fragment_table_units` (or a single
                # fragment's largest row still didn't fit the budget -- see
                # that function's docstring). Tables can't be split by
                # sentence/word boundaries like prose without destroying
                # their structure, so this is a hard stop, not a fallback --
                # never silently truncated or corrupted.
                raise ValueError(
                    "Table exceeds size guard; table fragmentation must occur before size guard."
                )

            source_unit_budget = _content_budget_for(
                unit.section_path,
                max_tokens,
                count_tokens,
                reserve_section_path_tokens,
            )

            pieces = _split_prose_unit(
                unit,
                source_unit_budget,
                count_tokens=count_tokens,
            )

            for piece in pieces:
                children.append(
                    ChildChunk(
                        index=len(children),
                        content=piece,
                        page_numbers=list(unit.page_numbers),
                        section_id=section_id,
                        section_path=unit.section_path,
                        section_level=unit.section_level,
                        source_units=(unit,),
                    )
                )

    return children
