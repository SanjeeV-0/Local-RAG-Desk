"""Builds the TEXT actually handed to the embedding model for a chunk --
distinct from `DocumentChunk.content` (what is stored and shown to users/the
reranker/generation) and from `chunk_metadata["section_path"]` (metadata).

This is the one place that decides what "heading-aware embedding input"
means, used identically by both `ChunkService.persist_chunks` (new
ingestion) and any future re-embedding workflow -- see the module docstring
in `app.ingestion.chunk_service` for why those two call sites must never
diverge. Pure and side-effect-free: no database access, no I/O, so it is
trivially unit-testable and safe to call from a batch/offline job as well as
the live ingestion path.
"""


def section_path_prefix_text(section_path: str | None) -> str:
    """Return just the prefix portion `build_embedding_text` would add ahead
    of a chunk's content -- i.e. `build_embedding_text(section_path, "")`
    reduced to only the part that doesn't depend on content. Empty when
    there is no section path.

    Used by `app.ingestion.size_guard`/`app.ingestion.table_chunker` to
    reserve exactly this many tokens of a chunk's embedding-input budget
    BEFORE splitting its content, so the complete, final
    `build_embedding_text` output (prefix + content) is guaranteed to fit --
    see those modules' `reserve_section_path_tokens` parameter. Must stay in
    exact sync with `build_embedding_text`'s own prefix formatting below.
    """

    if not section_path:
        return ""

    return f"{section_path}\n\n"


def build_embedding_text(section_path: str | None, content: str) -> str:
    """Return the text to embed for one chunk.

    For a chunk with a non-empty section path, the path is prepended ahead
    of the content, separated by a blank line -- e.g.:

        Database > Indexing > HNSW

        <original chunk content>

    This gives the embedding model the chunk's heading context as a short,
    clearly-separated prefix (never interleaved into the sentence text
    itself), so retrieval queries that name a section/topic can match on
    that context even when the chunk's own prose never repeats it verbatim.

    For a chunk with NO section path (root-level content, or a headingless
    document -- see `app.ingestion.structure_extractor`'s module docstring),
    `section_path` is `None` or `""` and this returns `content` completely
    unchanged -- there is no heading to add, and no placeholder is
    fabricated in its place.

    The embedding model's tokenizer truncates from the right (confirmed
    against the configured `sentence-transformers/all-MiniLM-L6-v2`
    tokenizer: `truncation_side == "right"`, `max_seq_length == 256`), so a
    prefix placed at the front of the text is always preserved first. When
    the configured embedding provider exposes `count_tokens`/
    `max_embedding_tokens` (see `app.embeddings.sentence_transformer`),
    `app.ingestion.size_guard`/`app.ingestion.table_chunker` already reserve
    room for this prefix and split oversized content BEFORE it ever reaches
    this function, so the combined output fits the model's real budget
    without relying on truncation. A provider without that capability falls
    back to the previous, word-count-estimated chunk sizing, where this
    prefix is pure, unbudgeted overhead and truncation remains possible.
    """

    prefix = section_path_prefix_text(section_path)

    if not prefix:
        return content

    return f"{prefix}{content}"
