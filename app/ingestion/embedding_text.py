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
    prefix placed at the front of the text is always preserved; only the
    tail of an already-oversized chunk's content can be truncated -- the
    same truncation that already happens today for any sufficiently long
    chunk, independent of this prefix (chunks are capped at 500
    whitespace-estimated tokens, which routinely exceeds 256 real
    wordpiece tokens even without a heading prefix).
    """

    if not section_path:
        return content

    return f"{section_path}\n\n{content}"
