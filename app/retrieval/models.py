from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk returned by vector retrieval.

    `section_id`/`section_path` are `None` for a headingless document's
    chunks (no matching DocumentSection row; the repository's outer join
    keeps such chunks retrievable rather than excluding them).
    """

    document_id: UUID
    chunk_id: UUID
    section_id: UUID | None
    section_path: str | None
    page_numbers: list[int]
    content: str
    distance: float
    rerank_score: float | None = None
    # Every sub-query (in `RetrievalService._search`'s `retrieval_queries`,
    # which includes the original query whenever it was kept as one of
    # them) whose per-sub-query vector search returned this chunk -- set
    # during candidate merging, NOT just the one sub-query tied to its kept
    # (lowest-distance) occurrence. Empty for a chunk built anywhere other
    # than `RetrievalService._search` (e.g. directly from the repository,
    # or in tests that construct a `RetrievedChunk` without this field).
    matched_sub_queries: tuple[str, ...] = ()

    @property
    def similarity(self) -> float:
        """Return cosine similarity derived from cosine distance."""
        return 1.0 - self.distance
