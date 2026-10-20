"""The only `Reranker` implementation: a sentence-transformers CrossEncoder
scoring (query, chunk) pairs directly (as opposed to the bi-encoder used for
embeddings, which scores query and chunk independently). `model_name`'s
default here is overridden by `Settings.reranker_model` in practice -- see
`app.core.dependencies.get_reranker`.
"""

from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from app.retrieval.models import RetrievedChunk


class CrossEncoderReranker:
    """Reranks retrieval candidates using a cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Return chunks ordered by cross-encoder relevance.

        Scores and reorders the ENTIRE input -- there is no internal top-N
        cutoff or batch-size limit here; truncation to the caller's final
        `limit` happens after this returns (see
        `RetrievalService._search`). `distance` (cosine distance from the
        original vector search) and `matched_sub_queries` are preserved
        unchanged; only `rerank_score` is newly populated on the returned
        chunks.
        """

        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            RetrievedChunk(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                section_id=chunk.section_id,
                section_path=chunk.section_path,
                page_numbers=chunk.page_numbers,
                content=chunk.content,
                distance=chunk.distance,
                rerank_score=float(score),
                matched_sub_queries=chunk.matched_sub_queries,
            )
            for chunk, score in ranked
        ]
