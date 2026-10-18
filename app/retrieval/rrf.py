"""Reciprocal Rank Fusion (RRF): combines two independent rankings of the
SAME candidate set into one fused ranking, using only rank POSITIONS, never
raw cosine distances or raw cross-encoder scores (which live on different,
non-comparable scales -- see `app.retrieval.reranker`'s module docstring for
why a cross-encoder's output is not globally calibrated).

    RRF(d) = 1 / (k + vector_rank(d)) + 1 / (k + reranker_rank(d))

`k=60` (`DEFAULT_RRF_K`) is the standard smoothing constant from Cormack,
Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and
Individual Rank Learning Methods" (SIGIR 2009) -- used here as the
conventional default, not tuned against this project's own evaluation data.

Used by `app.retrieval.service.RetrievalService` when
`ranking_strategy="rrf"` (see that module); also directly reusable by
offline evaluation tooling (`tests/evaluation/offline_reranker_diagnostics.py`
and related scripts) so the evaluation exercises the exact same fusion code
path as production, never a reimplementation.
"""

from collections.abc import Sequence
from uuid import UUID

from app.retrieval.models import RetrievedChunk

DEFAULT_RRF_K = 60


def fuse_rankings(
    vector_ranked: Sequence[RetrievedChunk],
    reranker_ranked: Sequence[RetrievedChunk],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Fuse two rankings of the IDENTICAL candidate set into one RRF-ordered
    list.

    Both arguments must already be in rank order -- a chunk's 1-indexed
    position in each list IS its rank in that ranking; this function does
    not re-sort its inputs by anything other than the RRF score it computes
    from those positions.

    Raises `ValueError` (never silently fabricates a rank or proceeds with
    bad input) when:
      - `k <= 0`;
      - either list contains a duplicate `chunk_id` (a duplicate would let
        one physical chunk contribute two different rank positions to its
        own fused score, inflating it);
      - the two lists do not contain EXACTLY the same set of `chunk_id`s
        (RRF requires both rankings to be of the same candidate pool --
        fusing mismatched sets would silently invent a rank for whichever
        list a chunk happened to be missing from).

    Deterministic tie-break: when two chunks land on the exact same fused
    RRF score, the lower-`chunk_id`-string one sorts first. This is a fixed,
    reproducible rule independent of either input list's original order or
    of dict/set iteration order, so repeated calls with the same two
    rankings always produce the same fused order.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")

    vector_ids = [chunk.chunk_id for chunk in vector_ranked]
    reranker_ids = [chunk.chunk_id for chunk in reranker_ranked]

    if len(vector_ids) != len(set(vector_ids)):
        raise ValueError("vector_ranked contains a duplicate chunk_id")

    if len(reranker_ids) != len(set(reranker_ids)):
        raise ValueError("reranker_ranked contains a duplicate chunk_id")

    if set(vector_ids) != set(reranker_ids):
        raise ValueError(
            "vector_ranked and reranker_ranked must contain exactly the same "
            "set of chunk_ids -- RRF cannot fuse mismatched candidate pools"
        )

    vector_rank_by_id: dict[UUID, int] = {
        chunk_id: rank for rank, chunk_id in enumerate(vector_ids, start=1)
    }
    reranker_rank_by_id: dict[UUID, int] = {
        chunk_id: rank for rank, chunk_id in enumerate(reranker_ids, start=1)
    }
    chunk_by_id: dict[UUID, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in reranker_ranked}

    def rrf_score(chunk_id: UUID) -> float:
        return 1.0 / (k + vector_rank_by_id[chunk_id]) + 1.0 / (k + reranker_rank_by_id[chunk_id])

    fused_ids = sorted(
        vector_ids,
        key=lambda chunk_id: (-rrf_score(chunk_id), str(chunk_id)),
    )

    return [chunk_by_id[chunk_id] for chunk_id in fused_ids]
