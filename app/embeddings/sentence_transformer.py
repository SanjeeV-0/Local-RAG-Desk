"""The only `EmbeddingProvider` implementation, used for both ingestion
(embedding chunks) and retrieval (embedding queries) -- the same model must
back both sides, since a query embedding is only meaningful compared against
chunk embeddings from the same model/space.

Default model's output dimension (384) is hard-coded into the database
schema (`DocumentChunk.embedding: Vector(384)` in `app.core.models`).
Changing `model_name` to a model with a different dimension requires a
schema migration; changing it to ANY different model -- same dimension or
not -- requires re-embedding every existing chunk, since vectors from
different models are not comparable even if same-sized. Nothing in this
codebase automates that re-embedding.

Also exposes `count_tokens`/`max_embedding_tokens`, an OPTIONAL, duck-typed
capability consumed by `app.ingestion.service.IngestionService` to make
chunk-sizing tokenizer-aware (see `app.ingestion.size_guard` and
`app.ingestion.table_chunker`). This is deliberately not part of the
`EmbeddingProvider` ABC -- any provider that doesn't expose these two
attributes (e.g. `DeterministicEmbeddingProvider`, test fakes) causes
ingestion to fall back to its previous, word-count-estimated behavior
unchanged, rather than being forced to implement a real tokenizer.
"""

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

# Reserves headroom in `max_embedding_tokens` against two things a single
# exact token count can't fully capture: (1) tokenizer non-additivity --
# `count_tokens(a) + count_tokens(b)` is not always exactly
# `count_tokens(a + b)` (BPE/WordPiece merges can occur across a boundary
# that didn't exist when the pieces were counted separately), which matters
# because the ingestion size guard sums per-sentence/per-paragraph counts
# rather than re-tokenizing the fully joined text at every step (see
# `app.ingestion.size_guard._split_paragraph`); and (2) the small, fixed
# cost of the "\n\n" separator between a section-path prefix and content,
# which is already counted when a section path is present but adds a tiny
# amount of unavoidable rounding risk when it is not. Eight tokens is a
# deliberately conservative, cheap-to-afford margin relative to this
# model's ~254-token usable budget.
_SAFETY_MARGIN_TOKENS = 8


class SentenceTransformerEmbeddingProvider:
    """Embedding provider backed by a Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def count_tokens(self, text: str) -> int:
        """Return the real number of wordpiece tokens `text` would produce,
        EXCLUDING special tokens (`[CLS]`/`[SEP]`) -- callers compare this
        against `max_embedding_tokens`, which already reserves room for
        them. Never truncates; this is a pure measurement."""

        return len(
            self.model.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    @property
    def max_embedding_tokens(self) -> int:
        """The usable content-token budget for one embedding input: the
        model's total sequence length, minus the special tokens it always
        adds (`[CLS]`/`[SEP]` for this model -- confirmed via
        `tokenizer.num_special_tokens_to_add`, not assumed), minus a fixed
        safety margin (see `_SAFETY_MARGIN_TOKENS`)."""

        total_budget = self.model.max_seq_length
        special_tokens = self.model.tokenizer.num_special_tokens_to_add(
            pair=False,
        )

        return max(
            1,
            total_budget - special_tokens - _SAFETY_MARGIN_TOKENS,
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        No `normalize_embeddings=True` is passed -- the raw model output is
        returned as-is. Cosine distance is still computed correctly at query
        time by pgvector's `cosine_distance` operator regardless of whether
        stored vectors are pre-normalized.
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings while preserving input order."""
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
        )

        return [embedding.tolist() for embedding in embeddings]
