"""Verifies heading-aware embedding input against the REAL configured
embedding model (not a mock) -- specifically the tokenization/truncation
behavior investigated for this milestone: `sentence-transformers/
all-MiniLM-L6-v2` truncates from the right (`truncation_side == "right"`),
so a heading prefix placed at the front of the embedding input is never
truncated away, even for an oversized chunk. This loads the real model
once (same as other tests in this suite that use
`SentenceTransformerEmbeddingProvider` directly) -- it does not call any
external API and does not touch the database.
"""

from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.ingestion.embedding_text import build_embedding_text


def test_configured_model_truncates_from_the_right() -> None:
    """Pins down the exact tokenizer behavior this milestone's design
    relies on -- if a future model swap changed this, heading prefixes
    could silently stop surviving truncation, and this test would catch
    it."""

    provider = SentenceTransformerEmbeddingProvider()

    assert provider.model.tokenizer.truncation_side == "right"
    assert provider.model.max_seq_length == 256


def test_heading_prefix_survives_embedding_an_oversized_chunk() -> None:
    """A chunk at the 500-whitespace-word size-guard ceiling already
    exceeds the model's 256-token limit on its own -- confirms embedding
    heading-aware text for such a chunk does not raise, still returns a
    384-dim vector, and (by virtue of right-side truncation) the heading
    prefix is never the part that gets cut."""

    provider = SentenceTransformerEmbeddingProvider()

    long_content = " ".join(["word"] * 500)
    embedding_text = build_embedding_text("Database > Indexing > HNSW", long_content)

    embedding = provider.embed_text(embedding_text)

    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)


def test_embedding_dimension_unaffected_by_heading_prefix_on_short_chunk() -> None:
    provider = SentenceTransformerEmbeddingProvider()

    embedding = provider.embed_text(build_embedding_text("Results", "Short chunk."))

    assert len(embedding) == 384
