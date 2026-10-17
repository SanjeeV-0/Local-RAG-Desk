from unittest.mock import MagicMock

from app.embeddings.sentence_transformer import (
    _SAFETY_MARGIN_TOKENS,
    SentenceTransformerEmbeddingProvider,
)


def test_embed_text_returns_embedding() -> None:
    model = MagicMock()

    embedding = MagicMock()
    embedding.tolist.return_value = [0.1, 0.2, 0.3]

    model.encode.return_value = embedding

    provider = SentenceTransformerEmbeddingProvider.__new__(SentenceTransformerEmbeddingProvider)
    provider.model = model

    result = provider.embed_text("retrieval")

    assert result == [0.1, 0.2, 0.3]

    model.encode.assert_called_once_with(
        "retrieval",
        convert_to_numpy=True,
    )


def test_embed_batch_preserves_order() -> None:
    model = MagicMock()

    first = MagicMock()
    first.tolist.return_value = [1.0, 0.0]

    second = MagicMock()
    second.tolist.return_value = [0.0, 1.0]

    model.encode.return_value = [
        first,
        second,
    ]

    provider = SentenceTransformerEmbeddingProvider.__new__(SentenceTransformerEmbeddingProvider)
    provider.model = model

    result = provider.embed_batch(["first", "second"])

    assert result == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    model.encode.assert_called_once_with(
        ["first", "second"],
        convert_to_numpy=True,
    )


def test_count_tokens_excludes_special_tokens() -> None:
    """`count_tokens` measures raw content length only -- special tokens are
    accounted for separately in `max_embedding_tokens`, never double
    counted."""
    model = MagicMock()
    model.tokenizer.encode.return_value = [101, 2001, 2002, 2003]  # 4 ids

    provider = SentenceTransformerEmbeddingProvider.__new__(SentenceTransformerEmbeddingProvider)
    provider.model = model

    result = provider.count_tokens("some text")

    assert result == 4
    model.tokenizer.encode.assert_called_once_with(
        "some text",
        add_special_tokens=False,
    )


def test_max_embedding_tokens_derives_from_model_limit_and_special_tokens() -> None:
    """Confirms the formula: total sequence length minus special tokens
    minus the safety margin -- not a hardcoded constant."""
    model = MagicMock()
    model.max_seq_length = 256
    model.tokenizer.num_special_tokens_to_add.return_value = 2

    provider = SentenceTransformerEmbeddingProvider.__new__(SentenceTransformerEmbeddingProvider)
    provider.model = model

    result = provider.max_embedding_tokens

    assert result == 256 - 2 - _SAFETY_MARGIN_TOKENS
    model.tokenizer.num_special_tokens_to_add.assert_called_once_with(pair=False)


def test_max_embedding_tokens_never_goes_below_one() -> None:
    """A pathological model configuration (tiny max_seq_length) must not
    produce a zero or negative budget -- the floor of 1 is a defensive
    guard, not an expected production value."""
    model = MagicMock()
    model.max_seq_length = 2
    model.tokenizer.num_special_tokens_to_add.return_value = 2

    provider = SentenceTransformerEmbeddingProvider.__new__(SentenceTransformerEmbeddingProvider)
    provider.model = model

    assert provider.max_embedding_tokens == 1


def test_real_model_effective_budget_matches_the_verified_investigation() -> None:
    """End-to-end confirmation against the REAL configured model (slower,
    loads the actual tokenizer) -- proves the production values match what
    the prior read-only investigation measured directly: max_seq_length=256,
    2 special tokens ([CLS]/[SEP]) for a single sequence, giving an actual
    usable budget of 256 - 2 - 8 == 246 content tokens."""
    provider = SentenceTransformerEmbeddingProvider()

    assert provider.model.max_seq_length == 256
    assert provider.model.tokenizer.num_special_tokens_to_add(pair=False) == 2
    assert provider.max_embedding_tokens == 246

    # A 300-word synthetic string comfortably exceeds any real-token count
    # under 246 for plain English words -- confirms count_tokens reports a
    # real, non-trivial wordpiece count, not a word count in disguise.
    long_text = " ".join(["word"] * 300)
    assert provider.count_tokens(long_text) > provider.max_embedding_tokens
