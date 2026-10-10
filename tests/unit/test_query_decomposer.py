import pytest

from app.retrieval.query_decomposer import QueryDecomposer


class StubProvider:
    def __init__(self, result: list[str] | Exception) -> None:
        self.result = result

    async def decompose(self, query: str) -> list[str]:
        if isinstance(self.result, Exception):
            raise self.result

        return self.result


@pytest.mark.asyncio
async def test_simple_query_preserves_original_query() -> None:
    decomposer = QueryDecomposer(StubProvider(["What is retrieval augmented generation?"]))

    result = await decomposer.decompose("What is retrieval augmented generation?")

    assert result.original_query == "What is retrieval augmented generation?"
    assert result.sub_queries == ("What is retrieval augmented generation?",)
    assert result.was_decomposed is False
    assert result.fallback_reason is None


@pytest.mark.asyncio
async def test_multi_part_query_is_decomposed() -> None:
    decomposer = QueryDecomposer(
        StubProvider(
            [
                "What is retrieval augmented generation?",
                "How does reranking improve retrieval?",
                "How should RAG context be assembled?",
            ]
        )
    )

    result = await decomposer.decompose(
        "What is RAG, how does reranking help, and how should context be assembled?"
    )

    assert result.sub_queries == (
        "What is RAG, how does reranking help, and how should context be assembled?",
        "What is retrieval augmented generation?",
        "How does reranking improve retrieval?",
        "How should RAG context be assembled?",
    )
    assert result.was_decomposed is True


@pytest.mark.asyncio
async def test_duplicate_queries_are_removed_case_insensitively() -> None:
    decomposer = QueryDecomposer(
        StubProvider(
            [
                "What is RAG?",
                " what is rag? ",
                "WHAT IS RAG?",
                "How does reranking work?",
            ]
        )
    )

    result = await decomposer.decompose("What is RAG?")

    assert result.sub_queries == (
        "What is RAG?",
        "How does reranking work?",
    )


@pytest.mark.asyncio
async def test_original_query_is_always_preserved() -> None:
    decomposer = QueryDecomposer(
        StubProvider(
            [
                "First focused question",
                "Second focused question",
            ]
        )
    )

    result = await decomposer.decompose("  Original user question  ")

    assert result.original_query == "Original user question"
    assert result.sub_queries[0] == "Original user question"


@pytest.mark.asyncio
async def test_empty_candidates_fall_back_to_original() -> None:
    decomposer = QueryDecomposer(StubProvider([]))

    result = await decomposer.decompose("Original question")

    assert result.sub_queries == ("Original question",)
    assert result.was_decomposed is False
    assert result.fallback_reason is None


@pytest.mark.asyncio
async def test_provider_failure_falls_back_to_original() -> None:
    decomposer = QueryDecomposer(StubProvider(TimeoutError("decomposition timed out")))

    result = await decomposer.decompose("Original question")

    assert result.sub_queries == ("Original question",)
    assert result.was_decomposed is False
    assert result.fallback_reason == "decomposition timed out"


@pytest.mark.asyncio
async def test_maximum_five_queries_are_enforced() -> None:
    decomposer = QueryDecomposer(
        StubProvider(
            [
                "Query one",
                "Query two",
                "Query three",
                "Query four",
                "Query five",
                "Query six",
                "Query seven",
            ]
        )
    )

    result = await decomposer.decompose("Original question")

    assert len(result.sub_queries) == 5
    assert result.sub_queries[0] == "Original question"


@pytest.mark.asyncio
async def test_configured_max_sub_queries_is_actually_used() -> None:
    """Settings-backed `max_sub_queries` (default 5, matching the previous
    hard-coded MAX_SUB_QUERIES constant) must actually bound the result --
    not just be accepted and ignored."""

    decomposer = QueryDecomposer(
        StubProvider(
            [
                "Query one",
                "Query two",
                "Query three",
            ]
        ),
        max_sub_queries=2,
    )

    result = await decomposer.decompose("Original question")

    assert len(result.sub_queries) == 2
    assert result.sub_queries[0] == "Original question"


@pytest.mark.asyncio
async def test_default_max_sub_queries_matches_previous_hard_coded_constant() -> None:
    """Constructing QueryDecomposer without `max_sub_queries` (every
    existing call site before this milestone) must preserve the exact
    previous hard-coded behavior (5)."""

    decomposer = QueryDecomposer(
        StubProvider([f"Query {i}" for i in range(10)]),
    )

    result = await decomposer.decompose("Original question")

    assert len(result.sub_queries) == 5


@pytest.mark.asyncio
async def test_empty_generated_queries_are_ignored() -> None:
    decomposer = QueryDecomposer(
        StubProvider(
            [
                "",
                "   ",
                "Useful query",
            ]
        )
    )

    result = await decomposer.decompose("Original question")

    assert result.sub_queries == (
        "Original question",
        "Useful query",
    )


@pytest.mark.asyncio
async def test_provider_returning_only_duplicates_is_not_decomposed() -> None:
    decomposer = QueryDecomposer(
        StubProvider(
            [
                "Original question",
                " original question ",
            ]
        )
    )

    result = await decomposer.decompose("Original question")

    assert result.sub_queries == ("Original question",)
    assert result.was_decomposed is False
