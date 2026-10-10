from app.retrieval.decomposition import (
    MAX_SUB_QUERIES,
    QueryDecompositionProvider,
    QueryDecompositionResult,
)


class QueryDecomposer:
    """Normalize and validate LLM-generated query decomposition.

    This is the boundary between "whatever a QueryDecompositionProvider
    returns" (untrusted: could be malformed JSON, too many queries,
    duplicates, or a network/API failure) and "a safe, bounded list of
    retrieval queries that RetrievalService can run directly". It owns the
    original-query-preservation and failure-fallback invariants documented
    below -- RetrievalService never has to handle a provider failure itself.
    """

    def __init__(
        self,
        provider: QueryDecompositionProvider,
        max_sub_queries: int = MAX_SUB_QUERIES,
    ) -> None:
        self.provider = provider
        # Defaults to the existing hard-coded constant, so constructing a
        # QueryDecomposer without this argument (every existing call site,
        # including unit tests) preserves today's exact behavior. Settings-
        # backed via `Settings.query_decomposition_max_sub_queries` in
        # production (see app.core.dependencies.get_query_decomposer).
        self.max_sub_queries = max_sub_queries

    async def decompose(self, query: str) -> QueryDecompositionResult:
        """Decompose `query`, never raising for provider-side failures.

        Any exception from `self.provider.decompose` (network error,
        malformed JSON, wrong response shape, non-string entries, ...) is
        caught here and converted into a fallback result containing only the
        original query -- retrieval must still be able to proceed with a
        single-query search even if the decomposition LLM is unreachable or
        misbehaves. Only an empty/whitespace-only `query` is a hard failure
        (raises `ValueError`), since there is no sensible query to fall back
        to in that case.
        """

        original_query = query.strip()

        if not original_query:
            raise ValueError("query must not be empty")

        try:
            candidates = await self.provider.decompose(original_query)
            sub_queries = self._normalize_queries(
                original_query,
                candidates,
                self.max_sub_queries,
            )

            return QueryDecompositionResult(
                original_query=original_query,
                sub_queries=tuple(sub_queries),
                was_decomposed=len(sub_queries) > 1,
            )
        except Exception as exc:
            return QueryDecompositionResult(
                original_query=original_query,
                sub_queries=(original_query,),
                was_decomposed=False,
                fallback_reason=str(exc),
            )

    @staticmethod
    def _normalize_queries(
        original_query: str,
        candidates: list[str],
        max_sub_queries: int,
    ) -> list[str]:
        """Build the final, bounded sub-query list.

        Guarantees (relied on by RetrievalService and by
        tests/unit/test_query_decomposer.py):
          - the original query is always present, always first, and can
            never be dropped by deduplication -- it is added before any
            LLM-provided candidate is considered;
          - blank/whitespace-only candidates are discarded;
          - duplicates are removed case-insensitively (`.casefold()`), so
            "What is RAG?" and "what is rag?" count as the same query;
          - the result never exceeds `max_sub_queries` entries (including
            the original), even if the provider returned more.
        """

        queries: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            normalized = value.strip()

            if not normalized:
                return

            key = normalized.casefold()

            if key in seen:
                return

            seen.add(key)
            queries.append(normalized)

        # Original query is always first and can never be removed.
        add(original_query)

        for candidate in candidates:
            add(candidate)

            if len(queries) >= max_sub_queries:
                break

        return queries
