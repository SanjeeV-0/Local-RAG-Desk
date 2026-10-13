import pytest

from app.retrieval.query_router import should_decompose


@pytest.mark.parametrize(
    "query",
    [
        "What distance operators does pgvector provide for comparing vectors?",
        "How many dimensions does the production embedding model's output vector have?",
        "What token budget does the chunking pipeline target for each chunk?",
        "Why did the production system settle on HNSW rather than IVFFlat for its vector index?",
    ],
)
def test_simple_factual_queries_bypass_decomposition(query: str) -> None:
    assert should_decompose(query) is False


@pytest.mark.parametrize(
    "query",
    [
        # Two independent questions joined by ", and <wh-word>".
        "What are the main failure modes of query decomposition, and what "
        "mitigation does the system apply for each one?",
        # Multiple literal question marks.
        "What is RAG? How does reranking improve it?",
    ],
)
def test_compound_questions_use_decomposition(query: str) -> None:
    assert should_decompose(query) is True


@pytest.mark.parametrize(
    "query",
    [
        # Oxford-style list of 3+ named items being compared.
        "What are the differences between Recall@K, Precision@K, MRR, and "
        "Hit Rate@K in terms of what each one fails to capture?",
        # Three items enumerated via repeated "from" fragments instead of
        # commas.
        "How does production telemetry distinguish an embedding provider "
        "failure from an index-related failure from a reranker failure, in "
        "terms of which latency signal moves?",
    ],
)
def test_aggregation_and_list_requests_use_decomposition(query: str) -> None:
    assert should_decompose(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "List all the distance operators pgvector supports.",
        "Enumerate the failure modes of query decomposition.",
    ],
)
def test_explicit_enumeration_requests_use_decomposition(query: str) -> None:
    assert should_decompose(query) is True


@pytest.mark.parametrize(
    "query",
    [
        # Two-item pairing ("difference between X and Y") is treated as a
        # single relationship question, not an aggregation request -- see
        # Stage 2 report for why this disagrees with a syntactically
        # similar-looking benchmark label.
        "What's the fundamental difference between how IVFFlat and HNSW "
        "organize vectors so they can be searched quickly?",
        # A plain "X and Y" pairing with no comma, compound wh-clause, or
        # enumeration language is a single relationship question.
        "What distinguishes a bi-encoder from a cross-encoder when scoring "
        "a query and a passage?",
    ],
)
def test_ambiguous_two_item_comparisons_bypass_decomposition(query: str) -> None:
    """These are deliberately conservative choices documented in the Stage 2
    report: the router treats a simple "X and Y" pairing as single-intent
    rather than guessing it apart, erring toward bypass only when there is
    no comma-separated list, no compound wh-clause introduced by ", and/or",
    and no enumeration language. A query phrased with a ", or <wh-word>"
    clause (e.g. "..., or can the pipeline operate without it?") IS treated
    as compound and routed to decomposition -- see
    test_query_router_benchmark_fixture.py's EXPECTED_FALSE_DECOMPOSE_IDS
    for that documented, accepted disagreement with the benchmark label."""

    assert should_decompose(query) is False


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_or_whitespace_only_query_routes_to_decomposer(query: str) -> None:
    """The router must never bypass an empty/whitespace query -- it defers
    to QueryDecomposer.decompose, which raises ValueError for this case, so
    that behavior is unchanged whether or not the router is enabled."""

    assert should_decompose(query) is True
