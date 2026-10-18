import asyncio
from collections import defaultdict

from app.core.database import async_session_factory
from app.core.dependencies import (
    get_embedding_provider,
    get_query_decomposer,
    get_reranker,
    get_retrieval_service,
)
from app.retrieval.service import RetrievalService
from tests.evaluation.benchmark_corpus_v2 import (
    BENCHMARK_V2_DOCUMENTS,
    BENCHMARK_V2_FORMAT_BY_TITLE,
    resolve_benchmark_v2_evaluation_cases,
)
from tests.evaluation.retrieval_cases_v2 import (
    RETRIEVAL_BENCHMARK_V2_CASE_SPECS,
)
from tests.evaluation.retrieval_metrics import (
    RetrievalCaseMetrics,
    aggregate_case_metrics,
    hit_rate_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
)

CANDIDATE_LIMITS = (20, 50, 100)
FINAL_LIMITS = (3, 5, 10)

# The single configuration the per-category and per-format breakdowns are
# reported at -- closest to V1's best-performing configuration, so results
# stay comparable across the two benchmark generations.
BREAKDOWN_CANDIDATE_LIMIT = 100
BREAKDOWN_FINAL_LIMIT = 10


def _calculate_case_metrics(
    *,
    case,
    retrieved_ids,
    k: int,
) -> RetrievalCaseMetrics:
    return RetrievalCaseMetrics(
        query=case.query,
        k=k,
        recall=recall_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            k,
        ),
        precision=precision_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            k,
        ),
        mrr=mrr_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            k,
        ),
        hit_rate=hit_rate_at_k(
            retrieved_ids,
            case.relevant_chunk_ids,
            k,
        ),
    )


def _case_source_format(spec) -> str:
    """Classify a case's gold evidence by source format.

    Returns "markdown" / "docx" / "pdf" if every gold fragment for this
    case comes from documents of that single format, or "mixed" if the
    case's gold evidence genuinely spans more than one source format.
    """
    formats = {
        BENCHMARK_V2_FORMAT_BY_TITLE[relevant_chunk.document_title]
        for relevant_chunk in spec.relevant_chunks
    }
    if len(formats) == 1:
        return next(iter(formats))
    return "mixed"


def _print_corpus_overview() -> None:
    format_counts: dict[str, int] = defaultdict(int)
    for _filename, _title, _logical_id, source_format in BENCHMARK_V2_DOCUMENTS:
        format_counts[source_format] += 1

    specs = RETRIEVAL_BENCHMARK_V2_CASE_SPECS
    total_gold_fragments = sum(len(spec.relevant_chunks) for spec in specs)

    gold_by_format: dict[str, int] = defaultdict(int)
    for spec in specs:
        for relevant_chunk in spec.relevant_chunks:
            gold_by_format[BENCHMARK_V2_FORMAT_BY_TITLE[relevant_chunk.document_title]] += 1

    single_chunk = sum(1 for spec in specs if len(spec.relevant_chunks) == 1)
    multi_chunk = len(specs) - single_chunk

    cross_document = 0
    case_format_counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        titles = {rc.document_title for rc in spec.relevant_chunks}
        if len(titles) > 1:
            cross_document += 1
        case_format_counts[_case_source_format(spec)] += 1

    decomposition_count = sum(1 for spec in specs if spec.expected_subqueries)

    print("=" * 88)
    print("BENCHMARK V2 CORPUS OVERVIEW")
    print("=" * 88)
    print(
        f"Documents              : {len(BENCHMARK_V2_DOCUMENTS)}  "
        f"(by format: {dict(format_counts)})"
    )
    print(f"Questions              : {len(specs)}")
    print(f"Gold fragment refs     : {total_gold_fragments}  (by format: {dict(gold_by_format)})")
    print(f"Single/multi-chunk Qs  : {single_chunk} / {multi_chunk}")
    print(f"Cross-document Qs      : {cross_document}")
    print(f"Decomposition Qs       : {decomposition_count}")
    print("Question primary-format counts (mixed = evidence spans >1 format):")
    for fmt in sorted(case_format_counts):
        print(f"    {fmt:>10}: {case_format_counts[fmt]}")
    print()


async def _run_benchmark_v2() -> None:
    async with async_session_factory() as session:
        cases = await resolve_benchmark_v2_evaluation_cases(
            session,
            RETRIEVAL_BENCHMARK_V2_CASE_SPECS,
        )

        # cases is built from RETRIEVAL_BENCHMARK_V2_CASE_SPECS in the same
        # order (see resolve_benchmark_v2_evaluation_cases), so zipping the
        # two lets every case be classified by its gold evidence's source
        # format without adding a format field to the shared
        # RetrievalEvaluationCase dataclass that V1 also uses.
        case_formats = [
            _case_source_format(spec)
            for spec in RETRIEVAL_BENCHMARK_V2_CASE_SPECS
        ]

        retrieval_service: RetrievalService = await get_retrieval_service(
            session=session,
            embedding_provider=get_embedding_provider(),
            reranker=get_reranker(),
            query_decomposer=get_query_decomposer(),
        )

        _print_corpus_overview()

        print("=" * 88)
        print("RETRIEVAL BENCHMARK V2 -- OVERALL")
        print("=" * 88)
        print(f"Evaluation cases : {len(cases)}")
        print(f"Candidate limits : {CANDIDATE_LIMITS}")
        print(f"Final limits     : {FINAL_LIMITS}")
        print()

        breakdown_case_metrics: list[RetrievalCaseMetrics] = []
        breakdown_metrics_by_category: dict[str, list[RetrievalCaseMetrics]] = defaultdict(list)
        breakdown_metrics_by_format: dict[str, list[RetrievalCaseMetrics]] = defaultdict(list)

        for candidate_limit in CANDIDATE_LIMITS:
            for final_limit in FINAL_LIMITS:
                case_metrics: list[RetrievalCaseMetrics] = []

                for case, case_format in zip(cases, case_formats, strict=True):
                    results = await retrieval_service.search(
                        query=case.query,
                        limit=final_limit,
                        candidate_limit=candidate_limit,
                    )

                    retrieved_ids = [
                        result.chunk_id
                        for result in results
                    ]

                    metrics = _calculate_case_metrics(
                        case=case,
                        retrieved_ids=retrieved_ids,
                        k=final_limit,
                    )
                    case_metrics.append(metrics)

                    completed = len(case_metrics)
                    if completed % 5 == 0 or completed == len(cases):
                        print(
                            f"[progress] candidate={candidate_limit} "
                            f"final={final_limit} "
                            f"questions={completed}/{len(cases)}",
                            flush=True,
                        )

                    is_breakdown_config = (
                        candidate_limit == BREAKDOWN_CANDIDATE_LIMIT
                        and final_limit == BREAKDOWN_FINAL_LIMIT
                    )
                    if is_breakdown_config:
                        breakdown_case_metrics.append(metrics)
                        breakdown_metrics_by_category[case.category].append(metrics)
                        breakdown_metrics_by_format[case_format].append(metrics)

                aggregate = aggregate_case_metrics(
                    case_metrics,
                    k=final_limit,
                )

                print(
                    f"candidate={candidate_limit:>3} "
                    f"final={final_limit:>2} | "
                    f"Recall={aggregate.mean_recall:.3f} | "
                    f"Precision={aggregate.mean_precision:.3f} | "
                    f"MRR={aggregate.mean_mrr:.3f} | "
                    f"HitRate={aggregate.mean_hit_rate:.3f}"
                )

        # Category- and format-level breakdowns at one fixed configuration
        # (candidate=100, final=10 -- the config closest to V1's reported
        # best) so weaknesses concentrated in one category or one source
        # format are visible rather than hidden behind a single aggregate
        # number (see the bias-auditing guidance in "Evaluation Metrics and
        # Benchmark Design for Retrieval Systems"). Reuses the results
        # already computed in the sweep above instead of re-querying.
        print()
        print("-" * 88)
        print(
            f"BY CATEGORY (candidate={BREAKDOWN_CANDIDATE_LIMIT}, "
            f"final={BREAKDOWN_FINAL_LIMIT})"
        )
        print("-" * 88)

        for category in sorted(breakdown_metrics_by_category):
            metrics_for_category = breakdown_metrics_by_category[category]

            aggregate = aggregate_case_metrics(
                metrics_for_category,
                k=BREAKDOWN_FINAL_LIMIT,
            )

            print(
                f"{category:>20} (n={len(metrics_for_category):>3}) | "
                f"Recall={aggregate.mean_recall:.3f} | "
                f"Precision={aggregate.mean_precision:.3f} | "
                f"MRR={aggregate.mean_mrr:.3f} | "
                f"HitRate={aggregate.mean_hit_rate:.3f}"
            )

        print()
        print("-" * 88)
        print(
            f"BY SOURCE FORMAT (candidate={BREAKDOWN_CANDIDATE_LIMIT}, "
            f"final={BREAKDOWN_FINAL_LIMIT})"
        )
        print("-" * 88)

        # Fixed, meaningful order rather than alphabetical (alphabetical
        # would put "docx" before "markdown" before "mixed" before "pdf",
        # which reads fine too, but this keeps the three real formats
        # grouped before the cross-format "mixed" bucket).
        for fmt in ("markdown", "docx", "pdf", "mixed"):
            metrics_for_format = breakdown_metrics_by_format.get(fmt)
            if not metrics_for_format:
                continue

            aggregate = aggregate_case_metrics(
                metrics_for_format,
                k=BREAKDOWN_FINAL_LIMIT,
            )

            print(
                f"{fmt:>20} (n={len(metrics_for_format):>3}) | "
                f"Recall={aggregate.mean_recall:.3f} | "
                f"Precision={aggregate.mean_precision:.3f} | "
                f"MRR={aggregate.mean_mrr:.3f} | "
                f"HitRate={aggregate.mean_hit_rate:.3f}"
            )


def test_retrieval_benchmark_v2() -> None:
    asyncio.run(_run_benchmark_v2())
