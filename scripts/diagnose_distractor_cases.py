import asyncio
import selectors

from app.core.database import async_session_factory
from app.core.dependencies import (
    get_embedding_provider,
    get_query_decomposer,
    get_reranker,
    get_retrieval_service,
)
from tests.evaluation.benchmark_corpus_v2 import (
    resolve_benchmark_v2_evaluation_cases,
)
from tests.evaluation.retrieval_cases_v2 import (
    RETRIEVAL_BENCHMARK_V2_CASE_SPECS,
)

TARGET_INDICES = set(range(74, 81))  # Questions 75–81
CANDIDATE_LIMIT = 100
FINAL_LIMIT = 10


async def main() -> None:
    async with async_session_factory() as session:
        cases = await resolve_benchmark_v2_evaluation_cases(
            session,
            RETRIEVAL_BENCHMARK_V2_CASE_SPECS,
        )
        service = await get_retrieval_service(
            session=session,
            embedding_provider=get_embedding_provider(),
            reranker=get_reranker(),
            query_decomposer=get_query_decomposer(),
        )

        for index, (spec, case) in enumerate(
            zip(RETRIEVAL_BENCHMARK_V2_CASE_SPECS, cases, strict=True)
        ):
            if index not in TARGET_INDICES:
                continue

            print("\n" + "=" * 88)
            print(f"Q{index + 1:03}: {case.query}")

            results = await service.search(
                query=case.query,
                limit=FINAL_LIMIT,
                candidate_limit=CANDIDATE_LIMIT,
                trace=True,
            )
            trace = service.last_trace
            if trace is None:
                print("No retrieval trace produced.")
                continue

            gold_ids = set(case.relevant_chunk_ids)
            candidate_rank = {
                item.chunk_id: rank
                for rank, item in enumerate(trace.candidates, 1)
            }
            final_rank = {
                item.chunk_id: rank
                for rank, item in enumerate(results, 1)
            }

            print(f"Subqueries: {trace.sub_queries}")
            print(
                f"Candidate pool: {trace.deduplicated_candidate_count} unique "
                f"chunks from {trace.raw_candidate_count} raw results"
            )

            print("\nGOLD CHUNKS")
            for gold_id in gold_ids:
                print(
                    f"id={gold_id} | "
                    f"vector_rank={candidate_rank.get(gold_id, 'NOT FOUND')} | "
                    f"final_rank={final_rank.get(gold_id, 'NOT IN TOP 10')}"
                )

            print("\nTOP 10 AFTER RERANKING")
            for rank, item in enumerate(results, 1):
                marker = " [GOLD]" if item.chunk_id in gold_ids else ""
                print(
                    f"{rank}. id={item.chunk_id} "
                    f"distance={item.distance:.4f} "
                    f"rerank_score={item.rerank_score}{marker}"
                )
                print(f"   {item.section_path}")
                print(f"   {item.content[:300].replace(chr(10), ' ')}")

            print("\nTOP 10 BEFORE RERANKING")
            for rank, item in enumerate(trace.candidates[:10], 1):
                marker = " [GOLD]" if item.chunk_id in gold_ids else ""
                print(
                    f"{rank}. id={item.chunk_id} "
                    f"distance={item.distance:.4f}{marker}"
                )
                print(f"   {item.section_path}")
                print(f"   {item.content[:200].replace(chr(10), ' ')}")


if __name__ == "__main__":
    loop_factory = lambda: asyncio.SelectorEventLoop(
        selectors.SelectSelector()
    )
    asyncio.run(main(), loop_factory=loop_factory)
