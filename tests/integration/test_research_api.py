from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.config.settings import Settings
from app.core.dependencies import (
    get_app_settings,
    get_generation_service,
    get_retrieval_service,
)
from app.generation.context import GenerationContext
from app.generation.generation_service import GenerationResult
from app.main import app
from app.retrieval.models import RetrievedChunk
from app.retrieval.trace import RetrievalTrace, RetrievalTraceCandidate


class FakeRetrievalService:
    def __init__(self, router_bypassed: bool = False) -> None:
        self.last_trace = None
        self.router_bypassed = router_bypassed
        # Captures whatever the endpoint actually passed, so tests can
        # assert the per-request router override reached this far.
        self.received_router_enabled: bool | None = None

    async def search(
        self,
        query: str,
        limit: int = 10,
        max_distance=None,
        document_id=None,
        section_id=None,
        candidate_limit=None,
        trace=False,
        router_enabled=None,
    ) -> list[RetrievedChunk]:
        self.received_router_enabled = router_enabled

        result = RetrievedChunk(
            document_id=uuid4(),
            chunk_id=uuid4(),
            section_id=uuid4(),
            section_path="Results",
            page_numbers=[1],
            content="RAG retrieves relevant information before generation.",
            distance=0.1,
        )

        if trace:
            candidate = RetrievalTraceCandidate(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                section_id=result.section_id,
                section_path=result.section_path,
                page_numbers=result.page_numbers,
                content=result.content,
                distance=result.distance,
                rerank_score=result.rerank_score,
            )
            self.last_trace = RetrievalTrace(
                query=query,
                original_query=query,
                sub_queries=[query],
                candidate_limit=candidate_limit if candidate_limit is not None else max(limit, 50),
                raw_candidate_count=1,
                deduplicated_candidate_count=1,
                candidates=[candidate],
                final_results=[candidate],
                router_bypassed=self.router_bypassed,
            )
        else:
            self.last_trace = None

        return [result]


class FakeGenerationService:
    def __init__(self) -> None:
        self.received_query = None
        self.received_context = None

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        self.received_query = query
        self.received_context = context

        return GenerationResult(
            text="RAG retrieves relevant information before generation.",
            model="fake-model",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )


@pytest.mark.asyncio
async def test_research_ask_endpoint() -> None:
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()

    app.dependency_overrides[get_generation_service] = lambda: generation_service

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/research/ask",
                json={
                    "query": "What is RAG?",
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["answer"] == ("RAG retrieves relevant information before generation.")

        assert body["model"] == "fake-model"

        assert len(body["sources"]) == 1
        assert body["sources"][0]["section_path"] == "Results"

        assert generation_service.received_query == "What is RAG?"

        assert "RAG retrieves relevant information" in generation_service.received_context.text

    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("router_enabled", [True, False, None])
@pytest.mark.asyncio
async def test_research_ask_accepts_router_enabled_field(router_enabled) -> None:
    """`router_enabled` is accepted whether explicitly True, explicitly
    False, or omitted (modeled as `None`, excluded from the JSON body) --
    all three must validate and reach `RetrievalService.search` as the
    exact value sent."""

    retrieval_service = FakeRetrievalService()
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service

    payload = {"query": "What is RAG?"}
    if router_enabled is not None:
        payload["router_enabled"] = router_enabled

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json=payload)

        assert response.status_code == 200
        assert retrieval_service.received_router_enabled == router_enabled

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_research_ask_omitted_router_enabled_is_none_not_false() -> None:
    """Existing request payloads that predate this field remain compatible
    -- omitting `router_enabled` must reach `RetrievalService` as `None`,
    not coerced to `False`."""

    retrieval_service = FakeRetrievalService()
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json={"query": "What is RAG?"})

        assert response.status_code == 200
        assert retrieval_service.received_router_enabled is None

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_research_ask_trace_reports_router_bypassed_true() -> None:
    """`router_bypassed` in /research/ask's trace dict must reflect the
    actual executed decision, following the same contract as
    /retrieval/search."""

    retrieval_service = FakeRetrievalService(router_bypassed=True)
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service
    app.dependency_overrides[get_app_settings] = lambda: Settings(trace_enabled=True)

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/research/ask",
                json={"query": "What is RAG?", "trace": True, "router_enabled": True},
            )

        assert response.status_code == 200
        body = response.json()
        assert "trace" in body
        assert body["trace"]["router_bypassed"] is True

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_research_ask_trace_reports_router_bypassed_false() -> None:
    """`router_bypassed` must be False when decomposition ran (or the
    router was disabled) -- the default shape of `FakeRetrievalService`."""

    retrieval_service = FakeRetrievalService(router_bypassed=False)
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service
    app.dependency_overrides[get_app_settings] = lambda: Settings(trace_enabled=True)

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/research/ask",
                json={"query": "What is RAG?", "trace": True},
            )

        assert response.status_code == 200
        body = response.json()
        assert "trace" in body
        assert body["trace"]["router_bypassed"] is False

    finally:
        app.dependency_overrides.clear()
