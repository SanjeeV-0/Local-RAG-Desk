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
from app.generation.errors import ProviderErrorCategory, ProviderUnavailableError
from app.generation.generation_service import GenerationResult
from app.main import app
from app.retrieval.decomposition_outcome import DecompositionOutcome
from app.retrieval.models import RetrievedChunk
from app.retrieval.trace import RetrievalTrace, RetrievalTraceCandidate


class FakeRetrievalService:
    def __init__(
        self,
        router_bypassed: bool = False,
        decomposition_status: str = "not_needed",
        decomposition_reason: str | None = None,
    ) -> None:
        self.last_trace = None
        self.last_decomposition_outcome: DecompositionOutcome | None = None
        self.router_bypassed = router_bypassed
        self.decomposition_status = decomposition_status
        self.decomposition_reason = decomposition_reason
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

        # Always set, exactly like the real `RetrievalService._search` --
        # independent of `trace`.
        self.last_decomposition_outcome = DecompositionOutcome(
            original_query=query,
            sub_queries=[query],
            status=self.decomposition_status,
            reason=self.decomposition_reason,
        )

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


@pytest.mark.parametrize(
    "status,reason",
    [
        ("bypassed", None),
        ("decomposed", None),
        ("not_needed", None),
        (
            "fallback",
            "Automatic query decomposition was unavailable for this request; "
            "the original query was used directly.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_research_ask_decomposition_outcome_present_without_trace(status, reason) -> None:
    """`decomposition` must be present on /research/ask's normal response
    for every reachable status, with NO trace requested at all -- the same
    regression fix as /retrieval/search, without relying on the trace-only
    field."""

    retrieval_service = FakeRetrievalService(
        decomposition_status=status,
        decomposition_reason=reason,
    )
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json={"query": "What is RAG?"})

        assert response.status_code == 200
        body = response.json()

        assert "trace" not in body
        assert body["trace_available"] is False
        assert body["decomposition"]["status"] == status
        assert body["decomposition"]["original_query"] == "What is RAG?"
        assert body["decomposition"]["sub_queries"] == ["What is RAG?"]
        assert body["decomposition"]["reason"] == reason

    finally:
        app.dependency_overrides.clear()


class _PiiRetrievalService(FakeRetrievalService):
    """A retrieval service whose only chunk contains a synthetic PII span,
    to exercise the end-to-end redaction path through the real /research/ask
    route (not just the unit-level ContextAssembler/GenerationService)."""

    async def search(self, query: str, limit: int = 10, **kwargs) -> list[RetrievedChunk]:
        self.last_decomposition_outcome = DecompositionOutcome(
            original_query=query,
            sub_queries=[query],
            status="not_needed",
        )
        self.last_trace = None

        return [
            RetrievedChunk(
                document_id=uuid4(),
                chunk_id=uuid4(),
                section_id=uuid4(),
                section_path="Results",
                page_numbers=[1],
                content="Contact jdoe@example.com for the full dataset.",
                distance=0.1,
            )
        ]


@pytest.mark.asyncio
async def test_research_ask_reports_redaction_outcome_for_pii_bearing_source() -> None:
    retrieval_service = _PiiRetrievalService()
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json={"query": "Who do I contact?"})

        assert response.status_code == 200
        body = response.json()

        assert "jdoe@example.com" not in str(body)
        assert body["redaction"]["redaction_applied"] is True
        assert body["redaction"]["affected_source_count"] == 1
        assert body["redaction"]["possibly_incomplete"] is True

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_research_ask_clean_request_reports_no_redaction() -> None:
    retrieval_service = FakeRetrievalService()
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json={"query": "What is RAG?"})

        assert response.status_code == 200
        body = response.json()

        assert body["redaction"]["redaction_applied"] is False
        assert body["redaction"]["affected_source_count"] == 0
        assert body["redaction"]["answer_redacted"] is False
        assert body["redaction"]["possibly_incomplete"] is False

    finally:
        app.dependency_overrides.clear()


class _GuardrailRejectingGenerationService:
    async def generate(self, query: str, context: GenerationContext) -> GenerationResult:
        raise ProviderUnavailableError(ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION)


class _EmptyRetrievalService(FakeRetrievalService):
    """Retrieval succeeds but finds nothing -- there is no safe content to
    fall back to if generation then fails."""

    async def search(self, query: str, limit: int = 10, **kwargs) -> list[RetrievedChunk]:
        self.last_decomposition_outcome = DecompositionOutcome(
            original_query=query,
            sub_queries=[query],
            status="not_needed",
        )
        self.last_trace = None
        return []


@pytest.mark.asyncio
async def test_research_ask_guardrail_rejection_with_no_sources_returns_safe_502() -> None:
    """When retrieval found nothing, there is no safe partial result to
    return -- a guardrail rejection must still surface as a clear, safe
    error, never a generic 500 and never raw provider details."""

    app.dependency_overrides[get_retrieval_service] = lambda: _EmptyRetrievalService()
    app.dependency_overrides[get_generation_service] = lambda: (
        _GuardrailRejectingGenerationService()
    )

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json={"query": "What is RAG?"})

        assert response.status_code == 502
        body = response.json()

        assert body["detail"]["error"]["category"] == "guardrail_or_policy_rejection"
        assert body["detail"]["error"]["retryable"] is False
        assert "message" in body["detail"]["error"]

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_research_ask_guardrail_rejection_with_sources_returns_extractive_fallback() -> None:
    """When retrieval already found safe, usable sources, a generation
    failure must not discard them -- the request should still succeed (200)
    with a useful, locally-built extractive summary of those sources,
    clearly labeled as such (never presented as a generated answer), plus
    an explicit `partial: true` and a safe (never raw) `generation_error`."""

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
    app.dependency_overrides[get_generation_service] = lambda: (
        _GuardrailRejectingGenerationService()
    )

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json={"query": "What is RAG?"})

        assert response.status_code == 200
        body = response.json()

        assert body["partial"] is True
        assert body["answer_method"] == "extractive_fallback"
        assert body["answer"] is not None
        assert "RAG retrieves relevant information before generation." in body["answer"]
        assert "[Source 1]" in body["answer"]
        assert body["model"] is None
        assert len(body["sources"]) == 1
        assert body["sources"][0]["section_path"] == "Results"
        assert body["decomposition"]["status"] == "not_needed"
        assert body["generation_error"]["category"] == "guardrail_or_policy_rejection"
        assert body["generation_error"]["retryable"] is False
        assert "message" in body["generation_error"]
        # Never the raw ProviderUnavailableError repr/str, only the fixed
        # safe message/category fields.
        assert "Traceback" not in str(body)

        # Regression: the fallback source contains no PII at all, so the
        # redaction outcome must NOT claim the response is incomplete due to
        # redaction -- that would previously render a stray, contentless
        # "Some content was withheld" banner (empty count, dangling period).
        # The generation failure itself is already conveyed separately via
        # `partial`/`answer_method`/`generation_error`.
        assert body["redaction"]["redaction_applied"] is False
        assert body["redaction"]["affected_source_count"] == 0
        assert body["redaction"]["possibly_incomplete"] is False

    finally:
        app.dependency_overrides.clear()


class _PiiPhoneRetrievalService(FakeRetrievalService):
    """Retrieval returns a source containing a phone number, to exercise the
    extractive fallback's handling of already-redacted content end-to-end
    through the real /research/ask route."""

    async def search(self, query: str, limit: int = 10, **kwargs) -> list[RetrievedChunk]:
        self.last_decomposition_outcome = DecompositionOutcome(
            original_query=query,
            sub_queries=[query],
            status="not_needed",
        )
        self.last_trace = None

        return [
            RetrievedChunk(
                document_id=uuid4(),
                chunk_id=uuid4(),
                section_id=uuid4(),
                section_path="Support",
                page_numbers=[1],
                content="For escalation, call 555-123-4567 during business hours.",
                distance=0.1,
            )
        ]


@pytest.mark.asyncio
async def test_research_ask_extractive_fallback_never_exposes_unredacted_phone_number() -> None:
    app.dependency_overrides[get_retrieval_service] = lambda: _PiiPhoneRetrievalService()
    app.dependency_overrides[get_generation_service] = lambda: (
        _GuardrailRejectingGenerationService()
    )

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/research/ask", json={"query": "How do I escalate?"}
            )

        assert response.status_code == 200
        body = response.json()

        assert body["partial"] is True
        assert body["answer_method"] == "extractive_fallback"
        assert "555-123-4567" not in str(body)
        assert "[REDACTED:PHONE]" in body["answer"]
        assert body["redaction"]["redaction_applied"] is True
        assert body["redaction"]["affected_source_count"] == 1
        # One redacted source -- the banner has real content to describe.
        assert body["redaction"]["possibly_incomplete"] is True

    finally:
        app.dependency_overrides.clear()


class _MultiplePiiRetrievalService(FakeRetrievalService):
    """Retrieval returns several sources, more than one containing PII --
    exercises the multi-source redaction count end-to-end."""

    async def search(self, query: str, limit: int = 10, **kwargs) -> list[RetrievedChunk]:
        self.last_decomposition_outcome = DecompositionOutcome(
            original_query=query,
            sub_queries=[query],
            status="not_needed",
        )
        self.last_trace = None

        return [
            RetrievedChunk(
                document_id=uuid4(),
                chunk_id=uuid4(),
                section_id=uuid4(),
                section_path="Support",
                page_numbers=[1],
                content="For escalation, call 555-123-4567 during business hours.",
                distance=0.1,
            ),
            RetrievedChunk(
                document_id=uuid4(),
                chunk_id=uuid4(),
                section_id=uuid4(),
                section_path="Contacts",
                page_numbers=[2],
                content="Email jdoe@example.com for billing questions.",
                distance=0.12,
            ),
            RetrievedChunk(
                document_id=uuid4(),
                chunk_id=uuid4(),
                section_id=uuid4(),
                section_path="General",
                page_numbers=[3],
                content="The documentation portal is updated weekly.",
                distance=0.2,
            ),
        ]


@pytest.mark.asyncio
async def test_research_ask_extractive_fallback_reports_multiple_redacted_sources() -> None:
    app.dependency_overrides[get_retrieval_service] = lambda: _MultiplePiiRetrievalService()
    app.dependency_overrides[get_generation_service] = lambda: (
        _GuardrailRejectingGenerationService()
    )

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json={"query": "How do I get help?"})

        assert response.status_code == 200
        body = response.json()

        assert "555-123-4567" not in str(body)
        assert "jdoe@example.com" not in str(body)
        assert "[REDACTED:PHONE]" in body["answer"]
        assert "[REDACTED:EMAIL]" in body["answer"]
        assert body["redaction"]["redaction_applied"] is True
        # Exactly two of the three sources contained PII.
        assert body["redaction"]["affected_source_count"] == 2
        assert body["redaction"]["possibly_incomplete"] is True

    finally:
        app.dependency_overrides.clear()


class _EmptyContentRetrievalService(FakeRetrievalService):
    """Retrieval returns a source whose content is entirely whitespace --
    there is nothing for the extractive fallback to extract, so the plain
    safe error state must be retained instead of inventing an answer."""

    async def search(self, query: str, limit: int = 10, **kwargs) -> list[RetrievedChunk]:
        self.last_decomposition_outcome = DecompositionOutcome(
            original_query=query,
            sub_queries=[query],
            status="not_needed",
        )
        self.last_trace = None

        return [
            RetrievedChunk(
                document_id=uuid4(),
                chunk_id=uuid4(),
                section_id=uuid4(),
                section_path="Results",
                page_numbers=[1],
                content="   ",
                distance=0.1,
            )
        ]


@pytest.mark.asyncio
async def test_research_ask_retains_safe_error_when_extractive_summary_not_possible() -> None:
    app.dependency_overrides[get_retrieval_service] = lambda: _EmptyContentRetrievalService()
    app.dependency_overrides[get_generation_service] = lambda: (
        _GuardrailRejectingGenerationService()
    )

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json={"query": "What is RAG?"})

        assert response.status_code == 200
        body = response.json()

        assert body["partial"] is True
        assert body["answer_method"] == "unavailable"
        assert body["answer"] is None
        assert body["generation_error"]["category"] == "guardrail_or_policy_rejection"

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_research_ask_normal_response_reports_partial_false() -> None:
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
    app.dependency_overrides[get_generation_service] = lambda: FakeGenerationService()

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/research/ask", json={"query": "What is RAG?"})

        assert response.status_code == 200
        body = response.json()

        assert body["partial"] is False
        assert body["answer_method"] == "generated"
        assert "generation_error" not in body
        assert body["answer"] is not None

    finally:
        app.dependency_overrides.clear()
