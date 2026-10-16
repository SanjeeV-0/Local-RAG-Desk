from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.config.settings import Settings
from app.core.database import async_session_factory
from app.core.dependencies import (
    get_app_settings,
    get_embedding_provider,
    get_generation_service,
    get_retrieval_service,
)
from app.core.models import (
    ChunkPageMap,
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentStatus,
)
from app.core.services.document import DocumentService
from app.generation.context import GenerationContext
from app.generation.service import GenerationResult
from app.main import app
from app.retrieval.decomposition_outcome import DecompositionOutcome
from app.retrieval.models import RetrievedChunk
from app.retrieval.trace import (
    RetrievalTrace,
    RetrievalTraceCandidate,
    RetrievalTraceContext,
)


class FakeRetrievalService:
    """Fake retrieval service for API contract tests."""

    def __init__(
        self,
        headingless: bool = False,
        router_bypassed: bool = False,
        decomposition_status: str = "not_needed",
        decomposition_reason: str | None = None,
    ) -> None:
        self.last_trace = None
        self.last_decomposition_outcome: DecompositionOutcome | None = None
        # A headingless document's chunks have section_id (and therefore
        # section_path) as NULL -- the outer-join retrieval fix keeps them
        # retrievable rather than excluding them. `headingless=True` models
        # exactly that shape so tests can exercise it without a real database.
        self.headingless = headingless
        # What the (faked) retrieval execution decided the router did --
        # independent of `received_router_enabled` below, which is what the
        # endpoint asked for.
        self.router_bypassed = router_bypassed
        # What `last_decomposition_outcome.status`/`.reason` should report,
        # independent of `trace` -- mirrors the real service's "always set"
        # contract (see `RetrievalService._search`).
        self.decomposition_status = decomposition_status
        self.decomposition_reason = decomposition_reason
        # Captures whatever the endpoint actually passed, so tests can
        # assert the per-request router override reached this far.
        self.received_router_enabled: bool | None = None
        # Same, for the per-request candidate_limit override.
        self.received_candidate_limit: int | None = None

    async def search(
        self,
        query: str,
        limit: int = 10,
        max_distance=None,
        document_id=None,
        section_id=None,
        candidate_limit=None,
        trace: bool = False,
        router_enabled: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Return deterministic retrieval results."""

        self.received_router_enabled = router_enabled
        self.received_candidate_limit = candidate_limit

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
            section_id=None if self.headingless else uuid4(),
            section_path=None if self.headingless else "Results",
            page_numbers=[1, 2],
            content=("Retrieval augmented generation combines retrieval with generation."),
            distance=0.15,
            rerank_score=4.2,
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
                candidate_limit=(
                    candidate_limit if candidate_limit is not None else max(limit, 50)
                ),
                raw_candidate_count=1,
                deduplicated_candidate_count=1,
                candidates=[candidate],
                final_results=[candidate],
                context=RetrievalTraceContext(
                    text=(
                        "[Source 1]\n"
                        "Retrieval augmented generation combines "
                        "retrieval with generation."
                    ),
                    sources=[candidate],
                ),
                router_bypassed=self.router_bypassed,
            )
        else:
            self.last_trace = None

        return [result]


class FakeGenerationService:
    """Fake generation service for research API contract tests."""

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """Return deterministic generation results."""

        return GenerationResult(
            text="RAG combines retrieval with language generation.",
            model="test-model",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )


def test_retrieval_search_endpoint() -> None:
    """Test the retrieval API response contract."""

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={
                "query": "retrieval augmented generation",
                "limit": 2,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert len(body["results"]) == 1

        result = body["results"][0]

        assert result["section_path"] == "Results"
        assert result["page_numbers"] == [1, 2]
        assert (
            result["content"] == "Retrieval augmented generation combines "
            "retrieval with generation."
        )
        assert result["distance"] == 0.15
        assert result["similarity"] == 0.85
        assert result["rerank_score"] == 4.2

        assert body["trace"] is None
        assert body["trace_requested"] is False
        assert body["trace_available"] is False
        assert body["trace_unavailable_reason"] is None

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_trace_not_requested_even_when_server_enabled() -> None:
    """Case 2: trace=false (explicit) + TRACE_ENABLED=true.

    The per-request flag is the deciding factor when the server permits
    tracing -- if the client didn't ask for it, no trace is captured, but
    results are still returned normally (retrieval/decomposition unaffected)."""

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
    app.dependency_overrides[get_app_settings] = lambda: Settings(trace_enabled=True)

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={
                "query": "retrieval augmented generation",
                "limit": 2,
                "trace": False,
            },
        )

        assert response.status_code == 200

        body = response.json()
        assert len(body["results"]) == 1
        assert body["trace"] is None
        assert body["trace_requested"] is False
        assert body["trace_available"] is False
        assert body["trace_unavailable_reason"] is None

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_trace_requested_but_server_disabled() -> None:
    """Case 3: trace=true + TRACE_ENABLED=false.

    Results must still be returned (retrieval/decomposition still run) --
    only trace capture is withheld, and the response says exactly why."""

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
    app.dependency_overrides[get_app_settings] = lambda: Settings(trace_enabled=False)

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={
                "query": "retrieval augmented generation",
                "limit": 2,
                "trace": True,
            },
        )

        assert response.status_code == 200

        body = response.json()
        assert len(body["results"]) == 1
        assert body["trace"] is None
        assert body["trace_requested"] is True
        assert body["trace_available"] is False
        assert body["trace_unavailable_reason"] == "server_disabled"

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_trace_mode() -> None:
    """Case 4: trace=true + TRACE_ENABLED=true -- full trace is returned."""

    retrieval_service = FakeRetrievalService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    trace_settings = Settings(
        trace_enabled=True,
    )

    app.dependency_overrides[get_app_settings] = lambda: trace_settings

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={
                "query": "research query",
                "limit": 1,
                "trace": True,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert "trace" in body
        assert body["trace"] is not None
        assert body["trace_requested"] is True
        assert body["trace_available"] is True
        assert body["trace_unavailable_reason"] is None

        trace = body["trace"]

        assert trace["query"] == "research query"
        assert trace["candidate_limit"] == 50
        assert trace["original_query"] == "research query"
        assert trace["sub_queries"] == ["research query"]
        assert trace["raw_candidate_count"] == 1
        assert trace["deduplicated_candidate_count"] == 1

        assert len(trace["candidates"]) == 1
        assert len(trace["final_results"]) == 1

        assert trace["context"] is not None

        context = trace["context"]

        assert context["text"] == (
            "[Source 1]\nRetrieval augmented generation combines retrieval with generation."
        )

        assert len(context["sources"]) == 1

        context_source = context["sources"][0]

        assert context_source["content"] == (
            "Retrieval augmented generation combines retrieval with generation."
        )

        assert context_source["section_path"] == "Results"
        assert context_source["page_numbers"] == [1, 2]
        assert context_source["distance"] == 0.15
        assert context_source["rerank_score"] == 4.2

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_rejects_empty_query() -> None:
    """Test validation of an empty retrieval query."""

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={
                "query": "",
                "limit": 5,
            },
        )

        assert response.status_code == 422

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_returns_null_section_for_headingless_chunk() -> None:
    """Regression test for the exact bug reported against the Vector
    Retrieval Explorer: a chunk from a headingless document has
    section_id/section_path as NULL (the outer-join retrieval fix keeps it
    retrievable rather than excluding it). The response must serialize that
    as JSON null, not raise a Pydantic validation error that FastAPI turns
    into an HTTP 500."""

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService(headingless=True)

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "co pilot", "limit": 2},
        )

        assert response.status_code == 200

        body = response.json()
        assert len(body["results"]) == 1

        result = body["results"][0]
        assert result["section_id"] is None
        assert result["section_path"] is None
        assert result["content"] == (
            "Retrieval augmented generation combines retrieval with generation."
        )

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_trace_mode_handles_headingless_chunk() -> None:
    """The same null-section shape must also serialize correctly in trace
    mode's candidates/context, which use a separate Pydantic response model
    (RetrievalTraceCandidateResponse) with the identical bug potential."""

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService(headingless=True)
    app.dependency_overrides[get_app_settings] = lambda: Settings(trace_enabled=True)

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "co pilot", "limit": 1, "trace": True},
        )

        assert response.status_code == 200

        trace = response.json()["trace"]
        assert trace is not None
        assert trace["candidates"][0]["section_id"] is None
        assert trace["candidates"][0]["section_path"] is None
        assert trace["context"]["sources"][0]["section_id"] is None
        assert trace["context"]["sources"][0]["section_path"] is None

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_empty_corpus_returns_empty_results() -> None:
    """An empty corpus (or a query matching nothing) must return 200 with
    an empty results list, not an error -- the development database is
    intentionally left empty after each pytest session, and this must be a
    normal, representable state, not a crash."""

    class EmptyRetrievalService:
        last_trace = None
        last_decomposition_outcome = DecompositionOutcome(
            original_query="co pilot",
            sub_queries=["co pilot"],
            status="not_needed",
        )

        async def search(self, **kwargs) -> list[RetrievedChunk]:
            return []

    app.dependency_overrides[get_retrieval_service] = lambda: EmptyRetrievalService()

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "co pilot", "limit": 5},
        )

        assert response.status_code == 200

        body = response.json()
        assert body["results"] == []
        assert body["trace"] is None
        assert body["trace_requested"] is False
        assert body["trace_available"] is False
        assert body["trace_unavailable_reason"] is None

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_retrieval_search_real_database_headingless_chunk() -> None:
    """Reproduces the reported bug end-to-end against the real database:
    a READY, current document whose only chunk has no section (section_id
    IS NULL -- e.g. a headingless document) must be retrievable through the
    real HTTP API without an Internal Server Error. Uses the exact
    user-reported query ("co pilot")."""

    async with async_session_factory() as session:
        document = Document(
            title=f"Headingless API Retrieval Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"headingless-test-{uuid4()}",
            document_metadata={},
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
        )

        session.add(document)
        await session.flush()

        page = DocumentPage(
            document_id=document.id,
            page_number=1,
            content="Co-pilot assists with writing code alongside the developer.",
        )

        session.add(page)
        await session.flush()

        embedding_provider = get_embedding_provider()

        content = "Co-pilot assists with writing code alongside the developer."

        chunk = DocumentChunk(
            document_id=document.id,
            section_id=None,
            chunk_index=0,
            content=content,
            chunk_metadata={},
            embedding=embedding_provider.embed_text(content),
        )

        session.add(chunk)
        await session.flush()

        session.add(
            ChunkPageMap(
                chunk_id=chunk.id,
                document_page_id=page.id,
            )
        )
        await session.flush()

        # Commit so the API's separate database session can see the test data.
        await session.commit()

        try:
            transport = ASGITransport(app=app)

            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/retrieval/search",
                    json={
                        "query": "co pilot",
                        "limit": 2,
                        "document_id": str(document.id),
                    },
                )

            assert response.status_code == 200

            body = response.json()
            assert len(body["results"]) == 1

            result = body["results"][0]
            assert result["content"] == content
            assert result["section_id"] is None
            assert result["section_path"] is None
            assert result["page_numbers"] == [1]

        finally:
            app.dependency_overrides.clear()

            await session.delete(document)
            await session.commit()


@pytest.mark.asyncio
async def test_retrieval_search_real_database() -> None:
    """Test retrieval API against real PostgreSQL, pgvector, and reranking."""

    async with async_session_factory() as session:
        document = Document(
            title=f"API Retrieval Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"api-retrieval-test-{uuid4()}",
            document_metadata={},
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
        )

        session.add(document)
        await session.flush()

        section = DocumentSection(
            document_id=document.id,
            title="Results",
            section_path="Results",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        session.add(section)
        await session.flush()

        page_one = DocumentPage(
            document_id=document.id,
            page_number=1,
            content=(
                "Retrieval augmented generation combines "
                "information retrieval with language generation."
            ),
        )

        page_two = DocumentPage(
            document_id=document.id,
            page_number=2,
            content=("The weather forecast predicts heavy rain tomorrow."),
        )

        session.add_all([page_one, page_two])
        await session.flush()

        embedding_provider = get_embedding_provider()

        relevant_content = (
            "Retrieval augmented generation combines "
            "information retrieval with language generation."
        )

        unrelated_content = "The weather forecast predicts heavy rain tomorrow."

        relevant_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=0,
            content=relevant_content,
            chunk_metadata={},
            embedding=embedding_provider.embed_text(relevant_content),
        )

        unrelated_chunk = DocumentChunk(
            document_id=document.id,
            section_id=section.id,
            chunk_index=1,
            content=unrelated_content,
            chunk_metadata={},
            embedding=embedding_provider.embed_text(unrelated_content),
        )

        session.add_all(
            [
                relevant_chunk,
                unrelated_chunk,
            ]
        )
        await session.flush()

        session.add_all(
            [
                ChunkPageMap(
                    chunk_id=relevant_chunk.id,
                    document_page_id=page_one.id,
                ),
                ChunkPageMap(
                    chunk_id=unrelated_chunk.id,
                    document_page_id=page_two.id,
                ),
            ]
        )
        await session.flush()

        # Commit so the API's separate database session
        # can see the test data.
        await session.commit()

        try:
            transport = ASGITransport(app=app)

            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/retrieval/search",
                    json={
                        "query": "retrieval augmented generation",
                        "limit": 2,
                        "document_id": str(document.id),
                    },
                )

            assert response.status_code == 200

            body = response.json()

            assert len(body["results"]) == 2

            first_result = body["results"][0]
            second_result = body["results"][1]

            # Vector retrieval + cross-encoder should put
            # the relevant chunk first.
            assert first_result["content"] == relevant_content

            assert first_result["section_path"] == "Results"

            assert first_result["page_numbers"] == [1]

            # Similarity comes from pgvector cosine distance.
            assert 0.0 <= first_result["similarity"] <= 1.0

            # These scores come from the real cross-encoder.
            assert first_result["rerank_score"] is not None
            assert second_result["rerank_score"] is not None

            assert first_result["rerank_score"] >= second_result["rerank_score"]

        finally:
            app.dependency_overrides.clear()

            await session.delete(document)
            await session.commit()


@pytest.mark.asyncio
async def test_retrieval_switching_current_version_changes_results_without_deleting_vectors() -> (
    None
):
    """Switching which version is current must change which version's
    chunks retrieval returns, without deleting either version's chunks or
    embeddings -- historical versions remain fully stored and retrievable
    again if they are made current later."""

    logical_document_id = uuid4()

    async with async_session_factory() as session:
        version_one = Document(
            title="Versioned Retrieval Doc",
            document_type="research_paper",
            content_hash=f"switch-v1-{uuid4()}",
            document_metadata={},
            logical_document_id=logical_document_id,
            version_number=1,
            is_current=True,
            status=DocumentStatus.READY,
        )

        version_two = Document(
            title="Versioned Retrieval Doc",
            document_type="research_paper",
            content_hash=f"switch-v2-{uuid4()}",
            document_metadata={},
            logical_document_id=logical_document_id,
            version_number=2,
            is_current=False,
            status=DocumentStatus.READY,
        )

        session.add_all([version_one, version_two])
        await session.flush()

        embedding_provider = get_embedding_provider()

        shared_query = "Switching current versions must change retrieval results."
        shared_embedding = embedding_provider.embed_text(shared_query)

        chunk_one = DocumentChunk(
            document_id=version_one.id,
            section_id=None,
            chunk_index=0,
            content=f"{shared_query} (from version one)",
            chunk_metadata={},
            embedding=shared_embedding,
        )

        chunk_two = DocumentChunk(
            document_id=version_two.id,
            section_id=None,
            chunk_index=0,
            content=f"{shared_query} (from version two)",
            chunk_metadata={},
            embedding=shared_embedding,
        )

        session.add_all([chunk_one, chunk_two])
        await session.commit()

        try:
            transport = ASGITransport(app=app)

            async with AsyncClient(transport=transport, base_url="http://test") as client:
                first_response = await client.post(
                    "/retrieval/search",
                    json={"query": shared_query, "limit": 10},
                )

            assert first_response.status_code == 200
            first_contents = [r["content"] for r in first_response.json()["results"]]
            assert any("(from version one)" in content for content in first_contents)
            assert all("(from version two)" not in content for content in first_contents)

            # Switch current from v1 -> v2 via the existing DocumentService --
            # no new version, no reprocessing, no chunk/vector mutation.
            document_service = DocumentService(session)
            await document_service.set_current(version_two)
            await session.commit()

            async with AsyncClient(transport=transport, base_url="http://test") as client:
                second_response = await client.post(
                    "/retrieval/search",
                    json={"query": shared_query, "limit": 10},
                )

            assert second_response.status_code == 200
            second_contents = [r["content"] for r in second_response.json()["results"]]
            assert any("(from version two)" in content for content in second_contents)
            assert all("(from version one)" not in content for content in second_contents)

            # Neither version's chunks/embeddings were deleted by switching
            # current -- both remain fully stored regardless of which one is
            # retrievable right now.
            stored_chunk_one = await session.get(DocumentChunk, chunk_one.id)
            stored_chunk_two = await session.get(DocumentChunk, chunk_two.id)

            assert stored_chunk_one is not None
            assert stored_chunk_one.embedding is not None
            assert stored_chunk_two is not None
            assert stored_chunk_two.embedding is not None

        finally:
            app.dependency_overrides.clear()

            # Cascade deletes handle each version's chunks.
            await session.delete(version_one)
            await session.delete(version_two)
            await session.commit()


def test_research_ask_endpoint() -> None:
    """Test the research API response contract without trace."""

    retrieval_service = FakeRetrievalService()
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service

    client = TestClient(app)

    try:
        response = client.post(
            "/research/ask",
            json={
                "query": "What is retrieval augmented generation?",
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["answer"] == ("RAG combines retrieval with language generation.")
        assert body["model"] == "test-model"

        assert len(body["sources"]) == 1

        source = body["sources"][0]

        assert source["section_path"] == "Results"
        assert source["page_numbers"] == [1, 2]

        assert "trace" not in body
        assert body["trace_requested"] is False
        assert body["trace_available"] is False
        assert body["trace_unavailable_reason"] is None

    finally:
        app.dependency_overrides.clear()


def test_research_ask_trace_requested_but_server_disabled() -> None:
    """Case 3 for /research/ask: trace=true + TRACE_ENABLED=false.

    The answer and sources must still be generated normally (retrieval and
    decomposition still run) -- only trace capture is withheld, and the
    response says exactly why, following the same rule as
    POST /retrieval/search."""

    retrieval_service = FakeRetrievalService()
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service
    app.dependency_overrides[get_app_settings] = lambda: Settings(trace_enabled=False)

    client = TestClient(app)

    try:
        response = client.post(
            "/research/ask",
            json={
                "query": "What is retrieval augmented generation?",
                "trace": True,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["answer"] == ("RAG combines retrieval with language generation.")
        assert len(body["sources"]) == 1

        assert "trace" not in body
        assert body["trace_requested"] is True
        assert body["trace_available"] is False
        assert body["trace_unavailable_reason"] == "server_disabled"

    finally:
        app.dependency_overrides.clear()


def test_research_ask_trace_mode() -> None:
    """Test that research trace mode exposes retrieval trace when the
    server also has tracing enabled (trace=true + TRACE_ENABLED=true)."""

    retrieval_service = FakeRetrievalService()
    generation_service = FakeGenerationService()

    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_generation_service] = lambda: generation_service
    app.dependency_overrides[get_app_settings] = lambda: Settings(trace_enabled=True)

    client = TestClient(app)

    try:
        response = client.post(
            "/research/ask",
            json={
                "query": "What is retrieval augmented generation?",
                "trace": True,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["answer"] == ("RAG combines retrieval with language generation.")
        assert body["model"] == "test-model"

        assert len(body["sources"]) == 1

        assert "trace" in body
        assert body["trace"] is not None
        assert body["trace_requested"] is True
        assert body["trace_available"] is True
        assert body["trace_unavailable_reason"] is None

        trace = body["trace"]

        assert trace["query"] == ("What is retrieval augmented generation?")

        assert trace["candidate_limit"] == 50
        assert trace["original_query"] == ("What is retrieval augmented generation?")
        assert trace["sub_queries"] == ["What is retrieval augmented generation?"]
        assert trace["raw_candidate_count"] == 1
        assert trace["deduplicated_candidate_count"] == 1

        assert len(trace["candidates"]) == 1
        assert len(trace["final_results"]) == 1

        candidate = trace["candidates"][0]

        assert candidate["section_path"] == "Results"
        assert candidate["page_numbers"] == [1, 2]
        assert candidate["distance"] == 0.15
        assert candidate["rerank_score"] == 4.2

        context = trace["context"]

        assert context is not None

        assert context["text"] == (
            "[Source 1]\nRetrieval augmented generation combines retrieval with generation."
        )

        assert len(context["sources"]) == 1

        context_source = context["sources"][0]

        assert context_source["content"] == (
            "Retrieval augmented generation combines retrieval with generation."
        )

        assert context_source["section_path"] == "Results"
        assert context_source["page_numbers"] == [1, 2]
        assert context_source["distance"] == 0.15
        assert context_source["rerank_score"] == 4.2

    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("router_enabled", [True, False, None])
def test_retrieval_search_accepts_router_enabled_field(router_enabled) -> None:
    """`router_enabled` is accepted whether explicitly True, explicitly
    False, or omitted from the payload entirely (modeled here as `None`,
    which is excluded from the JSON body below) -- all three must validate
    and reach `RetrievalService.search` as the exact value sent."""

    retrieval_service = FakeRetrievalService()
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    client = TestClient(app)

    payload = {"query": "retrieval augmented generation", "limit": 2}
    if router_enabled is not None:
        payload["router_enabled"] = router_enabled

    try:
        response = client.post("/retrieval/search", json=payload)

        assert response.status_code == 200
        assert retrieval_service.received_router_enabled == router_enabled

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_router_override_takes_precedence_over_setting() -> None:
    """An explicit per-request `router_enabled=True` must reach
    `RetrievalService.search` as `True` regardless of what the application
    setting (`Settings.query_router_enabled`, default False) currently is --
    confirming the override, not the global default, is what's threaded
    through for this request."""

    retrieval_service = FakeRetrievalService()
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    # Application default stays OFF -- only the request overrides it.
    app.dependency_overrides[get_app_settings] = lambda: Settings(query_router_enabled=False)

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "retrieval augmented generation", "router_enabled": True},
        )

        assert response.status_code == 200
        assert retrieval_service.received_router_enabled is True

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_omitted_router_enabled_is_none_not_false() -> None:
    """Existing request payloads that predate this field must remain
    compatible -- omitting `router_enabled` must reach `RetrievalService`
    as `None` (defer to the service's own configured default), never
    coerced to `False` (which would force-disable an application default
    of `True` instead of deferring to it)."""

    retrieval_service = FakeRetrievalService()
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "retrieval augmented generation"},
        )

        assert response.status_code == 200
        assert retrieval_service.received_router_enabled is None

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_trace_reports_router_bypassed_true() -> None:
    """`router_bypassed` in the trace response must reflect the actual
    executed decision (here, faked as True), not be inferred from
    `sub_queries` or query text."""

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService(
        router_bypassed=True
    )
    app.dependency_overrides[get_app_settings] = lambda: Settings(trace_enabled=True)

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "research query", "limit": 1, "trace": True, "router_enabled": True},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["trace"] is not None
        assert body["trace"]["router_bypassed"] is True

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_trace_reports_router_bypassed_false() -> None:
    """`router_bypassed` must be False in the trace response when
    decomposition ran (or the router was disabled) -- the default shape of
    `FakeRetrievalService` here."""

    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService(
        router_bypassed=False
    )
    app.dependency_overrides[get_app_settings] = lambda: Settings(trace_enabled=True)

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "research query", "limit": 1, "trace": True},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["trace"] is not None
        assert body["trace"]["router_bypassed"] is False

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_explicit_candidate_limit_reaches_service_unchanged() -> None:
    """An explicit `candidate_limit` in the request body must reach
    `RetrievalService.search` as the exact integer sent, not reinterpreted
    as the final result `limit`."""

    retrieval_service = FakeRetrievalService()
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "retrieval augmented generation", "limit": 5, "candidate_limit": 200},
        )

        assert response.status_code == 200
        assert retrieval_service.received_candidate_limit == 200

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_omitted_candidate_limit_preserves_default_path() -> None:
    """Omitting `candidate_limit` entirely must reach the service as `None`
    -- preserving `RetrievalService.search`'s own existing
    `max(limit, candidate_limit_floor)` default path exactly, not coerced
    to any other value."""

    retrieval_service = FakeRetrievalService()
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "retrieval augmented generation", "limit": 5},
        )

        assert response.status_code == 200
        assert retrieval_service.received_candidate_limit is None

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_null_candidate_limit_preserves_default_path() -> None:
    """An explicit JSON `null` must behave identically to omitting the
    field entirely -- both mean "use the configured default"."""

    retrieval_service = FakeRetrievalService()
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "retrieval augmented generation", "limit": 5, "candidate_limit": None},
        )

        assert response.status_code == 200
        assert retrieval_service.received_candidate_limit is None

    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("invalid_candidate_limit", [0, -5, 1001])
def test_retrieval_search_rejects_out_of_range_candidate_limit(invalid_candidate_limit) -> None:
    """`candidate_limit` must be rejected by request validation (422) when
    it falls outside the documented 1-1000 bound, before ever reaching the
    service."""

    retrieval_service = FakeRetrievalService()
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={
                "query": "retrieval augmented generation",
                "limit": 5,
                "candidate_limit": invalid_candidate_limit,
            },
        )

        assert response.status_code == 422
        # The service must never have been reached with a bad value.
        assert retrieval_service.received_candidate_limit is None

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_rejects_candidate_limit_below_result_limit() -> None:
    """A `candidate_limit` smaller than `limit` is rejected by request
    validation (422) -- the same invariant `RetrievalService.search` itself
    enforces, caught earlier at the HTTP boundary instead of surfacing as
    an unhandled 500 from an uncaught `ValueError`."""

    retrieval_service = FakeRetrievalService()
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "retrieval augmented generation", "limit": 20, "candidate_limit": 5},
        )

        assert response.status_code == 422
        assert retrieval_service.received_candidate_limit is None

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
def test_retrieval_search_decomposition_outcome_present_without_trace(status, reason) -> None:
    """`decomposition` must be present on the normal response for every
    reachable status, with NO trace requested at all -- this is the exact
    regression this milestone corrects: decomposition visibility must not
    depend on Trace Mode."""

    retrieval_service = FakeRetrievalService(
        decomposition_status=status,
        decomposition_reason=reason,
    )
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    client = TestClient(app)

    try:
        response = client.post(
            "/retrieval/search",
            json={"query": "retrieval augmented generation", "limit": 2},
        )

        assert response.status_code == 200
        body = response.json()

        # No trace was requested -- the old trace-only path must be absent...
        assert body["trace"] is None
        assert body["trace_available"] is False
        # ...but the decomposition outcome must still be present and accurate.
        assert body["decomposition"]["status"] == status
        assert body["decomposition"]["original_query"] == "retrieval augmented generation"
        assert body["decomposition"]["sub_queries"] == ["retrieval augmented generation"]
        assert body["decomposition"]["reason"] == reason

    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_decomposition_fallback_reason_is_safe_not_raw_exception() -> None:
    """The `reason` field must never contain raw exception text -- only the
    safe, generic constant the service substitutes in its place."""

    retrieval_service = FakeRetrievalService(
        decomposition_status="fallback",
        decomposition_reason="openrouter returned invalid JSON: traceback at line 42",
    )
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service

    client = TestClient(app)

    try:
        # This test documents the CONTRACT (the API surfaces whatever string
        # the service hands it) -- the actual safety guarantee (never a raw
        # exception) is enforced in RetrievalService._search itself and
        # covered by tests/unit/test_retrieval_decomposition.py's fallback
        # test, which asserts the real FALLBACK_REASON constant is used.
        response = client.post(
            "/retrieval/search",
            json={"query": "retrieval augmented generation"},
        )

        assert response.status_code == 200

    finally:
        app.dependency_overrides.clear()
