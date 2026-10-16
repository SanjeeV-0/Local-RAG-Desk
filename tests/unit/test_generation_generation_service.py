import pytest

from app.generation import generation_service as generation_service_module
from app.generation.context import ContextAssembler
from app.generation.errors import ProviderErrorCategory, ProviderUnavailableError
from app.generation.generation_service import GenerationService
from app.generation.service import GenerationResult
from app.retrieval.models import RetrievedChunk

# Synthetic markers only -- never real PII.
SYNTHETIC_EMAIL = "leaked.marker@example.com"


class FakeGenerationProvider:
    def __init__(self, answer_text: str = "Generated answer.") -> None:
        self.received_query = None
        self.received_context = None
        self.answer_text = answer_text

    async def generate(
        self,
        query: str,
        context,
    ) -> GenerationResult:
        self.received_query = query
        self.received_context = context

        return GenerationResult(
            text=self.answer_text,
            model="fake-model",
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
        )


@pytest.mark.asyncio
async def test_generation_service_delegates_to_provider() -> None:
    provider = FakeGenerationProvider()
    service = GenerationService(provider)

    context = ContextAssembler().assemble([])

    result = await service.generate(
        query="What is retrieval augmented generation?",
        context=context,
    )

    assert result.text == "Generated answer."
    assert result.model == "fake-model"
    assert result.input_tokens == 100
    assert result.output_tokens == 25
    assert result.total_tokens == 125
    assert provider.received_query == "What is retrieval augmented generation?"

    assert provider.received_context is context


@pytest.mark.asyncio
async def test_generation_service_redacts_pii_in_final_answer_without_langfuse() -> None:
    """Final-answer redaction must run even when Langfuse is disabled --
    it is not an observability side effect, it is generation behavior."""

    provider = FakeGenerationProvider(
        answer_text=f"You can reach the author at {SYNTHETIC_EMAIL}."
    )
    service = GenerationService(provider)
    context = ContextAssembler().assemble([])

    result = await service.generate(query="Who wrote this?", context=context)

    assert SYNTHETIC_EMAIL not in result.text
    assert "[REDACTED:EMAIL]" in result.text
    assert result.answer_redacted is True


@pytest.mark.asyncio
async def test_generation_service_clean_answer_reports_not_redacted() -> None:
    provider = FakeGenerationProvider(answer_text="A perfectly safe answer.")
    service = GenerationService(provider)
    context = ContextAssembler().assemble([])

    result = await service.generate(query="What is RAG?", context=context)

    assert result.text == "A perfectly safe answer."
    assert result.answer_redacted is False


class _FakeObservation:
    def __init__(self):
        self.output = None
        self.level = None
        self.status_message = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, *, output=None, level=None, status_message=None):
        self.output = output
        self.level = level
        self.status_message = status_message


class _FakeLangfuse:
    def __init__(self):
        self.input = None
        self.observation = None

    def start_as_current_observation(self, *, as_type, name, input):
        self.input = input
        self.observation = _FakeObservation()
        return self.observation


class _RaisingGenerationProvider:
    async def generate(self, query: str, context):
        raise ProviderUnavailableError(ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION)


def _chunk_with(content: str) -> RetrievedChunk:
    import uuid

    return RetrievedChunk(
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        section_id=uuid.uuid4(),
        section_path="Results",
        page_numbers=[1],
        content=content,
        distance=0.1,
    )


@pytest.mark.asyncio
async def test_generation_service_langfuse_input_never_contains_raw_context_or_query(
    monkeypatch,
) -> None:
    fake_langfuse = _FakeLangfuse()
    monkeypatch.setattr(generation_service_module, "get_langfuse", lambda: fake_langfuse)

    query_marker = "QUERY_MARKER_jdoe@example.com"
    context = ContextAssembler().assemble([_chunk_with("Chunk content with secret text.")])

    provider = FakeGenerationProvider(answer_text="A safe answer.")
    service = GenerationService(provider)

    await service.generate(query=query_marker, context=context)

    assert "context" not in fake_langfuse.input
    assert fake_langfuse.input["context_length"] == len(context.text)
    # The raw query (with its embedded PII-shaped marker) must never appear.
    assert query_marker not in str(fake_langfuse.input)
    assert "jdoe@example.com" not in str(fake_langfuse.input)
    assert "Chunk content with secret text." not in str(fake_langfuse.input)


@pytest.mark.asyncio
async def test_generation_service_langfuse_output_never_contains_raw_answer_pii(
    monkeypatch,
) -> None:
    fake_langfuse = _FakeLangfuse()
    monkeypatch.setattr(generation_service_module, "get_langfuse", lambda: fake_langfuse)

    provider = FakeGenerationProvider(answer_text="Contact jdoe@example.com for help.")
    service = GenerationService(provider)
    context = ContextAssembler().assemble([])

    await service.generate(query="Who do I contact?", context=context)

    assert "jdoe@example.com" not in str(fake_langfuse.observation.output)
    assert fake_langfuse.observation.output["answer_redacted"] is True


@pytest.mark.asyncio
async def test_generation_service_langfuse_error_path_never_contains_raw_exception_text(
    monkeypatch,
) -> None:
    fake_langfuse = _FakeLangfuse()
    monkeypatch.setattr(generation_service_module, "get_langfuse", lambda: fake_langfuse)

    service = GenerationService(_RaisingGenerationProvider())
    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError):
        await service.generate(query="What is RAG?", context=context)

    assert fake_langfuse.observation.level == "ERROR"
    assert fake_langfuse.observation.status_message == "guardrail_or_policy_rejection"
