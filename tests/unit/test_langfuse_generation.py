import pytest

from app.generation.context import ContextAssembler
from app.generation.errors import ProviderErrorCategory, ProviderUnavailableError
from app.generation.generation_service import GenerationService
from app.generation.service import GenerationResult

RAW_MARKER = "SYNTHETIC_TEST_RAW_EXCEPTION_MARKER"


class FakeGenerationProvider:
    def __init__(self) -> None:
        self.received_query = None
        self.received_context = None

    async def generate(
        self,
        query: str,
        context,
    ) -> GenerationResult:
        self.received_query = query
        self.received_context = context

        return GenerationResult(
            text="Generated answer.",
            model="fake-model",
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
        )


class FailingGenerationProvider:
    """Raises a caller-supplied `ProviderUnavailableError`, optionally
    exposing a `.model` attribute like the real `OpenRouterGenerationProvider`
    does."""

    def __init__(self, error: Exception, model: str | None = "configured-model") -> None:
        self.error = error
        if model is not None:
            self.model = model

    async def generate(self, query: str, context) -> GenerationResult:
        raise self.error


class FakeObservation:
    def __init__(
        self,
        name: str,
        observation_type: str,
        input_data: dict,
    ) -> None:
        self.name = name
        self.observation_type = observation_type
        self.input_data = input_data
        self.output = None
        self.level = None
        self.status_message = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def update(
        self,
        *,
        output=None,
        level=None,
        status_message=None,
    ) -> None:
        if output is not None:
            self.output = output

        if level is not None:
            self.level = level

        if status_message is not None:
            self.status_message = status_message


class FakeLangfuse:
    def __init__(self) -> None:
        self.observations = []

    def start_as_current_observation(
        self,
        *,
        as_type,
        name,
        input,
    ):
        observation = FakeObservation(
            name=name,
            observation_type=as_type,
            input_data=input,
        )

        self.observations.append(observation)

        return observation


@pytest.mark.asyncio
async def test_generation_creates_langfuse_observation(
    monkeypatch,
) -> None:
    fake_langfuse = FakeLangfuse()
    provider = FakeGenerationProvider()

    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = GenerationService(provider)

    context = ContextAssembler().assemble(
        [],
    )

    result = await service.generate(
        query="What is RAG?",
        context=context,
    )

    assert result.text == "Generated answer."
    assert result.model == "fake-model"

    assert len(fake_langfuse.observations) == 1

    observation = fake_langfuse.observations[0]

    assert observation.name == "answer-generation"
    assert observation.observation_type == "generation"

    # Raw context is never sent to Langfuse -- only safe, non-content
    # metadata (see app.generation.generation_service.GenerationService).
    assert observation.input_data == {
        "query": "What is RAG?",
        "context_length": 0,
        "source_count": 0,
        "redacted_source_count": 0,
    }

    assert observation.output == {
        "text": "Generated answer.",
        "answer_redacted": False,
        "model": "fake-model",
        "finish_reason": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
        },
    }


@pytest.mark.asyncio
async def test_generation_without_langfuse_still_works(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: None,
    )

    provider = FakeGenerationProvider()
    service = GenerationService(provider)

    context = ContextAssembler().assemble([])

    result = await service.generate(
        query="What is RAG?",
        context=context,
    )

    assert result.text == "Generated answer."
    assert result.model == "fake-model"

    assert provider.received_query == "What is RAG?"
    assert provider.received_context is context


@pytest.mark.asyncio
async def test_successful_generation_records_truncation_finish_reason(
    monkeypatch,
) -> None:
    """Regression for the incomplete/truncated-answer bug: when a provider's
    response was cut off by its token budget (`finish_reason == "length"`),
    that must be visible in the Langfuse observation -- otherwise a
    truncated answer is indistinguishable from a complete one in
    diagnostics."""
    fake_langfuse = FakeLangfuse()

    class TruncatedGenerationProvider:
        async def generate(self, query: str, context) -> GenerationResult:
            return GenerationResult(
                text="This answer was cut off mid-sent",
                model="fake-model",
                output_tokens=4096,
                finish_reason="length",
            )

    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = GenerationService(TruncatedGenerationProvider())
    context = ContextAssembler().assemble([])

    await service.generate(query="What is RAG?", context=context)

    observation = fake_langfuse.observations[0]
    assert observation.output["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_failed_generation_records_safe_category_model_and_http_metadata(
    monkeypatch,
) -> None:
    """Root-cause regression: the UI's observed `authentication_or_configuration_error`
    could not previously be diagnosed from Langfuse because the failed
    observation recorded nothing but the category string. This confirms the
    configured model, safe HTTP status, and safe provider error code are now
    captured too."""
    fake_langfuse = FakeLangfuse()
    error = ProviderUnavailableError(
        ProviderErrorCategory.AUTHENTICATION_OR_CONFIG,
        http_status=401,
        provider_error_code="invalid_api_key",
        provider_error_type="authentication_error",
        sanitized_provider_message="Invalid API key provided.",
    )
    provider = FailingGenerationProvider(error, model="openrouter/free")

    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = GenerationService(provider)
    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError):
        await service.generate(query="What is RAG?", context=context)

    observation = fake_langfuse.observations[0]
    assert observation.level == "ERROR"
    assert observation.status_message == "authentication_or_configuration_error"
    assert observation.output == {
        "category": "authentication_or_configuration_error",
        "retryable": False,
        "http_status": 401,
        "provider_error_code": "invalid_api_key",
        "provider_error_type": "authentication_error",
        "provider_message": "Invalid API key provided.",
        "model": "openrouter/free",
        "request_parameters": None,
    }


@pytest.mark.asyncio
async def test_failed_generation_missing_http_status_and_code_are_none_not_fabricated(
    monkeypatch,
) -> None:
    """A category with no real HTTP status (e.g. a timeout, which never
    reaches an HTTP response at all) must record `None`, never a guessed
    value -- this is the exact behavior `wrap_openai_exception` already
    guarantees; this test pins it at the observability layer too."""
    fake_langfuse = FakeLangfuse()
    error = ProviderUnavailableError(ProviderErrorCategory.TIMEOUT_OR_CONNECTIVITY)
    provider = FailingGenerationProvider(error)

    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = GenerationService(provider)
    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError):
        await service.generate(query="What is RAG?", context=context)

    observation = fake_langfuse.observations[0]
    assert observation.output["http_status"] is None
    assert observation.output["provider_error_code"] is None


@pytest.mark.asyncio
async def test_failed_generation_provider_without_model_attribute_records_none(
    monkeypatch,
) -> None:
    """`GenerationProvider` is a `Protocol` with no guaranteed `.model`
    attribute -- a provider implementation without one must record `None`,
    not raise `AttributeError` and not fabricate a model name."""
    fake_langfuse = FakeLangfuse()
    error = ProviderUnavailableError(ProviderErrorCategory.RATE_LIMITED)
    provider = FailingGenerationProvider(error, model=None)

    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = GenerationService(provider)
    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError):
        await service.generate(query="What is RAG?", context=context)

    observation = fake_langfuse.observations[0]
    assert observation.output["model"] is None


@pytest.mark.asyncio
async def test_failed_generation_never_records_raw_exception_text(
    monkeypatch,
) -> None:
    """The raw exception message/args must never reach Langfuse -- only the
    fixed, category-keyed safe fields already asserted in the other tests
    here."""
    fake_langfuse = FakeLangfuse()
    error = ProviderUnavailableError(ProviderErrorCategory.INVALID_REQUEST)
    # Simulate a raw provider body being attached the way a real caught
    # openai exception might carry one, to prove it is never read/forwarded.
    error.args = (RAW_MARKER,)
    provider = FailingGenerationProvider(error)

    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = GenerationService(provider)
    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError):
        await service.generate(query="What is RAG?", context=context)

    observation = fake_langfuse.observations[0]
    assert RAW_MARKER not in str(observation.output)
    assert RAW_MARKER not in str(observation.status_message)
    assert RAW_MARKER not in str(observation.input_data)


@pytest.mark.asyncio
async def test_unclassified_exception_records_only_class_name_and_model(
    monkeypatch,
) -> None:
    """An exception that never went through `wrap_openai_exception` (i.e.
    not a `ProviderUnavailableError`) must still never leak its message --
    only its class name and the configured model, exactly as before this
    change, now also structured into `output`."""
    fake_langfuse = FakeLangfuse()
    provider = FailingGenerationProvider(ValueError(RAW_MARKER), model="openrouter/free")

    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = GenerationService(provider)
    context = ContextAssembler().assemble([])

    with pytest.raises(ValueError):
        await service.generate(query="What is RAG?", context=context)

    observation = fake_langfuse.observations[0]
    assert observation.status_message == "ValueError"
    assert observation.output == {"category": "ValueError", "model": "openrouter/free"}
    assert RAW_MARKER not in str(observation.output)
    assert RAW_MARKER not in str(observation.status_message)


@pytest.mark.asyncio
async def test_guardrail_rejection_still_records_its_own_category(
    monkeypatch,
) -> None:
    """Regression: the existing guardrail-rejection category must continue
    to be reported accurately and must not be confused with
    authentication/configuration -- the new metadata fields are additive,
    never a reclassification."""
    fake_langfuse = FakeLangfuse()
    error = ProviderUnavailableError(
        ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION,
        http_status=403,
        provider_error_code="content_policy_violation",
    )
    provider = FailingGenerationProvider(error)

    monkeypatch.setattr(
        "app.generation.generation_service.get_langfuse",
        lambda: fake_langfuse,
    )

    service = GenerationService(provider)
    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError):
        await service.generate(query="What is RAG?", context=context)

    observation = fake_langfuse.observations[0]
    assert observation.output["category"] == "guardrail_or_policy_rejection"
    assert observation.output["retryable"] is False
    assert observation.output["http_status"] == 403
    assert observation.output["provider_error_code"] == "content_policy_violation"
