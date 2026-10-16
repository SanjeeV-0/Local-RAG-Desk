from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

from app.generation.errors import ProviderErrorCategory, ProviderUnavailableError
from app.retrieval.openrouter_decomposition import (
    OpenRouterQueryDecompositionProvider,
)


def make_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


@pytest.fixture
def provider() -> OpenRouterQueryDecompositionProvider:
    provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
    )
    provider.client.chat.completions.create = AsyncMock()
    return provider


@pytest.mark.asyncio
async def test_returns_queries_from_valid_response(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": ["What is RAG?", "How does reranking work?"]}'
    )

    result = await provider.decompose("What is RAG and how does reranking work?")

    assert result == [
        "What is RAG?",
        "How does reranking work?",
    ]

    provider.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_sends_configured_model_temperature_and_top_p(provider) -> None:
    """temperature is no longer hard-coded to 0 -- it (and model/top_p) must
    come straight from the provider's configured values."""

    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": ["What is RAG?"]}'
    )

    await provider.decompose("What is RAG?")

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["temperature"] == 0.0
    assert call.kwargs["top_p"] == 1.0


@pytest.mark.asyncio
async def test_omits_optional_fields_when_not_configured(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": ["What is RAG?"]}'
    )

    await provider.decompose("What is RAG?")

    call = provider.client.chat.completions.create.await_args
    assert "max_tokens" not in call.kwargs
    assert "response_format" not in call.kwargs
    assert "reasoning_effort" not in call.kwargs


@pytest.mark.asyncio
async def test_includes_optional_fields_when_configured() -> None:
    provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.5,
        max_tokens=128,
        top_p=0.8,
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )
    provider.client.chat.completions.create = AsyncMock(
        return_value=make_response('{"queries": ["What is RAG?"]}')
    )

    await provider.decompose("What is RAG?")

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["temperature"] == 0.5
    assert call.kwargs["top_p"] == 0.8
    assert call.kwargs["max_tokens"] == 128
    assert call.kwargs["response_format"] == {"type": "json_object"}
    assert call.kwargs["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_rejects_invalid_json(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response("not valid json")

    with pytest.raises(ValueError, match="invalid JSON"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_non_object_response(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response('["What is RAG?"]')

    with pytest.raises(ValueError, match="JSON object"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_missing_queries(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"result": ["What is RAG?"]}'
    )

    with pytest.raises(ValueError, match="queries"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_non_list_queries(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": "What is RAG?"}'
    )

    with pytest.raises(ValueError, match="queries"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_non_string_queries(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": ["What is RAG?", 123]}'
    )

    with pytest.raises(ValueError, match="strings"):
        await provider.decompose("What is RAG?")


@pytest.mark.asyncio
async def test_rejects_empty_response(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(None)

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.decompose("What is RAG?")

    assert exc_info.value.category == ProviderErrorCategory.INVALID_PROVIDER_RESPONSE


@pytest.mark.asyncio
async def test_omits_top_k_frequency_and_presence_penalty_when_not_configured(
    provider,
) -> None:
    provider.client.chat.completions.create.return_value = make_response(
        '{"queries": ["What is RAG?"]}'
    )

    await provider.decompose("What is RAG?")

    call = provider.client.chat.completions.create.await_args
    assert "extra_body" not in call.kwargs
    assert "frequency_penalty" not in call.kwargs
    assert "presence_penalty" not in call.kwargs


@pytest.mark.asyncio
async def test_sends_top_k_via_extra_body_when_configured() -> None:
    """top_k is NOT a standard OpenAI field -- it must travel via
    extra_body, OpenRouter's passthrough mechanism, never as a direct
    top-level kwarg the openai SDK itself would reject."""

    provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        top_k=10,
    )
    provider.client.chat.completions.create = AsyncMock(
        return_value=make_response('{"queries": ["What is RAG?"]}')
    )

    await provider.decompose("What is RAG?")

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["extra_body"] == {"top_k": 10}
    assert "top_k" not in call.kwargs


@pytest.mark.asyncio
async def test_sends_frequency_and_presence_penalty_when_configured() -> None:
    provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        frequency_penalty=1.0,
        presence_penalty=-1.0,
    )
    provider.client.chat.completions.create = AsyncMock(
        return_value=make_response('{"queries": ["What is RAG?"]}')
    )

    await provider.decompose("What is RAG?")

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["frequency_penalty"] == 1.0
    assert call.kwargs["presence_penalty"] == -1.0


def test_timeout_and_max_retries_propagate_into_client_construction() -> None:
    provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        timeout=15.0,
        max_retries=4,
    )

    assert provider.client.timeout == 15.0
    assert provider.client.max_retries == 4


def test_unconfigured_timeout_leaves_sdk_default_untouched() -> None:
    default_provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
    )

    baseline_client = AsyncOpenAI(api_key="test-key")

    assert default_provider.client.timeout == baseline_client.timeout


def test_default_max_retries_matches_sdk_default() -> None:
    provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
    )

    assert provider.client.max_retries == 2


@pytest.mark.asyncio
async def test_rate_limit_error_is_classified_as_retryable_with_safe_message(provider) -> None:
    import httpx
    from openai import RateLimitError

    raw_body_marker = "SYNTHETIC_TEST_RAW_BODY_MARKER_67890"
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        status_code=429,
        request=request,
        json={"error": {"message": raw_body_marker}},
    )
    provider.client.chat.completions.create.side_effect = RateLimitError(
        message=raw_body_marker,
        response=response,
        body={"error": {"message": raw_body_marker}},
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.decompose("What is RAG?")

    assert exc_info.value.category == ProviderErrorCategory.RATE_LIMITED
    assert exc_info.value.retryable is True
    assert raw_body_marker not in str(exc_info.value)


def test_generation_and_decomposition_settings_do_not_cross_contaminate() -> None:
    """Confirms the two providers are independently constructed/configured
    -- a value set on one provider instance must never appear on an
    unrelated instance of the other provider."""

    from app.generation.openrouter import OpenRouterGenerationProvider

    decomposition_provider = OpenRouterQueryDecompositionProvider(
        api_key="test-key",
        model="decomposition-model",
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        top_k=5,
        frequency_penalty=0.1,
    )

    generation_provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="generation-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        top_k=99,
        presence_penalty=0.9,
    )

    assert decomposition_provider.model == "decomposition-model"
    assert generation_provider.model == "generation-model"
    assert decomposition_provider.top_k == 5
    assert generation_provider.top_k == 99
    assert decomposition_provider.frequency_penalty == 0.1
    assert generation_provider.frequency_penalty is None
    assert decomposition_provider.presence_penalty is None
    assert generation_provider.presence_penalty == 0.9
