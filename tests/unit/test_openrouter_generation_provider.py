"""Unit tests for `OpenRouterGenerationProvider`'s request construction --
mirrors the conventions in tests/unit/test_openrouter_decomposition.py
(mocked `client.chat.completions.create`, no network calls). The live,
network-dependent check lives separately in
tests/integration/test_openrouter_generation.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

from app.generation.context import ContextAssembler
from app.generation.errors import ProviderErrorCategory, ProviderUnavailableError
from app.generation.openrouter import OpenRouterGenerationProvider


def make_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
        model="test-model",
        usage=None,
    )


def make_reasoning_response(reasoning: str | None) -> SimpleNamespace:
    """A response shaped like a reasoning-style OpenRouter model cut off
    before finishing: `content` is `None`, and the actual text lives in a
    non-standard `reasoning` field -- verified directly against a live
    free-tier model (see app.generation.openrouter's handling comment)."""

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, reasoning=reasoning),
            )
        ],
        model="test-reasoning-model",
        usage=None,
    )


@pytest.fixture
def provider() -> OpenRouterGenerationProvider:
    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
    )
    provider.client.chat.completions.create = AsyncMock()
    return provider


@pytest.mark.asyncio
async def test_sends_configured_model_temperature_and_top_p(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["temperature"] == 0.2
    assert call.kwargs["top_p"] == 1.0


@pytest.mark.asyncio
async def test_system_prompt_requests_evidence_driven_structure_not_fixed_template(
    provider,
) -> None:
    """The system prompt must guide the model toward structuring its answer
    around the actual question/evidence (optional, evidence-gated sections)
    rather than hardcoding a fixed set of sections into every answer, and
    must forbid overclaiming how many sources were actually reviewed."""

    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    system_message = call.kwargs["messages"][0]
    assert system_message["role"] == "system"
    content = system_message["content"]

    # Structure follows the question/evidence -- not a mandatory template.
    assert "rather than a fixed template" in content
    assert "Only include a section if the context actually supports it" in content

    # Anti-overclaiming instruction (never claim more documents/sources
    # were reviewed than were actually provided).
    assert "Never claim to have reviewed, summarized, or drawn" in content

    # Still forbids inventing unsupported facts.
    assert "never invent facts" in content.lower()


@pytest.mark.asyncio
async def test_omits_optional_fields_when_not_configured(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    assert "max_tokens" not in call.kwargs
    assert "response_format" not in call.kwargs
    assert "reasoning_effort" not in call.kwargs


@pytest.mark.asyncio
async def test_includes_optional_fields_when_configured() -> None:
    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
        response_format={"type": "json_object"},
        reasoning_effort="medium",
    )
    provider.client.chat.completions.create = AsyncMock(return_value=make_response("An answer."))

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["temperature"] == 0.7
    assert call.kwargs["top_p"] == 0.9
    assert call.kwargs["max_tokens"] == 256
    assert call.kwargs["response_format"] == {"type": "json_object"}
    assert call.kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_raises_on_empty_response(provider) -> None:
    provider.client.chat.completions.create.return_value = make_response(None)

    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.generate(query="What is RAG?", context=context)

    assert exc_info.value.category == ProviderErrorCategory.INVALID_PROVIDER_RESPONSE


@pytest.mark.asyncio
async def test_never_substitutes_reasoning_field_as_the_user_facing_answer(provider) -> None:
    """A reasoning-style model can leave `content: null` and populate a
    non-standard `reasoning`/chain-of-thought field instead (e.g. when cut
    off by the token budget before finishing). That field is internal
    deliberation, not a provider-documented user-facing output, and must
    NEVER be surfaced as if it were the model's actual answer -- the
    existing safe error/fallback behavior must be used instead."""

    provider.client.chat.completions.create.return_value = make_reasoning_response(
        "Internal chain-of-thought: the user wants to know about RAG, so I should..."
    )

    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.generate(query="What is RAG?", context=context)

    assert exc_info.value.category == ProviderErrorCategory.INVALID_PROVIDER_RESPONSE
    # The internal reasoning text must never leak into the safe exception
    # either -- only the fixed, generic message.
    assert "chain-of-thought" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_raises_on_null_content_regardless_of_reasoning_field_presence(provider) -> None:
    provider.client.chat.completions.create.return_value = make_reasoning_response(None)

    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.generate(query="What is RAG?", context=context)

    assert exc_info.value.category == ProviderErrorCategory.INVALID_PROVIDER_RESPONSE


@pytest.mark.asyncio
async def test_omits_top_k_frequency_and_presence_penalty_when_not_configured(
    provider,
) -> None:
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    assert "extra_body" not in call.kwargs
    assert "frequency_penalty" not in call.kwargs
    assert "presence_penalty" not in call.kwargs


@pytest.mark.asyncio
async def test_sends_top_k_via_extra_body_when_configured() -> None:
    """top_k is NOT a standard OpenAI field -- it must travel via
    extra_body, OpenRouter's passthrough mechanism, never as a direct
    top-level kwarg the openai SDK itself would reject."""

    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        top_k=40,
    )
    provider.client.chat.completions.create = AsyncMock(return_value=make_response("An answer."))

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["extra_body"] == {"top_k": 40}
    assert "top_k" not in call.kwargs


@pytest.mark.asyncio
async def test_sends_frequency_and_presence_penalty_when_configured() -> None:
    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        frequency_penalty=0.5,
        presence_penalty=-0.5,
    )
    provider.client.chat.completions.create = AsyncMock(return_value=make_response("An answer."))

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    assert call.kwargs["frequency_penalty"] == 0.5
    assert call.kwargs["presence_penalty"] == -0.5


def test_timeout_and_max_retries_propagate_into_client_construction() -> None:
    """Verifies `timeout`/`max_retries` reach the underlying `AsyncOpenAI`
    client construction -- not swallowed or ignored."""

    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        timeout=30.0,
        max_retries=5,
    )

    assert provider.client.timeout == 30.0
    assert provider.client.max_retries == 5


def test_unconfigured_timeout_leaves_sdk_default_untouched() -> None:
    """An unconfigured (`None`) timeout must not become an explicit
    `timeout=None` override -- the SDK's own library-internal default must
    apply exactly as if `timeout` had never been mentioned at all."""

    default_provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
    )

    baseline_client = AsyncOpenAI(api_key="test-key")

    assert default_provider.client.timeout == baseline_client.timeout


@pytest.mark.asyncio
async def test_plain_bad_request_is_classified_as_invalid_request_not_guardrail(provider) -> None:
    """A synthetic 400 BadRequestError with no moderation signal in its body
    must never leak its raw body text, AND must not be mislabeled as a
    guardrail/policy rejection -- a plain 400 is far more often a malformed
    request than a content decision."""

    import httpx
    from openai import BadRequestError

    raw_body_marker = "SYNTHETIC_TEST_RAW_BODY_MARKER_12345"
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        status_code=400,
        request=request,
        json={"error": {"message": raw_body_marker}},
    )
    provider.client.chat.completions.create.side_effect = BadRequestError(
        message=raw_body_marker,
        response=response,
        body={"error": {"message": raw_body_marker}},
    )

    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.generate(query="What is RAG?", context=context)

    assert exc_info.value.category == ProviderErrorCategory.INVALID_REQUEST
    assert raw_body_marker not in str(exc_info.value)
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_bad_request_with_moderation_signal_is_classified_as_guardrail_rejection(
    provider,
) -> None:
    """A 400 that DOES carry a recognizable moderation/guardrail signal in
    its error body is correctly classified as a policy rejection."""

    import httpx
    from openai import BadRequestError

    raw_body_marker = "SYNTHETIC_TEST_RAW_BODY_MARKER_67890"
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    body = {"error": {"message": raw_body_marker, "code": "content_policy_violation"}}
    response = httpx.Response(status_code=400, request=request, json=body)
    provider.client.chat.completions.create.side_effect = BadRequestError(
        message=raw_body_marker,
        response=response,
        body=body,
    )

    context = ContextAssembler().assemble([])

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.generate(query="What is RAG?", context=context)

    assert exc_info.value.category == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION
    assert raw_body_marker not in str(exc_info.value)
    assert exc_info.value.retryable is False


def test_default_max_retries_matches_sdk_default() -> None:
    """Confirms the Settings default (2) is identical to the openai SDK's
    own default, so leaving this unconfigured changes nothing."""

    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
    )

    assert provider.client.max_retries == 2
