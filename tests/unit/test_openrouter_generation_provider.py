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
    around the actual question/evidence (optional, evidence-gated content)
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
    assert "Only include something if the context actually supports it" in content

    # Anti-overclaiming instruction (never claim more documents/sources
    # were reviewed than were actually provided).
    assert "Never claim to have reviewed, summarized, or drawn" in content

    # Still forbids inventing unsupported facts.
    assert "never invent facts" in content.lower()


@pytest.mark.asyncio
async def test_system_prompt_forbids_source_by_source_enumeration(provider) -> None:
    """Regression for the 'reads like a source inventory' bug: the model
    was mirroring the [Source N] blocks it was given as an outline for its
    own answer. The system prompt must explicitly instruct synthesis across
    sources in natural prose, organized around the question -- not a
    mechanical per-source rundown -- and must scope citations to specific
    provenance points rather than a running per-sentence label."""

    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)

    call = provider.client.chat.completions.create.await_args
    content = call.kwargs["messages"][0]["content"]

    assert "not a source-by-source report" in content
    assert "NOT an outline to follow" in content
    assert "synthesize" in content.lower()
    assert "organize the answer around the user's question" in content.lower()
    assert "never as the answer's organizing structure" in content.lower()


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
async def test_finish_reason_is_surfaced_on_the_result(provider) -> None:
    """Regression for the incomplete/truncated-answer bug: the provider's
    own stop reason (`"length"` means cut off by `max_tokens`, not that the
    model chose to stop) must reach `GenerationResult` so a truncated answer
    is diagnosable, not silently indistinguishable from a complete one."""
    response = make_response("This answer was cut off mid-sent")
    response.choices[0].finish_reason = "length"
    provider.client.chat.completions.create.return_value = response

    context = ContextAssembler().assemble([])
    result = await provider.generate(query="What is RAG?", context=context)

    assert result.finish_reason == "length"


@pytest.mark.asyncio
async def test_finish_reason_defaults_to_none_when_not_exposed(provider) -> None:
    """A response shape that never set `finish_reason` at all (e.g. an
    older/minimal mock, or a provider that omits it) must yield `None`,
    never raise -- `getattr` with a default, not a direct attribute read."""
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])
    result = await provider.generate(query="What is RAG?", context=context)

    assert result.finish_reason is None


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


class FakeSystemInstructionProvider:
    """Mutable fake -- lets a test change what the NEXT `.get()` call
    returns, to prove `generate` re-reads it every call rather than caching
    whatever was configured at construction time."""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def get(self) -> str:
        return self.text


def _system_message(provider: OpenRouterGenerationProvider) -> str:
    call = provider.client.chat.completions.create.await_args
    messages = call.kwargs["messages"]
    system_message = next(m for m in messages if m["role"] == "system")
    return system_message["content"]


@pytest.mark.asyncio
async def test_no_system_instruction_provider_leaves_prompt_unchanged() -> None:
    """The default (`system_instruction_provider=None`, every existing call
    site) must produce EXACTLY the same system prompt as before this
    feature existed."""

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
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])
    await provider.generate(query="What is RAG?", context=context)

    system_message = _system_message(provider)
    assert system_message.startswith("You are a research assistant.")


@pytest.mark.asyncio
async def test_configured_empty_instruction_leaves_prompt_unchanged() -> None:
    """A configured provider that currently returns "" (no instruction set
    yet, or cleared back to empty) must behave identically to having no
    provider at all -- never prepend a stray blank line."""

    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        system_instruction_provider=FakeSystemInstructionProvider(""),
    )
    provider.client.chat.completions.create = AsyncMock()
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])
    await provider.generate(query="What is RAG?", context=context)

    system_message = _system_message(provider)
    assert system_message.startswith("You are a research assistant.")


@pytest.mark.asyncio
async def test_configured_instruction_is_prepended_ahead_of_the_default_prompt() -> None:
    """A real configured instruction must appear FIRST in the system
    message, ahead of the built-in instructions -- and therefore, since the
    retrieved-chunk context only ever lives in the user message, ahead of
    that too."""

    instruction_provider = FakeSystemInstructionProvider(
        "Always answer in formal British English."
    )
    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        system_instruction_provider=instruction_provider,
    )
    provider.client.chat.completions.create = AsyncMock()
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])
    await provider.generate(query="What is RAG?", context=context)

    system_message = _system_message(provider)
    assert system_message.startswith("Always answer in formal British English.")
    assert "You are a research assistant." in system_message
    assert system_message.index(
        "Always answer in formal British English."
    ) < system_message.index("You are a research assistant.")


@pytest.mark.asyncio
async def test_instruction_is_read_fresh_on_every_call_not_cached() -> None:
    """Changing the store's content between two calls on the SAME provider
    instance must be reflected on the very next call -- proving the
    instruction is never read once and frozen at construction time."""

    instruction_provider = FakeSystemInstructionProvider("First instruction.")
    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        system_instruction_provider=instruction_provider,
    )
    provider.client.chat.completions.create = AsyncMock()
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])

    await provider.generate(query="What is RAG?", context=context)
    assert _system_message(provider).startswith("First instruction.")

    instruction_provider.text = "Second instruction."

    await provider.generate(query="What is RAG?", context=context)
    assert _system_message(provider).startswith("Second instruction.")


@pytest.mark.asyncio
async def test_default_instruction_alone_is_prepended_ahead_of_the_default_prompt() -> None:
    """Configuring only the DEFAULT layer (no user instruction) must behave
    the same way a lone user instruction already does -- prepended ahead of
    the built-in prompt."""

    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        default_system_instruction_provider=FakeSystemInstructionProvider(
            "Always cite page numbers when available."
        ),
    )
    provider.client.chat.completions.create = AsyncMock()
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])
    await provider.generate(query="What is RAG?", context=context)

    system_message = _system_message(provider)
    assert system_message.startswith("Always cite page numbers when available.")
    assert "You are a research assistant." in system_message


@pytest.mark.asyncio
async def test_default_and_user_instructions_combine_as_default_then_user_then_builtin() -> None:
    """Both layers configured together must appear in the order
    [default] + [user] + [built-in prompt] -- never reordered, never one
    silently overriding the other."""

    provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        default_system_instruction_provider=FakeSystemInstructionProvider(
            "Always cite page numbers when available."
        ),
        system_instruction_provider=FakeSystemInstructionProvider(
            "Keep this answer under 100 words."
        ),
    )
    provider.client.chat.completions.create = AsyncMock()
    provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])
    await provider.generate(query="What is RAG?", context=context)

    system_message = _system_message(provider)

    default_pos = system_message.index("Always cite page numbers when available.")
    user_pos = system_message.index("Keep this answer under 100 words.")
    builtin_pos = system_message.index("You are a research assistant.")

    assert default_pos < user_pos < builtin_pos


@pytest.mark.asyncio
async def test_empty_default_instruction_does_not_leave_a_stray_blank_section() -> None:
    """An unconfigured/empty DEFAULT layer alongside a real USER layer must
    produce exactly the same prompt as the user-only case -- no leftover
    blank line or empty section where the default would have gone."""

    with_default_provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        default_system_instruction_provider=FakeSystemInstructionProvider(""),
        system_instruction_provider=FakeSystemInstructionProvider("User instruction."),
    )
    with_default_provider.client.chat.completions.create = AsyncMock()
    with_default_provider.client.chat.completions.create.return_value = make_response(
        "An answer."
    )

    user_only_provider = OpenRouterGenerationProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=None,
        top_p=1.0,
        response_format=None,
        reasoning_effort=None,
        system_instruction_provider=FakeSystemInstructionProvider("User instruction."),
    )
    user_only_provider.client.chat.completions.create = AsyncMock()
    user_only_provider.client.chat.completions.create.return_value = make_response("An answer.")

    context = ContextAssembler().assemble([])

    await with_default_provider.generate(query="What is RAG?", context=context)
    await user_only_provider.generate(query="What is RAG?", context=context)

    assert _system_message(with_default_provider) == _system_message(user_only_provider)
