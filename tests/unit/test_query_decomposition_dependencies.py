from unittest.mock import patch

import pytest

from app.config.settings import Settings
from app.core.dependencies import (
    get_generation_provider,
    get_query_decomposer,
    get_query_decomposition_provider,
)
from app.generation.openrouter import OpenRouterGenerationProvider
from app.retrieval.openrouter_decomposition import (
    OpenRouterQueryDecompositionProvider,
)
from app.retrieval.query_decomposer import QueryDecomposer


def test_query_decomposition_provider_uses_openrouter_settings() -> None:
    settings = Settings(
        openrouter_api_key="test-key",
        openrouter_decomposition_model="test-decomposition-model",
        openrouter_decomposition_temperature=0.3,
        openrouter_decomposition_max_tokens=64,
        openrouter_decomposition_top_p=0.8,
        openrouter_decomposition_response_format={"type": "json_object"},
        openrouter_decomposition_reasoning_effort="low",
        openrouter_decomposition_top_k=7,
        openrouter_decomposition_frequency_penalty=0.3,
        openrouter_decomposition_presence_penalty=-0.3,
        openrouter_timeout_seconds=12.5,
        openrouter_max_retries=3,
        openrouter_base_url="https://example.test/v1",
        openrouter_app_name="Test App",
    )

    get_query_decomposition_provider.cache_clear()

    with patch(
        "app.core.dependencies.get_settings",
        return_value=settings,
    ):
        provider = get_query_decomposition_provider()

    assert isinstance(provider, OpenRouterQueryDecompositionProvider)
    assert provider.model == "test-decomposition-model"
    assert provider.temperature == 0.3
    assert provider.max_tokens == 64
    assert provider.top_p == 0.8
    assert provider.response_format == {"type": "json_object"}
    assert provider.reasoning_effort == "low"
    assert provider.top_k == 7
    assert provider.frequency_penalty == 0.3
    assert provider.presence_penalty == -0.3
    assert provider.client.timeout == 12.5
    assert provider.client.max_retries == 3
    assert str(provider.client.base_url) == "https://example.test/v1/"

    get_query_decomposition_provider.cache_clear()


def test_generation_provider_uses_generation_settings_not_decomposition_settings() -> None:
    """Dependency wiring must not cross-wire the two roles' settings --
    the generation provider receives ONLY `openrouter_generation_*` values,
    which (per the approved defaults) differ from the decomposition ones."""

    settings = Settings(
        openrouter_api_key="test-key",
        openrouter_generation_model="test-generation-model",
        openrouter_generation_temperature=0.6,
        openrouter_generation_max_tokens=512,
        openrouter_generation_top_p=0.95,
        openrouter_generation_response_format={"type": "text"},
        openrouter_generation_reasoning_effort="high",
        openrouter_generation_top_k=42,
        openrouter_generation_frequency_penalty=0.6,
        openrouter_generation_presence_penalty=-0.6,
        openrouter_timeout_seconds=20.0,
        openrouter_max_retries=1,
        # Deliberately different decomposition values, to prove they never
        # leak into the generation provider.
        openrouter_decomposition_model="test-decomposition-model",
        openrouter_decomposition_temperature=0.0,
        openrouter_decomposition_top_k=7,
        openrouter_decomposition_frequency_penalty=0.1,
        openrouter_base_url="https://example.test/v1",
        openrouter_app_name="Test App",
    )

    get_generation_provider.cache_clear()

    with patch(
        "app.core.dependencies.get_settings",
        return_value=settings,
    ):
        provider = get_generation_provider()

    assert isinstance(provider, OpenRouterGenerationProvider)
    assert provider.model == "test-generation-model"
    assert provider.temperature == 0.6
    assert provider.max_tokens == 512
    assert provider.top_p == 0.95
    assert provider.response_format == {"type": "text"}
    assert provider.reasoning_effort == "high"
    assert provider.top_k == 42
    assert provider.frequency_penalty == 0.6
    assert provider.presence_penalty == -0.6
    assert provider.client.timeout == 20.0
    assert provider.client.max_retries == 1
    assert str(provider.client.base_url) == "https://example.test/v1/"

    # Decomposition's distinct top_k/frequency_penalty values never leaked
    # into this generation provider instance.
    assert provider.top_k != 7
    assert provider.frequency_penalty != 0.1

    get_generation_provider.cache_clear()


def test_missing_openrouter_api_key_produces_a_safe_actionable_diagnostic() -> None:
    """A missing `OPENROUTER_API_KEY` must fail fast with a clear message
    naming the exact environment variable -- never a secret value, and
    never left to surface as an opaque downstream provider/auth error."""

    settings = Settings(openrouter_api_key=None)

    get_generation_provider.cache_clear()

    with patch("app.core.dependencies.get_settings", return_value=settings):
        with pytest.raises(ValueError) as exc_info:
            get_generation_provider()

    assert "OPENROUTER_API_KEY" in str(exc_info.value)
    assert "must be configured" in str(exc_info.value)

    get_generation_provider.cache_clear()


def test_empty_string_openrouter_api_key_is_treated_as_missing() -> None:
    """An empty string is falsy, same as `None` -- both must be rejected by
    the same safe, actionable check rather than silently constructing a
    client with no usable credential."""

    settings = Settings(openrouter_api_key="")

    get_generation_provider.cache_clear()

    with patch("app.core.dependencies.get_settings", return_value=settings):
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY must be configured"):
            get_generation_provider()

    get_generation_provider.cache_clear()


def test_query_decomposer_uses_decomposition_provider() -> None:
    get_query_decomposition_provider.cache_clear()

    with patch(
        "app.core.dependencies.get_query_decomposition_provider",
    ) as provider_factory:
        provider = object()
        provider_factory.return_value = provider

        decomposer = get_query_decomposer()

    assert isinstance(decomposer, QueryDecomposer)
    assert decomposer.provider is provider
