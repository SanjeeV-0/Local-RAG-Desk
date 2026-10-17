import httpx
import openai
import pytest

from app.generation.errors import (
    ProviderErrorCategory,
    ProviderUnavailableError,
    classify_openai_exception,
    wrap_openai_exception,
)

RAW_MARKER = "SYNTHETIC_TEST_RAW_EXCEPTION_MARKER"


def _status_error(
    cls: type[openai.APIStatusError],
    status_code: int,
    body: dict | None = None,
) -> openai.APIStatusError:
    error_body = body if body is not None else {"error": {"message": RAW_MARKER}}
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        status_code=status_code,
        request=request,
        json=error_body,
    )
    return cls(message=RAW_MARKER, response=response, body=error_body)


def test_plain_bad_request_without_moderation_signal_is_invalid_request_not_guardrail() -> None:
    """A 400 with no recognizable moderation/guardrail signal in the body
    must NOT be assumed to be a policy rejection -- it is far more often a
    malformed request (bad parameter, oversized context, etc)."""

    exc = _status_error(openai.BadRequestError, 400)
    assert classify_openai_exception(exc) == ProviderErrorCategory.INVALID_REQUEST


def test_bad_request_with_moderation_code_is_classified_as_guardrail_rejection() -> None:
    exc = _status_error(
        openai.BadRequestError,
        400,
        body={"error": {"message": RAW_MARKER, "code": "content_policy_violation"}},
    )
    assert classify_openai_exception(exc) == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION


def test_bad_request_with_moderation_metadata_is_classified_as_guardrail_rejection() -> None:
    """OpenRouter-style error shape: a `metadata` object (e.g. with a
    `reasons` key) alongside the message -- only the KEY NAMES are
    inspected, never the values, which could contain request content."""

    exc = _status_error(
        openai.BadRequestError,
        400,
        body={
            "error": {
                "message": RAW_MARKER,
                "metadata": {"reasons": ["jailbreak"], "flagged_input": RAW_MARKER},
            }
        },
    )
    assert classify_openai_exception(exc) == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION


def test_plain_permission_denied_without_signal_is_auth_not_guardrail() -> None:
    """A 403 with no moderation signal is far more often an access/API-key
    restriction than a content decision."""

    exc = _status_error(openai.PermissionDeniedError, 403)
    assert classify_openai_exception(exc) == ProviderErrorCategory.AUTHENTICATION_OR_CONFIG


def test_permission_denied_with_moderation_signal_is_guardrail_rejection() -> None:
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": RAW_MARKER, "type": "moderation_error"}},
    )
    assert classify_openai_exception(exc) == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION


def test_permission_denied_with_pii_message_is_classified_as_guardrail_rejection() -> None:
    """Reproduces a real observed OpenRouter 403 whose ONLY signal is in the
    message text -- `code` is the int 403 (not a string, so it is never a
    candidate), and `metadata` carries no recognizable key names."""

    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={
            "error": {
                "message": "Request blocked: PII detected (PERSON)",
                "code": 403,
                "metadata": {"provider_name": None, "previous_errors": []},
            },
            "user_id": "user_abc123",
        },
    )
    assert classify_openai_exception(exc) == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION


def test_bad_request_with_moderation_message_is_classified_as_guardrail_rejection() -> None:
    exc = _status_error(
        openai.BadRequestError,
        400,
        body={"error": {"message": "This request was rejected by our content moderation system."}},
    )
    assert classify_openai_exception(exc) == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION


def test_permission_denied_with_unrelated_message_stays_authentication() -> None:
    """An explicit, unambiguous access/credential rejection message with no
    moderation/PII signal must remain AUTHENTICATION_OR_CONFIG -- a 403 is
    never assumed to be a guardrail rejection just because it has a message."""

    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": "Invalid API key provided.", "code": 403}},
    )
    assert classify_openai_exception(exc) == ProviderErrorCategory.AUTHENTICATION_OR_CONFIG


def test_ambiguous_permission_denied_message_defaults_conservatively() -> None:
    """A vague/ambiguous 403 message with no recognizable moderation or
    auth-specific wording must default conservatively to
    AUTHENTICATION_OR_CONFIG rather than being guessed as a guardrail
    rejection."""

    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": "Forbidden.", "code": 403}},
    )
    assert classify_openai_exception(exc) == ProviderErrorCategory.AUTHENTICATION_OR_CONFIG


def test_message_signal_never_overrides_wrapped_error_public_text() -> None:
    """The message text used for classification must never leak into the
    safe, public-facing error -- only the fixed, category-keyed string."""

    secret_message = "Request blocked: PII detected (PERSON) SYNTHETIC_MARKER_XYZ"
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": secret_message, "code": 403}},
    )
    wrapped = wrap_openai_exception(exc)

    assert wrapped.category == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION
    assert "SYNTHETIC_MARKER_XYZ" not in wrapped.safe_message
    assert "SYNTHETIC_MARKER_XYZ" not in str(wrapped)


def test_authentication_error_classified_correctly() -> None:
    exc = _status_error(openai.AuthenticationError, 401)
    assert classify_openai_exception(exc) == ProviderErrorCategory.AUTHENTICATION_OR_CONFIG


def test_rate_limit_error_classified_correctly() -> None:
    exc = _status_error(openai.RateLimitError, 429)
    assert classify_openai_exception(exc) == ProviderErrorCategory.RATE_LIMITED


def test_internal_server_error_classified_as_provider_http_error() -> None:
    exc = _status_error(openai.InternalServerError, 500)
    assert classify_openai_exception(exc) == ProviderErrorCategory.PROVIDER_HTTP_ERROR


def test_timeout_error_classified_correctly() -> None:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    exc = openai.APITimeoutError(request=request)
    assert classify_openai_exception(exc) == ProviderErrorCategory.TIMEOUT_OR_CONNECTIVITY


def test_connection_error_classified_correctly() -> None:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    exc = openai.APIConnectionError(request=request)
    assert classify_openai_exception(exc) == ProviderErrorCategory.TIMEOUT_OR_CONNECTIVITY


def test_unknown_non_provider_exception_is_reraised_unchanged() -> None:
    original = ValueError("not a provider error")

    with pytest.raises(ValueError, match="not a provider error"):
        classify_openai_exception(original)


def test_wrap_never_exposes_raw_exception_text() -> None:
    exc = _status_error(openai.BadRequestError, 400)
    wrapped = wrap_openai_exception(exc)

    assert isinstance(wrapped, ProviderUnavailableError)
    assert RAW_MARKER not in str(wrapped)
    assert RAW_MARKER not in wrapped.safe_message


def test_invalid_request_is_never_retryable() -> None:
    exc = _status_error(openai.BadRequestError, 400)
    wrapped = wrap_openai_exception(exc)

    assert wrapped.category == ProviderErrorCategory.INVALID_REQUEST
    assert wrapped.retryable is False


def test_policy_rejection_is_never_retryable() -> None:
    exc = _status_error(
        openai.BadRequestError,
        400,
        body={"error": {"message": RAW_MARKER, "code": "content_policy_violation"}},
    )
    wrapped = wrap_openai_exception(exc)

    assert wrapped.category == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION
    assert wrapped.retryable is False


def test_auth_error_is_never_retryable() -> None:
    exc = _status_error(openai.AuthenticationError, 401)
    wrapped = wrap_openai_exception(exc)

    assert wrapped.retryable is False


def test_rate_limited_is_retryable() -> None:
    exc = _status_error(openai.RateLimitError, 429)
    wrapped = wrap_openai_exception(exc)

    assert wrapped.retryable is True


def test_internal_server_error_is_retryable() -> None:
    exc = _status_error(openai.InternalServerError, 500)
    wrapped = wrap_openai_exception(exc)

    assert wrapped.retryable is True


def test_not_found_error_defaults_to_non_retryable_provider_http_error() -> None:
    exc = _status_error(openai.NotFoundError, 404)
    wrapped = wrap_openai_exception(exc)

    assert wrapped.category == ProviderErrorCategory.PROVIDER_HTTP_ERROR
    assert wrapped.retryable is False
