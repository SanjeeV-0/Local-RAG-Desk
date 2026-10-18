import json

import httpx
import openai
import pytest

from app.generation.errors import (
    ProviderErrorCategory,
    ProviderUnavailableError,
    _sanitize_provider_message,
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


# ---------------------------------------------------------------------------
# http_status / provider_error_code / provider_error_type / sanitized
# provider message -- the diagnostic fields added for Langfuse observability.
# ---------------------------------------------------------------------------


def test_http_status_is_captured_exactly_from_the_response() -> None:
    exc = _status_error(openai.PermissionDeniedError, 403)
    wrapped = wrap_openai_exception(exc)
    assert wrapped.http_status == 403


def test_http_status_is_none_for_a_non_http_exception() -> None:
    """Timeouts/connection errors never reach an HTTP response at all --
    this must be None, never a guessed/inferred value."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    exc = openai.APITimeoutError(request=request)
    wrapped = wrap_openai_exception(exc)
    assert wrapped.http_status is None


def test_provider_error_code_reads_the_nested_openrouter_shape() -> None:
    """Regression for a real, confirmed bug: OpenRouter nests `code`/`type`
    under `body["error"]`, but the openai SDK's own `exc.code`/`exc.type`
    attributes only ever parse the TOP level of the body (confirmed by
    reading openai._exceptions.APIError.__init__) -- so they are always
    None for OpenRouter's actual shape regardless of what the body
    contains. The fix reads the nested location first."""
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": RAW_MARKER, "code": "model_not_available"}},
    )
    wrapped = wrap_openai_exception(exc)
    assert wrapped.provider_error_code == "model_not_available"


def test_provider_error_type_reads_the_nested_openrouter_shape() -> None:
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": RAW_MARKER, "type": "insufficient_quota"}},
    )
    wrapped = wrap_openai_exception(exc)
    assert wrapped.provider_error_type == "insufficient_quota"


def test_provider_error_code_falls_back_to_flat_sdk_shape() -> None:
    """A provider using OpenAI's own flat (non-nested) error shape must
    still work -- the nested lookup is additive, not a replacement."""
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"message": RAW_MARKER, "code": "flat_shape_code"},
    )
    wrapped = wrap_openai_exception(exc)
    assert wrapped.provider_error_code == "flat_shape_code"


def test_numeric_error_code_mirroring_http_status_is_not_stringified() -> None:
    """Reproduces the exact real OpenRouter shape observed this session:
    `error.code` is the bare int 403 (mirroring the HTTP status), not a
    descriptive string -- this must yield None, never `"403"` fabricated
    from the int."""
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": RAW_MARKER, "code": 403}},
    )
    wrapped = wrap_openai_exception(exc)
    assert wrapped.provider_error_code is None
    assert wrapped.http_status == 403


def test_sanitized_provider_message_is_captured_when_safe() -> None:
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": "Invalid API key provided."}},
    )
    wrapped = wrap_openai_exception(exc)
    assert wrapped.sanitized_provider_message == "Invalid API key provided."


def test_sanitized_provider_message_redacts_bearer_token() -> None:
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={
            "error": {"message": "Rejected request with header Bearer sk-FAKESECRETVALUE1234567890"}
        },
    )
    wrapped = wrap_openai_exception(exc)
    assert "sk-FAKESECRETVALUE1234567890" not in wrapped.sanitized_provider_message
    assert "[REDACTED]" in wrapped.sanitized_provider_message


def test_sanitized_provider_message_redacts_api_key_assignment() -> None:
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": "api_key=sk-or-v1-FAKEKEYFAKEKEYFAKEKEY123456 is invalid"}},
    )
    wrapped = wrap_openai_exception(exc)
    assert "sk-or-v1-FAKEKEYFAKEKEYFAKEKEY123456" not in wrapped.sanitized_provider_message
    assert "[REDACTED]" in wrapped.sanitized_provider_message


def test_sanitized_provider_message_redacts_generic_long_opaque_token() -> None:
    """Even without a recognizable 'key=' or 'Bearer' prefix, a long opaque
    token-shaped string must still be redacted -- the broad, conservative
    catch-all pattern."""
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": "request id abcdefghij0123456789ABCDEFGHIJ rejected"}},
    )
    wrapped = wrap_openai_exception(exc)
    assert "abcdefghij0123456789ABCDEFGHIJ" not in wrapped.sanitized_provider_message
    assert "[REDACTED]" in wrapped.sanitized_provider_message


def test_sanitized_provider_message_redacts_pii_shaped_spans() -> None:
    """Defense in depth: the existing PII redactor is applied too, in case
    a provider echoes back a PII-shaped span from the original request."""
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": "Rejected content from user@example.com"}},
    )
    wrapped = wrap_openai_exception(exc)
    assert "user@example.com" not in wrapped.sanitized_provider_message


def test_sanitized_provider_message_is_truncated_to_a_bounded_length() -> None:
    long_message = "the request was rejected for policy reasons " * 50
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={"error": {"message": long_message}},
    )
    wrapped = wrap_openai_exception(exc)
    assert len(wrapped.sanitized_provider_message) <= 320
    assert wrapped.sanitized_provider_message.endswith("...[truncated]")


def test_sanitize_provider_message_none_and_empty_yield_none() -> None:
    """Unit-level check of `_sanitize_provider_message` directly: routing
    this through `wrap_openai_exception` can't isolate a missing message,
    since the SDK's own flat `exc.message` fallback is always populated by
    the `openai.APIError` constructor from whatever `message=` was passed
    when the exception was raised."""
    assert _sanitize_provider_message(None) is None
    assert _sanitize_provider_message("") is None


def test_provider_unavailable_error_backward_compatible_without_new_fields() -> None:
    """Every pre-existing call site that constructs `ProviderUnavailableError`
    with just a category (e.g. `ProviderUnavailableError(ProviderErrorCategory
    .INVALID_PROVIDER_RESPONSE)` in app.generation.openrouter) must keep
    working unchanged -- the new fields are optional and default to None."""
    error = ProviderUnavailableError(ProviderErrorCategory.INVALID_PROVIDER_RESPONSE)
    assert error.http_status is None
    assert error.provider_error_code is None
    assert error.provider_error_type is None
    assert error.sanitized_provider_message is None


def test_guardrail_classification_is_unchanged_by_the_new_diagnostic_fields() -> None:
    """Regression: adding diagnostic metadata extraction must never change
    WHICH category is assigned -- classification and metadata extraction
    are independent concerns."""
    exc = _status_error(
        openai.PermissionDeniedError,
        403,
        body={
            "error": {
                "message": "Request blocked: PII detected (PERSON)",
                "code": 403,
                "metadata": {"provider_name": None, "previous_errors": []},
            }
        },
    )
    wrapped = wrap_openai_exception(exc)
    assert wrapped.category == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION
    # The message is still captured for diagnostics -- it's the category
    # boolean decision that must be unaffected, not the presence of this field.
    assert wrapped.sanitized_provider_message is not None


# ---------------------------------------------------------------------------
# Regression for a real, confirmed bug found via a live OpenRouter 403:
# `openai`'s OWN `_make_status_error` (in `_client.py`, not `_exceptions.py`)
# already unwraps a `{"error": {...}}` envelope before an exception is ever
# constructed -- `data = body.get("error", body) if is_mapping(body) else
# body`. So `exc.body` is ALREADY the inner object for every exception that
# travels through the real SDK client, never the `{"error": {...}}` wrapper
# this module previously assumed it still had. `_status_error` above
# bypasses that unwrapping entirely (it constructs the exception by hand),
# which is exactly why the earlier nested-vs-flat fix still passed its own
# tests while remaining broken for real traffic -- this test goes through
# the real client method instead, to exercise the actual mechanism.
# ---------------------------------------------------------------------------


def _real_sdk_status_error(status_code: int, body: dict) -> openai.APIStatusError:
    """Construct an exception the same way the real `AsyncOpenAI` client
    does: via its own `_make_status_error_from_response`/`_make_status_error`,
    so the SDK's own envelope-unwrapping behavior is exercised for real,
    rather than assumed."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        status_code=status_code,
        request=request,
        content=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    client = openai.OpenAI(api_key="sk-test", base_url="https://openrouter.ai/api/v1")
    return client._make_status_error_from_response(response)


def test_real_sdk_already_unwraps_the_error_envelope_before_body_is_set() -> None:
    """Pins down the mechanism itself: `exc.body` for a real SDK exception
    has no `"error"` key at all -- it IS the inner object directly."""
    body = {"error": {"message": "Request blocked: PII detected (PERSON)", "code": 403}}
    exc = _real_sdk_status_error(403, body)
    assert isinstance(exc.body, dict)
    assert "error" not in exc.body
    assert exc.body.get("message") == "Request blocked: PII detected (PERSON)"


def test_real_openrouter_pii_guardrail_403_is_classified_correctly() -> None:
    """End-to-end regression for the actual production failure: a real
    OpenRouter 403 body shaped exactly like the one captured from a live
    request (`{"error": {"message": "Request blocked: PII detected
    (PERSON)", "code": 403, "metadata": {...}}}`), constructed through the
    real SDK client so the envelope-unwrapping is genuine, must be
    classified as a guardrail rejection -- NOT
    `authentication_or_configuration_error` -- and must carry the sanitized
    message for diagnostics."""
    body = {
        "error": {
            "message": "Request blocked: PII detected (PERSON)",
            "code": 403,
            "metadata": {
                "provider_name": None,
                "previous_errors": [
                    {"code": 403, "message": "Request blocked: PII detected (PERSON)"}
                ],
            },
        }
    }
    exc = _real_sdk_status_error(403, body)
    wrapped = wrap_openai_exception(exc)

    assert wrapped.category == ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION
    assert wrapped.http_status == 403
    assert wrapped.sanitized_provider_message == "Request blocked: PII detected (PERSON)"


def test_real_sdk_flat_auth_error_still_classifies_as_auth_config() -> None:
    """Regression guard: a genuine authentication/config failure (no
    guardrail signal anywhere in the body) must still classify correctly
    through the real SDK's unwrapping path -- the fix must not turn every
    403 into a guardrail rejection."""
    body = {"error": {"message": "Invalid API key provided.", "code": "invalid_api_key"}}
    exc = _real_sdk_status_error(401, body)
    wrapped = wrap_openai_exception(exc)

    assert wrapped.category == ProviderErrorCategory.AUTHENTICATION_OR_CONFIG
    assert wrapped.provider_error_code == "invalid_api_key"
    assert wrapped.sanitized_provider_message == "Invalid API key provided."
