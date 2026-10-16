"""Typed, safe error classification for OpenRouter provider calls.

Both OpenRouter call sites (`app.generation.openrouter.OpenRouterGenerationProvider.generate`
and `app.retrieval.openrouter_decomposition.OpenRouterQueryDecompositionProvider.decompose`)
wrap their `chat.completions.create(...)` call with the same classification
logic here, so a provider failure always surfaces as a `ProviderUnavailableError`
with a safe, generic, category-specific message -- never the raw exception
text or provider response body.

Categorization is best-effort. Critically, an HTTP 400/403 status ALONE is
NOT treated as proof of a guardrail/content-policy rejection -- a plain
`BadRequestError` is far more often a malformed request (bad parameter,
unsupported field, oversized context) than a moderation block, and a 403
`PermissionDeniedError` is far more often an access/API-key restriction than
a content decision. `GUARDRAIL_OR_POLICY_REJECTION` is only assigned when
the parsed error body carries a recognizable moderation-related signal (see
`_looks_like_guardrail_rejection` below); otherwise a 400 is reported as
`INVALID_REQUEST` and a 403 as `AUTHENTICATION_OR_CONFIG`, so callers are
never told "the guardrail blocked this" without actual evidence. This
classification never bypasses or weakens OpenRouter's own guardrails; it
only decides how a failure already returned by OpenRouter is reported
safely to the caller -- and does so honestly, without over-claiming the
cause.

Retry behavior: the installed `openai` SDK's own `_should_retry` (verified
locally) already retries only on HTTP 408/409/429/>=500 -- 400/401/403
(where a guardrail/auth rejection lands) are never retried by the SDK
itself, so no additional no-retry wrapper is needed here for policy
rejections.
"""

from __future__ import annotations

from enum import StrEnum

import openai


class ProviderErrorCategory(StrEnum):
    GUARDRAIL_OR_POLICY_REJECTION = "guardrail_or_policy_rejection"
    AUTHENTICATION_OR_CONFIG = "authentication_or_configuration_error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT_OR_CONNECTIVITY = "timeout_or_connectivity"
    PROVIDER_HTTP_ERROR = "provider_http_error"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    # A plain 400 with no recognizable moderation/guardrail signal in the
    # error body -- most often a malformed request, not a content decision.
    # Kept distinct from GUARDRAIL_OR_POLICY_REJECTION so callers are never
    # told "blocked by policy" without actual evidence of that.
    INVALID_REQUEST = "invalid_request"


_SAFE_MESSAGES: dict[ProviderErrorCategory, str] = {
    ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION: (
        "The research assistant's provider declined to process this request."
    ),
    ProviderErrorCategory.AUTHENTICATION_OR_CONFIG: (
        "The research assistant is temporarily misconfigured. Please try again later."
    ),
    ProviderErrorCategory.RATE_LIMITED: (
        "The research assistant is temporarily busy. Please try again shortly."
    ),
    ProviderErrorCategory.TIMEOUT_OR_CONNECTIVITY: (
        "The research assistant's provider did not respond in time. Please try again."
    ),
    ProviderErrorCategory.PROVIDER_HTTP_ERROR: (
        "The research assistant's provider returned an unexpected error."
    ),
    ProviderErrorCategory.INVALID_PROVIDER_RESPONSE: (
        "The research assistant's provider returned an invalid response."
    ),
    ProviderErrorCategory.INVALID_REQUEST: (
        "The research assistant's request to its provider was rejected as invalid."
    ),
}

# Categories that may succeed on a later attempt -- never includes a policy
# rejection, which must never be retried blindly.
_RETRYABLE_CATEGORIES = frozenset(
    {
        ProviderErrorCategory.RATE_LIMITED,
        ProviderErrorCategory.TIMEOUT_OR_CONNECTIVITY,
    }
)


class ProviderUnavailableError(Exception):
    """Raised instead of propagating a raw OpenRouter/openai SDK exception.

    `safe_message` is always a fixed, category-specific string -- it never
    contains the original exception's text or any provider response body.
    """

    def __init__(self, category: ProviderErrorCategory, retryable: bool | None = None) -> None:
        self.category = category
        self.safe_message = _SAFE_MESSAGES[category]
        self.retryable = (
            category in _RETRYABLE_CATEGORIES if retryable is None else retryable
        )
        super().__init__(self.safe_message)


# Substrings looked for (case-insensitively) in whatever safe, already-
# server-side-only diagnostic fields the `openai` SDK parsed out of the
# error body (`exc.code`, `exc.type`, `exc.param`) -- never the raw message
# text itself, and never exposed to the caller either way, only used here to
# decide which safe category to report. This is deliberately conservative:
# a 400/403 with none of these signals is NOT assumed to be a guardrail
# rejection (see module docstring).
_GUARDRAIL_SIGNAL_SUBSTRINGS = (
    "moderation",
    "flagged",
    "content_policy",
    "content_filter",
    "jailbreak",
    "guardrail",
    "pii",
    "harmful",
    "safety",
)


def _looks_like_guardrail_rejection(exc: openai.APIStatusError) -> bool:
    """Best-effort, evidence-based check for an actual moderation/guardrail
    signal in the parsed error body -- never assumed from status code alone.

    Inspects `exc.code`/`exc.type`/`exc.param` (parsed server-side by the
    SDK from the JSON error body) plus the top-level keys of `exc.body`
    (e.g. OpenRouter's own `error.metadata` wrapper), never the raw message
    string, so this stays a structural check rather than text matching on
    content that could itself contain user data.
    """

    candidates: list[str] = []

    for attr in ("code", "type", "param"):
        value = getattr(exc, attr, None)
        if isinstance(value, str):
            candidates.append(value.lower())

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error_obj = body.get("error")
        if isinstance(error_obj, dict):
            for key in ("code", "type"):
                value = error_obj.get(key)
                if isinstance(value, str):
                    candidates.append(value.lower())

            metadata = error_obj.get("metadata")
            if isinstance(metadata, dict):
                # The metadata wrapper's own key names (e.g. "reasons",
                # "flagged_input") are informative without reading any
                # value inside it -- values may contain request content and
                # are deliberately never inspected or logged here.
                candidates.extend(key.lower() for key in metadata)

    return any(
        signal in candidate for candidate in candidates for signal in _GUARDRAIL_SIGNAL_SUBSTRINGS
    )


def classify_openai_exception(exc: Exception) -> ProviderErrorCategory:
    """Map an `openai` SDK exception to a safe error category.

    Falls back to `PROVIDER_HTTP_ERROR` for any other `openai.APIStatusError`
    subclass not explicitly handled, and re-raises unknown, non-provider
    exceptions unchanged (callers should only pass exceptions already known
    to originate from the `openai` client call).
    """

    if isinstance(exc, openai.RateLimitError):
        return ProviderErrorCategory.RATE_LIMITED

    if isinstance(exc, openai.AuthenticationError):
        return ProviderErrorCategory.AUTHENTICATION_OR_CONFIG

    if isinstance(exc, openai.PermissionDeniedError):
        # A 403 is far more often an access/API-key restriction than a
        # content decision -- only escalate to a guardrail rejection when
        # the body actually says so.
        return (
            ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION
            if _looks_like_guardrail_rejection(exc)
            else ProviderErrorCategory.AUTHENTICATION_OR_CONFIG
        )

    if isinstance(exc, openai.BadRequestError):
        # A plain 400 is far more often a malformed request than a
        # moderation block -- only classify as a guardrail rejection when
        # the body carries an actual recognizable signal.
        return (
            ProviderErrorCategory.GUARDRAIL_OR_POLICY_REJECTION
            if _looks_like_guardrail_rejection(exc)
            else ProviderErrorCategory.INVALID_REQUEST
        )

    if isinstance(exc, openai.APITimeoutError | openai.APIConnectionError):
        return ProviderErrorCategory.TIMEOUT_OR_CONNECTIVITY

    if isinstance(exc, openai.APIStatusError):
        return ProviderErrorCategory.PROVIDER_HTTP_ERROR

    raise exc


def wrap_openai_exception(exc: Exception) -> ProviderUnavailableError:
    """Classify `exc` and return the corresponding safe `ProviderUnavailableError`.

    A 5xx `openai.InternalServerError` is the one `PROVIDER_HTTP_ERROR` case
    treated as retryable -- every other provider HTTP error defaults to
    non-retryable.
    """

    category = classify_openai_exception(exc)

    if category is ProviderErrorCategory.PROVIDER_HTTP_ERROR and isinstance(
        exc, openai.APIStatusError
    ):
        return ProviderUnavailableError(category, retryable=exc.status_code >= 500)

    return ProviderUnavailableError(category)
