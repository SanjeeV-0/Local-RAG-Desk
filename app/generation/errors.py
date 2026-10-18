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
the parsed error body carries a recognizable moderation-related signal --
preferably a structured field (`code`/`type`/`metadata` keys), with the
parsed `error.message` text itself checked only as a narrow fallback
against a curated phrase list (see `_looks_like_guardrail_rejection`
below); otherwise a 400 is reported as
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

import re
from enum import StrEnum

import openai

from app.privacy.pii_detector import redact


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

    `http_status`, `provider_error_code`, `provider_error_type`, and
    `sanitized_provider_message` are OPTIONAL, narrowly-scoped diagnostic
    fields for observability (see `wrap_openai_exception`, the only place
    that ever sets them to a non-`None` value) -- never surfaced in
    `safe_message` or to any HTTP caller (confirmed: `app.api.research`
    only ever reads `.category`/`.retryable`/`.safe_message` when building
    an HTTP response), only attached for a trusted, internal observability
    sink (Langfuse) to record. `http_status`/`provider_error_code`/
    `provider_error_type` are structural, already-parsed SDK/body fields
    (never free text); `sanitized_provider_message` is the one field that
    DOES derive from free text (the provider's `error.message`), and is
    never stored except through `_sanitize_provider_message`, which strips
    credential-shaped substrings, applies the existing PII redactor, and
    caps length BEFORE this constructor ever sees it -- this class itself
    never re-reads or re-sanitizes anything, it only stores what it's given.
    """

    def __init__(
        self,
        category: ProviderErrorCategory,
        retryable: bool | None = None,
        http_status: int | None = None,
        provider_error_code: str | None = None,
        provider_error_type: str | None = None,
        sanitized_provider_message: str | None = None,
    ) -> None:
        self.category = category
        self.safe_message = _SAFE_MESSAGES[category]
        self.retryable = (
            category in _RETRYABLE_CATEGORIES if retryable is None else retryable
        )
        self.http_status = http_status
        self.provider_error_code = provider_error_code
        self.provider_error_type = provider_error_type
        self.sanitized_provider_message = sanitized_provider_message
        super().__init__(self.safe_message)


# Substrings looked for (case-insensitively), in priority order, in the
# diagnostic fields the `openai` SDK parsed out of the error body
# (`exc.code`, `exc.type`, `exc.param`, `error.metadata` keys), and -- only
# as a narrow fallback when none of those carry a signal -- the parsed
# `error.message` text itself (see `_looks_like_guardrail_rejection`). None
# of this is ever exposed to the caller; it is used only to decide which
# safe, fixed category string to report. This is deliberately conservative:
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


def _parsed_error_object(exc: Exception) -> dict | None:
    """Return the nested `body["error"]` dict, if `exc.body` is a dict
    containing one -- this is OpenRouter's (and OpenAI's own) actual error
    JSON shape: `{"error": {"message": ..., "code": ..., "type": ...,
    "metadata": {...}}}`.

    CONFIRMED (by reading the installed SDK's `_client.py`, not just
    `_exceptions.py`): `AsyncOpenAI`/`OpenAI`'s own `_make_status_error`
    already unwraps that envelope itself before constructing the exception
    -- `data = body.get("error", body) if is_mapping(body) else body` --
    so for every real call made through this SDK, `exc.body` IS ALREADY the
    inner object (`{"message": ..., "code": ..., "metadata": {...}}`
    directly), never the `{"error": {...}}` wrapper. A prior version of
    this function assumed `exc.body` still carried that wrapper and dug for
    `body["error"]` -- which is always absent on real traffic, silently
    falling through to the useless flat fallback (see `_structured_field`)
    every time. Verified directly against a real captured OpenRouter 403
    body reconstructed through the SDK's actual `_make_status_error` (not
    just a hand-built `APIStatusError`, which is what this module's own
    tests had been constructing, masking the bug).

    This function therefore prefers `body` itself as the error object, with
    a defensive fallback to `body["error"]` only for a shape the SDK did
    NOT pass through that unwrapping step (e.g. a hand-built exception in a
    test, or a future SDK change) -- so both real traffic and the existing
    nested-body test fixtures are handled correctly.
    """

    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None

    nested = body.get("error")
    if isinstance(nested, dict):
        return nested

    return body


def _structured_field(exc: Exception, field_name: str) -> str | None:
    """Return `field_name` (e.g. `"code"`, `"type"`, `"message"`) from the
    nested `body["error"]` object if present (see `_parsed_error_object`),
    falling back to the SDK's own top-level `exc.<field_name>` attribute
    for any provider that DOES use OpenAI's flat error shape. Returns a
    string or `None` -- never fabricates a value, never returns a non-string
    (OpenRouter sometimes sets `error.code` to a bare int mirroring the HTTP
    status rather than a descriptive string; that case correctly yields
    `None` here, not a stringified number).
    """

    error_obj = _parsed_error_object(exc)
    if error_obj is not None:
        value = error_obj.get(field_name)
        if isinstance(value, str):
            return value

    value = getattr(exc, field_name, None)
    return value if isinstance(value, str) else None


def _looks_like_guardrail_rejection(exc: openai.APIStatusError) -> bool:
    """Best-effort, evidence-based check for an actual moderation/guardrail
    signal in the parsed error body -- never assumed from status code alone.

    Structured fields are checked first and are the preferred signal:
    `exc.code`/`exc.type`/`exc.param` (parsed server-side by the SDK from
    the JSON error body) plus the top-level keys of `exc.body` (e.g.
    OpenRouter's own `error.metadata` wrapper) -- these are machine-set
    values, not free text, so matching them carries no risk of keying off
    arbitrary user/document content.

    As a narrow fallback, the parsed `error.message` field's TEXT is also
    checked against the same curated signal list. This is a deliberate,
    narrowly-scoped exception to "never inspect message text": observed
    real OpenRouter guardrail rejections (e.g. a 403 whose *already-SDK-
    unwrapped* body -- see `_parsed_error_object` -- is
    `{"message": "Request blocked: PII detected (PERSON)", ...}`) carry
    their only machine-readable signal inside this field, with no matching
    `code`/`type`/`metadata` key. The message text is used ONLY to decide
    the boolean classification here; it is never stored, logged, or
    included in `ProviderUnavailableError.safe_message` (see
    `_SAFE_MESSAGES`, which is a fixed, category-keyed string, and
    `wrap_openai_exception`, which never forwards `exc` text to its return
    value). A 403/400 whose message carries none of these specific signals
    is NOT assumed to be a guardrail rejection -- it stays classified as an
    auth/config or invalid-request error, same as before this fallback was
    added.
    """

    candidates: list[str] = []

    for attr in ("code", "type", "param"):
        value = getattr(exc, attr, None)
        if isinstance(value, str):
            candidates.append(value.lower())

    error_obj = _parsed_error_object(exc)
    if error_obj is not None:
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

        message = error_obj.get("message")
        if isinstance(message, str):
            # Narrow fallback only -- see docstring above. Appended
            # last so structured-field signals are always checked
            # first/independently; this never replaces them.
            candidates.append(message.lower())

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


def _safe_provider_error_code(exc: Exception) -> str | None:
    """Return the provider's structured `error.code`, if present and a
    string -- a short, structured identifier (e.g. `"invalid_api_key"`),
    never the raw message or response body. Checks the nested OpenRouter
    shape first, then the SDK's flat top-level attribute (see
    `_structured_field`). Returns `None` rather than fabricating a value
    when absent, or when `code` is present but not a string (some OpenRouter
    responses set `error.code` to a bare int mirroring the HTTP status
    rather than a descriptive code -- that is correctly reported as "no
    string code available", not stringified).
    """

    return _structured_field(exc, "code")


def _safe_provider_error_type(exc: Exception) -> str | None:
    """Return the provider's structured `error.type`, if present and a
    string -- e.g. `"invalid_request_error"`. Same nested-then-flat lookup
    and the same "never fabricate" guarantee as `_safe_provider_error_code`.
    """

    return _structured_field(exc, "type")


# Broad, deliberately over-inclusive patterns for anything that LOOKS like a
# credential, applied to a provider error message before it is ever stored
# anywhere. Erring toward redacting a non-secret-looking token is the safe
# failure mode here; erring the other way is not. This is best-effort, not a
# guarantee -- an entirely novel secret format invented by a future provider
# could still slip through; see `_sanitize_provider_message`'s docstring.
_CREDENTIAL_LIKE_PATTERNS = (
    # Bearer-scheme tokens, wherever they appear in the text.
    re.compile(r"(?i)\bBearer\s+\S{10,}"),
    # Common vendor API-key prefixes (OpenAI/OpenRouter-style "sk-...").
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}"),
    # "key=...", "api_key: ...", "authorization: ...", "token=..." etc.
    re.compile(r"(?i)\b(?:api[_-]?key|authorization|secret|token)\b\s*[:=]\s*\S+"),
    # Any other long, opaque alphanumeric/underscore/dash run (20+ chars) --
    # generic enough to catch key formats not matched above, at the cost of
    # also redacting non-secret long identifiers; deliberately conservative.
    re.compile(r"\b[A-Za-z0-9_-]{20,}\b"),
)

# Hard cap on how much of a provider message is ever retained, independent
# of what credential/PII redaction removes -- bounds the worst case (an
# unexpectedly large or malformed body) rather than relying on redaction
# alone to keep the stored value small.
_PROVIDER_MESSAGE_MAX_LENGTH = 300


def _sanitize_provider_message(message: str | None) -> str | None:
    """Best-effort sanitization of a provider error message before it is
    ever recorded anywhere. Three layers, always in this order: (1) replace
    credential-shaped substrings (see `_CREDENTIAL_LIKE_PATTERNS`) with a
    fixed placeholder; (2) run the existing PII redactor
    (`app.privacy.pii_detector.redact`) over the result, in case the
    provider echoed back a PII-shaped span from the original request; (3)
    truncate to `_PROVIDER_MESSAGE_MAX_LENGTH`. Returns `None` for an empty
    or missing message -- never an empty string standing in for "nothing to
    report".

    This is best-effort, not a guarantee of completeness -- the patterns
    above are broad but finite; an unseen credential format could still
    pass through. Callers must treat this as diagnostic-quality text, not a
    value safe to use for anything beyond human review in a trusted sink.
    """

    if not message:
        return None

    sanitized = message

    for pattern in _CREDENTIAL_LIKE_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)

    sanitized = redact(sanitized).text

    if len(sanitized) > _PROVIDER_MESSAGE_MAX_LENGTH:
        sanitized = sanitized[:_PROVIDER_MESSAGE_MAX_LENGTH] + "...[truncated]"

    return sanitized


def _safe_http_status(exc: Exception) -> int | None:
    """Return the real HTTP status code from an `openai.APIStatusError`, if
    the exception is one -- never inferred from `ProviderErrorCategory`
    (several categories, e.g. `TIMEOUT_OR_CONNECTIVITY`, have no HTTP status
    at all; a plain `isinstance` check keeps this exact rather than guessed).
    """

    return exc.status_code if isinstance(exc, openai.APIStatusError) else None


def wrap_openai_exception(exc: Exception) -> ProviderUnavailableError:
    """Classify `exc` and return the corresponding safe `ProviderUnavailableError`.

    A 5xx `openai.InternalServerError` is the one `PROVIDER_HTTP_ERROR` case
    treated as retryable -- every other provider HTTP error defaults to
    non-retryable.
    """

    category = classify_openai_exception(exc)
    http_status = _safe_http_status(exc)
    provider_error_code = _safe_provider_error_code(exc)
    provider_error_type = _safe_provider_error_type(exc)
    sanitized_provider_message = _sanitize_provider_message(_structured_field(exc, "message"))

    if category is ProviderErrorCategory.PROVIDER_HTTP_ERROR and isinstance(
        exc, openai.APIStatusError
    ):
        return ProviderUnavailableError(
            category,
            retryable=exc.status_code >= 500,
            http_status=http_status,
            provider_error_code=provider_error_code,
            provider_error_type=provider_error_type,
            sanitized_provider_message=sanitized_provider_message,
        )

    return ProviderUnavailableError(
        category,
        http_status=http_status,
        provider_error_code=provider_error_code,
        provider_error_type=provider_error_type,
        sanitized_provider_message=sanitized_provider_message,
    )
