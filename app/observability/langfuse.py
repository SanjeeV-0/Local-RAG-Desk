"""Optional, purely-additive observability integration.

Every call site in the codebase (`RetrievalService.search`,
`GenerationService.generate`, `app.api.research.ask`) checks
`if langfuse is None` and runs the exact same logic either way -- Langfuse
can never change retrieval, decomposition, or generation behavior, only
observe it. This is independent of, and unrelated to, the retrieval trace
feature controlled by `Settings.trace_enabled` (see `app.retrieval.trace`
and `app.api.retrieval`) -- a request can have one, both, or neither
enabled at the same time.

Every call site is ALSO expected to pre-sanitize what it passes as
`input`/`output`/`metadata` (replacing raw content/exception text with safe
metadata -- see `GenerationService.generate`, `RetrievalService.search`,
`app.api.research.ask`). The `mask` function configured below is a
structural BACKSTOP, not the primary mechanism: it recursively redacts any
string value anywhere in whatever is passed to `start_observation`/
`update`/`set_trace_io`, so a future call site that forgets to pre-sanitize
is still covered. Its actual invocation scope (per-field, whole-value, not
per-leaf-string) was verified locally against the installed SDK (see
`tests/unit/test_langfuse_masking.py`) before being relied upon here -- this
is not an assumption that an SDK feature works as documented.
"""

from functools import lru_cache
from typing import Any

from langfuse import Langfuse

from app.config.settings import get_settings
from app.privacy.pii_detector import redact


def mask_sensitive_data(*, data: Any) -> Any:
    """Recursively redact PII-shaped spans in any string found within
    `data`. `data` is whatever single value Langfuse is about to attach to
    an observation's `input`/`output`/`metadata` field (a dict, list,
    string, or any other JSON-serializable value) -- not a single string,
    so this must walk nested structures itself rather than assume it will
    only ever receive a leaf value.
    """

    if isinstance(data, str):
        return redact(data).text

    if isinstance(data, dict):
        return {key: mask_sensitive_data(data=value) for key, value in data.items()}

    if isinstance(data, list):
        return [mask_sensitive_data(data=item) for item in data]

    if isinstance(data, tuple):
        return tuple(mask_sensitive_data(data=item) for item in data)

    return data


@lru_cache
def get_langfuse() -> Langfuse | None:
    """Return the configured Langfuse client, or None if disabled.

    Raises `ValueError` if `LANGFUSE_ENABLED=true` but the public/secret key
    is missing -- a misconfiguration is surfaced loudly rather than silently
    falling back to "disabled".
    """

    settings = get_settings()

    if not settings.langfuse_enabled:
        return None

    if not settings.langfuse_public_key:
        raise ValueError("LANGFUSE_PUBLIC_KEY must be configured when Langfuse is enabled.")

    if not settings.langfuse_secret_key:
        raise ValueError("LANGFUSE_SECRET_KEY must be configured when Langfuse is enabled.")

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
        mask=mask_sensitive_data,
    )
