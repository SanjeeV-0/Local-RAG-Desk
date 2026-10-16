"""Verifies the Langfuse masking backstop -- both as a pure function, and
that it is actually wired into (and invoked by) the installed Langfuse SDK's
own masking call path, per the explicit requirement to verify SDK masking
behavior before relying on it rather than assuming it works."""

from app.observability import langfuse as langfuse_module
from app.observability.langfuse import mask_sensitive_data

SYNTHETIC_EMAIL = "synthetic.marker@example.com"


def test_masks_top_level_string() -> None:
    assert mask_sensitive_data(data=f"Contact {SYNTHETIC_EMAIL} now.") == (
        "Contact [REDACTED:EMAIL] now."
    )


def test_masks_nested_dict_values() -> None:
    data = {
        "query": f"Email {SYNTHETIC_EMAIL}",
        "nested": {"content": "Call 555-123-4567 for help."},
    }
    masked = mask_sensitive_data(data=data)

    assert SYNTHETIC_EMAIL not in str(masked)
    assert "555-123-4567" not in str(masked)
    assert masked["query"] == "Email [REDACTED:EMAIL]"
    assert masked["nested"]["content"] == "Call [REDACTED:PHONE] for help."


def test_masks_values_inside_lists() -> None:
    data = {"sources": [f"from {SYNTHETIC_EMAIL}", "a safe string"]}
    masked = mask_sensitive_data(data=data)

    assert SYNTHETIC_EMAIL not in str(masked)
    assert masked["sources"][1] == "a safe string"


def test_non_string_scalars_pass_through_unchanged() -> None:
    data = {"count": 5, "ratio": 0.5, "flag": True, "nothing": None}
    masked = mask_sensitive_data(data=data)

    assert masked == data


def test_clean_data_is_returned_unchanged() -> None:
    data = {"model": "test-model", "source_count": 3}
    assert mask_sensitive_data(data=data) == data


def test_get_langfuse_constructs_client_with_mask_function(monkeypatch) -> None:
    """Confirms `get_langfuse()` actually passes `mask=` to the SDK's
    `Langfuse(...)` constructor -- not merely that the function exists."""

    captured_kwargs: dict[str, object] = {}

    class _FakeLangfuseClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(langfuse_module, "Langfuse", _FakeLangfuseClient)

    settings = langfuse_module.get_settings()
    monkeypatch.setattr(type(settings), "langfuse_enabled", True, raising=False)
    monkeypatch.setattr(type(settings), "langfuse_public_key", "pk-test", raising=False)
    monkeypatch.setattr(type(settings), "langfuse_secret_key", "sk-test", raising=False)

    langfuse_module.get_langfuse.cache_clear()
    try:
        client = langfuse_module.get_langfuse()
        assert isinstance(client, _FakeLangfuseClient)
        assert captured_kwargs["mask"] is mask_sensitive_data
    finally:
        langfuse_module.get_langfuse.cache_clear()


def test_sdk_mask_attribute_hook_invokes_configured_mask_function() -> None:
    """Verifies the installed Langfuse SDK's own internal masking call path
    (`_mask_attribute`) actually invokes whatever function was passed as
    `mask=` -- i.e. this is not merely a parameter the SDK silently ignores.
    Constructed via the SDK's internal mock-friendly path: a minimal object
    carrying `_mask`, mirroring `_LangfuseSpan._mask_attribute`'s own
    `self._langfuse_client._mask(data=data)` call exactly."""

    from langfuse._client.span import LangfuseSpan

    class _ClientStub:
        _mask = staticmethod(mask_sensitive_data)

    # `_mask_attribute` only reads `self._langfuse_client`, so a bare,
    # unconstructed instance is sufficient to exercise the real method.
    span = object.__new__(LangfuseSpan)
    span._langfuse_client = _ClientStub()

    result = span._mask_attribute(data={"query": f"Email {SYNTHETIC_EMAIL}"})

    assert SYNTHETIC_EMAIL not in str(result)
    assert result["query"] == "Email [REDACTED:EMAIL]"
