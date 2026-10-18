import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def _settings(**overrides) -> Settings:
    """Build a `Settings` instance without reading `.env` or the real
    process environment, so these tests are independent of whatever the
    developer's actual `.env` happens to contain."""

    return Settings(_env_file=None, **overrides)


def test_hnsw_ef_search_defaults_to_none() -> None:
    """Default must preserve today's behavior exactly: no SET LOCAL is ever
    issued unless explicitly configured (see DocumentRepository)."""

    assert _settings().hnsw_ef_search is None


@pytest.mark.parametrize("value", [1, 40, 100, 500, 1000])
def test_hnsw_ef_search_accepts_valid_values(value: int) -> None:
    assert _settings(hnsw_ef_search=value).hnsw_ef_search == value


@pytest.mark.parametrize("value", [0, -1, 1001, 5000])
def test_hnsw_ef_search_rejects_out_of_range_values(value: int) -> None:
    with pytest.raises(ValidationError, match="hnsw_ef_search must be between 1 and 1000"):
        _settings(hnsw_ef_search=value)


def test_hnsw_ef_search_rejects_non_integer_value() -> None:
    with pytest.raises(ValidationError):
        _settings(hnsw_ef_search="not-a-number")


# --- hnsw_iterative_scan ----------------------------------------------------


def test_hnsw_iterative_scan_defaults_to_none() -> None:
    assert _settings().hnsw_iterative_scan is None


@pytest.mark.parametrize("value", ["off", "relaxed_order", "strict_order"])
def test_hnsw_iterative_scan_accepts_valid_literal_values(value: str) -> None:
    assert _settings(hnsw_iterative_scan=value).hnsw_iterative_scan == value


def test_hnsw_iterative_scan_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        _settings(hnsw_iterative_scan="bogus")


# --- hnsw_max_scan_tuples ----------------------------------------------------


def test_hnsw_max_scan_tuples_defaults_to_none() -> None:
    assert _settings().hnsw_max_scan_tuples is None


@pytest.mark.parametrize("value", [1, 20000, 100000])
def test_hnsw_max_scan_tuples_accepts_positive_values(value: int) -> None:
    assert _settings(hnsw_max_scan_tuples=value).hnsw_max_scan_tuples == value


@pytest.mark.parametrize("value", [0, -1, -100])
def test_hnsw_max_scan_tuples_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValidationError, match="hnsw_max_scan_tuples must be a positive integer"):
        _settings(hnsw_max_scan_tuples=value)


# --- hnsw_scan_mem_multiplier ------------------------------------------------


def test_hnsw_scan_mem_multiplier_defaults_to_none() -> None:
    assert _settings().hnsw_scan_mem_multiplier is None


@pytest.mark.parametrize("value", [1, 2, 500, 1000])
def test_hnsw_scan_mem_multiplier_accepts_valid_values(value: int) -> None:
    assert _settings(hnsw_scan_mem_multiplier=value).hnsw_scan_mem_multiplier == value


@pytest.mark.parametrize("value", [0, -1, 1001, 5000])
def test_hnsw_scan_mem_multiplier_rejects_out_of_range_values(value: int) -> None:
    with pytest.raises(
        ValidationError,
        match="hnsw_scan_mem_multiplier must be between 1 and 1000",
    ):
        _settings(hnsw_scan_mem_multiplier=value)


# --- openrouter_timeout_seconds / openrouter_max_retries ---------------------


def test_openrouter_timeout_seconds_defaults_to_none() -> None:
    assert _settings().openrouter_timeout_seconds is None


@pytest.mark.parametrize("value", [0.1, 1.0, 30.0, 600.0])
def test_openrouter_timeout_seconds_accepts_positive_values(value: float) -> None:
    assert _settings(openrouter_timeout_seconds=value).openrouter_timeout_seconds == value


@pytest.mark.parametrize("value", [0, -1.0, -30.0])
def test_openrouter_timeout_seconds_rejects_non_positive_values(value: float) -> None:
    with pytest.raises(ValidationError, match="openrouter_timeout_seconds must be a positive"):
        _settings(openrouter_timeout_seconds=value)


def test_openrouter_generation_model_defaults_away_from_unstable_free_alias() -> None:
    """Regression: `"openrouter/free"` is an undocumented auto-routing
    alias, verified (via live, safe diagnostic calls) to resolve to a
    different underlying model on nearly every request -- including models
    entirely unsuited to producing a research answer (e.g. a content-safety
    classifier model). The default must be a single, concrete model id."""

    default_model = _settings().openrouter_generation_model

    assert default_model != "openrouter/free"
    assert default_model  # non-empty


def test_openrouter_generation_max_tokens_defaults_to_a_bounded_value() -> None:
    """Regression: an unbounded (`None`) token budget combined with a
    reasoning-style free-tier model risks very long generations or timeouts
    -- verified directly against a live model. The default must be a
    concrete, positive bound."""

    default_max_tokens = _settings().openrouter_generation_max_tokens

    assert default_max_tokens is not None
    assert default_max_tokens > 0


def test_openrouter_generation_model_default_when_nothing_configured() -> None:
    """Neither `OPENROUTER_GENERATION_MODEL` nor legacy `OPENROUTER_MODEL`
    set -- the compiled-in default must be used, unchanged."""

    settings = _settings()

    assert settings.openrouter_generation_model == "nvidia/nemotron-3-super-120b-a12b:free"


def test_legacy_openrouter_model_is_used_as_generation_fallback_when_unset() -> None:
    """Regression: a deployed `.env` using the legacy, pre-split
    `OPENROUTER_MODEL` variable name must not be silently ignored -- it
    should still configure the generation model when
    `OPENROUTER_GENERATION_MODEL` itself is not set."""

    settings = _settings(openrouter_model="legacy-vendor/legacy-model:free")

    assert settings.openrouter_generation_model == "legacy-vendor/legacy-model:free"


def test_explicit_generation_model_takes_precedence_over_legacy_variable() -> None:
    """`OPENROUTER_GENERATION_MODEL` must always win over the legacy
    `OPENROUTER_MODEL`, even when both are configured."""

    settings = _settings(
        openrouter_model="legacy-vendor/legacy-model:free",
        openrouter_generation_model="explicit-vendor/explicit-model:free",
    )

    assert settings.openrouter_generation_model == "explicit-vendor/explicit-model:free"


def test_explicit_generation_model_matching_default_is_not_overridden_by_legacy() -> None:
    """Edge case: an explicitly configured `OPENROUTER_GENERATION_MODEL`
    that happens to equal the compiled-in default must still be treated as
    explicitly set -- `model_fields_set`, not a value comparison, is what
    this precedence relies on."""

    default_model = _settings().openrouter_generation_model

    settings = _settings(
        openrouter_model="legacy-vendor/legacy-model:free",
        openrouter_generation_model=default_model,
    )

    assert settings.openrouter_generation_model == default_model


def test_legacy_fallback_logs_a_warning_identifying_the_mismatch(caplog) -> None:
    """The legacy fallback firing is a real, previously-silent condition
    that can cause generation to run on a model never reviewed for the
    generation workload -- this is exactly the scenario this test
    guards against regressing back to silent."""
    import logging

    with caplog.at_level(logging.WARNING, logger="app.config.settings"):
        _settings(openrouter_model="legacy-vendor/legacy-model:free")

    assert any(
        "OPENROUTER_GENERATION_MODEL is not set" in record.message for record in caplog.records
    )
    assert any("legacy-vendor/legacy-model:free" in record.message for record in caplog.records)


def test_no_warning_when_generation_model_is_explicitly_set(caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="app.config.settings"):
        _settings(
            openrouter_model="legacy-vendor/legacy-model:free",
            openrouter_generation_model="explicit-vendor/explicit-model:free",
        )

    assert not any(
        "OPENROUTER_GENERATION_MODEL is not set" in record.message for record in caplog.records
    )


def test_no_warning_when_legacy_variable_is_unset(caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="app.config.settings"):
        _settings()

    assert not any(
        "OPENROUTER_GENERATION_MODEL is not set" in record.message for record in caplog.records
    )


def test_legacy_openrouter_model_does_not_affect_decomposition_model() -> None:
    """`OPENROUTER_DECOMPOSITION_MODEL` must remain fully independent of the
    legacy `OPENROUTER_MODEL` fallback -- it keeps its own default/override
    regardless of whether the legacy variable is set."""

    settings = _settings(openrouter_model="legacy-vendor/legacy-model:free")

    assert settings.openrouter_decomposition_model == "openrouter/free"


def test_explicit_decomposition_model_is_unaffected_by_legacy_variable() -> None:
    settings = _settings(
        openrouter_model="legacy-vendor/legacy-model:free",
        openrouter_decomposition_model="decomposer-vendor/decomposer-model:free",
    )

    assert settings.openrouter_decomposition_model == "decomposer-vendor/decomposer-model:free"
    # The legacy variable still only ever reaches generation's model.
    assert settings.openrouter_generation_model == "legacy-vendor/legacy-model:free"


def test_openrouter_model_env_var_name_resolves_the_legacy_field(monkeypatch) -> None:
    """Confirms the legacy fallback is actually reachable via the real
    `OPENROUTER_MODEL` environment variable name (case-insensitive, matching
    every other setting's convention), not just via the Python keyword
    argument used by the other tests in this file."""

    monkeypatch.setenv("OPENROUTER_MODEL", "env-vendor/env-model:free")

    settings = Settings(_env_file=None)

    assert settings.openrouter_generation_model == "env-vendor/env-model:free"


def test_openrouter_max_retries_default_matches_sdk_default() -> None:
    assert _settings().openrouter_max_retries == 2


@pytest.mark.parametrize("value", [0, 1, 5, 10])
def test_openrouter_max_retries_accepts_non_negative_values(value: int) -> None:
    assert _settings(openrouter_max_retries=value).openrouter_max_retries == value


@pytest.mark.parametrize("value", [-1, -5])
def test_openrouter_max_retries_rejects_negative_values(value: int) -> None:
    with pytest.raises(ValidationError, match="openrouter_max_retries must be a non-negative"):
        _settings(openrouter_max_retries=value)


# --- per-task top_k / frequency_penalty / presence_penalty -------------------


@pytest.mark.parametrize(
    "field",
    ["openrouter_decomposition_top_k", "openrouter_generation_top_k"],
)
def test_top_k_defaults_to_none(field: str) -> None:
    assert getattr(_settings(), field) is None


@pytest.mark.parametrize(
    "field",
    ["openrouter_decomposition_top_k", "openrouter_generation_top_k"],
)
@pytest.mark.parametrize("value", [1, 10, 40, 100])
def test_top_k_accepts_positive_values(field: str, value: int) -> None:
    assert getattr(_settings(**{field: value}), field) == value


@pytest.mark.parametrize(
    "field",
    ["openrouter_decomposition_top_k", "openrouter_generation_top_k"],
)
@pytest.mark.parametrize("value", [0, -1, -10])
def test_top_k_rejects_non_positive_values(field: str, value: int) -> None:
    with pytest.raises(ValidationError, match=f"{field} must be a positive integer"):
        _settings(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "openrouter_decomposition_frequency_penalty",
        "openrouter_decomposition_presence_penalty",
        "openrouter_generation_frequency_penalty",
        "openrouter_generation_presence_penalty",
    ],
)
def test_penalty_fields_default_to_none(field: str) -> None:
    assert getattr(_settings(), field) is None


@pytest.mark.parametrize(
    "field",
    [
        "openrouter_decomposition_frequency_penalty",
        "openrouter_decomposition_presence_penalty",
        "openrouter_generation_frequency_penalty",
        "openrouter_generation_presence_penalty",
    ],
)
@pytest.mark.parametrize("value", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_penalty_fields_accept_valid_range(field: str, value: float) -> None:
    assert getattr(_settings(**{field: value}), field) == value


@pytest.mark.parametrize(
    "field",
    [
        "openrouter_decomposition_frequency_penalty",
        "openrouter_decomposition_presence_penalty",
        "openrouter_generation_frequency_penalty",
        "openrouter_generation_presence_penalty",
    ],
)
@pytest.mark.parametrize("value", [-2.1, 2.1, -5.0, 5.0])
def test_penalty_fields_reject_out_of_range_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match="Penalty values must be between -2.0 and 2.0"):
        _settings(**{field: value})


# --- retrieval/reranking operational controls --------------------------------


def test_retrieval_candidate_limit_floor_default_matches_previous_literal() -> None:
    assert _settings().retrieval_candidate_limit_floor == 50


@pytest.mark.parametrize("value", [1, 50, 500, 1000])
def test_retrieval_candidate_limit_floor_accepts_valid_values(value: int) -> None:
    assert _settings(retrieval_candidate_limit_floor=value).retrieval_candidate_limit_floor == value


@pytest.mark.parametrize("value", [0, -1, 1001])
def test_retrieval_candidate_limit_floor_rejects_out_of_range_values(value: int) -> None:
    with pytest.raises(ValidationError, match="retrieval_candidate_limit_floor must be between"):
        _settings(retrieval_candidate_limit_floor=value)


def test_retrieval_ranking_strategy_default_is_rrf() -> None:
    """Chosen based on the 25-question offline evaluation -- see the field's
    own comment in app.config.settings for the exact measured numbers."""
    assert _settings().retrieval_ranking_strategy == "rrf"


@pytest.mark.parametrize("value", ["rerank", "rrf"])
def test_retrieval_ranking_strategy_accepts_valid_values(value: str) -> None:
    assert _settings(retrieval_ranking_strategy=value).retrieval_ranking_strategy == value


def test_retrieval_ranking_strategy_rejects_invalid_value() -> None:
    """Enforced by the field's own Literal type -- no redundant custom
    validator, matching this file's existing Literal-typed settings
    (e.g. hnsw_iterative_scan)."""
    with pytest.raises(ValidationError, match="retrieval_ranking_strategy"):
        _settings(retrieval_ranking_strategy="not-a-real-strategy")


def test_retrieval_rrf_k_default_matches_the_standard_literature_constant() -> None:
    assert _settings().retrieval_rrf_k == 60


@pytest.mark.parametrize("value", [1, 60, 1000])
def test_retrieval_rrf_k_accepts_valid_values(value: int) -> None:
    assert _settings(retrieval_rrf_k=value).retrieval_rrf_k == value


@pytest.mark.parametrize("value", [0, -1])
def test_retrieval_rrf_k_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValidationError, match="retrieval_rrf_k must be greater than zero"):
        _settings(retrieval_rrf_k=value)


def test_research_ask_result_limit_default_matches_previous_literal() -> None:
    assert _settings().research_ask_result_limit == 10


@pytest.mark.parametrize("value", [1, 10, 50])
def test_research_ask_result_limit_accepts_valid_values(value: int) -> None:
    assert _settings(research_ask_result_limit=value).research_ask_result_limit == value


@pytest.mark.parametrize("value", [0, -1, 51])
def test_research_ask_result_limit_rejects_out_of_range_values(value: int) -> None:
    with pytest.raises(ValidationError, match="research_ask_result_limit must be between"):
        _settings(research_ask_result_limit=value)


def test_query_decomposition_max_sub_queries_default_matches_previous_constant() -> None:
    assert _settings().query_decomposition_max_sub_queries == 5


@pytest.mark.parametrize("value", [1, 5, 10, 20])
def test_query_decomposition_max_sub_queries_accepts_valid_values(value: int) -> None:
    assert (
        _settings(query_decomposition_max_sub_queries=value).query_decomposition_max_sub_queries
        == value
    )


@pytest.mark.parametrize("value", [0, -1, 21])
def test_query_decomposition_max_sub_queries_rejects_out_of_range_values(value: int) -> None:
    with pytest.raises(
        ValidationError,
        match="query_decomposition_max_sub_queries must be between",
    ):
        _settings(query_decomposition_max_sub_queries=value)
