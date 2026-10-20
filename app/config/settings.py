"""Single source of truth for all runtime configuration.

Every tunable value that isn't a hard-coded algorithm constant (chunking
thresholds, decomposition temperature, etc. -- those live next to the code
that uses them) lives here. Values are read from the environment / `.env`
(case-insensitive) with the defaults below applied when unset. Note that
`.env.example` only documents a subset of these fields -- every field here
still works via its environment variable even if absent from that example
file.

Query decomposition (`app.retrieval.openrouter_decomposition`) and answer
generation (`app.generation.openrouter`) are configured independently --
each has its own `openrouter_{decomposition,generation}_*` model/sampling
settings below, rather than sharing one a single model. Both still share
the connection-level settings (`openrouter_api_key`, `openrouter_base_url`,
`openrouter_app_name`), since those describe the OpenRouter account/client,
not a per-role choice.
"""

import logging
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_positive_top_k(value: int | None, field_name: str) -> int | None:
    """Shared validator body for both tasks' `*_top_k` fields -- kept as a
    plain module function (rather than duplicated per-field logic) since
    the rule is identical: a configured `top_k` must be a positive integer,
    `None` (unconfigured) is always valid."""

    if value is None:
        return value

    if value < 1:
        raise ValueError(f"{field_name} must be a positive integer.")

    return value


def _validate_penalty_range(value: float | None) -> float | None:
    """Shared validator body for both tasks' frequency/presence penalty
    fields -- OpenAI's documented valid range is -2.0 to 2.0 inclusive;
    `None` (unconfigured) is always valid."""

    if value is None:
        return value

    if not (-2.0 <= value <= 2.0):
        raise ValueError("Penalty values must be between -2.0 and 2.0.")

    return value


_logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    app_name: str = "Local RAG Desk"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/research_agent"
    # Root directory for LocalFileStorage. Changing this does not move
    # already-stored files -- it only changes where new ones are written.
    storage_root: str = "data"
    # Output dimension is fixed in the schema (Vector(384) on
    # DocumentChunk.embedding). Swapping this for a model with a different
    # dimension requires a migration AND re-embedding every existing chunk;
    # even a same-dimension swap requires re-embedding, since vectors from
    # different models are not comparable.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Adjacent-paragraph cosine-similarity boundary used by
    # `shred_semantically` (app.ingestion.semantic_shredder) to decide
    # whether two consecutive prose units belong in the same chunk. Default
    # (0.7) is the exact value this project previously hard-coded in
    # `IngestionService._process_document` -- leaving this unset changes
    # nothing. Lower it to merge more adjacent paragraphs into fewer, larger
    # chunks (more surrounding context per chunk, coarser boundaries);
    # raise it to split more aggressively (smaller, more topically-precise
    # chunks). Only affects ingestion of documents processed AFTER the
    # change -- it is not retroactive against already-ingested chunks.
    semantic_similarity_threshold: float = 0.7

    @field_validator("semantic_similarity_threshold")
    @classmethod
    def _validate_semantic_similarity_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("semantic_similarity_threshold must be between 0.0 and 1.0.")

        return value
    # pgvector HNSW query-time search breadth (`hnsw.ef_search`). `None`
    # (the default) means "don't touch it" -- vector search runs with
    # whatever `hnsw.ef_search` the Postgres server itself defaults to
    # (pgvector's own built-in default is 40), preserving today's behavior
    # exactly for anyone who doesn't set this. When set, it is applied via
    # a per-transaction `SET LOCAL` immediately before each vector search
    # query (see `DocumentRepository.search_similar_chunks`) -- never a
    # bare session-level `SET`, since this application's connections are
    # pooled and reused across unrelated requests; `SET LOCAL` is
    # guaranteed by PostgreSQL to revert at the end of the transaction
    # (commit or rollback), so it can never leak into another request that
    # later reuses the same pooled physical connection. Valid range per
    # pgvector: 1-1000.
    hnsw_ef_search: int | None = None

    @field_validator("hnsw_ef_search")
    @classmethod
    def _validate_hnsw_ef_search(cls, value: int | None) -> int | None:
        if value is None:
            return value

        if not (1 <= value <= 1000):
            raise ValueError("hnsw_ef_search must be between 1 and 1000 (pgvector's valid range).")

        return value

    # pgvector's iterative index scan mode -- lets a search that would
    # otherwise return fewer than `limit` rows (because post-filtering, e.g.
    # this application's `document_id`/`section_id`/`max_distance` WHERE
    # clauses, discards HNSW candidates) keep scanning instead of returning
    # a short result. `None` (the default) means "don't touch it" -- pgvector
    # 0.8.6 (the version installed against this project's configured
    # database, confirmed via `SELECT extversion FROM pg_extension WHERE
    # extname='vector'`) itself defaults `hnsw.iterative_scan` to `"off"`.
    # Applied the same way as `hnsw_ef_search`: a validated `SET LOCAL`
    # inside the same transaction as the vector search query, which
    # PostgreSQL guarantees reverts at that transaction's end (commit or
    # rollback), so it can never leak across a pooled connection's reuse by
    # a later, unrelated request.
    hnsw_iterative_scan: Literal["off", "relaxed_order", "strict_order"] | None = None

    # Row cap for one iterative scan pass (only meaningful when
    # `hnsw_iterative_scan` is not "off"). `None` (the default) leaves
    # pgvector's own built-in default of 20000 untouched. Confirmed via
    # `SHOW hnsw.max_scan_tuples` against the installed pgvector 0.8.6.
    # Applied via the same per-transaction `SET LOCAL` mechanism.
    hnsw_max_scan_tuples: int | None = None

    @field_validator("hnsw_max_scan_tuples")
    @classmethod
    def _validate_hnsw_max_scan_tuples(cls, value: int | None) -> int | None:
        if value is None:
            return value

        if value < 1:
            raise ValueError("hnsw_max_scan_tuples must be a positive integer.")

        return value

    # Memory multiplier for an iterative scan's internal tuple buffer.
    # `None` (the default) leaves pgvector's own built-in default of 1
    # untouched. Confirmed via `SHOW hnsw.scan_mem_multiplier` against the
    # installed pgvector 0.8.6 (valid range 1-1000, enforced by Postgres
    # itself, re-validated here too since this value is inlined into SQL --
    # see `DocumentRepository.search_similar_chunks`). Applied via the same
    # per-transaction `SET LOCAL` mechanism.
    hnsw_scan_mem_multiplier: int | None = None

    @field_validator("hnsw_scan_mem_multiplier")
    @classmethod
    def _validate_hnsw_scan_mem_multiplier(cls, value: int | None) -> int | None:
        if value is None:
            return value

        if not (1 <= value <= 1000):
            raise ValueError("hnsw_scan_mem_multiplier must be between 1 and 1000.")

        return value

    # Server-side capability ceiling for retrieval trace capture. A client's
    # per-request `trace=true` can never force a trace when this is false --
    # see app.api.retrieval / app.api.research's `effective_trace` logic.
    # Never gates retrieval or query decomposition themselves.
    trace_enabled: bool = False

    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "development"
    # Required by BOTH the query-decomposition provider and the generation
    # provider (app.core.dependencies raises eagerly if this is unset) --
    # in practice this makes decomposition mandatory, not optional, since
    # the DI wiring always constructs a decomposer for the retrieval service.
    openrouter_api_key: str | None = None

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_name: str = "Local RAG Desk"

    # Legacy, single-model configuration name from before generation and
    # decomposition had independent model settings. Deliberately NOT read
    # directly by any provider -- it only ever reaches
    # `openrouter_generation_model` via `_apply_legacy_openrouter_model_fallback`
    # below, and only when `OPENROUTER_GENERATION_MODEL` itself is unset.
    # `OPENROUTER_DECOMPOSITION_MODEL` is never affected by this: it keeps
    # its own independent default regardless of whether this legacy variable
    # is set. New deployments should set `OPENROUTER_GENERATION_MODEL`
    # directly; this exists only so an existing `OPENROUTER_MODEL=...` in a
    # deployed `.env` keeps working rather than being silently ignored.
    openrouter_model: str | None = None

    # Shared OpenAI-compatible CLIENT settings -- both
    # OpenRouterQueryDecompositionProvider and OpenRouterGenerationProvider
    # construct their `AsyncOpenAI` client identically, so these describe
    # the connection/transport, not a per-task sampling choice (same
    # reasoning as openrouter_api_key/base_url/app_name above).
    #
    # `None` (the default) means "don't override the SDK's own default" --
    # the `openai` SDK (confirmed via `inspect.signature(AsyncOpenAI.
    # __init__)`, installed version 2.54.0) itself defaults `timeout` to a
    # library-internal value when not given at all (passing `None`
    # explicitly is different from omitting it -- see the provider code for
    # how this distinction is preserved).
    openrouter_timeout_seconds: float | None = None

    @field_validator("openrouter_timeout_seconds")
    @classmethod
    def _validate_openrouter_timeout_seconds(cls, value: float | None) -> float | None:
        if value is None:
            return value

        if value <= 0:
            raise ValueError("openrouter_timeout_seconds must be a positive number of seconds.")

        return value

    # Matches the `openai` SDK's own current default (`max_retries=2`)
    # exactly, so setting this explicitly changes nothing by default --
    # the SDK already implements its own retry/backoff internally for
    # retryable errors; this project does not add a second, custom retry
    # loop on top of it.
    openrouter_max_retries: int = 2

    @field_validator("openrouter_max_retries")
    @classmethod
    def _validate_openrouter_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("openrouter_max_retries must be a non-negative integer.")

        return value

    # Query decomposition LLM configuration -- independent of answer
    # generation's settings below (see module docstring): a change to one
    # task's model/sampling parameters never affects the other, and
    # `OpenRouterQueryDecompositionProvider`/`OpenRouterGenerationProvider`
    # each only ever receive their own `openrouter_{decomposition,
    # generation}_*` prefix's values (see `app.core.dependencies`).
    openrouter_decomposition_model: str = "openrouter/free"
    openrouter_decomposition_temperature: float = 0.0
    openrouter_decomposition_max_tokens: int | None = None
    openrouter_decomposition_top_p: float = 1.0
    openrouter_decomposition_response_format: dict[str, object] | None = None
    openrouter_decomposition_reasoning_effort: str | None = None
    # OpenRouter-specific passthrough, NOT a standard OpenAI API field --
    # sent via `extra_body={"top_k": ...}` (see
    # OpenRouterQueryDecompositionProvider.decompose), only when configured.
    # Not every upstream model honors it; an unsupported model simply
    # ignores the field rather than erroring (OpenRouter's own passthrough
    # contract), but this project sends it regardless of model choice since
    # there is no reliable way to know in advance which models support it.
    openrouter_decomposition_top_k: int | None = None
    # Standard OpenAI chat-completion fields, included only when configured
    # (same omit-when-None pattern as max_tokens/response_format/
    # reasoning_effort above) -- valid range per the OpenAI API: -2.0 to 2.0.
    openrouter_decomposition_frequency_penalty: float | None = None
    openrouter_decomposition_presence_penalty: float | None = None

    @field_validator("openrouter_decomposition_top_k")
    @classmethod
    def _validate_openrouter_decomposition_top_k(cls, value: int | None) -> int | None:
        return _validate_positive_top_k(value, "openrouter_decomposition_top_k")

    @field_validator(
        "openrouter_decomposition_frequency_penalty",
        "openrouter_decomposition_presence_penalty",
    )
    @classmethod
    def _validate_openrouter_decomposition_penalties(cls, value: float | None) -> float | None:
        return _validate_penalty_range(value)

    # Answer generation LLM configuration -- independent of query
    # decomposition's settings above; see the matching comment there.
    #
    # `openrouter_generation_model` is deliberately a single, concrete
    # OpenRouter model id, NOT the "openrouter/free" auto-routing alias --
    # live verification (safe diagnostic calls, no content sent) showed that
    # alias resolves to a DIFFERENT underlying free model on nearly every
    # request (observed across one session: general chat/MoE models, a
    # coding-specific agent, and even a 4B-parameter content-safety
    # CLASSIFIER model entirely unsuited to producing a research answer).
    # That non-determinism explains both inconsistent generation failures
    # (different sub-providers/models behind the alias can fail in
    # different ways) and inconsistent answer quality. Pin to one verified,
    # general-purpose, instruction-capable free model instead; override via
    # `OPENROUTER_GENERATION_MODEL` in the environment if a different model
    # is preferred. `OPENROUTER_MODEL` (legacy) is also honored as a
    # fallback when `OPENROUTER_GENERATION_MODEL` itself is not set -- see
    # `openrouter_model` and `_apply_legacy_openrouter_model_fallback` above.
    openrouter_generation_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_generation_temperature: float = 0.2
    # Bounded (not `None`/unbounded) -- some free-tier models (including the
    # pinned default above) are reasoning models that can run very long, or
    # even time out, with no cap; a bounded budget also leaves enough room
    # for a structured, multi-part answer without an indefinite wait.
    #
    # Confirmed root cause of truncated, incomplete answers (2026-10-18): for
    # a reasoning model, `max_tokens` is shared between the model's hidden
    # reasoning/thinking tokens and its visible answer -- the 2048 token
    # budget this value previously held was frequently exhausted by
    # reasoning alone on a multi-part question synthesizing 10 retrieved
    # sources, cutting the visible answer off mid-sentence after little more
    # than a source-by-source inventory. Raised to give a reasoning model
    # enough headroom to both think and still produce a complete, coherent
    # answer; paired with `openrouter_generation_reasoning_effort` below so
    # the reasoning portion itself doesn't re-consume the larger budget.
    openrouter_generation_max_tokens: int | None = 4096
    openrouter_generation_top_p: float = 1.0
    openrouter_generation_response_format: dict[str, object] | None = None
    # "low" caps how much of `max_tokens` a reasoning-capable model spends on
    # hidden reasoning before writing its visible answer -- see the
    # `max_tokens` comment above for why this matters together with it. A
    # non-reasoning model simply ignores this field (OpenRouter passes it
    # through only to models that support it).
    openrouter_generation_reasoning_effort: str | None = "low"
    # OpenRouter-specific passthrough -- see the matching decomposition
    # field's comment above; identical contract, independent value.
    openrouter_generation_top_k: int | None = None
    openrouter_generation_frequency_penalty: float | None = None
    openrouter_generation_presence_penalty: float | None = None

    @field_validator("openrouter_generation_top_k")
    @classmethod
    def _validate_openrouter_generation_top_k(cls, value: int | None) -> int | None:
        return _validate_positive_top_k(value, "openrouter_generation_top_k")

    @field_validator(
        "openrouter_generation_frequency_penalty",
        "openrouter_generation_presence_penalty",
    )
    @classmethod
    def _validate_openrouter_generation_penalties(cls, value: float | None) -> float | None:
        return _validate_penalty_range(value)

    # Retrieval/reranking operational controls -- exposing existing
    # hard-coded literals, not new algorithm behavior. Each default below
    # is the EXACT value the code previously hard-coded, so leaving these
    # unset changes nothing.
    #
    # Distinct from `hnsw_ef_search` (HNSW's own internal search breadth,
    # above) -- this is the APPLICATION-level floor on how many candidate
    # chunks are fetched from the database per sub-query before reranking
    # (see `RetrievalService.search`'s `candidate_limit` parameter, which a
    # caller can still override per-request; this only governs its
    # default). Reranking itself has no separate "input count" control --
    # it scores however many candidates this floor (or an explicit
    # `candidate_limit`) produced (see `CrossEncoderReranker.rerank`'s own
    # docstring); that is unchanged by this setting.
    retrieval_candidate_limit_floor: int = 50

    @field_validator("retrieval_candidate_limit_floor")
    @classmethod
    def _validate_retrieval_candidate_limit_floor(cls, value: int) -> int:
        if not (1 <= value <= 1000):
            raise ValueError("retrieval_candidate_limit_floor must be between 1 and 1000.")

        return value

    # Final number of results `POST /research/ask` asks RetrievalService
    # for -- distinct from BOTH the HNSW search breadth and the candidate
    # limit above: this is the post-rerank result COUNT returned to the
    # answer-generation step, not how broadly the database/HNSW search.
    # `POST /retrieval/search` already exposes an equivalent, independent,
    # per-request `limit` field (`RetrievalSearchRequest.limit`, 1-50) --
    # this setting is the server-side default for the endpoint that does
    # NOT accept a per-request override.
    research_ask_result_limit: int = 10

    @field_validator("research_ask_result_limit")
    @classmethod
    def _validate_research_ask_result_limit(cls, value: int) -> int:
        if not (1 <= value <= 50):
            raise ValueError("research_ask_result_limit must be between 1 and 50.")

        return value

    # Hard ceiling on how many sub-queries (including the original query)
    # QueryDecomposer.decompose will ever keep -- exposes the previously
    # hard-coded `app.retrieval.decomposition.MAX_SUB_QUERIES` constant.
    # Does not change decomposition's actual behavior/quality, only the
    # bound on how many of the LLM's candidate sub-queries are kept.
    query_decomposition_max_sub_queries: int = 5

    @field_validator("query_decomposition_max_sub_queries")
    @classmethod
    def _validate_query_decomposition_max_sub_queries(cls, value: int) -> int:
        if not (1 <= value <= 20):
            raise ValueError("query_decomposition_max_sub_queries must be between 1 and 20.")

        return value

    # When False (the default), RetrievalService always calls
    # QueryDecomposer for every query, preserving today's exact behavior.
    # When True, a deterministic, rule-based pre-filter
    # (app.retrieval.query_router.should_decompose) may skip the
    # decomposition LLM call for queries it is confident are single-intent.
    # Ambiguous cases still fall through to decomposition either way -- this
    # flag only ever removes decomposition calls, never adds them.
    query_router_enabled: bool = False

    # Selects how RetrievalService produces its final, limit-truncated
    # result order from the reranked candidate pool -- see
    # `app.retrieval.service`'s module docstring and `app.retrieval.rrf` for
    # the full explanation. "rerank": the cross-encoder reranker's own score
    # order alone (this project's PREVIOUS default, before this evaluation).
    # "rrf" (the current default): fuses that ranking with the pre-rerank
    # vector-distance ranking via Reciprocal Rank Fusion.
    #
    # Chosen as the default based on a 25-question offline evaluation
    # (tests/evaluation/offline_reranker_diagnostics.py + a direct call to
    # this module's own fuse_rankings, same corpus/limits used throughout
    # this project's retrieval evaluations): versus the then-current
    # "rerank"-only default, "rrf" improved every aggregate metric
    # (Recall@3 0.760->0.820, Recall@5 0.840->0.860, MRR 0.737->0.768) and
    # won in 2 of 3 question categories (semantic_retrieval, distractor_heavy),
    # with a per-question record of 7 improved / 4 regressed / 14 unchanged
    # relative to "rerank". Caveats, disclosed rather than hidden: n=25 is a
    # small sample; the smallest category (cross_section, n=4) REGRESSED
    # under "rrf" (MRR 0.750->0.583); and "rrf" did not clearly beat plain
    # vector-only search on MRR (0.772 vector-only vs 0.768 "rrf" -- a
    # negligible difference, and not the comparison this default decision
    # was based on, since vector-only was never the prior production
    # default). Fusion overhead is negligible (pure rank arithmetic, no
    # extra model inference). Revert to "rerank" by setting this one value
    # if further validation disagrees.
    retrieval_ranking_strategy: Literal["rerank", "rrf"] = "rrf"

    # The RRF smoothing constant (see app.retrieval.rrf.DEFAULT_RRF_K for the
    # formula and citation). Irrelevant when retrieval_ranking_strategy is
    # "rerank". 60 is the conventional literature default, not tuned against
    # this project's evaluation data.
    retrieval_rrf_k: int = 60

    @field_validator("retrieval_rrf_k")
    @classmethod
    def _validate_retrieval_rrf_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("retrieval_rrf_k must be greater than zero.")

        return value

    @model_validator(mode="after")
    def _apply_legacy_openrouter_model_fallback(self) -> "Settings":
        """Precedence for the answer-generation model:

        1. `OPENROUTER_GENERATION_MODEL`, if explicitly set -- always wins.
        2. `OPENROUTER_MODEL` (legacy), if set and (1) is not.
        3. The compiled-in default (`openrouter_generation_model`'s own
           field default) otherwise.

        `model_fields_set` (not just "is the value non-default") is what
        this checks `openrouter_generation_model` against -- it reflects
        whether `OPENROUTER_GENERATION_MODEL` was actually provided (by the
        environment, `.env`, or a direct keyword argument), so an explicit
        setting that happens to match the compiled-in default still counts
        as "explicitly set" and is never silently overridden by the legacy
        variable.

        `openrouter_decomposition_model` is never touched here -- it keeps
        its own independent default/override regardless of this legacy
        variable, exactly as before this fallback was added.

        Logs one safe warning (the model name only -- never a credential or
        other secret) whenever this fallback actually fires, so an operator
        can see in logs that generation is running on a value that was never
        explicitly reviewed for the generation workload specifically
        (longer context, explicit `max_tokens`, etc.) -- unlike before,
        where this silently happened with no observable signal at all. This
        does not change which model is selected; it only makes an existing,
        already-silent precedence rule diagnosable.
        """

        if "openrouter_generation_model" not in self.model_fields_set and self.openrouter_model:
            self.openrouter_generation_model = self.openrouter_model
            _logger.warning(
                "OPENROUTER_GENERATION_MODEL is not set; falling back to the "
                "legacy OPENROUTER_MODEL value (%r) for answer generation. "
                "This value may not have been reviewed for the generation "
                "workload (which sends a longer prompt and an explicit "
                "max_tokens, unlike lighter-weight calls such as query "
                "decomposition) -- set OPENROUTER_GENERATION_MODEL explicitly "
                "to silence this warning and confirm the intended model.",
                self.openrouter_generation_model,
            )

        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
