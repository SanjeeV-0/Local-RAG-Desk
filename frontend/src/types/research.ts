/**
 * Type definitions strictly matching backend FastAPI contracts:
 * - app/api/research.py (POST /research/ask)
 * - app/api/retrieval.py (POST /retrieval/search)
 * - app/main.py (GET /health, GET /health/ready)
 */

export interface ResearchRequest {
  query: string;
  trace?: boolean;
  // Per-request override of the server's `query_router_enabled` default.
  // `null`/omitted means "use the configured application default"; `true`/
  // `false` controls the deterministic query router for THIS request only
  // (see app/retrieval/schemas.py's `RetrievalSearchRequest.router_enabled`
  // -- both endpoints share the identical contract). The frontend's Query
  // Router toggle always sends an explicit `true`/`false`, never `null`.
  router_enabled?: boolean | null;
}

export interface Source {
  document_id: string;
  chunk_id: string;
  section_id?: string | null;
  section_path: string | null;
  page_numbers: number[];
}

export interface CandidateResult {
  document_id: string;
  chunk_id: string;
  section_id?: string | null;
  section_path: string | null;
  page_numbers: number[];
  content: string;
  distance: number;
  rerank_score: number | null;
  matched_sub_queries: string[];
}

export interface TraceContextSource {
  document_id: string;
  chunk_id: string;
  section_id?: string | null;
  section_path: string | null;
  page_numbers: number[];
  content: string;
  distance: number;
  rerank_score: number | null;
  matched_sub_queries: string[];
}

export interface TraceContext {
  text: string;
  sources: TraceContextSource[];
}

/**
 * Debug trace for one retrieval operation. The retrieval pipeline always
 * decomposes `original_query` into 1-5 `sub_queries` (deduplicated, the
 * original query always included; decomposition only occurs for multiple
 * distinct information needs -- a single-intent query simply yields
 * `sub_queries == [original_query]`). Retrieval runs independently per
 * sub-query, candidates are merged/deduplicated across all of them
 * (`raw_candidate_count` before merge, `deduplicated_candidate_count`
 * after), and exactly one global reranker pass picks the final Top K from
 * the merged pool. The frontend only displays this -- it never re-derives
 * or re-runs any of this logic itself.
 */
export interface TraceData {
  query: string;
  original_query: string;
  sub_queries: string[];
  candidate_limit: number;
  raw_candidate_count: number;
  deduplicated_candidate_count: number;
  candidates: CandidateResult[];
  final_results: CandidateResult[];
  context: TraceContext | null;
  // True exactly when the deterministic query router was enabled for this
  // request AND actually decided to skip the decomposition call for
  // `original_query`; False whenever the router was disabled, or enabled
  // but decomposition still ran anyway (including when it ran and decided
  // a single sub-query -- i.e. `sub_queries.length === 1` does NOT by
  // itself mean the router bypassed anything). Reflects the backend's
  // actual executed decision (RetrievalTraceResponse.router_bypassed /
  // the research trace dict's "router_bypassed" key) -- never inferred
  // here from sub_queries length or query text. The decomposer, not the
  // router, is what produces `sub_queries` whenever decomposition runs.
  router_bypassed: boolean;
}

/**
 * Why trace data is or isn't present on a response. `TRACE_ENABLED` is a
 * server-side capability ceiling; request-level `trace` is the per-request
 * ask. The server only ever captures a trace when both are true
 * (`request.trace AND settings.trace_enabled`) -- neither ever controls
 * whether query decomposition or retrieval itself runs, only whether that
 * execution's trace is recorded and returned.
 */
export type TraceUnavailableReason = 'server_disabled';

/**
 * Always-present summary of what query decomposition actually did for one
 * request -- present on EVERY successful response from both
 * POST /research/ask and POST /retrieval/search, independent of Trace Mode
 * (unlike `TraceData`, which requires `trace_available`). The Query Router
 * controls whether decomposition runs; Trace Mode must never control
 * whether this outcome is visible.
 *
 * `status`:
 *  - "bypassed": the deterministic query router decided this query was
 *    confidently single-intent and skipped the decomposer entirely.
 *  - "decomposed": the decomposer ran and produced more than one retrieval
 *    sub-query -- shown in `sub_queries`, in their real, actual order.
 *  - "not_needed": the decomposer ran and decided a single sub-query was
 *    sufficient. Looks identical to "bypassed" in `sub_queries` (both are
 *    `[original_query]`) -- `status` is what distinguishes them.
 *  - "fallback": the decomposer's provider/parsing failed and retrieval
 *    fell back to the original query alone. `reason` is a short, safe,
 *    user-facing message in this case -- never a raw exception.
 *
 * Never inferred here from query text, subquery count, or router settings
 * alone -- always read verbatim from the backend's actual executed outcome.
 */
export interface DecompositionOutcome {
  original_query: string;
  sub_queries: string[];
  status: 'bypassed' | 'decomposed' | 'not_needed' | 'fallback';
  reason: string | null;
}

/**
 * Always-present summary of whether sensitive (PII-shaped) content was
 * redacted or withheld for this request -- present on EVERY successful
 * response from both POST /research/ask and POST /retrieval/search. A
 * response with `possibly_incomplete: true` must never be presented to the
 * user as a guaranteed-complete answer -- some source content or part of
 * the generated answer may have been withheld.
 *
 * Detection is rule-based (email, phone, dashed SSN, Luhn-validated card
 * numbers) and does not catch every form of sensitive information -- this
 * is a best-effort privacy safeguard, not a comprehensive one.
 */
export interface RedactionOutcome {
  redaction_applied: boolean;
  affected_source_count: number;
  answer_redacted: boolean;
  possibly_incomplete: boolean;
}

/**
 * Safe, category-classified provider failure reported in a partial
 * response -- see `app.generation.errors.ProviderUnavailableError`. Never
 * contains the raw provider response body or exception text.
 */
export interface GenerationError {
  category: string;
  retryable: boolean;
  message: string;
}

export interface ResearchResponse {
  // `null` exactly when `partial` is true: generation was rejected/failed
  // but retrieval already found safe, usable sources -- see
  // `generation_error` for why, and `sources`/`decomposition` for what was
  // still safely recovered.
  answer: string | null;
  model: string | null;
  sources: Source[];
  decomposition: DecompositionOutcome;
  redaction: RedactionOutcome;
  // True when generation could not produce an answer (a classified provider
  // failure) but retrieval/decomposition already succeeded -- `sources` and
  // `decomposition` are still a safe, complete result in that case. Always
  // present and `false` on a normal response.
  partial: boolean;
  // How `answer` was produced:
  //  - "generated": the normal path -- an LLM-generated answer.
  //  - "extractive_fallback": generation failed, but a local, deterministic
  //    extractive summary of the already-redacted retrieved sources was
  //    built instead (no provider call) -- `answer` is non-null but is NOT
  //    LLM-generated and must be labeled as such, never presented as a
  //    normal generated answer.
  //  - "unavailable": generation failed and no safe extractive summary
  //    could be built either -- `answer` is null.
  answer_method: 'generated' | 'extractive_fallback' | 'unavailable';
  generation_error?: GenerationError;
  trace_requested: boolean;
  trace_available: boolean;
  trace_unavailable_reason: TraceUnavailableReason | null;
  trace?: TraceData | null;
}

/**
 * Contracts for vector retrieval endpoint (POST /retrieval/search)
 */
export interface RetrievalSearchRequest {
  query: string;
  limit?: number; // 1 to 50, default 10
  document_id?: string | null;
  section_id?: string | null;
  // Per-request ask for trace capture. Combined with the server's
  // TRACE_ENABLED setting as request.trace AND settings.trace_enabled --
  // see `RetrievalSearchResponse` for how the server reports the outcome.
  trace?: boolean;
  // See `ResearchRequest.router_enabled` -- identical per-request override
  // contract, shared by both endpoints.
  router_enabled?: boolean | null;
  // Optional per-request override of the per-sub-query vector-search pool
  // size (server validates 1-1000, and rejects a value smaller than
  // `limit`). `null`/omitted means "use the server's own default"
  // (`max(limit, Settings.retrieval_candidate_limit_floor)`) -- the UI must
  // never invent or display a numeric default here itself; the actual
  // value used is only ever knowable from `RetrievalTraceResponse.candidate_limit`
  // once a request completes with tracing available.
  candidate_limit?: number | null;
}

export interface RetrievedChunkResponse {
  document_id: string;
  chunk_id: string;
  section_id: string | null;
  section_path: string | null;
  page_numbers: number[];
  content: string;
  distance: number;
  similarity: number;
  rerank_score: number | null;
  // Every sub-query (from `decomposition.sub_queries`, including the
  // original query when kept as one of them) whose vector search returned
  // this chunk. A single-entry `decomposition.sub_queries` means every
  // chunk's list is just that one entry -- nothing meaningful to filter by.
  matched_sub_queries: string[];
}

export interface RetrievalSearchResponse {
  results: RetrievedChunkResponse[];
  decomposition: DecompositionOutcome;
  redaction: RedactionOutcome;
  trace_requested: boolean;
  trace_available: boolean;
  trace_unavailable_reason: TraceUnavailableReason | null;
  trace: TraceData | null;
}

/**
 * System health contracts
 */
export interface BackendHealthResponse {
  status: string;
  environment?: string;
}

export interface BackendReadinessResponse {
  status: 'ready' | 'not_ready';
  checks: {
    database: boolean;
  };
}
