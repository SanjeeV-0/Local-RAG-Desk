# Single-User Precision Audit

**Scope of this document:** this re-checks the codebase specifically for a **single-user, upload-then-ask** use case, where the thing that matters most is *"when I upload a document and ask a question, is the answer precise and trustworthy?"* — not multi-tenant security, not internet-scale concurrency.

This changes the priority order a lot compared to a generic "production readiness" review:

- Auth, CORS, rate limiting, multi-tenant isolation — **mostly irrelevant** if this only ever runs on your own machine/network. Flagged but deprioritized.
- Upload fidelity, retrieval precision, and answer grounding — **the real priorities**, because these directly determine whether the answer you get back is correct.

All findings below are verified against the actual source in this repo (file:line cited), not assumed from docstrings or README claims.

---

## Priority 0 — Fix/verify before trusting any answer

These directly affect whether the system gives you a *correct* answer, not just *an* answer.

### 0.1 Your actual running config is using the lowest-effort free model for both generation and decomposition
- `.env:10` only sets `OPENROUTER_MODEL=openrouter/free`.
- `app/config/settings.py:467-468` — because `OPENROUTER_GENERATION_MODEL` is not set, generation silently falls back to this legacy `OPENROUTER_MODEL` value (`openrouter/free`), not the stronger compiled-in default (`nvidia/nemotron-3-super-120b-a12b:free`, `settings.py:284`).
- `app/config/settings.py:233` — decomposition's own default is also the generic `openrouter/free` (never touched by the legacy fallback, per the comment at `settings.py:453`).
- **Effect:** both the query-decomposition step and the final answer-generation step are currently running on OpenRouter's generic free router pick, not a specific, known-capable model. This is the single biggest lever on answer *quality* you're not currently using.
- **Action:** set `OPENROUTER_GENERATION_MODEL` and `OPENROUTER_DECOMPOSITION_MODEL` explicitly in `.env` to concrete models you've checked are good at instruction-following/structured output (decomposition needs reliable JSON; generation needs to follow "answer only from context + cite sources"). Since this is single-user, a few cents/request is not a real constraint — don't default to the free tier for precision-critical work.

### 0.2 A failed/garbled extraction can silently produce a "successful" document with ~no usable content
- No code anywhere in `app/ingestion/service.py` checks "did this document actually produce chunks?" before marking it `READY`.
- No API response field exposes chunk count either — confirmed by grep across `app/`, there is no `chunk_count`/`num_chunks` anywhere in schemas or responses.
- Combined with `app/ingestion/loaders/pdf.py` (pure `pypdf.extract_text()`, no OCR, no layout analysis): a **scanned/image-only PDF** ingests as `READY` with zero or near-zero chunks, and you have no signal that happened short of manually checking the retrieval trace.
- **Action (process, no code change needed today):** after uploading a PDF, run a query you know should hit that document and check the Retrieval Explorer trace (or query the `document_chunks` table directly) to confirm it actually produced a reasonable number of chunks. Treat "0 relevant results from a document I just uploaded" as a red flag to go check the raw extracted text, not as "the question wasn't in the doc."
- **Action (code, worth doing):** add a post-ingestion check that fails the version (or at least flags it) if `chunks_created == 0` or suspiciously low relative to document length.

### 0.3 No answer-grounding/citation verification — the model's citations are a convention, not a checked fact
- `app/generation/openrouter.py` prompts the model to cite `[Source N]` and not invent facts, but nothing in the pipeline verifies that a cited source actually supports the claim next to it, or that every factual claim has a citation at all.
- This is the main remaining source of imprecision once retrieval is working correctly: a good retrieval pass can still produce an answer that subtly misstates or extrapolates beyond what the sources say, and today you'd only catch that by reading the sources yourself.
- **Action:** when precision matters, open the Sources panel / retrieval trace and spot-check the cited chunks against the claim, especially for any numeric, comparative, or "according to X" statement. There is no automated faithfulness check to rely on instead.

### 0.4 Tables and multi-column layout are a real blind spot for your actual target corpus
- PDF tables are **not supported** at all — `pypdf.extract_text()` has no table structure, so a table in a PDF ingests as garbled flat text (confirmed; `.md` and `.docx` tables *are* handled correctly via pipe-table parsing).
- Multi-column academic-paper layouts (explicitly your target corpus per `README.md`) are a known weak point of `pypdf.extract_text()` — reading order across columns can get interleaved/garbled, with no detection or warning.
- **Action:** for any PDF with real data tables or a two-column academic layout, either (a) convert it to Markdown/docx first if feasible, or (b) manually verify the retrieved chunks for that section look coherent before trusting an answer that relies on it.

### 0.5 Retrieval is vector-only — exact technical terms can be missed
- Despite the README calling this "Hybrid Retrieval," there is no lexical/BM25/keyword path anywhere — `app/core/repositories/document.py`'s `search_similar_chunks` is pure pgvector cosine search, fused across decomposed sub-queries via RRF (`app/retrieval/rrf.py`), not fused with a keyword index.
- **Why this matters for precision on a technical corpus:** dense embeddings are good at semantic similarity but can under-rank chunks that are the literal, exact answer when the question uses a specific acronym, API name, parameter name, or version number the embedding model doesn't weight heavily (e.g. "HNSW `ef_search`", "KV cache", a specific library function name).
- **Action:** when asking about a specific named term/acronym/parameter, also try rephrasing the query to include the exact term verbatim, and check the trace to see if the expected chunk is even in the candidate pool before reranking. A real fix would be adding a Postgres full-text (`tsvector`) path and fusing it via the existing RRF code — RRF already supports fusing two rankings, it's just currently fed two vector-based rankings instead of vector+keyword.

---

## Priority 1 — Per-upload checks (workflow, not code)

Given 0.2 and 0.4 above, until the code gets a validation step, treat every upload with this checklist:

1. After upload, check the document's status is `READY` (not silently stuck or `FAILED`).
2. Run at least one question you know the answer to against the new document and confirm the retrieved sources actually come from it.
3. If the document has tables or a multi-column layout, open the Retrieval Explorer trace for a question about that content and visually confirm the chunk text is coherent, not garbled.
4. If a question about a document you just uploaded returns "no relevant information," don't assume the information isn't there — check chunk count/content first.

---

## Priority 2 — Config tuning available today (no code changes)

Since this is single-user (low query volume, no concurrency pressure), you can trade a bit of latency for meaningfully better recall/precision:

| Setting | Current default | Recommendation for single-user precision focus |
|---|---|---|
| `OPENROUTER_GENERATION_MODEL` | falls back to `openrouter/free` (see 0.1) | pin an explicit, capable instruction-following model |
| `OPENROUTER_DECOMPOSITION_MODEL` | `openrouter/free` default | pin an explicit model reliable at structured/JSON output |
| `RETRIEVAL_CANDIDATE_LIMIT_FLOOR` (`settings.py:343`) | `50` | raise (e.g. 100) — more candidates into the reranker costs you latency, not correctness, and you have no concurrency to protect |
| `RESEARCH_ASK_RESULT_LIMIT` (`settings.py:361`) | `10` | consider raising slightly if answers feel under-sourced |
| `QUERY_DECOMPOSITION_MAX_SUB_QUERIES` (`settings.py:376`) | `5` | fine as-is; raise only if you regularly ask multi-part questions |
| `HNSW_EF_SEARCH` (env, unset = pgvector default 40) | unset | raise (e.g. 100-200) for better recall; single-user query volume can easily absorb the extra query latency |
| `embedding_model` (`settings.py:74`) | `all-MiniLM-L6-v2` (384-dim, general-purpose) | a larger general-domain or technical-leaning sentence-transformers model (e.g. a `bge-base`/`e5-base` class model) will likely improve precision on technical content — **but this requires a full re-embed of every existing document and a DB column-width migration (`Vector(384)` in `app/core/models.py`), it's not a drop-in env change** |

---

## Priority 3 — Worth fixing regardless of single-user context

These aren't about multi-user security, they're plain correctness/hygiene issues that will bite *you* specifically:

- **`.gitignore:34` is a bare `tests/`.** Verified directly: any *new* test file you add today is silently untracked by git unless you `git add -f` it (confirmed with `git check-ignore -v`). The 118 existing tracked test files predate this rule. Fix the pattern (e.g. scope it to `tests/**/__pycache__/` or whatever it was meant to ignore) before you add more tests, or you'll quietly lose them.
- **Embedding/reranking calls block the event loop** (`app/retrieval/service.py:330`, `app/retrieval/cross_encoder.py:44` call synchronous `model.encode()`/`model.predict()` directly inside `async def`, no `asyncio.to_thread`). For single-user this mostly matters if you kick off an upload and then immediately ask a question — the question will wait behind the upload's embedding work since the whole process (including answering other requests) stalls during inference. Cheap, worthwhile fix even without a multi-user concern.
- **No upload size limit** (`app/api/documents.py:92`, `await file.read()` unbounded). Not a security issue for single-user, but there's nothing stopping an accidental huge-file upload from exhausting memory or hanging the process for a long time with no feedback.
- **Dead code**: `app/ingestion/chunker.py`, `structure_chunker.py`, `testing_similarity.py` are unused (verified by grep, zero real importers) — harmless but worth deleting so you don't accidentally edit the wrong chunker later.

---

## Priority 4 — Deprioritized for single-user local use

These were flagged in a general production-readiness review but matter much less here, *as long as this only runs on your own machine or a network only you can reach*:

- No authentication/authorization on any endpoint.
- No CORS policy, no rate limiting.
- No Docker/CI/CD pipeline.
- Local-disk-only file storage (fine for one user, one machine).
- PII redaction applied only at query/generation time, not at ingestion-time storage — relevant only if the documents you upload could contain other people's sensitive personal data at rest.

**Caveat:** if you ever expose this on a network reachable by anyone other than you (even just port-forwarding at home), revisit this section — at minimum put a single shared bearer-token check in front of it.

---

## Recommended action order

1. Pin explicit, capable models for `OPENROUTER_GENERATION_MODEL` and `OPENROUTER_DECOMPOSITION_MODEL` (0.1) — free, immediate quality win.
2. Adopt the per-upload verification checklist (Priority 1) until a chunk-count check exists in code.
3. Raise `HNSW_EF_SEARCH` and `RETRIEVAL_CANDIDATE_LIMIT_FLOOR` (Priority 2) — free, immediate recall win, no downside at single-user query volume.
4. Fix `.gitignore` (Priority 3) before adding any more tests.
5. For any document with tables or multi-column layout, manually spot-check ingestion quality (0.4) — this is a real content-fidelity gap, not a config knob.
6. Treat "no citation verification" (0.3) as a standing caveat — manually check sources for anything you'll rely on.
7. Longer-term, if you keep hitting precision issues on exact-term queries, consider adding a keyword/full-text fusion path (0.5) — the RRF plumbing to support it already exists.
