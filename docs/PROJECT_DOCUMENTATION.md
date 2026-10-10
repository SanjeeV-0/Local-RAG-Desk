# Production AI Research & Knowledge Agent — Project Documentation

> **Scope of this document.** This is a from-the-code audit of the repository as it exists
> today. It describes what is actually implemented, not the aspirational roadmap in the
> top-level `README.md` (which describes a multi-phase "V0 → V7" vision). Where the two
> disagree, this document follows the code. Anything that is not wired into a running code
> path is explicitly labeled as such rather than described as working.
>
> Every claim below was verified by reading the referenced source file. Where a value is
> hard-coded, it is labeled "hard-coded". Where a value comes from `Settings`, the
> environment variable and Python field name are given together. No model name, limit,
> temperature, dimension, or default in this document is inferred or guessed.

---

## 1. Project Overview

The **Production AI Research & Knowledge Agent** is a backend (FastAPI) + frontend (React/
Vite) application that lets a user:

1. **Ingest documents** (Markdown, PDF, `.docx`) into a managed corpus, with versioning,
   retry of failed ingestions, and the ability to promote any previously-READY version back
   to "current".
2. **Ask natural-language questions** against that corpus and get an LLM-generated answer
   grounded in retrieved source chunks ("Research & Synthesis" / RAG).
3. **Inspect retrieval mechanics directly** via a debugging UI ("Retrieval Explorer") that
   shows query decomposition, the candidate pool, deduplication counts, and reranked results
   without going through generation.

### Major technologies

| Concern | Technology | Where |
|---|---|---|
| API server | FastAPI | `app/main.py`, `app/api/*` |
| Database | PostgreSQL | `app/core/database.py`, `alembic/` |
| Vector search | pgvector (cosine distance, HNSW index) | `app/core/models.py`, `alembic/versions/86b2f3d39bfd_*` |
| ORM | SQLAlchemy 2.0 (async) | `app/core/models.py`, `app/core/repositories/*` |
| Migrations | Alembic | `alembic/versions/*` |
| Embeddings | `sentence-transformers` (local, CPU/GPU) | `app/embeddings/sentence_transformer.py` |
| Reranking | `sentence-transformers` `CrossEncoder` | `app/retrieval/cross_encoder.py` |
| Query decomposition LLM | OpenRouter (OpenAI-compatible API, via `openai` SDK) | `app/retrieval/openrouter_decomposition.py` |
| Generation LLM | OpenRouter (OpenAI-compatible API, via `openai` SDK) | `app/generation/openrouter.py` |
| Observability | Langfuse (optional) | `app/observability/langfuse.py` |
| Frontend | React 18 + TypeScript + Vite | `frontend/src/*` |

### High-level architecture (actual, current)

```text
                              ┌────────────────────────┐
                              │        Frontend         │
                              │  (React / Vite / TS)    │
                              │                          │
                              │  Document Library   ─┐   │
                              │  Research & Synthesis ┼──┼──► fetch() JSON/multipart
                              │  Retrieval Explorer  ─┘   │
                              └───────────┬────────────┘
                                          │ HTTP
                                          ▼
                              ┌────────────────────────┐
                              │        FastAPI           │
                              │  app/api/documents.py    │
                              │  app/api/retrieval.py    │
                              │  app/api/research.py     │
                              └──────┬───────┬──────────┘
                                     │       │
                 ┌───────────────────┘       └─────────────────────┐
                 ▼                                                  ▼
   ┌──────────────────────────┐                       ┌──────────────────────────────┐
   │  Document management      │                       │  Retrieval / RAG              │
   │  IngestionService          │                       │  RetrievalService              │
   │  DocumentService            │                       │   ├─ QueryDecomposer          │
   │  DocumentDeletionService    │                       │   │   └─ OpenRouter (LLM)      │
   │  DocumentRepository          │                       │   ├─ SentenceTransformer       │
   │  StoredFileRepository         │                       │   │   (embeddings)            │
   │  LocalFileStorage               │                       │   ├─ pgvector similarity     │
   └──────────────┬───────────────┘                       │   └─ CrossEncoderReranker      │
                  │                                        │  GenerationService              │
                  │                                        │   └─ OpenRouter (LLM)           │
                  ▼                                        └──────────────┬─────────────────┘
      ┌─────────────────────────┐                                        │
      │ PostgreSQL + pgvector     │◄───────────────────────────────────────┘
      │  documents, stored_files,  │
      │  document_pages, sections,   │
      │  chunks (+ embedding column)  │
      └───────────────────────────┘
                  ▲
                  │ (optional, both paths)
      ┌────────────────────────┐
      │       Langfuse            │  (observability only — never gates behavior)
      └────────────────────────┘
```

### What each top-level component owns (and does **not** own)

| Component | Owns | Does NOT own |
|---|---|---|
| `app/api/*` | HTTP contract: request validation, status codes, translating service exceptions to HTTP errors | Business logic, persistence, embeddings, LLM calls |
| `app/core/services/*` | Document/version lifecycle transitions, transaction boundaries | HTTP concerns, physical file I/O (delegates to `FileStorage`) |
| `app/core/repositories/*` | All SQL (SQLAlchemy Core/ORM queries) | Business rules about *when* to call a query |
| `app/ingestion/*` | Turning raw file bytes into pages → sections → chunks | Storage of the file itself, DB transaction commits (caller decides) |
| `app/storage/*` | Physical byte storage/retrieval/deletion, path-traversal safety | Database metadata about files |
| `app/embeddings/*` | Turning text into vectors | Deciding what gets embedded, or how vectors are queried |
| `app/retrieval/*` | Query decomposition, candidate retrieval orchestration, merge/dedup, reranking, trace capture | Generation/answer synthesis, HTTP contract |
| `app/generation/*` | Context assembly from retrieved chunks, calling the generation LLM | Retrieval itself |
| `app/observability/langfuse.py` | Wrapping select operations in Langfuse spans when configured | Being a source of truth for retrieval correctness (see §16) |
| `frontend/src/*` | Presentation, request construction, response rendering | Retrieval/decomposition/reranking logic (explicitly never re-implemented client-side) |

---

## 2. Repository Structure

```text
app/
  main.py                  FastAPI app construction, router registration, /health endpoints
  config/
    settings.py             Pydantic Settings — the single source of all configuration
  core/
    database.py              Async SQLAlchemy engine/session factory
    database_health.py        SELECT 1 liveness check for GET /health/ready
    logging.py                 stdlib logging configuration
    models.py                   All SQLAlchemy ORM models (Document, StoredFile, pages, sections, chunks, ...)
    dependencies.py             FastAPI dependency-injection wiring (the "composition root")
    repositories/
      document.py                DocumentRepository — all Document/page/section/chunk SQL
      stored_file.py               StoredFileRepository — StoredFile SQL
    services/
      document.py                 DocumentService — version lifecycle state machine
      document_deletion.py          DocumentDeletionService — DB+file deletion coordination
  api/
    documents.py               POST/GET/DELETE /documents* — document management HTTP boundary
    retrieval.py                 POST /retrieval/search — debug/engineering retrieval endpoint
    research.py                   POST /research/ask — RAG question-answering endpoint
  ingestion/
    service.py                    IngestionService — orchestrates the whole ingestion pipeline
    schemas.py                     Pydantic I/O schemas for ingestion + the documents API
    normalizer.py                   Content normalization + SHA-256 content hashing
    loaders/                         DocumentLoader implementations (markdown, pdf, docx) + resolver
    structure_extractor.py             Markdown-heading-aware structural parsing
    structure.py                        StructuralUnit/TableData/UnitType dataclasses
    section_builder.py                   Builds a section hierarchy from structural units
    section_service.py                    Persists DocumentSection rows
    section_map.py                         In-memory section-path → section-id map
    semantic_shredder.py                    Embedding-similarity-based paragraph grouping
    size_guard.py                            Final token-budget chunk splitting
    chunk_service.py                          Persists DocumentChunk rows + embeddings + page mappings
    similarity.py                              cosine_similarity() helper used by semantic_shredder
    table_chunker.py, table_serializer.py       Table splitting/Markdown serialization — ACTIVE
                                                 (called from structure_extractor.py and size_guard.py)
    chunker.py, structure_chunker.py,          NOT part of the active ingestion pipeline —
    testing_similarity.py                      see §20 "Known Limitations"
  storage/
    interface.py                 FileStorage abstract base class
    local.py                      LocalFileStorage — filesystem implementation
    keys.py                        build_storage_key() — immutable storage-key format
  embeddings/
    provider.py                   EmbeddingProvider abstract base class
    sentence_transformer.py         SentenceTransformerEmbeddingProvider
  retrieval/
    service.py                    RetrievalService — the whole retrieval pipeline
    models.py                      RetrievedChunk dataclass
    trace.py                        RetrievalTrace/RetrievalTraceCandidate/RetrievalTraceContext dataclasses
    schemas.py                      Pydantic request/response schemas for POST /retrieval/search
    query_decomposer.py              QueryDecomposer — normalizes/validates LLM decomposition output
    decomposition.py                  QueryDecompositionResult, QueryDecompositionProvider protocol, MAX_SUB_QUERIES
    openrouter_decomposition.py        OpenRouterQueryDecompositionProvider — the LLM call for decomposition
    reranker.py                        Reranker protocol
    cross_encoder.py                    CrossEncoderReranker — the actual reranker implementation
  generation/
    generation_service.py          GenerationService — wraps a GenerationProvider with Langfuse
    service.py                      GenerationProvider protocol + GenerationResult dataclass
    openrouter.py                    OpenRouterGenerationProvider — the LLM call for answer generation
    context.py                       ContextAssembler — turns retrieved chunks into prompt context
  observability/
    langfuse.py                    get_langfuse() — lazily-constructed, settings-gated Langfuse client
  agents/, evaluation/, llm/, memory/   Empty directories — no files exist yet (see §20)

alembic/
  versions/*.py               10 migrations, from initial `documents` table through document
                               versioning, HNSW index, and nullable chunk sections (see §7)

tests/
  unit/                       Pure-function/class tests, no DB (chunking, decomposition, trace shapes, ...)
  integration/                Tests that hit the real configured PostgreSQL database, plus FastAPI
                               contract tests using fakes (see §23)
  evaluation/                 Retrieval quality metrics (recall@K, MRR@K) — dataset is a placeholder (see §20)
  conftest.py                 Windows asyncio policy + a session-end DB/storage cleanup backstop

frontend/
  src/
    api/                      documents.ts, research.ts — the only two modules allowed to call fetch()
    types/                     document.ts, research.ts — TypeScript types mirroring the Pydantic schemas
    components/                 UploadDocumentModal, DocumentWorkspace, RetrievalWorkspace, TracePanel,
                                 AnswerPanel, SourcesPanel, SourceCard, Header, StatusBadge, ConfirmModal,
                                 LoadingOverlay, ResearchInput, DownloadButton
    pages/ResearchPage.tsx         Top-level page: owns tab state and the global Trace Mode toggle
    App.tsx                        Renders ResearchPage
  package.json                  Scripts: dev, build (tsc && vite build), lint (tsc --noEmit), preview

docs/
  PROJECT_DOCUMENTATION.md    This file.
  ENGINEERING_NOTES/           Informal running engineering log (pre-dates this document; aspirational
                               in places — this document is the authoritative current-state reference)

docker/                       Empty directory — no Dockerfile/compose file currently exists
.env.example                  Documents only a subset of settings — see §9 for the full list
pyproject.toml                 Python dependencies, pytest config, Ruff config
```

---

## 3. End-to-End Document Ingestion Pipeline

Entry point: `POST /documents` in `app/api/documents.py`, handled by
`IngestionService.ingest_file` in `app/ingestion/service.py`.

```text
multipart upload (file, document_type, title?, source?, logical_document_id?)
  │
  ▼
app/api/documents.py: upload_document()
  - reads file bytes, rejects empty files (422)
  - resolves a DocumentLoader from the filename extension (422 if unsupported)
  - writes bytes to a temp file (tempfile.NamedTemporaryFile)
  │
  ▼
IngestionService.ingest_file(path, loader, title, document_type, logical_document_id, source, original_filename)
  1. loader.load(path)                        → list[LoadedPage]  (page-level text extraction)
  2. combined_content = "\n\n".join(pages)
  3. DocumentService.ingest_document(DocumentInput(...), logical_document_id)
       - calculate_content_hash(combined_content)         (SHA-256 of whitespace-normalized text)
       - DocumentRepository.get_by_logical_and_content_hash(logical_document_id, content_hash)
       - if a Document row already exists with this exact (logical_document_id, content_hash):
             return it unchanged, created=False   ← IDEMPOTENCY (see §5)
       - else: create a new Document row, version_number = latest_version_number + 1,
               is_current=False, status=UPLOADED
  4. if not created: return the existing Document as-is (no new file, no reprocessing)
  5. file_hash = sha256(raw file bytes)             (distinct from the content_hash above — see §6)
  6. storage_key = build_storage_key(logical_document_id, document.id, file_id)
  7. FileStorage.store(file_bytes, storage_key)      ← physical write happens BEFORE the DB row
  8. create StoredFile row, session.commit()
       - on any exception here: session.rollback(), then best-effort delete the just-written
         physical file, then re-raise (no orphaned DB row pointing at a missing/invalid file)
  9. DocumentService.mark_processing(document); session.commit()
       - status → PROCESSING, processing_attempt += 1, processing_started_at = now()
  10. IngestionService._process_document(document, pages):
        a. StructureExtractor.extract(pages)                 → list[StructuralUnit]
        b. SectionBuilder.build(units)                         → list[SectionNode]
        c. SectionService.persist_sections(document.id, nodes)   → SectionMap (path → DocumentSection.id)
        d. shred_semantically(units, embedding_provider, threshold=0.7)   → list[SemanticUnit]
        e. apply_size_guard(semantic_units, section_map, max_tokens=500)   → list[ChildChunk]
        f. ChunkService.persist_chunks(document.id, page_ids, chunks)
             - embeds ALL chunk texts in one embedding_provider.embed_batch() call
             - persists one DocumentChunk row per chunk, with its embedding vector
             - persists a ChunkPageMap row per (chunk, source page) pair
        g. DocumentService.mark_ready(document); session.commit()
             - demotes the previous current version (if any) to is_current=False
             - sets this document: status=READY, is_current=True, processing_completed_at=now()
        h. on ANY exception in (a)-(g):
             - session.rollback()
             - re-fetch the Document row (it may have been mutated before the failing step)
             - DocumentService.mark_failed(document, str(exc)); session.commit()
                 status=FAILED, is_current=False, failed_at=now(), last_error=str(exc)
             - the exception object gets `.failed_document` attached (the persisted FAILED row)
               so the HTTP layer can report it as a normal 201 response, not a 500 (see below)
             - re-raise
  11. return the final Document (READY) or propagate the exception
```

### Failure behavior (verified in `app/ingestion/service.py` and `app/api/documents.py`)

- **File hashing happens before any DB row for a *new version's physical file* exists.**
  `_process_document` failures (extraction/chunking/embedding) **do not touch StoredFile or
  the physical file at all** — only pages/sections/chunks and the `Document` row's status are
  rolled back and then marked FAILED. The original uploaded file remains on disk and
  `StoredFile` remains valid, so a retry can reuse both.
- If `ingest_file` raises and the exception carries `.failed_document` (i.e. something was
  already persisted as FAILED), `app/api/documents.py::upload_document` treats this as a
  **normal, trackable outcome and returns HTTP 201** with that FAILED version's
  `DocumentVersionResponse` — not a 500. A FAILED version is first-class, retryable state, not
  a request error.
- If nothing was ever persisted (e.g. the loader itself threw before any `Document` row was
  created), the exception is **not** swallowed — it propagates as a real error.
- The temp file created for the upload is always deleted in a `finally` block, regardless of
  outcome.

---

## 4. Document Lifecycle

`DocumentStatus` (`app/core/models.py`): `UPLOADED`, `PROCESSING`, `READY`, `FAILED`.

```text
POST /documents (new content)
        │
        ▼
    UPLOADED  ──mark_processing()──►  PROCESSING ──mark_ready()──►  READY
                                           │
                                           └──mark_failed()──►  FAILED ──retry──► PROCESSING (loop)
```

Transitions are implemented in `app/core/services/document.py` (`DocumentService`):

| Method | Effect on the `Document` row |
|---|---|
| `ingest_document(...)` | Creates a new row at `UPLOADED`, or returns an existing row unchanged if content already exists for this logical document (idempotent — see §5) |
| `mark_processing(document)` | `status=PROCESSING`; `processing_attempt += 1`; `processing_started_at=now()`; clears `processing_completed_at`, `failed_at`, `last_error` |
| `mark_ready(document)` | Demotes the logical document's previous current version (if different); sets this row `status=READY`, `is_current=True`, `processing_completed_at=now()`; clears `failed_at`/`last_error` |
| `mark_failed(document, error)` | `status=FAILED`; `is_current=False`; `processing_completed_at=now()`; `failed_at=now()`; `last_error=error` |

`processing_attempt` counts **real processing executions**, not retry requests — it is
incremented only inside `mark_processing`, so a request that is rejected before processing
starts (e.g. retrying a non-FAILED version) never inflates it.

### Retry (`POST /documents/{logical_document_id}/versions/{version_id}/retry`)

Implemented by `IngestionService.retry_document` (`app/ingestion/service.py`):

1. Look up the `Document` row by `version_id`; raise `ValueError` if missing.
2. **Raise `ValueError` unless `status == FAILED`** — only FAILED versions may be retried (the
   route maps this to HTTP 409).
3. Look up the existing `StoredFile` for this version (raise `ValueError` if missing — this
   should not happen since FAILED versions always retain their file, see §6).
4. Resolve the loader from the StoredFile's **originally recorded filename** (not a new
   upload) if no loader is explicitly passed.
5. Re-read the **same physical file** via `FileStorage.retrieve(storage_key)`.
6. Run the same `mark_processing → _process_document` sequence as a fresh upload, reusing the
   same `Document` row and `StoredFile` row — **no new version, no new storage key, no new
   file write**.

If the retry fails again, the same `Document` row is marked FAILED again (with an incremented
`processing_attempt` and a fresh `last_error`) — this is indistinguishable, from the lifecycle
state machine's point of view, from a first-time ingestion failure.

---

## 5. Document Versions

### Identifiers (do not confuse these two)

- **`logical_document_id`** (`Document.logical_document_id`) — stable identity across every
  version of "the same document". Used by routes that operate on the whole document (list,
  get, list-versions, delete-everything).
- **`id` / "version id"** (`Document.id`, the primary key) — one specific ingested attempt.
  Used by routes that operate on one version directly (retry, set-current, delete-one-version).

Every route in `app/api/documents.py` is explicit about which of the two it takes — there is
no overloaded `{document_id}` path parameter anywhere in this API.

### Other version fields

| Field | Meaning |
|---|---|
| `content_hash` | SHA-256 of whitespace-normalized **extracted document content** (`app/ingestion/normalizer.py`). Drives idempotency (below). |
| `version_number` | `1, 2, 3, ...` per `logical_document_id`, assigned as `get_latest_version_number() + 1` at creation time. Never reused or renumbered. |
| `is_current` | Exactly 0 or 1 row per `logical_document_id` may have this `True` at any time — enforced by a **partial unique index** at the database level (`ix_documents_current_version`, `WHERE is_current = true`; see §7). |

### Idempotency

`DocumentService.ingest_document` looks up `(logical_document_id, content_hash)` via
`DocumentRepository.get_by_logical_and_content_hash`. If a row already exists for that exact
pair, the **existing row is returned unchanged** (`created=False`) — no new version, no new
file write, no reprocessing, regardless of version status (even a PROCESSING or FAILED
version's row is returned as-is if its content_hash matches). This also means uploading the
same byte-identical file again while an earlier upload is still PROCESSING returns that same
in-flight row rather than racing a second ingestion.

Uploading **different content** under the same `logical_document_id` always creates a new
`Document` row with the next `version_number`, `is_current=False`, `status=UPLOADED` — it does
not touch the existing current version until/unless the new version reaches `READY`.

### Current-version switching

Two ways a version becomes current:

1. **Automatically**, when a new version finishes processing successfully
   (`DocumentService.mark_ready`), or when the current version is deleted and a replacement is
   promoted (`DocumentService.delete_version` promotes the newest remaining `READY` version —
   never a FAILED/PROCESSING one; if none is READY, the logical document is left with **no**
   current version).
2. **Manually**, via `POST /documents/{logical_document_id}/versions/{version_id}/set-current`
   → `DocumentService.set_current` (`app/core/services/document.py`):
   - **Rejects (raises `ValueError`, mapped to HTTP 409) any version whose `status != READY`.**
     PROCESSING and FAILED versions can never become current.
   - **Idempotent**: calling it on an already-current READY version is a no-op success.
   - Otherwise: demotes the logical document's current version (if different) to
     `is_current=False`, flushes that, then sets the target to `is_current=True`, flushes that,
     then commits **both changes in one transaction**. The two sequential flushes (demote,
     then promote) inside one transaction mean the partial unique index is never violated
     mid-operation, and a failure between them rolls back to the original, single-current
     state.
   - Performs **no file operations and no vector/chunk operations** — it is purely an
     `is_current` flag move.

### Historical versions remain fully stored

Switching current (or uploading a new version) never deletes a previous version's
`DocumentPage`, `DocumentSection`, `DocumentChunk`, or embedding rows. They remain in
PostgreSQL exactly as ingested. The **only** thing that changes which version's chunks are
retrievable is the `is_current` flag, because `DocumentRepository.search_similar_chunks`
filters on `Document.status == READY AND Document.is_current.is_(True)` (see §12). This is
why a historical version can be made current again later and its original chunks/embeddings
are immediately searchable again, unchanged.

### Example

```text
Logical Document A
  v1 → READY, is_current=True     (searchable)
  v2 → READY, is_current=False    (stored, NOT searchable)
  v3 → FAILED                      (stored, NOT searchable, retry-eligible)

POST /documents/A/versions/v2-id/set-current
  → v1.is_current = False
  → v2.is_current = True
  → v2's chunks (which were never deleted) become immediately searchable
  → v1's chunks remain in the database but are now excluded from retrieval
  → v3 is untouched (still FAILED); it did not participate in current-version switching at all
```

---

## 6. File Storage

### `FileStorage` abstraction (`app/storage/interface.py`)

An `ABC` with four async methods: `store(content, storage_key)`, `retrieve(storage_key)`,
`delete(storage_key)`, `exists(storage_key)`. Only one implementation exists:
`LocalFileStorage` (`app/storage/local.py`), writing under `Settings.storage_root`
(default `"data"`, env `STORAGE_ROOT`).

### Storage key format (`app/storage/keys.py`)

```python
f"documents/{logical_document_id}/{document_version_id}/{file_id}"
```

This key is **generated independently of the user's original filename** — the user-supplied
filename is stored only as metadata (`StoredFile.original_filename`), never used to build a
filesystem path. This is deliberate: it means filenames with special characters, duplicate
names across uploads, or anything else attacker/user-controlled in the filename never reach
the filesystem path, and `logical_document_id`/`document_version_id`/`file_id` are
server-generated UUIDs.

### Path-traversal protection (`LocalFileStorage._resolve_path`)

```python
path = (self.root / storage_key).resolve()
path.relative_to(self.root)  # raises ValueError if path escaped root
```

Every storage operation resolves the key under `self.root` and verifies the resolved
absolute path is still inside it, raising `ValueError("storage_key resolves outside the
storage root")` otherwise. Since storage keys are always built by `build_storage_key()` from
server-generated UUIDs (never from user input directly), this is primarily defense in depth
against a malformed key reaching this layer, not a mitigation for a specific attack
observed in this codebase.

### `StoredFile` model (`app/core/models.py`)

One row per `Document` version that has a physical file (`document_id` is `unique=True` — a
version has at most one StoredFile, ever). Fields: `id`, `document_id`, `original_filename`,
`content_hash` (SHA-256 of the **raw file bytes** — distinct from `Document.content_hash`,
which hashes **extracted text**), `size_bytes`, `storage_key` (`unique=True`), `created_at`.

### Ordering and consistency guarantees (verified in code)

- **Upload**: the physical file is written (`FileStorage.store`) **before** the `StoredFile`
  row is committed. If the DB commit then fails, the code explicitly rolls back the session
  and **best-effort deletes the just-written physical file** (failure to delete it is silently
  swallowed so it doesn't mask the original DB error) — see `IngestionService.ingest_file`'s
  `try/except` around the `StoredFile` creation.
- **Version deletion** (`DocumentDeletionService.delete_version`): the `StoredFile` row is
  deleted first (deleting it before `Document` avoids a possible FK ordering issue), then the
  `Document` row, then **`session.commit()`**, and **only after that commit succeeds** is the
  physical file deleted via `FileStorage.delete(storage_key)`. If that final physical deletion
  fails, it propagates as a real exception — **the DB is already committed and consistent, but
  the physical file would be orphaned on disk.** There is no reconciliation/cleanup job for
  this case in the current codebase (see §20).
- **Logical document deletion** (`DocumentDeletionService.delete_logical_document`): same
  DB-commit-then-file-delete ordering, applied to every version's `StoredFile` and `Document`
  row at once (all DB deletes happen, then one `commit()`, then all physical files are
  deleted).
- **Failed versions retain their files.** A FAILED version's `StoredFile` and physical file
  are never deleted by the ingestion pipeline itself — only an explicit version/logical-document
  delete call removes them. This is what makes retry possible without re-uploading.

---

## 7. Database Model

All models live in `app/core/models.py`. Six tables:

```text
Document  (1) ───< StoredFile        (1:0..1, document_id UNIQUE)
Document  (1) ───< DocumentPage      (1:N, ON DELETE CASCADE)
Document  (1) ───< DocumentSection   (1:N, ON DELETE CASCADE, self-referential parent_section_id)
Document  (1) ───< DocumentChunk     (1:N, ON DELETE CASCADE)
DocumentSection (1) ───< DocumentChunk  (1:0..N, ON DELETE CASCADE, section_id NULLABLE)
DocumentChunk  (1) ───< ChunkPageMap >─── (1) DocumentPage   (many-to-many join table,
                                                                composite PK (chunk_id, document_page_id),
                                                                both legs ON DELETE CASCADE)
```

| Table | Key columns | Notes |
|---|---|---|
| `documents` | `id` (PK), `logical_document_id`, `content_hash`, `version_number`, `is_current`, `status`, `processing_attempt`, `last_error`, `failed_at`, `processing_started_at`, `processing_completed_at`, `metadata` (JSONB, Python attr `document_metadata`) | One row **per ingested version**, not per logical document |
| `stored_files` | `id`, `document_id` (unique FK → documents.id, **no** `ON DELETE CASCADE**), `original_filename`, `content_hash`, `size_bytes`, `storage_key` (unique) | `document_id` has no DB-level cascade — deletion order in `DocumentDeletionService` matters (see §6) |
| `document_pages` | `id`, `document_id` (FK, CASCADE), `page_number`, `content` | One row per `LoadedPage` returned by a loader |
| `document_sections` | `id`, `document_id` (FK, CASCADE), `parent_section_id` (self-FK, CASCADE), `title`, `section_path`, `section_level`, `section_index`, `section_metadata` (JSONB) | Hierarchical; built once per ingestion by `SectionBuilder`/`SectionService` |
| `document_chunks` | `id`, `document_id` (FK, CASCADE), `section_id` (FK → document_sections, CASCADE, **nullable**), `chunk_index`, `content`, `chunk_metadata` (JSONB), `embedding` (`pgvector` `Vector(384)`, nullable) | `section_id` is nullable specifically so headingless documents still produce retrievable chunks (see §8) |
| `chunk_page_map` | `chunk_id` + `document_page_id` composite PK, both FK CASCADE | Many-to-many: one chunk can span multiple pages, one page can back multiple chunks |

### Indexes/constraints that encode real invariants

- `uq_documents_logical_content_hash` — **unique** on `(logical_document_id, content_hash)`.
  This is what makes re-uploading identical content a no-op rather than a duplicate version
  (enforced at the DB level, not just in application code).
- `ix_documents_current_version` — a **partial unique index** on `logical_document_id` **WHERE
  `is_current = true`**. This is the database-level guarantee that a logical document can
  never have more than one current version, independent of any application bug.
- `ix_document_chunks_embedding_hnsw` — an HNSW index on `document_chunks.embedding` using
  `vector_cosine_ops` (migration `86b2f3d39bfd_add_chunk_embedding_hnsw_index.py`). The
  migration does not override pgvector's built-in HNSW build parameters (`m`,
  `ef_construction`) — whatever the installed pgvector version defaults to is what's in effect;
  this repository does not set or document specific values for them.
- `document_chunks.section_id` was made nullable by migration
  `e107da913094_allow_chunks_without_sections.py` — see §8.

### Migration history (chronological, oldest → newest via `down_revision` chain)

`f966038dee4a` (create `documents`) → `f9c70be32bf9` (add pages) → `7d01a8f0608b` (add chunks)
→ `73ec85f9f102` (add sections) → `7403624356ba` (add chunk embeddings) →
`86b2f3d39bfd` (HNSW index) → `32b24d0ab7d2` (add lifecycle status/timestamps) →
`9f0853bbd01b` (add versioning: `logical_document_id`, `version_number`, `is_current`,
backfilling every pre-existing row as its own version 1) → `7991853c5dd5` (add `StoredFile`)
→ `e107da913094` (allow chunks without sections).

---

## 8. Chunking and Provenance

Pipeline (see §3 step 10 for the orchestration): `StructureExtractor.extract` →
`fragment_table_units` → `SectionBuilder.build` → `SectionService.persist_sections` →
`shred_semantically` → `apply_size_guard` → `ChunkService.persist_chunks`.

- **`StructureExtractor`** (`app/ingestion/structure_extractor.py`) walks each `LoadedPage`'s
  text line by line, recognizing a Markdown heading pattern (`^#{1,6}\s+...`), a list pattern
  (`^[-*+]\s+` or `^\d+[.)]\s+`), and GitHub-Flavored-Markdown pipe tables (a header row
  immediately followed by a `|---|---|`-style separator row). It tracks a heading stack to
  build a `section_path` like `"Introduction > Background"` for every unit, and emits
  `StructuralUnit`s of type `HEADING`, `PARAGRAPH`, `LIST`, or **`TABLE`** (with a populated
  `TableData` of parsed `headers`/`rows`, re-serialized to canonical Markdown via
  `table_serializer.table_to_markdown`). Table detection is intentionally limited to this one
  real, well-defined text format — `MarkdownLoader` passes a `.md` file's own table syntax
  straight through, and `DocxLoader` renders python-docx's genuinely structured
  `document.tables` data as the same pipe-table syntax specifically so both are detected by
  this one code path. **`PDFLoader` has no structural table extraction** (`pypdf.extract_text()`
  returns flat text with no layout information), so PDF tables are not detected and ingest as
  ordinary (often garbled) paragraph text — a real, documented limitation, not a gap this
  extractor papers over. Single-column tables (only one header cell) are also not detected,
  since the pipe-row/separator regexes require at least two columns.
- **`fragment_table_units`** (`app/ingestion/size_guard.py`) pre-fragments every TABLE unit,
  via the existing `app.ingestion.table_chunker.split_table`, so its content already fits the
  chunk token budget before `shred_semantically`/`apply_size_guard` ever see it (see below for
  why). Called unconditionally for every table (even ones already under budget, which simply
  come back as a single fragment), tagging each resulting unit's `.metadata` with
  `content_type: "table"`, `table_id`, `table_title`, `table_fragment_index`, and
  `table_fragment_count` — propagated into the persisted chunk's `chunk_metadata` by
  `ChunkService.persist_chunks`, so a table chunk is distinguishable from a prose chunk after
  persistence. **Known limitation**: `split_table` only splits at row boundaries; if a single
  row (plus headers) already exceeds the budget on its own, it cannot be shrunk further, and
  the resulting chunk remains oversized (tripping `apply_size_guard`'s safety net below).
- **`SectionBuilder`/`SectionService`** turn the sequence of `section_path` strings into a
  deduplicated hierarchy of `DocumentSection` rows (one per unique path, in first-appearance
  order), returning a `SectionMap` (`path → DocumentSection.id`) used to attach `section_id` to
  chunks later. Fragmenting tables beforehand never changes which section paths exist, since
  every fragment keeps its source table's original `section_path` unchanged.
- **`shred_semantically`** (`app/ingestion/semantic_shredder.py`) embeds every prose unit
  (`PARAGRAPH`/`LIST`) once in a batch, then walks units in order and groups **adjacent** prose
  units into a `SemanticUnit` as long as: they're in the same section, and the cosine
  similarity between consecutive prose embeddings is `>= 0.7` (hard-coded threshold, passed as
  a literal in `IngestionService._process_document`). A `HEADING` always forces a break. A
  `TABLE` unit is always isolated into its own `SemanticUnit`, never merged with surrounding
  prose. This is **local, adjacent-pair similarity**, not global clustering.
- **`apply_size_guard`** (`app/ingestion/size_guard.py`) converts each `SemanticUnit` into one
  or more final `ChildChunk`s, respecting a `max_tokens=500` budget (hard-coded literal passed
  from `IngestionService`, shared by `fragment_table_units` for the same document). "Tokens"
  here means **whitespace-split word count** (`estimate_tokens`), not a real tokenizer —
  explicitly documented in the code as a deterministic, dependency-free placeholder (this also
  means pipe characters in a table row each count as their own "token", inflating a table
  fragment's apparent size relative to prose of similar visual length). Oversized prose is
  split first at paragraph boundaries, then sentence boundaries, then hard word-count
  boundaries; an oversized `TABLE` unit cannot be split this way (splitting mid-row would
  destroy its structure) and instead raises `ValueError` — a safety net that should never fire
  in practice, since `fragment_table_units` already pre-fits every table fragment, except for
  the single-oversized-row edge case noted above.
- **`ChunkService.persist_chunks`** embeds all final chunk texts in one
  `embedding_provider.embed_batch()` call, persists one `DocumentChunk` per chunk (with
  `section_id`, `chunk_index`, `content`, `embedding`, and `chunk_metadata` containing
  `page_numbers`/`section_path`, plus the table metadata above when the chunk's source unit is
  a TABLE), and persists one `ChunkPageMap` row per page the chunk's source content actually
  spans. Table chunks flow through this exact same persistence code as prose chunks — no
  separate table-persistence path exists.

### Headingless documents

If a document (or a portion of it) has no Markdown headings, `section_path` is `""` for its
units, so no `DocumentSection` is created for that content and `apply_size_guard` sets
`section_id=None` on the resulting chunks. This is exactly why `DocumentChunk.section_id` is
nullable (migration `e107da913094`) and why `DocumentRepository.search_similar_chunks` uses an
**outer** join to `DocumentSection` (see §12) — a headingless chunk is still fully retrievable,
just with `section_id`/`section_path` reported as `null` in every API response that surfaces
it (`RetrievedChunkResponse`, `RetrievalTraceCandidateResponse`, research `sources`).

### Page provenance

Page numbers are authoritative from the loader outward: `PDFLoader` assigns real PDF page
numbers; `MarkdownLoader`/`DocxLoader` have no reliable page concept and always report page 1.
A chunk's `page_numbers` (surfaced via `chunk_metadata` and via the `ChunkPageMap` join in
`search_similar_chunks`) is the sorted, deduplicated union of every source page any of its
constituent structural units came from — so a chunk that happens to straddle a page boundary
correctly reports both page numbers.

---

## 9. Embeddings and Vector Storage

| Aspect | Value | Source |
|---|---|---|
| Provider | `SentenceTransformerEmbeddingProvider` (`app/embeddings/sentence_transformer.py`) | only implementation of `EmbeddingProvider` |
| Model | `sentence-transformers/all-MiniLM-L6-v2` | `Settings.embedding_model`, env `EMBEDDING_MODEL` |
| Vector dimension | `384` | `DocumentChunk.embedding: Mapped[list[float] | None] = mapped_column(Vector(384), ...)` in `app/core/models.py` — **hard-coded in the schema**, not derived from the model at runtime |
| Normalization | **Not applied.** `model.encode(text, convert_to_numpy=True)` is called with no `normalize_embeddings=True` — raw model output is stored as-is | `app/embeddings/sentence_transformer.py` |
| Where generated | During ingestion, in `ChunkService.persist_chunks` (one batched call per document); during retrieval, once per retrieval sub-query in `RetrievalService._search` | `app/ingestion/chunk_service.py`, `app/retrieval/service.py` |
| Where stored | `document_chunks.embedding` column | `app/core/models.py` |
| How queried | `DocumentChunk.embedding.cosine_distance(query_embedding)` (pgvector's cosine-distance operator), ordered ascending, filtered through the HNSW index | `app/core/repositories/document.py::search_similar_chunks` |
| Lifetime | Loaded once per process via `@lru_cache` on `get_embedding_provider()` | `app/core/dependencies.py` |

**Changing `EMBEDDING_MODEL` requires re-embedding every existing chunk.** The vector column's
dimension (`384`) is fixed in the schema; a model with a different output dimension would fail
to insert (dimension mismatch) without a new migration altering the column, and even a
same-dimension model swap would leave old vectors incomparable to new queries since they come
from a different embedding space. Nothing in this codebase currently re-embeds existing data
automatically — that would be a manual operation.

**Embedding model vs. reranker model vs. generation LLM are three independent, separately
configured components** — see the table in §21. They are never the same model and must not be
conflated.

---

## 10. Retrieval Pipeline

Implemented end-to-end in `RetrievalService._search` (`app/retrieval/service.py`), called by
both `POST /retrieval/search` (`app/api/retrieval.py`) and `POST /research/ask`
(`app/api/research.py`) — there is exactly one retrieval implementation shared by both.

```text
RetrievalService.search(query, limit=10, max_distance=None, document_id=None,
                         section_id=None, candidate_limit=None, trace=False)
  │
  ├─ candidate_limit defaults to max(limit, 50) if not supplied
  ├─ raises ValueError if candidate_limit < limit
  ├─ (optional) wraps everything below in a Langfuse "retriever" observation
  │
  ▼
_search(...)
  │
  ├─ if self.query_decomposer is None:  retrieval_queries = (query,)
  │  else:  decomposition = await query_decomposer.decompose(query)
  │         retrieval_queries = decomposition.sub_queries     ← ALWAYS runs if a decomposer
  │                                                               is configured, regardless of
  │                                                               `trace` (see §11, §14)
  │
  ├─ for each retrieval_query in retrieval_queries:
  │     embedding = embedding_provider.embed_text(retrieval_query)
  │     candidates = DocumentRepository.search_similar_chunks(embedding, limit=candidate_limit,
  │                                                            max_distance, document_id, section_id)
  │     raw_candidate_count += len(candidates)
  │     for each candidate: keep it in merged_candidates[chunk_id] only if it's new OR has a
  │                          strictly smaller distance than what's already there  ← MERGE/DEDUP
  │
  ├─ candidates = sorted(merged_candidates.values(), key=distance)   ← deduplicated pool
  ├─ deduplicated_candidate_count = len(merged_candidates)
  │
  ├─ if self.reranker is None:  final_results = candidates[:limit]
  │  else:  reranked = reranker.rerank(query, candidates)   ← ONE call, ORIGINAL query, on the
  │                                                              WHOLE deduplicated pool
  │         final_results = reranked[:limit]
  │
  └─ if trace: build a RetrievalTrace capturing original_query, sub_queries, raw/deduplicated
              counts, the full candidate pool, final_results, and assembled context
```

### Key facts

- **`candidate_limit` is applied per sub-query**, not split across sub-queries. If
  decomposition produces 3 sub-queries and `candidate_limit=50` (the default when `limit=10`),
  each sub-query's database call can return up to 50 candidates — raw candidates can be up to
  `3 × 50 = 150` before merge, not capped at 50 total.
- **Deduplication keeps the best (lowest) distance** seen for a given `chunk_id` across all
  sub-queries, not the first one found.
- **Reranking is a single global pass** over the entire deduplicated candidate pool, using the
  caller's **original** query string (never an individual sub-query) — this is why the
  documentation and code both describe it as "one global reranker after merge", not
  per-sub-query reranking.
- `document_id` as a filter parameter here means **one specific version's `Document.id`**, not
  a `logical_document_id` — it restricts retrieval to chunks belonging to that exact row.

### Current-version filtering (why historical vectors stay stored but excluded)

`DocumentRepository.search_similar_chunks` (`app/core/repositories/document.py`) filters:

```sql
WHERE document_chunks.embedding IS NOT NULL
  AND documents.status = 'READY'
  AND documents.is_current IS TRUE
```

This is the **only** place that enforces "only the current, READY version is searchable." It
is a `WHERE` clause on a live query, not a data-deletion rule — a previous version's chunks and
embeddings are never touched by a current-version switch (§5), so making that version current
again later makes its original chunks searchable again immediately, with no re-embedding.

An **outer join** to `DocumentSection` (rather than inner) means a chunk with `section_id IS
NULL` (headingless — §8) is still returned, with `section_path` reported as `None`/`null`.

---

## 11. Query Decomposition

### What it is / is not

Implemented across three files: `app/retrieval/decomposition.py` (data types + provider
protocol), `app/retrieval/query_decomposer.py` (`QueryDecomposer`, normalization/validation),
`app/retrieval/openrouter_decomposition.py` (`OpenRouterQueryDecompositionProvider`, the actual
LLM call).

It is **query planning for retrieval**, not generic query rewriting/expansion. The system
prompt in `OpenRouterQueryDecompositionProvider.decompose` is explicit:

> "Split the user's question only when it contains multiple distinct information needs...
> Do not generate paraphrases, semantic variants, or arbitrary query expansions."

### Flow (`QueryDecomposer.decompose`, `app/retrieval/query_decomposer.py`)

```python
original_query = query.strip()
if not original_query:
    raise ValueError("query must not be empty")

try:
    candidates = await provider.decompose(original_query)  # list[str] from the LLM
    sub_queries = _normalize_queries(original_query, candidates)
    return QueryDecompositionResult(
        original_query, tuple(sub_queries), was_decomposed=len(sub_queries) > 1
    )
except Exception as exc:
    # ANY exception from the provider — network error, malformed JSON, wrong shape,
    # non-string elements, etc. — is caught here.
    return QueryDecompositionResult(
        original_query,
        sub_queries=(original_query,),
        was_decomposed=False,
        fallback_reason=str(exc),
    )
```

### `_normalize_queries` — the invariants

```python
add(original_query)  # ALWAYS first; can never be removed by dedup
for candidate in candidates:
    add(candidate)
    if len(queries) >= MAX_SUB_QUERIES:  # MAX_SUB_QUERIES = 5 (app/retrieval/decomposition.py)
        break
```

`add()` strips whitespace, skips blank strings, and deduplicates **case-insensitively**
(`.casefold()`) against everything already added. Net effect, verified directly from this
code:

- **The original query is always preserved** — it is unconditionally the first entry and can
  never be deduplicated away or dropped.
- **1–5 unique, non-empty queries, always.** Never 0 (the original is always present), never
  more than 5.
- **Decomposition failure → `[original_query]` only**, with `was_decomposed=False`.

### `OpenRouterQueryDecompositionProvider.decompose` — exact request shape

```python
await self.client.chat.completions.create(
    model=self.model,  # Settings.openrouter_model — the SAME model setting used for
    # generation (app/core/dependencies.py). There is no separate
    # decomposition-model setting.
    temperature=0,  # HARD-CODED — not configurable via Settings
    messages=[system, user],
)
```

No `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `response_format`,
`timeout`, or retry parameters are set — everything beyond `model`/`temperature`/`messages`
uses the OpenRouter/`openai`-SDK defaults. The response is parsed as `json.loads(message)` and
validated to be `{"queries": [str, ...]}`; any shape mismatch raises `ValueError`/`RuntimeError`,
which `QueryDecomposer` catches as described above.

**`was_decomposed` and `fallback_reason` are computed but not currently surfaced anywhere.**
`RetrievalService._search` only reads `decomposition.sub_queries` — neither field appears in
`RetrievalTrace`, `RetrievalTraceResponse`, or the research API's trace payload. The only
externally-visible signal that decomposition didn't happen (or failed) is that `sub_queries`
contains exactly one entry.

### Decomposition is independent of tracing — verified

`RetrievalService._search` calls `self.query_decomposer.decompose(query)` **unconditionally**
whenever `self.query_decomposer is not None`, before the `if trace:` block that builds the
trace object. The `trace` parameter only controls whether a `RetrievalTrace` is *recorded* —
it has no code path that skips decomposition or retrieval. This is also explicitly covered by
a dedicated regression test:
`tests/unit/test_retrieval_decomposition.py::test_decomposition_and_merge_happen_even_when_trace_is_disabled`.

**In practice, decomposition is effectively mandatory, not optional, in the deployed system.**
`RetrievalService` itself treats `query_decomposer=None` as "skip decomposition", but the
FastAPI dependency graph (`app/core/dependencies.py::get_retrieval_service`) always injects a
real `QueryDecomposer`. That decomposer's provider
(`get_query_decomposition_provider`) **raises `ValueError` eagerly** if
`Settings.openrouter_api_key` is not configured — before any retrieval or decomposition logic
runs. So as currently wired, **`POST /retrieval/search` and `POST /research/ask` both fail
(500, unhandled dependency-resolution error) if `OPENROUTER_API_KEY` is not set**, even though
decomposition is architecturally optional at the `RetrievalService` level.

---

## 12. Candidate Retrieval

`DocumentRepository.search_similar_chunks` is called once per sub-query from
`RetrievalService._search`. Parameters and defaults:

| Parameter | Where it comes from | Default |
|---|---|---|
| `limit` (→ `candidate_limit` of the retrieval call) | `RetrievalService.search(candidate_limit=...)` | `max(limit, 50)` if not explicitly passed |
| `max_distance` | `RetrievalService.search(max_distance=...)` | `None` (no distance filter) — **not exposed** through either `RetrievalSearchRequest` or `ResearchRequest`; only reachable if `RetrievalService.search` is called directly (e.g. from a test) |
| `document_id` | `RetrievalSearchRequest.document_id` | `None` |
| `section_id` | `RetrievalSearchRequest.section_id` | `None` |
| final `limit` (Top K after rerank) | `RetrievalSearchRequest.limit` (`1..50`, default `10`) for `/retrieval/search`; hard-coded `10` for `/research/ask` (`app/api/research.py::_execute_research`) | — |

### Worked example (illustrative — values are inputs you choose, not fixed outputs)

```text
Original query decomposes into 3 sub-queries. limit=10 → candidate_limit defaults to 50.

  sub-query 1 → up to 50 candidates
  sub-query 2 → up to 50 candidates
  sub-query 3 → up to 50 candidates
  ------------------------------------
  raw_candidate_count  = sum of all three calls' result counts (≤ 150)
  deduplicated pool    = unique chunk_ids across all three, each keeping its best (lowest)
                         distance (≤ raw_candidate_count, exact number depends on overlap)
  → entire deduplicated pool is reranked in ONE call
  → final_results = top 10 by rerank score
```

The exact raw/deduplicated counts for any real query are only known by running it — they
depend on corpus content and are reported live via `trace.raw_candidate_count` /
`trace.deduplicated_candidate_count` (see §15).

---

## 13. Reranking

| Aspect | Value | Source |
|---|---|---|
| Implementation | `CrossEncoderReranker` (`app/retrieval/cross_encoder.py`), the only implementation of the `Reranker` protocol (`app/retrieval/reranker.py`) | — |
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `Settings.reranker_model`, env `RERANKER_MODEL` |
| Input | `(query, chunk.content)` pairs for **every** chunk in the deduplicated candidate pool | `rerank(query, chunks)` |
| Scoring | `CrossEncoder.predict(pairs)` — one relevance score per pair | `sentence_transformers.CrossEncoder` |
| Ordering | Sorted descending by score | `sorted(..., key=lambda item: float(item[1]), reverse=True)` |
| Output | A **new** list of `RetrievedChunk`s, identical to the input except `rerank_score` is now populated (original `distance` is preserved, unchanged) | — |
| Final Top K | **Not** the reranker's job — `RetrievalService._search` truncates `reranked[:limit]` after reranking | `app/retrieval/service.py` |
| Batch size / score threshold / cutoff | None — the reranker scores and reorders the **entire** pool it's given, with no internal limit | — |

Why global, after merge (not per-sub-query): reranking per sub-query would score each
sub-query's candidates only against *that* sub-query's text, making cross-sub-query ranking
incomparable; reranking once, against the merged pool and the user's original query, produces
one consistent relevance ordering regardless of which sub-query originally surfaced a chunk.
The trade-off is cost/latency: the reranker call's cost scales with the size of the
**deduplicated pool**, which itself scales with `candidate_limit × number of sub-queries`
(before dedup).

---

## 14. RAG / Generation

**`ContextAssembler.assemble(chunks)`** (`app/generation/context.py`) turns the final reranked
chunks into:

```text
text = "\n\n".join(f"[Source {i}]\n{chunk.content}" for i, chunk in enumerate(chunks, start=1))
sources = [GenerationContextSource(document_id, chunk_id, section_id, section_path, page_numbers), ...]
```

If constructed with `max_characters` set, it stops adding sources once the accumulated text
would exceed that budget (chunks are **not** truncated mid-text — a chunk is either fully
included or excluded). **`ContextAssembler()` is always constructed with no `max_characters`**
in the actual call sites (`app/retrieval/service.py`, `app/api/research.py`) — there is
currently no context-length cap applied in practice.

**`GenerationService.generate(query, context)`** (`app/generation/generation_service.py`)
wraps `GenerationProvider.generate` in an optional Langfuse `"generation"` observation and
otherwise just delegates.

**`OpenRouterGenerationProvider.generate`** (`app/generation/openrouter.py`) — the only
`GenerationProvider` implementation:

```python
await self.client.chat.completions.create(
    model=self.model,  # Settings.openrouter_model
    messages=[
        {
            "role": "system",
            "content": "You are a research assistant. Answer the user's "
            "question using the provided research context. "
            "Do not invent facts that are not supported by the "
            "context.",
        },
        {"role": "user", "content": f"Research context:\n\n{context.text}\n\nQuestion:\n\n{query}"},
    ],
)
```

**No `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`,
`response_format`, `timeout`, or retry parameters are set** — this call relies entirely on
OpenRouter/the selected model's own defaults. The only non-default thing configured on the
client is a `X-Title` header (`Settings.openrouter_app_name`) set on the `AsyncOpenAI` client
in `__init__`.

`GenerationResult` carries `text`, `model` (from the response, falling back to the configured
model string), and `input_tokens`/`output_tokens`/`total_tokens` (from `response.usage`, or
`None` if the provider didn't return usage).

### Retrieval vs. context assembly vs. generation vs. RAG — the distinction this codebase makes

- **Retrieval** = `RetrievalService.search(...)` — decomposition through reranked Top K. Ends
  with a `list[RetrievedChunk]`. No LLM call for the answer itself (only, optionally, for
  decomposition).
- **Context assembly** = `ContextAssembler.assemble(...)` — a pure, deterministic
  transformation of retrieved chunks into a prompt-ready string + source list. No LLM call.
- **Generation** = `GenerationService`/`OpenRouterGenerationProvider` — the actual LLM call
  that produces the answer text.
- **RAG** (as this codebase implements it) = retrieval → context assembly → generation,
  orchestrated by `app/api/research.py::_execute_research`. `POST /retrieval/search` performs
  only the first step and never calls the generation LLM.

---

## 15. Retrieval Explorer

This is a **debugging/engineering tool**, not a second answer-generation UI. It is backed
entirely by `POST /retrieval/search` (`app/api/retrieval.py`), which never calls
`GenerationService` — it stops after reranked Top K.

### Trace availability contract (`app/retrieval/schemas.py`, `app/api/retrieval.py`)

```text
effective_trace = request.trace AND settings.trace_enabled
```

| `request.trace` | `TRACE_ENABLED` | `effective_trace` | Retrieval/decomposition run? | `trace` in response | `trace_requested` | `trace_available` | `trace_unavailable_reason` |
|---|---|---|---|---|---|---|---|
| false | false | false | **yes** | `null` | `false` | `false` | `null` |
| false | true | false | **yes** | `null` | `false` | `false` | `null` |
| true | false | false | **yes** | `null` | `true` | `false` | `"server_disabled"` |
| true | true | true | **yes** | populated | `true` | `true` | `null` |

**Tracing never gates retrieval or decomposition in any of these four cases** — only whether
the execution's trace is *recorded and returned*. This exact rule, and all four combinations,
are covered by tests in `tests/integration/test_retrieval_api.py` and mirrored for
`POST /research/ask` (`app/api/research.py`, same `effective_trace` computation, same
`trace_requested`/`trace_available`/`trace_unavailable_reason` fields added to its response
dict — though for backward compatibility `/research/ask`'s `"trace"` key is **omitted
entirely** when unavailable, rather than explicitly `null` like `/retrieval/search`).

### What the trace contains (`RetrievalTraceResponse`, `app/retrieval/schemas.py`)

`query`, `original_query`, `sub_queries`, `candidate_limit`, `raw_candidate_count`,
`deduplicated_candidate_count`, `candidates` (the full deduplicated pool, pre-rerank —
`rerank_score` is `null` on these since reranking hasn't happened to them individually in the
trace-candidate conversion), `final_results` (post-rerank, `rerank_score` populated),
`context` (the assembled generation context + its sources, built the same way `/research/ask`
would, even though `/retrieval/search` never calls generation).

### Why it exists separately from RAG

`POST /research/ask` only returns `answer`, `model`, `sources` (document/chunk/section/page
identifiers — no raw content, distance, or rerank score), plus trace metadata if requested.
That's deliberately too little to debug *why* a particular chunk was or wasn't retrieved.
Retrieval Explorer exists to answer exactly that question — original query, whether/how it was
decomposed, the full candidate pool with distances, and the final reranked ordering — without
needing to also run (and pay for) generation.

---

## 16. Langfuse / Observability

`app/observability/langfuse.py::get_langfuse()` is an `@lru_cache`d factory:

```python
if not settings.langfuse_enabled:
    return None
if not settings.langfuse_public_key:
    raise ValueError(...)
if not settings.langfuse_secret_key:
    raise ValueError(...)
return Langfuse(
    public_key=...,
    secret_key=...,
    base_url=settings.langfuse_base_url,
    environment=settings.langfuse_environment,
)
```

Every call site checks `if langfuse is None` first and runs the **exact same logic either
way** — Langfuse is purely additive observability, never a behavioral gate. Three
instrumented spots:

| Site | Span type | `name` | Input captured | Output captured |
|---|---|---|---|---|
| `RetrievalService.search` | `"retriever"` | `"document-retrieval"` | query, limit, candidate_limit, max_distance, document_id, section_id | result_count + each result's chunk_id/document_id/section_id/section_path/page_numbers/distance/rerank_score |
| `GenerationService.generate` | `"generation"` | `"answer-generation"` | query, assembled context text | answer text, model, token usage (input/output/total) |
| `app/api/research.py::ask` | `"span"` | `"research-request"` | query, requested trace flag | answer, model, source count |

Errors in any of these are reported via `observation.update(level="ERROR",
status_message=str(exc))` before the exception is re-raised (Langfuse always sees failures,
even though they still propagate as real HTTP errors).

### Langfuse vs. Retrieval Explorer — the actual difference

- **Retrieval Explorer** (`RetrievalTrace`/`trace` field) is a **per-request, synchronous,
  returned-to-the-caller** snapshot of one retrieval execution — decomposition, full candidate
  pool, dedup counts, reranked results, assembled context. It is gated by
  `request.trace AND settings.trace_enabled` (§11, §15) and is consumed directly by the
  frontend's Retrieval Explorer UI.
- **Langfuse** is an **external, asynchronous observability backend** that records spans for
  later inspection in the Langfuse dashboard — it is gated independently by
  `LANGFUSE_ENABLED`/`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, has no relationship to
  `TRACE_ENABLED`, and its content (query, result summaries, token usage) is coarser than the
  full `RetrievalTrace` payload — it does not include the full candidate pool's chunk content,
  for example.

Both can be on, off, or independently configured at the same time — there is no code path
where one controls the other.

---

## 17. API Layer

### Documents (`app/api/documents.py`, prefix `/documents`)

| Method & path | Purpose | Key request fields | Key response | Notable status codes |
|---|---|---|---|---|
| `POST /documents` | Upload/ingest a file as a new document or a new version of an existing `logical_document_id` | multipart: `file` (required), `document_type` (required), `title`/`source`/`logical_document_id` (optional) | `DocumentVersionResponse` | `201` (including a FAILED outcome — see §3); `422` empty/unsupported file |
| `GET /documents` | List every logical document with its current version (or `null`) | — | `list[LogicalDocumentResponse]` | `200` |
| `GET /documents/{logical_document_id}` | One logical document's identity + current version | — | `LogicalDocumentResponse` | `404` if unknown |
| `GET /documents/{logical_document_id}/versions` | Every version, newest first, any status | — | `list[DocumentVersionResponse]` | `404` if unknown |
| `POST /documents/{logical_document_id}/versions/{version_id}/retry` | Retry a FAILED version in place | no body | `DocumentVersionResponse` | `200` (even on re-failure); `404` unknown/mismatched version; `409` non-FAILED version |
| `POST /documents/{logical_document_id}/versions/{version_id}/set-current` | Promote a READY version to current | no body | `DocumentVersionResponse` | `200` (idempotent if already current); `404` unknown/mismatched version; `409` non-READY version |
| `DELETE /documents/{logical_document_id}/versions/{version_id}` | Delete one version (DB + physical file) | — | `204` | `404` unknown/mismatched version |
| `DELETE /documents/{logical_document_id}` | Delete an entire logical document, all versions | — | `204` | `404` unknown |

Every route that takes both IDs checks `version.logical_document_id == logical_document_id`
and returns the **same 404** whether the version doesn't exist at all or belongs to a
different logical document — this deliberately avoids revealing that a version id exists
elsewhere.

`DocumentVersionResponse` fields (`app/ingestion/schemas.py`): `id`, `logical_document_id`,
`version_number`, `is_current`, `status`, `title`, `document_type`, `source`,
`processing_attempt`, `created_at`, `updated_at`, `processing_started_at`,
`processing_completed_at`, `failed_at`, `last_error`. `LogicalDocumentResponse`:
`logical_document_id`, `current_version: DocumentVersionResponse | None`.

### Retrieval (`app/api/retrieval.py`, prefix `/retrieval`)

`POST /retrieval/search` — request `RetrievalSearchRequest` (`query`, `limit` 1–50 default 10,
`document_id?`, `section_id?`, `trace` default `false`); response `RetrievalSearchResponse`
(`results: list[RetrievedChunkResponse]`, `trace_requested`, `trace_available`,
`trace_unavailable_reason`, `trace: RetrievalTraceResponse | null`). See §15 for the full
trace-availability contract. `RetrievedChunkResponse` fields: `document_id`, `chunk_id`,
`section_id` (nullable), `section_path` (nullable), `page_numbers`, `content`, `distance`,
`similarity` (`1.0 - distance`), `rerank_score` (nullable).

### Research (`app/api/research.py`, prefix `/research`)

`POST /research/ask` — request `ResearchRequest` (`query`, `trace` default `false`); response
is a plain `dict[str, object]` (not a declared Pydantic `response_model`) with `answer`,
`model`, `sources` (list of `{document_id, chunk_id, section_id, section_path, page_numbers}`
— no content/distance/rerank_score), `trace_requested`, `trace_available`,
`trace_unavailable_reason`, and `trace` (same shape as the retrieval trace, **only present as
a key at all when available** — omitted, not `null`, otherwise, for backward compatibility
with clients written before this field existed).

### Health (`app/main.py`)

`GET /health` → `{"status": "healthy", "environment": ...}`. `GET /health/ready` → `{"status":
"ready"|"not_ready", "checks": {"database": bool}}`, backed by a real `SELECT 1`
(`app/core/database_health.py`).

---

## 18. Frontend Architecture

Single page app (`App.tsx` → `ResearchPage.tsx`), with `ResearchPage` owning all cross-tab
state: `activeTab` (`'research' | 'retrieval' | 'documents' | 'trace'`), the global **Trace
Mode** toggle (`traceEnabled`, rendered in `Header.tsx`), the last research response, and a
`targetDocId` used to cross-navigate between tabs (e.g. "Inspect Doc" from a retrieval result
jumps to the Document Library pre-selected on that document).

### API client layer (the only modules allowed to call `fetch`)

- **`frontend/src/api/documents.ts`** — one function per document-management endpoint
  (`listDocuments`, `getDocument`, `getDocumentVersions`, `uploadDocument`,
  `retryDocumentVersion`, `setCurrentVersion`, `deleteDocumentVersion`,
  `deleteLogicalDocument`). Defines `DocumentApiUnavailableError`, thrown **only** when a
  response is HTTP 404 with literally `{"detail": "Not Found"}` (FastAPI/Starlette's generic
  no-route-matched body) — every application-level 404 (e.g. `"Document not found: <id>"`) is
  a normal error, not evidence the API isn't exposed. This is how the frontend distinguishes
  "this specific resource doesn't exist" from "this route isn't implemented by the backend at
  all."
- **`frontend/src/api/research.ts`** — `askResearchQuestion`, `searchRetrievedChunks`,
  `checkBackendHealth`, `checkBackendReadiness`. `searchRetrievedChunks` sends the global Trace
  Mode toggle's value as `trace` on every request.

### TypeScript types (`frontend/src/types/document.ts`, `research.ts`)

Hand-written interfaces mirroring the Pydantic schemas exactly, including nullability
(`section_path: string | null`, etc.) and the trace-availability fields
(`trace_requested`/`trace_available`/`trace_unavailable_reason: 'server_disabled' | null`).
These are **not** code-generated from the backend — they must be kept in sync by hand when the
backend contract changes (as happened across the version-management and trace-contract work
reflected in this repository's history).

### Document Library (`DocumentWorkspace.tsx` + `UploadDocumentModal.tsx`)

List view = one card per **logical document**; versions are never shown as top-level cards.
Selecting a card loads its full version history via a separate `GET .../versions` call (the
list endpoint only returns the current version). Each version row shows its status, a
"Current" label or a "Make Current" button (only for READY, non-current versions — PROCESSING/
FAILED never get this button), a "Retry" button (FAILED only), and "Delete". The upload modal
offers "Upload as new document" (default) vs. "Upload as new version of: [existing logical
document ▼]" (populated from `GET /documents`) — selecting the latter sends the chosen
`logical_document_id` in the multipart body, reusing the single `POST /documents` contract
rather than a second endpoint.

### Research & Synthesis (`ResearchInput.tsx`, `AnswerPanel.tsx`, `SourcesPanel.tsx`,
`SourceCard.tsx`, `DownloadButton.tsx`, `TracePanel.tsx`)

Submits via `askResearchQuestion(query, traceEnabled)`. Renders the answer, a source list
(enriched with content/distance/rerank score from the trace's `final_results` when trace data
is available, by matching on `chunk_id`), a JSON download of the raw response, and — only when
`response.trace` is present — an inline `TracePanel`.

### Retrieval Explorer (`RetrievalWorkspace.tsx`)

Sends the **same global `traceEnabled` toggle** (passed down as a prop from `ResearchPage`) as
`trace` on every `/retrieval/search` call — there is deliberately no second, page-local trace
toggle, specifically so the toggle's displayed state can never drift from what's actually sent
in the request. UI behavior is driven off the explicit response fields, not off `trace`
truthiness alone:

- Trace Mode OFF (`trace_requested=false`): no trace-related banner at all, just results.
- Trace Mode ON and available (`trace_available=true`): an inline "Query Decomposition"
  section (original query, sub-queries as chips, raw/deduplicated candidate counts, final
  Top K count) plus a "View Trace & Rerank Scores" button that opens the full `TracePanel`.
- Trace Mode ON but server-disabled (`trace_requested=true, trace_available=false,
  trace_unavailable_reason="server_disabled"`): an explicit warning banner stating that
  retrieval and decomposition still ran normally and only the trace representation is
  unavailable — worded to never imply decomposition itself didn't happen.

### Relationship between frontend Trace Mode and backend `TRACE_ENABLED`

The frontend toggle only ever controls the **request-level** `trace` flag. It has no way to
observe or change `Settings.trace_enabled` directly — it only finds out the server-side
capability ceiling indirectly, from `trace_available`/`trace_unavailable_reason` in each
response, and never silently flips its own toggle state based on that (per the explicit UX
rule documented in-code and in the component).

### Loading/error/empty states

Every async view (`DocumentWorkspace`, `RetrievalWorkspace`, `ResearchPage`) tracks `isLoading`
and a distinct error-vs-empty state, and `DocumentApiUnavailableError` is surfaced as its own
"API not exposed" banner distinct from an ordinary "no results"/"not found" state.

---

## 19. Failure and Recovery Behavior

| Failure point | What happens | What remains intact |
|---|---|---|
| Upload: file read fails before any DB row | Exception propagates as a real error (422/500 depending on cause) | Nothing was created |
| Upload: physical file write succeeds, `StoredFile` commit fails | `session.rollback()`; best-effort delete of the just-written physical file (failure to delete is swallowed); re-raise | The `Document` row from step 3 of §3 may still exist at `UPLOADED` — it is not cleaned up by this specific `except` block |
| Ingestion processing fails (extraction/chunking/embedding) | `session.rollback()`; re-fetch `Document`; `mark_failed()`; commit; exception gets `.failed_document` attached and is re-raised | `StoredFile` and physical file untouched; reported to the caller as a normal FAILED version, not a 500 (§3) |
| Retry of a non-FAILED version | `IngestionService.retry_document` raises `ValueError`; route maps to `409` | No state change |
| Retry fails again | Same `Document` row marked FAILED again, `processing_attempt` incremented | Same as above |
| Version deletion: DB commit fails | Exception propagates; physical file is **never** touched (file deletion only happens after a successful commit) | DB and file both intact |
| Version deletion: DB commit succeeds, physical file deletion fails | Exception propagates | DB is already committed/consistent; the file is orphaned on disk — no reconciliation job exists (§20) |
| `set-current` on a non-READY version | `DocumentService.set_current` raises `ValueError`; route maps to `409` | No state change |
| `set-current` fails mid-transaction | Nothing is committed (the demote-then-promote pair commits once together); session close rolls back | Never leaves two current versions or zero when one was intended |
| Query decomposition provider throws | Caught inside `QueryDecomposer.decompose`; falls back to `[original_query]` with `was_decomposed=False` | Retrieval proceeds using just the original query |
| `OPENROUTER_API_KEY` not configured | `get_query_decomposition_provider`/`get_generation_provider` raise `ValueError` at dependency-resolution time | `/retrieval/search` and `/research/ask` both fail outright (unhandled → 500) — see §11 |
| Reranker unavailable | Not applicable — `CrossEncoderReranker` is unconditionally constructed in `app/core/dependencies.py`; there is no configuration path to disable it |
| Generation provider throws | Propagates out of `GenerationService.generate`/`/research/ask`; Langfuse (if enabled) records the error before re-raise | Retrieval results are discarded; no partial answer is fabricated |

---

## 20. Evaluation

`tests/evaluation/` contains:

- `retrieval_metrics.py` — `recall_at_k(retrieved_ids, relevant_ids, k)` and
  `mrr_at_k(retrieved_ids, relevant_ids, k)`, both pure functions over `UUID` sequences/sets.
  Fully implemented and unit-tested (`test_retrieval_metrics.py`).
- `retrieval_dataset.py` — `RetrievalEvaluationCase(query: str, relevant_chunk_ids:
  frozenset[UUID])`, a plain dataclass.
- `retrieval_cases.py` — defines `RETRIEVAL_EVALUATION_CASES`, a list containing **one
  placeholder case** whose `relevant_chunk_ids` are literally `UUID("...")` — not a real UUID.
  **This module is not imported anywhere else in the repository** (verified by search) — it is
  not wired into any test, script, or CI step that actually runs an evaluation against the
  corpus.

**What exists:** the metric functions (recall@K, MRR@K) and the data-shape scaffolding.
**What does not exist:** a populated, real dataset; any harness that runs `RetrievalService`
against `RETRIEVAL_EVALUATION_CASES` and reports recall/MRR; graded (non-binary) relevance;
NDCG@K; a baseline run to compare against. This should be read as "metrics implemented,
end-to-end offline evaluation not yet assembled," not as a working evaluation pipeline.

Offline evaluation (this), production observability (Langfuse, §16), and human/LLM feedback
are three distinct concerns in this codebase; only the first two have any implementation at
all, and offline evaluation's implementation is partial as described above.

---

## 21. Configuration & Change Guide

### 21.1 Runtime configuration (`app/config/settings.py`, `Settings` — Pydantic `BaseSettings`,
reads from `.env` via `pydantic-settings`, case-insensitive, `extra="ignore"`)

> **Note:** `.env.example` documents every setting listed below (kept in sync as of this
> milestone). If a future setting is added without updating `.env.example`, it still works via
> its environment variable using the same `UPPER_SNAKE_CASE` name as the Python field — reading
> this table (or `app/config/settings.py` directly) remains the authoritative source either way.

| Env var | Python field | Type | Default | Group | Effect |
|---|---|---|---|---|---|
| `APP_NAME` | `app_name` | `str` | `"Production AI Research & Knowledge Agent"` | Application | FastAPI title; also the default OpenRouter `X-Title` if `OPENROUTER_APP_NAME` isn't set separately |
| `APP_VERSION` | `app_version` | `str` | `"0.1.0"` | Application | FastAPI `version` |
| `ENVIRONMENT` | `environment` | `str` | `"development"` | Application | Reported by `GET /health`; not otherwise branched on in the inspected code |
| `LOG_LEVEL` | `log_level` | `str` | `"INFO"` | Application | `logging.basicConfig` level (`app/core/logging.py`) |
| `DATABASE_URL` | `database_url` | `str` | `postgresql+psycopg://postgres:postgres@localhost:5432/research_agent` | Database | SQLAlchemy async engine URL (`app/core/database.py`) |
| `STORAGE_ROOT` | `storage_root` | `str` | `"data"` | Storage | Root directory for `LocalFileStorage` |
| `EMBEDDING_MODEL` | `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embeddings | Model name passed to `SentenceTransformer(...)` |
| `RERANKER_MODEL` | `reranker_model` | `str` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Reranking | Model name passed to `CrossEncoder(...)` |
| `TRACE_ENABLED` | `trace_enabled` | `bool` | `False` | Tracing | Server-side ceiling on retrieval trace capture — see §15 |
| `LANGFUSE_ENABLED` | `langfuse_enabled` | `bool` | `False` | Observability | If false, `get_langfuse()` returns `None` and every instrumented site is a plain passthrough |
| `LANGFUSE_PUBLIC_KEY` | `langfuse_public_key` | `str \| None` | `None` | Observability | Required (raises if missing) when `LANGFUSE_ENABLED=true` |
| `LANGFUSE_SECRET_KEY` | `langfuse_secret_key` | `str \| None` | `None` | Observability | Required (raises if missing) when `LANGFUSE_ENABLED=true` |
| `LANGFUSE_BASE_URL` | `langfuse_base_url` | `str` | `"https://cloud.langfuse.com"` | Observability | Langfuse client base URL |
| `LANGFUSE_ENVIRONMENT` | `langfuse_environment` | `str` | `"development"` | Observability | Tag reported to Langfuse |
| `OPENROUTER_API_KEY` | `openrouter_api_key` | `str \| None` | `None` | OpenRouter/LLM | Required (raises if missing) by **both** the decomposition provider and the generation provider — see §11 |
| `OPENROUTER_BASE_URL` | `openrouter_base_url` | `str` | `"https://openrouter.ai/api/v1"` | OpenRouter/LLM | Base URL for both OpenRouter clients |
| `OPENROUTER_APP_NAME` | `openrouter_app_name` | `str` | `"Production AI Research & Knowledge Agent"` | OpenRouter/LLM | Sent as the `X-Title` header on both OpenRouter clients |
| `OPENROUTER_TIMEOUT_SECONDS` | `openrouter_timeout_seconds` | `float \| None` | `None` | OpenRouter/LLM (shared) | **Shared** by both providers' `AsyncOpenAI` client construction. `None` = don't override the `openai` SDK's own default timeout (distinct from passing `timeout=None` explicitly). Must be > 0 when set. |
| `OPENROUTER_MAX_RETRIES` | `openrouter_max_retries` | `int` | `2` | OpenRouter/LLM (shared) | **Shared** by both providers. Matches the `openai` SDK's own current default exactly — the SDK already retries retryable errors internally; this project adds no custom retry loop. Must be ≥ 0. |

> `openrouter_decomposition_*` and `openrouter_generation_*` below are configured **independently** — each provider only ever reads its own prefix's fields (see `app/core/dependencies.py::get_query_decomposition_provider` / `get_generation_provider`, and `tests/unit/test_query_decomposition_dependencies.py`'s cross-contamination test). Changing one task's settings never affects the other.

| Env var | Python field | Type | Default | Effect |
|---|---|---|---|---|
| `OPENROUTER_DECOMPOSITION_MODEL` | `openrouter_decomposition_model` | `str` | `"openrouter/free"` | Model used only for query decomposition |
| `OPENROUTER_DECOMPOSITION_TEMPERATURE` | `openrouter_decomposition_temperature` | `float` | `0.0` | Sampling temperature, always sent explicitly |
| `OPENROUTER_DECOMPOSITION_MAX_TOKENS` | `openrouter_decomposition_max_tokens` | `int \| None` | `None` | Omitted from the request when unset |
| `OPENROUTER_DECOMPOSITION_TOP_P` | `openrouter_decomposition_top_p` | `float` | `1.0` | Always sent explicitly |
| `OPENROUTER_DECOMPOSITION_RESPONSE_FORMAT` | `openrouter_decomposition_response_format` | `dict \| None` | `None` | Omitted when unset |
| `OPENROUTER_DECOMPOSITION_REASONING_EFFORT` | `openrouter_decomposition_reasoning_effort` | `str \| None` | `None` | Omitted when unset |
| `OPENROUTER_DECOMPOSITION_TOP_K` | `openrouter_decomposition_top_k` | `int \| None` | `None` | **OpenRouter-specific passthrough**, sent via `extra_body={"top_k": ...}` — NOT a standard OpenAI field, not every upstream model honors it. Omitted when unset. Must be a positive integer when set. |
| `OPENROUTER_DECOMPOSITION_FREQUENCY_PENALTY` | `openrouter_decomposition_frequency_penalty` | `float \| None` | `None` | Standard OpenAI field, omitted when unset. Valid range -2.0 to 2.0. |
| `OPENROUTER_DECOMPOSITION_PRESENCE_PENALTY` | `openrouter_decomposition_presence_penalty` | `float \| None` | `None` | Standard OpenAI field, omitted when unset. Valid range -2.0 to 2.0. |
| `OPENROUTER_GENERATION_MODEL` | `openrouter_generation_model` | `str` | `"openrouter/free"` | Model used only for answer generation |
| `OPENROUTER_GENERATION_TEMPERATURE` | `openrouter_generation_temperature` | `float` | `0.2` | Sampling temperature, always sent explicitly |
| `OPENROUTER_GENERATION_MAX_TOKENS` | `openrouter_generation_max_tokens` | `int \| None` | `None` | Omitted from the request when unset |
| `OPENROUTER_GENERATION_TOP_P` | `openrouter_generation_top_p` | `float` | `1.0` | Always sent explicitly |
| `OPENROUTER_GENERATION_RESPONSE_FORMAT` | `openrouter_generation_response_format` | `dict \| None` | `None` | Omitted when unset |
| `OPENROUTER_GENERATION_REASONING_EFFORT` | `openrouter_generation_reasoning_effort` | `str \| None` | `None` | Omitted when unset |
| `OPENROUTER_GENERATION_TOP_K` | `openrouter_generation_top_k` | `int \| None` | `None` | Same OpenRouter-specific `extra_body` passthrough contract as decomposition's, independent value. Must be a positive integer when set. |
| `OPENROUTER_GENERATION_FREQUENCY_PENALTY` | `openrouter_generation_frequency_penalty` | `float \| None` | `None` | Standard OpenAI field, omitted when unset. Valid range -2.0 to 2.0. |
| `OPENROUTER_GENERATION_PRESENCE_PENALTY` | `openrouter_generation_presence_penalty` | `float \| None` | `None` | Standard OpenAI field, omitted when unset. Valid range -2.0 to 2.0. |

**pgvector HNSW query-time configuration** (installed pgvector version confirmed as `0.8.6` against this project's configured database via `SELECT extversion FROM pg_extension WHERE extname='vector'`). All four below are query-time GUCs only — **none change the HNSW index's build parameters** (`m`, `ef_construction`, set once in `alembic/versions/86b2f3d39bfd_*` and never touched by these settings). Each is applied via a `SET LOCAL` inside the exact same transaction as the vector search query (`DocumentRepository.search_similar_chunks`/`_apply_hnsw_session_settings`) — PostgreSQL guarantees `SET LOCAL` reverts at that transaction's end (commit or rollback), which is what makes this safe on a pooled connection reused by a later, unrelated request; this project never issues a bare session-level `SET` for any of these.

| Env var | Python field | Type | Default | Effect |
|---|---|---|---|---|
| `HNSW_EF_SEARCH` | `hnsw_ef_search` | `int \| None` | `None` | `None` = leave pgvector's own built-in default (40) untouched. Valid range 1-1000. |
| `HNSW_ITERATIVE_SCAN` | `hnsw_iterative_scan` | `Literal["off","relaxed_order","strict_order"] \| None` | `None` | `None` = leave pgvector's own built-in default (`"off"`) untouched. |
| `HNSW_MAX_SCAN_TUPLES` | `hnsw_max_scan_tuples` | `int \| None` | `None` | `None` = leave pgvector's own built-in default (20000) untouched. Must be a positive integer when set. |
| `HNSW_SCAN_MEM_MULTIPLIER` | `hnsw_scan_mem_multiplier` | `int \| None` | `None` | `None` = leave pgvector's own built-in default (1) untouched. Valid range 1-1000. |

**Retrieval/reranking operational controls** — each default below is the *exact* value this project previously hard-coded, so leaving these unset changes nothing. Keep these four concepts distinct: `HNSW_EF_SEARCH` above (HNSW's own internal search breadth) → `RETRIEVAL_CANDIDATE_LIMIT_FLOOR` (application-level candidate-pool floor) → `RESEARCH_ASK_RESULT_LIMIT` (final result count, `/research/ask` only) → reranker input count (no separate control — the reranker always scores however many candidates the floor/explicit `candidate_limit` produced; see `CrossEncoderReranker.rerank`'s own docstring, unchanged by this milestone).

| Env var | Python field | Type | Default | Effect |
|---|---|---|---|---|
| `RETRIEVAL_CANDIDATE_LIMIT_FLOOR` | `retrieval_candidate_limit_floor` | `int` | `50` | Default `candidate_limit` floor in `RetrievalService.search` when a caller doesn't pass one explicitly (`max(limit, floor)`). Valid range 1-1000. |
| `RESEARCH_ASK_RESULT_LIMIT` | `research_ask_result_limit` | `int` | `10` | Final result count `POST /research/ask` requests from `RetrievalService.search` (not request-overridable — `POST /retrieval/search` already exposes an equivalent per-request `limit`, 1-50). Valid range 1-50. |
| `QUERY_DECOMPOSITION_MAX_SUB_QUERIES` | `query_decomposition_max_sub_queries` | `int` | `5` | Hard ceiling on kept sub-queries (including the original) in `QueryDecomposer`. Valid range 1-20. |

### 21.2 Hard-coded tunable constants (changing these requires a code change, not a config change)

| Constant | Value | File | What it controls |
|---|---|---|---|
| Semantic shredding similarity threshold | `0.7` | `app/ingestion/service.py` (literal passed to `shred_semantically`) | How aggressively adjacent paragraphs are grouped into one chunk before size-guarding |
| Chunk token budget | `500` | `app/ingestion/service.py` (literal passed to `apply_size_guard`) | Maximum "tokens" (whitespace-split words) per final chunk |
| Token estimator | whitespace word count | `app/ingestion/size_guard.py::estimate_tokens` | Not a real tokenizer — explicitly documented in code as a deterministic placeholder |
| Generation prompt / system message | fixed strings | `app/generation/openrouter.py` | The instruction given to the generation LLM |
| Decomposition prompt / system message | fixed strings | `app/retrieval/openrouter_decomposition.py` | The instruction given to the decomposition LLM |
| Embedding vector dimension | `384` | `app/core/models.py` (`Vector(384)`) | Must match the embedding model's output dimension; changing the model to a different dimension requires a schema migration |
| HNSW index build params (`m`, `ef_construction`) | pgvector library defaults (not overridden) | `alembic/versions/86b2f3d39bfd_*` | Not set in this repository at all — deliberately out of scope for the query-time settings above |
| `GenerationResult.input_tokens` etc. | — | — | *Not* hard-coded — taken from `response.usage` when present |

> Previously listed here and now Settings-backed instead (see §21.1 above): decomposition `temperature` (now `OPENROUTER_DECOMPOSITION_TEMPERATURE`, always was a field — this row was stale), decomposition max sub-queries (now `QUERY_DECOMPOSITION_MAX_SUB_QUERIES`), default `candidate_limit` (now `RETRIEVAL_CANDIDATE_LIMIT_FLOOR`), and the research final `limit` (now `RESEARCH_ASK_RESULT_LIMIT`).

### 21.3 Data/schema-dependent settings

- `EMBEDDING_MODEL` is schema-dependent: the `document_chunks.embedding` column is a
  fixed-width `Vector(384)`. Changing to a model with a different output dimension requires a
  migration to alter the column **and** re-embedding every existing chunk; even a
  same-dimension swap requires re-embedding (old and new vectors are not comparable).
- `RERANKER_MODEL`/`OPENROUTER_{DECOMPOSITION,GENERATION}_MODEL` changes take effect on the next
  process start (the reranker/embedding-provider factories are `@lru_cache`d in
  `app/core/dependencies.py`; the two OpenRouter provider factories are also `@lru_cache`d) and
  do not require any data migration — they only change how future requests are scored/generated.
- `HNSW_EF_SEARCH`/`HNSW_ITERATIVE_SCAN`/`HNSW_MAX_SCAN_TUPLES`/`HNSW_SCAN_MEM_MULTIPLIER` take
  effect on the very next request (read fresh from `Settings` per request in
  `get_retrieval_service`, applied per-transaction) — no process restart or migration needed.

### 21.4 Architectural decisions (not configuration — would require code changes to alter)

- Decomposition and generation are configured **independently** (`OPENROUTER_DECOMPOSITION_*`
  vs. `OPENROUTER_GENERATION_*`) — this is already the case, not a future possibility; a truly
  shared setting would require deliberately collapsing the two prefixes back into one.
- No fallback-model mechanism exists for either OpenRouter provider (a primary-request failure
  is not retried against a different model) — explicitly out of scope for the configuration
  milestone that added `OPENROUTER_MAX_RETRIES`/`OPENROUTER_TIMEOUT_SECONDS`; adding one is a new
  control-flow path, not just another passthrough setting.
- No caching layer exists at any level (embeddings, query decomposition, LLM responses, or
  provider prompt caching) — evaluated and deliberately deferred; see the project's own
  configuration-audit notes for the per-layer benefit/risk tradeoffs considered.
- The reranker is unconditionally present (no "retrieval without reranking" configuration path
  exists in the DI wiring, even though `RetrievalService` itself supports `reranker=None`).
- There is exactly one retrieval implementation shared by both `/retrieval/search` and
  `/research/ask` (not a configuration toggle — a structural fact about `app/core/dependencies.py`).
- Reranking has no separate "input count" setting — it always scores the entire deduplicated
  candidate pool produced by `RETRIEVAL_CANDIDATE_LIMIT_FLOOR`/an explicit `candidate_limit`;
  adding a distinct reranker-input cutoff was explicitly out of scope for this milestone.

---

## 22. "How Do I Change X?" Cookbook

**I want better retrieval recall.**
Increase `candidate_limit` by calling `RetrievalService.search(..., candidate_limit=N)` with a
larger `N` (not currently exposed through either HTTP API's request schema — you'd need to add
a field to `RetrievalSearchRequest`/`ResearchRequest` and thread it through, or change the
`max(limit, 50)` default in `app/retrieval/service.py`). Trade-off: more candidates per
sub-query means a larger deduplicated pool, which costs more reranker time.

**I want faster retrieval.**
Lower `candidate_limit` (same mechanism as above), or reduce `MAX_SUB_QUERIES` in
`app/retrieval/decomposition.py` so fewer sub-query retrieval round-trips happen per request.
Both reduce the size of the pool the reranker has to score.

**I want the reranker to see more candidates.**
Same lever as "better recall" above — `candidate_limit` is exactly the knob that controls the
pool size the reranker receives (`RetrievalService._search` reranks the **entire** deduplicated
pool, not a sub-slice of it).

**I want fewer query sub-queries.**
Lower `MAX_SUB_QUERIES` in `app/retrieval/decomposition.py` (currently `5`, hard-coded — not a
`Settings` field).

**I want a different embedding model.**
Change `EMBEDDING_MODEL` in `.env`/environment. If the new model's output dimension differs
from `384`, you must also write a migration altering `document_chunks.embedding`'s column
type. Either way, **every existing chunk must be re-embedded** — there is no automatic
re-embedding job in this codebase; it would be a manual, one-off operation you'd need to write.

**I want a different generation model.**
Change `OPENROUTER_MODEL` in `.env`/environment — note this also changes the model used for
query decomposition, since both currently share this one setting (`app/core/dependencies.py`).

**I want more deterministic decomposition/generation.**
Decomposition is already deterministic (`temperature=0`, hard-coded in
`app/retrieval/openrouter_decomposition.py`). **Generation has no temperature set at all** —
it uses whatever default the OpenRouter model applies. To make it deterministic, you would need
to add `temperature=0` (or another value) to the `chat.completions.create(...)` call in
`app/generation/openrouter.py`; there is currently no `Settings` field for this.

**I want to disable retrieval tracing.**
Set `TRACE_ENABLED=false` (the default). This makes `effective_trace` always `False`
regardless of what any client requests, per the four-case table in §15 — no frontend change
needed.

**I want to make a READY historical version current.**
API: `POST /documents/{logical_document_id}/versions/{version_id}/set-current` (no body).
UI: in the Document Library's version history, click "Make Current" on any READY, non-current
version row (`DocumentWorkspace.tsx`).

**I want to add another supported document type.**
Implement a new `DocumentLoader` subclass under `app/ingestion/loaders/` (see
`markdown.py`/`pdf.py`/`docx.py` for the pattern — just `load(path) -> list[LoadedPage]`), then
register its file extension(s) in `LOADERS_BY_SUFFIX` in `app/ingestion/loaders/resolver.py`.
No other ingestion code needs to change — `IngestionService` and the upload route already
resolve the loader generically by extension. Add unit tests mirroring
`tests/unit/test_markdown_loader.py`/`test_pdf_loader.py`/`test_docx_loader.py`, and consider
an API contract test mirroring the existing `.docx` coverage in
`tests/integration/test_documents_api.py`.

---

## 23. Testing

- **`tests/unit/`** — pure-function/class tests with no database: chunking
  (`test_chunker.py`, `test_structure_chunker.py`, `test_table_chunker.py`,
  `test_table_serializer.py`, `test_size_guard.py`, `test_semantic_shredder.py`,
  `test_structure_extractor.py`, `test_section_builder.py`, `test_section_map.py`), loaders
  (`test_markdown_loader.py`, `test_pdf_loader.py`, `test_docx_loader.py`,
  `test_loader_resolver.py`), decomposition (`test_query_decomposer.py`,
  `test_openrouter_decomposition.py`, `test_query_decomposition_dependencies.py`,
  `test_retrieval_decomposition.py`), retrieval internals (`test_retrieval_service.py`,
  `test_retrieval_trace.py`, `test_cross_encoder.py`), generation
  (`test_generation_service.py`, `test_generation_generation_service.py`), embeddings
  (`test_embedding_provider.py`, `test_embeddings.py`, `test_sentence_transformer_embedding.py`),
  Langfuse (`test_langfuse.py`, `test_langfuse_generation.py`, `test_langfuse_retrieval.py`,
  `test_research_langfuse.py`), health (`test_health.py`), and more.
- **`tests/integration/`** — split into two styles:
  - **Fake-based FastAPI contract tests** (e.g. `test_documents_api.py`,
    `test_retrieval_api.py`, `test_research_api.py`) — use `TestClient`/`AsyncClient` with
    `app.dependency_overrides` pointing at hand-written fakes (`FakeIngestionService`,
    `FakeDocumentRepository`, `FakeRetrievalService`, etc.), so no real DB, embedding model, or
    network call is involved. These pin the exact HTTP contract (status codes, response
    shapes, 404-vs-409 semantics).
  - **Real-database tests** (e.g. `test_document_service.py`, `test_document_repository.py`,
    `test_chunk_embeddings.py`, `test_vector_retrieval.py`, several `test_retrieval_search_real_database*`
    functions inside `test_retrieval_api.py`) — connect to the actual configured
    `DATABASE_URL`, use the **real** `SentenceTransformerEmbeddingProvider` (deterministic
    given the same model/input, but not a mock), and commit real rows, typically cleaning up
    by hand with `session.delete(...)` / `session.commit()` at the end of each test.
- **`tests/evaluation/`** — see §20.
- **`tests/conftest.py`** — two responsibilities:
  1. On Windows, sets `asyncio.WindowsSelectorEventLoopPolicy()` (required for `psycopg`'s
     async driver on Windows).
  2. A **`pytest_sessionfinish`** hook that unconditionally deletes every row from
     `ChunkPageMap`, `DocumentChunk`, `DocumentSection`, `DocumentPage`, `StoredFile`, and
     `Document` (in that FK-safe order) and removes every file under `Settings.storage_root`
     except `.gitkeep`, **after the entire pytest session finishes, regardless of pass/fail**.
     This exists because real-database integration tests commit directly into whatever
     database `DATABASE_URL` points at — **there is no separate test database and no
     per-test transaction/SAVEPOINT isolation** — so if a test's own hand-written cleanup
     never runs (e.g. an assertion failed first), this hook is the guaranteed backstop that
     prevents test fixtures from persisting into whatever the developer is pointing
     `DATABASE_URL` at (commonly a local dev database also used for manual testing). A
     cleanup failure is reported loudly to the terminal but does not fail the test run.
- **External-network-dependent test**: `tests/integration/test_openrouter_generation.py::test_real_openrouter_generation`
  calls the real OpenRouter API if `OPENROUTER_API_KEY` is configured (it `pytest.skip()`s
  otherwise). It is marked `@pytest.mark.integration`. This is a real, acknowledged exception
  to "no network dependency in tests" — it exists specifically to validate the real
  integration, and can fail for reasons outside the codebase (e.g. the free-tier daily rate
  limit being exhausted), which is not a code regression.
- **Frontend**: no test runner is configured (`package.json` has no `test` script). Correctness
  is checked via TypeScript (`npm run lint` → `tsc --noEmit`) and a production build
  (`npm run build` → `tsc && vite build`), not unit/component tests.

---

## 24. Developer Workflow

Commands below are the actual scripts/entry points defined in this repository.

```bash
# --- Backend setup ---
# Python 3.12 required (pyproject.toml: requires-python = ">=3.12,<3.13")
# Install dependencies per your tool of choice from pyproject.toml's [project.dependencies]
# and [dependency-groups.dev].

# Configure environment: copy .env.example to .env and fill in DATABASE_URL at minimum.
# OPENROUTER_API_KEY is required for /retrieval/search and /research/ask to work at all (§11).
cp .env.example .env

# --- Database migrations ---
alembic upgrade head

# --- Run the backend ---
uvicorn app.main:app --reload

# --- Run the frontend ---
cd frontend
npm install
npm run dev        # Vite dev server
npm run build       # tsc && vite build — production build + type-check
npm run lint          # tsc --noEmit — type-check only

# --- Backend tests ---
pytest                                   # full suite (requires a reachable DATABASE_URL —
                                          # real-database integration tests are not mocked out)
pytest tests/unit                          # no database required
pytest tests/integration/test_retrieval_api.py -q

# --- Lint/format (Python) ---
ruff check .
ruff format --check .
```

Practical flows:

- **Upload a document** → Document Library → "Upload / Ingest Document" → choose "Upload as
  new document" → pick a file → set `document_type` → submit (`POST /documents`).
- **Upload a new version** → same modal → "Upload as new version of" → pick the existing
  logical document from the dropdown → submit (same `POST /documents`, with
  `logical_document_id` set).
- **Retry a failed version** → Document Library → open the document → "Retry" on the FAILED
  version row (`POST .../retry`).
- **Change current version** → "Make Current" on any READY, non-current version row
  (`POST .../set-current`).
- **Delete a version / whole document** → "Delete" on a version row, or "Delete Logical
  Document" on the detail view (`DELETE .../versions/{id}` or `DELETE /documents/{id}`).
- **Inspect retrieval** → Retrieval Explorer tab, toggle Trace Mode in the header, run a query.
- **Inspect Langfuse** → configure `LANGFUSE_ENABLED=true` + keys, then view traces in the
  Langfuse dashboard at `LANGFUSE_BASE_URL` — this repository does not provide an embedded
  Langfuse UI.

---

## 25. End-to-End Example

**A user uploads `annual_report.pdf` as document type `research_paper`.**

1. `POST /documents` receives the multipart body; `PDFLoader` is resolved from the `.pdf`
   extension.
2. `IngestionService.ingest_file` extracts per-page text via `pypdf.PdfReader`.
3. `DocumentService.ingest_document` hashes the combined extracted text; no matching
   `(logical_document_id, content_hash)` exists, so a **new logical document** is created:
   `version_number=1`, `status=UPLOADED`, `is_current=False`.
4. The raw PDF bytes are hashed separately (`StoredFile.content_hash`) and stored via
   `LocalFileStorage.store` at `documents/{logical_document_id}/{version_id}/{file_id}`; a
   `StoredFile` row is committed.
5. `mark_processing` → `status=PROCESSING`, `processing_attempt=1`.
6. `StructureExtractor` turns each page's text into headings/paragraphs/lists;
   `SectionBuilder`/`SectionService` persist the resulting section hierarchy.
7. `shred_semantically` groups adjacent paragraphs within the same section whose embeddings
   are `>= 0.7` cosine-similar; `apply_size_guard` splits anything over 500 estimated tokens.
8. `ChunkService.persist_chunks` embeds every final chunk in one batch call
   (`all-MiniLM-L6-v2`, 384-dim) and persists `DocumentChunk` + `ChunkPageMap` rows.
9. `mark_ready` → `status=READY`, `is_current=True`. This logical document now has exactly one
   searchable version.
10. A user asks: *"What were the company's revenue and headcount trends, and how did
    management explain the changes?"* via `POST /research/ask`.
11. `QueryDecomposer` (if `OPENROUTER_API_KEY` is configured) may split this into, e.g.:
    `["What were the company's revenue and headcount trends, and how did management explain
    the changes?", "What was the company's revenue trend?", "What was the company's headcount
    trend?", "How did management explain these changes?"]` (original always first; exact
    output depends on the LLM — this is illustrative, not a guaranteed output).
12. Each sub-query is embedded and run through `search_similar_chunks` independently (filtered
    to `status=READY AND is_current=True` — this document's version 1 qualifies).
13. Candidates are merged by `chunk_id`, keeping each chunk's best (lowest) distance seen
    across sub-queries; counts are tracked as `raw_candidate_count`/`deduplicated_candidate_count`.
14. The entire deduplicated pool is reranked **once**, against the *original* question, by
    `CrossEncoderReranker`.
15. The top results (10, for `/research/ask`) are kept.
16. `ContextAssembler` turns them into `[Source 1]\n...`-formatted text plus a source list.
17. `OpenRouterGenerationProvider` calls the configured `OPENROUTER_MODEL` with that context
    and the original question; the answer text and token usage are returned.
18. If `trace=true` **and** `TRACE_ENABLED=true`, the response also includes the full
    decomposition/candidate/rerank trace (§15); otherwise `trace_available=false` and, if
    `trace=true` was requested anyway, `trace_unavailable_reason="server_disabled"`.
19. If `LANGFUSE_ENABLED=true`, three Langfuse observations were recorded along the way
    (research-request span, retriever span, generation span) — independently of whether the
    retrieval trace was captured.

**If ingestion had failed instead** (step 7 or 8 throwing, e.g. an embedding call failing):
the session is rolled back, the `Document` row is re-fetched and marked `FAILED` with
`last_error` set, `StoredFile` and the physical PDF are untouched, and `POST /documents`
still returns **201** with that FAILED version — the user can then call
`POST .../retry` to reprocess the exact same stored file without re-uploading.

**If a `version 2` of `annual_report.pdf` is later uploaded with corrected numbers:** a new
`Document` row is created (`version_number=2`, `is_current=False`, `status=UPLOADED`), goes
through the same pipeline, and on reaching `READY` automatically demotes version 1
(`is_current=False`) and becomes current itself. Version 1's chunks/embeddings are **not**
deleted — they remain in `document_chunks`, just excluded from retrieval by the
`is_current`/`status` filter, and can be made current again later via `set-current` without
any re-ingestion.

---

## 26. Important Design Invariants

Each of these was verified directly against the implementation referenced.

1. **A logical document can have multiple versions**, each its own `Document` row sharing one
   `logical_document_id`. (`app/core/models.py`, `app/core/services/document.py`)
2. **At most one version per logical document may have `is_current=True`**, enforced both by a
   partial unique index (`ix_documents_current_version`) and by `DocumentService`'s
   demote-then-promote sequencing. (`alembic/versions/9f0853bbd01b_*`, `app/core/services/document.py`)
3. **Re-uploading byte-for-byte-equivalent extracted content under the same
   `logical_document_id` is a no-op** (returns the existing version unchanged), enforced at
   the DB level by `uq_documents_logical_content_hash`. (`app/core/services/document.py`)
4. **Only a `READY` version may become current** — `set_current` raises for any other status;
   already-current is idempotent. (`app/core/services/document.py::set_current`)
5. **Only `READY` and `is_current=True` chunks participate in normal retrieval** —
   enforced in the `WHERE` clause of `search_similar_chunks`, not by deleting anything.
   (`app/core/repositories/document.py`)
6. **Historical (non-current) versions' chunks and embeddings remain stored** and become
   searchable again immediately if that version is made current again — nothing about
   current-version switching touches `DocumentChunk`/`DocumentPage`/`DocumentSection` rows.
   (`app/core/services/document.py::set_current`, `app/core/repositories/document.py`)
7. **A FAILED version retains its `StoredFile` and physical file**, which is what makes retry
   possible without re-upload. (`app/ingestion/service.py::_process_document`,
   `app/ingestion/service.py::retry_document`)
8. **The physical file for a version is only deleted after its DB deletion has committed**,
   never before. (`app/core/services/document_deletion.py`)
9. **Query decomposition always preserves the original query** as the first, non-removable
   entry in `sub_queries`. (`app/retrieval/query_decomposer.py::_normalize_queries`)
10. **Decomposition produces 1–5 unique, non-empty queries** (deduplicated case-insensitively).
    (`app/retrieval/query_decomposer.py`, `MAX_SUB_QUERIES` in `app/retrieval/decomposition.py`)
11. **Decomposition failure falls back to `[original_query]`** — any exception from the
    provider is caught, never propagated to the caller. (`app/retrieval/query_decomposer.py`)
12. **Candidates from every sub-query are merged and deduplicated (keeping the best distance
    per `chunk_id`) before reranking** — reranking never happens per sub-query.
    (`app/retrieval/service.py::_search`)
13. **Exactly one global reranker pass**, over the whole deduplicated pool, using the
    original query — not per-sub-query, not per-candidate-batch. (`app/retrieval/service.py`)
14. **`trace` (request-level) and `TRACE_ENABLED` (server-level) together gate only whether a
    trace is captured/returned — never whether decomposition or retrieval execute.**
    (`app/retrieval/service.py::_search`; regression test
    `tests/unit/test_retrieval_decomposition.py::test_decomposition_and_merge_happen_even_when_trace_is_disabled`)
15. **The Retrieval Explorer's trace is distinct from Langfuse observability** — independently
    configured, independently gated, and carrying different (and differently-shaped) data.
    (§16)
16. **Storage keys never derive from user-supplied filenames**, and `LocalFileStorage`
    verifies every resolved path stays under its configured root before any file operation.
    (`app/storage/keys.py`, `app/storage/local.py::_resolve_path`)
17. **Table chunks participate in the exact same persistence, provenance, and retrieval
    system as prose chunks** — no separate table-retrieval path exists. A table chunk is only
    distinguished by `chunk_metadata["content_type"] == "table"`; it is still filtered by the
    same `status=READY AND is_current=true` rule, still subject to the same current-version
    switching and deletion semantics, and still flows through `ContextAssembler` unchanged.
    (`app/ingestion/size_guard.py::fragment_table_units`, `app/ingestion/chunk_service.py`;
    `tests/integration/test_table_retrieval.py`)

---

## 27. Known Limitations / Deferred Work

These are factual gaps found in the repository, not a speculative roadmap.

- **`OPENROUTER_API_KEY` is effectively required**, not optional, for both `/retrieval/search`
  and `/research/ask` — their dependency chains eagerly construct an OpenRouter-backed query
  decomposer even though `RetrievalService` itself supports running with no decomposer at all.
  See §11.
- **Table-aware chunking IS wired into the active ingestion pipeline** (`table_chunker.py`,
  `table_serializer.py`, called from `StructureExtractor`/`size_guard.fragment_table_units` —
  see §8). What remains unused is only the alternate candidate-grouping strategy in
  `structure_chunker.py` (plus the unrelated `chunker.py` and the `testing_similarity.py` test
  double), which `IngestionService` does not call. Table detection is Markdown-pipe-table-only:
  supported for `.md` files (native syntax) and `.docx` (python-docx's structured
  `document.tables`, rendered as the same pipe syntax). **PDF tables are not supported** —
  `pypdf.extract_text()` has no layout/table structure to extract, so a table in a PDF ingests
  as plain, often-garbled paragraph text. Single-column tables and a table whose single row
  (with headers) alone exceeds the 500-token chunk budget are also not handled (see §8).
- **The evaluation dataset is a placeholder.** `tests/evaluation/retrieval_cases.py` contains
  one case with invalid `UUID("...")` literals and is not imported/run by anything — see §20.
- **No reconciliation/orphan-cleanup job exists** for a physical file left behind if deletion
  fails after its DB row is already committed (§6, §19) — this is explicitly deferred, not
  automated.
- **Ingestion is synchronous, in-request work**, not a background job/queue — `POST /documents`
  blocks until chunking/embedding finishes (or fails). `app/api/documents.py`'s own docstring
  states this directly ("mirrors the existing `IngestionService.ingest_file` flow exactly and
  does not introduce any background job, queue, or async task semantics").
- **`test_real_openrouter_generation` depends on a live OpenRouter call** and real account
  quota — it is explicitly marked `@pytest.mark.integration` and self-skips without an API key,
  but it is a genuine, acknowledged external-network dependency in the test suite (§23).
- **`.env.example` under-documents configuration** — only 5 of the ~17 settings in `Settings`
  appear in it (§21.1).
- **`app/agents/`, `app/evaluation/`, `app/llm/`, `app/memory/`, and `docker/` are empty
  directories** — no files exist in any of them. They appear to be placeholders for the
  `README.md` roadmap's later phases (agentic research, vLLM serving, memory, containerization)
  and currently contain no implementation whatsoever.
- **Generation has no temperature, max-token, or other sampling parameter set at all** — see
  §14/§21.2. This is a real gap if deterministic or length-bounded answers are required.
- **`was_decomposed`/`fallback_reason` from `QueryDecompositionResult` are computed but never
  surfaced** through the trace or API (§11) — there is currently no way for a caller to
  directly observe "decomposition was attempted and failed" versus "the query simply had one
  information need," other than `sub_queries` having length 1 in both cases.

---

## 28. Glossary

| Term | Meaning in this codebase |
|---|---|
| **Logical document** | The stable identity (`logical_document_id`) shared by every ingested version of "the same" source document, across content changes. |
| **Version** | One specific ingested attempt — a single `Document` row, identified by its own `id`, belonging to exactly one logical document and one `version_number`. |
| **Current version** | The one version (if any) of a logical document with `is_current=True`; the only version whose chunks normal retrieval can return. |
| **Chunk** | A `DocumentChunk` row — the unit of text that gets embedded and retrieved, produced by the semantic-shredding + size-guard pipeline (§8). |
| **Embedding** | The `list[float]`/`Vector(384)` produced by the configured `EMBEDDING_MODEL` for a chunk or a query; stored on `DocumentChunk.embedding`. |
| **Candidate pool** | The merged, deduplicated set of chunks retrieved across all of a query's sub-queries, before reranking (§10, §12). |
| **Reranking** | The single global `CrossEncoderReranker` pass over the candidate pool, producing `rerank_score` and the final ordering (§13). |
| **Decomposition** | Splitting a multi-intent query into 1–5 focused retrieval sub-queries via an LLM, with the original query always preserved (§11). |
| **Retrieval trace** | The optional, per-request `RetrievalTrace`/`RetrievalTraceResponse` payload capturing decomposition, candidate pool, and final results for one retrieval execution (§15). |
| **RAG** | Retrieval (§10) → context assembly (§14) → generation (§14), as orchestrated by `POST /research/ask`. |
| **Langfuse** | The optional external observability backend recording spans for retrieval/generation calls, independent of the retrieval trace (§16). |
| **Provenance** | The `document_id`/`chunk_id`/`section_id`/`section_path`/`page_numbers` attached to every retrieved chunk and generation source, tracing it back to its exact origin in the source document. |
