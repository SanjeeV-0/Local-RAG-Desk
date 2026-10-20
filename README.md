# Local RAG Desk

**A local-first research workspace for document retrieval, grounded answers, and AI-assisted knowledge exploration.**

Local RAG Desk combines semantic search, reranking, document management, and LLM-powered research in a single application. It lets you explore your documents, retrieve relevant evidence, and ask questions about your knowledge base through a web interface.

Built with FastAPI, PostgreSQL, pgvector, Sentence Transformers, and OpenRouter.

## Overview

Local RAG Desk is designed to make personal and project documentation easier to search, understand, and analyze using Retrieval-Augmented Generation (RAG).

Rather than relying entirely on a language model's pretrained knowledge, the application retrieves relevant passages from your indexed documents and supplies them as context for research responses.

The project focuses on retrieval quality, source-grounded answers, observability, and safe handling of document content.

## Features

### Research & Synthesis
- Ask natural-language questions about your indexed knowledge base.
- Retrieve relevant document passages to ground research responses.
- Generate answers using an OpenRouter-compatible language model.
- Fall back to an extractive summary when generation is unavailable, with the fallback clearly identified.
- Preserve source attribution so answers can be traced to retrieved evidence.

### Semantic Retrieval
- Generate document embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Store and search embeddings in PostgreSQL with pgvector.
- Use an HNSW index with cosine distance for vector retrieval.
- Apply cross-encoder reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Merge and deduplicate candidate results across query variants.
- Support Reciprocal Rank Fusion (RRF) as a ranking strategy.

### Query Processing
- Retain the original query while optionally decomposing complex questions into subqueries.
- Support a configurable query-complexity router, disabled by default.
- Expose retrieval configuration for experimentation and evaluation.

### Document Library & Versions
- Manage indexed documents and their versions.
- Deduplicate repeated content where supported by the ingestion pipeline.
- Preserve document metadata and chunk-level provenance for retrieval.

### Vector Retrieval Explorer
- Inspect retrieved candidates and ranking behavior.
- Explore how retrieval configuration affects the results.
- Investigate candidate selection and reranking without relying only on final generated answers.

### Observability & Error Handling
- Integrate Langfuse for tracing retrieval and answer generation.
- Capture generation completion metadata, including the model's finish reason when available.
- Classify provider errors and record sanitized diagnostic information.
- Preserve a safe extractive fallback when answer generation fails.

### Privacy & Security
- Apply PII redaction to supported sensitive data in retrieved context before it reaches the generation model.
- Preserve provider-error sanitization and safe error responses.
- Treat retrieved documents as evidence, not trusted instructions.
- Recognize prompt injection, jailbreaks, instruction overrides, and data-exfiltration attempts as security risks requiring dedicated testing.

**Security note:** These protections reduce risk but do not guarantee that every sensitive value or prompt-injection attempt will be detected. Review the configured provider's data-processing policies and guardrails before using sensitive documents.

## Architecture

```text
                 ┌────────────────────────┐
                 │      React + Vite      │
                 │   TypeScript frontend  │
                 └───────────┬────────────┘
                             │ HTTP API
                 ┌───────────▼────────────┐
                 │       FastAPI          │
                 │ Research & Documents   │
                 └───────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
      ┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
      │ Query        │ │ Retrieval  │ │ Generation │
      │ Processing   │ │ & Reranking│ │ Service    │
      └──────────────┘ └─────┬──────┘ └─────┬──────┘
                             │              │
                    ┌────────▼────────┐ ┌───▼──────────┐
                    │ PostgreSQL      │ │ OpenRouter   │
                    │ + pgvector      │ │ LLM API      │
                    └─────────────────┘ └──────────────┘
```

Langfuse provides observability for supported application operations.

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| Database | PostgreSQL |
| Vector search | pgvector |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Language model provider | OpenRouter |
| Tracing | Langfuse |
| Database migrations | Alembic |

## How Retrieval Works

The retrieval pipeline is designed to find relevant evidence before generation.

1. **Query processing:** The original query is retained, with optional decomposition into subqueries.
2. **Embedding:** Queries are embedded into 384-dimensional vectors.
3. **Vector search:** PostgreSQL and pgvector retrieve semantically similar chunks.
4. **Candidate merging:** Results from query variants are merged and deduplicated by chunk ID.
5. **Ranking:** Candidate passages are ranked using the configured ranking strategy, including cross-encoder reranking or RRF.
6. **Context preparation:** Selected passages are assembled with source information and supported PII redaction.
7. **Answer generation:** The generation service submits the prepared context to the configured language model.
8. **Fallback:** If generation fails and sources are available, the application can return a clearly labelled extractive summary instead.

An extractive fallback is not equivalent to a synthesized model answer.

## Getting Started

### Prerequisites

- Python 3.12
- Node.js and npm
- PostgreSQL with the pgvector extension
- An OpenRouter API key for LLM generation
- Optional: a Langfuse project for tracing

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your repository URL and directory name.

### 2. Set up the Python environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your project uses a different dependency-management file, follow the repository's existing setup instructions.

### 3. Configure environment variables

Create a local `.env` file using the configuration expected by the application.

At minimum, configure the database connection and OpenRouter API key. Configure the generation model using the application's supported settings.

Example structure:

```dotenv
DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/DATABASE_NAME

OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
OPENROUTER_MODEL=openrouter/free

LANGFUSE_ENABLED=false
```

This is an illustrative starting point, not a complete configuration template. Check the application's settings for all required variables and defaults before running it. Keep `.env` out of version control.

### 4. Prepare the database

Ensure PostgreSQL and pgvector are available, then apply the project's Alembic migrations using the repository's configured migration workflow.

For example:

```bash
alembic upgrade head
```

### 5. Start the backend

Use the application's existing FastAPI entry point. For example, if the entry point is `app.main:app`:

```bash
uvicorn app.main:app --reload
```

The API is typically available at:

```text
http://127.0.0.1:8000
```

Check the health endpoints supported by the application to verify startup and readiness.

### 6. Start the frontend

From the frontend directory:

```bash
npm install
npm run dev
```

Open the local URL printed by Vite.

> Exact installation commands, environment variables, and scripts should be verified against the current repository before publishing this README.

## Using Local RAG Desk

1. Start PostgreSQL, the backend, and the frontend.
2. Add documents through the document-management interface.
3. Allow ingestion and embedding to complete.
4. Use **Research & Synthesis** to ask questions about the indexed content.
5. Use **Vector Retrieval Explorer** to inspect retrieval behavior.
6. Use **Document Library & Versions** to review indexed documents and their versions.
7. When a response is labelled as an extractive fallback, treat it as a partial summary rather than a successful synthesized answer.

## Evaluation & Testing

The repository includes unit tests, integration tests, and retrieval-evaluation utilities.

Run the unit tests using the project's configured test runner. For example:

```bash
pytest tests/unit
```

Run selected integration tests according to the repository's database and service requirements.

Retrieval evaluation should measure metrics such as:

- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)
- Hit rate

Changes to retrieval, ranking, context assembly, and generation should be evaluated separately where possible. A successful HTTP response alone does not prove that an answer is complete, accurate, or well grounded.

## Known Limitations

- Answer quality depends on retrieval relevance, context selection, the configured model, and the model provider's availability and policies.
- Free-model routing may select different backend models and can have variable availability.
- Provider-side safety filters can reject requests, including requests involving personal information.
- PII detection is not comprehensive; person-name detection in particular is not guaranteed by the existing regex-based detector.
- Prompt-injection detection and guardrails do not guarantee immunity to malicious instructions in retrieved documents.
- An extractive fallback may be incomplete and should not be mistaken for a generated synthesis.
- End-to-end answer quality must be validated independently from retrieval metrics and provider connectivity.

## Roadmap

Potential areas for continued improvement include:

- More natural synthesis across multiple retrieved passages.
- Better detection of truncated or incomplete answers.
- Expanded prompt-injection and adversarial-document evaluations.
- Improved generation failure diagnostics and user-facing error guidance.
- More systematic evaluation of retrieval and ranking configurations.
- Stronger privacy-aware handling of sensitive document content.

## Contributing

Contributions, bug reports, and focused evaluation improvements are welcome.

When proposing changes:
- Preserve source provenance and existing security controls.
- Include regression tests for behavior changes.
- Avoid committing secrets, personal data, or private documents.
- Report evaluation results honestly, including known limitations.

## License

Add the license that applies to this repository before publishing or distributing the project.

---

**Local RAG Desk** — explore your documents, retrieve relevant evidence, and build a more useful local research workflow.
