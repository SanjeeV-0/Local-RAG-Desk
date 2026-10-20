I suggest we create a living project engineering document that we update at major milestones.

Local RAG Desk — Engineering Notes
1. Project Goal

We are building a production-grade AI research and engineering knowledge agent over a controlled corpus of AI/LLM/RAG/Agentic-AI technical documents.

The eventual system will support:

Documents
   ↓
Multimodal ingestion
   ↓
Document normalization
   ↓
Chunking
   ↓
Embeddings
   ↓
Hybrid retrieval
   ↓
Reranking
   ↓
Agentic research
   ↓
Evidence verification
   ↓
LLM generation
   ↓
Citations

We are deliberately building this incrementally.

Our principle is:

Don't add a technology because it is popular. Add it when it solves a measurable problem.

2. Development Environment
Python 3.12

We chose Python 3.12 instead of Python 3.14 because the project will eventually use a large AI ecosystem, and Python 3.12 currently gives us a safer compatibility baseline.

We created:

.venv/

This isolates project dependencies from the global Python installation.

Activation:

.\.venv\Scripts\Activate.ps1
3. pyproject.toml

This is the central project configuration file.

It defines:

Project metadata
Dependencies
Python version
pytest configuration
Ruff configuration
Build configuration

For example:

requires-python = ">=3.12,<3.13"

means this project explicitly targets Python 3.12.

We also configured Ruff:

[tool.ruff]
line-length = 100
target-version = "py312"

So our codebase has a consistent coding standard.

4. Git

We initialized Git so every major architectural milestone can be tracked.

Our first commit represented:

Project foundation

Later we'll have commits such as:

feat: add ingestion pipeline
feat: add document chunking
feat: add vector retrieval
feat: add hybrid retrieval
feat: add agentic research workflow

This is much better than building the entire project and having one giant commit.

5. FastAPI

We created:

app/main.py

with:

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

FastAPI is our application/API layer.

Conceptually:

Client
   ↓
FastAPI
   ↓
Application Services
   ↓
Database / RAG / Agents / LLM

We started with:

GET /health

which verifies that the application process is alive.

6. Testing with pytest

We created:

tests/unit/test_health.py

The test uses:

TestClient(app)

to call:

GET /health

and verify:

{
    "status": "healthy"
}

This establishes an important engineering principle:

Code
 ↓
Test
 ↓
Pass
 ↓
Continue

We don't want to build 20 components and discover at the end that something fundamental broke.

7. Ruff

Ruff is our static analysis and code quality tool.

We run:

ruff check .

It catches things such as:

Import problems
Unused variables
Formatting issues
Potential bugs
Line length

We encountered this ourselves when Ruff detected:

E501 Line too long

Rather than ignoring it, we fixed the code.

8. Pydantic Settings

We created:

app/config/settings.py

with:

class Settings(BaseSettings):
    ...

This gives us centralized configuration.

The architecture is:

.env
 ↓
Pydantic Settings
 ↓
Application

Instead of hardcoding:

DATABASE_URL = "..."

throughout the application.

Later this will contain things like:

DATABASE_URL
LLM_MODEL
EMBEDDING_MODEL
VECTOR_DB_URL
REDIS_URL
LANGSMITH_API_KEY
9. .env vs .env.example

We created:

.env.example

which contains the structure of our configuration but no secrets.

Actual credentials go into:

.env

and .gitignore prevents .env from being committed.

Architecture:

.env.example
     ↓
Team knows required variables

.env
     ↓
Local secrets
     ↓
NOT committed

This is basic but important security hygiene.

10. Logging

We created:

app/core/logging.py

Instead of:

print("Application started")

we use Python's logging system:

logger.info("Application started")

Our format is:

timestamp | level | logger | message

For example:

2026-08-24 17:03:12 | INFO | app.main | Application started

This becomes important when we eventually introduce:

LangSmith
LangFuse
Cloud logging
Distributed tracing

because production systems need observable behavior.

11. Liveness vs Readiness

We introduced two concepts.

Liveness
/health

Answers:

Is the application process alive?

Readiness
/health/ready

Answers:

Is the application capable of serving requests?

Currently readiness checks:

Database

Eventually it can check:

Database
Redis
Vector database
LLM server
Embedding service

This distinction is important in production deployments.

12. PostgreSQL

We chose PostgreSQL as our primary relational database.

We created:

research_agent

and verified:

PostgreSQL 18.6

Architecture:

Application
     ↓
SQLAlchemy
     ↓
Psycopg
     ↓
PostgreSQL

PostgreSQL will eventually store things such as:

Document metadata
Chunks
Sources
Conversations
Memory
Evaluation data
Relationships
13. SQLAlchemy

We use SQLAlchemy as our ORM/database abstraction.

We created:

app/core/database.py

The important concept is:

AsyncEngine
      ↓
AsyncSession
      ↓
SQL queries

We use asynchronous database operations because our FastAPI application will perform many I/O-heavy operations.

For example:

API request
   ↓
Database query
   ↓
waiting...
   ↓
continue

The application doesn't need to block an entire worker while waiting for database I/O.

14. SQLAlchemy Base

We created:

class Base(DeclarativeBase):
    pass

This is the root of our ORM model hierarchy.

Later:

Base
 ├── Document
 ├── Chunk
 ├── Source
 ├── Conversation
 └── Evaluation

Alembic uses:

Base.metadata

to understand our database schema.

This is also why we couldn't call an ORM field metadata directly—it is reserved by SQLAlchemy's Declarative API.

We solved it with:

document_metadata = mapped_column("metadata", ...)

So:

Python:
document_metadata

PostgreSQL:
metadata
15. Database Health Check

We created:

app/core/database_health.py

which executes:

SELECT 1

The idea is:

Application
    ↓
Can I reach PostgreSQL?
    ↓
SELECT 1

This gives us a lightweight readiness check.

16. Alembic

We installed Alembic for database schema migrations.

Without migrations, someone might manually modify:

PostgreSQL tables

which becomes difficult to reproduce.

With Alembic:

SQLAlchemy model
       ↓
Alembic migration
       ↓
PostgreSQL schema

We initialized:

alembic/
├── env.py
├── versions/
├── script.py.mako
└── README

and connected it to:

Base.metadata
17. First Database Model — Document

Our first domain model is:

Document

It represents the source document, not its chunks.

Current conceptual schema:

Document
│
├── id
├── title
├── authors
├── source
├── publication_date
├── document_type
├── content_hash
├── metadata
├── created_at
└── updated_at
Why content_hash?

This is important for ingestion.

Suppose we receive:

Paper.pdf

today and receive the same paper again tomorrow.

We can calculate:

SHA-256(document content)

and compare it with:

content_hash

Because the database column is:

UNIQUE

we have database-level protection against duplicate documents.

18. Database Migration

We generated:

create_documents_table

with Alembic and applied it using:

alembic upgrade head

Our PostgreSQL database now contains:

documents
alembic_version

This was our first real schema migration.

19. Integration Testing

We created:

tests/integration/test_database.py

This test does:

Python
 ↓
SQLAlchemy
 ↓
Psycopg
 ↓
PostgreSQL
 ↓
SELECT 1

We also created a document persistence test:

tests/integration/test_document_repository.py

which verifies:

Create Document
      ↓
INSERT
      ↓
PostgreSQL
      ↓
SELECT
      ↓
Verify
      ↓
DELETE test record

This is different from a unit test.

Unit test

Tests a component in isolation.

Integration test

Tests whether multiple real components work together.

20. Repository Pattern

We created:

app/core/repositories/document.py

The repository handles database operations such as:

get_by_id()
get_by_content_hash()
create()

Architecture:

Service
   ↓
Repository
   ↓
SQLAlchemy
   ↓
PostgreSQL

The repository's job is:

How do I access the data?

It should not contain business decisions.

21. Service Layer

We created:

app/core/services/document.py

The service sits above the repository.

Architecture:

API
 ↓
Service
 ↓
Repository
 ↓
Database

The service answers:

What should the application do?

The repository answers:

How do I retrieve/store the data?

This separation becomes particularly valuable once ingestion becomes complex.

22. Current Architecture

At this exact point, our project looks like:

                    ┌───────────────┐
                    │    Client     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   FastAPI     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    Service    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Repository   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  SQLAlchemy   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  PostgreSQL   │
                    └───────────────┘

With cross-cutting infrastructure:

        ┌─────────────────────────┐
        │ Configuration           │
        │ Pydantic Settings       │
        ├─────────────────────────┤
        │ Logging                 │
        ├─────────────────────────┤
        │ Testing                 │
        │ pytest + Ruff           │
        ├─────────────────────────┤
        │ Migrations              │
        │ Alembic                 │
        └─────────────────────────┘
23. What We Have NOT Built Yet

This is important.

We have not yet built the AI/RAG part.

We still need:

PDF/document ingestion
        ↓
Layout-aware parsing
        ↓
Document normalization
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector database
        ↓
HNSW
        ↓
BM25
        ↓
Hybrid retrieval
        ↓
Cross-encoder reranking
        ↓
HyDE
        ↓
Semantic sharding
        ↓
LLM inference
        ↓
LangGraph
        ↓
Memory
        ↓
Evidence verification
        ↓
Evaluation
        ↓
Observability

That's intentional.

We first established a production software foundation rather than immediately writing a load_pdf → embed → chat script.

24. The Most Important Concepts You've Learned So Far

If you're preparing for interviews, you should be able to explain these without looking at the code:

Concept	Why we use it
Virtual environment	Dependency isolation
pyproject.toml	Project/dependency/tool configuration
FastAPI	API/application boundary
Pydantic Settings	Centralized configuration
.env	Local secrets/configuration
Logging	Production observability foundation
Liveness	Process health
Readiness	Dependency/application health
PostgreSQL	Persistent relational storage
Psycopg	PostgreSQL driver
SQLAlchemy	Database abstraction/ORM
AsyncSession	Non-blocking DB I/O
Repository	Isolate persistence logic
Service	Encapsulate application/business logic
Alembic	Version-controlled DB schema changes
ORM Model	Python representation of DB entity
Unit Test	Test component behavior
Integration Test	Test real component interaction
Content Hash	Document deduplication
Git	Version/control and reproducibility
And this is exactly why we're going one step at a time.

When we eventually write:

LangGraph + RAG + Hybrid Retrieval + HyDE + HNSW + vLLM

you'll understand the engineering foundation underneath it, rather than just knowing how to import the libraries.