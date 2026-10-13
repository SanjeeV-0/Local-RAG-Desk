"""FastAPI dependency-injection wiring -- the application's composition root.

This module owns *which concrete implementation* backs each protocol/ABC
(EmbeddingProvider, Reranker, QueryDecomposer, FileStorage, GenerationProvider)
and how per-request services (RetrievalService, IngestionService, ...) are
assembled from them plus the current request's DB session. It does not
contain business logic itself -- every factory here just reads `Settings`
and constructs an object.

Process-lifetime singletons (models, API clients) are `@lru_cache`d so they
are constructed once per process, not once per request; everything that
needs a request-scoped `AsyncSession` is a plain (uncached) function that
FastAPI re-resolves per request.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.core.database import get_db_session
from app.core.repositories.document import DocumentRepository
from app.core.services.document import DocumentService
from app.core.services.document_deletion import DocumentDeletionService
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.generation.generation_service import GenerationService
from app.generation.openrouter import OpenRouterGenerationProvider
from app.ingestion.service import IngestionService
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.openrouter_decomposition import (
    OpenRouterQueryDecompositionProvider,
)
from app.retrieval.query_decomposer import QueryDecomposer
from app.retrieval.reranker import Reranker
from app.retrieval.service import RetrievalService
from app.storage.interface import FileStorage
from app.storage.local import LocalFileStorage


@lru_cache
def get_query_decomposition_provider() -> OpenRouterQueryDecompositionProvider:
    """Return the configured OpenRouter query decomposition provider.

    Raises eagerly (at dependency-resolution time, before any retrieval
    logic runs) if OPENROUTER_API_KEY is unset. Because `get_retrieval_service`
    below always requests a QueryDecomposer, this effectively makes
    OPENROUTER_API_KEY a hard requirement for POST /retrieval/search and
    POST /research/ask, even though RetrievalService itself treats a missing
    decomposer as "skip decomposition" rather than an error.
    """

    settings = get_settings()

    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY must be configured.")

    return OpenRouterQueryDecompositionProvider(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_decomposition_model,
        temperature=settings.openrouter_decomposition_temperature,
        max_tokens=settings.openrouter_decomposition_max_tokens,
        top_p=settings.openrouter_decomposition_top_p,
        response_format=settings.openrouter_decomposition_response_format,
        reasoning_effort=settings.openrouter_decomposition_reasoning_effort,
        base_url=settings.openrouter_base_url,
        app_name=settings.openrouter_app_name,
        top_k=settings.openrouter_decomposition_top_k,
        frequency_penalty=settings.openrouter_decomposition_frequency_penalty,
        presence_penalty=settings.openrouter_decomposition_presence_penalty,
        timeout=settings.openrouter_timeout_seconds,
        max_retries=settings.openrouter_max_retries,
    )


def get_query_decomposer() -> QueryDecomposer:
    """Return the application query decomposer."""

    settings = get_settings()

    return QueryDecomposer(
        provider=get_query_decomposition_provider(),
        max_sub_queries=settings.query_decomposition_max_sub_queries,
    )


@lru_cache
def get_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    """Return the application embedding provider."""

    settings = get_settings()

    return SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model,
    )


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    """Return the application cross-encoder reranker."""

    settings = get_settings()

    return CrossEncoderReranker(
        model_name=settings.reranker_model,
    )


def get_app_settings() -> Settings:
    """Return application settings."""

    return get_settings()


async def get_retrieval_service(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
    embedding_provider: Annotated[
        SentenceTransformerEmbeddingProvider,
        Depends(get_embedding_provider),
    ],
    reranker: Annotated[
        Reranker,
        Depends(get_reranker),
    ],
    query_decomposer: Annotated[
        QueryDecomposer,
        Depends(get_query_decomposer),
    ],
) -> RetrievalService:
    """Create a retrieval service for the current database session.

    `query_decomposer` and `reranker` are always supplied here -- there is
    currently no configuration path to run retrieval without either one,
    even though `RetrievalService` itself supports `query_decomposer=None`
    / `reranker=None` at the constructor level (used by unit tests).
    """

    settings = get_settings()

    return RetrievalService(
        repository=DocumentRepository(
            session,
            hnsw_ef_search=settings.hnsw_ef_search,
            hnsw_iterative_scan=settings.hnsw_iterative_scan,
            hnsw_max_scan_tuples=settings.hnsw_max_scan_tuples,
            hnsw_scan_mem_multiplier=settings.hnsw_scan_mem_multiplier,
        ),
        embedding_provider=embedding_provider,
        reranker=reranker,
        query_decomposer=query_decomposer,
        candidate_limit_floor=settings.retrieval_candidate_limit_floor,
        router_enabled=settings.query_router_enabled,
    )


@lru_cache
def get_generation_provider() -> OpenRouterGenerationProvider:
    """Return the configured OpenRouter generation provider."""

    settings = get_settings()

    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY must be configured.")

    return OpenRouterGenerationProvider(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_generation_model,
        temperature=settings.openrouter_generation_temperature,
        max_tokens=settings.openrouter_generation_max_tokens,
        top_p=settings.openrouter_generation_top_p,
        response_format=settings.openrouter_generation_response_format,
        reasoning_effort=settings.openrouter_generation_reasoning_effort,
        base_url=settings.openrouter_base_url,
        app_name=settings.openrouter_app_name,
        top_k=settings.openrouter_generation_top_k,
        frequency_penalty=settings.openrouter_generation_frequency_penalty,
        presence_penalty=settings.openrouter_generation_presence_penalty,
        timeout=settings.openrouter_timeout_seconds,
        max_retries=settings.openrouter_max_retries,
    )


def get_generation_service(
    provider: Annotated[
        OpenRouterGenerationProvider,
        Depends(get_generation_provider),
    ],
) -> GenerationService:
    """Create the application generation service."""

    return GenerationService(
        provider=provider,
    )


@lru_cache
def get_file_storage() -> FileStorage:
    """Return the application file storage backend."""

    settings = get_settings()

    return LocalFileStorage(Path(settings.storage_root))


async def get_ingestion_service(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
    embedding_provider: Annotated[
        SentenceTransformerEmbeddingProvider,
        Depends(get_embedding_provider),
    ],
    file_storage: Annotated[
        FileStorage,
        Depends(get_file_storage),
    ],
) -> IngestionService:
    """Create an ingestion service for the current database session."""

    return IngestionService(
        session=session,
        embedding_provider=embedding_provider,
        file_storage=file_storage,
    )


def get_document_repository(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
) -> DocumentRepository:
    """Create a document repository for the current database session."""

    return DocumentRepository(session)


def get_document_service(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
) -> DocumentService:
    """Create a document service for the current database session."""

    return DocumentService(session)


def get_document_deletion_service(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
    file_storage: Annotated[
        FileStorage,
        Depends(get_file_storage),
    ],
) -> DocumentDeletionService:
    """Create a document deletion service for the current database session."""

    return DocumentDeletionService(session=session, file_storage=file_storage)
