"""FastAPI application entry point.

Builds the `FastAPI` app, registers the four routers (documents, retrieval,
research, settings), and exposes two liveness/readiness endpoints. Contains
no business logic itself -- every route's actual behavior lives in
`app/api/*` and the service/repository layers they call.
"""

import logging

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.research import router as research_router
from app.api.retrieval import router as retrieval_router
from app.api.settings import router as settings_router
from app.config.settings import get_settings
from app.core.database_health import check_database_connection
from app.core.logging import configure_logging

configure_logging()

settings = get_settings()

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)
app.include_router(retrieval_router)
app.include_router(research_router)
app.include_router(documents_router)
app.include_router(settings_router)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Application started")


@app.get("/health")
async def health_check() -> dict[str, str]:
    logger.info("Health check requested")

    return {
        "status": "healthy",
        "environment": settings.environment,
    }


@app.get("/health/ready")
async def readiness_check() -> dict[str, object]:
    database_healthy = await check_database_connection()

    return {
        "status": "ready" if database_healthy else "not_ready",
        "checks": {
            "database": database_healthy,
        },
    }
