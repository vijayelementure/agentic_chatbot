from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

from agentic_rag import __version__
from agentic_rag.agent import AgenticRAG
from agentic_rag.api.routes_ask import router as ask_router
from agentic_rag.api.routes_health import router as health_router
from agentic_rag.api.routes_ingest import router as ingest_router
from agentic_rag.exceptions import AgenticRAGError
from agentic_rag.logging_config import configure_logging, get_logger
from agentic_rag.settings import get_settings

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_for_runtime()
    logger.info("Starting up — initializing agent...")
    app.state.agent = AgenticRAG(settings=settings)
    logger.info("Agent ready. Vector store has %d chunks.", app.state.agent.store.count())
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Agentic RAG API",
        description=(
            "Ask questions grounded in Google Drive docs + website content, "
            "via a Gemini-powered agent."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    origins = settings.cors_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(AgenticRAGError)
    async def handle_app_error(request: Request, exc: AgenticRAGError):
        logger.error("Unhandled application error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": exc.__class__.__name__, "detail": str(exc)},
        )

    app.include_router(health_router)
    app.include_router(ask_router)
    app.include_router(ingest_router)

    return app


app = create_app()
handler = Mangum(app)
