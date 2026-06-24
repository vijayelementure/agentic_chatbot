from __future__ import annotations
from threading import Lock
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from agentic_rag.agent import AgenticRAG
from agentic_rag.api.schemas import IngestStatusResponse
from agentic_rag.ingest import run_ingestion
from agentic_rag.logging_config import get_logger
from agentic_rag.settings import get_settings

logger = get_logger(__name__)
router = APIRouter(tags=["ingest"])

lock = Lock()
status: dict = {"state": "idle", "detail": None}


def _(request_app_state) -> None:
    with lock:
        status.update(state="running", detail=None)
    try:
        result = run_ingestion()
        request_app_state.agent = AgenticRAG(settings=get_settings())
        with lock:
            status.update(
                state="done",
                detail=None,
                drive_documents=result.drive_documents,
                website_pages=result.website_pages,
                chunks_written=result.chunks_written,
                total_chunks_in_store=result.total_chunks_in_store,
            )
    except Exception as e:
        logger.exception("Background ingestion failed")
        with lock:
            status.update(state="error", detail=str(e))


@router.post("/ingest", response_model=IngestStatusResponse)
def trigger_ingest(request: Request, background_tasks: BackgroundTasks) -> IngestStatusResponse:
    with lock:
        if status["state"] == "running":
            return IngestStatusResponse(state="running", detail="Ingestion already in progress.")
    background_tasks.add_task(run_ingestion_job, request.app.state)
    return IngestStatusResponse(state="started", detail="Ingestion started in the background.")


@router.get("/ingest/status", response_model=IngestStatusResponse)
def ingest_status() -> IngestStatusResponse:
    with lock:
        return IngestStatusResponse(**status)
