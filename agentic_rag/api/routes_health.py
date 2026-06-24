from __future__ import annotations
from fastapi import APIRouter, Depends
from agentic_rag import __version__
from agentic_rag.agent import AgenticRAG
from agentic_rag.api.dependencies import get_agent
from agentic_rag.api.schemas import HealthResponse
router = APIRouter(tags=["health"])
@router.get("/health", response_model=HealthResponse)
def health(agent: AgenticRAG = Depends(get_agent)) -> HealthResponse:
    return HealthResponse(status="ok", chunks_indexed=agent.store.count(), version=__version__)
