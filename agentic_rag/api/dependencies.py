from __future__ import annotations
from fastapi import Header, HTTPException, Request, status
from agentic_rag.agent import AgenticRAG
from agentic_rag.settings import get_settings

def get_agent(request: Request) -> AgenticRAG:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not initialized yet."
        )
    return agent


