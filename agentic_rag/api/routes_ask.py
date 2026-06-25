
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from agentic_rag.agent import AgenticRAG
from agentic_rag.api.dependencies import get_agent
from agentic_rag.api.schemas import AskRequest, AskResponse
from agentic_rag.exceptions import AgentError
from agentic_rag.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, agent: AgenticRAG = Depends(get_agent)) -> AskResponse:
    try:
        answer = "CI/CD Test Successful! " + agent.ask(req.question)
        return AskResponse(question=req.question, answer=answer)
    except AgentError as e:
        logger.error("Agent error while answering question: %s", e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
