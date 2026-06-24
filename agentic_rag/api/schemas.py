from __future__ import annotations
from pydantic import BaseModel, Field
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="The user's question")
    

class AskResponse(BaseModel):
    question: str
    answer: str


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    version: str


class IngestStatusResponse(BaseModel):
    state: str  # idle | running | done | error
    detail: str | None = None
    drive_documents: int | None = None
    website_pages: int | None = None
    chunks_written: int | None = None
    total_chunks_in_store: int | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
