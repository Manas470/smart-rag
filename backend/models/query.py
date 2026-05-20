"""
SmartRAG — Query Request/Response Pydantic Models
"""
from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to use for answer")
    filter_document_ids: Optional[list[str]] = Field(
        default=None, description="Scope retrieval to specific document IDs (optional)"
    )
    stream: bool = Field(default=False, description="Stream the answer tokens (SSE)")


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    relevance_score: float
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    query_type: str                    # simple | complex | hybrid
    sources: list[SourceChunk]
    hallucination_score: float         # 0.0 (certain) → 1.0 (likely hallucinated)
    is_flagged: bool                   # True if hallucination_score > threshold
    confidence: str                    # "high" | "medium" | "low"
    latency_ms: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    message: str
