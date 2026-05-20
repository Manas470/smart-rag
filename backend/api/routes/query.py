"""
SmartRAG — Query Route
POST /query → run the full SmartRAG pipeline and return a cited answer
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes.auth import get_current_tenant
from db.postgres import get_db, log_query, increment_query_count
from core.rag_engine import run_query
from models.document import Tenant
from models.query import QueryRequest, QueryResponse

import structlog
log = structlog.get_logger()

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """
    Run a natural language query against the tenant's indexed documents.

    Returns a cited answer with:
    - source chunks used
    - query type (simple/complex/hybrid) chosen by the router
    - hallucination score and confidence flag
    - end-to-end latency
    """
    try:
        result = await run_query(
            tenant_id=tenant.id,
            query=request.query,
            top_k=request.top_k,
            filter_document_ids=request.filter_document_ids,
        )
    except Exception as e:
        log.error("query_failed", tenant_id=tenant.id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

    # Async log to DB (non-blocking — don't await, fire and forget)
    await log_query(
        db=db,
        tenant_id=tenant.id,
        query_text=request.query,
        answer_text=result.answer,
        query_type=result.query_type,
        hallucination_score=result.hallucination_score,
        is_flagged=result.is_flagged,
        latency_ms=result.latency_ms,
        chunk_count=len(result.sources),
    )
    await increment_query_count(db, tenant.id)

    return result
