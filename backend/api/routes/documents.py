"""
SmartRAG — Document Management Routes
POST /documents/upload   → ingest a file
GET  /documents          → list documents for tenant
DELETE /documents/{id}   → remove document and its vectors
"""
import asyncio
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes.auth import get_current_tenant
from db.postgres import (
    get_db,
    create_document_record,
    update_document_status,
    get_documents_for_tenant,
)
from db.vector_store import delete_document_chunks
from core.ingestion import ingest_document
from models.document import Tenant
from models.query import IngestResponse
from config import settings

import structlog
log = structlog.get_logger()

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/csv",
}


async def _run_ingestion_background(
    tenant_id: str,
    document_id: str,
    filename: str,
    content: bytes,
    content_type: str,
):
    """Background task: ingest and update DB status."""
    from db.postgres import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            chunk_count = await ingest_document(
                tenant_id=tenant_id,
                document_id=document_id,
                filename=filename,
                content=content,
                content_type=content_type,
            )
            await update_document_status(db, document_id, "ready", chunk_count)
            await db.commit()
            log.info("ingestion_complete", document_id=document_id, chunks=chunk_count)
        except Exception as e:
            await update_document_status(db, document_id, "failed", error_message=str(e))
            await db.commit()
            log.error("ingestion_failed", document_id=document_id, error=str(e))


@router.post("/upload", response_model=IngestResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Upload and asynchronously index a document."""
    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, DOCX, TXT, MD, CSV",
        )

    # Validate size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_DOCUMENT_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed: {settings.MAX_DOCUMENT_SIZE_MB}MB",
        )

    # Validate document quota
    if tenant.document_count >= settings.MAX_DOCUMENTS_PER_TENANT:
        raise HTTPException(
            status_code=429,
            detail=f"Document limit reached ({settings.MAX_DOCUMENTS_PER_TENANT}). Upgrade your plan.",
        )

    # Create DB record (status=processing)
    doc = await create_document_record(
        db=db,
        tenant_id=tenant.id,
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=len(content),
    )

    # Kick off async ingestion (returns immediately to caller)
    background_tasks.add_task(
        _run_ingestion_background,
        tenant_id=tenant.id,
        document_id=doc.id,
        filename=file.filename,
        content=content,
        content_type=file.content_type,
    )

    return IngestResponse(
        document_id=doc.id,
        filename=file.filename,
        status="processing",
        chunk_count=0,
        message="Document uploaded. Indexing in progress — poll GET /documents to check status.",
    )


@router.get("/", response_model=list[dict])
async def list_documents(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """List all documents for the authenticated tenant."""
    docs = await get_documents_for_tenant(db, tenant.id)
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "chunk_count": d.chunk_count,
            "size_bytes": d.size_bytes,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and its vectors from ChromaDB."""
    await delete_document_chunks(tenant.id, document_id)
    await update_document_status(db, document_id, "deleted")
    log.info("document_deleted", document_id=document_id, tenant_id=tenant.id)
