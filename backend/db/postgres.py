"""
SmartRAG — PostgreSQL Async Database Layer
"""
import secrets
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, update
from datetime import datetime

from config import settings
from models.document import Base, Tenant, Document, QueryLog


# ── Engine & Session ──────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── Tenant Operations ─────────────────────────────────────────────────────────

async def create_tenant(db: AsyncSession, name: str, email: str) -> Tenant:
    api_key = f"srag_{secrets.token_urlsafe(32)}"
    tenant = Tenant(name=name, email=email, api_key=api_key)
    db.add(tenant)
    await db.flush()
    await db.refresh(tenant)
    return tenant


async def get_tenant_by_api_key(db: AsyncSession, api_key: str) -> Optional[Tenant]:
    result = await db.execute(select(Tenant).where(Tenant.api_key == api_key, Tenant.is_active == True))
    return result.scalar_one_or_none()


async def increment_query_count(db: AsyncSession, tenant_id: str):
    await db.execute(
        update(Tenant)
        .where(Tenant.id == tenant_id)
        .values(query_count=Tenant.query_count + 1)
    )


# ── Document Operations ───────────────────────────────────────────────────────

async def create_document_record(
    db: AsyncSession,
    tenant_id: str,
    filename: str,
    content_type: str,
    size_bytes: int,
) -> Document:
    doc = Document(
        tenant_id=tenant_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        status="processing",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    await db.execute(
        update(Tenant)
        .where(Tenant.id == tenant_id)
        .values(document_count=Tenant.document_count + 1)
    )
    return doc


async def update_document_status(
    db: AsyncSession,
    document_id: str,
    status: str,
    chunk_count: int = 0,
    error_message: Optional[str] = None,
):
    values = {"status": status, "chunk_count": chunk_count}
    if status == "ready":
        values["indexed_at"] = datetime.utcnow()
    if error_message:
        values["error_message"] = error_message

    await db.execute(update(Document).where(Document.id == document_id).values(**values))


async def get_documents_for_tenant(db: AsyncSession, tenant_id: str) -> list[Document]:
    result = await db.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


# ── Query Log Operations ──────────────────────────────────────────────────────

async def log_query(
    db: AsyncSession,
    tenant_id: str,
    query_text: str,
    answer_text: str,
    query_type: str,
    hallucination_score: float,
    is_flagged: bool,
    latency_ms: int,
    chunk_count: int,
) -> QueryLog:
    log = QueryLog(
        tenant_id=tenant_id,
        query_text=query_text,
        answer_text=answer_text,
        query_type=query_type,
        hallucination_score=hallucination_score,
        is_flagged=is_flagged,
        latency_ms=latency_ms,
        chunk_count=chunk_count,
    )
    db.add(log)
    await db.flush()
    return log
