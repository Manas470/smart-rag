"""
SmartRAG — Document & Tenant Data Models (SQLAlchemy + Pydantic)
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4
import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pydantic import BaseModel, Field


# ── SQLAlchemy ORM ────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Tenant(Base):
    """A tenant is an account (user or organisation) that owns documents + API keys."""
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    email: Mapped[str] = mapped_column(sa.String(255), unique=True, nullable=False)
    api_key: Mapped[str] = mapped_column(sa.String(64), unique=True, nullable=False)
    plan: Mapped[str] = mapped_column(sa.String(32), default="free")   # free | pro | enterprise
    is_active: Mapped[bool] = mapped_column(sa.Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)
    query_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    document_count: Mapped[int] = mapped_column(sa.Integer, default=0)


class Document(Base):
    """Metadata for an ingested document. Vectors live in ChromaDB."""
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(sa.String, sa.ForeignKey("tenants.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    status: Mapped[str] = mapped_column(sa.String(32), default="processing")  # processing | ready | failed
    error_message: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime, nullable=True)


class QueryLog(Base):
    """Audit log of every query — for analytics and hallucination tracking."""
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(sa.String, primary_key=True, default=lambda: str(uuid4()))
    tenant_id: Mapped[str] = mapped_column(sa.String, sa.ForeignKey("tenants.id"), nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    query_type: Mapped[str] = mapped_column(sa.String(32))      # simple | complex | hybrid
    retrieval_score: Mapped[float] = mapped_column(sa.Float, default=0.0)
    hallucination_score: Mapped[float] = mapped_column(sa.Float, default=0.0)
    is_flagged: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(sa.Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)


# ── Pydantic Schemas (API) ─────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
    size_bytes: int
    created_at: datetime

    model_config = {"from_attributes": True}


class TenantResponse(BaseModel):
    id: str
    name: str
    email: str
    plan: str
    query_count: int
    document_count: int
    created_at: datetime

    model_config = {"from_attributes": True}
