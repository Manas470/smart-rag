"""
SmartRAG — Health & Tenant Management Routes
GET  /health          → liveness probe (no auth)
GET  /me              → current tenant info
POST /auth/register   → create new tenant + API key
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr

from api.routes.auth import get_current_tenant
from db.postgres import get_db, create_tenant
from models.document import Tenant

router = APIRouter(tags=["System"])


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr


class RegisterResponse(BaseModel):
    tenant_id: str
    api_key: str
    message: str


@router.get("/health")
async def health():
    return {"status": "ok", "service": "SmartRAG"}


@router.post("/auth/register", response_model=RegisterResponse)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Create a new tenant account and receive an API key."""
    try:
        tenant = await create_tenant(db, name=request.name, email=request.email)
    except Exception:
        raise HTTPException(status_code=409, detail="Email already registered.")
    return RegisterResponse(
        tenant_id=tenant.id,
        api_key=tenant.api_key,
        message="Save your API key — it won't be shown again. Use it as: Authorization: Bearer <key>",
    )


@router.get("/me")
async def get_me(tenant: Tenant = Depends(get_current_tenant)):
    """Return current tenant profile and usage stats."""
    return {
        "id": tenant.id,
        "name": tenant.name,
        "email": tenant.email,
        "plan": tenant.plan,
        "query_count": tenant.query_count,
        "document_count": tenant.document_count,
    }
