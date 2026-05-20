"""
SmartRAG — API Key Authentication
Multi-tenant: every request must carry an API key in the header.
Header: Authorization: Bearer srag_<key>
"""
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres import get_db, get_tenant_by_api_key
from models.document import Tenant

bearer_scheme = HTTPBearer()


async def get_current_tenant(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Tenant:
    """
    Dependency: resolves the API key to a Tenant object.
    Raises 401 if the key is missing or invalid.
    """
    api_key = credentials.credentials
    tenant = await get_tenant_by_api_key(db, api_key)
    if not tenant:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key. Generate one at /auth/register.",
        )
    if not tenant.is_active:
        raise HTTPException(status_code=403, detail="Tenant account is suspended.")
    return tenant
