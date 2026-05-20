"""
SmartRAG — ChromaDB Vector Store Manager
Each tenant gets an isolated collection: smartrag_{tenant_id}
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional
from config import settings


_client: Optional[chromadb.AsyncHttpClient] = None


async def get_chroma_client() -> chromadb.AsyncHttpClient:
    global _client
    if _client is None:
        _client = await chromadb.AsyncHttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def collection_name(tenant_id: str) -> str:
    """Each tenant's documents are in an isolated Chroma collection."""
    return f"smartrag_{tenant_id}"


async def get_or_create_collection(tenant_id: str):
    client = await get_chroma_client()
    return await client.get_or_create_collection(
        name=collection_name(tenant_id),
        metadata={"hnsw:space": "cosine"},
    )


async def upsert_chunks(
    tenant_id: str,
    chunk_ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
):
    collection = await get_or_create_collection(tenant_id)
    await collection.upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


async def query_collection(
    tenant_id: str,
    query_embedding: list[float],
    n_results: int = 20,
    where: Optional[dict] = None,
) -> dict:
    collection = await get_or_create_collection(tenant_id)
    results = await collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return results


async def delete_document_chunks(tenant_id: str, document_id: str):
    """Remove all chunks for a specific document from the collection."""
    collection = await get_or_create_collection(tenant_id)
    await collection.delete(where={"document_id": document_id})
