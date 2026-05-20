"""
SmartRAG — Hybrid Retriever
Combines dense vector search (ChromaDB) + sparse BM25 keyword search,
fused with Reciprocal Rank Fusion (RRF).

Why hybrid? Dense retrieval excels at semantic similarity.
BM25 excels at exact keyword/entity matches. RRF merges both rank lists
without needing to calibrate scores across different spaces.
"""
from dataclasses import dataclass
from typing import Optional
from rank_bm25 import BM25Okapi

from core.embeddings import embed_query
from db.vector_store import query_collection
from config import settings


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    vector_score: float
    bm25_score: float
    rrf_score: float
    page_number: Optional[int] = None


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[tuple[str, float]]:
    """
    Merge two ranked lists using RRF.
    Returns: list of (chunk_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}

    for rank, item in enumerate(vector_results):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + vector_weight * (1.0 / (k + rank + 1))

    for rank, item in enumerate(bm25_results):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + bm25_weight * (1.0 / (k + rank + 1))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def retrieve_chunks(
    tenant_id: str,
    query: str,
    top_k: int = 20,
    use_bm25: bool = True,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    filter_document_ids: Optional[list[str]] = None,
) -> list[RetrievedChunk]:
    """
    First-stage retrieval: returns top_k candidate chunks using hybrid search.
    These candidates are then passed to the cross-encoder reranker.
    """
    # ── Dense vector retrieval ────────────────────────────────────────────────
    query_embedding = await embed_query(query)

    where_filter = None
    if filter_document_ids:
        where_filter = {"document_id": {"$in": filter_document_ids}}

    vector_raw = await query_collection(
        tenant_id=tenant_id,
        query_embedding=query_embedding,
        n_results=top_k,
        where=where_filter,
    )

    # Unpack ChromaDB response format
    chunks_map: dict[str, dict] = {}
    vector_ranked: list[dict] = []

    if vector_raw["ids"] and vector_raw["ids"][0]:
        for idx, chunk_id in enumerate(vector_raw["ids"][0]):
            meta = vector_raw["metadatas"][0][idx]
            doc_text = vector_raw["documents"][0][idx]
            distance = vector_raw["distances"][0][idx]
            vector_score = 1.0 - distance  # cosine distance → similarity

            chunks_map[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": meta.get("document_id", ""),
                "document_name": meta.get("document_name", ""),
                "content": doc_text,
                "page_number": meta.get("page_number"),
                "vector_score": vector_score,
                "bm25_score": 0.0,
            }
            vector_ranked.append({"chunk_id": chunk_id})

    if not chunks_map:
        return []

    # ── BM25 sparse retrieval (over the same candidate pool) ─────────────────
    bm25_ranked: list[dict] = []
    if use_bm25 and chunks_map:
        corpus = [chunks_map[cid]["content"].lower().split() for cid in chunks_map]
        chunk_ids = list(chunks_map.keys())
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query.lower().split())

        scored = sorted(zip(chunk_ids, bm25_scores), key=lambda x: x[1], reverse=True)
        for cid, score in scored:
            chunks_map[cid]["bm25_score"] = float(score)
            bm25_ranked.append({"chunk_id": cid})

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    if use_bm25 and bm25_ranked:
        fused = reciprocal_rank_fusion(vector_ranked, bm25_ranked, vector_weight=vector_weight, bm25_weight=bm25_weight)
    else:
        fused = [(item["chunk_id"], 1.0 / (60 + i + 1)) for i, item in enumerate(vector_ranked)]

    result: list[RetrievedChunk] = []
    for chunk_id, rrf_score in fused[:top_k]:
        c = chunks_map[chunk_id]
        result.append(RetrievedChunk(
            chunk_id=chunk_id,
            document_id=c["document_id"],
            document_name=c["document_name"],
            content=c["content"],
            vector_score=c["vector_score"],
            bm25_score=c["bm25_score"],
            rrf_score=rrf_score,
            page_number=c.get("page_number"),
        ))

    return result
