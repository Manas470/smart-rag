"""
SmartRAG — Cross-Encoder Reranker

Why reranking? First-stage retrieval is a fast approximation — we retrieve
20 candidates cheaply. The cross-encoder reads the query and each document
TOGETHER (full attention, not just embeddings) giving far more accurate
relevance scores. We then take the top-K reranked results.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, runs in ~80ms per batch on CPU
  - Trained on 500K MS-MARCO passage pairs
  - Can be swapped for a domain-fine-tuned model

Typical gain: 15-30% improvement in NDCG@5 vs. embedding-only ranking.
"""
import asyncio
from functools import lru_cache
from typing import Optional
import structlog

from core.retriever import RetrievedChunk
from config import settings

log = structlog.get_logger()


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Lazy-load the cross-encoder model once (cached after first call)."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
        log.info("cross_encoder_loaded", model=settings.RERANKER_MODEL)
        return model
    except Exception as e:
        log.warning("cross_encoder_load_failed", error=str(e))
        return None


async def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Rerank candidates using a cross-encoder. Returns top_k chunks sorted by
    cross-encoder score. Falls back to RRF score ordering if model unavailable.
    """
    if not chunks:
        return []

    model = _load_cross_encoder()
    if model is None:
        # Graceful fallback: use first-stage RRF scores
        log.warning("reranker_fallback", reason="model_not_available")
        return sorted(chunks, key=lambda c: c.rrf_score, reverse=True)[:top_k]

    pairs = [(query, chunk.content) for chunk in chunks]

    # Run cross-encoder inference in a thread pool (CPU-bound)
    loop = asyncio.get_event_loop()
    scores: list[float] = await loop.run_in_executor(
        None, lambda: model.predict(pairs, show_progress_bar=False).tolist()
    )

    scored_chunks = sorted(
        zip(scores, chunks), key=lambda x: x[0], reverse=True
    )

    reranked: list[RetrievedChunk] = []
    for score, chunk in scored_chunks[:top_k]:
        # Inject the cross-encoder score as the primary relevance score
        chunk.vector_score = float(score)
        reranked.append(chunk)

    return reranked
