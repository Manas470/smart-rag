"""
SmartRAG — Main RAG Engine

This is the orchestration layer. For every query it:
  1. Routes to the right retrieval strategy (QueryRouter)
  2. Retrieves candidate chunks (HybridRetriever)
  3. Reranks candidates (CrossEncoderReranker)
  4. Generates the answer with citations (LLM)
  5. Scores faithfulness (HallucinationDetector)
  6. Returns a structured QueryResponse

All components are async and designed for production load.
"""
import time
import uuid
from typing import Optional
from openai import AsyncOpenAI

from config import settings
from core.query_router import classify_query, get_retrieval_config, QueryType
from core.retriever import retrieve_chunks, RetrievedChunk
from core.reranker import rerank_chunks
from core.hallucination import score_faithfulness, confidence_label
from models.query import QueryResponse, SourceChunk

import structlog
log = structlog.get_logger()

_ANSWER_SYSTEM = """You are a precise, helpful assistant that answers questions using ONLY
the provided document context. Do not use outside knowledge.

Rules:
- Answer based strictly on the context provided
- If the context doesn't contain enough information, say so clearly
- Always be concise and direct
- Do not invent facts, statistics, or details not present in the context
- You may quote directly from the context when helpful

Context will be provided as numbered sections. Reference them naturally (e.g., "According to the document...").
"""


async def _generate_answer(query: str, context_chunks: list[RetrievedChunk]) -> str:
    """Generate an answer from the query and reranked context chunks."""
    context_text = "\n\n".join(
        f"[{i+1}] {chunk.content}"
        for i, chunk in enumerate(context_chunks)
    )

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}",
            },
        ],
        max_tokens=settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


async def run_query(
    tenant_id: str,
    query: str,
    top_k: int = 5,
    filter_document_ids: Optional[list[str]] = None,
) -> QueryResponse:
    """
    Full SmartRAG pipeline. Returns a structured QueryResponse.
    """
    start_ms = time.time()
    query_id = str(uuid.uuid4())

    log.info("rag_query_start", query_id=query_id, tenant_id=tenant_id, query=query[:80])

    # ── Step 1: Query Routing ────────────────────────────────────────────────
    query_type, route_reason = await classify_query(query)
    retrieval_config = get_retrieval_config(query_type)
    log.info("query_routed", query_type=query_type, reason=route_reason)

    # ── Step 2: Hybrid Retrieval ─────────────────────────────────────────────
    candidates = await retrieve_chunks(
        tenant_id=tenant_id,
        query=query,
        top_k=retrieval_config["top_k_first_stage"],
        use_bm25=retrieval_config["use_bm25"],
        vector_weight=retrieval_config["vector_weight"],
        bm25_weight=retrieval_config["bm25_weight"],
        filter_document_ids=filter_document_ids,
    )
    log.info("retrieval_done", candidates=len(candidates))

    if not candidates:
        return QueryResponse(
            query_id=query_id,
            query=query,
            answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
            query_type=query_type,
            sources=[],
            hallucination_score=0.0,
            is_flagged=False,
            confidence="high",
            latency_ms=int((time.time() - start_ms) * 1000),
        )

    # ── Step 3: Reranking ─────────────────────────────────────────────────────
    reranked = await rerank_chunks(
        query=query,
        chunks=candidates,
        top_k=min(top_k, retrieval_config["top_k_rerank"]),
    )
    log.info("reranking_done", top_chunks=len(reranked))

    # ── Step 4: Answer Generation ─────────────────────────────────────────────
    answer = await _generate_answer(query, reranked)
    log.info("answer_generated", answer_len=len(answer))

    # ── Step 5: Hallucination Check ───────────────────────────────────────────
    context_texts = [c.content for c in reranked]
    h_score, h_reason = await score_faithfulness(answer, context_texts, query)
    is_flagged = h_score > settings.HALLUCINATION_THRESHOLD
    confidence = confidence_label(h_score)
    log.info("hallucination_scored", score=h_score, flagged=is_flagged)

    # ── Build Response ────────────────────────────────────────────────────────
    sources = [
        SourceChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_name=c.document_name,
            content=c.content,
            relevance_score=round(c.vector_score, 4),
            page_number=c.page_number,
        )
        for c in reranked
    ]

    latency_ms = int((time.time() - start_ms) * 1000)
    log.info("rag_query_complete", query_id=query_id, latency_ms=latency_ms)

    return QueryResponse(
        query_id=query_id,
        query=query,
        answer=answer,
        query_type=query_type,
        sources=sources,
        hallucination_score=round(h_score, 4),
        is_flagged=is_flagged,
        confidence=confidence,
        latency_ms=latency_ms,
    )
