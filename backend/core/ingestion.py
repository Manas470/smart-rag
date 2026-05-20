"""
SmartRAG — Document Ingestion Pipeline

Supports: PDF, DOCX, TXT, MD
Pipeline:
  1. Parse raw bytes → text
  2. Chunk with overlap (RecursiveCharacterTextSplitter)
  3. Embed chunks in batches
  4. Upsert to ChromaDB with metadata
  5. Update document status in PostgreSQL
"""
import io
import uuid
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import structlog

from config import settings
from core.embeddings import embed_texts
from db.vector_store import upsert_chunks

log = structlog.get_logger()


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n\n".join(pages_text)


def _extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    import docx
    doc = docx.Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text(content: bytes, content_type: str, filename: str) -> str:
    """Route to the correct extractor based on MIME type or filename."""
    ct = content_type.lower()
    fn = filename.lower()

    if "pdf" in ct or fn.endswith(".pdf"):
        return _extract_text_from_pdf(content)
    elif "docx" in ct or "word" in ct or fn.endswith(".docx"):
        return _extract_text_from_docx(content)
    elif "text" in ct or fn.endswith((".txt", ".md", ".csv")):
        return content.decode("utf-8", errors="replace")
    else:
        # Best-effort: try UTF-8 decode
        return content.decode("utf-8", errors="replace")


def chunk_text(text: str, document_id: str, document_name: str) -> list[dict]:
    """
    Split text into overlapping chunks. Returns list of chunk dicts
    with id, text, and metadata ready for ChromaDB upsert.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    result = []
    for i, chunk_text in enumerate(chunks):
        if not chunk_text.strip():
            continue
        chunk_id = f"{document_id}_chunk_{i}"
        result.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": i,
            },
        })
    return result


async def ingest_document(
    tenant_id: str,
    document_id: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> int:
    """
    Full ingestion pipeline for one document.
    Returns the number of chunks created.
    Raises on failure (caller logs and updates DB status).
    """
    log.info("ingestion_start", document_id=document_id, filename=filename)

    # ── 1. Extract text ───────────────────────────────────────────────────────
    raw_text = extract_text(content, content_type, filename)
    if not raw_text.strip():
        raise ValueError(f"No text could be extracted from {filename}")
    log.info("text_extracted", chars=len(raw_text))

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    chunks = chunk_text(raw_text, document_id, filename)
    if not chunks:
        raise ValueError("Document produced no chunks after splitting")
    log.info("chunking_done", num_chunks=len(chunks))

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    texts = [c["text"] for c in chunks]
    embeddings = await embed_texts(texts)
    log.info("embedding_done", num_embeddings=len(embeddings))

    # ── 4. Upsert to ChromaDB ─────────────────────────────────────────────────
    await upsert_chunks(
        tenant_id=tenant_id,
        chunk_ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )
    log.info("vector_upsert_done", tenant_id=tenant_id)

    return len(chunks)
