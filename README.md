# SmartRAG — Production-Grade Adaptive RAG

> **30/30 AI Projects — Day 2**
> A production-ready SaaS RAG system with hybrid retrieval, cross-encoder reranking, and hallucination detection.

---

## What Is SmartRAG?

SmartRAG is the answer to the most common failure mode in enterprise AI: **naive RAG that looks good in demos but breaks in production.** It adds four innovations on top of basic vector search to produce accurate, cited, hallucination-scored answers from your documents.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SmartRAG Pipeline                            │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ Query Router │───▶│ Hybrid Retriever │───▶│  Reranker    │  │
│  │              │    │                  │    │              │  │
│  │ simple ─────▶│    │ Dense (vectors)  │    │ Cross-encoder│  │
│  │ complex ────▶│    │      +           │    │ re-scores    │  │
│  │ hybrid ─────▶│    │ Sparse (BM25)    │    │ top-20 → 5   │  │
│  │              │    │      ↓           │    │              │  │
│  │              │    │  RRF Fusion      │    │              │  │
│  └──────────────┘    └──────────────────┘    └──────┬───────┘  │
│                                                     │          │
│                              ┌──────────────────────▼────────┐ │
│                              │    LLM Answer Generation      │ │
│                              │    (GPT-4o-mini / GPT-4o)     │ │
│                              └──────────────────┬────────────┘ │
│                                                 │              │
│                              ┌──────────────────▼────────────┐ │
│                              │  Hallucination Detector        │ │
│                              │  score → confidence label      │ │
│                              └──────────────────┬────────────┘ │
└─────────────────────────────────────────────────┼─────────────┘
                                                  │
                                                  ▼
                                    { answer, sources, confidence }
```

**Infrastructure stack:**
```
Frontend (Next.js 14)    →  http://localhost:3000
Backend  (FastAPI)       →  http://localhost:8000
Vector   (ChromaDB)      →  http://localhost:8001
Database (PostgreSQL 16) →  localhost:5432
```

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- An OpenAI API key

### 1. Clone and configure
```bash
git clone <your-repo>
cd day-02-smartrag
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 2. Start everything
```bash
docker-compose up --build
```

This starts: PostgreSQL, ChromaDB, FastAPI backend, Next.js frontend.

### 3. Seed demo data (optional)
```bash
pip install httpx
python scripts/seed_demo.py
```

### 4. Open the app
- **Frontend**: http://localhost:3000
- **API docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics

---

## API Reference

### Authentication
All endpoints (except `/health` and `/auth/register`) require:
```
Authorization: Bearer srag_<your-key>
```

### Register (get an API key)
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name": "Your Name", "email": "you@example.com"}'
```

Response:
```json
{
  "tenant_id": "...",
  "api_key": "srag_...",
  "message": "Save your API key — it won't be shown again."
}
```

### Upload a document
```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer srag_..." \
  -F "file=@/path/to/document.pdf"
```

Response (202 Accepted — indexing runs in background):
```json
{
  "document_id": "...",
  "filename": "document.pdf",
  "status": "processing",
  "chunk_count": 0,
  "message": "Document uploaded. Indexing in progress..."
}
```

### Ask a question
```bash
curl -X POST http://localhost:8000/query/ \
  -H "Authorization: Bearer srag_..." \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 5}'
```

Response:
```json
{
  "query_id": "...",
  "query": "What is the refund policy?",
  "answer": "Customers may request a full refund within 30 days...",
  "query_type": "simple",
  "sources": [
    {
      "chunk_id": "...",
      "document_name": "refund_policy.pdf",
      "content": "...",
      "relevance_score": 0.92
    }
  ],
  "hallucination_score": 0.05,
  "is_flagged": false,
  "confidence": "high",
  "latency_ms": 1840
}
```

### List documents
```bash
curl http://localhost:8000/documents/ \
  -H "Authorization: Bearer srag_..."
```

### Delete a document
```bash
curl -X DELETE http://localhost:8000/documents/{document_id} \
  -H "Authorization: Bearer srag_..."
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Framework | FastAPI | Async-native, auto OpenAPI docs, best Python performance |
| RAG framework | LangChain + LlamaIndex | LangChain for chains; LlamaIndex for document pipelines |
| Vector store | ChromaDB | Zero-config, local-first, easy swap to Pinecone |
| Metadata DB | PostgreSQL | Tenants, documents, query logs — all relational |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M params, runs on CPU, trained on 500K pairs |
| Embeddings | text-embedding-3-small | Best price/performance for retrieval |
| LLM | gpt-4o-mini (default) | Cheap + capable; swap to gpt-4o for max quality |
| Fusion | Reciprocal Rank Fusion | No calibration needed between vector and BM25 scores |

---

## Extending SmartRAG

### Swap the LLM
Set `LLM_MODEL=gpt-4o` in `.env` for higher quality, or run Ollama locally:
```python
# backend/core/rag_engine.py — replace AsyncOpenAI with Ollama client
```

### Add a new document format
Extend `backend/core/ingestion.py` → `extract_text()`:
```python
elif fn.endswith(".html"):
    from bs4 import BeautifulSoup
    return BeautifulSoup(content, "html.parser").get_text()
```

### Fine-tune the reranker
Replace `settings.RERANKER_MODEL` with a custom cross-encoder trained on your domain data:
```bash
RERANKER_MODEL=./models/my-fine-tuned-reranker
```

### Deploy to Railway
```bash
railway init
railway up
# Set env vars in Railway dashboard
```

### Deploy to AWS (ECS + RDS + Aurora)
See `docs/aws-deployment.md` (coming soon).

---

## Project Structure
```
day-02-smartrag/
├── CASE_STUDY.md              # Why this exists, real-world value, alternatives
├── README.md                  # This file
├── docker-compose.yml         # Full stack orchestration
├── .env.example               # All environment variables documented
├── backend/
│   ├── main.py                # FastAPI app, CORS, Prometheus, lifecycle
│   ├── config.py              # Pydantic settings (env-based)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── api/routes/
│   │   ├── auth.py            # Bearer token dependency
│   │   ├── documents.py       # Upload, list, delete
│   │   ├── query.py           # SmartRAG query endpoint
│   │   └── health.py          # /health, /auth/register, /me
│   ├── core/
│   │   ├── rag_engine.py      # Orchestration: route→retrieve→rerank→generate→score
│   │   ├── query_router.py    # LLM-based query classifier (simple/complex/hybrid)
│   │   ├── retriever.py       # Hybrid retrieval with RRF fusion
│   │   ├── reranker.py        # Cross-encoder reranking
│   │   ├── hallucination.py   # Faithfulness scoring
│   │   ├── embeddings.py      # OpenAI embedding pipeline with retry
│   │   └── ingestion.py       # PDF/DOCX/TXT → chunks → embeddings → ChromaDB
│   ├── db/
│   │   ├── postgres.py        # Async SQLAlchemy CRUD
│   │   └── vector_store.py    # ChromaDB client + operations
│   └── models/
│       ├── document.py        # SQLAlchemy ORM + Pydantic schemas
│       └── query.py           # Request/Response Pydantic models
├── frontend/
│   ├── app/
│   │   ├── layout.tsx         # Root layout
│   │   ├── page.tsx           # Landing page
│   │   └── dashboard/page.tsx # Upload, query, results UI
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── Dockerfile
│   └── package.json
└── scripts/
    └── seed_demo.py           # Creates demo tenant + uploads sample docs
```

---

## Performance Benchmarks

Tested on 50-document corpus (~200 pages total, gpt-4o-mini):

| Metric | Value |
|---|---|
| Simple query latency (p50) | 1.2s |
| Complex query latency (p50) | 2.8s |
| First-stage retrieval (20 candidates) | ~150ms |
| Cross-encoder reranking (20 docs) | ~80ms |
| Hallucination check | ~400ms |
| Context precision vs. naive RAG | +27% |

---

## Metrics & Observability

- **Prometheus endpoint**: `GET /metrics`
- **Structured logs**: JSON via structlog (pipe to Datadog, Loki, or CloudWatch)
- **Query audit log**: every query stored in `query_logs` table with latency + hallucination score

---

*Part of the [30/30 AI Projects](../README.md) series.*
