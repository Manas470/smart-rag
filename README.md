# 🧠 SmartRAG

> **Production-grade Adaptive RAG** — hybrid retrieval, cross-encoder reranking, and hallucination detection as a deployable SaaS.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?logo=langchain)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)

---

## What Is SmartRAG?

SmartRAG is the answer to the most common failure mode in enterprise AI: **naive RAG that works in demos but breaks in production.**

Standard vector search feeds the LLM bad context. SmartRAG fixes that with four layers:

| Layer | What it does | Why it matters |
|---|---|---|
| **Adaptive query routing** | Classifies each query as simple / complex / hybrid | Simple lookups get fast retrieval; complex reasoning gets broader, deeper search |
| **Hybrid retrieval** | Dense vector + BM25 sparse, fused with RRF | Semantic similarity + exact keyword matching — neither alone is sufficient |
| **Cross-encoder reranking** | 22M-param neural model re-scores top-20 candidates | Full query-document attention, not just embedding proximity |
| **Hallucination detection** | LLM-as-judge faithfulness scoring | Every answer gets a confidence score; uncertain answers are flagged |

---

## Use It for Your Own Project

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- An [OpenAI API key](https://platform.openai.com/api-keys) (needs at least $5 credit)
- Git

### Clone and configure

```bash
git clone https://github.com/manas470/smart-rag.git
cd smart-rag

# Create your local config file
cp .env.example .env
```

Open `.env` and set your OpenAI key:

```env
OPENAI_API_KEY=sk-...your-key-here...
```

Everything else in `.env` works out of the box for local development.

### Start everything with one command

```bash
docker-compose up --build
```

**First build takes 5–10 minutes** — it downloads Python packages, builds the Next.js app, and pre-downloads the reranker model (~85MB). Subsequent starts take ~30 seconds.

You'll know it's ready when you see:

```
smartrag_backend   | Application startup complete.
smartrag_frontend  | ✓ Ready on http://0.0.0.0:3000
```

### Open the app

| Service | URL | What's there |
|---|---|---|
| **Frontend UI** | http://localhost:3000 | Landing page + upload dashboard + query interface |
| **API docs** | http://localhost:8000/docs | Interactive Swagger — try every endpoint in browser |
| **Metrics** | http://localhost:8000/metrics | Prometheus metrics endpoint |

### Seed demo data (optional — 60 seconds)

```bash
pip install httpx
python scripts/seed_demo.py
```

This creates a demo account, uploads 3 sample documents, and runs test queries so you can see SmartRAG working immediately without uploading your own files.

---

## API Quick Start

### 1. Get an API key

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name": "Your Name", "email": "you@example.com"}'
```

```json
{
  "tenant_id": "...",
  "api_key": "srag_...",
  "message": "Save your API key — it won't be shown again."
}
```

### 2. Upload a document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer srag_YOUR_KEY" \
  -F "file=@/path/to/your-document.pdf"
```

Supported formats: **PDF, DOCX, TXT, MD, CSV** (max 50MB each)

The response comes back immediately (HTTP 202) — indexing runs in the background. Poll `GET /documents/` to check when status is `"ready"`.

### 3. Ask a question

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Authorization: Bearer srag_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy after 30 days?", "top_k": 5}'
```

```json
{
  "query_id": "abc-123",
  "query": "What is the refund policy after 30 days?",
  "answer": "After 30 days and within 90 days, customers are eligible for a 50% partial refund...",
  "query_type": "simple",
  "sources": [
    {
      "document_name": "refund_policy.pdf",
      "content": "...",
      "relevance_score": 0.94
    }
  ],
  "hallucination_score": 0.04,
  "is_flagged": false,
  "confidence": "high",
  "latency_ms": 1620
}
```

---

## Adapting SmartRAG for Your Use Case

### Swap the LLM

In `.env`, change `LLM_MODEL`:

```env
LLM_MODEL=gpt-4o          # higher quality
LLM_MODEL=gpt-4o-mini     # default — best price/quality
```

To use a **local model** (no API cost), replace the OpenAI client in `backend/core/rag_engine.py` with an Ollama client and set `base_url=http://localhost:11434`.

### Swap the vector store

The entire vector store is abstracted in `backend/db/vector_store.py`. To switch from ChromaDB to Qdrant or Pinecone, only this file changes — nothing else in the pipeline needs to know.

### Add a new document format

Extend the `extract_text()` function in `backend/core/ingestion.py`:

```python
elif fn.endswith(".html"):
    from bs4 import BeautifulSoup
    return BeautifulSoup(content, "html.parser").get_text()
```

### Change the reranker model

In `.env`:

```env
# Smaller/faster (current default)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Higher quality (needs more RAM)
RERANKER_MODEL=cross-encoder/ms-marco-electra-base
```

### Tune retrieval aggressiveness

```env
CHUNK_SIZE=512              # tokens per chunk (smaller = more precise retrieval)
CHUNK_OVERLAP=64            # overlap between chunks
TOP_K_RETRIEVAL=20          # candidates from first-stage retrieval
TOP_K_RERANK=5              # chunks sent to LLM after reranking
HALLUCINATION_THRESHOLD=0.5 # above this score → answer is flagged
```

---

## Deploy to Production

### Railway (easiest — ~15 minutes)

1. Push this repo to your GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo
4. Add environment variables from your `.env` file in the Railway dashboard
5. Railway auto-detects `docker-compose.yml` and deploys all services

### Render

Use the `docker-compose.yml` as a reference to create individual services on Render (one for backend, one for frontend). Use Render's managed PostgreSQL and add ChromaDB as a private service.

### AWS (ECS + RDS)

Replace `postgres` in docker-compose with AWS RDS (PostgreSQL), replace `chromadb` with a managed Qdrant Cloud or self-hosted ECS task, deploy backend and frontend as ECS services behind an ALB.

---

## Architecture

```
User query
    │
    ▼
┌──────────────────────────────────────────────────┐
│               SmartRAG pipeline                  │
│                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌───────┐ │
│  │ Query router│──▶│   Hybrid     │──▶│Rerank │ │
│  │             │   │  retriever   │   │       │ │
│  │ simple ─────│   │ Dense+BM25   │   │top20  │ │
│  │ complex ────│   │    + RRF     │   │ → top5│ │
│  │ hybrid ─────│   │              │   │       │ │
│  └─────────────┘   └──────────────┘   └───┬───┘ │
│                                           │      │
│                              ┌────────────▼────┐ │
│                              │  LLM generation │ │
│                              └────────┬────────┘ │
│                                       │          │
│                              ┌────────▼────────┐ │
│                              │  Hallucination  │ │
│                              │    detector     │ │
│                              └────────┬────────┘ │
└───────────────────────────────────────┼──────────┘
                                        ▼
                           { answer + sources + confidence }
```

**Services:**

```
Frontend   (Next.js 14)    →  :3000
Backend    (FastAPI)       →  :8000
Vector DB  (ChromaDB)      →  :8001
Database   (PostgreSQL 16) →  :5432
```

---

## Project Structure

```
smart-rag/
├── .env.example               # All config variables documented
├── docker-compose.yml         # Full stack — one command to run
├── CASE_STUDY.md              # Real-world problem, use cases, alternatives
├── THOUGHTS.md                # Builder's notes — decisions, tradeoffs, benchmarks
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── config.py              # Pydantic settings from env vars
│   ├── api/routes/
│   │   ├── auth.py            # Bearer token dependency
│   │   ├── documents.py       # Upload, list, delete
│   │   ├── query.py           # SmartRAG query endpoint
│   │   └── health.py          # /health, /auth/register, /me
│   ├── core/
│   │   ├── rag_engine.py      # Pipeline orchestrator
│   │   ├── query_router.py    # LLM-based query classifier
│   │   ├── retriever.py       # Hybrid retrieval + RRF fusion
│   │   ├── reranker.py        # Cross-encoder reranking
│   │   ├── hallucination.py   # Faithfulness scoring
│   │   ├── embeddings.py      # OpenAI embedding pipeline
│   │   └── ingestion.py       # Document parsing + chunking + indexing
│   ├── db/
│   │   ├── postgres.py        # Async SQLAlchemy CRUD
│   │   └── vector_store.py    # ChromaDB client
│   └── models/                # SQLAlchemy ORM + Pydantic schemas
├── frontend/
│   ├── app/page.tsx           # Landing page
│   └── app/dashboard/page.tsx # Upload + query UI
└── scripts/
    └── seed_demo.py           # Create demo account + upload sample docs
```

---

## Performance

Tested on 50-document corpus (~200 pages), GPT-4o-mini, M2 MacBook Pro:

| Query type | p50 latency | vs naive RAG accuracy |
|---|---|---|
| Simple (lookup) | ~1.6s | +23% |
| Complex (reasoning) | ~2.5s | +29% |
| Hallucination detection recall | — | 94% |

---

## Contributing

```bash
git clone https://github.com/manas470/smart-rag.git
cd smart-rag
cp .env.example .env   # add your OPENAI_API_KEY
docker-compose up --build
```

PRs welcome. Open an issue first for large changes.

---

## License

MIT © 2026 venkatamanas Raghupatruni

