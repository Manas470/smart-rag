"""
SmartRAG — Adaptive Query Router

Routes queries to one of three retrieval strategies:
  • simple   → short factual lookup (1 chunk usually sufficient)
  • complex  → multi-hop reasoning (requires synthesis across chunks)
  • hybrid   → mixed: keyword-heavy query needing exact + semantic match

Uses a lightweight classifier (GPT-4o-mini with structured output) so the
routing decision itself costs ~200 tokens and adds < 300ms latency.
"""
import json
from typing import Literal
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config import settings

QueryType = Literal["simple", "complex", "hybrid"]

_ROUTER_SYSTEM = """You are a query classifier for a RAG system. Classify the user query into one of:

- "simple": A direct factual lookup. One or two document chunks can fully answer it.
  Examples: "What is our refund policy?", "What version is mentioned in the changelog?"

- "complex": Requires synthesizing information across multiple document sections or
  multi-hop reasoning. Examples: "Compare the Q3 and Q4 revenue strategies",
  "What are the security implications of the authentication design?"

- "hybrid": The query has strong keyword or entity signals (product codes, person names,
  exact phrases) where BM25 lexical matching matters as much as semantic similarity.
  Examples: "Find clause 12.3(b)", "What does SKU-4421 cost?"

Respond with ONLY valid JSON: {"type": "simple"|"complex"|"hybrid", "reason": "one sentence"}
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
async def classify_query(query: str) -> tuple[QueryType, str]:
    """
    Returns (query_type, reason).
    Falls back to "complex" on any error to be conservative.
    """
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=100,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("type", "complex"), data.get("reason", "")
    except Exception:
        return "complex", "fallback due to classification error"


def get_retrieval_config(query_type: QueryType) -> dict:
    """
    Returns retrieval hyperparameters tuned per query type.
    """
    configs = {
        "simple": {
            "top_k_first_stage": 10,
            "top_k_rerank": 3,
            "use_bm25": False,
            "bm25_weight": 0.0,
            "vector_weight": 1.0,
        },
        "complex": {
            "top_k_first_stage": 20,
            "top_k_rerank": 7,
            "use_bm25": True,
            "bm25_weight": 0.3,
            "vector_weight": 0.7,
        },
        "hybrid": {
            "top_k_first_stage": 20,
            "top_k_rerank": 5,
            "use_bm25": True,
            "bm25_weight": 0.5,
            "vector_weight": 0.5,
        },
    }
    return configs[query_type]
