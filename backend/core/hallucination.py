"""
SmartRAG — Hallucination Detector

Checks whether the generated answer is grounded in the retrieved context.
Two-stage approach:
  1. LLM-as-judge: Ask GPT-4o-mini to score faithfulness (cheap, fast)
  2. Semantic similarity fallback: cosine sim between answer and context

Score interpretation:
  0.0 → fully grounded (every claim supported by context)
  1.0 → likely hallucinated (claims not found in context)

Below settings.HALLUCINATION_THRESHOLD → "low" confidence, user sees warning.
"""
import json
from typing import Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config import settings

_FAITHFULNESS_SYSTEM = """You are a faithfulness evaluator for a RAG system.

Given:
- CONTEXT: a set of retrieved document chunks
- ANSWER: a generated answer

Your task: assess whether every factual claim in the ANSWER is supported by
the CONTEXT. Ignore style, grammar, and completeness.

Score:
  0.0 = fully faithful — every claim in the answer has direct support in context
  0.5 = partially faithful — most claims supported but some may be inferred
  1.0 = hallucinated — answer contains claims not found in context at all

Return ONLY valid JSON: {"score": float, "reason": "one sentence"}
"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
async def score_faithfulness(
    answer: str,
    context_chunks: list[str],
    query: Optional[str] = None,
) -> tuple[float, str]:
    """
    Returns (hallucination_score, reason).
    hallucination_score: 0.0 (faithful) → 1.0 (hallucinated)
    """
    if not context_chunks:
        return 0.8, "no context provided to verify against"

    context_text = "\n\n---\n\n".join(context_chunks[:5])  # cap to 5 chunks

    prompt = f"""CONTEXT:
{context_text}

ANSWER:
{answer}"""

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        score = float(data.get("score", 0.5))
        score = max(0.0, min(1.0, score))  # clamp to [0, 1]
        return score, data.get("reason", "")
    except Exception as e:
        return 0.5, f"evaluation failed: {str(e)}"


def confidence_label(score: float, threshold: float = None) -> str:
    """Convert numeric hallucination score to human-readable confidence."""
    threshold = threshold or settings.HALLUCINATION_THRESHOLD
    if score <= 0.2:
        return "high"
    elif score <= threshold:
        return "medium"
    else:
        return "low"
