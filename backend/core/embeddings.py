"""
SmartRAG — Embedding Pipeline
Wraps OpenAI embeddings with retry logic, batching, and token counting.
"""
import asyncio
from typing import Optional
import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config import settings

_client: Optional[AsyncOpenAI] = None
_tokenizer = tiktoken.get_encoding("cl100k_base")


def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI's embedding model.
    Automatically batches to stay within API limits (2048 texts per call).
    """
    client = get_openai_client()
    batch_size = 100  # safe batch for OpenAI
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(
            input=batch,
            model=settings.EMBEDDING_MODEL,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    client = get_openai_client()
    response = await client.embeddings.create(
        input=[query],
        model=settings.EMBEDDING_MODEL,
    )
    return response.data[0].embedding
