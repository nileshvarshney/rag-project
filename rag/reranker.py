"""
reranker.py — Ollama-embedding-based reranking for retrieved chunks.

After initial retrieval the same Ollama embedding model used for indexing
scores each chunk against the question via cosine similarity, then keeps only
the top-N highest-scoring results.

This is lighter-weight than a cross-encoder and requires no external models —
everything runs through the already-running Ollama server.
"""

import asyncio
import logging
import math
from functools import lru_cache

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def rerank_documents(question: str, docs: list[Document]) -> list[Document]:
    """Re-score docs against the question using Ollama embeddings; keep top-N.

    Args:
        question: The user's query.
        docs:     Candidate documents from the retriever.

    Returns:
        Up to RERANKER_TOP_N documents ordered by cosine similarity descending.
        Returns docs unchanged when RERANKER_ENABLED is False or docs is empty.
    """
    if not config.RERANKER_ENABLED or not docs:
        return docs

    embedder = _get_embedder()
    q_emb    = embedder.embed_query(question)
    doc_embs = embedder.embed_documents([d.page_content for d in docs])

    scored  = sorted(
        zip((_cosine(q_emb, e) for e in doc_embs), docs),
        key=lambda x: x[0],
        reverse=True,
    )
    selected = [doc for _, doc in scored[: config.RERANKER_TOP_N]]

    logger.debug(
        "Reranker: %d → %d chunk(s)  (top score=%.4f)",
        len(docs), len(selected), scored[0][0] if scored else 0.0,
    )
    return selected


async def async_rerank_documents(question: str, docs: list[Document]) -> list[Document]:
    """Async wrapper — runs embedding calls in a thread to keep the event loop free."""
    if not config.RERANKER_ENABLED or not docs:
        return docs
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rerank_documents, question, docs)
