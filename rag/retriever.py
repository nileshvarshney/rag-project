"""
retriever.py — Vector store connection and retriever construction.

Supports three search strategies, switchable via config.SEARCH_STRATEGY:

  "similarity"  Plain cosine/ANN search on embeddings.
                Fast, good default for most queries.

  "mmr"         Maximal Marginal Relevance. Still vector-based, but
                re-ranks results to reduce near-duplicate chunks.
                Useful when your documents are repetitive.

  "hybrid"      BM25 (keyword) + semantic (vector) combined via
                Reciprocal Rank Fusion (RRF).

                RAG concept: embeddings are great at capturing meaning
                but can miss exact keyword matches (e.g. product codes,
                names, abbreviations). BM25 is the opposite — it matches
                keywords precisely but is blind to synonyms and paraphrases.
                Combining both covers each other's blind spots.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 singleton cache
# ---------------------------------------------------------------------------
# Building a BM25 index requires fetching all documents from ChromaDB and
# running tokenisation + IDF computation — pure CPU work that produces the
# same result as long as the collection hasn't changed.  We cache the result
# and only rebuild when the ChromaDB document count differs from the count at
# build time.
#
# Invalidation blind spot: if an ingest run adds N chunks and deletes N others
# in the same pass, the count stays the same and the cache is not invalidated.
# In practice this cannot happen: incremental ingest adds then deletes in two
# separate ChromaDB operations, so the count is always correct after the run.
#
# Thread safety: Python's GIL makes the read-then-write on _bm25_cache safe
# for a single-process server.  With multiple uvicorn workers each process
# holds its own copy of module state and its own warm cache — no sharing
# needed.

@dataclass
class _BM25Cache:
    retriever: BM25Retriever
    cache_key: str  # collection count + config signature + kb_filter


# One cache entry per (config+count+kb) key so different KB filters maintain
# independent indexes without blowing each other's cache.
_bm25_cache: dict[str, _BM25Cache] = {}


def _bm25_cache_key(doc_count: int, kb_filter: str = "") -> str:
    """Stable string that changes whenever the index must be rebuilt.

    Includes collection size, config params that affect chunk content, and
    the KB filter so a KB-scoped BM25 index never pollutes another KB's cache.
    """
    base = f"{doc_count}:{config.CHUNK_SIZE}:{config.CHUNK_OVERLAP}:{config.EMBED_MODEL}"
    return f"{base}:{kb_filter}" if kb_filter else base


def load_vector_store() -> Chroma:
    """
    Connect to an existing ChromaDB collection on disk.
    Does NOT re-embed anything — just opens the persisted store.
    """
    if not Path(config.CHROMA_PATH).exists():
        raise FileNotFoundError(
            f"ChromaDB not found at '{config.CHROMA_PATH}'.\n"
            "Run 'python ingest.py' to embed your documents first."
        )

    embeddings = OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    chroma_client = chromadb.PersistentClient(
        path=config.CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    vector_store = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        client=chroma_client,
        embedding_function=embeddings,
    )
    return vector_store


def _kb_where(kb_filter: str) -> dict | None:
    """Return a ChromaDB where clause for KB filtering, or None."""
    return {"kb": {"$eq": kb_filter}} if kb_filter else None


def _build_semantic_retriever(
    vector_store: Chroma, k: int, kb_filter: str = ""
) -> BaseRetriever:
    """
    Vector-based retriever. Uses config.SEARCH_STRATEGY to pick
    "similarity" or "mmr" — falls back to "similarity" for hybrid.
    When kb_filter is set, ChromaDB's where clause limits results to that KB.
    """
    search_type = config.SEARCH_STRATEGY if config.SEARCH_STRATEGY != "hybrid" else "similarity"
    search_kwargs: dict = {"k": k}
    where = _kb_where(kb_filter)
    if where:
        search_kwargs["filter"] = where
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def _build_bm25_retriever(
    vector_store: Chroma, k: int, kb_filter: str = ""
) -> BM25Retriever:
    """Return a BM25 retriever, rebuilding the index only when the collection changes.

    When kb_filter is set, fetches only that KB's documents from ChromaDB so
    the BM25 index covers just that slice of the collection.  Each (config +
    count + kb) combination gets its own cache entry so KB-scoped indexes
    don't interfere with each other or with the unfiltered index.
    """
    current_count: int = vector_store._collection.count()
    key = _bm25_cache_key(current_count, kb_filter)

    if key in _bm25_cache:
        logger.debug("BM25 cache hit (key=%s)", key)
        _bm25_cache[key].retriever.k = k
        return _bm25_cache[key].retriever

    logger.info("BM25 index rebuild — key=%s", key)

    where = _kb_where(kb_filter)
    get_kwargs: dict = {"include": ["documents", "metadatas"]}
    if where:
        get_kwargs["where"] = where
    raw = vector_store.get(**get_kwargs)

    if not raw["documents"]:
        raise RuntimeError(
            "ChromaDB collection is empty. "
            "Run 'python ingest.py' to add documents before querying."
        )
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    retriever = BM25Retriever.from_documents(docs, k=k)
    _bm25_cache[key] = _BM25Cache(retriever=retriever, cache_key=key)
    logger.info("BM25 index ready — %d chunk(s) (kb=%r).", len(docs), kb_filter or "*")
    return retriever


def _build_hybrid_retriever(
    vector_store: Chroma, k: int, kb_filter: str = ""
) -> EnsembleRetriever:
    """
    Hybrid retriever: BM25 + semantic combined via Reciprocal Rank Fusion.
    Both legs respect the optional KB filter.
    """
    semantic = _build_semantic_retriever(vector_store, k, kb_filter)
    bm25     = _build_bm25_retriever(vector_store, k, kb_filter)

    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[config.BM25_WEIGHT, config.SEMANTIC_WEIGHT],
    )


def get_retriever(k: int = config.TOP_K, kb_filter: str = "") -> BaseRetriever:
    """
    Load the vector store and return a retriever for the configured strategy.

    kb_filter: if non-empty, retrieval is scoped to that knowledge base.
      "similarity" → semantic retriever with optional KB where clause
      "mmr"        → same with MMR re-ranking
      "hybrid"     → BM25 + semantic, both KB-filtered
    """
    store = load_vector_store()

    strategy = config.SEARCH_STRATEGY
    if strategy == "hybrid":
        logger.info("strategy=hybrid  (bm25=%s, semantic=%s, k=%d, kb=%r)",
                    config.BM25_WEIGHT, config.SEMANTIC_WEIGHT, k, kb_filter or "*")
        return _build_hybrid_retriever(store, k, kb_filter)

    logger.info("strategy=%s  k=%d  kb=%r", strategy, k, kb_filter or "*")
    return _build_semantic_retriever(store, k, kb_filter)


# ---------------------------------------------------------------------------
# Query-result LRU cache
# ---------------------------------------------------------------------------
# Caches (question, k, strategy) → tuple[Document, ...] so repeated identical
# queries skip the full retrieve pipeline (vector search + BM25 scoring +
# RRF merge) and return immediately from memory.
#
# Key design choices:
#   - Keyed on strategy as well as question+k so a config change never
#     returns stale results from a different search strategy.
#   - Returns a tuple (immutable) so callers cannot mutate the cached value
#     and corrupt future lookups.
#   - maxsize=256 keeps memory bounded; with ~1 KB per Document and k≤16 per
#     query, worst-case is ~4 MB — safe for a typical server process.
#   - Cache is process-local; multiple uvicorn workers each maintain their
#     own independent cache (no sharing needed — each worker warms up quickly).
#
# Cache invalidation: call clear_retrieval_cache() after any ingest run that
# adds or removes documents. The BM25 singleton uses count-based invalidation
# but the query cache has no awareness of collection changes.

def _normalise_query(question: str) -> str:
    """Normalise whitespace and case so minor variations share a cache entry.

    "What is X?" and "what is x?" produce the same embedding and the same
    BM25 results — there is no benefit in caching them separately.
    """
    return " ".join(question.strip().lower().split())


@lru_cache(maxsize=256)
def retrieve_documents(
    question: str, k: int, strategy: str, kb_filter: str = ""
) -> tuple[Document, ...]:
    """Return cached retrieval results for (normalised question, k, strategy, kb_filter).

    kb_filter is part of the cache key so KB-scoped queries never return
    results from the wrong KB on a cache hit.
    """
    retriever = get_retriever(k=k, kb_filter=kb_filter)
    docs = retriever.invoke(question)
    logger.debug(
        "Cache MISS — retrieved %d chunk(s)  (question=%r, k=%d, strategy=%s, kb=%r)",
        len(docs), question[:60], k, strategy, kb_filter or "*",
    )
    return tuple(docs)


def retrieve_documents_normalised(
    question: str, k: int, strategy: str, kb_filter: str = ""
) -> tuple[Document, ...]:
    """Public wrapper that normalises the question before the LRU cache lookup."""
    return retrieve_documents(_normalise_query(question), k, strategy, kb_filter)


def clear_retrieval_cache() -> None:
    """Invalidate all cached retrieval results.

    Call this after any ingest run that modifies the ChromaDB collection.
    Without invalidation, queries answered before the ingest would continue
    returning pre-ingest chunks until the process restarts.
    """
    retrieve_documents.cache_clear()
    logger.info("Retrieval LRU cache cleared.")
