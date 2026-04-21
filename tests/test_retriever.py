"""
test_retriever.py — Unit tests for rag/retriever.py.

Strategy:
  - load_vector_store: mock chromadb.PersistentClient, OllamaEmbeddings, and
    Chroma so tests never touch disk or Ollama.  tmp_path controls whether the
    CHROMA_PATH directory "exists" by pointing config at a real or missing dir.

  - _build_bm25_retriever: the cache is module-level state, so an autouse
    fixture resets it to None before and after every test.  A helper builds a
    fake Chroma store whose count() and get() are controllable via arguments.

  - _build_hybrid_retriever: verifies EnsembleRetriever composition and weight
    forwarding without touching real retrievers.

  - get_retriever: patches load_vector_store so no disk I/O occurs, then
    checks that each strategy produces the right retriever type.
"""

import pytest
from unittest.mock import MagicMock, patch, call

import config
import rag.retriever as retriever_module
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.retriever import (
    _build_bm25_retriever,
    _build_hybrid_retriever,
    get_retriever,
    load_vector_store,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_fake_store(doc_count: int = 3, docs=None, metadatas=None) -> MagicMock:
    """Return a MagicMock Chroma store with controllable count/get/as_retriever."""
    if docs is None:
        docs = [f"chunk content {i}" for i in range(doc_count)]
    if metadatas is None:
        metadatas = [{"source": "test.txt"}] * len(docs)

    store = MagicMock()
    store._collection.count.return_value = doc_count
    store.get.return_value = {"documents": docs, "metadatas": metadatas}
    store.as_retriever.return_value = MagicMock(spec=BaseRetriever)
    return store


@pytest.fixture(autouse=True)
def reset_bm25_cache():
    """Isolate every test from module-level BM25 cache state."""
    retriever_module._bm25_cache = None
    yield
    retriever_module._bm25_cache = None


# ---------------------------------------------------------------------------
# load_vector_store
# ---------------------------------------------------------------------------

class TestLoadVectorStore:
    """Tests for load_vector_store() — ChromaDB connection setup."""

    def test_raises_when_chroma_path_missing(self, tmp_path) -> None:
        """FileNotFoundError with a helpful message when CHROMA_PATH doesn't exist."""
        missing = str(tmp_path / "no_db_here")
        with patch.object(config, "CHROMA_PATH", missing):
            with pytest.raises(FileNotFoundError, match="ChromaDB not found"):
                load_vector_store()

    def test_creates_persistent_client_with_configured_path(self, tmp_path) -> None:
        """PersistentClient must receive the path from config, not a hardcoded value."""
        with patch.object(config, "CHROMA_PATH", str(tmp_path)), \
             patch("rag.retriever.chromadb.PersistentClient") as mock_client, \
             patch("rag.retriever.OllamaEmbeddings"), \
             patch("rag.retriever.Chroma"):
            load_vector_store()

        _, kwargs = mock_client.call_args
        assert kwargs["path"] == str(tmp_path)

    def test_creates_chroma_with_configured_collection(self, tmp_path) -> None:
        """Chroma must be opened with the collection name from config."""
        with patch.object(config, "CHROMA_PATH", str(tmp_path)), \
             patch("rag.retriever.chromadb.PersistentClient"), \
             patch("rag.retriever.OllamaEmbeddings"), \
             patch("rag.retriever.Chroma") as mock_chroma:
            load_vector_store()

        _, kwargs = mock_chroma.call_args
        assert kwargs["collection_name"] == config.CHROMA_COLLECTION

    def test_returns_chroma_instance(self, tmp_path) -> None:
        """Return value must be the Chroma object, not the client or embeddings."""
        with patch.object(config, "CHROMA_PATH", str(tmp_path)), \
             patch("rag.retriever.chromadb.PersistentClient"), \
             patch("rag.retriever.OllamaEmbeddings"), \
             patch("rag.retriever.Chroma") as mock_chroma:
            result = load_vector_store()

        assert result is mock_chroma.return_value

    def test_embeddings_use_configured_model(self, tmp_path) -> None:
        """OllamaEmbeddings must be initialised with EMBED_MODEL and OLLAMA_BASE_URL."""
        with patch.object(config, "CHROMA_PATH", str(tmp_path)), \
             patch("rag.retriever.chromadb.PersistentClient"), \
             patch("rag.retriever.OllamaEmbeddings") as mock_embed, \
             patch("rag.retriever.Chroma"):
            load_vector_store()

        mock_embed.assert_called_once_with(
            model=config.EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )


# ---------------------------------------------------------------------------
# _build_bm25_retriever — cache behaviour
# ---------------------------------------------------------------------------

class TestBuildBM25Retriever:
    """Tests for _build_bm25_retriever() — BM25 singleton cache logic."""

    def test_cold_start_returns_bm25_retriever(self) -> None:
        store = _make_fake_store(doc_count=3)
        result = _build_bm25_retriever(store, k=5)
        assert isinstance(result, BM25Retriever)

    def test_cold_start_fetches_documents_from_store(self) -> None:
        """On first call the function must pull documents from ChromaDB."""
        store = _make_fake_store(doc_count=3)
        _build_bm25_retriever(store, k=5)
        store.get.assert_called_once()

    def test_cold_start_populates_cache_with_doc_count(self) -> None:
        """Cache must record the current collection count for future comparisons."""
        store = _make_fake_store(doc_count=4)
        _build_bm25_retriever(store, k=5)
        assert retriever_module._bm25_cache is not None
        assert retriever_module._bm25_cache.doc_count == 4

    def test_cache_hit_skips_store_get(self) -> None:
        """Subsequent call with same count must not fetch documents again."""
        store = _make_fake_store(doc_count=3)
        _build_bm25_retriever(store, k=5)   # cold start
        store.get.reset_mock()
        _build_bm25_retriever(store, k=5)   # cache hit
        store.get.assert_not_called()

    def test_cache_hit_updates_k_in_place(self) -> None:
        """A cache hit must update k without rebuilding the index."""
        store = _make_fake_store(doc_count=3)
        _build_bm25_retriever(store, k=5)
        result = _build_bm25_retriever(store, k=2)
        assert result.k == 2

    def test_k_applied_on_cold_start(self) -> None:
        store = _make_fake_store(doc_count=3)
        result = _build_bm25_retriever(store, k=8)
        assert result.k == 8

    def test_cache_miss_when_count_increases(self) -> None:
        """After an ingest that adds documents, the index must be rebuilt."""
        store = _make_fake_store(doc_count=3)
        _build_bm25_retriever(store, k=5)

        # Simulate ingest adding two more documents
        store._collection.count.return_value = 5
        store.get.return_value = {
            "documents": [f"chunk {i}" for i in range(5)],
            "metadatas": [{"source": "test.txt"}] * 5,
        }
        store.get.reset_mock()
        _build_bm25_retriever(store, k=5)

        store.get.assert_called_once()

    def test_cache_updated_after_rebuild(self) -> None:
        """After a cache miss the stored doc_count must reflect the new total."""
        store = _make_fake_store(doc_count=3)
        _build_bm25_retriever(store, k=5)

        store._collection.count.return_value = 7
        store.get.return_value = {
            "documents": [f"c{i}" for i in range(7)],
            "metadatas": [{}] * 7,
        }
        _build_bm25_retriever(store, k=5)

        assert retriever_module._bm25_cache.doc_count == 7

    def test_same_retriever_object_returned_on_cache_hit(self) -> None:
        """Cache hit must return the identical Python object, not a copy."""
        store = _make_fake_store(doc_count=3)
        first  = _build_bm25_retriever(store, k=5)
        second = _build_bm25_retriever(store, k=5)
        assert first is second

    def test_raises_when_collection_is_empty(self) -> None:
        """An empty ChromaDB collection must raise RuntimeError, not silently fail."""
        store = _make_fake_store(doc_count=0, docs=[], metadatas=[])
        with pytest.raises(RuntimeError, match="ChromaDB collection is empty"):
            _build_bm25_retriever(store, k=5)

    def test_metadata_attached_to_indexed_documents(self) -> None:
        """Metadata from ChromaDB must be passed into the BM25 index documents."""
        meta = {"source": "special.txt", "page": 3}
        store = _make_fake_store(
            doc_count=1,
            docs=["some content"],
            metadatas=[meta],
        )
        # If metadata were lost, the retriever would still build — we verify
        # that get() is called with metadatas included so nothing is dropped.
        _build_bm25_retriever(store, k=5)
        store.get.assert_called_once_with(include=["documents", "metadatas"])


# ---------------------------------------------------------------------------
# _build_hybrid_retriever
# ---------------------------------------------------------------------------

class TestBuildHybridRetriever:
    """Tests for _build_hybrid_retriever() — BM25 + semantic composition."""

    def test_returns_ensemble_retriever(self) -> None:
        store = _make_fake_store(doc_count=3)
        result = _build_hybrid_retriever(store, k=5)
        assert isinstance(result, EnsembleRetriever)

    def test_ensemble_contains_exactly_two_retrievers(self) -> None:
        """The EnsembleRetriever must have one BM25 leg and one semantic leg."""
        store = _make_fake_store(doc_count=3)
        result = _build_hybrid_retriever(store, k=5)
        assert len(result.retrievers) == 2

    def test_uses_configured_bm25_weight(self) -> None:
        store = _make_fake_store(doc_count=3)
        result = _build_hybrid_retriever(store, k=5)
        assert result.weights[0] == config.BM25_WEIGHT

    def test_uses_configured_semantic_weight(self) -> None:
        store = _make_fake_store(doc_count=3)
        result = _build_hybrid_retriever(store, k=5)
        assert result.weights[1] == config.SEMANTIC_WEIGHT

    def test_calls_as_retriever_for_semantic_leg(self) -> None:
        """_build_semantic_retriever must delegate to vector_store.as_retriever()."""
        store = _make_fake_store(doc_count=3)
        _build_hybrid_retriever(store, k=5)
        store.as_retriever.assert_called_once()

    def test_as_retriever_receives_k(self) -> None:
        """The k value must appear in the search_kwargs passed to as_retriever."""
        store = _make_fake_store(doc_count=3)
        _build_hybrid_retriever(store, k=6)
        _, kwargs = store.as_retriever.call_args
        assert kwargs["search_kwargs"]["k"] == 6


# ---------------------------------------------------------------------------
# get_retriever
# ---------------------------------------------------------------------------

class TestGetRetriever:
    """Tests for get_retriever() — strategy dispatch and k forwarding."""

    def test_raises_when_chroma_path_missing(self, tmp_path) -> None:
        """Missing CHROMA_PATH must propagate as FileNotFoundError before querying."""
        with patch.object(config, "CHROMA_PATH", str(tmp_path / "missing")):
            with pytest.raises(FileNotFoundError):
                get_retriever()

    def test_hybrid_strategy_returns_ensemble_retriever(self) -> None:
        store = _make_fake_store(doc_count=3)
        with patch("rag.retriever.load_vector_store", return_value=store), \
             patch.object(config, "SEARCH_STRATEGY", "hybrid"):
            result = get_retriever(k=5)
        assert isinstance(result, EnsembleRetriever)

    def test_similarity_strategy_does_not_return_ensemble(self) -> None:
        """Similarity strategy must use a plain vector retriever, not an ensemble."""
        store = _make_fake_store(doc_count=3)
        with patch("rag.retriever.load_vector_store", return_value=store), \
             patch.object(config, "SEARCH_STRATEGY", "similarity"):
            result = get_retriever(k=5)
        assert not isinstance(result, EnsembleRetriever)

    def test_similarity_strategy_calls_as_retriever(self) -> None:
        store = _make_fake_store(doc_count=3)
        with patch("rag.retriever.load_vector_store", return_value=store), \
             patch.object(config, "SEARCH_STRATEGY", "similarity"):
            get_retriever(k=5)
        store.as_retriever.assert_called_once()

    def test_k_forwarded_to_semantic_retriever(self) -> None:
        """The k argument must appear in search_kwargs, not be ignored."""
        store = _make_fake_store(doc_count=3)
        with patch("rag.retriever.load_vector_store", return_value=store), \
             patch.object(config, "SEARCH_STRATEGY", "similarity"):
            get_retriever(k=11)
        _, kwargs = store.as_retriever.call_args
        assert kwargs["search_kwargs"]["k"] == 11

    def test_mmr_strategy_uses_as_retriever(self) -> None:
        """MMR is implemented via as_retriever(search_type='mmr'), not ensemble."""
        store = _make_fake_store(doc_count=3)
        with patch("rag.retriever.load_vector_store", return_value=store), \
             patch.object(config, "SEARCH_STRATEGY", "mmr"):
            get_retriever(k=5)
        _, kwargs = store.as_retriever.call_args
        assert kwargs["search_type"] == "mmr"
