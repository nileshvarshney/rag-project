# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fully offline Retrieval-Augmented Generation (RAG) pipeline for querying local documents using local LLMs. No external API calls — everything runs on-device via Ollama + ChromaDB + LangGraph + FastAPI.

## Commands

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
make sync-dev        # install runtime + dev + test deps
cp .env.example .env
# pull required Ollama models before running:
# ollama pull qwen2.5:7b-instruct-q4_K_M
# ollama pull nomic-embed-text:latest
```

### Development
```bash
make test            # run full pytest suite (no Ollama/ChromaDB needed — fully mocked)
pytest tests/test_retriever.py -v   # run a single test file
make ingest          # ingest documents from docs/ into ChromaDB
make serve           # start FastAPI server at http://localhost:8000
python app.py        # interactive CLI query mode
python app.py --k 6  # CLI with custom k-chunk retrieval
```

### Dependency management
```bash
# after editing requirements/*.in files:
make compile         # re-pin lock files
make sync            # install runtime deps only
make sync-dev        # install runtime + dev + test deps
```

## Architecture

### Entry points
- `app.py` — interactive CLI
- `server.py` — FastAPI server (Chat UI, Admin UI, Data Load UI at port 8000)
- `ingest.py` — document ingestion pipeline (one-time / incremental)

### Configuration
`config.py` validates all environment variables at import time and fails fast with a full error list if any are invalid. All tunables are in `.env` (see `.env.example` for reference). Key variables:
- `LLM_MODEL`, `EMBED_MODEL`, `OLLAMA_BASE_URL` — local LLM setup
- `CHROMA_PATH`, `CHROMA_COLLECTION` — vector store
- `TOP_K`, `SEARCH_STRATEGY` (`similarity`|`mmr`|`hybrid`), `BM25_WEIGHT`, `SEMANTIC_WEIGHT`
- `GRADING_ENABLED`, `RERANKER_ENABLED`, `QUERY_EXPANSION_ENABLED` — pipeline feature flags

### Ingestion pipeline (`ingest.py`)
```
docs/ → load_documents() → preprocess_documents() → split into chunks
      → SHA-256 hash per chunk → skip unchanged chunks
      → OllamaEmbeddings → ChromaDB
```
Preprocessing (`rag/preprocessor.py`) cleans artifacts, detects document structure, and adds contextual headers to chunks.

### Query pipeline — LangGraph agent (`rag/graph.py`, `rag/nodes.py`)
All five nodes are `async`. The graph forks at `route_node`:
```
question
  → route_node
      ├─ "direct"    → generate → verify → END
      └─ "retrieval" → retrieve → grade → generate → verify → END
```
1. **route_node** — zero-shot JSON classifier decides if retrieval is needed
2. **retrieve_node** — optional query expansion → parallel fetch across expanded queries → optional cross-encoder reranking (`rag/reranker.py`)
3. **grade_node** — parallel LLM-based relevance filtering (`rag/grader.py`); if all chunks fail, passes an empty list rather than halting
4. **generate_node** — formats surviving chunks as context, prompts LLM
5. **verify_node** — self-RAG grounding check (is answer supported by the retrieved context?)

There is also a simpler LCEL chain (`rag/chain.py`) without branching for use cases that don't need the full graph.

### Retriever (`rag/retriever.py`)
Three strategies selectable via `SEARCH_STRATEGY`:
- **similarity** — cosine/ANN vector search
- **mmr** — Maximal Marginal Relevance (diversifies results)
- **hybrid** — BM25 keyword search + semantic search fused via Reciprocal Rank Fusion (RRF); BM25 index is a singleton cached in memory and invalidated when ChromaDB doc count changes

### Performance patterns
- BM25 singleton cache with count-based invalidation
- LRU query-result cache (256 entries) in retriever
- Chunk grading LRU cache (4096 entries, keyed by question_hash + content_hash)
- Parallel chunk grading and parallel retrieval via `asyncio.gather`
- Tenacity retry with exponential backoff (4 attempts, 2–30s) for Ollama calls (`rag/retry.py`)
- Cross-encoder reranker (`rag/reranker.py`) is lazy-loaded as a singleton

## Testing
All tests mock Ollama and ChromaDB — the full test suite runs without any running services.

Key patterns:
- Reset BM25 singleton between tests via the `reset_bm25_cache` fixture
- Use `RunnableLambda` stubs (not plain callables) when mocking LCEL chains
- Patch config flags surgically: `patch.object(config, 'GRADING_ENABLED', False)`

Test files map to modules: `test_ingest.py`, `test_chain.py`, `test_retriever.py`, `test_grader.py`.
