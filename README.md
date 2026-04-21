# Local RAG Project

A local Retrieval-Augmented Generation (RAG) pipeline that lets you query your own documents using a fully offline LLM stack — no API keys, no data leaving your machine.

## Stack

| Component | Tool |
|---|---|
| LLM | Ollama — `qwen2.5:7b-instruct-q4_K_M` |
| Embeddings | Ollama — `nomic-embed-text:latest` |
| Vector store | ChromaDB (local, persistent) |
| Orchestration | LangChain + LangGraph |
| API server | FastAPI + uvicorn |
| Web UI | Vue 3 + Tailwind CSS (served by FastAPI) |
| Reranking | sentence-transformers (CrossEncoder) |
| Retry | tenacity (exponential back-off) |
| Rate limiting | slowapi |
| Language | Python 3.10+ |
| Tests | pytest |
| Dependency locking | pip-tools |

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- Python 3.10+

Pull the required models before running:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text:latest
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install pinned dependencies
make sync-dev                   # dev + test dependencies (recommended)
# or: pip install -r requirements/dev.txt

# 3. Copy and edit config overrides
cp .env.example .env
```

## Usage

### Web UI

Start the FastAPI server — the Vue 3 web UI is served at the same address:

```bash
python server.py           # http://localhost:8000
python server.py --port 9000
uvicorn server:app --reload   # development with auto-reload
```

Open **http://localhost:8000** in your browser. Three pages are available:

| Page | Description |
|---|---|
| **💬 Chat** | Ask questions and get streaming answers grounded in your documents |
| **⚙️ Admin** | Monitor Ollama/ChromaDB health and edit all configuration settings |
| **📁 Data Load** | Upload `.txt`/`.pdf` files, run incremental or full-rebuild ingestion, and manage the vector store |

The Chat page streams tokens directly from the LangGraph pipeline via Server-Sent Events. The Admin → Configuration tab saves changes to `.env` (server restart required to apply). The Data Load page streams live ingestion logs and shows per-source chunk counts.

The UI requires no separate process — it is served by the same FastAPI server as the API.

---

### CLI / API

### Step 1 — Add your documents

Drop `.txt` or `.pdf` files into the `docs/` folder.

### Step 2 — Ingest documents

```bash
# Incremental (default) — only embeds new or changed chunks
python ingest.py

# Full rebuild — re-embeds everything (use after changing CHUNK_SIZE or EMBED_MODEL)
python ingest.py --full-rebuild
```

Incremental ingest computes a SHA-256 hash per chunk and skips chunks already in ChromaDB, making repeated runs fast when most documents haven't changed.

> **After any ingest run**, call `clear_retrieval_cache()` if running a long-lived server process, or simply restart the server.

### Step 3 — Query your documents

**CLI (interactive)**

```bash
python app.py              # default settings
python app.py --k 6        # retrieve 6 chunks per query
```

Type a question and get an answer grounded in your documents. Type `quit` to exit.

Each query prints a live trace with a unique request ID:

```
════════════════════════════════════════════════════════════
  Question:   What is a slowly changing dimension?
  Request ID: a3f1c2d4-7e8b-4f2a-9c1d-0e5f6a7b8c9d
════════════════════════════════════════════════════════════
  [→ route]    route=retrieval
  [↓ retrieve] 8 chunk(s) retrieved
  [✓ grade]    5 chunk(s) after grading
  [✎ generate] "A slowly changing dimension (SCD) is..."
  [✔ verify]   grounded=yes

──────────────────────────────────────────────────────────
  Answer
──────────────────────────────────────────────────────────
A slowly changing dimension (SCD) is a dimension that...
```

**API server**

```bash
python server.py                     # 0.0.0.0:8000
python server.py --port 9000
uvicorn server:app --reload          # development with auto-reload
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — `{"status": "ok"}` |
| `POST` | `/query` | Blocking — returns full answer + metadata |
| `POST` | `/query/stream` | Streaming — Server-Sent Events (SSE) |

**POST /query**

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is a fact table?"}'
```

```json
{
  "request_id": "a3f1c2d4-...",
  "answer": "A fact table stores quantitative measurements...",
  "grounded": true,
  "route": "retrieval",
  "num_docs": 5
}
```

**POST /query/stream** (SSE)

Events arrive as `data: <json>\n\n`:

```
data: {"type": "node",  "node": "route",    "route": "retrieval"}
data: {"type": "node",  "node": "retrieve", "num_docs": 8}
data: {"type": "node",  "node": "grade",    "num_docs": 5}
data: {"type": "token", "token": "A "}
data: {"type": "token", "token": "fact "}
data: {"type": "done",  "request_id": "a3f1c2d4-...", "grounded": true}
```

Rate limit: 10 requests/minute per IP on both POST endpoints.

## Project Structure

```
rag-project/
├── ui/                  # Vue 3 + Tailwind CSS web UI (served by FastAPI)
│   ├── index.html       # SPA shell with sidebar navigation
│   └── components/
│       ├── Chat.js      # Chat page — SSE streaming token output
│       ├── Admin.js     # Admin page — health monitor + config editor
│       └── DataLoad.js  # Data Load page — file upload + ingest pipeline
├── app.py               # Interactive CLI (LangGraph agent)
├── server.py            # FastAPI server (/query, /query/stream)
├── ingest.py            # Document ingestion pipeline (incremental + full-rebuild)
├── evaluate.py          # RAGAS evaluation (faithfulness, relevance, etc.)
├── config.py            # All settings — validated at import time
├── Makefile             # compile / sync / sync-dev / test / serve
├── .env.example         # Config override template (copy to .env)
├── requirements/
│   ├── base.in          # Abstract runtime deps (edit these)
│   ├── base.txt         # Pinned runtime lock file (generated)
│   ├── dev.in           # Abstract dev/eval deps (edit these)
│   └── dev.txt          # Pinned dev lock file (generated)
├── docs/                # Add your .txt and .pdf documents here
├── tests/               # pytest test suite
│   ├── conftest.py      # Shared fixtures
│   ├── test_ingest.py   # Ingestion pipeline tests
│   ├── test_chain.py    # LCEL chain tests
│   ├── test_retriever.py# Retriever + BM25 cache tests
│   └── test_grader.py   # Grader + fallback logic tests
└── rag/                 # Core RAG package
    ├── state.py         # RAGState TypedDict (includes request_id)
    ├── graph.py         # LangGraph agent assembly
    ├── nodes.py         # Async node functions (route/retrieve/grade/generate/verify)
    ├── chain.py         # LCEL chain (alternative to graph)
    ├── retriever.py     # ChromaDB + similarity/MMR/hybrid search + LRU cache
    ├── grader.py        # Self-RAG chunk relevance grader (sync + async)
    ├── reranker.py      # Cross-encoder reranker (lazy-loaded singleton)
    ├── query_expander.py# LLM query expansion (generates N variations)
    └── retry.py         # Tenacity retry helpers (sync + async)
```

## How It Works

### Ingestion (offline, run once)

```
docs/
  → load (.txt, .pdf)
  → split into overlapping chunks
  → hash each chunk (SHA-256 of source path + content)
  → compare hashes with ChromaDB
  → embed and add new/changed chunks only
  → delete chunks whose source was removed
```

### Query pipeline (LangGraph agent, per question)

```
question + UUID request_id
  → route          — retrieval or direct answer?
  → query expand   — generate N alternative phrasings (optional)
  → retrieve       — parallel fetch for all query variations; deduplicate
  → rerank         — cross-encoder scores each (question, chunk) pair (optional)
  → grade          — parallel LLM filter — all chunks graded concurrently
  → generate       — answer from graded context
  → verify         — grounding check (Self-RAG hallucination detection)
```

Graph shape:

```
START → route
  ├─ "retrieval" → retrieve → grade → generate → verify → END
  └─ "direct"   ─────────────────── → generate → verify → END
```

All five nodes are `async def`. Grade runs `asyncio.gather` so all k LLM grading calls are in-flight simultaneously — latency drops from O(k × llm) to O(llm).

## Configuration

All settings live in `config.py`, validated at import time, and overridable via `.env`:

### Core

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:7b-instruct-q4_K_M` | Ollama LLM model |
| `EMBED_MODEL` | `nomic-embed-text:latest` | Ollama embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `CHROMA_COLLECTION` | `rag_docs` | ChromaDB collection name |
| `DOCS_DIR` | `./docs` | Source documents directory |
| `VERBOSE` | `true` | `true` → DEBUG logging, `false` → INFO only |

### Chunking

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `700` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |

### Retrieval

| Variable | Default | Description |
|---|---|---|
| `TOP_K` | `8` | Chunks retrieved per query |
| `GRADING_ENABLED` | `true` | Self-RAG relevance grading |
| `SEARCH_STRATEGY` | `hybrid` | `similarity` / `mmr` / `hybrid` |
| `BM25_WEIGHT` | `0.5` | Hybrid — BM25 keyword weight (must sum to 1.0 with `SEMANTIC_WEIGHT`) |
| `SEMANTIC_WEIGHT` | `0.5` | Hybrid — semantic vector weight |

### Cross-encoder reranking

| Variable | Default | Description |
|---|---|---|
| `RERANKER_ENABLED` | `false` | Enable retrieve-then-rerank |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace model (~80 MB, downloaded on first use) |
| `RERANKER_TOP_N` | `4` | Chunks kept after reranking |

Recommended models:

| Model | Size | Quality |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB | Fast (default) |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~120 MB | Better |
| `cross-encoder/ms-marco-electra-base` | ~440 MB | Best |

### Query expansion

| Variable | Default | Description |
|---|---|---|
| `QUERY_EXPANSION_ENABLED` | `false` | Generate N query variations before retrieval |
| `QUERY_EXPANSION_N` | `3` | Number of alternative phrasings to generate |

### Search strategies

| Strategy | When to use |
|---|---|
| `similarity` | Fast cosine/ANN vector search |
| `mmr` | Repetitive documents — penalises near-duplicate chunks |
| `hybrid` | Technical content with exact terms (product codes, acronyms, names) |

## Dependency Management

Requirements are split and locked with [pip-tools](https://pip-tools.readthedocs.io):

```
requirements/base.in   → base.txt   (runtime lock)
requirements/dev.in    → dev.txt    (dev + test lock, includes base)
```

```bash
# Install pinned deps
make sync         # runtime only
make sync-dev     # all (dev + test)

# Re-pin after editing a .in file — commit both .in and .txt
make compile
```

Always commit both the `.in` source and the generated `.txt` lock file so every environment installs identical package versions.

## Tests

All tests run without Ollama or ChromaDB — every external call is mocked.

```bash
make test
# or: pytest tests/ -v
```

### Coverage

| File | What's tested | Tests |
|---|---|---|
| `test_ingest.py` | `load_documents`, `split_documents`, `build_vector_store`, `incremental_ingest`, `main` | 25 |
| `test_chain.py` | `print_chunks`, `format_docs`, `grade_and_format`, `build_rag_chain` | 24 |
| `test_retriever.py` | `load_vector_store`, `_build_bm25_retriever` (cache logic), `_build_hybrid_retriever`, `get_retriever` | 28 |
| `test_grader.py` | `_parse_grade`, `grade_documents` (filtering, fallback, error handling) | 23 |

### Key mock patterns

- `test_retriever.py` — `autouse` fixture resets the module-level BM25 singleton before every test to prevent state leakage.
- `test_grader.py` — patches `retry_invoke` to control LLM responses; patches `config.GRADING_ENABLED` surgically with `patch.object`.
- `test_chain.py` — uses `RunnableLambda` instead of `MagicMock` for LLM stubs; LCEL's `|` pipe requires real `Runnable` objects.

## Production Features

| Feature | Where |
|---|---|
| Structured logging (`logging` module) | All modules — replaces all `print()` calls |
| Tenacity retry on all Ollama calls | `rag/retry.py` — 4 attempts, exponential back-off 2–30 s |
| Config validation at import | `config.validate()` — fails fast with all errors listed |
| Request / trace IDs | `RAGState.request_id` — UUID generated at entry point, logged on every node |
| Async pipeline | All 5 LangGraph nodes are `async def` |
| Parallel grading | `asyncio.gather` in `async_grade_documents` — O(llm) not O(k × llm) |
| Parallel retrieval | `asyncio.gather` across all expanded query variations |
| BM25 singleton cache | Module-level `_BM25Cache`, invalidated by ChromaDB document count |
| Query-result LRU cache | `@lru_cache(maxsize=256)` on `retrieve_documents(question, k, strategy)` |
| Cross-encoder reranking | `rag/reranker.py` — CPU work offloaded to thread pool via `run_in_executor` |
| Query expansion | `rag/query_expander.py` — LLM generates N phrasings; results merged + deduplicated |
| Input validation | Pydantic `field_validator` on `QueryRequest` — strips whitespace, enforces max length |
| Rate limiting | `slowapi` — 10 req/min per IP on POST endpoints |
| Incremental ingest | SHA-256 hash per chunk — only new/changed chunks are re-embedded |
| Lock files | `pip-tools` — `requirements/base.txt` and `requirements/dev.txt` pin all transitive deps |

## Resetting the Vector Store

```bash
rm -rf chroma_db
python ingest.py
```

Run this after changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBED_MODEL`, since those changes make existing embeddings stale regardless of document content.

## Supported Document Types

| Format | Notes |
|---|---|
| `.txt` | Plain text, UTF-8 |
| `.pdf` | Text-based PDFs (scanned PDFs require OCR preprocessing) |
