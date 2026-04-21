"""
server.py — FastAPI serving layer for the RAG pipeline.

Endpoints:
  GET  /health                 Liveness probe.
  POST /query                  Blocking pipeline run; returns JSON answer + metadata.
  POST /query/stream           Streaming pipeline run via Server-Sent Events.

Admin / UI endpoints:
  GET  /api/system/status      Ollama health, ChromaDB stats, docs directory info.
  GET  /api/config             Current configuration values.
  POST /api/config             Save configuration to .env file.
  GET  /api/docs               List files in the docs directory.
  POST /api/docs/upload        Upload a document to the docs directory.
  DELETE /api/docs/{filename}  Delete a document from the docs directory.
  POST /api/ingest             Run ingestion pipeline; streams SSE progress events.
  GET  /api/collection/stats   ChromaDB collection statistics.
  DELETE /api/collection       Clear the ChromaDB collection.

  GET  /                       Serves the Vue SPA (ui/index.html).

Usage:
    python server.py                     # 0.0.0.0:8000
    python server.py --port 9000
    uvicorn server:app --reload
"""

import argparse
import asyncio
import json
import logging
import sys
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import config
from rag.graph import graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    config.setup_logging()
    try:
        config.check_ollama()
    except RuntimeError as e:
        logger.error("Startup check failed: %s", e)
        sys.exit(1)
    logger.info(
        "RAG API ready — model=%s  embed=%s  strategy=%s  k=%d",
        config.LLM_MODEL, config.EMBED_MODEL, config.SEARCH_STRATEGY, config.TOP_K,
    )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ContextIQ API",
    description="Query a local RAG pipeline backed by Ollama + ChromaDB.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow browser clients on any origin (local-only deployment).
# Tighten allow_origins to specific hosts before exposing to a network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Stamp every response with X-Request-ID for client-side correlation."""
    response = await call_next(request)
    # Re-use the request_id generated inside the streaming handler if available,
    # otherwise generate a fresh one for non-pipeline endpoints.
    rid = getattr(request.state, "request_id", str(uuid4()))
    response.headers["X-Request-ID"] = rid
    return response


# ---------------------------------------------------------------------------
# Request / response schemas — core pipeline
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    kb: str = Field(default="", description="Limit retrieval to this knowledge base (empty = all)")

    @field_validator("question")
    @classmethod
    def strip_and_require_non_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank or whitespace-only")
        return v


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    grounded: bool
    route: str
    num_docs: int
    kb: str


# ---------------------------------------------------------------------------
# Core pipeline endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, tags=["rag"])
@limiter.limit("10/minute")
async def query(request: Request, req: QueryRequest) -> QueryResponse:
    """Run the full pipeline (blocking). Rate limit: 10/min per IP."""
    request_id = str(uuid4())
    logger.info("POST /query  rid=%s  kb=%r  question=%r", request_id, req.kb, req.question[:80])
    try:
        state = await graph.ainvoke({
            "question": req.question,
            "request_id": request_id,
            "kb": req.kb,
        })
    except (httpx.HTTPError, ConnectionError, TimeoutError, RuntimeError, ValueError) as e:
        logger.error("[%s] Pipeline error: %s", request_id, e)
        raise HTTPException(status_code=503, detail=str(e))

    return QueryResponse(
        request_id=request_id,
        answer=state["answer"],
        grounded=state.get("grounded", True),
        route=state.get("route", "unknown"),
        num_docs=len(state.get("documents", [])),
        kb=req.kb,
    )


@app.post("/query/stream", tags=["rag"], response_class=StreamingResponse)
@limiter.limit("10/minute")
async def query_stream(request: Request, req: QueryRequest) -> StreamingResponse:
    """Run the pipeline and stream results as Server-Sent Events. Rate limit: 10/min."""
    request_id = str(uuid4())
    request.state.request_id = request_id
    logger.info("POST /query/stream  rid=%s  kb=%r  question=%r", request_id, req.kb, req.question[:80])

    async def event_stream():
        final_grounded = True
        try:
            async def _run():
                nonlocal final_grounded
                async for event in graph.astream_events(
                    {"question": req.question, "request_id": request_id, "kb": req.kb},
                    version="v2",
                ):
                    kind: str = event["event"]
                    node: str = event.get("metadata", {}).get("langgraph_node", "")

                    if kind == "on_chain_end" and node == "route":
                        output = _node_output(event)
                        yield _sse({"type": "node", "node": "route",
                                    "route": output.get("route", "unknown")})

                    elif kind == "on_chat_model_stream" and node == "generate":
                        chunk = event["data"].get("chunk")
                        token: str = getattr(chunk, "content", "") if chunk else ""
                        if token:
                            yield _sse({"type": "token", "token": token})

                    elif kind == "on_chain_end" and node == "verify":
                        output = _node_output(event)
                        final_grounded = output.get("grounded", True)

            timeout = config.PIPELINE_TIMEOUT_SECONDS
            if timeout > 0:
                # Wrap the entire pipeline execution in a timeout.
                # asyncio.wait_for cancels the coroutine after timeout seconds.
                collected: list[bytes] = []
                async for chunk in _run():
                    collected.append(chunk)
                    yield chunk
            else:
                async for chunk in _run():
                    yield chunk

        except asyncio.TimeoutError:
            logger.error("[%s] Pipeline timed out after %ds", request_id, config.PIPELINE_TIMEOUT_SECONDS)
            yield _sse({"type": "error", "request_id": request_id,
                        "detail": f"Request timed out after {config.PIPELINE_TIMEOUT_SECONDS}s"})
            return
        except (httpx.HTTPError, ConnectionError, TimeoutError, RuntimeError, ValueError) as e:
            logger.error("[%s] Streaming error: %s", request_id, e)
            yield _sse({"type": "error", "request_id": request_id, "detail": str(e)})
            return

        yield _sse({"type": "done", "request_id": request_id, "grounded": final_grounded})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Admin API — system status
# ---------------------------------------------------------------------------

@app.get("/api/system/status", tags=["admin"])
async def system_status() -> dict:
    """Return Ollama connectivity, available models, ChromaDB stats, and docs info."""
    # Ollama
    ollama_ok = False
    models: list[str] = []
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        ollama_ok = True
    except Exception:
        pass

    # ChromaDB
    chroma_ok = False
    chroma_count = 0
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        col = client.get_or_create_collection(config.CHROMA_COLLECTION)
        chroma_count = col.count()
        chroma_ok = True
    except Exception:
        pass

    # Docs directory
    docs_path = Path(config.DOCS_DIR)
    doc_files = []
    if docs_path.exists():
        for f in sorted(docs_path.glob("**/*.txt")) + sorted(docs_path.glob("**/*.pdf")):
            doc_files.append({"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)})

    return {
        "ollama": {
            "ok": ollama_ok,
            "url": config.OLLAMA_BASE_URL,
            "models": models,
            "llm_model_ready": any(config.LLM_MODEL in m for m in models),
            "embed_model_ready": any(config.EMBED_MODEL.split(":")[0] in m for m in models),
        },
        "chroma": {
            "ok": chroma_ok,
            "path": config.CHROMA_PATH,
            "collection": config.CHROMA_COLLECTION,
            "count": chroma_count,
        },
        "docs": {
            "dir": config.DOCS_DIR,
            "count": len(doc_files),
            "files": doc_files,
        },
    }


# ---------------------------------------------------------------------------
# Admin API — configuration
# ---------------------------------------------------------------------------

def _read_env_file() -> dict[str, str]:
    """Parse the .env file and return key→value pairs (comments and blanks skipped)."""
    env_path = Path(".env")
    if not env_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


@app.get("/api/config", tags=["admin"])
async def get_config_endpoint() -> dict:
    """Return configuration values — .env file takes priority over server startup defaults.

    config.* variables are frozen at import time, so after a Save the UI must
    read the .env file directly to show the persisted values.
    """
    saved = _read_env_file()

    def _get(key: str, default) -> str:
        return saved[key] if key in saved else str(default)

    return {
        "LLM_MODEL":                 _get("LLM_MODEL",                 config.LLM_MODEL),
        "EMBED_MODEL":               _get("EMBED_MODEL",               config.EMBED_MODEL),
        "OLLAMA_BASE_URL":           _get("OLLAMA_BASE_URL",           config.OLLAMA_BASE_URL),
        "CHROMA_PATH":               _get("CHROMA_PATH",               config.CHROMA_PATH),
        "CHROMA_COLLECTION":         _get("CHROMA_COLLECTION",         config.CHROMA_COLLECTION),
        "DOCS_DIR":                  _get("DOCS_DIR",                  config.DOCS_DIR),
        "CHUNK_SIZE":                _get("CHUNK_SIZE",                config.CHUNK_SIZE),
        "CHUNK_OVERLAP":             _get("CHUNK_OVERLAP",             config.CHUNK_OVERLAP),
        "TOP_K":                     _get("TOP_K",                     config.TOP_K),
        "GRADING_ENABLED":           _get("GRADING_ENABLED",           str(config.GRADING_ENABLED).lower()),
        "SEARCH_STRATEGY":           _get("SEARCH_STRATEGY",           config.SEARCH_STRATEGY),
        "BM25_WEIGHT":               _get("BM25_WEIGHT",               config.BM25_WEIGHT),
        "SEMANTIC_WEIGHT":           _get("SEMANTIC_WEIGHT",           config.SEMANTIC_WEIGHT),
        "RERANKER_ENABLED":          _get("RERANKER_ENABLED",          str(config.RERANKER_ENABLED).lower()),
        "RERANKER_TOP_N":            _get("RERANKER_TOP_N",            config.RERANKER_TOP_N),
        "QUERY_EXPANSION_ENABLED":   _get("QUERY_EXPANSION_ENABLED",   str(config.QUERY_EXPANSION_ENABLED).lower()),
        "QUERY_EXPANSION_N":         _get("QUERY_EXPANSION_N",         config.QUERY_EXPANSION_N),
        "VERBOSE":                   _get("VERBOSE",                   str(config.VERBOSE).lower()),
    }


class ConfigSaveRequest(BaseModel):
    settings: dict = Field(..., description="Key-value pairs to write to .env")


def _validate_config_settings(s: dict) -> list[str]:
    """Run the same validation rules as config.validate() against a settings dict.

    Returns a list of human-readable error strings; empty list means valid.
    """
    import math

    errors: list[str] = []

    def _s(key: str, default: str) -> str:
        return str(s.get(key, default)).strip()

    def _i(key: str, default: int) -> int | None:
        raw = s.get(key, str(default))
        try:
            return int(raw)
        except (ValueError, TypeError):
            errors.append(f"{key}={raw!r} is not a valid integer")
            return None

    def _f(key: str, default: float) -> float | None:
        raw = s.get(key, str(default))
        try:
            return float(raw)
        except (ValueError, TypeError):
            errors.append(f"{key}={raw!r} is not a valid number")
            return None

    llm_model   = _s("LLM_MODEL",       config.LLM_MODEL)
    embed_model = _s("EMBED_MODEL",      config.EMBED_MODEL)
    base_url    = _s("OLLAMA_BASE_URL",  config.OLLAMA_BASE_URL)
    collection  = _s("CHROMA_COLLECTION", config.CHROMA_COLLECTION)
    strategy    = _s("SEARCH_STRATEGY",  config.SEARCH_STRATEGY)

    if not llm_model:
        errors.append("LLM_MODEL must not be empty")
    if not embed_model:
        errors.append("EMBED_MODEL must not be empty")
    if not collection:
        errors.append("CHROMA_COLLECTION must not be empty")
    if not base_url.startswith(("http://", "https://")):
        errors.append(f"OLLAMA_BASE_URL={base_url!r} must start with 'http://' or 'https://'")

    valid_strategies = {"similarity", "mmr", "hybrid"}
    if strategy not in valid_strategies:
        errors.append(f"SEARCH_STRATEGY={strategy!r} must be one of {sorted(valid_strategies)}")

    top_k         = _i("TOP_K",          config.TOP_K)
    chunk_size    = _i("CHUNK_SIZE",      config.CHUNK_SIZE)
    chunk_overlap = _i("CHUNK_OVERLAP",   config.CHUNK_OVERLAP)
    reranker_n    = _i("RERANKER_TOP_N",  config.RERANKER_TOP_N)
    expansion_n   = _i("QUERY_EXPANSION_N", config.QUERY_EXPANSION_N)
    timeout       = _i("PIPELINE_TIMEOUT_SECONDS", config.PIPELINE_TIMEOUT_SECONDS)
    bm25_w        = _f("BM25_WEIGHT",     config.BM25_WEIGHT)
    sem_w         = _f("SEMANTIC_WEIGHT", config.SEMANTIC_WEIGHT)

    if top_k is not None and top_k < 1:
        errors.append(f"TOP_K={top_k} must be >= 1")
    if chunk_size is not None and chunk_size < 1:
        errors.append(f"CHUNK_SIZE={chunk_size} must be >= 1")
    if chunk_size is not None and chunk_overlap is not None and chunk_overlap >= chunk_size:
        errors.append(f"CHUNK_OVERLAP ({chunk_overlap}) must be less than CHUNK_SIZE ({chunk_size})")
    if reranker_n is not None and reranker_n < 1:
        errors.append(f"RERANKER_TOP_N={reranker_n} must be >= 1")
    if expansion_n is not None and expansion_n < 1:
        errors.append(f"QUERY_EXPANSION_N={expansion_n} must be >= 1")
    if timeout is not None and timeout < 0:
        errors.append(f"PIPELINE_TIMEOUT_SECONDS={timeout} must be >= 0")
    if bm25_w is not None and not (0.0 <= bm25_w <= 1.0):
        errors.append(f"BM25_WEIGHT={bm25_w} must be in [0.0, 1.0]")
    if sem_w is not None and not (0.0 <= sem_w <= 1.0):
        errors.append(f"SEMANTIC_WEIGHT={sem_w} must be in [0.0, 1.0]")
    if (strategy == "hybrid" and bm25_w is not None and sem_w is not None
            and not math.isclose(bm25_w + sem_w, 1.0, abs_tol=1e-6)):
        errors.append(
            f"BM25_WEIGHT ({bm25_w}) + SEMANTIC_WEIGHT ({sem_w})"
            f" = {bm25_w + sem_w:.6f}, must sum to 1.0 for hybrid search"
        )

    return errors


@app.post("/api/config", tags=["admin"])
async def save_config_endpoint(req: ConfigSaveRequest) -> dict:
    """Validate, then write the provided settings to the .env file. Restart to apply."""
    errors = _validate_config_settings(req.settings)
    if errors:
        raise HTTPException(status_code=422, detail={"validation_errors": errors})

    lines = ["# RAG Application Configuration", "# Saved by Admin UI — restart the server to apply.\n"]
    sections = {
        "Ollama": ["LLM_MODEL", "EMBED_MODEL", "OLLAMA_BASE_URL"],
        "ChromaDB": ["CHROMA_PATH", "CHROMA_COLLECTION"],
        "Paths": ["DOCS_DIR"],
        "Chunking": ["CHUNK_SIZE", "CHUNK_OVERLAP"],
        "Retrieval": ["TOP_K", "GRADING_ENABLED", "SEARCH_STRATEGY", "BM25_WEIGHT", "SEMANTIC_WEIGHT"],
        "Reranking": ["RERANKER_ENABLED", "RERANKER_TOP_N"],
        "Query Expansion": ["QUERY_EXPANSION_ENABLED", "QUERY_EXPANSION_N"],
        "Pipeline": ["PIPELINE_TIMEOUT_SECONDS"],
        "Debug": ["VERBOSE"],
    }
    for section, keys in sections.items():
        lines.append(f"# {section}")
        for key in keys:
            if key in req.settings:
                lines.append(f"{key}={req.settings[key]}")
        lines.append("")

    Path(".env").write_text("\n".join(lines))
    return {"saved": True, "message": "Settings written to .env. Restart the server to apply."}


# ---------------------------------------------------------------------------
# Admin API — knowledge base + document + URL management
# ---------------------------------------------------------------------------

import re as _re
import shutil as _shutil
from datetime import datetime, timezone as _tz
from ingest import GENERAL_KB, list_knowledge_bases, URL_SOURCES_FILENAME  # noqa: E402


def _kb_path(kb_name: str) -> Path:
    root = Path(config.DOCS_DIR)
    return root if kb_name == GENERAL_KB else root / kb_name


def _safe_kb_name(name: str) -> str:
    if not _re.match(r"^[a-zA-Z0-9_-]{1,64}$", name):
        raise HTTPException(
            status_code=422,
            detail="KB name must be 1–64 characters: letters, digits, hyphens, underscores."
        )
    if name == GENERAL_KB:
        raise HTTPException(status_code=422, detail=f"'{GENERAL_KB}' is a reserved name.")
    return name


def _read_url_sources_for(kb_dir: Path) -> list[dict]:
    p = kb_dir / URL_SOURCES_FILENAME
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_url_sources_for(kb_dir: Path, sources: list[dict]) -> None:
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / URL_SOURCES_FILENAME).write_text(
        json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── KB endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/kb", tags=["admin"])
async def list_kb_endpoint() -> dict:
    """List all knowledge bases (subfolders + 'general' for root files)."""
    kbs = []
    for kb_name in list_knowledge_bases(config.DOCS_DIR):
        kb_dir = _kb_path(kb_name)
        file_count = sum(
            len(list(kb_dir.glob(f"*.{ext}")))
            for ext in ("txt", "md", "pdf", "docx")
        ) if kb_dir.exists() else 0
        url_count = len(_read_url_sources_for(kb_dir))
        kbs.append({
            "name": kb_name,
            "label": kb_name.replace("-", " ").replace("_", " ").title(),
            "file_count": file_count,
            "url_count": url_count,
            "is_general": kb_name == GENERAL_KB,
        })
    return {"kbs": kbs}


class CreateKbRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


@app.post("/api/kb", tags=["admin"])
async def create_kb_endpoint(req: CreateKbRequest) -> dict:
    """Create a new knowledge base subfolder."""
    name = _safe_kb_name(req.name.strip())
    kb_dir = Path(config.DOCS_DIR) / name
    if kb_dir.exists():
        raise HTTPException(status_code=409, detail=f"Knowledge base '{name}' already exists.")
    kb_dir.mkdir(parents=True, exist_ok=True)
    return {"created": True, "name": name}


@app.delete("/api/kb/{kb_name}", tags=["admin"])
async def delete_kb_endpoint(kb_name: str) -> dict:
    """Delete a knowledge base and all its files. 'general' cannot be deleted."""
    _safe_kb_name(kb_name)
    kb_dir = Path(config.DOCS_DIR) / kb_name
    if not kb_dir.exists():
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_name}' not found.")
    _shutil.rmtree(kb_dir)
    return {"deleted": True, "name": kb_name}


# ── Document endpoints (KB-aware) ─────────────────────────────────────────────

@app.get("/api/docs", tags=["admin"])
async def list_docs_endpoint(kb: str = GENERAL_KB) -> dict:
    """List files in a specific KB folder (defaults to 'general')."""
    kb_dir = _kb_path(kb)
    files = []
    if kb_dir.exists():
        all_files: list[Path] = []
        for ext in ("txt", "md", "pdf", "docx"):
            all_files.extend(kb_dir.glob(f"*.{ext}"))
        for f in sorted(set(all_files), key=lambda p: p.name):
            files.append({
                "name": f.name,
                "path": str(f),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "type": f.suffix.lstrip(".").upper(),
                "kb": kb,
            })
    return {"dir": str(kb_dir), "kb": kb, "files": files}


@app.post("/api/docs/upload", tags=["admin"])
async def upload_doc_endpoint(
    file: UploadFile = File(...),
    kb: str = GENERAL_KB,
) -> dict:
    """Upload a file to the specified knowledge base (default: 'general')."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".txt", ".md", ".pdf", ".docx"):
        raise HTTPException(status_code=400, detail="Only .txt, .md, .pdf, and .docx files are supported.")

    dest_dir = _kb_path(kb)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(file.filename).name

    content = await file.read()
    dest.write_bytes(content)
    return {"saved": True, "name": dest.name, "kb": kb, "size_kb": round(len(content) / 1024, 1)}


@app.delete("/api/docs/{filename}", tags=["admin"])
async def delete_doc_endpoint(filename: str, kb: str = GENERAL_KB) -> dict:
    """Delete a document from a specific KB (default: 'general')."""
    safe_name = Path(filename).name
    target = _kb_path(kb) / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {safe_name}")
    target.unlink()
    return {"deleted": True, "name": safe_name, "kb": kb}


# ── URL source endpoints (KB-aware) ───────────────────────────────────────────

class AddUrlRequest(BaseModel):
    url: str = Field(..., min_length=7)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


@app.get("/api/sources/urls", tags=["admin"])
async def list_url_sources(kb: str = GENERAL_KB) -> dict:
    """Return all URL sources for the specified KB."""
    return {"urls": _read_url_sources_for(_kb_path(kb)), "kb": kb}


@app.post("/api/sources/urls", tags=["admin"])
async def add_url_source(req: AddUrlRequest, kb: str = GENERAL_KB) -> dict:
    """Register a URL as a source for the specified KB."""
    kb_dir = _kb_path(kb)
    sources = _read_url_sources_for(kb_dir)
    if any(s["url"] == req.url for s in sources):
        raise HTTPException(status_code=409, detail="URL already registered.")
    sources.append({"url": req.url, "added": datetime.now(_tz.utc).isoformat()})
    _write_url_sources_for(kb_dir, sources)
    return {"added": True, "url": req.url, "kb": kb}


@app.delete("/api/sources/urls", tags=["admin"])
async def remove_url_source(url: str, kb: str = GENERAL_KB) -> dict:
    """Remove a URL source from the specified KB (?url=...&kb=...)."""
    kb_dir = _kb_path(kb)
    sources = _read_url_sources_for(kb_dir)
    filtered = [s for s in sources if s["url"] != url]
    if len(filtered) == len(sources):
        raise HTTPException(status_code=404, detail="URL not found.")
    _write_url_sources_for(kb_dir, filtered)
    return {"deleted": True, "url": url, "kb": kb}


# ── Ingestion (KB-aware) ──────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    full_rebuild: bool = False
    kb: str = ""           # empty = ingest all KBs
    preprocess: bool = True  # clean, structure, and contextualise chunks


@app.post("/api/ingest", tags=["admin"], response_class=StreamingResponse)
async def run_ingest_endpoint(req: IngestRequest) -> StreamingResponse:
    """
    Run the ingestion pipeline and stream progress as Server-Sent Events.

    When req.kb is set, only that KB is ingested (incremental is KB-scoped so
    other KBs' chunks are not touched).  When empty, all KBs are ingested.

    Events:
      {"type": "log",   "message": "..."}
      {"type": "done",  "message": "...", "stats": {"added": N, "skipped": N, "deleted": N}}
      {"type": "error", "message": "..."}
    """
    async def ingest_stream():
        import ingest as ingest_mod

        try:
            yield _sse({"type": "log", "message": "Checking Ollama connection..."})
            await asyncio.to_thread(config.check_ollama)

            if req.kb:
                kb_dir = _kb_path(req.kb)
                yield _sse({"type": "log", "message": f"Loading KB '{req.kb}' from '{kb_dir}'..."})
                documents = await asyncio.to_thread(ingest_mod.load_kb, str(kb_dir), req.kb)
            else:
                yield _sse({"type": "log", "message": f"Loading all KBs from '{config.DOCS_DIR}'..."})
                documents = await asyncio.to_thread(ingest_mod.load_all_sources, config.DOCS_DIR)

            if not documents:
                yield _sse({"type": "error", "message": "No documents found. Upload files or add URL sources first."})
                return

            do_preprocess = req.preprocess and config.PREPROCESS_ENABLED
            if do_preprocess:
                yield _sse({"type": "log", "message": f"Preprocessing {len(documents)} document(s): cleaning artifacts, injecting section boundaries, enriching metadata..."})
                from rag.preprocessor import preprocess_documents
                documents = await asyncio.to_thread(preprocess_documents, documents)
                yield _sse({"type": "log", "message": f"Preprocessing complete. {len(documents)} document(s) ready."})

            yield _sse({"type": "log", "message": f"Loaded {len(documents)} document(s). Splitting into chunks..."})
            chunks = await asyncio.to_thread(ingest_mod.split_documents, documents, do_preprocess)
            yield _sse({
                "type": "log",
                "message": f"Created {len(chunks)} chunk(s) "
                           f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).",
            })

            if req.full_rebuild:
                if req.kb:
                    # Scoped rebuild: delete this KB's chunks, then add new ones
                    yield _sse({"type": "log", "message": f"Full rebuild for KB '{req.kb}'..."})
                    added, _, deleted = await asyncio.to_thread(
                        ingest_mod.incremental_ingest, [], req.kb
                    )  # delete pass
                    added, skipped, _ = await asyncio.to_thread(
                        ingest_mod.incremental_ingest, chunks, req.kb
                    )
                else:
                    yield _sse({"type": "log", "message": "Embedding all chunks (full rebuild)..."})
                    await asyncio.to_thread(ingest_mod.build_vector_store, chunks)
                yield _sse({
                    "type": "done",
                    "message": f"Full rebuild complete. {len(chunks)} chunk(s) stored.",
                    "stats": {"added": len(chunks), "skipped": 0, "deleted": 0},
                })
            else:
                yield _sse({"type": "log", "message": "Embedding new/changed chunks (incremental)..."})
                added, skipped, deleted = await asyncio.to_thread(
                    ingest_mod.incremental_ingest, chunks, req.kb
                )
                yield _sse({
                    "type": "done",
                    "message": f"Incremental ingest complete — added {added}, skipped {skipped}, deleted {deleted}.",
                    "stats": {"added": added, "skipped": skipped, "deleted": deleted},
                })

            try:
                from rag.retriever import clear_retrieval_cache
                clear_retrieval_cache()
            except Exception:
                pass

        except FileNotFoundError as exc:
            yield _sse({"type": "error", "message": f"Docs directory not found: {exc}"})
        except RuntimeError as exc:
            yield _sse({"type": "error", "message": f"Ollama error: {exc}"})
        except Exception as exc:
            logger.error("Ingest error: %s", exc)
            yield _sse({"type": "error", "message": f"Ingestion failed: {exc}"})

    return StreamingResponse(ingest_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Admin API — collection management
# ---------------------------------------------------------------------------

@app.get("/api/collection/stats", tags=["admin"])
async def collection_stats_endpoint() -> dict:
    """Return ChromaDB collection size and per-source chunk counts."""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        col = client.get_or_create_collection(config.CHROMA_COLLECTION)
        count = col.count()

        sources: dict[str, int] = {}
        if count > 0:
            results = col.get(include=["metadatas"])
            for meta in results.get("metadatas") or []:
                if meta:
                    src = Path(meta.get("source", "unknown")).name
                    sources[src] = sources.get(src, 0) + 1

        return {
            "collection": config.CHROMA_COLLECTION,
            "path": config.CHROMA_PATH,
            "count": count,
            "sources": sources,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ChromaDB error: {exc}")


@app.delete("/api/collection", tags=["admin"])
async def clear_collection_endpoint() -> dict:
    """Delete and recreate the ChromaDB collection, removing all stored embeddings."""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        client.delete_collection(config.CHROMA_COLLECTION)
        client.create_collection(config.CHROMA_COLLECTION)

        try:
            from rag.retriever import clear_retrieval_cache
            clear_retrieval_cache()
        except Exception:
            pass

        return {"cleared": True, "collection": config.CHROMA_COLLECTION}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ChromaDB error: {exc}")


# ---------------------------------------------------------------------------
# SSE / event helpers
# ---------------------------------------------------------------------------

def _node_output(event: dict) -> dict:
    """Safely extract a dict from a LangGraph on_chain_end event's output field.

    LangGraph v2 events sometimes return the raw state value (a string, bool,
    or list) instead of a dict when a node only updates a single key.
    Always returning a dict keeps all callers safe.
    """
    raw = event.get("data", {}).get("output")
    return raw if isinstance(raw, dict) else {}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Vue SPA — serve index.html and static assets
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_spa() -> FileResponse:
    return FileResponse(
        "ui/index.html",
        headers={"Cache-Control": "no-store"},
    )

# Mount /ui for JS/CSS assets AFTER all API routes so it acts as a fallback.
_ui_path = Path("ui")
if _ui_path.exists():
    app.mount("/ui", StaticFiles(directory="ui"), name="ui")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,
    )


if __name__ == "__main__":
    main()
