"""
config.py — Central configuration for the RAG pipeline.

Keeping all settings here makes it easy to experiment:
swap models, tune chunk sizes, or change the DB path
without hunting through multiple files.
"""

import logging
import math
import os
import sys
import urllib.error
import urllib.request
from dotenv import load_dotenv

load_dotenv()  # reads .env if present; falls back to env vars

# Disable ChromaDB's anonymous telemetry.
# Must be set before chromadb is imported anywhere.
os.environ["ANONYMIZED_TELEMETRY"] = "False"


# ---------------------------------------------------------------------------
# Safe env-var helpers
# ---------------------------------------------------------------------------
# os.getenv always returns a string. Passing that string directly to int()
# or float() raises ValueError if the value is non-numeric (e.g. "abc").
# These helpers surface a clear error message instead of a raw traceback.

def _int(key: str, default: int) -> int:
    """Read an env var and convert it to int.

    Args:
        key:     Environment variable name.
        default: Fallback value if the variable is not set.

    Returns:
        The integer value of the env var.

    Raises:
        ValueError: If the env var is set but cannot be parsed as an integer.
    """
    val = os.getenv(key, str(default))
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Config error: {key}={val!r} is not a valid integer")


def _float(key: str, default: float) -> float:
    """Read an env var and convert it to float.

    Args:
        key:     Environment variable name.
        default: Fallback value if the variable is not set.

    Returns:
        The float value of the env var.

    Raises:
        ValueError: If the env var is set but cannot be parsed as a float.
    """
    val = os.getenv(key, str(default))
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Config error: {key}={val!r} is not a valid number")


# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------

LLM_MODEL       = os.getenv("LLM_MODEL",       "qwen2.5:7b-instruct-q4_K_M")
EMBED_MODEL     = os.getenv("EMBED_MODEL",      "nomic-embed-text:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")


# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------

CHROMA_PATH       = os.getenv("CHROMA_PATH",       "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs")


# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
# Smaller chunks → more precise retrieval but lose surrounding context.
# Larger chunks → more context but noisier matches.
# Overlap ensures sentences that span a boundary aren't split mid-thought.

CHUNK_SIZE    = _int("CHUNK_SIZE",    700)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 64)


# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

TOP_K = _int("TOP_K", 8)

GRADING_ENABLED = os.getenv("GRADING_ENABLED", "true").lower() == "true"

_VALID_STRATEGIES = {"similarity", "mmr", "hybrid"}
SEARCH_STRATEGY   = os.getenv("SEARCH_STRATEGY", "hybrid")

# Weights for hybrid search — must sum to 1.0.
# Equal weights give BM25 (exact keyword) and semantic (meaning) equal say.
# This is important for precise technical terms like "SCD Type 2" or
# "materialized view" that semantic similarity alone fails to rank well.
BM25_WEIGHT     = _float("BM25_WEIGHT",     0.5)
SEMANTIC_WEIGHT = _float("SEMANTIC_WEIGHT", 0.5)


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
# After retrieval, chunks are re-scored via cosine similarity against the
# question using the same Ollama embedding model used for indexing, then the
# top-N results are kept.  No external models required.
# Disabled by default; set RERANKER_ENABLED=true to activate.

RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
RERANKER_TOP_N   = _int("RERANKER_TOP_N", 4)


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
# Before retrieval the LLM generates N alternative phrasings of the question.
# All variations are retrieved and merged (deduplicated by content).
# Disabled by default; set QUERY_EXPANSION_ENABLED=true to activate.

QUERY_EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "false").lower() == "true"
QUERY_EXPANSION_N       = _int("QUERY_EXPANSION_N", 3)


# ---------------------------------------------------------------------------
# Document preprocessing
# ---------------------------------------------------------------------------
# When enabled, raw documents are cleaned, structured, and enriched before
# chunking:
#   - Artifact removal: PDF hyphenation, ligatures, control characters
#   - Section markers: headers become === SECTION === splitter boundaries
#   - Context prefix: every chunk gets [Document | KB | Section] prepended
#     so retrieved chunks are self-contained (contextual retrieval)
# Disable only for debugging or if your documents are already well-structured.

PREPROCESS_ENABLED = os.getenv("PREPROCESS_ENABLED", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
# Hard timeout (seconds) on the full graph execution per request.
# Prevents a slow Ollama call from hanging the server indefinitely.
# Set 0 to disable.

PIPELINE_TIMEOUT_SECONDS = _int("PIPELINE_TIMEOUT_SECONDS", 120)

# ---------------------------------------------------------------------------
# Debug / verbosity
# ---------------------------------------------------------------------------
# When True, pipeline emits DEBUG-level log messages (retrieved chunks,
# grader decisions, routing). Set VERBOSE=false to see only INFO and above.

VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging for all project modules.

    Maps VERBOSE → DEBUG level (every chunk/grade decision visible) or
    INFO level (only high-level progress messages). Third-party libraries
    are capped at WARNING so they don't drown out pipeline output.
    """
    level = logging.DEBUG if VERBOSE else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    for name in ("rag", "ingest", "app", "server", "evaluate", "generate_testset", "rag.preprocessor"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = False
        if not lg.handlers:
            lg.addHandler(handler)

    for lib in ("chromadb", "httpx", "httpcore", "langchain_core", "openai", "urllib3", "instructor"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate() -> None:
    """Validate all config values and raise ValueError listing every problem.

    Collecting all errors before raising means the user sees every bad value
    at once instead of fixing one mistake and re-running to discover the next.

    Called automatically at module import so misconfiguration fails fast,
    before any pipeline code runs.
    """
    errors: list[str] = []

    # Required non-empty strings
    if not LLM_MODEL.strip():
        errors.append("LLM_MODEL must not be empty")
    if not EMBED_MODEL.strip():
        errors.append("EMBED_MODEL must not be empty")
    if not CHROMA_COLLECTION.strip():
        errors.append("CHROMA_COLLECTION must not be empty")
    if not OLLAMA_BASE_URL.startswith(("http://", "https://")):
        errors.append(
            f"OLLAMA_BASE_URL={OLLAMA_BASE_URL!r} must start with 'http://' or 'https://'"
        )

    # Numeric bounds
    if TOP_K < 1:
        errors.append(f"TOP_K={TOP_K} must be >= 1")
    if CHUNK_SIZE < 1:
        errors.append(f"CHUNK_SIZE={CHUNK_SIZE} must be >= 1")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})"
        )
    if not (0.0 <= BM25_WEIGHT <= 1.0):
        errors.append(f"BM25_WEIGHT={BM25_WEIGHT} must be in [0.0, 1.0]")
    if not (0.0 <= SEMANTIC_WEIGHT <= 1.0):
        errors.append(f"SEMANTIC_WEIGHT={SEMANTIC_WEIGHT} must be in [0.0, 1.0]")

    # Search strategy
    # Reranker bounds
    if PIPELINE_TIMEOUT_SECONDS < 0:
        errors.append(f"PIPELINE_TIMEOUT_SECONDS={PIPELINE_TIMEOUT_SECONDS} must be >= 0")
    if RERANKER_TOP_N < 1:
        errors.append(f"RERANKER_TOP_N={RERANKER_TOP_N} must be >= 1")

    # Query expansion bounds
    if QUERY_EXPANSION_N < 1:
        errors.append(f"QUERY_EXPANSION_N={QUERY_EXPANSION_N} must be >= 1")

    if SEARCH_STRATEGY not in _VALID_STRATEGIES:
        errors.append(
            f"SEARCH_STRATEGY={SEARCH_STRATEGY!r} is not valid. "
            f"Choose one of: {sorted(_VALID_STRATEGIES)}"
        )

    # Hybrid weights must sum to 1.0 (checked only when strategy is hybrid;
    # non-hybrid strategies ignore the weights, so a bad sum there is harmless)
    if SEARCH_STRATEGY == "hybrid" and not math.isclose(
        BM25_WEIGHT + SEMANTIC_WEIGHT, 1.0, abs_tol=1e-6
    ):
        errors.append(
            f"BM25_WEIGHT ({BM25_WEIGHT}) + SEMANTIC_WEIGHT ({SEMANTIC_WEIGHT})"
            f" = {BM25_WEIGHT + SEMANTIC_WEIGHT:.6f}, must sum to 1.0 for hybrid search"
        )

    if errors:
        bullet = "\n  • "
        raise ValueError(f"Config validation failed:{bullet}{bullet.join(errors)}")


validate()


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def check_ollama() -> None:
    """Verify Ollama is reachable before starting a pipeline run.

    Makes a lightweight HTTP request to the Ollama tags endpoint.
    Called by ingest.py and app.py at startup so the user gets a clear
    message rather than a cryptic connection error buried in a stack trace.

    Raises:
        RuntimeError: If Ollama is not reachable at OLLAMA_BASE_URL.
    """
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except (urllib.error.URLError, OSError):
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
            "Make sure Ollama is running:  ollama serve\n"
            f"Then pull the required models:\n"
            f"  ollama pull {LLM_MODEL}\n"
            f"  ollama pull {EMBED_MODEL}"
        )
