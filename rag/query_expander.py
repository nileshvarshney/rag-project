"""
query_expander.py — LLM-based query expansion before retrieval.

RAG concept: a user's vague or ambiguous question often misses relevant chunks
because the exact wording doesn't appear in the source documents.  By asking
the LLM to rephrase the query using synonyms, related terms, or different
angles, we cast a wider net and retrieve chunks that a single query would miss.

The caller always receives [original] + variations so the original query is
never dropped and result sets are merged with deduplication.

Enabled via config.QUERY_EXPANSION_ENABLED (default: false) to avoid the
extra LLM round-trip when the feature is not needed.

Caching: expansions are cached by (normalised_question, n) so the same
question never triggers a second LLM call.  Cache holds up to 512 entries;
oldest entries are evicted when full (simple LRU via OrderedDict).
"""

import json
import logging
import re
from collections import OrderedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from .retry import retry_invoke, async_retry_invoke

logger = logging.getLogger(__name__)

_EXPAND_PROMPT = ChatPromptTemplate.from_template(
    """Generate {n} alternative ways to ask the following question.
Each variation must preserve the original intent but use different wording,
synonyms, or a different angle — helping retrieve documents that may not
match the original phrasing.

Original question: {question}

Respond with a JSON object and nothing else:
{{"variations": ["variation 1", "variation 2", ...]}}"""
)

_llm = ChatOllama(
    model=config.LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0.4,
)

_expand_chain = _EXPAND_PROMPT | _llm | StrOutputParser()

# LRU cache: (normalised_question, n) → list[str]
_CACHE_SIZE = 512
_cache: OrderedDict[tuple[str, int], list[str]] = OrderedDict()


def _cache_get(key: tuple) -> list[str] | None:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key: tuple, value: list[str]) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)


def _normalise(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _parse_variations(raw: str, n: int) -> list[str]:
    """Extract the variations list from the model's JSON response.

    Falls back to [] on any parse error so the caller can still use the
    original query alone — expansion failure is never fatal.
    """
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        variations = data.get("variations", [])
        if isinstance(variations, list):
            return [str(v).strip() for v in variations if str(v).strip()][:n]
    except (json.JSONDecodeError, AttributeError):
        pass
    return []


def expand_query(question: str) -> list[str]:
    """Return the original question plus up to QUERY_EXPANSION_N variations.

    Results are cached by (normalised question, n) — repeated calls with the
    same question are free after the first invocation.

    Args:
        question: The user's original question string.

    Returns:
        List starting with the original question, followed by LLM-generated
        variations.  If expansion is disabled or the LLM fails, returns a
        single-element list containing only the original question.
    """
    if not config.QUERY_EXPANSION_ENABLED:
        return [question]

    n = config.QUERY_EXPANSION_N
    key = (_normalise(question), n)

    cached = _cache_get(key)
    if cached is not None:
        logger.debug("Query expansion cache HIT for %r", question[:60])
        return cached

    try:
        raw = retry_invoke(_expand_chain, {"question": question, "n": n})
        variations = _parse_variations(raw, n)
    except Exception:  # noqa: BLE001 — expansion failure must never crash retrieval
        logger.warning("Query expansion failed; using original query only.")
        return [question]

    queries = [question] + [v for v in variations if _normalise(v) != _normalise(question)]
    _cache_put(key, queries)
    logger.debug(
        "Query expansion: 1 → %d queries\n  %s",
        len(queries),
        "\n  ".join(f"{i}. {q}" for i, q in enumerate(queries, 1)),
    )
    return queries


async def async_expand_query(question: str) -> list[str]:
    """Async version of expand_query — uses async_retry_invoke.

    Shares the same cache as the sync version so a prior sync call warms
    the cache for subsequent async calls and vice-versa.
    """
    if not config.QUERY_EXPANSION_ENABLED:
        return [question]

    n = config.QUERY_EXPANSION_N
    key = (_normalise(question), n)

    cached = _cache_get(key)
    if cached is not None:
        logger.debug("Query expansion cache HIT for %r", question[:60])
        return cached

    try:
        raw = await async_retry_invoke(_expand_chain, {"question": question, "n": n})
        variations = _parse_variations(raw, n)
    except Exception:  # noqa: BLE001
        logger.warning("Query expansion failed; using original query only.")
        return [question]

    queries = [question] + [v for v in variations if _normalise(v) != _normalise(question)]
    _cache_put(key, queries)
    logger.debug(
        "Query expansion: 1 → %d queries\n  %s",
        len(queries),
        "\n  ".join(f"{i}. {q}" for i, q in enumerate(queries, 1)),
    )
    return queries
