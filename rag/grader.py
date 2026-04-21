"""
grader.py — Self-RAG chunk relevance grader.

Self-RAG concept: before passing retrieved chunks to the LLM for generation,
we ask the LLM to judge whether each chunk is actually relevant to the
question. Irrelevant chunks are filtered out, giving the LLM a cleaner,
more focused context window.

Why this helps:
  Vector similarity finds chunks that are topically close to a query,
  but "close" is not the same as "useful". A chunk about a related topic
  can score highly in cosine space yet contain nothing that helps answer
  the specific question. Grading catches those false positives.

Grade caching:
  Grades are cached by (question_hash, content_hash) so the same
  chunk is never re-graded for the same question.  Cache holds up to
  4 096 entries; eviction is oldest-first (LRU via OrderedDict).

Fallback when all chunks are filtered:
  Rather than injecting irrelevant context (the old behaviour), we return
  an empty list so generate_node detects zero documents and falls back to
  a direct LLM answer.  This is the honest path — if nothing retrieved is
  relevant, the model should say so from its own knowledge.

Trade-off:
  Grading adds one LLM call per chunk before the final generation call.
  At k=5 that's 5 extra round-trips to Ollama. Disable with
  GRADING_ENABLED=False when you care more about latency than precision.
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict

import httpx
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import config
from .retry import retry_invoke, async_retry_invoke

logger = logging.getLogger(__name__)

GRADE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading whether a document chunk contains information useful for answering a question.

Question: {question}

Document chunk:
{chunk}

Does this chunk contain information that helps answer the question?
Respond with a JSON object and nothing else: {{"relevant": "yes"}} or {{"relevant": "no"}}"""
)

_grader_chain = (
    GRADE_PROMPT
    | ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,  # determinism: same chunk → same grade every time
    )
    | StrOutputParser()
)

# Grade cache: (question_md5, content_md5) → bool
_CACHE_SIZE = 4096
_grade_cache: OrderedDict[tuple[str, str], bool] = OrderedDict()


def _grade_key(question: str, content: str) -> tuple[str, str]:
    q = hashlib.md5(question.encode(), usedforsecurity=False).hexdigest()
    c = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
    return q, c


def _cache_get(key: tuple) -> bool | None:
    if key in _grade_cache:
        _grade_cache.move_to_end(key)
        return _grade_cache[key]
    return None


def _cache_put(key: tuple, value: bool) -> None:
    _grade_cache[key] = value
    _grade_cache.move_to_end(key)
    if len(_grade_cache) > _CACHE_SIZE:
        _grade_cache.popitem(last=False)


def _parse_grade(raw: str) -> bool:
    """Extract yes/no from the model's response.

    We look for '"yes"' anywhere in the output rather than doing strict
    JSON parsing — small models sometimes add whitespace or comments
    around the JSON object.
    """
    return '"yes"' in raw.lower()


def grade_documents(question: str, docs: list[Document]) -> list[Document]:
    """Grade each chunk for relevance to question. Return only relevant ones.

    Falls back to direct generation (empty list) when all chunks are filtered,
    rather than injecting irrelevant context into the LLM.

    Args:
        question: The user's original query.
        docs:     Chunks retrieved from the vector store.

    Returns:
        Filtered list containing only chunks the grader marked relevant.
        Returns empty list if all chunks fail grading (triggers direct path).
    """
    if not config.GRADING_ENABLED or not docs:
        return docs

    relevant: list[Document] = []
    cache_hits = 0

    logger.debug("Grading %d chunk(s) for relevance...", len(docs))

    for i, doc in enumerate(docs, 1):
        key = _grade_key(question, doc.page_content)
        cached = _cache_get(key)
        if cached is not None:
            cache_hits += 1
            if cached:
                relevant.append(doc)
            logger.debug("Chunk %d: %s (cached)", i, "relevant" if cached else "filtered")
            continue

        try:
            raw = retry_invoke(_grader_chain, {
                "question": question,
                "chunk": doc.page_content,
            })
            passed = _parse_grade(raw)
            label = "relevant" if passed else "filtered"
            logger.debug("Chunk %d: %s  (model said: %s)", i, label, raw.strip()[:60])
        except (httpx.HTTPError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            logger.debug("Chunk %d: grading error (%s) — keeping chunk", i, e)
            passed = True

        _cache_put(key, passed)
        if passed:
            relevant.append(doc)

    if cache_hits:
        logger.debug("Grade cache: %d/%d hits", cache_hits, len(docs))

    if not relevant:
        logger.info(
            "No chunks passed grading — falling back to direct generation "
            "(model will answer from its own knowledge)."
        )
        return []

    logger.debug("%d/%d chunk(s) passed grading.", len(relevant), len(docs))
    return relevant


async def async_grade_documents(question: str, docs: list[Document]) -> list[Document]:
    """Async version of grade_documents — grades all chunks concurrently.

    Replaces the sequential loop with asyncio.gather so all k LLM grading
    calls are in-flight simultaneously. Latency drops from O(k × llm_time)
    to O(llm_time) — a k× speedup at the cost of k× concurrent connections
    to Ollama (which handles this fine for local single-model serving).

    Grade results are cached — chunks whose (question, content) pair has
    been seen before skip the LLM call entirely.
    """
    if not config.GRADING_ENABLED or not docs:
        return docs

    logger.debug("Grading %d chunk(s) concurrently...", len(docs))

    async def _grade_one(i: int, doc: Document) -> bool:
        key = _grade_key(question, doc.page_content)
        cached = _cache_get(key)
        if cached is not None:
            logger.debug("Chunk %d: %s (cached)", i, "relevant" if cached else "filtered")
            return cached

        try:
            raw = await async_retry_invoke(_grader_chain, {
                "question": question,
                "chunk": doc.page_content,
            })
            passed = _parse_grade(raw)
            label = "relevant" if passed else "filtered"
            logger.debug("Chunk %d: %s  (model said: %s)", i, label, raw.strip()[:60])
        except (httpx.HTTPError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            logger.debug("Chunk %d: grading error (%s) — keeping chunk", i, e)
            passed = True

        _cache_put(key, passed)
        return passed

    results: list[bool] = await asyncio.gather(
        *[_grade_one(i, doc) for i, doc in enumerate(docs, 1)]
    )

    relevant = [doc for doc, passed in zip(docs, results) if passed]

    if not relevant:
        logger.info(
            "No chunks passed grading — falling back to direct generation "
            "(model will answer from its own knowledge)."
        )
        return []

    logger.debug("%d/%d chunk(s) passed grading.", len(relevant), len(docs))
    return relevant
