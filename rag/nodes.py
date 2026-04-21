"""
nodes.py — Individual node functions for the LangGraph RAG agent.

Each function matches the LangGraph node contract:
  - Input:  the full RAGState dict
  - Output: a partial dict with only the keys this node modifies

LangGraph merges each return value into the running state, so nodes only
need to declare what they change — not copy everything they didn't touch.

Node order: route → [retrieve → grade] → generate → verify
"""

import asyncio
import hashlib
import logging
import time
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from .state import RAGState
from .retriever import retrieve_documents_normalised, clear_retrieval_cache
from .grader import async_grade_documents
from .reranker import async_rerank_documents
from .query_expander import async_expand_query
from .retry import async_retry_invoke

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared LLM instances
# ---------------------------------------------------------------------------
# temperature=0 for classification tasks (route, grade, verify) — determinism
# matters more than creativity.  generate_node uses 0.1 for natural answers.

_llm_classify = ChatOllama(
    model=config.LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0,
)

_llm_generate = ChatOllama(
    model=config.LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0.1,
)


# ---------------------------------------------------------------------------
# Node 1: route
# ---------------------------------------------------------------------------

_ROUTE_PROMPT = ChatPromptTemplate.from_template(
    """Decide whether the following question requires searching a knowledge base
(documents, specific facts, detailed information) or can be answered directly
as a general conversational question.

Question: {question}

Respond with a JSON object and nothing else:
  {{"route": "retrieval"}} — question needs document search
  {{"route": "direct"}}    — question can be answered without documents"""
)

_route_chain = _ROUTE_PROMPT | _llm_classify | StrOutputParser()


async def route_node(state: RAGState) -> dict[str, Any]:
    """Decide whether retrieval is needed for this question.

    Falls back to "retrieval" if the model produces unexpected output —
    it is always safe to try retrieval even when unnecessary.
    """
    rid = state["request_id"]
    t0 = time.perf_counter()
    logger.debug("[%s] Routing question: '%s'", rid, state["question"])

    raw = await async_retry_invoke(_route_chain, {"question": state["question"]})

    if '"direct"' in raw:
        decision = "direct"
    else:
        decision = "retrieval"

    logger.debug("[%s] route=%s  (%.0f ms)  model said: %s",
                 rid, decision, (time.perf_counter() - t0) * 1000, raw.strip()[:80])
    return {"route": decision, "documents": []}


# ---------------------------------------------------------------------------
# Node 2: retrieve
# ---------------------------------------------------------------------------

def _dedup_docs(docs):
    """Deduplicate by stable content hash (handles whitespace differences)."""
    seen: set[str] = set()
    unique = []
    for doc in docs:
        key = hashlib.md5(
            " ".join(doc.page_content.split()).encode(),
            usedforsecurity=False,
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


async def retrieve_node(state: RAGState) -> dict[str, Any]:
    """Fetch the top-k most relevant chunks from ChromaDB.

    When QUERY_EXPANSION_ENABLED the LLM generates N query variations; all
    are retrieved concurrently via asyncio.gather, then merged with
    deduplication by content hash (whitespace-insensitive).
    """
    rid = state["request_id"]
    question = state["question"]
    t0 = time.perf_counter()

    queries = await async_expand_query(question)

    k = config.TOP_K
    strategy = config.SEARCH_STRATEGY
    kb_filter = state.get("kb", "")
    loop = asyncio.get_event_loop()

    batches: list[tuple] = await asyncio.gather(
        *[loop.run_in_executor(None, retrieve_documents_normalised, q, k, strategy, kb_filter)
          for q in queries]
    )

    # Flatten and deduplicate by content hash (not string equality)
    all_docs = [doc for batch in batches for doc in batch]
    docs = _dedup_docs(all_docs)

    logger.debug(
        "[%s] Retrieved %d unique chunk(s) across %d quer%s  (%.0f ms)",
        rid, len(docs), len(queries), "y" if len(queries) == 1 else "ies",
        (time.perf_counter() - t0) * 1000,
    )
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        logger.debug("[%s] Chunk %d [%s]:\n%s", rid, i, source, doc.page_content[:200])

    docs = await async_rerank_documents(question, docs)
    return {"documents": docs}


# ---------------------------------------------------------------------------
# Node 3: grade
# ---------------------------------------------------------------------------

async def grade_node(state: RAGState) -> dict[str, Any]:
    """Filter retrieved chunks to only those relevant to the question.

    Uses async_grade_documents which fires all k LLM grading calls in
    parallel via asyncio.gather, reducing grading latency from O(k × llm)
    to O(llm) regardless of how many chunks were retrieved.

    When the grader filters everything out it returns [] so generate_node
    detects zero documents and falls back to direct generation rather than
    injecting irrelevant context.
    """
    rid = state["request_id"]
    t0 = time.perf_counter()
    logger.debug("[%s] Grading %d chunk(s) in parallel...", rid, len(state["documents"]))
    relevant = await async_grade_documents(state["question"], state["documents"])
    logger.debug(
        "[%s] grade: %d/%d passed  (%.0f ms)",
        rid, len(relevant), len(state["documents"]), (time.perf_counter() - t0) * 1000,
    )
    return {"documents": relevant}


# ---------------------------------------------------------------------------
# Node 4: generate
# ---------------------------------------------------------------------------

_GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
Use only the following context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)

_DIRECT_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the following question concisely.

Question: {question}

Answer:"""
)

_generate_rag_chain    = _GENERATE_PROMPT | _llm_generate | StrOutputParser()
_generate_direct_chain = _DIRECT_PROMPT   | _llm_generate | StrOutputParser()


async def generate_node(state: RAGState) -> dict[str, Any]:
    """Generate an answer using retrieved context (RAG) or directly (no docs).

    Two prompt paths:
      - RAG path (route == "retrieval" AND documents non-empty): injects
        retrieved chunks as context; instructs the model to answer only from
        that context.
      - Direct path: plain conversational prompt with no retrieval context.
        Triggered when route == "direct" OR when grading filtered all docs.
    """
    rid = state["request_id"]
    t0 = time.perf_counter()

    use_rag = state["route"] == "retrieval" and state["documents"]
    logger.debug("[%s] Generating answer (path=%s)...", rid, "rag" if use_rag else "direct")

    if use_rag:
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        answer = await async_retry_invoke(_generate_rag_chain, {
            "context":  context,
            "question": state["question"],
        })
    else:
        answer = await async_retry_invoke(_generate_direct_chain, {"question": state["question"]})

    logger.debug("[%s] generate done  (%.0f ms)", rid, (time.perf_counter() - t0) * 1000)
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node 5: verify
# ---------------------------------------------------------------------------

_VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are checking whether an answer is grounded in the provided context.
An answer is grounded if it only contains information present in the context
and does not introduce facts, claims, or details not found there.

Question: {question}

Context:
{context}

Answer: {answer}

Is the answer grounded in the context above?
Respond with a JSON object and nothing else:
  {{"grounded": "yes"}} — answer only uses information from the context
  {{"grounded": "no"}}  — answer contains information not in the context"""
)

_verify_chain = _VERIFY_PROMPT | _llm_classify | StrOutputParser()


async def verify_node(state: RAGState) -> dict[str, Any]:
    """Check whether the generated answer is grounded in the retrieved context.

    For direct-route answers there is no context to verify against, so we
    mark them as grounded by convention (the model answered from its own
    training knowledge, which is expected for conversational questions).

    For RAG answers, grounded=False flags potential hallucination — the UI
    shows a warning badge so the user knows to verify the answer.
    """
    rid = state["request_id"]
    t0 = time.perf_counter()
    logger.debug("[%s] Checking answer for grounding...", rid)

    if state["route"] == "direct" or not state["documents"]:
        logger.debug("[%s] Direct route — skipping grounding check.", rid)
        return {"grounded": True}

    context = "\n\n".join(doc.page_content for doc in state["documents"])
    raw = await async_retry_invoke(_verify_chain, {
        "question": state["question"],
        "context":  context,
        "answer":   state["answer"],
    })

    grounded = '"yes"' in raw.lower()
    status = "PASS" if grounded else "FAIL — answer may contain hallucinations"
    logger.debug(
        "[%s] grounding=%s  (%.0f ms)  model said: %s",
        rid, status, (time.perf_counter() - t0) * 1000, raw.strip()[:80],
    )

    return {"grounded": grounded}
