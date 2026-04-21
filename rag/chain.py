"""
chain.py — Assembles the RAG chain using LangChain Expression Language (LCEL).

Pipeline stages:
  1. Retrieve  — fetch TOP_K chunks from the vector store
  2. Inspect   — print chunks so you can see what was retrieved
  3. Grade     — (Self-RAG) filter out chunks the LLM deems irrelevant
  4. Format    — concatenate surviving chunks into a context string
  5. Generate  — prompt the LLM and stream the answer

The grading step requires both the question and the docs at the same time,
so RunnableParallel keeps them as separate keys ("question", "docs") through
steps 1-3, then a RunnableLambda merges them into the {"context", "question"}
shape that RAG_PROMPT expects.
"""

import logging
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, Runnable

import config
from .retriever import retrieve_documents
from .grader import grade_documents
from .reranker import rerank_documents
from .query_expander import expand_query

logger = logging.getLogger(__name__)


RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant.
Use only the following context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""")


def print_chunks(docs: list[Document]) -> list[Document]:
    """Print retrieved chunks for inspection, then pass them through unchanged.

    This is a side-effect step inserted into the LCEL chain. It lets you
    observe exactly what context the LLM receives — the most important
    variable when debugging retrieval quality.

    Args:
        docs: Chunks returned by the retriever.

    Returns:
        The same docs list, unmodified, so the next chain step receives it.
    """
    logger.debug("Retrieved %d chunk(s)", len(docs))
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        logger.debug("Chunk %d [%s]:\n%s", i, source, doc.page_content)
    return docs


def format_docs(docs: list[Document]) -> str:
    """Concatenate document chunks into a single context string for the prompt.

    Chunks are joined with double newlines so the LLM can distinguish
    boundaries between separate passages.

    Args:
        docs: Graded and filtered document chunks.

    Returns:
        Single string of all chunk contents, separated by blank lines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def grade_and_format(inputs: dict[str, Any]) -> dict[str, str]:
    """Rerank, grade, then format chunks into context for the prompt.

    Sits between retrieval and generation in the LCEL chain. Receives the
    parallel outputs ``{"question": str, "docs": list[Document]}`` and
    returns ``{"question": str, "context": str}`` — the shape RAG_PROMPT
    expects.

    Stage order:
      1. Rerank  — cross-encoder rescores (question, chunk) pairs jointly
      2. Grade   — LLM filter removes irrelevant survivors
      3. Format  — surviving chunks concatenated into a context string

    Args:
        inputs: Dict with keys ``"question"`` (str) and ``"docs"`` (list).

    Returns:
        Dict with keys ``"question"`` (str) and ``"context"`` (str).
    """
    question: str = inputs["question"]
    docs: list[Document] = inputs["docs"]

    docs = rerank_documents(question, docs)
    relevant_docs = grade_documents(question, docs)
    return {
        "question": question,
        "context":  format_docs(relevant_docs),
    }


def build_rag_chain(k: int = config.TOP_K) -> Runnable:
    """Build and return the full LCEL RAG chain.

    Chain anatomy (read left to right with ``|`` as "then"):
      RunnableParallel → RunnableLambda(grade_and_format) → RAG_PROMPT → LLM → parser

    Args:
        k: Number of chunks to retrieve per query variation. With expansion
           enabled, each of the N+1 queries retrieves k chunks, so the grader
           may receive up to k*(N+1) unique candidates before filtering.

    Returns:
        A compiled LCEL Runnable that accepts a question string and returns
        an answer string. Supports ``.invoke()``, ``.stream()``, and
        ``.batch()``.
    """
    llm = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
    )

    strategy = config.SEARCH_STRATEGY

    def retrieve_expanded(question: str) -> list[Document]:
        """Retrieve chunks for the original question plus any expansions.

        Uses retrieve_documents so repeated identical queries are served from
        the LRU cache instead of hitting the vector store again.
        """
        queries = expand_query(question)
        seen: set[str] = set()
        docs: list[Document] = []
        for q in queries:
            for doc in retrieve_documents(q, k, strategy):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    docs.append(doc)
        return docs

    rag_chain: Runnable = (
        # Stage 1 & 2: expand query, retrieve + deduplicate, print chunks
        RunnableParallel({
            "docs":     RunnableLambda(retrieve_expanded) | RunnableLambda(print_chunks),
            "question": RunnablePassthrough(),
        })
        # Stage 3 & 4: rerank, grade, format into context string
        | RunnableLambda(grade_and_format)
        # Stage 5: generate answer
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain
