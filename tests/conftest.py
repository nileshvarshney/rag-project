"""
conftest.py — Shared pytest fixtures for the RAG project test suite.

pytest automatically loads this file before any test module, making all
fixtures defined here available without explicit imports.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_docs() -> list[Document]:
    """Two minimal Documents that stand in for real retrieved chunks."""
    return [
        Document(page_content="Paris is the capital of France.",
                 metadata={"source": "docs/geography.txt"}),
        Document(page_content="The Eiffel Tower is in Paris.",
                 metadata={"source": "docs/geography.txt"}),
    ]


@pytest.fixture
def long_doc() -> Document:
    """A Document whose content exceeds the default CHUNK_SIZE (512 chars)."""
    # 600 words × ~5 chars each ≈ 3000 chars — safely above any chunk size
    return Document(
        page_content=" ".join(["word"] * 600),
        metadata={"source": "docs/long.txt"},
    )


# ---------------------------------------------------------------------------
# LCEL-compatible fake components
# ---------------------------------------------------------------------------
# Using RunnableLambda instead of MagicMock means LCEL's | pipe operator
# composes correctly in chain tests — MagicMock doesn't support __or__.

@pytest.fixture
def fake_retriever(sample_docs: list[Document]) -> RunnableLambda:
    """A retriever that always returns sample_docs, regardless of the query."""
    return RunnableLambda(lambda _query: sample_docs)


@pytest.fixture
def fake_llm() -> RunnableLambda:
    """An LLM stub that returns a fixed AIMessage with no Ollama call."""
    return RunnableLambda(lambda _prompt: AIMessage(content="stubbed answer"))
