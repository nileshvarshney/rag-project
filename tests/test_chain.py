"""
test_chain.py — Tests for the LCEL RAG chain (rag/chain.py).

Strategy:
  - print_chunks / format_docs / grade_and_format: pure or near-pure
    functions — test inputs and outputs directly, no mocking needed
    beyond grade_documents (which calls Ollama).

  - build_rag_chain: the assembled LCEL chain. We mock two things:
      1. get_retriever  → RunnableLambda (LCEL-compatible fake)
      2. ChatOllama     → RunnableLambda (LCEL-compatible fake)
    Using RunnableLambda instead of MagicMock is critical: LCEL's | pipe
    operator calls __or__ on each step, which MagicMock doesn't support.
    RunnableLambda is a real Runnable, so pipes compose correctly and the
    full chain can be invoked end-to-end without touching any model.
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from rag.chain import print_chunks, format_docs, grade_and_format, build_rag_chain


# ---------------------------------------------------------------------------
# print_chunks
# ---------------------------------------------------------------------------

class TestPrintChunks:
    """Tests for print_chunks() — side-effect inspection step in the chain."""

    def test_returns_docs_unchanged(
        self, sample_docs: list[Document], capsys: pytest.CaptureFixture
    ) -> None:
        """The function must be a pass-through: same list, same objects."""
        result = print_chunks(sample_docs)
        assert result == sample_docs

    def test_returns_empty_list_unchanged(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        result = print_chunks([])
        assert result == []

    def test_prints_chunk_count(
        self, sample_docs: list[Document], capsys: pytest.CaptureFixture
    ) -> None:
        """The count shown in stdout must match the number of docs passed."""
        print_chunks(sample_docs)
        stdout = capsys.readouterr().out
        assert f"{len(sample_docs)} chunk(s)" in stdout

    def test_prints_source_metadata(
        self, sample_docs: list[Document], capsys: pytest.CaptureFixture
    ) -> None:
        """Source file path should appear in printed output."""
        print_chunks(sample_docs)
        stdout = capsys.readouterr().out
        assert "docs/geography.txt" in stdout

    def test_prints_chunk_content(
        self, sample_docs: list[Document], capsys: pytest.CaptureFixture
    ) -> None:
        """Each chunk's page_content should appear in the printed output."""
        print_chunks(sample_docs)
        stdout = capsys.readouterr().out
        for doc in sample_docs:
            assert doc.page_content in stdout

    def test_unknown_source_falls_back_to_label(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Docs with no 'source' metadata key should show 'unknown'."""
        docs = [Document(page_content="no source", metadata={})]
        print_chunks(docs)
        stdout = capsys.readouterr().out
        assert "unknown" in stdout


# ---------------------------------------------------------------------------
# format_docs
# ---------------------------------------------------------------------------

class TestFormatDocs:
    """Tests for format_docs() — chunk list → single context string."""

    def test_joins_multiple_docs_with_double_newline(
        self, sample_docs: list[Document]
    ) -> None:
        """Chunks must be separated by exactly two newlines."""
        result = format_docs(sample_docs)
        expected = "Paris is the capital of France.\n\nThe Eiffel Tower is in Paris."
        assert result == expected

    def test_empty_list_returns_empty_string(self) -> None:
        assert format_docs([]) == ""

    def test_single_doc_has_no_separator(self) -> None:
        docs = [Document(page_content="only chunk")]
        assert format_docs(docs) == "only chunk"

    def test_preserves_whitespace_inside_chunks(self) -> None:
        docs = [Document(page_content="line one\nline two")]
        assert format_docs(docs) == "line one\nline two"

    def test_separator_is_exactly_two_newlines(
        self, sample_docs: list[Document]
    ) -> None:
        """Guard against single-newline or triple-newline joining."""
        result = format_docs(sample_docs)
        # The only double-newline in the result should be the separator
        parts = result.split("\n\n")
        assert len(parts) == len(sample_docs)


# ---------------------------------------------------------------------------
# grade_and_format
# ---------------------------------------------------------------------------

class TestGradeAndFormat:
    """Tests for grade_and_format() — the grading + formatting bridge step."""

    def test_returns_question_key(
        self, sample_docs: list[Document]
    ) -> None:
        with patch("rag.chain.grade_documents", return_value=sample_docs):
            result = grade_and_format({"question": "test?", "docs": sample_docs})
        assert "question" in result

    def test_returns_context_key(
        self, sample_docs: list[Document]
    ) -> None:
        with patch("rag.chain.grade_documents", return_value=sample_docs):
            result = grade_and_format({"question": "test?", "docs": sample_docs})
        assert "context" in result

    def test_passes_question_through_unchanged(
        self, sample_docs: list[Document]
    ) -> None:
        question = "What is the capital of France?"
        with patch("rag.chain.grade_documents", return_value=sample_docs):
            result = grade_and_format({"question": question, "docs": sample_docs})
        assert result["question"] == question

    def test_context_equals_formatted_docs(
        self, sample_docs: list[Document]
    ) -> None:
        """The 'context' value must equal format_docs applied to graded docs."""
        with patch("rag.chain.grade_documents", return_value=sample_docs):
            result = grade_and_format({"question": "q", "docs": sample_docs})
        assert result["context"] == format_docs(sample_docs)

    def test_calls_grade_documents_with_correct_args(
        self, sample_docs: list[Document]
    ) -> None:
        """grade_documents must receive the question and the raw docs list."""
        with patch("rag.chain.grade_documents", return_value=sample_docs) as mock_grade:
            grade_and_format({"question": "my question", "docs": sample_docs})
        mock_grade.assert_called_once_with("my question", sample_docs)

    def test_context_is_empty_when_all_docs_filtered(
        self, sample_docs: list[Document]
    ) -> None:
        """If grading removes every chunk, context should be an empty string."""
        with patch("rag.chain.grade_documents", return_value=[]):
            result = grade_and_format({"question": "q", "docs": sample_docs})
        assert result["context"] == ""

    def test_context_reflects_filtered_subset(
        self, sample_docs: list[Document]
    ) -> None:
        """Context should contain only docs that passed grading."""
        kept = sample_docs[:1]  # only the first chunk survives grading
        with patch("rag.chain.grade_documents", return_value=kept):
            result = grade_and_format({"question": "q", "docs": sample_docs})
        assert result["context"] == kept[0].page_content


# ---------------------------------------------------------------------------
# build_rag_chain
# ---------------------------------------------------------------------------

class TestBuildRagChain:
    """Tests for build_rag_chain() — the assembled LCEL pipeline.

    All Ollama I/O is eliminated by replacing ChatOllama and get_retriever
    with RunnableLambda instances that the LCEL pipe operator can compose.
    """

    def test_returns_a_runnable(
        self,
        fake_retriever: RunnableLambda,
        fake_llm: RunnableLambda,
        sample_docs: list[Document],
    ) -> None:
        """build_rag_chain must return a LangChain Runnable."""
        with patch("rag.chain.get_retriever", return_value=fake_retriever), \
             patch("rag.chain.ChatOllama", return_value=fake_llm), \
             patch("rag.chain.grade_documents", return_value=sample_docs):
            chain = build_rag_chain(k=2)
        assert isinstance(chain, Runnable)

    def test_chain_invoke_returns_string(
        self,
        fake_retriever: RunnableLambda,
        fake_llm: RunnableLambda,
        sample_docs: list[Document],
    ) -> None:
        """The assembled chain must return a plain string when invoked."""
        with patch("rag.chain.get_retriever", return_value=fake_retriever), \
             patch("rag.chain.ChatOllama", return_value=fake_llm), \
             patch("rag.chain.grade_documents", return_value=sample_docs):
            chain = build_rag_chain(k=2)
            result = chain.invoke("What is the capital of France?")
        assert isinstance(result, str)

    def test_chain_returns_llm_output(
        self,
        fake_retriever: RunnableLambda,
        sample_docs: list[Document],
    ) -> None:
        """The string returned by the chain must come from the LLM output."""
        expected = "The capital of France is Paris."
        specific_llm = RunnableLambda(
            lambda _: AIMessage(content=expected)
        )
        with patch("rag.chain.get_retriever", return_value=fake_retriever), \
             patch("rag.chain.ChatOllama", return_value=specific_llm), \
             patch("rag.chain.grade_documents", return_value=sample_docs):
            chain = build_rag_chain(k=2)
            result = chain.invoke("Capital of France?")
        assert result == expected

    def test_uses_k_parameter_for_retriever(
        self,
        fake_llm: RunnableLambda,
        sample_docs: list[Document],
    ) -> None:
        """The k argument must be forwarded to get_retriever, not ignored."""
        with patch("rag.chain.get_retriever") as mock_get_retriever, \
             patch("rag.chain.ChatOllama", return_value=fake_llm):
            mock_get_retriever.return_value = RunnableLambda(lambda _: sample_docs)
            build_rag_chain(k=3)
        mock_get_retriever.assert_called_once_with(k=3)

    def test_chain_invokes_grade_documents(
        self,
        fake_retriever: RunnableLambda,
        fake_llm: RunnableLambda,
        sample_docs: list[Document],
    ) -> None:
        """grade_documents must be called as part of every chain invocation."""
        with patch("rag.chain.get_retriever", return_value=fake_retriever), \
             patch("rag.chain.ChatOllama", return_value=fake_llm), \
             patch("rag.chain.grade_documents",
                   return_value=sample_docs) as mock_grade:
            chain = build_rag_chain(k=2)
            chain.invoke("test question")
        mock_grade.assert_called_once()

    def test_chain_passes_question_to_grade_documents(
        self,
        fake_retriever: RunnableLambda,
        fake_llm: RunnableLambda,
        sample_docs: list[Document],
    ) -> None:
        """grade_documents must receive the user's original question."""
        question = "What is the Eiffel Tower?"
        with patch("rag.chain.get_retriever", return_value=fake_retriever), \
             patch("rag.chain.ChatOllama", return_value=fake_llm), \
             patch("rag.chain.grade_documents",
                   return_value=sample_docs) as mock_grade:
            chain = build_rag_chain(k=2)
            chain.invoke(question)
        actual_question = mock_grade.call_args[0][0]
        assert actual_question == question

    def test_chain_works_when_grading_filters_all_docs(
        self,
        fake_retriever: RunnableLambda,
        fake_llm: RunnableLambda,
    ) -> None:
        """If grading rejects every chunk, the chain must not raise —
        grade_and_format falls back to the direct prompt path."""
        with patch("rag.chain.get_retriever", return_value=fake_retriever), \
             patch("rag.chain.ChatOllama", return_value=fake_llm), \
             patch("rag.chain.grade_documents", return_value=[]):
            chain = build_rag_chain(k=2)
            result = chain.invoke("any question")
        assert isinstance(result, str)
