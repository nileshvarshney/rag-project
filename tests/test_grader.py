"""
test_grader.py — Unit tests for rag/grader.py.

Strategy:
  - _parse_grade: pure function — parametrize over every meaningful input
    variant without any mocking.

  - grade_documents: patch retry_invoke to control the model's response,
    and patch config.GRADING_ENABLED where the test needs a specific value.
    No Ollama calls are made.

Key edge cases:
  - All chunks graded irrelevant → fallback returns the original list (not []).
  - Grading exception → chunk is kept conservatively (never silently lost).
  - GRADING_ENABLED=False → docs returned without any LLM call.
"""

import pytest
import httpx
from unittest.mock import patch, call

import config
from langchain_core.documents import Document

from rag.grader import _parse_grade, grade_documents


# ---------------------------------------------------------------------------
# _parse_grade
# ---------------------------------------------------------------------------

class TestParseGrade:
    """Tests for _parse_grade() — model response string → bool."""

    @pytest.mark.parametrize("raw, expected", [
        ('{"relevant": "yes"}',            True),   # canonical yes
        ('{"relevant": "no"}',             False),  # canonical no
        ('{"relevant": "YES"}',            True),   # case-insensitive
        ('{"relevant": "Yes"}',            True),
        ('  {"relevant": "yes"}  ',        True),   # surrounding whitespace
        ('prefix text "yes" suffix',       True),   # yes anywhere in output
        ('{"relevant": "no", "extra": 1}', False),  # no with extra keys
        ('',                               False),  # empty string
        ('some garbage output',            False),  # no yes/no marker
        ('{"relevant": "maybe"}',          False),  # ambiguous → conservative False
        ('{"relevant": "no yes"}',         False),  # "yes" token not present with quotes
    ])
    def test_parse_variants(self, raw: str, expected: bool) -> None:
        assert _parse_grade(raw) == expected


# ---------------------------------------------------------------------------
# grade_documents
# ---------------------------------------------------------------------------

class TestGradeDocuments:
    """Tests for grade_documents() — filter + fallback logic."""

    # ------------------------------------------------------------------
    # Short-circuit paths (no LLM calls)
    # ------------------------------------------------------------------

    def test_returns_docs_unchanged_when_grading_disabled(
        self, sample_docs: list[Document]
    ) -> None:
        """GRADING_ENABLED=False must bypass all LLM calls and return docs as-is."""
        with patch.object(config, "GRADING_ENABLED", False), \
             patch("rag.grader.retry_invoke") as mock_invoke:
            result = grade_documents("any question", sample_docs)

        assert result is sample_docs
        mock_invoke.assert_not_called()

    def test_returns_empty_list_for_empty_docs(self) -> None:
        """Empty input must produce empty output without touching the LLM."""
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke") as mock_invoke:
            result = grade_documents("any question", [])

        assert result == []
        mock_invoke.assert_not_called()

    # ------------------------------------------------------------------
    # Filtering behaviour
    # ------------------------------------------------------------------

    def test_all_relevant_chunks_are_kept(
        self, sample_docs: list[Document]
    ) -> None:
        """When every chunk is graded relevant, the full list is returned."""
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke", return_value='{"relevant": "yes"}'):
            result = grade_documents("question", sample_docs)

        assert result == sample_docs

    def test_mixed_grading_returns_only_relevant_subset(
        self, sample_docs: list[Document]
    ) -> None:
        """Only chunks the model marked relevant should survive."""
        # sample_docs has two entries; mark first relevant, second not
        responses = ['{"relevant": "yes"}', '{"relevant": "no"}']
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke", side_effect=responses):
            result = grade_documents("question", sample_docs)

        assert result == [sample_docs[0]]

    def test_all_irrelevant_falls_back_to_original_docs(
        self, sample_docs: list[Document]
    ) -> None:
        """If every chunk is filtered, the function falls back to the full
        original list rather than passing an empty context to the LLM."""
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke", return_value='{"relevant": "no"}'):
            result = grade_documents("question", sample_docs)

        assert result is sample_docs

    def test_order_of_relevant_chunks_is_preserved(self) -> None:
        """Surviving chunks must appear in the same order as the input."""
        docs = [
            Document(page_content=f"chunk {i}", metadata={}) for i in range(4)
        ]
        # chunks 0, 2 are relevant; 1, 3 are not
        responses = [
            '{"relevant": "yes"}',
            '{"relevant": "no"}',
            '{"relevant": "yes"}',
            '{"relevant": "no"}',
        ]
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke", side_effect=responses):
            result = grade_documents("q", docs)

        assert result == [docs[0], docs[2]]

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_grading_exception_keeps_chunk(
        self, sample_docs: list[Document]
    ) -> None:
        """If retry_invoke raises after exhaustion, the chunk must be kept
        (conservative fallback — never silently discard a chunk on error)."""
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke",
                   side_effect=httpx.ConnectError("ollama down")):
            result = grade_documents("question", sample_docs[:1])

        assert result == sample_docs[:1]

    def test_timeout_error_keeps_chunk(self) -> None:
        doc = Document(page_content="some text", metadata={})
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke", side_effect=TimeoutError("timed out")):
            result = grade_documents("question", [doc])

        assert result == [doc]

    def test_partial_error_keeps_failed_chunk_alongside_others(
        self, sample_docs: list[Document]
    ) -> None:
        """If only one chunk errors, the others are still graded normally."""
        # First chunk errors → kept. Second chunk is relevant → kept.
        responses = [
            httpx.ConnectError("down"),
            '{"relevant": "yes"}',
        ]
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke", side_effect=responses):
            result = grade_documents("question", sample_docs)

        assert result == sample_docs  # both kept (one via error, one via yes)

    # ------------------------------------------------------------------
    # Argument forwarding
    # ------------------------------------------------------------------

    def test_question_forwarded_to_retry_invoke(
        self, sample_docs: list[Document]
    ) -> None:
        """retry_invoke must receive the exact question string."""
        question = "What is a data warehouse?"
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke",
                   return_value='{"relevant": "yes"}') as mock_invoke:
            grade_documents(question, sample_docs[:1])

        inputs = mock_invoke.call_args[0][1]  # second positional arg (the dict)
        assert inputs["question"] == question

    def test_chunk_content_forwarded_to_retry_invoke(
        self, sample_docs: list[Document]
    ) -> None:
        """retry_invoke must receive the chunk's page_content, not its metadata."""
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke",
                   return_value='{"relevant": "yes"}') as mock_invoke:
            grade_documents("q", sample_docs[:1])

        inputs = mock_invoke.call_args[0][1]
        assert inputs["chunk"] == sample_docs[0].page_content

    def test_retry_invoke_called_once_per_doc(
        self, sample_docs: list[Document]
    ) -> None:
        """retry_invoke must be called exactly len(docs) times — no more, no less."""
        with patch.object(config, "GRADING_ENABLED", True), \
             patch("rag.grader.retry_invoke",
                   return_value='{"relevant": "yes"}') as mock_invoke:
            grade_documents("q", sample_docs)

        assert mock_invoke.call_count == len(sample_docs)
