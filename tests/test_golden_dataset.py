"""
test_golden_dataset.py — Integration tests for the RAG chain against the
data-warehouse golden dataset (tests/golden_dataset_dwh.json).

Run with:
    pytest -m integration -v -s

Tests are skipped automatically when Ollama is unreachable.

Early-exit policy:
  Processing halts as soon as MAX_FAILURES entries fail.  The test then
  prints a detailed failure report and fails the suite.

Scoring:
  Both expected and actual answers are lower-cased, tokenised, stop-words
  removed.  Score = |expected_keywords ∩ actual_keywords| / |expected_keywords|.
  A score ≥ PASS_THRESHOLD counts as a pass.

Why RETRIEVAL_K > config.TOP_K:
  Integration tests retrieve more candidates than production so that
  lower-ranked but topically correct chunks are not missed.
"""

from __future__ import annotations

import json
import re
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import config
from rag.chain import build_rag_chain

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOLDEN_DATASET_PATH  = Path(__file__).parent / "golden_dataset_dwh.json"
PASS_THRESHOLD       = 0.25    # fraction of key terms from expected that must appear
MAX_FAILURES         = 3       # stop processing after this many failures
RETRIEVAL_K          = 10      # retrieve more chunks than production default (5)
                               # so specific topics surface even if ranked lower

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "it", "its", "that", "this", "which",
    "as", "if", "not", "no", "than", "then", "so", "also", "can", "each",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "any",
    "used", "use", "using", "about", "he", "she", "they", "we", "you",
    "their", "our", "your", "his", "her", "them", "us", "what",
}

_SEP = "─" * 68


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOP_WORDS}


def _score(expected: str, actual: str) -> float:
    exp_kw = _keywords(expected)
    if not exp_kw:
        return 1.0
    return len(exp_kw & _keywords(actual)) / len(exp_kw)


def _load_dataset() -> list[dict[str, Any]]:
    with GOLDEN_DATASET_PATH.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    id: str
    topic: str
    difficulty: str
    question: str
    expected: str
    actual: str
    score: float
    missing_keywords: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.missing_keywords = _keywords(self.expected) - _keywords(self.actual)

    def format(self, index: int) -> str:
        missing_list = sorted(self.missing_keywords)
        missing_str  = ", ".join(missing_list[:15])
        if len(missing_list) > 15:
            missing_str += f" … (+{len(missing_list) - 15} more)"

        actual_display = self.actual.strip() if self.actual.strip() else "(empty — chain returned nothing)"

        return "\n".join([
            f"Failure #{index}  [{self.id}]",
            f"  Topic      : {self.topic}  |  Difficulty: {self.difficulty}",
            f"  Score      : {self.score:.2f}  (threshold {PASS_THRESHOLD})",
            f"  Question   : {self.question}",
            f"  Expected   : {textwrap.shorten(self.expected, 220)}",
            f"  Actual     : {textwrap.shorten(actual_display, 220)}",
            f"  Missing kw : {missing_str or '(none — answer may be completely wrong)'}",
        ])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ollama_available() -> None:
    try:
        urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
    except (urllib.error.URLError, OSError):
        pytest.skip(
            f"Ollama not reachable at {config.OLLAMA_BASE_URL} — "
            "skipping integration tests"
        )


@pytest.fixture(scope="session")
def rag_chain(ollama_available: None):  # noqa: ARG001
    # Use higher k so niche topics (SCD, materialized views) surface in retrieval
    return build_rag_chain(k=RETRIEVAL_K)


# ---------------------------------------------------------------------------
# Main golden-dataset test
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_golden_dataset(rag_chain) -> None:
    """
    Run the RAG chain against every golden-dataset entry.

    Stops as soon as MAX_FAILURES entries score below PASS_THRESHOLD and
    prints a full failure report immediately — no need to dig through logs.
    """
    dataset   = _load_dataset()
    failures: list[FailureRecord] = []
    processed = 0

    print(f"\n{_SEP}")
    print(
        f"  Golden Dataset — {len(dataset)} entries | "
        f"k={RETRIEVAL_K} | threshold={PASS_THRESHOLD} | max_failures={MAX_FAILURES}"
    )
    print(_SEP)

    for entry in dataset:
        processed += 1
        actual: str = rag_chain.invoke(entry["question"])
        score  = _score(entry["expected_answer"], actual)
        passed = score >= PASS_THRESHOLD
        status = "PASS" if passed else "FAIL"
        actual_short = textwrap.shorten(actual.strip() or "(empty)", 80)

        print(
            f"  [{processed:>2}/{len(dataset)}] {status}  {entry['id']}"
            f"  score={score:.2f}  → {actual_short}"
        )

        if not passed:
            rec = FailureRecord(
                id         = entry["id"],
                topic      = entry["topic"],
                difficulty = entry["difficulty"],
                question   = entry["question"],
                expected   = entry["expected_answer"],
                actual     = actual,
                score      = score,
            )
            failures.append(rec)

            # Print the failure inline so it's visible immediately
            print(f"\n{_SEP}")
            print(rec.format(len(failures)))
            print(_SEP)

            if len(failures) >= MAX_FAILURES:
                remaining = len(dataset) - processed
                print(
                    f"\n  *** Reached {MAX_FAILURES} failures after {processed} "
                    f"entries — stopping early ({remaining} entries not processed) ***"
                )
                break

    total_passed = processed - len(failures)
    print(f"\n  Result: {total_passed}/{processed} processed entries passed  "
          f"(dataset total: {len(dataset)})")
    print(_SEP)

    if not failures:
        return

    # Also raise with the report so pytest -v shows it in the summary
    report_lines = [f"\n{len(failures)} failure(s) — processing stopped early.\n"]
    for i, f in enumerate(failures):
        report_lines.append(f.format(i + 1))
        report_lines.append("")   # blank line between failures

    pytest.fail("\n".join(report_lines), pytrace=False)
