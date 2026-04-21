"""
evaluate.py — RAGAS-based evaluation of the RAG pipeline.

RAGAS concept: our keyword-overlap golden dataset test checks whether the
right keywords appear in the answer. RAGAS goes deeper — it uses an LLM as a
judge to measure pipeline quality along four orthogonal dimensions:

  faithfulness       Is every claim in the answer supported by the retrieved
                     context? A score of 1 means no hallucinations; 0 means
                     the answer contradicts or ignores the context entirely.

  answer_relevancy   Does the answer actually address the question? Catches
                     on-topic but unhelpful answers (e.g. tangential facts).
                     Uses embeddings to compare answer ↔ question direction.

  context_precision  Are the *most* relevant chunks ranked first in the
                     retrieved list? A pipeline that retrieves the right
                     documents but buries them behind noise scores low here.
                     Needs a reference (ground-truth) answer.

  context_recall     Did retrieval capture *all* the information needed to
                     answer the question? High recall = the context contains
                     every claim present in the reference answer.
                     Needs a reference (ground-truth) answer.

All metrics are 0–1 (higher is better). They complement keyword overlap:
keyword overlap catches missing facts; RAGAS catches hallucinations and
retrieval order problems.

LLM-as-judge note: metrics that call the judge LLM (faithfulness,
context_precision, context_recall) are only as reliable as that LLM.
A 7B local model works but gives noisier scores than GPT-4. Use trends
over multiple runs rather than treating individual scores as ground truth.

Usage:
    python evaluate.py                     # evaluate all 20 golden entries
    python evaluate.py --samples 5         # quick 5-question sanity check
    python evaluate.py --no-context        # skip context_precision/recall
    python evaluate.py --history           # print run history and exit
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import config
from rag.retriever import get_retriever
from rag.grader import grade_documents
from rag.chain import RAG_PROMPT, format_docs
from rag.retry import retry_invoke

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = "tests/golden_dataset_dwh.json"
METRICS_LOG_PATH    = "evaluation/metrics_log.json"


# ---------------------------------------------------------------------------
# Step 1: Collect RAG pipeline outputs
# ---------------------------------------------------------------------------

def collect_rag_outputs(dataset: list[dict], k: int = config.TOP_K) -> list[dict]:
    """
    Run the full RAG pipeline (retrieve → grade → generate) on each entry
    and record the inputs and outputs that RAGAS needs:

      user_input         — the question
      response           — the generated answer
      retrieved_contexts — the *graded* chunks actually used (list of str)
      reference          — the expected answer from the golden dataset

    Why not just call build_rag_chain().invoke()?  The LCEL chain returns
    only the final string; RAGAS also needs the intermediate context list.
    We therefore call retrieval and generation separately so we can capture
    both outputs in one pass.
    """
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    retriever = get_retriever(k=k)

    # Reuse the same RAG_PROMPT from chain.py; build a minimal generation chain.
    llm = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
    )
    gen_chain = RAG_PROMPT | llm | StrOutputParser()

    outputs = []
    for i, entry in enumerate(dataset, 1):
        question     = entry["question"]
        ground_truth = entry["expected_answer"]

        logger.info("[%2d/%d] %s  %s...", i, len(dataset), entry["id"], question[:65])

        raw_docs    = retriever.invoke(question)
        graded_docs = grade_documents(question, raw_docs)
        used_docs   = graded_docs if graded_docs else raw_docs

        context = format_docs(used_docs)
        answer  = retry_invoke(gen_chain, {"context": context, "question": question})

        outputs.append({
            "user_input":         question,
            "response":           answer,
            "retrieved_contexts": [doc.page_content for doc in used_docs],
            "reference":          ground_truth,
        })

    return outputs


# ---------------------------------------------------------------------------
# Step 2: Run RAGAS metrics
# ---------------------------------------------------------------------------

def run_ragas(samples: list[dict], use_context_metrics: bool = True) -> dict[str, float]:
    """
    Evaluate the collected outputs with RAGAS.

    RAGAS 0.4.x changed to use the `instructor` library for structured LLM
    output.  Metrics now require an InstructorLLM (not LangchainLLMWrapper).
    Ollama exposes an OpenAI-compatible REST API at /v1, so we point the
    instructor + openai clients there — no API key or internet needed.

    Embeddings likewise need the modern ragas embedding_factory (not the
    deprecated LangchainEmbeddingsWrapper).  We again reuse Ollama's /v1
    endpoint with the same nomic-embed-text model used by the RAG pipeline.
    """
    import instructor
    import openai as oai
    import warnings
    import instructor
    import openai as oai
    from ragas import evaluate, EvaluationDataset
    from ragas.llms.base import InstructorLLM
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import OllamaEmbeddings

    # ragas.metrics (not .collections): these extend Metric, which evaluate()
    # requires.  ragas.metrics.collections uses BaseMetric — incompatible.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    logger.info("Initialising judge LLM and embeddings via Ollama /v1 endpoint...")

    # RAGAS 0.4.x metrics require InstructorLLM (structured JSON output via the
    # instructor library).  Ollama's OpenAI-compatible /v1 endpoint works here.
    llm_client = instructor.from_openai(
        oai.OpenAI(base_url=f"{config.OLLAMA_BASE_URL}/v1", api_key="ollama"),
        mode=instructor.Mode.JSON,
    )
    judge_llm = InstructorLLM(
        client=llm_client,
        model=config.LLM_MODEL,
        provider="ollama",
    )

    # LangchainEmbeddingsWrapper (deprecated but functional) provides the
    # embed_query() method that AnswerRelevancy needs internally.
    # The modern ragas OpenAIEmbeddings lacks embed_query and causes an error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        embed_model = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
        )

    metrics = [Faithfulness(llm=judge_llm), AnswerRelevancy(llm=judge_llm)]
    if use_context_metrics:
        metrics += [ContextPrecision(llm=judge_llm), ContextRecall(llm=judge_llm)]

    dataset = EvaluationDataset.from_list(samples)

    logger.info("Scoring %d sample(s) across %d metric(s). Each metric makes one or more LLM calls per sample.",
                len(samples), len(metrics))

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        embeddings=embed_model,   # explicit: prevents evaluate() from auto-creating
                                  # OpenAI embeddings (which requires OPENAI_API_KEY)
        show_progress=True,
        raise_exceptions=False,   # return NaN for failed samples instead of crashing
    )

    # EvaluationResult.to_pandas() gives one row per sample; take column means.
    df = result.to_pandas()
    metric_cols = [m.name for m in metrics]
    return {col: float(df[col].mean()) for col in metric_cols if col in df.columns}


# ---------------------------------------------------------------------------
# Step 3: Performance tracking
# ---------------------------------------------------------------------------

def append_metrics_log(scores: dict[str, float], num_samples: int) -> None:
    """
    Append one run's results to evaluation/metrics_log.json.

    Each entry records the timestamp, pipeline config, number of samples, and
    all metric scores. Over time this builds a trend chart you can review with
    --history to see whether config changes improved or degraded quality.
    """
    log_path = Path(METRICS_LOG_PATH)
    log_path.parent.mkdir(exist_ok=True)

    history: list[dict] = []
    if log_path.exists():
        with open(log_path) as f:
            history = json.load(f)

    history.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "llm_model":       config.LLM_MODEL,
            "embed_model":     config.EMBED_MODEL,
            "chunk_size":      config.CHUNK_SIZE,
            "chunk_overlap":   config.CHUNK_OVERLAP,
            "top_k":           config.TOP_K,
            "search_strategy": config.SEARCH_STRATEGY,
        },
        "num_samples":   num_samples,
        "ragas_metrics": scores,
    })

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Results saved to %s", METRICS_LOG_PATH)


def print_history(log_path: str = METRICS_LOG_PATH) -> None:
    """Print all recorded runs as a comparison table."""
    path = Path(log_path)
    if not path.exists():
        print("No metrics history found. Run evaluate.py to create one.")
        return

    with open(path) as f:
        history = json.load(f)

    w = 80
    print(f"\n{'─' * w}")
    print(f"  Evaluation History  ({len(history)} run(s) in {log_path})")
    print(f"{'─' * w}")
    header = f"  {'Timestamp':<20} {'N':>4}  {'Faithful':>9} {'AnsRel':>8} {'CtxPrec':>8} {'CtxRec':>8}"
    print(header)
    print(f"  {'─' * (w - 2)}")
    for run in history:
        m  = run["ragas_metrics"]
        ts = run["timestamp"]
        n  = run["num_samples"]

        def _fmt(key: str) -> str:
            v = m.get(key)
            return f"{v:.3f}" if v is not None and v == v else "  n/a"

        print(f"  {ts:<20} {n:>4}  "
              f"{_fmt('faithfulness'):>9} "
              f"{_fmt('answer_relevancy'):>8} "
              f"{_fmt('context_precision'):>8} "
              f"{_fmt('context_recall'):>8}")
    print(f"{'─' * w}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    config.setup_logging()
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline with RAGAS metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of golden entries to evaluate (default: all 20)",
    )
    parser.add_argument(
        "--no-context", action="store_true",
        help="Skip context_precision and context_recall (faster, no ground-truth needed)",
    )
    parser.add_argument(
        "--history", action="store_true",
        help="Print performance history and exit",
    )
    args = parser.parse_args()

    if args.history:
        print_history()
        return

    print("=== RAG Pipeline — RAGAS Evaluation ===\n")

    try:
        config.check_ollama()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    with open(GOLDEN_DATASET_PATH) as f:
        dataset = json.load(f)

    if args.samples:
        dataset = dataset[: args.samples]

    # --- Step 1: Collect ---
    logger.info("Step 1/3  Collecting RAG outputs for %d question(s)...", len(dataset))
    outputs = collect_rag_outputs(dataset)

    # --- Step 2: RAGAS ---
    logger.info("Step 2/3  Running RAGAS metrics...")
    scores = run_ragas(outputs, use_context_metrics=not args.no_context)

    # --- Step 3: Track & display ---
    logger.info("Step 3/3  Recording results...")
    append_metrics_log(scores, len(dataset))

    # Pretty-print current run results
    bar_width = 24
    print(f"\n{'─' * 55}")
    print("  RAGAS Scores (0–1, higher is better)")
    print(f"{'─' * 55}")
    for metric, score in scores.items():
        bar  = "█" * round(score * bar_width)
        rest = "░" * (bar_width - len(bar))
        print(f"  {metric:<22}  {score:.3f}  {bar}{rest}")
    print(f"{'─' * 55}")

    print_history()


if __name__ == "__main__":
    main()
