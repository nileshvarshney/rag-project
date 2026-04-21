"""
generate_testset.py — Synthetic test data generation using RAGAS TestsetGenerator.

RAGAS concept: instead of hand-writing golden Q&A pairs, the TestsetGenerator
uses an LLM to automatically create diverse, grounded questions directly from
your source documents.  The generator builds a knowledge graph from the chunks,
then synthesises questions of varying complexity:

  Simple        — Single-hop, answered from one chunk.
                  E.g. "What does SCD Type 2 require?"
  Reasoning     — Requires inferring from content within a chunk.
                  E.g. "Why does hybrid search outperform BM25 alone?"
  Multi-context — Needs information from multiple chunks.
                  E.g. "How does ETL relate to the Integration Layer?"

Each generated entry includes a reference answer grounded in the document,
making the output immediately usable as a golden dataset for test_golden_dataset.py.

Practical note on local LLMs
------------------------------
The generator makes many LLM calls (knowledge graph extraction, question
synthesis, answer generation). With a 7B model this takes ~10-15 minutes for
10 questions.  Review the output — smaller models occasionally produce vague
or overly generic questions.  Edit before using as ground truth.

Usage:
    python generate_testset.py                    # 10 questions, saved to tests/
    python generate_testset.py --size 5           # quick 5-question run
    python generate_testset.py --output my.json   # custom output path
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import config
from ingest import load_documents

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "tests/golden_dataset_synthetic.json"


def generate_testset(documents, test_size: int = 10) -> list[dict]:
    """
    Build a synthetic Q&A dataset from the provided documents.

    TestsetGenerator workflow:
      1. Extract entities and relationships into a KnowledgeGraph.
      2. Synthesise (question, reference_answer) pairs of varied complexity.
      3. Return a Testset object whose .to_list() gives dict-per-row.

    The returned dicts are reformatted to match golden_dataset_dwh.json so
    they can be passed directly to test_golden_dataset.py or evaluate.py.
    """
    import instructor
    import openai as oai
    from ragas.testset import TestsetGenerator
    from ragas.llms.base import InstructorLLM
    from ragas.embeddings.base import embedding_factory

    logger.info("Initialising LLM and embeddings via Ollama /v1 endpoint...")

    # RAGAS 0.4.x TestsetGenerator also expects InstructorLLM (structured output)
    # and a modern embedding provider.  Both are served by Ollama's /v1 API.
    llm_client = instructor.from_openai(
        oai.OpenAI(base_url=f"{config.OLLAMA_BASE_URL}/v1", api_key="ollama"),
        mode=instructor.Mode.JSON,
    )
    llm = InstructorLLM(client=llm_client, model=config.LLM_MODEL, provider="ollama")

    embed_client = oai.OpenAI(base_url=f"{config.OLLAMA_BASE_URL}/v1", api_key="ollama")
    embeddings = embedding_factory(
        "openai", model=config.EMBED_MODEL, client=embed_client, interface="modern"
    )

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    logger.info("Generating %d question(s) from %d document(s). "
                "Building knowledge graph then synthesising Q&A pairs "
                "(may take 10-20 min with a local 7B model)...",
                test_size, len(documents))

    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=test_size,
        raise_exceptions=False,  # log failures instead of crashing
    )

    rows = testset.to_list()

    formatted = []
    for i, row in enumerate(rows, 1):
        question = row.get("user_input") or row.get("question") or ""
        reference = row.get("reference") or row.get("ground_truth") or ""

        if not question.strip():
            continue

        formatted.append({
            "id":              f"syn_{i:03d}",
            "question":        question.strip(),
            "expected_answer": reference.strip(),
            "topic":           "synthetic",
            "difficulty":      "auto",
        })

    return formatted


def main() -> None:
    config.setup_logging()
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data from docs using RAGAS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--size", type=int, default=10,
        help="Number of questions to generate (default: 10)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print("=== Synthetic Test Data Generation (RAGAS) ===\n")

    try:
        config.check_ollama()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    documents = load_documents(config.DOCS_DIR)
    if not documents:
        logger.error("No documents found. Add .txt or .pdf files to the docs/ folder.")
        sys.exit(1)

    testset = generate_testset(documents, test_size=args.size)

    if not testset:
        logger.error("No questions were generated. The LLM may have failed to parse the documents.")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(testset, f, indent=2)

    logger.info("%d question(s) written to %s", len(testset), out_path)

    print("\nSample generated entries:")
    for entry in testset[:3]:
        print(f"\n  [{entry['id']}]")
        print(f"  Q: {entry['question'][:80]}")
        print(f"  A: {entry['expected_answer'][:80]}...")

    print(f"\nNext steps:")
    print(f"  Review {out_path} and edit any vague questions.")
    print(f"  Then run: python evaluate.py   (to score with RAGAS)")
    print(f"  Or run:   pytest tests/test_golden_dataset.py -m integration")
    print(f"            (using the synthetic set as the GOLDEN_DATASET_PATH)")


if __name__ == "__main__":
    main()
