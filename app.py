"""
app.py — Interactive CLI for the LangGraph RAG agent.

Usage:
    python app.py           # uses TOP_K from config.py
    python app.py --k 2
    python app.py --k 6

LangGraph concept: .stream() with stream_mode="updates" yields one dict
per node as it completes: {"node_name": {keys_it_changed}}.
This lets us print a live trace of the graph executing — you can see
each node fire in order rather than waiting for the full pipeline to finish.
"""

import argparse
import sys
from typing import Any
from uuid import uuid4

import httpx

import config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with a single attribute:
          k (int): number of chunks to retrieve per query.
    """
    parser = argparse.ArgumentParser(description="LangGraph RAG agent CLI")
    parser.add_argument(
        "--k",
        type=int,
        default=config.TOP_K,
        help=f"Chunks to retrieve (default: {config.TOP_K})",
    )
    return parser.parse_args()


def run(question: str) -> None:
    """Run the LangGraph agent for a single question and print the trace.

    Streams node-by-node updates so each pipeline step is visible as it
    completes rather than waiting for the full run. Catches all exceptions
    so a single failed query does not kill the interactive loop.

    Args:
        question: The user's natural-language question.
    """
    from rag.graph import graph  # noqa: PLC0415

    request_id = str(uuid4())
    print(f"\n{'═' * 60}")
    print(f"  Question:   {question}")
    print(f"  Request ID: {request_id}")
    print(f"{'═' * 60}")

    final_state: dict[str, Any] = {}

    try:
        for step in graph.stream(
            {"question": question, "request_id": request_id},
            stream_mode="updates",
        ):
            node_name: str = next(iter(step))
            updates: dict[str, Any] = step[node_name]
            final_state.update(updates)
            _print_step_summary(node_name, updates)
    except FileNotFoundError as e:
        # chroma_db/ missing — ingest.py hasn't been run yet
        print(f"\nError: {e}")
        return
    except (httpx.HTTPError, ConnectionError, TimeoutError, RuntimeError, ValueError) as e:
        print(f"\nPipeline error: {e}")
        return

    print(f"\n{'─' * 60}")
    print(f"  Answer")
    print(f"{'─' * 60}")
    print(final_state.get("answer", "(no answer produced)"))

    if not final_state.get("grounded", True):
        print("\n  [!] Grounding check FAILED — answer may not be fully supported by your documents.")

    print()


def _print_step_summary(node: str, updates: dict[str, Any]) -> None:
    """Print a one-line status line when a graph node completes.

    Args:
        node:    Name of the node that just finished.
        updates: Partial state dict returned by that node.
    """
    icons: dict[str, str] = {
        "route":    "→",
        "retrieve": "↓",
        "grade":    "✓",
        "generate": "✎",
        "verify":   "✔",
    }
    icon = icons.get(node, "•")

    if node == "route":
        detail = f"route={updates.get('route')}"
    elif node == "retrieve":
        detail = f"{len(updates.get('documents', []))} chunk(s) retrieved"
    elif node == "grade":
        detail = f"{len(updates.get('documents', []))} chunk(s) after grading"
    elif node == "generate":
        answer_preview = (updates.get("answer") or "")[:60].replace("\n", " ")
        detail = f'"{answer_preview}..."'
    elif node == "verify":
        detail = "grounded=yes" if updates.get("grounded") else "grounded=NO"
    else:
        detail = str(updates)[:80]

    print(f"  [{icon} {node}] {detail}")


def main() -> None:
    """Entry point: parse args, check Ollama, then start the question loop."""
    config.setup_logging()
    args = parse_args()

    try:
        config.check_ollama()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Patch TOP_K before the graph imports the retriever
    config.TOP_K = args.k

    print(f"=== LangGraph RAG Agent | k={args.k} | strategy={config.SEARCH_STRATEGY} ===")
    print("Type 'quit' to exit.\n")

    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            run(question)
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
