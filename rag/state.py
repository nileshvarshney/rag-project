"""
state.py — Shared state definition for the LangGraph RAG agent.

LangGraph concept: a StateGraph passes a single state dict between nodes.
Each node receives the full current state and returns a partial dict
containing only the keys it modified. LangGraph merges these updates
back into the state automatically.

Using TypedDict (rather than a plain dict) gives you type hints and
makes the data contract between nodes explicit at a glance.
"""

from typing import TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):
    # Unique ID for this request — generated at the entry point (app.py /
    # server.py) so every log line across route → retrieve → grade →
    # generate → verify can be correlated by a single string.
    request_id: str

    # Set by the user at invocation time.
    question: str

    # Optional knowledge-base filter.  Empty string = search all KBs.
    # When set, retrieval is scoped to chunks whose 'kb' metadata matches.
    kb: str

    # Set by route_node: "retrieval" or "direct".
    # Controls which branch the graph takes after routing.
    route: str

    # Set by retrieve_node, then filtered in place by grade_node.
    # Empty list when route == "direct" (no retrieval needed).
    documents: list[Document]

    # Set by generate_node: the LLM's answer.
    answer: str

    # Set by verify_node: True if the answer is grounded in the
    # retrieved context, False if the model may have hallucinated.
    grounded: bool
