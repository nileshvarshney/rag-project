"""
graph.py — Assembles the LangGraph RAG agent.

Graph shape:

  START
    │
    ▼
  route ──── "direct" ─────────────────────────────────────┐
    │                                                       │
  "retrieval"                                              │
    │                                                       │
    ▼                                                       │
  retrieve                                                  │
    │                                                       │
    ▼                                                       │
  grade ──── "no_docs" (all filtered) ────────────────────►│
    │                                                       │
  "has_docs"                                               │
    │                                                       ▼
    └────────────────────────────────────────────────► generate
                                                            │
                                                            ▼
                                                          verify
                                                            │
                                                            ▼
                                                           END

Adaptive fallback after grading:
  If the grader filters every retrieved chunk (documents=[]), the pipeline
  routes directly to the generate node with no context, which triggers the
  direct-answer prompt instead of injecting irrelevant context.
"""

from typing import Any

from langgraph.graph import StateGraph, END

from .state import RAGState
from .nodes import route_node, retrieve_node, grade_node, generate_node, verify_node


def _route_decision(state: RAGState) -> str:
    """Route after the router node: "retrieval" → retrieve, "direct" → generate."""
    return state["route"]


def _grade_decision(state: RAGState) -> str:
    """Route after grading: if no documents survived, go direct to generate."""
    return "has_docs" if state["documents"] else "no_docs"


def build_graph() -> Any:
    """Wire nodes and edges into a compiled LangGraph agent."""
    workflow = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────
    workflow.add_node("route",    route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade",    grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify",   verify_node)

    # ── Entry point ───────────────────────────────────────────────────────
    workflow.set_entry_point("route")

    # ── Conditional edge: route → retrieve OR generate ────────────────────
    workflow.add_conditional_edges(
        "route",
        _route_decision,
        {
            "retrieval": "retrieve",
            "direct":    "generate",
        },
    )

    # ── Unconditional edge: retrieve → grade ──────────────────────────────
    workflow.add_edge("retrieve", "grade")

    # ── Conditional edge: grade → generate (adaptive fallback) ───────────
    # When the grader filters all documents (no_docs), generate is called
    # with an empty documents list which triggers the direct-answer prompt.
    workflow.add_conditional_edges(
        "grade",
        _grade_decision,
        {
            "has_docs": "generate",
            "no_docs":  "generate",   # same destination — generate handles both
        },
    )

    # ── Unconditional edges ───────────────────────────────────────────────
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify",   END)

    return workflow.compile()


# Module-level compiled graph — import this in server.py and app.py
graph = build_graph()
