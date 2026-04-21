# rag/ — core RAG package
#
# Submodules:
#   state     — RAGState TypedDict shared across all nodes
#   retriever — ChromaDB connection and retrieval strategies
#   grader    — Self-RAG chunk relevance grading
#   chain     — LCEL chain (kept as a simpler alternative to the graph)
#   nodes     — LangGraph node functions
#   graph     — LangGraph agent assembly
