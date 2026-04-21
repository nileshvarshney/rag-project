"""
ingest.py — Document ingestion pipeline.

RAG concept: before you can answer questions, you must:
  1. Load raw documents
  2. Split them into chunks (so embeddings are focused, not diluted)
  3. Embed each chunk (convert text → vector)
  4. Store vectors in ChromaDB (so you can retrieve them later)

Supported sources:
  Files   — .txt, .md (plain text), .pdf (page-aware), .docx (Word)
  URLs    — listed in {DOCS_DIR}/.url_sources.json; fetched at ingest time

Incremental ingest (default):
  Each chunk gets a stable ID = SHA-256(source_path + content).
  On re-run, only new or changed chunks are embedded; unchanged chunks are
  skipped; chunks whose source was deleted are removed from ChromaDB.
  This makes repeated ingest runs fast when most documents haven't changed.

Full rebuild (--full-rebuild):
  Drops the collection and re-embeds everything from scratch. Use this when
  you change CHUNK_SIZE, CHUNK_OVERLAP, or EMBED_MODEL, since those changes
  make existing embeddings stale regardless of content hashes.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import chromadb
import httpx
from chromadb.config import Settings
from chromadb.errors import ChromaError
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config
from rag.preprocessor import preprocess_documents, contextualise_chunks

logger = logging.getLogger(__name__)

# File that stores the list of URL sources for a docs directory / KB folder.
URL_SOURCES_FILENAME = ".url_sources.json"

# Name given to documents loaded directly from the root DOCS_DIR (not in any
# subfolder KB).  Using "general" makes the KB label human-readable in the UI.
GENERAL_KB = "general"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tag_kb(documents: list[Document], kb: str) -> list[Document]:
    """Inject 'kb' metadata into every document in-place and return the list."""
    for doc in documents:
        doc.metadata["kb"] = kb
    return documents


def load_documents(docs_dir: str, recursive: bool = True) -> list[Document]:
    """Load all supported file-based documents from a directory.

    Supported formats:
      .txt / .md  — UTF-8 plain text (markdown treated as text)
      .pdf        — page-aware extraction via pypdf
      .docx       — Word document text extraction via docx2txt

    Args:
        docs_dir:  Path to the directory to load from.
        recursive: If True use ``**/<ext>`` glob (descends subdirs).
                   If False use ``*.<ext>`` glob (root level only).

    Returns:
        List of LangChain Document objects (no 'kb' tag — callers add it).

    Raises:
        FileNotFoundError: If docs_dir does not exist.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(
            f"Docs directory not found: '{docs_dir}'\n"
            "Create it and add documents before ingesting."
        )

    prefix = "**/" if recursive else ""
    documents: list[Document] = []

    for glob, loader_cls, kwargs in [
        (f"{prefix}*.txt",  TextLoader,      {"encoding": "utf-8"}),
        (f"{prefix}*.md",   TextLoader,      {"encoding": "utf-8"}),
        (f"{prefix}*.pdf",  PyPDFLoader,     {}),
        (f"{prefix}*.docx", Docx2txtLoader,  {}),
    ]:
        loader = DirectoryLoader(
            docs_dir,
            glob=glob,
            loader_cls=loader_cls,
            loader_kwargs=kwargs if kwargs else None,
            show_progress=True,
        )
        documents.extend(loader.load())

    logger.info("Loaded %d file document(s) from '%s'", len(documents), docs_dir)
    return documents


def load_url_documents(docs_dir: str) -> list[Document]:
    """Fetch documents from URLs listed in .url_sources.json inside docs_dir.

    Returns Document objects (no 'kb' tag — callers add it).
    Failed URLs are skipped with a warning.
    """
    sources_file = Path(docs_dir) / URL_SOURCES_FILENAME
    if not sources_file.exists():
        return []

    try:
        entries: list[dict] = json.loads(sources_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read URL sources file: %s", e)
        return []

    documents: list[Document] = []
    for entry in entries:
        url = entry.get("url", "").strip()
        if not url:
            continue
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
            logger.info("Fetched URL '%s' → %d doc(s)", url, len(docs))
        except Exception as exc:
            logger.warning("Failed to fetch URL '%s': %s", url, exc)

    logger.info("Loaded %d URL document(s) from %d source(s)", len(documents), len(entries))
    return documents


# ---------------------------------------------------------------------------
# Knowledge-base aware loading
# ---------------------------------------------------------------------------

def list_knowledge_bases(docs_dir: str) -> list[str]:
    """Return the list of KB names present in docs_dir.

    KBs are the non-hidden subdirectories of docs_dir.  The root itself
    (for files placed directly in docs_dir) is represented as GENERAL_KB.
    """
    root = Path(docs_dir)
    kbs = [GENERAL_KB]
    if root.exists():
        for p in sorted(root.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                kbs.append(p.name)
    return kbs


def load_kb(kb_dir: str, kb_name: str) -> list[Document]:
    """Load all documents (files + URLs) for a single knowledge base.

    Files are loaded non-recursively from kb_dir (one level only) so that
    KB subfolders are not double-counted when load_all_sources iterates
    every KB separately.  URL sources are loaded from kb_dir/.url_sources.json.

    All returned documents are tagged with metadata['kb'] = kb_name.
    """
    # Non-recursive for KB root (GENERAL_KB loads root files only;
    # named KBs load their folder files without descending further).
    docs = load_documents(kb_dir, recursive=False)
    url_docs = load_url_documents(kb_dir)
    all_docs = docs + url_docs
    _tag_kb(all_docs, kb_name)
    logger.info("KB '%s': %d doc(s) loaded", kb_name, len(all_docs))
    return all_docs


def load_all_sources(docs_dir: str) -> list[Document]:
    """Load documents from ALL knowledge bases, each tagged with its KB name.

    - Root-level files → kb='general'
    - docs/hr/* → kb='hr'
    - docs/finance/* → kb='finance'
    etc.
    """
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Docs directory not found: '{docs_dir}'\n"
            "Create it and add documents before ingesting."
        )
    all_docs: list[Document] = []

    # Root files (non-recursive so subfolder KB files aren't double-loaded)
    root_docs = load_kb(docs_dir, GENERAL_KB)
    all_docs.extend(root_docs)

    # Each non-hidden subfolder = a named KB
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            kb_docs = load_kb(str(sub), sub.name)
            all_docs.extend(kb_docs)

    logger.info("Total: %d document(s) across all KBs", len(all_docs))
    return all_docs


def split_documents(
    documents: list[Document],
    contextualise: bool | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks.

    Uses RecursiveCharacterTextSplitter, which tries to split on paragraph,
    sentence, and word boundaries in order — preserving semantic units better
    than a naive fixed-size splitter.

    When preprocessing is enabled, '\\n\\n===' is the top-priority separator
    so that every === SECTION === marker injected by the preprocessor becomes
    a guaranteed chunk boundary.  No chunk will ever span two sections.

    After splitting, if contextualise is True (the default when preprocessing
    is enabled), each chunk is prefixed with a compact context line:
      [Document title | KB:name | §Section]
    so every retrieved chunk is self-contained (contextual retrieval).

    Args:
        documents:     Full documents returned by load_documents() and
                       optionally preprocessed by preprocess_documents().
        contextualise: Override for the contextualisation step.  Defaults to
                       config.PREPROCESS_ENABLED when not specified.

    Returns:
        Flat list of chunk Documents with enriched metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # "\n\n===" is tried first so section boundaries (=== HEADER ===) are
        # always honoured before falling back to paragraph / line / word splits.
        # This prevents content from two different sections being merged into
        # one chunk, which dilutes embeddings and breaks retrieval.
        separators=["\n\n===", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunk(s) (size=%d, overlap=%d)",
                len(chunks), config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    should_contextualise = contextualise if contextualise is not None else config.PREPROCESS_ENABLED
    if should_contextualise:
        chunks = contextualise_chunks(chunks)

    return chunks


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and persist them in ChromaDB.

    Deletes the existing collection before writing so that re-running this
    function never creates duplicate chunks.

    Args:
        chunks: Document chunks produced by split_documents().

    Returns:
        The populated Chroma vector store instance.
    """
    logger.info("Embedding %d chunk(s) with '%s' via Ollama...", len(chunks), config.EMBED_MODEL)

    embeddings = OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    chroma_client = chromadb.PersistentClient(
        path=config.CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete the existing collection so re-ingestion starts fresh.
    # Without this, every run appends to the old data and duplicates all chunks.
    try:
        chroma_client.delete_collection(config.CHROMA_COLLECTION)
        logger.info("Replaced existing collection '%s'", config.CHROMA_COLLECTION)
    except ValueError:
        pass  # collection doesn't exist yet — that's fine

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION,
        client=chroma_client,
    )

    logger.info("Stored %d chunk(s) in ChromaDB at '%s'", len(chunks), config.CHROMA_PATH)
    return vector_store


# ---------------------------------------------------------------------------
# Incremental ingest helpers
# ---------------------------------------------------------------------------

def _chunk_id(doc: Document) -> str:
    """Return a stable 32-hex-char ID for a chunk.

    ID = SHA-256(source_path + config_sig + page_content), truncated to 128 bits.

    The config signature includes CHUNK_SIZE, CHUNK_OVERLAP, and EMBED_MODEL so
    changing any of these automatically invalidates all chunk IDs and forces
    incremental ingest to re-embed — no need to remember --full-rebuild after
    a config change.

    128 bits gives a collision probability of < 1-in-10^28 for 10 million
    chunks — safe for any realistic document corpus.
    """
    source = doc.metadata.get("source", "")
    cfg_sig = f"{config.CHUNK_SIZE}:{config.CHUNK_OVERLAP}:{config.EMBED_MODEL}"
    raw = f"{source}::{cfg_sig}::{doc.page_content}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def incremental_ingest(
    chunks: list[Document], kb_filter: str = ""
) -> tuple[int, int, int]:
    """Embed and store only chunks that are new or changed.

    When kb_filter is non-empty, the comparison is scoped to that KB's
    existing chunks.  This prevents per-KB ingest runs from accidentally
    deleting chunks belonging to other KBs.

    Algorithm:
      1. Compute a hash-based ID for every incoming chunk.
      2. Fetch existing IDs from ChromaDB — either all IDs (kb_filter="") or
         only IDs where metadata.kb == kb_filter.
      3. Add chunks whose IDs are absent (new or changed content).
      4. Delete IDs that are in the scoped existing set but absent from the
         incoming set (file deleted or no longer part of this KB).

    Returns:
        Tuple of (added, skipped, deleted) chunk counts.
    """
    embeddings = OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    chroma_client = chromadb.PersistentClient(
        path=config.CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    vector_store = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        client=chroma_client,
        embedding_function=embeddings,
    )

    # Fetch only the IDs we care about — scoped to KB if requested.
    if kb_filter:
        existing_ids = set(
            vector_store._collection.get(
                include=[], where={"kb": {"$eq": kb_filter}}
            )["ids"]
        )
    else:
        existing_ids = set(vector_store._collection.get(include=[])["ids"])

    new_chunk_map: dict[str, Document] = {_chunk_id(doc): doc for doc in chunks}
    new_ids = set(new_chunk_map)

    to_add_ids    = new_ids - existing_ids
    to_delete_ids = existing_ids - new_ids
    skipped       = len(new_ids & existing_ids)

    if to_add_ids:
        docs_to_add = [new_chunk_map[cid] for cid in to_add_ids]
        logger.info("Embedding %d new/changed chunk(s) with '%s'...",
                    len(docs_to_add), config.EMBED_MODEL)
        vector_store.add_documents(docs_to_add, ids=list(to_add_ids))

    if to_delete_ids:
        logger.info("Deleting %d stale chunk(s) from ChromaDB...", len(to_delete_ids))
        vector_store.delete(ids=list(to_delete_ids))

    return len(to_add_ids), skipped, len(to_delete_ids)


def main() -> None:
    """Run the ingestion pipeline: load → preprocess → split → embed → store."""
    config.setup_logging()

    parser = argparse.ArgumentParser(
        description="Ingest documents into ChromaDB for RAG retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Pipeline: load → preprocess → split → contextualise → embed\n\n"
            "Default mode: incremental — only embeds new or changed chunks.\n"
            "Use --full-rebuild when CHUNK_SIZE, CHUNK_OVERLAP, or EMBED_MODEL changes,\n"
            "since those invalidate existing embeddings regardless of content.\n\n"
            "Preprocessing (enabled by default, PREPROCESS_ENABLED=true):\n"
            "  Cleans text artifacts, injects section markers as chunk boundaries,\n"
            "  and prefixes each chunk with [Document | KB | Section] context.\n"
            "  Use --no-preprocess to skip (useful for already-structured docs)."
        ),
    )
    parser.add_argument(
        "--full-rebuild", action="store_true",
        help="Drop the collection and re-embed everything from scratch.",
    )
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="Skip the preprocessing step (cleaning, section markers, contextualisation).",
    )
    args = parser.parse_args()

    do_preprocess = config.PREPROCESS_ENABLED and not args.no_preprocess
    mode_parts = ["full rebuild" if args.full_rebuild else "incremental"]
    if do_preprocess:
        mode_parts.append("with preprocessing")
    print(f"=== RAG Ingestion Pipeline ({', '.join(mode_parts)}) ===\n")

    try:
        config.check_ollama()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    try:
        documents = load_all_sources(config.DOCS_DIR)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    if not documents:
        logger.error(
            "No documents found. Add files to docs/ subfolders or register URL sources."
        )
        sys.exit(1)

    try:
        # Optional preprocessing: clean text, inject structure, enrich metadata
        if do_preprocess:
            logger.info(
                "Preprocessing %d document(s) — cleaning artifacts, "
                "injecting section markers, detecting document types…",
                len(documents),
            )
            documents = preprocess_documents(documents)

        chunks = split_documents(documents, contextualise=do_preprocess)

        if args.full_rebuild:
            build_vector_store(chunks)
            logger.info("Full rebuild complete — %d chunk(s) stored.", len(chunks))
        else:
            added, skipped, deleted = incremental_ingest(chunks)
            logger.info(
                "Incremental ingest complete — %d added, %d skipped (unchanged), %d deleted.",
                added, skipped, deleted,
            )

    except (ValueError, RuntimeError, httpx.HTTPError, ConnectionError, TimeoutError, ChromaError) as e:
        logger.error("Error during ingestion: %s", e)
        sys.exit(1)

    logger.info("You can now run app.py to query your documents.")


if __name__ == "__main__":
    main()
