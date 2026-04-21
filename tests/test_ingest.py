"""
test_ingest.py — Tests for the document ingestion pipeline (ingest.py).

Strategy:
  - load_documents: use tmp_path (real filesystem) — no mocking needed.
    File I/O is the thing being tested; faking it adds no value.
  - split_documents: pure function — no mocking needed.
  - build_vector_store: mock chromadb, OllamaEmbeddings, and Chroma so
    tests run without a running Ollama instance or ChromaDB on disk.
  - main: mock all three sub-functions to test the orchestration logic
    (argument validation, exit codes, call order) in isolation.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# load_documents
# ---------------------------------------------------------------------------

class TestLoadDocuments:
    """Tests for load_documents() — file discovery and parsing."""

    def test_raises_for_nonexistent_directory(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError with a helpful message."""
        from ingest import load_documents

        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Docs directory not found"):
            load_documents(str(missing))

    def test_returns_empty_list_for_empty_directory(self, tmp_path: Path) -> None:
        """An empty docs folder should yield zero documents, not an error."""
        from ingest import load_documents

        result = load_documents(str(tmp_path))
        assert result == []

    def test_loads_single_txt_file(self, tmp_path: Path) -> None:
        """A plain text file should be returned as a single Document."""
        from ingest import load_documents

        (tmp_path / "note.txt").write_text("Hello, RAG world!", encoding="utf-8")
        result = load_documents(str(tmp_path))

        assert len(result) == 1
        assert "Hello, RAG world!" in result[0].page_content

    def test_loads_multiple_txt_files(self, tmp_path: Path) -> None:
        """Each .txt file should produce at least one Document."""
        from ingest import load_documents

        (tmp_path / "a.txt").write_text("File A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("File B", encoding="utf-8")
        result = load_documents(str(tmp_path))

        assert len(result) == 2

    def test_ignores_non_supported_extensions(self, tmp_path: Path) -> None:
        """Files with unsupported extensions should be silently skipped."""
        from ingest import load_documents

        (tmp_path / "data.csv").write_text("col1,col2", encoding="utf-8")
        (tmp_path / "note.txt").write_text("valid", encoding="utf-8")
        result = load_documents(str(tmp_path))

        assert len(result) == 1

    def test_txt_document_carries_source_metadata(self, tmp_path: Path) -> None:
        """Loaded Documents should include a 'source' key in their metadata."""
        from ingest import load_documents

        (tmp_path / "doc.txt").write_text("content", encoding="utf-8")
        result = load_documents(str(tmp_path))

        assert "source" in result[0].metadata

    def test_loads_files_in_subdirectories(self, tmp_path: Path) -> None:
        """The glob pattern **/*.txt should recurse into subdirectories."""
        from ingest import load_documents

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested content", encoding="utf-8")
        result = load_documents(str(tmp_path))

        assert len(result) == 1
        assert "nested content" in result[0].page_content


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------

class TestSplitDocuments:
    """Tests for split_documents() — chunking behaviour."""

    def test_short_document_becomes_single_chunk(self) -> None:
        """A document shorter than CHUNK_SIZE should not be split."""
        from ingest import split_documents

        docs = [Document(page_content="Short text.", metadata={"source": "x"})]
        chunks = split_documents(docs)

        assert len(chunks) == 1

    def test_long_document_is_split_into_multiple_chunks(
        self, long_doc: Document
    ) -> None:
        """A document exceeding CHUNK_SIZE (512) must produce multiple chunks."""
        from ingest import split_documents

        chunks = split_documents([long_doc])

        assert len(chunks) > 1

    def test_all_chunks_are_documents(self, long_doc: Document) -> None:
        """Every element returned should be a LangChain Document."""
        from ingest import split_documents

        chunks = split_documents([long_doc])

        assert all(isinstance(c, Document) for c in chunks)

    def test_source_metadata_is_preserved_in_chunks(self, long_doc: Document) -> None:
        """Chunk metadata must include the source from the parent document."""
        from ingest import split_documents

        chunks = split_documents([long_doc])

        assert all(c.metadata.get("source") == "docs/long.txt" for c in chunks)

    def test_no_chunk_exceeds_chunk_size(self, long_doc: Document) -> None:
        """No chunk should be longer than CHUNK_SIZE characters."""
        import config
        from ingest import split_documents

        chunks = split_documents([long_doc])

        # Allow a small tolerance for separator characters kept at split points
        for chunk in chunks:
            assert len(chunk.page_content) <= config.CHUNK_SIZE + config.CHUNK_OVERLAP

    def test_multiple_documents_are_all_split(self, long_doc: Document) -> None:
        """When multiple documents are passed, all are chunked."""
        from ingest import split_documents

        chunks = split_documents([long_doc, long_doc])

        # Two long documents should yield more chunks than one
        single_chunks = split_documents([long_doc])
        assert len(chunks) > len(single_chunks)


# ---------------------------------------------------------------------------
# build_vector_store
# ---------------------------------------------------------------------------

class TestBuildVectorStore:
    """Tests for build_vector_store() — ChromaDB write path."""

    # Patch targets are the names as imported in ingest.py, not their
    # original module paths. This is the standard unittest.mock rule:
    # patch where the name is used, not where it is defined.

    def _patches(self):
        """Return a stack of patches shared across several tests."""
        return [
            patch("ingest.OllamaEmbeddings"),
            patch("ingest.Chroma.from_documents"),
        ]

    def test_deletes_existing_collection_before_writing(
        self, sample_docs: list[Document]
    ) -> None:
        """Re-ingestion must drop the old collection to prevent duplicates."""
        from ingest import build_vector_store

        mock_client = MagicMock()

        with patch("ingest.chromadb.PersistentClient", return_value=mock_client), \
             patch("ingest.OllamaEmbeddings"), \
             patch("ingest.Chroma.from_documents"):
            build_vector_store(sample_docs)

        mock_client.delete_collection.assert_called_once()

    def test_delete_exception_is_swallowed(
        self, sample_docs: list[Document]
    ) -> None:
        """If the collection doesn't exist yet, delete_collection raises —
        that exception must not propagate."""
        from ingest import build_vector_store

        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = Exception("collection not found")

        with patch("ingest.chromadb.PersistentClient", return_value=mock_client), \
             patch("ingest.OllamaEmbeddings"), \
             patch("ingest.Chroma.from_documents"):
            build_vector_store(sample_docs)  # must not raise

    def test_from_documents_receives_correct_chunks(
        self, sample_docs: list[Document]
    ) -> None:
        """The exact chunk list passed in must be forwarded to Chroma."""
        from ingest import build_vector_store

        with patch("ingest.chromadb.PersistentClient"), \
             patch("ingest.OllamaEmbeddings"), \
             patch("ingest.Chroma.from_documents") as mock_from_docs:
            build_vector_store(sample_docs)

        _, kwargs = mock_from_docs.call_args
        assert kwargs["documents"] == sample_docs

    def test_from_documents_uses_configured_collection_name(
        self, sample_docs: list[Document]
    ) -> None:
        """The ChromaDB collection name must come from config, not be hardcoded."""
        import config
        from ingest import build_vector_store

        with patch("ingest.chromadb.PersistentClient"), \
             patch("ingest.OllamaEmbeddings"), \
             patch("ingest.Chroma.from_documents") as mock_from_docs:
            build_vector_store(sample_docs)

        _, kwargs = mock_from_docs.call_args
        assert kwargs["collection_name"] == config.CHROMA_COLLECTION

    def test_embeddings_use_configured_model(
        self, sample_docs: list[Document]
    ) -> None:
        """OllamaEmbeddings must be initialised with the model from config."""
        import config
        from ingest import build_vector_store

        with patch("ingest.chromadb.PersistentClient"), \
             patch("ingest.OllamaEmbeddings") as mock_embed, \
             patch("ingest.Chroma.from_documents"):
            build_vector_store(sample_docs)

        mock_embed.assert_called_once_with(
            model=config.EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    """Tests for main() — orchestration, argument validation, exit codes."""

    def test_exits_1_when_ollama_unreachable(self) -> None:
        """If Ollama is down, main() should print an error and exit with code 1."""
        from ingest import main

        with patch("ingest.config.check_ollama",
                   side_effect=RuntimeError("cannot connect")), \
             pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_exits_1_when_docs_directory_missing(self) -> None:
        """A missing docs directory should produce exit code 1."""
        from ingest import main

        with patch("ingest.config.check_ollama"), \
             patch("ingest.load_documents",
                   side_effect=FileNotFoundError("no such dir")), \
             pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_exits_1_when_no_documents_found(self) -> None:
        """An empty docs folder should produce exit code 1, not a silent no-op."""
        from ingest import main

        with patch("ingest.config.check_ollama"), \
             patch("ingest.load_documents", return_value=[]), \
             pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_exits_1_on_ingestion_error(
        self, sample_docs: list[Document]
    ) -> None:
        """An exception during split or embed should produce exit code 1."""
        from ingest import main

        with patch("ingest.config.check_ollama"), \
             patch("ingest.load_documents", return_value=sample_docs), \
             patch("ingest.split_documents",
                   side_effect=RuntimeError("embedding failed")), \
             pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_success_calls_pipeline_in_order(
        self, sample_docs: list[Document]
    ) -> None:
        """On a clean run, all three pipeline steps must be called in order."""
        from ingest import main

        chunks = [Document(page_content="chunk")]

        # Attach child mocks to a shared manager so mock_calls records
        # every call across all four functions in a single ordered list.
        mock_manager = MagicMock()
        mock_manager.load.return_value = sample_docs
        mock_manager.split.return_value = chunks

        with patch("ingest.config.check_ollama", mock_manager.check_ollama), \
             patch("ingest.load_documents",       mock_manager.load), \
             patch("ingest.split_documents",      mock_manager.split), \
             patch("ingest.build_vector_store",   mock_manager.store):
            main()

        assert mock_manager.mock_calls == [
            call.check_ollama(),
            call.load(config_docs_dir()),
            call.split(sample_docs),
            call.store(chunks),
        ]

    def test_success_does_not_raise(
        self, sample_docs: list[Document]
    ) -> None:
        """A fully successful run must complete without raising or sys.exit."""
        from ingest import main

        with patch("ingest.config.check_ollama"), \
             patch("ingest.load_documents", return_value=sample_docs), \
             patch("ingest.split_documents", return_value=sample_docs), \
             patch("ingest.build_vector_store"):
            main()  # passes if no exception is raised


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def config_docs_dir() -> str:
    """Return the DOCS_DIR from config so tests stay in sync with it."""
    import config
    return config.DOCS_DIR
