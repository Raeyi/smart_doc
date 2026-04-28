"""
Unit tests for SmartDocPlatform.search_documents method (main.py:227-268)
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_document(content: str, source: str = "test.txt") -> Document:
    """Helper to create a Document with metadata."""
    return Document(page_content=content, metadata={"source": source})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Minimal mock config object."""
    cfg = MagicMock()
    cfg.app.app_name = "TestApp"
    cfg.app.version = "0.0.1"
    cfg.app.log_file = None
    cfg.app.log_level = "WARNING"
    cfg.model.llm_provider = "deepseek"
    cfg.vector_store.k = 4
    cfg.vector_store.search_type = "similarity"
    return cfg


@pytest.fixture
def mock_retriever():
    """Mock SmartRetriever instance."""
    retriever = MagicMock()
    return retriever


@pytest.fixture
def platform(mock_config, mock_retriever):
    """Create a SmartDocPlatform instance with mocked dependencies."""
    with patch("main.config", mock_config), \
         patch("main.setup_logger", return_value=MagicMock()), \
         patch("main.get_document_loader", return_value=MagicMock()), \
         patch("main.get_vector_store_manager", return_value=MagicMock()), \
         patch("main.get_retriever", return_value=mock_retriever), \
         patch("main.init_llm", return_value=None), \
         patch("main.get_qa_system", return_value=MagicMock()):

        from main import SmartDocPlatform
        p = SmartDocPlatform()
        # Manually set the retriever to our mock
        p.retriever = mock_retriever
        p.llm = None
        p.vector_store_manager = MagicMock()
        return p


# ---------------------------------------------------------------------------
# Tests: search_documents – normal paths
# ---------------------------------------------------------------------------

class TestSearchDocumentsNormal:
    """Tests for normal execution paths of search_documents."""

    def test_similarity_search(self, platform, mock_retriever):
        """search_type='similarity' should call retriever.retrieve(query, 'vector', k)."""
        docs = [_make_document("hello world")]
        mock_retriever.retrieve.return_value = docs

        result = platform.search_documents("test query", search_type="similarity")

        mock_retriever.retrieve.assert_called_once_with("test query", "vector", None)
        assert result["success"] is True
        assert result["search_type"] == "similarity"
        assert result["result_count"] == 1
        assert result["results"][0]["rank"] == 1

    def test_mmr_search(self, platform, mock_retriever):
        """search_type='mmr' should call retriever.max_marginal_relevance_search."""
        docs = [_make_document("mmr result")]
        mock_retriever.max_marginal_relevance_search.return_value = docs

        result = platform.search_documents("test", search_type="mmr", k=5)

        mock_retriever.max_marginal_relevance_search.assert_called_once_with("test", 5)
        assert result["success"] is True
        assert result["search_type"] == "mmr"
        assert result["result_count"] == 1

    def test_hybrid_search(self, platform, mock_retriever):
        """search_type='hybrid' should call retriever.hybrid_retrieve."""
        docs = [_make_document("hybrid result")]
        mock_retriever.hybrid_retrieve.return_value = docs

        result = platform.search_documents("test", search_type="hybrid")

        mock_retriever.hybrid_retrieve.assert_called_once_with("test", k=None)
        assert result["success"] is True
        assert result["search_type"] == "hybrid"

    def test_unknown_search_type_defaults_to_similarity(self, platform, mock_retriever):
        """An unknown search_type should fall back to vector/similarity search."""
        docs = [_make_document("fallback")]
        mock_retriever.retrieve.return_value = docs

        result = platform.search_documents("test", search_type="unknown_type")

        mock_retriever.retrieve.assert_called_once_with("test", "vector", None)
        assert result["success"] is True
        assert result["search_type"] == "unknown_type"

    def test_custom_k_parameter(self, platform, mock_retriever):
        """The k parameter should be passed through to the retriever."""
        mock_retriever.retrieve.return_value = []

        platform.search_documents("test", search_type="similarity", k=10)

        mock_retriever.retrieve.assert_called_once_with("test", "vector", 10)


# ---------------------------------------------------------------------------
# Tests: search_documents – result formatting
# ---------------------------------------------------------------------------

class TestSearchDocumentsFormatting:
    """Tests for result formatting logic."""

    def test_content_truncated_when_over_500_chars(self, platform, mock_retriever):
        """Content longer than 500 chars should be truncated with '...'."""
        long_content = "A" * 600
        mock_retriever.retrieve.return_value = [_make_document(long_content)]

        result = platform.search_documents("test")

        assert result["results"][0]["content"].endswith("...")
        assert len(result["results"][0]["content"]) == 503  # 500 + "..."

    def test_content_not_truncated_when_under_500_chars(self, platform, mock_retriever):
        """Content 500 chars or less should not be truncated."""
        short_content = "short content"
        mock_retriever.retrieve.return_value = [_make_document(short_content)]

        result = platform.search_documents("test")

        assert result["results"][0]["content"] == short_content

    def test_content_exactly_500_chars_not_truncated(self, platform, mock_retriever):
        """Content exactly 500 chars should NOT be truncated (condition is > 500)."""
        content_500 = "A" * 500
        mock_retriever.retrieve.return_value = [_make_document(content_500)]

        result = platform.search_documents("test")

        assert not result["results"][0]["content"].endswith("...")
        assert len(result["results"][0]["content"]) == 500

    def test_multiple_results_ranking(self, platform, mock_retriever):
        """Multiple results should have sequential rank numbers."""
        docs = [_make_document(f"doc{i}") for i in range(5)]
        mock_retriever.retrieve.return_value = docs

        result = platform.search_documents("test")

        ranks = [r["rank"] for r in result["results"]]
        assert ranks == [1, 2, 3, 4, 5]

    def test_result_metadata_preserved(self, platform, mock_retriever):
        """Document metadata should be preserved in results."""
        doc = Document(
            page_content="content",
            metadata={"source": "file.pdf", "page": 3}
        )
        mock_retriever.retrieve.return_value = [doc]

        result = platform.search_documents("test")

        assert result["results"][0]["metadata"]["source"] == "file.pdf"
        assert result["results"][0]["metadata"]["page"] == 3

    def test_result_length_field(self, platform, mock_retriever):
        """The 'length' field should match the original content length."""
        content = "Hello, this is a test document."
        mock_retriever.retrieve.return_value = [_make_document(content)]

        result = platform.search_documents("test")

        assert result["results"][0]["length"] == len(content)

    def test_result_structure_fields(self, platform, mock_retriever):
        """Result dict should contain all expected top-level keys."""
        mock_retriever.retrieve.return_value = [_make_document("x")]

        result = platform.search_documents("test")

        assert "success" in result
        assert "query" in result
        assert "search_type" in result
        assert "result_count" in result
        assert "results" in result
        assert "timestamp" in result
        assert result["query"] == "test"
        assert result["result_count"] == 1

    def test_empty_results(self, platform, mock_retriever):
        """Empty search results should return success with result_count=0."""
        mock_retriever.retrieve.return_value = []

        result = platform.search_documents("test")

        assert result["success"] is True
        assert result["result_count"] == 0
        assert result["results"] == []


# ---------------------------------------------------------------------------
# Tests: search_documents – error handling
# ---------------------------------------------------------------------------

class TestSearchDocumentsErrors:
    """Tests for error and exception handling."""

    def test_retriever_exception_returns_error(self, platform, mock_retriever):
        """If retriever raises an exception, result should indicate failure."""
        mock_retriever.retrieve.side_effect = RuntimeError("DB connection lost")

        result = platform.search_documents("test")

        assert result["success"] is False
        assert "DB connection lost" in result["error"]

    def test_mmr_exception_returns_error(self, platform, mock_retriever):
        """Exception in MMR search should be caught and reported."""
        mock_retriever.max_marginal_relevance_search.side_effect = ValueError("bad param")

        result = platform.search_documents("test", search_type="mmr")

        assert result["success"] is False
        assert "bad param" in result["error"]

    def test_hybrid_exception_returns_error(self, platform, mock_retriever):
        """Exception in hybrid search should be caught and reported."""
        mock_retriever.hybrid_retrieve.side_effect = ConnectionError("network error")

        result = platform.search_documents("test", search_type="hybrid")

        assert result["success"] is False
        assert "network error" in result["error"]


# ---------------------------------------------------------------------------
# Tests: search_documents – boundary / edge cases
# ---------------------------------------------------------------------------

class TestSearchDocumentsEdgeCases:
    """Tests for boundary and edge cases."""

    def test_k_none_uses_default(self, platform, mock_retriever):
        """When k=None, retriever should receive None (it uses its own default)."""
        mock_retriever.retrieve.return_value = []

        platform.search_documents("test", k=None)

        mock_retriever.retrieve.assert_called_once_with("test", "vector", None)

    def test_k_zero(self, platform, mock_retriever):
        """k=0 is a valid (though unusual) input that should be passed through."""
        mock_retriever.retrieve.return_value = []

        platform.search_documents("test", k=0)

        mock_retriever.retrieve.assert_called_once_with("test", "vector", 0)

    def test_empty_query_string(self, platform, mock_retriever):
        """Empty query string should still execute (no validation on query)."""
        mock_retriever.retrieve.return_value = []

        result = platform.search_documents("")

        assert result["success"] is True
        assert result["query"] == ""

    def test_unicode_query(self, platform, mock_retriever):
        """Unicode queries (e.g., Chinese) should work correctly."""
        mock_retriever.retrieve.return_value = [_make_document("中文结果")]

        result = platform.search_documents("深度学习")

        assert result["success"] is True
        assert result["query"] == "深度学习"

    def test_timestamp_format(self, platform, mock_retriever):
        """Timestamp should be a valid ISO format string."""
        mock_retriever.retrieve.return_value = []

        result = platform.search_documents("test")

        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(result["timestamp"])
