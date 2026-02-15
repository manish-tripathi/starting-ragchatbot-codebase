"""Shared fixtures and helpers for backend tests."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# We need to be able to import from the backend package and from tests/
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# Mock heavy dependencies before they are imported by application modules
import unittest.mock as _mock
_chromadb_mock = _mock.MagicMock()
_st_mock = _mock.MagicMock()
sys.modules.setdefault("chromadb", _chromadb_mock)
sys.modules.setdefault("chromadb.config", _mock.MagicMock())
sys.modules.setdefault("chromadb.utils", _mock.MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", _mock.MagicMock())
sys.modules.setdefault("sentence_transformers", _st_mock)
sys.modules.setdefault("anthropic", _mock.MagicMock())

from vector_store import SearchResults


# ---------------------------------------------------------------------------
# SearchResults fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_search_results():
    """SearchResults with sample documents and metadata."""
    return SearchResults(
        documents=[
            "Machine learning is a subset of AI.",
            "Neural networks are inspired by the brain.",
        ],
        metadata=[
            {"course_title": "Intro to AI", "lesson_number": 1},
            {"course_title": "Intro to AI", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def mock_empty_results():
    """Empty SearchResults (no error)."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_error_results():
    """SearchResults carrying an error message."""
    return SearchResults.empty("Search error: connection refused")


# ---------------------------------------------------------------------------
# VectorStore mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store(mock_search_results):
    """MagicMock of VectorStore with configurable search() return."""
    store = MagicMock()
    store.search.return_value = mock_search_results
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    return store


# ---------------------------------------------------------------------------
# Anthropic client mock
# ---------------------------------------------------------------------------

from helpers import make_text_response, make_tool_use_response


@pytest.fixture
def mock_anthropic_client():
    """Mock of anthropic.Anthropic whose messages.create() is configurable."""
    client = MagicMock()
    # Default: return a simple text response
    client.messages.create.return_value = make_text_response("Hello!")
    return client


# ---------------------------------------------------------------------------
# ToolManager mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tool_manager():
    """MagicMock of ToolManager with configurable execute_tool() return."""
    tm = MagicMock()
    tm.execute_tool.return_value = "Tool executed successfully."
    tm.get_tool_definitions.return_value = [{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object", "properties": {}}}]
    tm.get_last_sources.return_value = [{"name": "Intro to AI - Lesson 1", "link": "https://example.com/lesson/1"}]
    tm.reset_sources.return_value = None
    return tm
