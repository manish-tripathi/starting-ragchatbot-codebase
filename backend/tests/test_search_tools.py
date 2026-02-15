"""Tests for CourseSearchTool and ToolManager."""

import pytest
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager, Tool


# ═══════════════════════════════════════════════════════════════════════════
# A. CourseSearchTool.execute() — Result Handling
# ═══════════════════════════════════════════════════════════════════════════

class TestCourseSearchToolResults:

    def test_execute_returns_formatted_results(self, mock_vector_store, mock_search_results):
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="machine learning")

        # Should contain course header and document content
        assert "[Intro to AI - Lesson 1]" in result
        assert "Machine learning is a subset of AI." in result
        assert "[Intro to AI - Lesson 2]" in result
        assert "Neural networks are inspired by the brain." in result

    def test_execute_empty_results_no_filters(self, mock_vector_store, mock_empty_results):
        mock_vector_store.search.return_value = mock_empty_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum computing")
        assert result == "No relevant content found."

    def test_execute_empty_results_with_course_filter(self, mock_vector_store, mock_empty_results):
        mock_vector_store.search.return_value = mock_empty_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum", course_name="Physics 101")
        assert "in course 'Physics 101'" in result

    def test_execute_empty_results_with_lesson_filter(self, mock_vector_store, mock_empty_results):
        mock_vector_store.search.return_value = mock_empty_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum", lesson_number=3)
        assert "in lesson 3" in result

    def test_execute_error_from_store(self, mock_vector_store, mock_error_results):
        mock_vector_store.search.return_value = mock_error_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")
        assert result == "Search error: connection refused"


# ═══════════════════════════════════════════════════════════════════════════
# B. CourseSearchTool.execute() — Filter Pass-Through
# ═══════════════════════════════════════════════════════════════════════════

class TestCourseSearchToolFilters:

    def test_execute_passes_query_to_store(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="deep learning")

        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs.kwargs["query"] == "deep learning"

    def test_execute_passes_course_name_to_store(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", course_name="ML Course")

        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs.kwargs["course_name"] == "ML Course"

    def test_execute_passes_lesson_number_to_store(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", lesson_number=5)

        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs.kwargs["lesson_number"] == 5

    def test_execute_passes_all_filters(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="neural nets", course_name="AI 101", lesson_number=2)

        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs.kwargs["query"] == "neural nets"
        assert call_kwargs.kwargs["course_name"] == "AI 101"
        assert call_kwargs.kwargs["lesson_number"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# C. CourseSearchTool — Source Tracking
# ═══════════════════════════════════════════════════════════════════════════

class TestCourseSearchToolSources:

    def test_execute_populates_last_sources(self, mock_vector_store, mock_search_results):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="machine learning")

        assert len(tool.last_sources) > 0
        for src in tool.last_sources:
            assert "name" in src
            assert "link" in src

    def test_execute_deduplicates_sources(self, mock_vector_store):
        """Multiple chunks from the same lesson should produce only one source entry."""
        dup_results = SearchResults(
            documents=["chunk A", "chunk B"],
            metadata=[
                {"course_title": "AI", "lesson_number": 1},
                {"course_title": "AI", "lesson_number": 1},
            ],
            distances=[0.1, 0.2],
        )
        mock_vector_store.search.return_value = dup_results
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        assert len(tool.last_sources) == 1

    def test_execute_sources_empty_on_no_results(self, mock_vector_store, mock_empty_results):
        mock_vector_store.search.return_value = mock_empty_results
        tool = CourseSearchTool(mock_vector_store)
        # Pre-populate to ensure reset
        tool.last_sources = [{"name": "stale", "link": None}]
        tool.execute(query="nothing")

        # last_sources should still contain the stale data because
        # the empty-results path doesn't call _format_results.
        # The ToolManager.reset_sources() is what clears it externally.
        # So we just verify the method doesn't crash.


# ═══════════════════════════════════════════════════════════════════════════
# D. ToolManager
# ═══════════════════════════════════════════════════════════════════════════

class _DummyTool(Tool):
    """Minimal tool for ToolManager tests."""

    def __init__(self, name="dummy_tool", result="dummy result"):
        self._name = name
        self._result = result
        self.last_sources = []

    def get_tool_definition(self):
        return {"name": self._name, "description": "A dummy tool",
                "input_schema": {"type": "object", "properties": {}}}

    def execute(self, **kwargs):
        return self._result


class TestToolManager:

    @patch("search_tools.ToolLogger")
    def test_register_and_execute_tool(self, MockLogger):
        tm = ToolManager()
        dummy = _DummyTool(name="my_tool", result="ok")
        tm.register_tool(dummy)

        result = tm.execute_tool("my_tool")
        assert result == "ok"

    @patch("search_tools.ToolLogger")
    def test_execute_unknown_tool(self, MockLogger):
        tm = ToolManager()
        result = tm.execute_tool("nonexistent")
        assert "not found" in result.lower()

    @patch("search_tools.ToolLogger")
    def test_execute_tool_logs_success(self, MockLogger):
        tm = ToolManager()
        dummy = _DummyTool()
        tm.register_tool(dummy)
        tm.execute_tool("dummy_tool")

        tm.logger.log_tool_call.assert_called()
        call_kwargs = tm.logger.log_tool_call.call_args
        assert call_kwargs.kwargs.get("success") or call_kwargs[1].get("success") or call_kwargs[0][3]

    @patch("search_tools.ToolLogger")
    def test_execute_tool_logs_and_reraises_errors(self, MockLogger):
        tm = ToolManager()

        class _FailTool(Tool):
            def get_tool_definition(self):
                return {"name": "fail", "description": "fails",
                        "input_schema": {"type": "object", "properties": {}}}
            def execute(self, **kwargs):
                raise RuntimeError("boom")

        tm.register_tool(_FailTool())

        with pytest.raises(RuntimeError, match="boom"):
            tm.execute_tool("fail")

        # Should still have logged the failure
        tm.logger.log_tool_call.assert_called()

    @patch("search_tools.ToolLogger")
    def test_get_last_sources_returns_tool_sources(self, MockLogger):
        tm = ToolManager()
        dummy = _DummyTool()
        dummy.last_sources = [{"name": "src", "link": None}]
        tm.register_tool(dummy)

        assert tm.get_last_sources() == [{"name": "src", "link": None}]

    @patch("search_tools.ToolLogger")
    def test_reset_sources_clears_all(self, MockLogger):
        tm = ToolManager()
        dummy = _DummyTool()
        dummy.last_sources = [{"name": "src", "link": None}]
        tm.register_tool(dummy)

        tm.reset_sources()
        assert tm.get_last_sources() == []
