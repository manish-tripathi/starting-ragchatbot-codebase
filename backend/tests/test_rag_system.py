"""Tests for RAGSystem — query pipeline, source management, sessions, errors."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def rag():
    """RAGSystem with all heavy dependencies mocked out."""
    # Build a fake config with all required attributes
    config = MagicMock()
    config.CHUNK_SIZE = 500
    config.CHUNK_OVERLAP = 50
    config.CHROMA_PATH = "/tmp/test_chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "fake-key"
    config.ANTHROPIC_MODEL = "claude-test"
    config.MAX_HISTORY = 5

    with patch("rag_system.DocumentProcessor"), \
         patch("rag_system.VectorStore") as MockVS, \
         patch("rag_system.AIGenerator") as MockAI, \
         patch("rag_system.SessionManager") as MockSM, \
         patch("rag_system.ToolManager") as MockTM, \
         patch("rag_system.CourseSearchTool"), \
         patch("rag_system.CourseOutlineTool"):

        from rag_system import RAGSystem
        system = RAGSystem(config)

        # Expose mocks for assertions
        system._mock_ai = system.ai_generator
        system._mock_sm = system.session_manager
        system._mock_tm = system.tool_manager

        # Defaults
        system._mock_ai.generate_response.return_value = "AI answer."
        system._mock_tm.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        system._mock_tm.get_last_sources.return_value = [{"name": "Src", "link": "http://example.com"}]
        system._mock_sm.get_conversation_history.return_value = None

        return system


# ═══════════════════════════════════════════════════════════════════════════
# A. Query Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestQueryPipeline:

    def test_query_returns_response_and_sources_tuple(self, rag):
        result = rag.query("What is AI?")
        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_builds_prompt_with_user_query(self, rag):
        rag.query("What is deep learning?")

        call_kwargs = rag._mock_ai.generate_response.call_args.kwargs
        assert "What is deep learning?" in call_kwargs["query"]

    def test_query_passes_tools_to_ai_generator(self, rag):
        rag.query("test")

        call_kwargs = rag._mock_ai.generate_response.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == rag._mock_tm.get_tool_definitions()

    def test_query_passes_tool_manager_to_ai_generator(self, rag):
        rag.query("test")

        call_kwargs = rag._mock_ai.generate_response.call_args.kwargs
        assert call_kwargs["tool_manager"] is rag._mock_tm


# ═══════════════════════════════════════════════════════════════════════════
# B. Source Management
# ═══════════════════════════════════════════════════════════════════════════

class TestSourceManagement:

    def test_query_resets_sources_before_execution(self, rag):
        rag.query("test")

        # reset_sources should be called before generate_response
        rag._mock_tm.reset_sources.assert_called_once()

    def test_query_retrieves_sources_after_execution(self, rag):
        rag.query("test")

        rag._mock_tm.get_last_sources.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# C. Session Management
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionManagement:

    def test_query_retrieves_session_history(self, rag):
        rag._mock_sm.get_conversation_history.return_value = "User: hi\nAssistant: hello"

        rag.query("follow up", session_id="sess_1")

        rag._mock_sm.get_conversation_history.assert_called_with("sess_1")
        call_kwargs = rag._mock_ai.generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] == "User: hi\nAssistant: hello"

    def test_query_adds_exchange_to_session(self, rag):
        rag.query("What is AI?", session_id="sess_1")

        rag._mock_sm.add_exchange.assert_called_once()
        args = rag._mock_sm.add_exchange.call_args
        assert args[0][0] == "sess_1"  # session_id
        assert "What is AI?" in args[0][1]  # user query (wrapped in prompt)

    def test_query_works_without_session(self, rag):
        response, sources = rag.query("no session")

        assert response == "AI answer."
        rag._mock_sm.add_exchange.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# D. Error Propagation
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorPropagation:

    def test_query_propagates_ai_generator_errors(self, rag):
        rag._mock_ai.generate_response.side_effect = Exception("API 500")

        with pytest.raises(Exception, match="API 500"):
            rag.query("will fail")
