"""Tests for AIGenerator — direct responses, tool execution flow, multi-round tools, and error handling."""

import pytest
from unittest.mock import MagicMock, patch, call

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from helpers import make_text_response, make_tool_use_response


SAMPLE_TOOLS = [
    {
        "name": "search_course_content",
        "description": "Search course content",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }
]


@pytest.fixture
def ai(mock_anthropic_client):
    """AIGenerator with a mocked Anthropic client."""
    with patch("ai_generator.anthropic") as mock_module:
        mock_module.Anthropic.return_value = mock_anthropic_client
        gen = AIGenerator(api_key="fake-key", model="claude-test")
        # Directly replace the client so our mock is used
        gen.client = mock_anthropic_client
        return gen


# ═══════════════════════════════════════════════════════════════════════════
# A. Direct Responses (No Tool Use)
# ═══════════════════════════════════════════════════════════════════════════

class TestDirectResponses:

    def test_direct_response_returns_text(self, ai, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_text_response("Paris is the capital.")

        result = ai.generate_response("What is the capital of France?")
        assert result == "Paris is the capital."

    def test_system_prompt_included(self, ai, mock_anthropic_client):
        ai.generate_response("hello")

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert "AI assistant" in call_kwargs["system"]

    def test_conversation_history_appended_to_system(self, ai, mock_anthropic_client):
        history = "User: hi\nAssistant: hello"
        ai.generate_response("follow up", conversation_history=history)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "Previous conversation:" in call_kwargs["system"]
        assert history in call_kwargs["system"]

    def test_no_history_uses_base_system_prompt(self, ai, mock_anthropic_client):
        ai.generate_response("hello")

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "Previous conversation:" not in call_kwargs["system"]

    def test_tools_passed_to_api_call(self, ai, mock_anthropic_client):
        ai.generate_response("search something", tools=SAMPLE_TOOLS)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == SAMPLE_TOOLS
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_means_no_tool_params(self, ai, mock_anthropic_client):
        ai.generate_response("hello", tools=None)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs


# ═══════════════════════════════════════════════════════════════════════════
# B. Tool Execution Flow (Single Round)
# ═══════════════════════════════════════════════════════════════════════════

class TestToolExecutionFlow:

    def _setup_tool_flow(self, mock_anthropic_client, tool_result_text="Tool output here"):
        """Configure the mock client for a tool_use → final text flow."""
        tool_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "machine learning"},
            tool_use_id="toolu_abc123",
        )
        final_response = make_text_response("Here is the answer based on course content.")

        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]
        return tool_response, final_response

    def test_tool_use_triggers_execution(self, ai, mock_anthropic_client, mock_tool_manager):
        self._setup_tool_flow(mock_anthropic_client)

        ai.generate_response("tell me about ML", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        # tool_manager.execute_tool should have been called
        mock_tool_manager.execute_tool.assert_called_once()

    def test_tool_manager_execute_called_with_correct_args(self, ai, mock_anthropic_client, mock_tool_manager):
        self._setup_tool_flow(mock_anthropic_client)

        ai.generate_response("tell me about ML", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="machine learning"
        )

    def test_tool_result_included_in_followup_messages(self, ai, mock_anthropic_client, mock_tool_manager):
        self._setup_tool_flow(mock_anthropic_client)

        ai.generate_response("tell me about ML", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        # The second call to messages.create should have tool_result in messages
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]

        # Last message should be the tool_result from the user role
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_abc123"

    def test_followup_call_includes_tools_parameter(self, ai, mock_anthropic_client, mock_tool_manager):
        """The follow-up API call after tool execution MUST include the 'tools' parameter."""
        self._setup_tool_flow(mock_anthropic_client)

        ai.generate_response("tell me about ML", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        # The second API call must include tools
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
        assert "tools" in second_call_kwargs, (
            "Follow-up API call is missing 'tools' parameter. "
            "The Anthropic API requires 'tools' when messages contain tool_use/tool_result blocks."
        )
        assert second_call_kwargs["tools"] == SAMPLE_TOOLS

    def test_followup_returns_final_text(self, ai, mock_anthropic_client, mock_tool_manager):
        self._setup_tool_flow(mock_anthropic_client)

        result = ai.generate_response("tell me about ML", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)
        assert result == "Here is the answer based on course content."

    def test_no_tool_manager_skips_execution(self, ai, mock_anthropic_client):
        """If tool_manager is None but stop_reason is tool_use, fall through to text extraction."""
        # Return a response with both a tool_use block and a text block
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_block.id = "toolu_xyz"

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I would search but no tool manager."

        response = MagicMock()
        response.content = [text_block, tool_block]
        response.stop_reason = "tool_use"

        mock_anthropic_client.messages.create.return_value = response

        # With tool_manager=None, should just return content[0].text
        result = ai.generate_response("test", tools=SAMPLE_TOOLS, tool_manager=None)
        assert result == "I would search but no tool manager."


# ═══════════════════════════════════════════════════════════════════════════
# C. Multi-Round Tool Execution
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiRoundToolExecution:

    def _setup_two_round_flow(self, mock_anthropic_client, mock_tool_manager):
        """Configure mocks for a 2-round tool flow: tool1 → tool2 → final text."""
        round1_response = make_tool_use_response(
            tool_name="get_course_outline",
            tool_input={"course_name": "Intro to AI"},
            tool_use_id="toolu_round1",
        )
        round2_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "neural networks"},
            tool_use_id="toolu_round2",
        )
        final_response = make_text_response("Here is the multi-round answer.")

        mock_anthropic_client.messages.create.side_effect = [
            round1_response, round2_response, final_response
        ]
        mock_tool_manager.execute_tool.side_effect = [
            "Outline: Lesson 1, Lesson 2, Lesson 3",
            "Neural networks are covered in Lesson 2.",
        ]
        return round1_response, round2_response, final_response

    def test_two_rounds_makes_three_api_calls(self, ai, mock_anthropic_client, mock_tool_manager):
        self._setup_two_round_flow(mock_anthropic_client, mock_tool_manager)

        ai.generate_response("Find a topic from course outline", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        assert mock_anthropic_client.messages.create.call_count == 3

    def test_two_rounds_executes_both_tools(self, ai, mock_anthropic_client, mock_tool_manager):
        self._setup_two_round_flow(mock_anthropic_client, mock_tool_manager)

        ai.generate_response("Find a topic from course outline", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        assert mock_tool_manager.execute_tool.call_count == 2

    def test_two_rounds_returns_final_text(self, ai, mock_anthropic_client, mock_tool_manager):
        self._setup_two_round_flow(mock_anthropic_client, mock_tool_manager)

        result = ai.generate_response("Find a topic from course outline", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        assert result == "Here is the multi-round answer."

    def test_two_rounds_messages_accumulate(self, ai, mock_anthropic_client, mock_tool_manager):
        self._setup_two_round_flow(mock_anthropic_client, mock_tool_manager)

        ai.generate_response("Find a topic from course outline", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        # 3rd API call should have 5 messages: user, assistant_tool1, user_result1, assistant_tool2, user_result2
        third_call_kwargs = mock_anthropic_client.messages.create.call_args_list[2].kwargs
        messages = third_call_kwargs["messages"]
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    def test_two_rounds_in_loop_calls_include_tools(self, ai, mock_anthropic_client, mock_tool_manager):
        """The in-loop API calls include tools; the post-loop forced-text call does not."""
        self._setup_two_round_flow(mock_anthropic_client, mock_tool_manager)

        ai.generate_response("Find a topic from course outline", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        calls = mock_anthropic_client.messages.create.call_args_list
        # First two calls (inside the loop) should include tools
        for i in range(2):
            assert "tools" in calls[i].kwargs, f"In-loop API call {i} missing 'tools' parameter"
        # Third call (post-loop forced text) should NOT include tools
        assert "tools" not in calls[2].kwargs, "Post-loop forced-text call should not include 'tools'"

    def test_two_rounds_different_tools(self, ai, mock_anthropic_client, mock_tool_manager):
        self._setup_two_round_flow(mock_anthropic_client, mock_tool_manager)

        ai.generate_response("Find a topic from course outline", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0][0] == "get_course_outline"
        assert calls[1][0][0] == "search_course_content"

    def test_early_termination_after_first_round(self, ai, mock_anthropic_client, mock_tool_manager):
        """When Claude returns text after round 1, only 2 API calls are made."""
        tool_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "ML basics"},
            tool_use_id="toolu_only",
        )
        final_response = make_text_response("Done after one round.")

        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        result = ai.generate_response("search ML", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        assert mock_anthropic_client.messages.create.call_count == 2
        assert result == "Done after one round."


# ═══════════════════════════════════════════════════════════════════════════
# D. Max Rounds Enforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestMaxRoundsEnforcement:

    def test_max_rounds_constant(self):
        assert AIGenerator.MAX_TOOL_ROUNDS == 2

    def test_stops_after_max_rounds(self, ai, mock_anthropic_client, mock_tool_manager):
        """When Claude wants a 3rd tool call, code stops at 3 API calls and returns text."""
        round1_response = make_tool_use_response(
            tool_name="get_course_outline",
            tool_input={"course_name": "AI"},
            tool_use_id="toolu_r1",
        )
        round2_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "deep learning"},
            tool_use_id="toolu_r2",
        )
        # Post-loop final call returns text (even though Claude might want more tools)
        final_response = make_text_response("Forced final answer after max rounds.")

        mock_anthropic_client.messages.create.side_effect = [
            round1_response, round2_response, final_response
        ]
        mock_tool_manager.execute_tool.side_effect = ["Outline data", "Search results"]

        result = ai.generate_response("complex query", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        assert mock_anthropic_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result == "Forced final answer after max rounds."


# ═══════════════════════════════════════════════════════════════════════════
# E. Error Scenarios
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorScenarios:

    def test_api_error_propagates(self, ai, mock_anthropic_client):
        mock_anthropic_client.messages.create.side_effect = Exception("API down")

        with pytest.raises(Exception, match="API down"):
            ai.generate_response("hello")

    def test_tool_execution_error_propagates(self, ai, mock_anthropic_client, mock_tool_manager):
        tool_response = make_tool_use_response("search_course_content", {"query": "test"})
        mock_anthropic_client.messages.create.return_value = tool_response
        mock_tool_manager.execute_tool.side_effect = RuntimeError("Tool crashed")

        with pytest.raises(RuntimeError, match="Tool crashed"):
            ai.generate_response("test", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

    def test_tool_error_in_second_round_propagates(self, ai, mock_anthropic_client, mock_tool_manager):
        """Tool crash in round 2 raises; execute_tool called twice."""
        round1_response = make_tool_use_response(
            tool_name="get_course_outline",
            tool_input={"course_name": "AI"},
            tool_use_id="toolu_r1",
        )
        round2_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_use_id="toolu_r2",
        )
        mock_anthropic_client.messages.create.side_effect = [round1_response, round2_response]
        mock_tool_manager.execute_tool.side_effect = [
            "Round 1 result",
            RuntimeError("Tool crashed in round 2"),
        ]

        with pytest.raises(RuntimeError, match="Tool crashed in round 2"):
            ai.generate_response("test", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        assert mock_tool_manager.execute_tool.call_count == 2

    def test_api_error_in_second_round_propagates(self, ai, mock_anthropic_client, mock_tool_manager):
        """API error on 2nd call propagates; execute_tool called once."""
        round1_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_use_id="toolu_r1",
        )
        mock_anthropic_client.messages.create.side_effect = [
            round1_response,
            Exception("API error on second call"),
        ]

        with pytest.raises(Exception, match="API error on second call"):
            ai.generate_response("test", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        assert mock_tool_manager.execute_tool.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# F. System Prompt
# ═══════════════════════════════════════════════════════════════════════════

class TestSystemPrompt:

    def test_prompt_mentions_sequential_tools(self):
        assert "ONE tool at a time" in AIGenerator.SYSTEM_PROMPT or "2 tools sequentially" in AIGenerator.SYSTEM_PROMPT

    def test_prompt_says_one_tool_at_a_time(self):
        assert "ONE tool at a time" in AIGenerator.SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════
# G. max_tokens Handling
# ═══════════════════════════════════════════════════════════════════════════

class TestMaxTokensHandling:

    def test_max_tokens_forces_text_retry(self, ai, mock_anthropic_client, mock_tool_manager):
        """When stop_reason='max_tokens' mid-loop, tools are stripped and a text-only retry is made."""
        truncated_response = make_text_response("Now let me search for...", stop_reason="max_tokens")
        final_response = make_text_response("Here is the synthesized answer.")

        mock_anthropic_client.messages.create.side_effect = [truncated_response, final_response]

        result = ai.generate_response("summarize everything", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        assert result == "Here is the synthesized answer."
        assert mock_anthropic_client.messages.create.call_count == 2

        # The retry call should NOT have tools or tool_choice
        retry_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in retry_kwargs
        assert "tool_choice" not in retry_kwargs


# ═══════════════════════════════════════════════════════════════════════════
# H. Parallel Tool Call Limiting
# ═══════════════════════════════════════════════════════════════════════════

class TestParallelToolCallLimiting:

    def _make_multi_tool_response(self, count=3):
        """Build a mock response with multiple tool_use blocks."""
        blocks = []
        for i in range(count):
            block = MagicMock()
            block.type = "tool_use"
            block.name = "search_course_content"
            block.input = {"query": f"topic {i}"}
            block.id = f"toolu_{i}"
            blocks.append(block)
        response = MagicMock()
        response.content = blocks
        response.stop_reason = "tool_use"
        return response

    def test_parallel_tool_calls_limited(self, ai, mock_anthropic_client, mock_tool_manager):
        """Only the first tool_use block should be executed, even if model emits 3."""
        multi_response = self._make_multi_tool_response(3)
        final_response = make_text_response("Answer from first tool result.")

        mock_anthropic_client.messages.create.side_effect = [multi_response, final_response]

        result = ai.generate_response("broad query", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        # Only one tool should have actually been executed
        mock_tool_manager.execute_tool.assert_called_once()
        assert result == "Answer from first tool result."

    def test_skipped_tools_get_result_message(self, ai, mock_anthropic_client, mock_tool_manager):
        """Every tool_use_id must get a tool_result, including skipped ones."""
        multi_response = self._make_multi_tool_response(3)
        final_response = make_text_response("Done.")

        mock_anthropic_client.messages.create.side_effect = [multi_response, final_response]

        ai.generate_response("broad query", tools=SAMPLE_TOOLS, tool_manager=mock_tool_manager)

        # The second API call's messages should include tool_results for all 3 tool_use ids
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
        tool_result_msg = second_call_kwargs["messages"][-1]
        assert tool_result_msg["role"] == "user"

        tool_results = tool_result_msg["content"]
        assert len(tool_results) == 3
        assert all(r["type"] == "tool_result" for r in tool_results)

        # First one should have real content, others should have skip message
        ids = {r["tool_use_id"] for r in tool_results}
        assert ids == {"toolu_0", "toolu_1", "toolu_2"}
        assert "Skipped" in tool_results[1]["content"]
        assert "Skipped" in tool_results[2]["content"]
