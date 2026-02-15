"""Test helpers importable from test modules."""

from unittest.mock import MagicMock


def make_text_response(text: str, stop_reason: str = "end_turn"):
    """Build a mock Claude API response that contains a single text block."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.content = [text_block]
    response.stop_reason = stop_reason
    return response


def make_tool_use_response(tool_name: str, tool_input: dict, tool_use_id: str = "toolu_01"):
    """Build a mock Claude API response that requests a tool call."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input
    tool_block.id = tool_use_id

    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response
