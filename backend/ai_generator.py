import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Selection Rules:
- **get_course_outline**: Use when the user asks about a course outline, structure, syllabus, lesson list, what lessons are in a course, what a course covers, or how many lessons a course has. Present the course title, course link, and every lesson with its number and title as returned.
- **search_course_content**: Use for questions about specific topics, concepts, or detailed educational content within lessons.
- Call ONE tool at a time. Do not call multiple tools in parallel.
- You may call up to 2 tools sequentially per query (e.g., get an outline first, then search for details).
- Synthesize your answer from outlines and the data already returned — do not search every lesson individually.
- If a tool yields no results, state this clearly without offering alternatives.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching.
- **Course outline/structure/syllabus/lesson list questions**: Use get_course_outline, then present the complete course title, course link, and full lesson list exactly as returned.
- **Course-specific content questions**: Use search_course_content, then answer.
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis.
 - Do not mention "based on the search results".

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 2048
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters
        messages = [{"role": "user", "content": query}]
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Tool loop: up to MAX_TOOL_ROUNDS iterations
        for round_num in range(self.MAX_TOOL_ROUNDS):
            response = self.client.messages.create(**api_params)

            # If the model hit the token limit mid-response, strip tools and
            # retry once so it synthesises from what it already has.
            if response.stop_reason == "max_tokens":
                api_params.pop("tool_choice", None)
                api_params.pop("tools", None)
                response = self.client.messages.create(**api_params)
                return self._extract_text(response)

            # Exit loop if no tool use or no tool_manager
            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text(response)

            # Execute tools, append results to messages
            tool_results = self._execute_tool_calls(response, tool_manager)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            api_params["messages"] = messages

        # Post-loop: force text-only response after max rounds exhausted
        # Remove tool_choice and tools so the model generates a text answer
        # instead of trying to call yet another tool
        api_params.pop("tool_choice", None)
        api_params.pop("tools", None)
        response = self.client.messages.create(**api_params)
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response):
        """Return the first text block from a response, or a fallback message."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return "I'm sorry, I wasn't able to produce a text response."

    def _execute_tool_calls(self, response, tool_manager):
        """Execute one tool call from a response; skip extras to limit token use.

        Every tool_use block must have a matching tool_result (API requirement),
        so skipped blocks get a short placeholder message.
        """
        tool_results = []
        executed = False
        for block in response.content:
            if block.type == "tool_use":
                if not executed:
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                    executed = True
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Skipped: one tool per round. Use existing results or call another tool next round.",
                    })
        return tool_results