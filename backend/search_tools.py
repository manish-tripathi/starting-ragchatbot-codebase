import time
from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults
from tool_logger import ToolLogger


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search for specific topics or concepts within course lesson content. Do NOT use this for course outlines, syllabi, or lesson lists — use get_course_outline instead.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        seen_sources = {}  # Deduplicate by source name

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI (deduplicated)
            source_name = course_title
            if lesson_num is not None:
                source_name += f" - Lesson {lesson_num}"

            if source_name not in seen_sources:
                link = None
                if lesson_num is not None:
                    link = self.store.get_lesson_link(course_title, lesson_num)
                seen_sources[source_name] = {"name": source_name, "link": link}

            display_doc = doc[:400] + "..." if len(doc) > 400 else doc
            formatted.append(f"{header}\n{display_doc}")

        # Store deduplicated sources for retrieval
        self.last_sources = list(seen_sources.values())

        return "\n\n".join(formatted)

class CourseOutlineTool(Tool):
    """Tool for retrieving the full outline of a course"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []

    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "name": "get_course_outline",
            "description": "Get the full outline of a course including its title, link, and complete lesson list with lesson numbers and titles. Use this when the user asks about what a course covers, its structure, syllabus, or lesson list.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title to look up (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_name"]
            }
        }

    def execute(self, course_name: str) -> str:
        outline = self.store.get_course_outline(course_name)
        if not outline:
            return f"No course found matching '{course_name}'."

        # Track source for the UI
        self.last_sources = [{"name": outline["title"], "link": outline.get("course_link")}]

        # Format the outline
        lines = [f"Course: {outline['title']}"]
        if outline.get("course_link"):
            lines.append(f"Link: {outline['course_link']}")

        lessons = outline.get("lessons", [])
        if lessons:
            lines.append(f"\nLessons ({len(lessons)}):")
            for lesson in lessons:
                lines.append(f"  Lesson {lesson['lesson_number']}: {lesson['lesson_title']}")
        else:
            lines.append("\nNo lessons found for this course.")

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
        self.logger = ToolLogger()
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            self.logger.log_tool_call(
                tool_name, kwargs, "", success=False, duration_ms=0,
                error=f"Tool '{tool_name}' not found",
            )
            return f"Tool '{tool_name}' not found"

        start = time.perf_counter()
        try:
            result = self.tools[tool_name].execute(**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            self.logger.log_tool_call(
                tool_name, kwargs, result, success=True, duration_ms=duration_ms,
            )
            return result
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            self.logger.log_tool_call(
                tool_name, kwargs, "", success=False, duration_ms=duration_ms,
                error=str(exc),
            )
            raise
    
    def get_last_sources(self) -> list:
        """Get sources from all tools used in the last query"""
        all_sources = []
        seen = set()
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                for source in tool.last_sources:
                    key = source.get("name", "")
                    if key not in seen:
                        seen.add(key)
                        all_sources.append(source)
        return all_sources

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []