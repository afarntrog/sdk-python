"""Structured output tool implementation.

This module provides a real tool implementation for structured output that integrates
with the existing tool execution and error handling infrastructure.
"""

import logging
from typing import Any, Type
from typing_extensions import override
from pydantic import BaseModel, ValidationError

from ...types._events import ToolResultEvent
from ...types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse, ToolResult
from .structured_output_utils import convert_pydantic_to_tool_spec

logger = logging.getLogger(__name__)


class StructuredOutputTool(AgentTool):
    """Tool implementation for structured output validation.
    
    This class converts structured output from a "pseudo-tool" into a real tool
    that can leverage the existing error handling and retry infrastructure.
    """

    def __init__(self, output_type: Type[BaseModel]) -> None:
        """Initialize a structured output tool.

        Args:
            output_type: The Pydantic model class that defines the expected output structure.
        """
        super().__init__()
        self._output_type = output_type
        self._tool_spec = convert_pydantic_to_tool_spec(output_type)
        self._tool_name = self._tool_spec.get("name")

    @property
    def tool_name(self) -> str:
        """Get the name of the tool.

        Returns:
            The name of the tool (same as the Pydantic model class name).
        """
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the tool specification for this structured output tool.

        Returns:
            The tool specification generated from the Pydantic model.
        """
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        """Identifies this as a structured output tool implementation.

        Returns:
            "structured_output".
        """
        return "structured_output"

    @property
    def output_type(self) -> Type[BaseModel]:
        """Get the Pydantic model type for this tool.

        Returns:
            The Pydantic model class.
        """
        return self._output_type

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Validate the structured output and return appropriate result.

        Args:
            tool_use: The tool use request containing the data to validate.
            invocation_state: Context for the tool invocation, including agent state.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result (success or error).
        """
        tool_input = tool_use.get("input", {})
        tool_use_id = str(tool_use.get("toolUseId", ""))

        try:
            # Attempt to create and validate the Pydantic object
            validated_object = self._output_type(**tool_input)
            
            logger.debug(f"Successfully validated structured output for {self._tool_name}")
            
            # Store in invocation state with namespaced key
            key = f"structured_output_{tool_use_id}"
            invocation_state[key] = validated_object
            
            # Create clean success result
            result: ToolResult = {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"Successfully validated {self._tool_name} structured output"}],
            }
            
            yield ToolResultEvent(result)

        except ValidationError as e:
            # Create detailed error message for the LLM
            error_details = []
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"]) if error["loc"] else "root"
                error_details.append(f"Field '{field_path}': {error['msg']}")
            
            error_message = (
                f"Validation failed for {self._tool_name}. Please fix the following errors:\n" +
                "\n".join(f"- {detail}" for detail in error_details)
            )
            
            logger.warning(f"Structured output validation failed for {self._tool_name}: {error_message}")
            
            # Create error result that will be sent back to the LLM
            result: ToolResult = {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": error_message}],
            }
            
            yield ToolResultEvent(result)

        except Exception as e:
            # Ensure cleanup on unexpected errors
            key = f"structured_output_{tool_use_id}"
            invocation_state.pop(key, None)
            
            # Handle any other unexpected errors
            error_message = f"Unexpected error validating {self._tool_name}: {str(e)}"
            logger.exception(f"Unexpected error in structured output tool {self._tool_name}")
            
            result: ToolResult = {
                "toolUseId": tool_use_id,
                "status": "error", 
                "content": [{"text": error_message}],
            }
            
            yield ToolResultEvent(result)
