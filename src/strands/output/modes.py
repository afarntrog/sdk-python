"""Concrete output mode implementations."""

from typing import Any, Type, Optional, TYPE_CHECKING
from pydantic import BaseModel

from .base import OutputMode
from strands.tools.structured_output_tool import StructuredOutputTool
from strands.tools.structured_output import convert_pydantic_to_tool_spec

if TYPE_CHECKING:
    from strands.models.model import Model
    from strands.types.tools import ToolSpec


class ToolMode(OutputMode):
    """Use function calling for structured output (DEFAULT).
    
    This is the most reliable approach across all model providers and ensures
    consistent behavior regardless of model capabilities.
    """

    def get_tool_specs(self, output_type: Type[BaseModel]) -> list["ToolSpec"]:
        """Convert Pydantic model to tool specifications."""
        return [StructuredOutputTool(output_type).tool_spec]

    def get_tool_instances(self, output_type: Type[BaseModel]) -> list["StructuredOutputTool"]:
        """Create actual tool instances for structured output.
        
        Args:
            output_type: The Pydantic model class to create tools for.
            
        Returns:
            List containing a single StructuredOutputTool instance.
        """
        return [StructuredOutputTool(output_type)]

    def extract_result(self, model_response: Any) -> Any:
        """Extract result from tool call response."""
        # Implementation will be added when integrating with event loop
        return model_response

    def is_supported_by_model(self, model: "Model") -> bool:
        """Tool-based output is supported by all models that support function calling."""
        return True  # All our models support function calling


class NativeMode(OutputMode):
    """Use model's native structured output capabilities.
    
    Only use when explicitly requested and supported by the model.
    Falls back to ToolMode if not supported.
    """

    def get_tool_specs(self, output_type: Type[BaseModel]) -> list["ToolSpec"]:
        """Return empty list - will use native JSON schema instead."""
        raise NotImplementedError()

    def extract_result(self, model_response: Any) -> Any:
        """Extract result from native structured output."""
        # Implementation will be added when integrating with model providers
        raise NotImplementedError()

    def is_supported_by_model(self, model: "Model") -> bool:
        """Check if model supports native structured output."""
        raise NotImplementedError()


class PromptMode(OutputMode):
    """Use prompting to guide output format.
    
    Only use when explicitly requested. Less reliable than tool-based approach
    but can work with models that have limited function calling support.
    """

    def __init__(self, template: Optional[str] = None):
        """Initialize with optional custom template.
        
        Args:
            template: Custom template for prompting (uses default if None)
        """
        self.template = template or "Please respond with JSON matching this schema: {schema}"

    def get_tool_specs(self, output_type: Type[BaseModel]) -> list["ToolSpec"]:
        """Return empty list - will inject schema into system prompt instead."""
        raise NotImplementedError()

    def extract_result(self, model_response: Any) -> Any:
        """Extract result from prompted response."""
        raise NotImplementedError()

    def is_supported_by_model(self, model: "Model") -> bool:
        """Prompting-based output works with all models."""
        raise NotImplementedError()
