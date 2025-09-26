"""Output type registry and resolution utilities."""

from typing import Type, Union, Optional, Dict, Any
from pydantic import BaseModel

from .base import OutputMode, OutputSchema
from .modes import ToolMode


class OutputRegistry:
    """Registry for output type resolution and caching."""

    def __init__(self):
        """Initialize empty registry."""
        self._tool_spec_cache: Dict[str, Any] = {}

    def resolve_output_schema(
        self,
        output_type: Optional[Union[Type[BaseModel], OutputSchema]] = None,
        output_mode: Optional[OutputMode] = None,
    ) -> Optional[OutputSchema]:
        """Resolve output type and mode into OutputSchema.
        
        Args:
            output_type: Output type specification
            output_mode: Output mode (defaults to ToolMode if not specified)
            
        Returns:
            Resolved OutputSchema or None if no output type specified
        """
        if not output_type:
            return None

        if isinstance(output_type, OutputSchema):
            return output_type

        # Default to ToolMode if no mode specified
        resolved_mode = output_mode or ToolMode()

        return OutputSchema(output_type, resolved_mode)
