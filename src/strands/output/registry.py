"""Output type registry and resolution utilities."""

from typing import Type, Union, Optional, Dict, Any
from pydantic import BaseModel

from .base import OutputMode, OutputSchema
from .modes import ToolOutput


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
            output_mode: Output mode (defaults to ToolOutput if not specified)
            
        Returns:
            Resolved OutputSchema or None if no output type specified
        """
        if not output_type:
            return None

        if isinstance(output_type, OutputSchema):
            return output_type

        # Default to ToolOutput if no mode specified
        resolved_mode = output_mode or ToolOutput()

        return OutputSchema(output_type, resolved_mode)

    def get_cached_tool_specs(self, schema: OutputSchema) -> Optional[list]:
        """Get cached tool specifications for schema.
        
        Args:
            schema: Output schema to get tool specs for
            
        Returns:
            Cached tool specs or None if not cached
        """
        cache_key = self._get_cache_key(schema)
        return self._tool_spec_cache.get(cache_key)

    def cache_tool_specs(self, schema: OutputSchema, tool_specs: list) -> None:
        """Cache tool specifications for schema.
        
        Args:
            schema: Output schema
            tool_specs: Tool specifications to cache
        """
        cache_key = self._get_cache_key(schema)
        self._tool_spec_cache[cache_key] = tool_specs

    def _get_cache_key(self, schema: OutputSchema) -> str:
        """Generate cache key for schema."""
        type_name = schema.type.__name__
        mode_name = schema.mode.__class__.__name__
        return f"{mode_name}:{type_name}"


def validate_output_type(output_type: Type[BaseModel]) -> None:
    """Validate that output type is a Pydantic model.
    
    Args:
        output_type: Type to validate
        
    Raises:
        ValueError: If type is not a Pydantic model
    """
    if not issubclass(output_type, BaseModel):
        raise ValueError(f"Output type must be a Pydantic model, got {output_type}")


def validate_output_schema(schema: OutputSchema) -> None:
    """Validate output schema.
    
    Args:
        schema: Schema to validate
        
    Raises:
        ValueError: If schema is invalid
    """
    validate_output_type(schema.type)
