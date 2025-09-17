"""Output type registry and utilities."""

from typing import Any, Dict, List, Optional, Type, Union
from functools import lru_cache

from .base import OutputMode, OutputSchema
from .modes import ToolOutput


class OutputRegistry:
    """Registry for managing output types and schemas."""
    
    def __init__(self):
        self._schemas: Dict[str, OutputSchema] = {}
        self._tool_spec_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_schema(self, name: str, schema: OutputSchema) -> None:
        """Register an output schema by name."""
        self._schemas[name] = schema
        # Clear cache for this schema
        cache_key = f"{name}_{id(schema.output_type)}"
        self._tool_spec_cache.pop(cache_key, None)
    
    def get_schema(self, name: str) -> Optional[OutputSchema]:
        """Get a registered schema by name."""
        return self._schemas.get(name)
    
    def resolve_output_schema(
        self,
        output_type: Optional[Union[Type, List[Type], "OutputSchema"]] = None,
        output_mode: Optional[OutputMode] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        default_schema: Optional[OutputSchema] = None
    ) -> OutputSchema:
        """Resolve output schema from various inputs."""
        # If output_type is already an OutputSchema, return it
        if isinstance(output_type, OutputSchema):
            return output_type
        
        # If name is provided, try to get registered schema
        if name and name in self._schemas:
            registered = self._schemas[name]
            # Override with provided parameters
            return OutputSchema(
                output_type=output_type or registered.output_type,
                mode=output_mode or registered.mode,
                name=name,
                description=description or registered.description
            )
        
        # Use default schema if provided and no specific type/mode
        if default_schema and not output_type and not output_mode:
            return default_schema
        
        # Handle list of types (for now, just use the first one)
        if isinstance(output_type, list) and output_type:
            output_type = output_type[0]
        
        # Create new schema
        return OutputSchema(
            output_type=output_type,
            mode=output_mode or ToolOutput(),
            name=name,
            description=description
        )
    
    def get_tool_specs(self, schema: OutputSchema) -> List[Dict[str, Any]]:
        """Get cached tool specifications for a schema."""
        if not schema.output_type:
            return []
        
        cache_key = f"{schema.name or 'unnamed'}_{id(schema.output_type)}"
        
        if cache_key not in self._tool_spec_cache:
            self._tool_spec_cache[cache_key] = schema.mode.get_tool_specs(schema.output_type)
        
        return self._tool_spec_cache[cache_key]
    
    def validate_schema(self, schema: OutputSchema) -> bool:
        """Validate an output schema."""
        if not isinstance(schema, OutputSchema):
            return False
        
        if not isinstance(schema.mode, OutputMode):
            return False
        
        # Check if output_type is a valid Pydantic model
        if schema.output_type:
            try:
                # Check if it has Pydantic model characteristics
                if not (hasattr(schema.output_type, 'model_fields') or 
                       hasattr(schema.output_type, '__annotations__')):
                    return False
            except Exception:
                return False
        
        return True
    
    def clear_cache(self) -> None:
        """Clear the tool spec cache."""
        self._tool_spec_cache.clear()


# Global registry instance
_global_registry = OutputRegistry()


def get_global_registry() -> OutputRegistry:
    """Get the global output registry."""
    return _global_registry


def convert_type_to_schema(
    output_type: Type,
    mode: Optional[OutputMode] = None,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> OutputSchema:
    """Convert a type to an output schema."""
    return OutputSchema(
        output_type=output_type,
        mode=mode or ToolOutput(),
        name=name,
        description=description
    )


@lru_cache(maxsize=128)
def get_type_name(output_type: Type) -> str:
    """Get a string name for a type."""
    if hasattr(output_type, '__name__'):
        return output_type.__name__
    return str(output_type)


def validate_output_type(output_type: Type) -> bool:
    """Validate that a type can be used for structured output."""
    if output_type is None:
        return False
    
    # Check for Pydantic model
    if hasattr(output_type, 'model_fields'):
        return True
    
    # Check for dataclass
    if hasattr(output_type, '__dataclass_fields__'):
        return True
    
    # Check for TypedDict
    if hasattr(output_type, '__annotations__') and hasattr(output_type, '__total__'):
        return True
    
    return False
