"""Output type registry and resolution system."""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from .base import OutputMode, OutputSchema
from .modes import NativeOutput, PromptedOutput, ToolOutput

if TYPE_CHECKING:
    from ..models.model import Model
    from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)



class OutputRegistry:
    """Registry for managing output type specifications and caching.

    This registry handles:
    - Resolution of output type specifications
    - Caching of tool specifications
    - Validation of output schemas
    - Model compatibility checking
    """

    def __init__(self):
        """Initialize the output registry."""
        self._tool_spec_cache: Dict[str, list["ToolSpec"]] = {}
        self._schema_cache: Dict[str, OutputSchema] = {}
        self._model_compatibility_cache: Dict[str, Dict[str, bool]] = {}

    def resolve_output_schema(
        self,
        output_type: Optional[Type[BaseModel]],
        output_mode: Optional[OutputMode] = None,
        model: Optional["Model"] = None,
    ) -> Optional[OutputSchema]:
        """Resolve output type specification into OutputSchema.

        Args:
            output_type: Output type (Pydantic BaseModel class)
            output_mode: Optional output mode (defaults to ToolOutput)
            model: Model instance for compatibility checking

        Returns:
            Resolved OutputSchema or None if no output type specified

        Raises:
            ValueError: If output specification is invalid or incompatible
        """
        if not output_type:
            return None

        # Validate that output_type is a BaseModel subclass
        if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
            raise ValueError(f"output_type must be a Pydantic BaseModel subclass, got {type(output_type)}")

        # Create cache key for schema resolution
        cache_key = self._get_schema_cache_key(output_type, output_mode)

        # Check cache first
        if cache_key in self._schema_cache:
            cached_schema = self._schema_cache[cache_key]
            if model:
                self._validate_schema_compatibility(cached_schema, model)
            return cached_schema

        # Resolve output mode with model compatibility checking
        resolved_mode = self._resolve_output_mode(output_mode, model)

        # Create new schema with single type wrapped in list
        schema = OutputSchema(types=[output_type], mode=resolved_mode)

        # Cache the schema
        self._schema_cache[cache_key] = schema

        return schema

    def get_tool_specs(self, schema: OutputSchema) -> list["ToolSpec"]:
        """Get tool specifications for an output schema.

        Args:
            schema: Output schema to generate tools for

        Returns:
            List of tool specifications

        Raises:
            ValueError: If tool spec generation fails
        """
        # Create cache key
        cache_key = self._get_tool_spec_cache_key(schema)

        # Check cache first
        if cache_key in self._tool_spec_cache:
            return self._tool_spec_cache[cache_key]

        try:
            # Generate tool specs
            tool_specs = schema.mode.get_tool_specs(schema.types)

            # Cache the result
            self._tool_spec_cache[cache_key] = tool_specs

            return tool_specs

        except Exception as e:
            logger.error(f"Failed to generate tool specs for schema {schema}: {e}")
            raise ValueError(f"Cannot generate tool specifications: {e}") from e

    def validate_output_types(self, output_types: list[Type[BaseModel]]) -> None:
        """Validate that output types are valid Pydantic models.

        Args:
            output_types: List of output types to validate

        Raises:
            ValueError: If any output type is invalid
        """
        for output_type in output_types:
            if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
                raise ValueError(f"Output type {output_type} must be a Pydantic BaseModel subclass")

            # Validate that the model can generate a JSON schema
            try:
                output_type.model_json_schema()
            except Exception as e:
                raise ValueError(f"Cannot generate JSON schema for {output_type.__name__}: {e}") from e

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._tool_spec_cache.clear()
        self._schema_cache.clear()
        self._model_compatibility_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache sizes
        """
        return {
            "tool_spec_cache_size": len(self._tool_spec_cache),
            "schema_cache_size": len(self._schema_cache),
            "compatibility_cache_size": len(self._model_compatibility_cache),
        }

    def _resolve_output_mode(
        self, output_mode: Optional[OutputMode], model: Optional["Model"]
    ) -> OutputMode:
        """Resolve output mode with model compatibility checking.

        Args:
            output_mode: Requested output mode
            model: Model instance for compatibility checking

        Returns:
            Compatible output mode (may fallback to ToolOutput)
        """
        # Default to ToolOutput if no mode specified
        if not output_mode:
            return ToolOutput()

        # If no model provided, return as-is
        if not model:
            return output_mode

        # Check if mode is supported by model
        if self._is_mode_supported_by_model(output_mode, model):
            return output_mode

        # Handle fallback for unsupported modes
        if isinstance(output_mode, NativeOutput):
            logger.warning(
                f"Model {model.__class__.__name__} does not support native structured output. "
                "Falling back to tool-based approach."
            )
            return ToolOutput()

        # PromptedOutput and ToolOutput should always be supported
        logger.warning(
            f"Model {model.__class__.__name__} unexpectedly doesn't support {output_mode.__class__.__name__}. "
            "Falling back to tool-based approach."
        )
        return ToolOutput()

    def _validate_schema_compatibility(self, schema: OutputSchema, model: "Model") -> None:
        """Validate that schema is compatible with model.

        Args:
            schema: Output schema to validate
            model: Model to check compatibility with

        Raises:
            ValueError: If schema is incompatible with model
        """
        if not self._is_mode_supported_by_model(schema.mode, model):
            raise ValueError(
                f"Output mode {schema.mode.__class__.__name__} is not supported by "
                f"model {model.__class__.__name__}"
            )

        # Validate output types
        self.validate_output_types(schema.types)

    def _is_mode_supported_by_model(self, mode: OutputMode, model: "Model") -> bool:
        """Check if output mode is supported by model with caching.

        Args:
            mode: Output mode to check
            model: Model to check compatibility with

        Returns:
            True if mode is supported by model
        """
        model_key = model.__class__.__name__
        mode_key = mode.__class__.__name__

        # Check cache
        if model_key in self._model_compatibility_cache:
            if mode_key in self._model_compatibility_cache[model_key]:
                return self._model_compatibility_cache[model_key][mode_key]

        # Check compatibility
        try:
            is_supported = mode.is_supported_by_model(model)
        except Exception as e:
            logger.warning(f"Error checking {mode_key} support for {model_key}: {e}")
            is_supported = False

        # Cache result
        if model_key not in self._model_compatibility_cache:
            self._model_compatibility_cache[model_key] = {}
        self._model_compatibility_cache[model_key][mode_key] = is_supported

        return is_supported

    def _get_schema_cache_key(
        self, output_type: Type[BaseModel], output_mode: Optional[OutputMode]
    ) -> str:
        """Generate cache key for schema resolution.

        Args:
            output_type: Output type (BaseModel class)
            output_mode: Output mode

        Returns:
            Cache key string
        """
        type_key = output_type.__name__

        mode_key = output_mode.__class__.__name__ if output_mode else "ToolOutput"

        return f"{type_key}:{mode_key}"

    def _get_tool_spec_cache_key(self, schema: OutputSchema) -> str:
        """Generate cache key for tool spec caching.

        Args:
            schema: Output schema

        Returns:
            Cache key string
        """
        type_names = sorted([t.__name__ for t in schema.types])
        type_key = f"[{','.join(type_names)}]"
        mode_key = schema.mode.__class__.__name__

        # Include mode-specific parameters for more precise caching
        if isinstance(schema.mode, PromptedOutput):
            template_hash = hash(schema.mode.template)
            mode_key = f"{mode_key}:{template_hash}"

        return f"{type_key}:{mode_key}"


# Global registry instance
_global_registry: Optional[OutputRegistry] = None


def get_global_registry() -> OutputRegistry:
    """Get the global output registry instance.

    Returns:
        Global OutputRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = OutputRegistry()
    return _global_registry


@lru_cache(maxsize=128)
def resolve_output_schema(
    output_type_name: str,
    output_mode_name: str = "ToolOutput",
    model_name: Optional[str] = None,
) -> str:
    """Cached function for output schema resolution (for frequently used combinations).

    This is a convenience function for caching common schema resolutions.

    Args:
        output_type_name: Name of the output type
        output_mode_name: Name of the output mode
        model_name: Optional model name for compatibility

    Returns:
        Serialized schema information
    """
    # This is a simplified caching mechanism for common patterns
    # In practice, this would be used by the registry for frequently accessed schemas
    return f"{output_type_name}:{output_mode_name}:{model_name or 'any'}"


def clear_global_cache() -> None:
    """Clear the global registry cache."""
    registry = get_global_registry()
    registry.clear_cache()