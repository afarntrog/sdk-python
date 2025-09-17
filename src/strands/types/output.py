"""Type definitions for structured output system."""

from typing import Any, Dict, List, Optional, Type, Union, Protocol, runtime_checkable

from ..output.base import OutputMode, OutputSchema


@runtime_checkable
class StructuredOutputCapable(Protocol):
    """Protocol for models that support structured output."""
    
    def supports_native_structured_output(self) -> bool:
        """Check if the model supports native structured output."""
        ...
    
    def get_structured_output_config(self, output_type: Type) -> Dict[str, Any]:
        """Get model-specific configuration for structured output."""
        ...


# Type aliases for common output configurations
OutputType = Union[Type, None]
OutputModeType = Union[OutputMode, str, None]
OutputSchemaType = Union[OutputSchema, Dict[str, Any], None]
OutputTypeSpec = Union[Type, List[Type], OutputSchema, None]

# Common output mode names
OUTPUT_MODE_TOOL = "tool"
OUTPUT_MODE_NATIVE = "native" 
OUTPUT_MODE_PROMPTED = "prompted"
