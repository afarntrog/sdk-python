"""Type definitions for structured output system."""

from typing import Union, Type, TypeVar
from pydantic import BaseModel

from ..output.base import OutputSchema, OutputMode

# Type variable for generic structured output
T = TypeVar("T", bound=BaseModel)

# Type alias for output type specifications that can be passed to agents
OutputTypeSpec = Union[
    Type[BaseModel],              # Single output type
    list[Type[BaseModel]],        # Multiple possible output types
    OutputSchema,                 # Complete output schema with mode
]

# Type alias for output mode specifications
OutputModeSpec = Union[
    OutputMode,                   # Specific output mode instance
    Type[OutputMode],            # Output mode class (will be instantiated)
    str,                         # Output mode name (for string-based configuration)
]

__all__ = [
    "T",
    "OutputTypeSpec",
    "OutputModeSpec",
]