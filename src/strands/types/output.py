"""Type definitions for structured output system."""

from typing import Type, TypeVar, Union

from pydantic import BaseModel

from ..output.base import OutputMode

# Type variable for generic structured output
T = TypeVar("T", bound=BaseModel)

# Type alias for output mode specifications
OutputModeSpec = Union[
    OutputMode,                   # Specific output mode instance
    Type[OutputMode],            # Output mode class (will be instantiated)
    str,                         # Output mode name (for string-based configuration)
]

__all__ = [
    "T",
    "OutputModeSpec",
]