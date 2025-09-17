"""Base classes for output type system."""

from abc import ABC, abstractmethod
from typing import Any, Type, Union, Optional, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from strands.models.model import Model
    from strands.tools.tool_spec import ToolSpec


class OutputMode(ABC):
    """Base class for different structured output modes."""

    @abstractmethod
    def get_tool_specs(self, output_types: list[Type[BaseModel]]) -> list["ToolSpec"]:
        """Convert output types to tool specifications.
        
        Args:
            output_types: List of Pydantic model types to convert
            
        Returns:
            List of tool specifications for the output types
        """
        pass

    @abstractmethod
    def extract_result(self, model_response: Any) -> Any:
        """Extract structured result from model response.
        
        Args:
            model_response: Raw response from the model
            
        Returns:
            Extracted structured output
        """
        pass

    @abstractmethod
    def is_supported_by_model(self, model: "Model") -> bool:
        """Check if this output mode is supported by the given model.
        
        Args:
            model: Model instance to check support for
            
        Returns:
            True if the model supports this output mode
        """
        pass


class OutputSchema:
    """Container for output type information and processing mode."""

    def __init__(
        self,
        types: Union[Type[BaseModel], list[Type[BaseModel]]],
        mode: Optional[OutputMode] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize output schema.
        
        Args:
            types: Pydantic model type(s) for structured output
            mode: Output mode to use (defaults to ToolOutput)
            name: Optional name for the output schema
            description: Optional description of the output schema
        """
        self.types = types if isinstance(types, list) else [types]
        if mode is None:
            from .modes import ToolOutput
            mode = ToolOutput()
        self.mode = mode
        self.name = name
        self.description = description

    @property
    def single_type(self) -> Type[BaseModel]:
        """Get single output type (for schemas with only one type)."""
        if len(self.types) != 1:
            raise ValueError(f"Expected single output type, got {len(self.types)}")
        return self.types[0]

    @property
    def is_single_type(self) -> bool:
        """Check if schema has exactly one output type."""
        return len(self.types) == 1
