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
    def get_tool_specs(self, output_type: Type[BaseModel]) -> list["ToolSpec"]:
        """Convert output type to tool specifications.
        
        Args:
            output_type: Pydantic model type to convert
            
        Returns:
            List of tool specifications for the output type
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
        type: Type[BaseModel],
        mode: Optional[OutputMode] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize output schema.
        
        Args:
            type: Pydantic model type for structured output
            mode: Output mode to use (defaults to ToolMode)
            name: Optional name for the output schema
            description: Optional description of the output schema
        """
        self.type = type
        if mode is None:
            from .modes import ToolMode
            mode = ToolMode()
        self.mode = mode
        self.name = name
        self.description = description
