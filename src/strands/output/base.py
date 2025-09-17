"""Base classes for structured output system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union


class OutputMode(ABC):
    """Abstract base class for output modes."""
    
    @abstractmethod
    def get_tool_specs(self, output_type: Type) -> List[Dict[str, Any]]:
        """Generate tool specifications for the output type.
        
        Args:
            output_type: The Pydantic model or type to generate specs for
            
        Returns:
            List of tool specifications
        """
        pass
    
    @abstractmethod
    def extract_result(self, response: Any, output_type: Type) -> Any:
        """Extract structured result from model response.
        
        Args:
            response: The model response
            output_type: Expected output type
            
        Returns:
            Extracted and validated result
        """
        pass
    
    @abstractmethod
    def is_supported_by_model(self, model: Any) -> bool:
        """Check if this output mode is supported by the given model.
        
        Args:
            model: The model instance to check
            
        Returns:
            True if supported, False otherwise
        """
        pass


class OutputSchema:
    """Container for output configuration."""
    
    def __init__(
        self,
        output_type: Optional[Type] = None,
        mode: Optional[OutputMode] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Initialize output schema.
        
        Args:
            output_type: The expected output type (Pydantic model)
            mode: The output mode to use (defaults to ToolOutput)
            name: Optional name for the output
            description: Optional description
        """
        from .modes import ToolOutput  # Import here to avoid circular imports
        
        self.output_type = output_type
        self.mode = mode or ToolOutput()
        self.name = name
        self.description = description
    
    def __repr__(self) -> str:
        return f"OutputSchema(type={self.output_type}, mode={self.mode.__class__.__name__})"
