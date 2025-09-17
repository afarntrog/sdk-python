"""Base classes for the structured output system."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..models.model import Model
    from ..types.tools import ToolSpec

T = TypeVar("T", bound=BaseModel)


class OutputMode(ABC):
    """Abstract base class for different structured output modes.

    Output modes define how structured output is obtained from the model:
    - ToolOutput: Uses function calling (default, most reliable)
    - NativeOutput: Uses model's native structured output capabilities
    - PromptedOutput: Uses prompting to guide output format
    """

    @abstractmethod
    def get_tool_specs(self, output_types: list[Type[BaseModel]]) -> list["ToolSpec"]:
        """Convert output types to tool specifications.

        Args:
            output_types: List of Pydantic model classes to convert

        Returns:
            List of tool specifications for the model to use
        """
        pass

    @abstractmethod
    def extract_result(self, model_response: Any, expected_type: Type[T]) -> T:
        """Extract structured result from model response.

        Args:
            model_response: Raw response from the model
            expected_type: Expected output type for validation

        Returns:
            Validated structured output instance

        Raises:
            ValueError: If response cannot be converted to expected type
        """
        pass

    @abstractmethod
    def is_supported_by_model(self, model: "Model") -> bool:
        """Check if this output mode is supported by the given model.

        Args:
            model: Model instance to check compatibility with

        Returns:
            True if the model supports this output mode
        """
        pass


class OutputSchema:
    """Container for output type information and processing mode.

    This class encapsulates the structured output configuration including
    the output types, processing mode, and metadata.
    """

    def __init__(
        self,
        types: Union[Type[T], list[Type[BaseModel]]],
        mode: Optional[OutputMode] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize output schema.

        Args:
            types: Single output type or list of possible output types
            mode: Output processing mode (defaults to ToolOutput)
            name: Optional name for the structured output
            description: Optional description for the structured output
        """
        from .modes import ToolOutput  # Import here to avoid circular imports

        self.types = types if isinstance(types, list) else [types]
        self.mode = mode or ToolOutput()  # Default to tool-based approach
        self.name = name
        self.description = description

    @property
    def is_single_type(self) -> bool:
        """Check if this schema has a single output type."""
        return len(self.types) == 1

    @property
    def primary_type(self) -> Type[BaseModel]:
        """Get the primary (first) output type.

        Returns:
            The first output type in the list

        Raises:
            ValueError: If no output types are defined
        """
        if not self.types:
            raise ValueError("No output types defined in schema")
        return self.types[0]

    def get_effective_name(self) -> str:
        """Get the effective name for this output schema.

        Returns:
            Explicit name if provided, otherwise the primary type name
        """
        if self.name:
            return self.name
        return self.primary_type.__name__

    def get_effective_description(self) -> str:
        """Get the effective description for this output schema.

        Returns:
            Explicit description if provided, otherwise derived from primary type
        """
        if self.description:
            return self.description

        primary_type = self.primary_type
        if primary_type.__doc__:
            return primary_type.__doc__.strip()

        return f"{primary_type.__name__} structured output"

    def __repr__(self) -> str:
        """String representation of the output schema."""
        type_names = [t.__name__ for t in self.types]
        mode_name = self.mode.__class__.__name__
        return f"OutputSchema(types={type_names}, mode={mode_name})"