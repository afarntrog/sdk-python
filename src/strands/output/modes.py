"""Concrete implementations of output modes."""

import json
import logging
from typing import TYPE_CHECKING, Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .base import OutputMode

if TYPE_CHECKING:
    from ..models.model import Model
    from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ToolOutput(OutputMode):
    """Use function calling for structured output (DEFAULT).

    This is the most reliable approach across all model providers and ensures
    consistent behavior regardless of model capabilities. It converts Pydantic
    models to function tools that the model can call to return structured data.
    """

    def get_tool_specs(self, output_types: list[Type[BaseModel]]) -> list["ToolSpec"]:
        """Convert output types to tool specifications for function calling.

        Args:
            output_types: List of Pydantic model classes to convert

        Returns:
            List of tool specifications for function calling
        """
        from ..tools.structured_output import convert_pydantic_to_tool_spec

        tool_specs = []
        for output_type in output_types:
            try:
                tool_spec = convert_pydantic_to_tool_spec(output_type)
                tool_specs.append(tool_spec)
            except Exception as e:
                logger.error(f"Failed to convert {output_type.__name__} to tool spec: {e}")
                raise ValueError(f"Cannot convert {output_type.__name__} to tool specification") from e

        return tool_specs

    def extract_result(self, model_response: Any, expected_type: Type[T]) -> T:
        """Extract structured result from function call response.

        Args:
            model_response: Response from function call (should be dict or BaseModel)
            expected_type: Expected output type for validation

        Returns:
            Validated structured output instance

        Raises:
            ValueError: If response cannot be converted to expected type
        """
        try:
            # If already the correct type, return as-is
            if isinstance(model_response, expected_type):
                return model_response

            # If it's a dict, try to parse it
            if isinstance(model_response, dict):
                return expected_type(**model_response)

            # If it's another BaseModel, try to convert via dict
            if isinstance(model_response, BaseModel):
                return expected_type(**model_response.model_dump())

            # Try to parse as JSON if it's a string
            if isinstance(model_response, str):
                try:
                    data = json.loads(model_response)
                    return expected_type(**data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in model response: {e}") from e

            raise ValueError(f"Cannot convert response type {type(model_response)} to {expected_type.__name__}")

        except ValidationError as e:
            raise ValueError(f"Response validation failed for {expected_type.__name__}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to extract {expected_type.__name__} from response: {e}") from e

    def is_supported_by_model(self, model: "Model") -> bool:
        """Tool-based output is supported by all models that support function calling.

        All models in the Strands SDK support function calling, so this always returns True.

        Args:
            model: Model instance to check

        Returns:
            Always True for tool-based output
        """
        return True


class NativeOutput(OutputMode):
    """Use model's native structured output capabilities.

    Only use when explicitly requested and supported by the model.
    Falls back to ToolOutput if not supported.

    This mode leverages the model's built-in JSON schema support for
    structured output when available (e.g., OpenAI's structured outputs).
    """

    def get_tool_specs(self, output_types: list[Type[BaseModel]]) -> list["ToolSpec"]:
        """Return empty list as native output doesn't use function calling.

        Native output uses the model's built-in JSON schema support instead
        of function calling tools.

        Args:
            output_types: List of output types (used for schema generation)

        Returns:
            Empty list - native output doesn't use tools
        """
        return []

    def extract_result(self, model_response: Any, expected_type: Type[T]) -> T:
        """Extract structured result from native model response.

        Args:
            model_response: Raw response from model's native structured output
            expected_type: Expected output type for validation

        Returns:
            Validated structured output instance

        Raises:
            ValueError: If response cannot be converted to expected type
        """
        try:
            # Native output should already be structured, but validate anyway
            if isinstance(model_response, expected_type):
                return model_response

            if isinstance(model_response, dict):
                return expected_type(**model_response)

            if isinstance(model_response, str):
                # Parse JSON response from native structured output
                try:
                    data = json.loads(model_response)
                    return expected_type(**data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON from native output: {e}") from e

            raise ValueError(f"Unexpected native output type: {type(model_response)}")

        except ValidationError as e:
            raise ValueError(f"Native output validation failed for {expected_type.__name__}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to extract {expected_type.__name__} from native output: {e}") from e

    def is_supported_by_model(self, model: "Model") -> bool:
        """Check if model supports native structured output.

        Args:
            model: Model instance to check compatibility with

        Returns:
            True if the model supports native structured output
        """
        try:
            return model.supports_native_structured_output()
        except AttributeError:
            # Fallback for models that don't implement the new interface yet
            logger.warning(f"Model {model.__class__.__name__} doesn't implement supports_native_structured_output()")
            return False


class PromptedOutput(OutputMode):
    """Use prompting to guide output format.

    Only use when explicitly requested. Less reliable than tool-based approach
    but can work with models that have limited function calling support.

    This mode injects JSON schema information into the system prompt to guide
    the model toward producing structured output.
    """

    def __init__(self, template: str | None = None):
        """Initialize prompted output mode.

        Args:
            template: Custom template for schema injection. Must contain {schema} placeholder.
                     Defaults to a standard template if not provided.
        """
        self.template = template or (
            "Please respond with valid JSON that matches this schema exactly. "
            "Do not include any additional text or formatting outside the JSON.\n\n"
            "Schema: {schema}"
        )

        # Validate template has required placeholder
        if "{schema}" not in self.template:
            raise ValueError("Template must contain {schema} placeholder")

    def get_tool_specs(self, output_types: list[Type[BaseModel]]) -> list["ToolSpec"]:
        """Return empty list as prompted output doesn't use function calling.

        Prompted output injects schema into the system prompt instead of using tools.

        Args:
            output_types: List of output types (used for schema generation)

        Returns:
            Empty list - prompted output doesn't use tools
        """
        return []

    def extract_result(self, model_response: Any, expected_type: Type[T]) -> T:
        """Extract structured result from prompted model response.

        Args:
            model_response: Raw text response from model
            expected_type: Expected output type for validation

        Returns:
            Validated structured output instance

        Raises:
            ValueError: If response cannot be converted to expected type
        """
        try:
            # For prompted output, response is typically a string
            if isinstance(model_response, str):
                # Try to extract JSON from the response
                response_text = model_response.strip()

                # Look for JSON in the response (handle cases where model adds extra text)
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    # No JSON found, try parsing the entire response
                    try:
                        data = json.loads(response_text)
                    except json.JSONDecodeError:
                        raise ValueError(f"No valid JSON found in prompted response: {response_text}")
                else:
                    # Extract JSON portion
                    json_text = response_text[json_start:json_end]
                    try:
                        data = json.loads(json_text)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in prompted response: {e}") from e

                return expected_type(**data)

            # Handle cases where response is already structured
            if isinstance(model_response, dict):
                return expected_type(**model_response)

            if isinstance(model_response, expected_type):
                return model_response

            raise ValueError(f"Unexpected prompted output type: {type(model_response)}")

        except ValidationError as e:
            raise ValueError(f"Prompted output validation failed for {expected_type.__name__}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to extract {expected_type.__name__} from prompted output: {e}") from e

    def is_supported_by_model(self, model: "Model") -> bool:
        """Prompting-based output works with all models.

        All models can process prompts with schema information, so this
        always returns True.

        Args:
            model: Model instance to check

        Returns:
            Always True for prompted output
        """
        return True

    def get_schema_prompt(self, output_types: list[Type[BaseModel]]) -> str:
        """Generate schema prompt for the given output types.

        Args:
            output_types: List of output types to generate schema for

        Returns:
            Formatted prompt with schema information
        """
        if len(output_types) == 1:
            schema = output_types[0].model_json_schema()
        else:
            # For multiple types, create a union schema
            schemas = {f"{t.__name__}": t.model_json_schema() for t in output_types}
            schema = {
                "oneOf": [{"$ref": f"#/definitions/{name}"} for name in schemas.keys()],
                "definitions": schemas,
            }

        schema_json = json.dumps(schema, indent=2)
        return self.template.format(schema=schema_json)