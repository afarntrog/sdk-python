"""Strategy implementations for structured output across different model providers."""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ..models.model import Model
from ..types.content import Messages
from .exceptions import StructuredOutputError, StructuredOutputValidationError, StructuredOutputParsingError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputStrategy(ABC):
    """Abstract base class for structured output strategies."""
    
    @abstractmethod
    async def execute(
        self,
        model: Model,
        output_type: Type[T],
        messages: Messages,
        system_prompt: Optional[str] = None
    ) -> Optional[T]:
        """Execute structured output strategy.
        
        Args:
            model: The model instance to use.
            output_type: The Pydantic model class to parse into.
            messages: The conversation messages.
            system_prompt: Optional system prompt.
            
        Returns:
            Parsed structured output or None if parsing failed.
            
        Raises:
            StructuredOutputError: If the strategy execution fails.
        """
        pass


class NativeStrategy(StructuredOutputStrategy):
    """Strategy for providers with native structured output support (OpenAI, LiteLLM)."""
    
    async def execute(
        self,
        model: Model,
        output_type: Type[T],
        messages: Messages,
        system_prompt: Optional[str] = None
    ) -> Optional[T]:
        """Use provider's native structured output API."""
        provider_name = model.__class__.__name__
        try:
            # Validate that the model has the structured_output method
            if not hasattr(model, 'structured_output'):
                raise StructuredOutputError(
                    f"Model {provider_name} does not support structured_output method",
                    provider=provider_name,
                    strategy="native"
                )
            
            # Use the existing model.structured_output() method for native providers
            # This method uses the provider's native API (e.g., OpenAI's beta.chat.completions.parse)
            events = model.structured_output(output_type, messages, system_prompt)
            async for event in events:
                if "output" in event and event["output"] is not None:
                    # Validate that the output is of the expected type
                    output = event["output"]
                    if isinstance(output, output_type):
                        return output
                    else:
                        raise StructuredOutputValidationError(
                            f"Output type mismatch: expected {output_type.__name__}, got {type(output).__name__}",
                            provider=provider_name,
                            strategy="native"
                        )
            return None
        except ValidationError as e:
            raise StructuredOutputValidationError(
                f"Native strategy validation failed: {e}", 
                provider=provider_name, 
                strategy="native"
            ) from e
        except Exception as e:
            raise StructuredOutputError(
                f"Native strategy failed: {e}", 
                provider=provider_name, 
                strategy="native"
            ) from e


class JsonSchemaStrategy(StructuredOutputStrategy):
    """Strategy for providers with JSON schema support (Ollama, LlamaCpp)."""
    
    async def execute(
        self,
        model: Model,
        output_type: Type[T],
        messages: Messages,
        system_prompt: Optional[str] = None
    ) -> Optional[T]:
        """Use JSON schema format parameter."""
        provider_name = model.__class__.__name__
        try:
            # Validate that the model has the structured_output method
            if not hasattr(model, 'structured_output'):
                raise StructuredOutputError(
                    f"Model {provider_name} does not support structured_output method",
                    provider=provider_name,
                    strategy="json_schema"
                )
            
            # Validate that the output_type has model_json_schema method
            if not hasattr(output_type, 'model_json_schema'):
                raise StructuredOutputError(
                    f"Output type {output_type.__name__} does not support model_json_schema method",
                    provider=provider_name,
                    strategy="json_schema"
                )
            
            # Use the existing model.structured_output() method for JSON schema providers
            # This method uses the format parameter with model_json_schema()
            events = model.structured_output(output_type, messages, system_prompt)
            async for event in events:
                if "output" in event and event["output"] is not None:
                    # Validate that the output is of the expected type
                    output = event["output"]
                    if isinstance(output, output_type):
                        return output
                    else:
                        raise StructuredOutputValidationError(
                            f"Output type mismatch: expected {output_type.__name__}, got {type(output).__name__}",
                            provider=provider_name,
                            strategy="json_schema"
                        )
            return None
        except ValidationError as e:
            raise StructuredOutputValidationError(
                f"JSON schema strategy validation failed: {e}", 
                provider=provider_name, 
                strategy="json_schema"
            ) from e
        except Exception as e:
            raise StructuredOutputError(
                f"JSON schema strategy failed: {e}", 
                provider=provider_name, 
                strategy="json_schema"
            ) from e


class ToolCallingStrategy(StructuredOutputStrategy):
    """Strategy for providers using tool calling mechanism (Bedrock, Anthropic)."""
    
    async def execute(
        self,
        model: Model,
        output_type: Type[T],
        messages: Messages,
        system_prompt: Optional[str] = None
    ) -> Optional[T]:
        """Use tool calling mechanism with improved error handling."""
        provider_name = model.__class__.__name__
        try:
            # Validate that the model has required methods for tool calling
            if not hasattr(model, 'structured_output'):
                raise StructuredOutputError(
                    f"Model {provider_name} does not support structured_output method",
                    provider=provider_name,
                    strategy="tool_calling"
                )
            
            if not hasattr(model, 'stream'):
                raise StructuredOutputError(
                    f"Model {provider_name} does not support stream method required for tool calling",
                    provider=provider_name,
                    strategy="tool_calling"
                )
            
            # Validate that the output_type is a proper Pydantic model
            if not issubclass(output_type, BaseModel):
                raise StructuredOutputError(
                    f"Output type {output_type.__name__} must be a Pydantic BaseModel",
                    provider=provider_name,
                    strategy="tool_calling"
                )
            
            # Use the existing model.structured_output() method for tool-based providers
            # This method converts Pydantic model to tool spec and forces tool usage
            events = model.structured_output(output_type, messages, system_prompt)
            
            output_found = False
            async for event in events:
                if "output" in event and event["output"] is not None:
                    output_found = True
                    output = event["output"]
                    
                    # Validate that the output is of the expected type
                    if isinstance(output, output_type):
                        return output
                    else:
                        # Try to create the model from the output if it's a dict
                        if isinstance(output, dict):
                            try:
                                return output_type(**output)
                            except ValidationError as ve:
                                raise StructuredOutputValidationError(
                                    f"Failed to validate tool output as {output_type.__name__}: {ve}",
                                    provider=provider_name,
                                    strategy="tool_calling"
                                ) from ve
                        else:
                            raise StructuredOutputValidationError(
                                f"Tool output type mismatch: expected {output_type.__name__}, got {type(output).__name__}",
                                provider=provider_name,
                                strategy="tool_calling"
                            )
            
            if not output_found:
                raise StructuredOutputError(
                    "No tool output found in response - tool calling may have failed",
                    provider=provider_name,
                    strategy="tool_calling"
                )
            
            return None
            
        except ValidationError as e:
            raise StructuredOutputValidationError(
                f"Tool calling strategy validation failed: {e}", 
                provider=provider_name, 
                strategy="tool_calling"
            ) from e
        except Exception as e:
            # Check for common tool calling failure patterns
            error_msg = str(e).lower()
            if "tool_use" in error_msg or "tool use" in error_msg:
                raise StructuredOutputError(
                    f"Tool calling mechanism failed: {e}",
                    provider=provider_name,
                    strategy="tool_calling"
                ) from e
            elif "stop_reason" in error_msg:
                raise StructuredOutputError(
                    f"Model stopped without using tool: {e}",
                    provider=provider_name,
                    strategy="tool_calling"
                ) from e
            else:
                raise StructuredOutputError(
                    f"Tool calling strategy failed: {e}", 
                    provider=provider_name, 
                    strategy="tool_calling"
                ) from e


class PromptBasedStrategy(StructuredOutputStrategy):
    """Universal fallback strategy using prompt engineering."""
    
    def __init__(self):
        """Initialize the prompt-based strategy with templates."""
        self.structured_output_prompt = """You must respond with valid JSON that matches this exact schema:

{json_schema}

Requirements:
- Response must be valid JSON only
- No additional text before or after the JSON
- All required fields must be present
- Follow the exact field names and types specified

User request: {user_prompt}

JSON Response:"""

        self.structured_output_prompt_with_examples = """You must respond with valid JSON matching this schema:

Schema:
{json_schema}

Example valid response:
{example_json}

Rules:
1. Return ONLY valid JSON, no other text
2. Include all required fields
3. Use exact field names from schema
4. Follow data types specified

User request: {user_prompt}

JSON:"""
    
    async def execute(
        self,
        model: Model,
        output_type: Type[T],
        messages: Messages,
        system_prompt: Optional[str] = None
    ) -> Optional[T]:
        """Universal fallback using prompt engineering."""
        provider_name = model.__class__.__name__
        try:
            # Validate that the model has stream method
            if not hasattr(model, 'stream'):
                raise StructuredOutputError(
                    f"Model {provider_name} does not support stream method",
                    provider=provider_name,
                    strategy="prompt_based"
                )
            
            # Generate prompt components
            schema_str = self._pydantic_to_prompt_schema(output_type)
            example_str = self._generate_example_json(output_type)
            user_prompt = self._extract_user_prompt(messages)
            
            # Try with examples first, then fallback to basic prompt
            for attempt, use_examples in enumerate([True, False], 1):
                try:
                    if use_examples:
                        structured_prompt = self.structured_output_prompt_with_examples.format(
                            json_schema=schema_str,
                            example_json=example_str,
                            user_prompt=user_prompt
                        )
                    else:
                        structured_prompt = self.structured_output_prompt.format(
                            json_schema=schema_str,
                            user_prompt=user_prompt
                        )
                    
                    # Create new messages with structured prompt
                    structured_messages = self._create_structured_messages(messages, structured_prompt)
                    
                    # Get response using regular streaming
                    response_chunks = model.stream(structured_messages, system_prompt=system_prompt)
                    full_response = await self._collect_full_response(response_chunks)
                    
                    # Extract and parse JSON
                    json_data = self._extract_json_from_response(full_response)
                    if json_data:
                        try:
                            return output_type(**json_data)
                        except ValidationError as ve:
                            if attempt == 1:  # Try again with simpler prompt
                                logger.warning(f"Validation failed on attempt {attempt}, trying simpler prompt: {ve}")
                                continue
                            else:
                                raise StructuredOutputValidationError(
                                    f"Validation failed: {ve}",
                                    provider=provider_name,
                                    strategy="prompt_based"
                                ) from ve
                    else:
                        if attempt == 1:  # Try again with simpler prompt
                            logger.warning(f"JSON extraction failed on attempt {attempt}, trying simpler prompt")
                            continue
                        else:
                            raise StructuredOutputParsingError(
                                "Failed to extract valid JSON from response",
                                provider=provider_name,
                                strategy="prompt_based"
                            )
                            
                except Exception as e:
                    if attempt == 1:  # Try again with simpler prompt
                        logger.warning(f"Attempt {attempt} failed, trying simpler prompt: {e}")
                        continue
                    else:
                        raise e
            
            return None
            
        except ValidationError as e:
            raise StructuredOutputValidationError(
                f"Prompt-based strategy validation failed: {e}", 
                provider=provider_name, 
                strategy="prompt_based"
            ) from e
        except Exception as e:
            raise StructuredOutputError(
                f"Prompt-based strategy failed: {e}", 
                provider=provider_name, 
                strategy="prompt_based"
            ) from e
    
    def _pydantic_to_prompt_schema(self, model: Type[BaseModel]) -> str:
        """Convert Pydantic model to human-readable schema for prompts."""
        try:
            schema = model.model_json_schema()
            
            # Simplify schema for prompt clarity
            simplified = {
                "type": "object",
                "properties": {},
                "required": schema.get("required", [])
            }
            
            for prop_name, prop_def in schema.get("properties", {}).items():
                simplified["properties"][prop_name] = {
                    "type": prop_def.get("type", "string"),
                    "description": prop_def.get("description", "")
                }
            
            return json.dumps(simplified, indent=2)
        except Exception as e:
            # Fallback to basic schema
            return json.dumps({
                "type": "object",
                "properties": {field: {"type": "string"} for field in model.model_fields.keys()},
                "required": list(model.model_fields.keys())
            }, indent=2)
    
    def _generate_example_json(self, model: Type[BaseModel]) -> str:
        """Generate example JSON for the prompt."""
        try:
            example_data = {}
            schema = model.model_json_schema()
            
            for prop_name, prop_def in schema.get("properties", {}).items():
                prop_type = prop_def.get("type", "string")
                if prop_type == "string":
                    example_data[prop_name] = f"example_{prop_name}"
                elif prop_type == "integer":
                    example_data[prop_name] = 42
                elif prop_type == "number":
                    example_data[prop_name] = 3.14
                elif prop_type == "boolean":
                    example_data[prop_name] = True
                elif prop_type == "array":
                    example_data[prop_name] = ["example_item"]
                else:
                    example_data[prop_name] = f"example_{prop_name}"
            
            return json.dumps(example_data, indent=2)
        except Exception:
            # Fallback to basic example
            return json.dumps({field: f"example_{field}" for field in model.model_fields.keys()}, indent=2)
    
    def _extract_user_prompt(self, messages: Messages) -> str:
        """Extract the user's prompt from messages."""
        try:
            # Find the last user message
            for message in reversed(messages):
                if message.get("role") == "user":
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("text"):
                                return item["text"]
                    elif isinstance(content, str):
                        return content
            return "Generate the requested data"
        except Exception:
            return "Generate the requested data"
    
    def _create_structured_messages(self, original_messages: Messages, structured_prompt: str) -> Messages:
        """Create new messages with structured prompt."""
        return [{"role": "user", "content": [{"text": structured_prompt}]}]
    
    async def _collect_full_response(self, response_chunks) -> str:
        """Collect full response from streaming chunks."""
        full_text = ""
        try:
            async for chunk in response_chunks:
                if "data" in chunk:
                    full_text += chunk["data"]
                elif "text" in chunk:
                    full_text += chunk["text"]
                # Handle different chunk formats from various providers
                elif isinstance(chunk, dict):
                    for key in ["content", "message", "response"]:
                        if key in chunk and isinstance(chunk[key], str):
                            full_text += chunk[key]
        except Exception as e:
            logger.warning(f"Error collecting response chunks: {e}")
        
        return full_text.strip()
    
    def _extract_json_from_response(self, response_text: str) -> Optional[dict]:
        """Extract JSON from model response with multiple fallback strategies."""
        if not response_text:
            return None
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find JSON block markers
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # ```json {...} ```
            r'```\s*(\{.*?\})\s*```',      # ``` {...} ```
            r'(\{.*?\})',                  # First {...} block
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Clean and retry
        cleaned = self._clean_json_response(response_text)
        if cleaned:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _clean_json_response(self, text: str) -> str:
        """Clean common JSON formatting issues."""
        if not text:
            return ""
        
        # Remove common prefixes/suffixes
        text = re.sub(r'^[^{]*', '', text)  # Remove text before first {
        text = re.sub(r'[^}]*$', '', text)  # Remove text after last }
        
        # Fix common issues
        text = text.replace('```json', '').replace('```', '')
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
