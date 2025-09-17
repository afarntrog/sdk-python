"""Concrete output mode implementations."""

from typing import Any, Dict, List, Type

from .base import OutputMode


class ToolOutput(OutputMode):
    """Tool-based output mode using function calling."""
    
    def get_tool_specs(self, output_type: Type) -> List[Dict[str, Any]]:
        """Generate tool specifications for the output type."""
        from ..tools.structured_output import convert_pydantic_to_tool_spec
        
        if output_type is None:
            return []
        
        return [convert_pydantic_to_tool_spec(output_type)]
    
    def extract_result(self, response: Any, output_type: Type) -> Any:
        """Extract result from tool call response."""
        # For tool-based output, the result should be in the tool call
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                import json
                args = json.loads(tool_call.function.arguments)
                return output_type(**args) if output_type else args
        
        return None
    
    def is_supported_by_model(self, model: Any) -> bool:
        """Tool output is supported by all models."""
        return True


class NativeOutput(OutputMode):
    """Native structured output mode using model-specific features."""
    
    def get_tool_specs(self, output_type: Type) -> List[Dict[str, Any]]:
        """Native output doesn't use tool specs."""
        return []
    
    def extract_result(self, response: Any, output_type: Type) -> Any:
        """Extract result from native structured output."""
        # Implementation depends on model-specific response format
        if hasattr(response, 'structured_output'):
            return response.structured_output
        
        # Fallback to parsing response content
        if hasattr(response, 'content'):
            import json
            try:
                data = json.loads(response.content)
                return output_type(**data) if output_type else data
            except (json.JSONDecodeError, TypeError):
                pass
        
        return None
    
    def is_supported_by_model(self, model: Any) -> bool:
        """Check if model supports native structured output."""
        if hasattr(model, 'supports_native_structured_output'):
            return model.supports_native_structured_output()
        return False


class PromptedOutput(OutputMode):
    """Prompt-based output mode with custom templates."""
    
    def __init__(self, template: str = None):
        """Initialize with optional custom template.
        
        Args:
            template: Custom prompt template for structured output
        """
        self.template = template or (
            "Please respond with a JSON object that matches this schema: {schema}. "
            "Only return the JSON, no additional text."
        )
    
    def get_tool_specs(self, output_type: Type) -> List[Dict[str, Any]]:
        """Prompted output doesn't use tool specs."""
        return []
    
    def extract_result(self, response: Any, output_type: Type) -> Any:
        """Extract result from prompted response."""
        content = response.content if hasattr(response, 'content') else str(response)
        
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return output_type(**data) if output_type else data
            except (json.JSONDecodeError, TypeError):
                pass
        
        return None
    
    def is_supported_by_model(self, model: Any) -> bool:
        """Prompted output is supported by all models."""
        return True
    
    def get_prompt_template(self, output_type: Type) -> str:
        """Get the prompt template with schema information."""
        if output_type and hasattr(output_type, 'model_json_schema'):
            schema = output_type.model_json_schema()
            return self.template.format(schema=schema)
        return self.template
