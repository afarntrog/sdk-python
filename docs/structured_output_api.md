# Structured Output API Documentation

## Overview

The Strands SDK provides a powerful and intuitive structured output system that allows you to get type-safe, validated responses from AI models. This system is inspired by PydanticAI and integrates seamlessly with the main agent event loop for full metrics collection and streaming support.

## Key Features

- **Intuitive API**: Pass output types directly to agent calls
- **Multiple Output Modes**: Tool-based, native, and prompted approaches
- **Automatic Fallback**: Graceful degradation when models don't support native structured output
- **Full Integration**: Works with streaming, metrics, hooks, and all existing SDK features
- **Type Safety**: Runtime validation with Pydantic models
- **Backward Compatibility**: Existing methods continue to work with deprecation warnings

## Quick Start

```python
from strands import Agent, ToolOutput
from pydantic import BaseModel

class WeatherReport(BaseModel):
    location: str
    temperature: float
    conditions: str
    humidity: int

# Create agent with structured output
agent = Agent(
    model_id="gpt-4o",
    output_type=WeatherReport  # Optional: set default output type
)

# Get structured output
result = agent("What's the weather like in San Francisco?", output_type=WeatherReport)
weather = result.get_structured_output(WeatherReport)

print(f"Location: {weather.location}")
print(f"Temperature: {weather.temperature}°F")
print(f"Conditions: {weather.conditions}")
```

## Core Classes

### OutputSchema

Container class that holds output type information and configuration.

```python
from strands.output import OutputSchema, ToolOutput

# Single output type
schema = OutputSchema([WeatherReport])

# Multiple output types (agent can choose which to use)
schema = OutputSchema([WeatherReport, NewsUpdate, StockPrice])

# Custom output mode
schema = OutputSchema([WeatherReport], mode=ToolOutput())
```

**Parameters:**
- `types: List[Type[BaseModel]]` - List of Pydantic model classes for structured output
- `mode: Optional[OutputMode]` - Output mode strategy (defaults to ToolOutput)
- `name: Optional[str]` - Human-readable name for the output schema
- `description: Optional[str]` - Description of what this output represents

### OutputMode Classes

#### ToolOutput (Default)

Uses function calling to generate structured output. Most reliable across all model providers.

```python
from strands.output import ToolOutput

mode = ToolOutput()
agent = Agent(model_id="gpt-4o", output_mode=mode)
```

**Advantages:**
- Works with all model providers
- Most reliable and consistent
- Supports complex nested structures
- Built-in validation

#### NativeOutput

Uses model provider's native structured output capabilities when available.

```python
from strands.output import NativeOutput

mode = NativeOutput()
agent = Agent(model_id="gpt-4o", output_mode=mode)  # Falls back to ToolOutput if not supported
```

**Advantages:**
- Potentially faster (no tool call overhead)
- May have better adherence to schema
- Direct integration with model capabilities

**Limitations:**
- Only supported by some models (OpenAI GPT-4o, some LiteLLM models)
- Automatically falls back to ToolOutput if not supported

#### PromptedOutput

Uses carefully crafted prompts to guide the model toward structured output.

```python
from strands.output import PromptedOutput

mode = PromptedOutput(
    template="Please respond with a JSON object matching this schema: {schema}"
)
agent = Agent(model_id="claude-3", output_mode=mode)
```

**Parameters:**
- `template: str` - Template string with `{schema}` placeholder for JSON schema insertion

**Use Cases:**
- Models without function calling or native structured output
- When you want more control over the prompting strategy
- Legacy model support

## Agent Integration

### Constructor Parameters

```python
agent = Agent(
    model_id="gpt-4o",
    output_type=MyModel,           # Default output type for all calls
    output_mode=ToolOutput(),      # Default output mode strategy
    # ... other agent parameters
)
```

### Call-time Overrides

```python
# Override output type for specific calls
result = agent("Generate a report", output_type=DifferentModel)

# Override both type and mode
result = agent(
    "Generate a report",
    output_type=DifferentModel,
    output_mode=NativeOutput()
)
```

### AgentResult Methods

```python
class AgentResult:
    structured_output: Optional[Any]  # The parsed structured output

    def get_structured_output(self, output_type: Type[T]) -> T:
        """Get structured output with type validation."""
        # Returns the structured output cast to the expected type
        # Raises ValueError if no structured output or type mismatch
```

## Streaming Support

Structured output works seamlessly with the streaming interface:

```python
async def stream_with_structured_output():
    events = agent.stream_async(
        "Generate weather report",
        output_schema=OutputSchema([WeatherReport])
    )

    async for event in events:
        if hasattr(event, 'get') and 'structured_output' in event:
            # Structured output event
            output = event['structured_output']
            output_type = event['output_type']
            print(f"Got {output_type}: {output}")
        elif hasattr(event, 'get') and 'result' in event:
            # Final result
            result = event['result']
            if result.structured_output:
                weather = result.get_structured_output(WeatherReport)
                print(f"Final weather: {weather}")
```

## Model Provider Support

| Provider | ToolOutput | NativeOutput | PromptedOutput |
|----------|------------|--------------|----------------|
| OpenAI | ✅ | ✅ | ✅ |
| Bedrock | ✅ | ❌ | ✅ |
| Anthropic | ✅ | ❌ | ✅ |
| LiteLLM | ✅ | ⚠️* | ✅ |
| Ollama | ✅ | ❌ | ✅ |
| Others | ✅ | ❌ | ✅ |

*LiteLLM: Depends on underlying model support

## Advanced Usage

### Multiple Output Types

```python
from typing import Union

class WeatherReport(BaseModel):
    location: str
    temperature: float

class ErrorResponse(BaseModel):
    error: str
    code: int

# Agent can choose which type to return
schema = OutputSchema([WeatherReport, ErrorResponse])
result = agent("What's the weather?", output_schema=schema)

# Handle different types
if result.structured_output:
    if isinstance(result.structured_output, WeatherReport):
        print(f"Weather: {result.structured_output.temperature}°F")
    elif isinstance(result.structured_output, ErrorResponse):
        print(f"Error: {result.structured_output.error}")
```

### Custom Validation

```python
from pydantic import BaseModel, validator

class ValidatedReport(BaseModel):
    temperature: float
    location: str

    @validator('temperature')
    def validate_temperature(cls, v):
        if v < -100 or v > 150:
            raise ValueError('Temperature out of realistic range')
        return v

# Validation happens automatically
result = agent("Weather report", output_type=ValidatedReport)
report = result.get_structured_output(ValidatedReport)  # Raises if validation fails
```

### Nested Models

```python
class Location(BaseModel):
    city: str
    state: str
    country: str
    coordinates: Optional[Tuple[float, float]] = None

class DetailedWeather(BaseModel):
    location: Location
    current: WeatherConditions
    forecast: List[WeatherConditions]
    metadata: WeatherMetadata

# Works seamlessly with complex nested structures
result = agent("Detailed weather for NYC", output_type=DetailedWeather)
weather = result.get_structured_output(DetailedWeather)
```

## Error Handling

```python
from strands.types.exceptions import ValidationException

try:
    result = agent("Generate report", output_type=MyModel)
    data = result.get_structured_output(MyModel)
except ValueError as e:
    # No structured output available or type mismatch
    print(f"Structured output error: {e}")
except ValidationException as e:
    # Model returned invalid data
    print(f"Validation error: {e}")
```

## Migration from Legacy API

### Old API (Deprecated)

```python
# Deprecated - will show warnings
weather = agent.structured_output(WeatherReport, "What's the weather?")
weather = await agent.structured_output_async(WeatherReport, "What's the weather?")
```

### New API

```python
# Recommended approach
result = agent("What's the weather?", output_type=WeatherReport)
weather = result.get_structured_output(WeatherReport)

# For async
events = agent.stream_async("What's the weather?", output_schema=OutputSchema([WeatherReport]))
async for event in events:
    if 'result' in event:
        weather = event['result'].get_structured_output(WeatherReport)
```

## Best Practices

### 1. Choose the Right Output Mode

- **Use ToolOutput (default)** for maximum reliability and compatibility
- **Use NativeOutput** only when you need the performance benefit and know your model supports it
- **Use PromptedOutput** for legacy models or when you need custom prompting

### 2. Design Clear Schemas

```python
class GoodSchema(BaseModel):
    """Clear, well-documented schema with validation."""

    temperature: float = Field(description="Temperature in Fahrenheit")
    location: str = Field(description="City and state/country")
    conditions: str = Field(description="Brief weather description")

    @validator('temperature')
    def validate_temp(cls, v):
        if not -100 <= v <= 150:
            raise ValueError('Invalid temperature')
        return v
```

### 3. Handle Multiple Output Types Strategically

```python
# Use Union types for related outputs
class SuccessResponse(BaseModel):
    data: WeatherReport
    status: str = "success"

class ErrorResponse(BaseModel):
    error: str
    status: str = "error"

schema = OutputSchema([SuccessResponse, ErrorResponse])
```

### 4. Use Type Hints

```python
def process_weather(agent: Agent) -> WeatherReport:
    result = agent("Get weather", output_type=WeatherReport)
    return result.get_structured_output(WeatherReport)
```

## Performance Considerations

- **ToolOutput**: Slight overhead from function call, but highly reliable
- **NativeOutput**: Potentially faster, but limited model support
- **Schema Complexity**: More complex schemas may require more tokens/processing
- **Caching**: Output schemas are cached for performance across calls

## Troubleshooting

### Common Issues

1. **"No structured output available"**
   - Check that you passed `output_type` or `output_schema`
   - Verify the model successfully called the structured output tool

2. **Validation Errors**
   - Review your Pydantic model definition
   - Check for required fields that might be missing
   - Validate your field types and constraints

3. **Model Not Generating Structured Output**
   - Try being more specific in your prompt
   - Consider using a different output mode
   - Check model provider support

### Debug Mode

```python
import logging
logging.getLogger('strands.output').setLevel(logging.DEBUG)

# Now you'll see detailed logs about output mode selection, tool registration, etc.
```

## API Reference

For complete API reference, see the individual class documentation:

- [`Agent`](./agent.md)
- [`OutputSchema`](./output_schema.md)
- [`OutputMode`](./output_mode.md)
- [`AgentResult`](./agent_result.md)