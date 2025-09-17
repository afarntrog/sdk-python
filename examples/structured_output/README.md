# Structured Output Examples

This directory contains comprehensive examples demonstrating the new structured output capabilities of the Strands SDK. These examples showcase various use cases, patterns, and best practices for getting type-safe, validated responses from AI models.

## Overview

The Strands SDK's structured output system allows you to:
- Get type-safe responses using Pydantic models
- Choose from multiple output strategies (tool-based, native, prompted)
- Integrate seamlessly with streaming and metrics
- Maintain backward compatibility with existing code

## Examples

### 1. Basic Weather Example (`basic_weather_example.py`)

**What it demonstrates:**
- Fundamental structured output usage
- Defining Pydantic models
- Agent configuration with output types
- Both sync and async interfaces

**Key concepts:**
- `output_type` parameter in agent calls
- `get_structured_output()` method
- Default output types in agent constructor
- Streaming with structured output

**Run it:**
```bash
python basic_weather_example.py
```

### 2. Multiple Output Types (`multiple_output_types.py`)

**What it demonstrates:**
- Using multiple output types in a single schema
- Conditional response handling
- Error and fallback response types
- Project management use case

**Key concepts:**
- `OutputSchema([Type1, Type2, Type3])`
- `isinstance()` for type checking
- Agent choosing appropriate response type
- Robust error handling patterns

**Run it:**
```bash
python multiple_output_types.py
```

### 3. Output Modes Comparison (`output_modes_comparison.py`)

**What it demonstrates:**
- Comparison of ToolOutput, NativeOutput, and PromptedOutput
- Performance characteristics
- Automatic fallback behavior
- When to use each mode

**Key concepts:**
- `ToolOutput()` - Function calling approach (default)
- `NativeOutput()` - Model's native capabilities
- `PromptedOutput()` - Prompt-based approach
- Trade-offs and selection criteria

**Run it:**
```bash
python output_modes_comparison.py
```

### 4. Real-World Data Extraction (`real_world_data_extraction.py`)

**What it demonstrates:**
- Practical data extraction scenarios
- Contact information parsing
- Invoice processing
- Customer feedback analysis
- Meeting notes structuring

**Key concepts:**
- Complex nested models
- Data validation with Pydantic
- Enum types for classifications
- Business process automation

**Run it:**
```bash
python real_world_data_extraction.py
```

## Getting Started

### Prerequisites

```bash
# Install the Strands SDK (with structured output support)
pip install strands-agents

# Required dependencies
pip install pydantic>=2.0
```

### Basic Setup

```python
from strands import Agent, ToolOutput
from pydantic import BaseModel, Field

# Define your data model
class MyModel(BaseModel):
    field1: str = Field(description="Description of field1")
    field2: int = Field(description="Description of field2")

# Create agent
agent = Agent(
    model_id="gpt-4o",
    output_type=MyModel,        # Optional default
    output_mode=ToolOutput()    # Optional mode selection
)

# Get structured output
result = agent("Your prompt here", output_type=MyModel)
data = result.get_structured_output(MyModel)
```

## Common Patterns

### 1. Single Output Type

```python
class WeatherReport(BaseModel):
    temperature: float
    conditions: str

result = agent("Weather in NYC", output_type=WeatherReport)
weather = result.get_structured_output(WeatherReport)
```

### 2. Multiple Output Types

```python
schema = OutputSchema([SuccessResponse, ErrorResponse])
result = agent("Process this request", output_schema=schema)

if isinstance(result.structured_output, SuccessResponse):
    # Handle success
elif isinstance(result.structured_output, ErrorResponse):
    # Handle error
```

### 3. Async Streaming

```python
async def stream_example():
    events = agent.stream_async(
        "Generate report",
        output_schema=OutputSchema([MyModel])
    )

    async for event in events:
        if 'result' in event:
            data = event['result'].get_structured_output(MyModel)
            break
```

### 4. Custom Validation

```python
class ValidatedModel(BaseModel):
    score: float

    @validator('score')
    def validate_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Score must be between 0 and 100')
        return v
```

## Best Practices

### 1. Model Design

- Use clear, descriptive field names
- Add comprehensive field descriptions
- Include validation where appropriate
- Use appropriate types (enums, unions, etc.)

```python
class GoodModel(BaseModel):
    """Clear description of what this model represents."""

    temperature: float = Field(
        description="Temperature in Fahrenheit",
        ge=-100,
        le=150
    )
    location: str = Field(description="City and state/country")
    conditions: WeatherCondition = Field(description="Weather conditions")
```

### 2. Output Mode Selection

- **Default to ToolOutput** for reliability
- **Use NativeOutput** when you need performance and know your model supports it
- **Use PromptedOutput** for legacy models or custom prompting strategies

### 3. Error Handling

```python
try:
    result = agent("Your prompt", output_type=MyModel)
    data = result.get_structured_output(MyModel)
except ValueError as e:
    # Handle missing or invalid structured output
    print(f"Structured output error: {e}")
```

### 4. Performance Optimization

- Reuse OutputSchema instances when possible
- Use simpler models for better performance
- Consider NativeOutput for high-throughput scenarios

## Migration from Legacy API

### Old (Deprecated)

```python
# These methods show deprecation warnings
weather = agent.structured_output(WeatherReport, "Weather in NYC")
weather = await agent.structured_output_async(WeatherReport, "Weather in NYC")
```

### New (Recommended)

```python
# Sync
result = agent("Weather in NYC", output_type=WeatherReport)
weather = result.get_structured_output(WeatherReport)

# Async
events = agent.stream_async("Weather in NYC", output_schema=OutputSchema([WeatherReport]))
async for event in events:
    if 'result' in event:
        weather = event['result'].get_structured_output(WeatherReport)
        break
```

## Troubleshooting

### Common Issues

1. **"No structured output available"**
   - Ensure you passed `output_type` or `output_schema`
   - Check that the model successfully generated structured output

2. **Validation errors**
   - Review your Pydantic model definition
   - Check for missing required fields
   - Verify field types and constraints

3. **Model not generating structured output**
   - Try being more specific in your prompt
   - Consider using a different output mode
   - Check model provider support

### Debug Mode

```python
import logging
logging.getLogger('strands.output').setLevel(logging.DEBUG)
```

## Performance Benchmarks

Run the output modes comparison example to see performance characteristics:

```bash
python output_modes_comparison.py
```

This will show you timing comparisons between different output modes with your specific model and setup.

## Contributing

When adding new examples:

1. Follow the existing pattern and structure
2. Include comprehensive docstrings and comments
3. Demonstrate clear use cases and concepts
4. Add error handling examples
5. Update this README with your new example

## Support

- [API Documentation](../../docs/structured_output_api.md)
- [Migration Guide](../../docs/structured_output_migration.md)
- [GitHub Issues](https://github.com/anthropics/strands/issues)

## License

These examples are part of the Strands SDK and follow the same license terms.