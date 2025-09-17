# Structured Output Migration Guide

This guide helps you migrate from the legacy `structured_output()` methods to the new unified structured output system in the Strands SDK.

## Overview

The new structured output system provides:
- **Better Integration**: Full integration with the main event loop for metrics and streaming
- **More Intuitive API**: Pass output types directly to agent calls
- **Multiple Output Modes**: Choose from tool-based, native, or prompted approaches
- **Type Safety**: Enhanced runtime validation and type checking
- **Future-Proof**: Built for extensibility and performance

## Quick Migration Reference

| Legacy API | New API |
|------------|---------|
| `agent.structured_output(Model, prompt)` | `agent(prompt, output_type=Model).get_structured_output(Model)` |
| `await agent.structured_output_async(Model, prompt)` | `await agent.stream_async(prompt, output_schema=OutputSchema([Model]))` |

## Migration Steps

### Step 1: Update Imports

**Before:**
```python
from strands import Agent
```

**After:**
```python
from strands import Agent, ToolOutput
from strands.output import OutputSchema, NativeOutput, PromptedOutput
```

### Step 2: Replace Method Calls

#### Sync Usage

**Before (Deprecated):**
```python
agent = Agent(model_id="gpt-4o")

# Direct method call
weather = agent.structured_output(WeatherReport, "What's the weather in NYC?")
print(f"Temperature: {weather.temperature}")
```

**After (Recommended):**
```python
agent = Agent(
    model_id="gpt-4o",
    output_type=WeatherReport,  # Optional: set as default
    output_mode=ToolOutput()    # Optional: choose output strategy
)

# New unified interface
result = agent("What's the weather in NYC?", output_type=WeatherReport)
weather = result.get_structured_output(WeatherReport)
print(f"Temperature: {weather.temperature}")
```

#### Async Usage

**Before (Deprecated):**
```python
async def get_weather():
    weather = await agent.structured_output_async(
        WeatherReport,
        "What's the weather in NYC?"
    )
    return weather
```

**After (Recommended):**
```python
async def get_weather():
    # Option 1: Use streaming interface
    events = agent.stream_async(
        "What's the weather in NYC?",
        output_schema=OutputSchema([WeatherReport])
    )

    async for event in events:
        if hasattr(event, 'get') and event.get('result'):
            result = event['result']
            if result.structured_output:
                return result.get_structured_output(WeatherReport)

    # Option 2: If you need simple async without streaming
    # Note: This runs sync agent() in a thread pool
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def sync_call():
        result = agent("What's the weather in NYC?", output_type=WeatherReport)
        return result.get_structured_output(WeatherReport)

    with ThreadPoolExecutor() as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, sync_call)
```

### Step 3: Handle Multiple Output Types

**Before (Not directly supported):**
```python
# Had to use separate calls or complex prompting
try:
    weather = agent.structured_output(WeatherReport, prompt)
except:
    error = agent.structured_output(ErrorResponse, prompt)
```

**After (Native support):**
```python
from strands.output import OutputSchema

schema = OutputSchema([WeatherReport, ErrorResponse])
result = agent(prompt, output_schema=schema)

if isinstance(result.structured_output, WeatherReport):
    weather = result.get_structured_output(WeatherReport)
    print(f"Weather: {weather.temperature}°F")
elif isinstance(result.structured_output, ErrorResponse):
    error = result.get_structured_output(ErrorResponse)
    print(f"Error: {error.message}")
```

## Common Migration Patterns

### Pattern 1: Simple Model Response

**Before:**
```python
class UserProfile(BaseModel):
    name: str
    age: int
    email: str

def create_user_profile(description: str) -> UserProfile:
    return agent.structured_output(UserProfile, f"Create a user profile: {description}")
```

**After:**
```python
class UserProfile(BaseModel):
    name: str
    age: int
    email: str

def create_user_profile(description: str) -> UserProfile:
    result = agent(f"Create a user profile: {description}", output_type=UserProfile)
    return result.get_structured_output(UserProfile)
```

### Pattern 2: Error Handling

**Before:**
```python
try:
    data = agent.structured_output(MyModel, prompt)
    process_data(data)
except Exception as e:
    handle_error(e)
```

**After:**
```python
try:
    result = agent(prompt, output_type=MyModel)
    data = result.get_structured_output(MyModel)
    process_data(data)
except ValueError as e:
    # No structured output or validation failed
    handle_structured_output_error(e)
except Exception as e:
    # Other errors (network, authentication, etc.)
    handle_general_error(e)
```

### Pattern 3: Conditional Prompting

**Before:**
```python
def get_analysis(data: str, detailed: bool = False):
    if detailed:
        return agent.structured_output(DetailedAnalysis, f"Analyze in detail: {data}")
    else:
        return agent.structured_output(SimpleAnalysis, f"Analyze briefly: {data}")
```

**After:**
```python
def get_analysis(data: str, detailed: bool = False):
    if detailed:
        result = agent(f"Analyze in detail: {data}", output_type=DetailedAnalysis)
        return result.get_structured_output(DetailedAnalysis)
    else:
        result = agent(f"Analyze briefly: {data}", output_type=SimpleAnalysis)
        return result.get_structured_output(SimpleAnalysis)

# Or better yet, use multiple output types:
def get_analysis_improved(data: str):
    schema = OutputSchema([DetailedAnalysis, SimpleAnalysis])
    prompt = f"Analyze this data (choose detail level): {data}"
    result = agent(prompt, output_schema=schema)
    return result.structured_output  # Agent chooses the appropriate type
```

### Pattern 4: Batch Processing

**Before:**
```python
async def process_batch(items: List[str]) -> List[ProcessedItem]:
    results = []
    for item in items:
        result = await agent.structured_output_async(ProcessedItem, f"Process: {item}")
        results.append(result)
    return results
```

**After:**
```python
async def process_batch(items: List[str]) -> List[ProcessedItem]:
    results = []
    schema = OutputSchema([ProcessedItem])

    for item in items:
        events = agent.stream_async(f"Process: {item}", output_schema=schema)
        async for event in events:
            if hasattr(event, 'get') and event.get('result'):
                result = event['result']
                if result.structured_output:
                    processed = result.get_structured_output(ProcessedItem)
                    results.append(processed)
                    break

    return results

# Or for better performance, use the sync interface:
def process_batch_sync(items: List[str]) -> List[ProcessedItem]:
    results = []
    for item in items:
        result = agent(f"Process: {item}", output_type=ProcessedItem)
        processed = result.get_structured_output(ProcessedItem)
        results.append(processed)
    return results
```

## Advanced Migration Scenarios

### Custom Output Modes

If you were using custom prompting strategies with the old system:

**Before:**
```python
# Custom prompt construction
custom_prompt = f"""
{user_prompt}

Please respond with JSON matching this schema:
{json.dumps(MyModel.model_json_schema())}
"""

response = agent.structured_output(MyModel, custom_prompt)
```

**After:**
```python
from strands.output import PromptedOutput

# Use PromptedOutput mode with custom template
custom_mode = PromptedOutput(
    template="""
{user_input}

Please respond with JSON matching this schema:
{schema}

Ensure all required fields are included.
"""
)

agent = Agent(model_id="gpt-4o", output_mode=custom_mode)
result = agent(user_prompt, output_type=MyModel)
response = result.get_structured_output(MyModel)
```

### Model-Specific Optimizations

**Before:**
```python
# Different handling for different models
if model_id.startswith("gpt"):
    response = agent.structured_output(MyModel, prompt)
else:
    # Custom handling for other models
    custom_prompt = f"Use strict JSON format: {prompt}"
    response = agent.structured_output(MyModel, custom_prompt)
```

**After:**
```python
from strands.output import NativeOutput, ToolOutput

# Let the system choose the best approach automatically
if model_id.startswith("gpt"):
    # Try native structured output, fall back to tool-based
    agent = Agent(model_id=model_id, output_mode=NativeOutput())
else:
    # Use reliable tool-based approach
    agent = Agent(model_id=model_id, output_mode=ToolOutput())

result = agent(prompt, output_type=MyModel)
response = result.get_structured_output(MyModel)
```

## Handling Deprecation Warnings

During migration, you'll see deprecation warnings:

```
DeprecationWarning: structured_output() is deprecated.
Use agent(prompt, output_type=YourModel).get_structured_output(YourModel) instead.
```

### Suppress Warnings Temporarily

```python
import warnings

# Suppress during migration period
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="strands")
    # Your legacy code here
    result = agent.structured_output(MyModel, prompt)
```

### Gradual Migration Strategy

1. **Phase 1**: Keep existing code but suppress warnings
2. **Phase 2**: Migrate new code to use the new API
3. **Phase 3**: Gradually refactor existing code
4. **Phase 4**: Remove all legacy method usage

## Testing Your Migration

### Unit Tests

**Before:**
```python
def test_weather_extraction():
    agent = Agent(model_id="gpt-4o")
    weather = agent.structured_output(WeatherReport, "Weather in NYC")
    assert isinstance(weather, WeatherReport)
    assert weather.temperature > 0
```

**After:**
```python
def test_weather_extraction():
    agent = Agent(model_id="gpt-4o")
    result = agent("Weather in NYC", output_type=WeatherReport)
    weather = result.get_structured_output(WeatherReport)
    assert isinstance(weather, WeatherReport)
    assert weather.temperature > 0

    # Test the result object too
    assert result.structured_output is not None
    assert isinstance(result.structured_output, WeatherReport)
```

### Integration Tests

```python
async def test_async_migration():
    agent = Agent(model_id="gpt-4o")
    schema = OutputSchema([WeatherReport])

    events = agent.stream_async("Weather in NYC", output_schema=schema)
    weather = None

    async for event in events:
        if hasattr(event, 'get') and event.get('result'):
            result = event['result']
            if result.structured_output:
                weather = result.get_structured_output(WeatherReport)
                break

    assert weather is not None
    assert isinstance(weather, WeatherReport)
```

## Performance Considerations

### Before vs After Performance

The new system provides several performance benefits:

1. **Full Event Loop Integration**: Metrics and telemetry now work with structured output
2. **Output Mode Selection**: Choose optimal strategy for your use case
3. **Schema Caching**: Output schemas are cached for repeated use
4. **Streaming Support**: Real-time structured output events

### Benchmarking

Compare performance of old vs new:

```python
import time
from strands import Agent
from strands.output import OutputSchema, ToolOutput, NativeOutput

agent = Agent(model_id="gpt-4o")

# Benchmark legacy method (deprecated)
start = time.time()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    old_result = agent.structured_output(MyModel, prompt)
old_time = time.time() - start

# Benchmark new method with ToolOutput
agent.output_mode = ToolOutput()
start = time.time()
result = agent(prompt, output_type=MyModel)
new_result = result.get_structured_output(MyModel)
new_time = time.time() - start

# Benchmark new method with NativeOutput (if supported)
agent.output_mode = NativeOutput()
start = time.time()
result = agent(prompt, output_type=MyModel)
native_result = result.get_structured_output(MyModel)
native_time = time.time() - start

print(f"Legacy method: {old_time:.2f}s")
print(f"New ToolOutput: {new_time:.2f}s")
print(f"New NativeOutput: {native_time:.2f}s")
```

## Troubleshooting Migration Issues

### Common Issues and Solutions

#### 1. "No structured output available"

**Problem**: Getting `ValueError` when calling `get_structured_output()`

**Solution**: Ensure you're passing `output_type` or `output_schema`:

```python
# Wrong
result = agent("Generate data")
data = result.get_structured_output(MyModel)  # Error!

# Correct
result = agent("Generate data", output_type=MyModel)
data = result.get_structured_output(MyModel)  # Works!
```

#### 2. Type validation errors

**Problem**: Pydantic validation failing with new system

**Solution**: Check your model definitions and add better validation:

```python
class ImprovedModel(BaseModel):
    name: str = Field(description="Clear description")
    age: int = Field(description="Age in years", ge=0, le=150)

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
```

#### 3. Async streaming complexity

**Problem**: New async interface is more complex than the old `structured_output_async()`

**Solution**: Create helper functions:

```python
async def get_structured_output_async(agent, prompt, output_type):
    """Helper to simplify async structured output."""
    schema = OutputSchema([output_type])
    events = agent.stream_async(prompt, output_schema=schema)

    async for event in events:
        if hasattr(event, 'get') and event.get('result'):
            result = event['result']
            if result.structured_output:
                return result.get_structured_output(output_type)

    raise ValueError("No structured output received")

# Usage
weather = await get_structured_output_async(agent, "Weather in NYC", WeatherReport)
```

#### 4. Multiple model support

**Problem**: Need to support multiple model providers with different capabilities

**Solution**: Use factory pattern with output mode detection:

```python
def create_agent_for_model(model_id: str) -> Agent:
    """Create agent with optimal output mode for the given model."""
    if model_id.startswith("gpt"):
        # OpenAI supports native structured output
        return Agent(model_id=model_id, output_mode=NativeOutput())
    else:
        # Use reliable tool-based approach for others
        return Agent(model_id=model_id, output_mode=ToolOutput())

# Usage
agent = create_agent_for_model("gpt-4o")
result = agent(prompt, output_type=MyModel)
```

## Migration Checklist

### Pre-Migration

- [ ] Review current usage of `structured_output()` and `structured_output_async()`
- [ ] Identify custom prompting strategies that might benefit from `PromptedOutput`
- [ ] Plan migration phases (new code first, then legacy code)
- [ ] Set up testing for new API

### During Migration

- [ ] Update imports to include new output classes
- [ ] Replace method calls with new unified interface
- [ ] Add output mode selection where beneficial
- [ ] Update error handling for new exception types
- [ ] Test both sync and async patterns
- [ ] Update documentation and examples

### Post-Migration

- [ ] Remove deprecation warning suppressions
- [ ] Optimize output modes for your use cases
- [ ] Monitor performance improvements
- [ ] Update CI/CD pipelines if needed
- [ ] Train team on new patterns

## Getting Help

If you encounter issues during migration:

1. **Check Examples**: Review the [examples directory](../examples/structured_output/)
2. **API Documentation**: See the [full API reference](./structured_output_api.md)
3. **GitHub Issues**: Report bugs or ask questions
4. **Community**: Join discussions about best practices

## Timeline for Legacy Support

- **Current**: Legacy methods work with deprecation warnings
- **Next Minor Release**: Warnings become more prominent
- **Next Major Release**: Legacy methods may be removed

We recommend migrating as soon as possible to take advantage of the new features and ensure future compatibility.

---

## Summary

The new structured output system provides a more powerful, intuitive, and integrated approach to getting structured data from AI models. While migration requires some code changes, the benefits include:

- Better integration with SDK features
- More flexible output strategies
- Improved performance options
- Future-proof architecture
- Enhanced type safety

Follow this guide step-by-step, test thoroughly, and don't hesitate to reach out if you need help with your migration!