# Structured Output for Strands Agents

Automatically parse model responses into typed Pydantic models with universal provider support and intelligent fallback strategies.

## Quick Start

```python
from strands import Agent
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    occupation: str

agent = Agent()
result = agent("Create a user profile for John, age 30, engineer", output_type=UserProfile)

print(f"Text: {result}")  # Full response text
print(f"Data: {result.structured_output}")  # UserProfile(name="John", age=30, ...)
```

## Key Features

✅ **Universal Provider Support** - Works with OpenAI, Bedrock, Anthropic, Ollama, LlamaCpp, and more  
✅ **Automatic Strategy Selection** - Chooses the best parsing method for your model provider  
✅ **Intelligent Fallbacks** - Graceful degradation ensures reliability across all providers  
✅ **Streaming Compatible** - Real-time text + structured output in final event  
✅ **Performance Monitoring** - Built-in metrics for success rates and parsing performance  
✅ **Type Safety** - Full Pydantic validation with rich error handling  

## Usage Patterns

### Synchronous
```python
result = agent("Generate data", output_type=MyModel)
data = result.structured_output
```

### Asynchronous  
```python
result = await agent.invoke_async("Generate data", output_type=MyModel)
data = result.structured_output
```

### Streaming
```python
async for event in agent.stream_async("Generate data", output_type=MyModel):
    if "data" in event:
        print(event["data"], end="")  # Real-time text
    elif "result" in event:
        data = event["result"].structured_output  # Final structured output
```

## Provider Strategies

| Provider | Strategy | Reliability |
|----------|----------|-------------|
| OpenAI, LiteLLM | Native API | Highest |
| Ollama, LlamaCpp | JSON Schema | High |
| Bedrock, Anthropic | Tool Calling | High |
| Any Provider | Prompt Engineering | Universal Fallback |

## Advanced Examples

### Complex Nested Models
```python
class Address(BaseModel):
    street: str
    city: str
    state: str

class Person(BaseModel):
    name: str
    age: int
    address: Address
    skills: List[str]

result = agent("Create a person profile with address and skills", output_type=Person)
```

### Data Extraction
```python
class ExpenseReport(BaseModel):
    total: float
    expenses: List[dict]
    categories: List[str]

receipt_text = "Coffee $4.50, Gas $45.00, Lunch $12.75"
result = agent(f"Extract expenses from: {receipt_text}", output_type=ExpenseReport)
```

### Error Handling
```python
result = agent("Generate data", output_type=MyModel)

if result.structured_output is not None:
    # Use structured data
    process_data(result.structured_output)
else:
    # Fallback to text processing
    process_text(str(result))
```

## Performance Monitoring

```python
# Access built-in metrics
metrics = agent.event_loop_metrics.get_summary()
so_metrics = metrics['structured_output']

print(f"Success rate: {so_metrics['success_rate']:.1%}")
print(f"Strategy used: {so_metrics['strategy_used']}")
print(f"Avg parsing time: {so_metrics['average_parsing_time']:.3f}s")
```

## Documentation

📖 **[Complete Documentation](docs/structured_output.md)** - Comprehensive guide with advanced examples  
🔧 **[Working Examples](examples/structured_output_examples.py)** - Practical code samples  
🏗️ **[API Reference](src/strands/structured_output/)** - Technical implementation details  

## Architecture

The structured output system uses a strategy pattern with automatic provider detection:

1. **Provider Detection** - Identifies your model provider capabilities
2. **Strategy Selection** - Chooses optimal parsing approach (Native → JSON Schema → Tool Calling → Prompt Engineering)
3. **Execution** - Runs structured output with automatic fallback on failure
4. **Integration** - Seamlessly integrates with existing agent streaming and metrics

## Migration from Manual Parsing

### Before
```python
response = agent("Generate JSON data")
text = str(response)
try:
    data = json.loads(text)
    obj = MyModel(**data)
except (json.JSONDecodeError, ValidationError):
    # Handle errors manually
    pass
```

### After
```python
result = agent("Generate data", output_type=MyModel)
obj = result.structured_output  # Automatically parsed and validated
```

## Requirements

- **Pydantic** - For model definitions and validation
- **Strands Agents** - Core agent framework
- **Model Provider** - Any supported provider (OpenAI, Bedrock, Ollama, etc.)

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install strands-agents pydantic
   ```

2. **Define Your Model**
   ```python
   from pydantic import BaseModel
   
   class MyData(BaseModel):
       name: str
       value: int
   ```

3. **Use with Agent**
   ```python
   from strands import Agent
   
   agent = Agent()
   result = agent("Generate data", output_type=MyData)
   ```

4. **Access Results**
   ```python
   text_content = str(result)           # Full text response
   structured_data = result.structured_output  # Parsed Pydantic model
   ```

## Best Practices

- **Clear Models** - Use descriptive field names and include docstrings
- **Specific Prompts** - Provide clear instructions for data generation
- **Error Handling** - Always check if `structured_output` is not None
- **Monitor Performance** - Use built-in metrics to track success rates
- **Test Fallbacks** - Verify behavior when structured parsing fails

## Support

- **Universal Compatibility** - Works with all Strands-supported model providers
- **Automatic Fallbacks** - Graceful degradation ensures reliability
- **Performance Monitoring** - Built-in observability for production use
- **Type Safety** - Full Pydantic validation and error reporting

---

**Ready to get started?** Check out the [complete documentation](docs/structured_output.md) and [working examples](examples/structured_output_examples.py)!
