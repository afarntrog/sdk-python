# Structured Output API Reference

## Agent Methods

### `agent(prompt, output_type=None, **kwargs)`
**Synchronous structured output interface**

```python
result = agent("Generate data", output_type=MyModel)
```

**Parameters:**
- `prompt` (str | list | None): Input prompt in various formats
- `output_type` (Type[BaseModel] | None): Pydantic model class for structured parsing
- `**kwargs`: Additional parameters passed to event loop

**Returns:** `AgentResult` with `structured_output` field

---

### `agent.invoke_async(prompt, output_type=None, **kwargs)`
**Asynchronous structured output interface**

```python
result = await agent.invoke_async("Generate data", output_type=MyModel)
```

**Parameters:**
- `prompt` (str | list | None): Input prompt in various formats  
- `output_type` (Type[BaseModel] | None): Pydantic model class for structured parsing
- `**kwargs`: Additional parameters passed to event loop

**Returns:** `AgentResult` with `structured_output` field

---

### `agent.stream_async(prompt, output_type=None, **kwargs)`
**Streaming interface with structured output**

```python
async for event in agent.stream_async("Generate data", output_type=MyModel):
    if "result" in event:
        data = event["result"].structured_output
```

**Parameters:**
- `prompt` (str | list | None): Input prompt in various formats
- `output_type` (Type[BaseModel] | None): Pydantic model class for structured parsing  
- `**kwargs`: Additional parameters passed to event loop

**Yields:** Stream events, final event contains `AgentResult` with `structured_output`

## AgentResult Class

### Fields
- `stop_reason` (str): Why the agent stopped ("end_turn", "max_tokens", etc.)
- `message` (dict): Final message from the model
- `metrics` (EventLoopMetrics): Performance and execution metrics
- `state` (dict): Final event loop state
- `structured_output` (BaseModel | None): Parsed structured output when `output_type` provided

### Methods
- `__str__()`: Returns the text content of the agent's response

```python
result = agent("Generate data", output_type=MyModel)

# Access text content
text = str(result)

# Access structured data  
data = result.structured_output

# Access metrics
metrics = result.metrics.get_summary()
```

## StructuredOutputManager

### `detect_provider_capabilities(model)`
**Detect available strategies for a model provider**

```python
from strands.structured_output import StructuredOutputManager

manager = StructuredOutputManager()
strategies = manager.detect_provider_capabilities(agent.model)
print(f"Available strategies: {strategies}")
```

**Parameters:**
- `model` (Model): The model instance to check

**Returns:** `List[str]` - Available strategy names in priority order

---

### `execute_structured_output(model, output_type, messages, system_prompt=None)`
**Execute structured output parsing**

```python
result = await manager.execute_structured_output(
    model=agent.model,
    output_type=MyModel,
    messages=conversation_messages,
    system_prompt="You are a helpful assistant"
)
```

**Parameters:**
- `model` (Model): Model instance to use
- `output_type` (Type[BaseModel]): Pydantic model class
- `messages` (List[dict]): Conversation messages
- `system_prompt` (str | None): Optional system prompt

**Returns:** `BaseModel | None` - Parsed model instance or None if parsing failed

## Strategies

### Available Strategies

1. **NativeStrategy** - Uses provider's native structured output API
   - Providers: OpenAI, LiteLLM
   - Reliability: Highest

2. **JsonSchemaStrategy** - Uses JSON schema format parameter
   - Providers: Ollama, LlamaCpp  
   - Reliability: High

3. **ToolCallingStrategy** - Uses tool calling mechanism
   - Providers: Bedrock, Anthropic
   - Reliability: High

4. **PromptBasedStrategy** - Uses prompt engineering (universal fallback)
   - Providers: Any text-generation model
   - Reliability: Universal

### Strategy Selection

Strategies are automatically selected based on provider capabilities:

```python
# Automatic selection (recommended)
result = agent("Generate data", output_type=MyModel)

# Manual strategy inspection
from strands.structured_output import StructuredOutputManager
manager = StructuredOutputManager()
capabilities = manager.detect_provider_capabilities(agent.model)
```

## Metrics and Monitoring

### EventLoopMetrics.structured_output

Access structured output metrics through the agent's event loop metrics:

```python
result = agent("Generate data", output_type=MyModel)
metrics = result.metrics.get_summary()
so_metrics = metrics['structured_output']

print(f"Attempts: {so_metrics['attempts']}")
print(f"Successes: {so_metrics['successes']}")  
print(f"Success rate: {so_metrics['success_rate']:.1%}")
print(f"Strategy used: {so_metrics['strategy_used']}")
print(f"Total parsing time: {so_metrics['total_parsing_time']:.3f}s")
print(f"Average parsing time: {so_metrics['average_parsing_time']:.3f}s")
```

### Available Metrics

- `attempts` (int): Total structured output parsing attempts
- `successes` (int): Number of successful parses
- `success_rate` (float): Success rate (successes/attempts)
- `strategy_used` (str): Last strategy used ("native", "json_schema", "tool_calling", "prompt_based")
- `total_parsing_time` (float): Total time spent parsing (seconds)
- `average_parsing_time` (float): Average parsing time per attempt (seconds)

## Error Handling

### Exception Types

```python
from strands.structured_output.exceptions import (
    StructuredOutputError,           # Base exception
    StructuredOutputValidationError, # Pydantic validation failed
    StructuredOutputParsingError     # JSON/response parsing failed
)
```

### Error Handling Patterns

```python
try:
    result = agent("Generate data", output_type=MyModel)
    
    if result.structured_output is not None:
        # Success - use structured data
        process_data(result.structured_output)
    else:
        # Parsing failed but text available
        process_text(str(result))
        
except StructuredOutputValidationError as e:
    print(f"Validation error: {e}")
    print(f"Provider: {e.provider}, Strategy: {e.strategy}")
    
except StructuredOutputError as e:
    print(f"Structured output error: {e}")
    
except Exception as e:
    print(f"General error: {e}")
```

## Configuration

### Provider-Specific Configuration

```python
from strands import Agent
from strands.models import BedrockModel, OpenAIModel, OllamaModel

# Bedrock (uses tool calling)
bedrock_agent = Agent(model=BedrockModel(model_id="us.amazon.nova-pro-v1:0"))

# OpenAI (uses native API)  
openai_agent = Agent(model=OpenAIModel(model_id="gpt-4"))

# Ollama (uses JSON schema)
ollama_agent = Agent(model=OllamaModel(host="http://localhost:11434", model_id="llama3"))

# All use the same interface
result = bedrock_agent("Generate data", output_type=MyModel)
result = openai_agent("Generate data", output_type=MyModel)  
result = ollama_agent("Generate data", output_type=MyModel)
```

### Model Requirements

For structured output to work optimally:

1. **Pydantic Models** - Must inherit from `BaseModel`
2. **Clear Field Types** - Use standard Python types (str, int, bool, List, etc.)
3. **Reasonable Complexity** - Very complex nested models may have lower success rates
4. **Provider Compatibility** - All providers supported with automatic fallback

### Best Practices

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class OptimalModel(BaseModel):
    """Clear docstring describing the model."""
    
    # Use descriptive names and types
    user_name: str = Field(description="Full name of the user")
    age_years: int = Field(description="Age in years", ge=0, le=150)
    
    # Provide defaults for optional fields
    is_active: bool = Field(default=True)
    
    # Use constraints for validation
    status: str = Field(pattern="^(active|inactive|pending)$")
    
    # Lists and nested objects work well
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[dict] = None
```

## Integration Examples

### With Existing Agent Workflows

```python
# Structured output integrates seamlessly with existing patterns
agent = Agent(tools=[my_tool])

# Regular agent usage
response = agent("Use the calculator tool to compute 2+2")

# Structured output usage  
result = agent("Generate a report", output_type=ReportModel)

# Both work together
result = agent("Use tools to gather data, then format as structured output", 
               output_type=DataModel)
```

### With Streaming and Tools

```python
async for event in agent.stream_async(
    "Research the topic and create a structured summary",
    output_type=SummaryModel
):
    if "tool_use" in event:
        print(f"Using tool: {event['tool_use']['name']}")
    elif "data" in event:
        print(event["data"], end="")
    elif "result" in event:
        summary = event["result"].structured_output
        print(f"\nStructured summary: {summary}")
```

This API reference covers all the key interfaces and patterns for using structured output with Strands Agents.
