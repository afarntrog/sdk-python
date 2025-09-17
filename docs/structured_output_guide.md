# Structured Output Guide

This guide covers the new unified structured output system in Strands Agents, which provides multiple ways to get structured data from AI models.

## Quick Start

```python
from strands import Agent
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    email: str

# Create agent with default structured output
agent = Agent(output_type=UserProfile)
result = agent("Create a user profile for John, age 30, email john@example.com")
user = result.get_structured_output(UserProfile)
print(f"Name: {user.name}, Age: {user.age}")
```

## Core Concepts

### Output Modes

The system supports three output modes:

1. **ToolOutput** (default) - Uses function calling for maximum compatibility
2. **NativeOutput** - Uses model's native structured output when available
3. **PromptedOutput** - Uses prompt engineering with custom templates

### Output Schema

The `OutputSchema` class defines what structured output you want:

```python
from strands import OutputSchema, ToolOutput

schema = OutputSchema(
    types=[UserProfile],
    mode=ToolOutput(),
    name="user_profile_output",
    description="Extract user profile information"
)
```

## Usage Patterns

### 1. Agent Constructor with Default Output Type

```python
# Set default output type for all calls
agent = Agent(output_type=UserProfile)
result = agent("Extract user info from: John Smith, 25, john@email.com")
user = result.get_structured_output(UserProfile)
```

### 2. Runtime Output Type Specification

```python
# Specify output type per call
agent = Agent()
result = agent("Create user profile", output_type=UserProfile)
user = result.get_structured_output(UserProfile)
```

### 3. Multiple Output Types

```python
class Task(BaseModel):
    name: str
    priority: str

class Project(BaseModel):
    title: str
    tasks: list[Task]

# Agent can handle multiple types
agent = Agent()
result1 = agent("Create a task", output_type=Task)
result2 = agent("Create a project", output_type=Project)
```

### 4. Custom Output Modes

```python
from strands import NativeOutput, PromptedOutput

# Use native structured output when available
agent = Agent(output_type=UserProfile, output_mode=NativeOutput())

# Use custom prompt template
template = "Extract the following information and format as JSON: {prompt}"
agent = Agent(output_type=UserProfile, output_mode=PromptedOutput(template=template))
```

### 5. Model Provider Capabilities

```python
from strands.models import OpenAIModel, BedrockModel

# OpenAI supports native structured output
openai_model = OpenAIModel(model_id="gpt-4")
agent = Agent(model=openai_model, output_type=UserProfile, output_mode=NativeOutput())

# Bedrock uses function calling
bedrock_model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
agent = Agent(model=bedrock_model, output_type=UserProfile)  # Automatically uses ToolOutput
```

## Advanced Features

### Automatic Fallback

The system automatically falls back to compatible modes:

```python
# If NativeOutput isn't supported, automatically falls back to ToolOutput
agent = Agent(
    output_type=UserProfile, 
    output_mode=NativeOutput()  # Will fallback to ToolOutput if not supported
)
```

### Streaming with Structured Output

```python
async def stream_example():
    agent = Agent(output_type=UserProfile)
    async for event in agent.stream_async("Create user profile"):
        if event.get("type") == "structured_output":
            user = event["data"]
            print(f"Got user: {user.name}")
```

### Complex Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    address: Address

class Employee(BaseModel):
    name: str
    position: str
    company: Company

agent = Agent(output_type=Employee)
result = agent("Extract employee info: John Smith, Software Engineer at Acme Corp, 123 Main St, New York, USA")
employee = result.get_structured_output(Employee)
```

## Backward Compatibility

Existing methods continue to work with deprecation warnings:

```python
# Old way (deprecated but still works)
user = agent.structured_output(UserProfile, "Create user profile")

# New way (recommended)
result = agent("Create user profile", output_type=UserProfile)
user = result.get_structured_output(UserProfile)
```

## Error Handling

```python
try:
    result = agent("Create user profile", output_type=UserProfile)
    user = result.get_structured_output(UserProfile)
except ValueError as e:
    print(f"Structured output error: {e}")
```

## Best Practices

1. **Use descriptive model docstrings** - They help the AI understand what to extract
2. **Provide clear field descriptions** - Use Pydantic field descriptions
3. **Start with ToolOutput mode** - Most compatible across all models
4. **Use NativeOutput for supported models** - Better performance when available
5. **Handle validation errors** - Always wrap in try/catch blocks

## Model Provider Support

| Provider | Native Support | Recommended Mode |
|----------|----------------|------------------|
| OpenAI | ✅ Yes | NativeOutput |
| Anthropic | ❌ No | ToolOutput |
| Bedrock | ❌ No | ToolOutput |
| Ollama | ✅ Yes | NativeOutput |
| LlamaCpp | ✅ Yes | NativeOutput |
| Others | ❌ No | ToolOutput |

## Migration Guide

### From Old to New API

```python
# Old API
user = agent.structured_output(UserProfile, "Create user")

# New API
result = agent("Create user", output_type=UserProfile)
user = result.get_structured_output(UserProfile)
```

### Benefits of New API

- Unified interface with regular agent calls
- Better type safety and validation
- Support for multiple output modes
- Automatic model capability detection
- Improved error handling and debugging
