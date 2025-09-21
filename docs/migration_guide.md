# Migration Guide: Structured Output

This guide helps you migrate from the old structured output API to the new unified system.

## Overview of Changes

The new structured output system provides:
- ✅ Unified interface with regular agent calls
- ✅ Multiple output modes (tool-based, native, prompted)
- ✅ Better type safety and validation
- ✅ Automatic model capability detection
- ✅ Improved error handling

## Migration Steps

### Step 1: Update Basic Usage

**Old API:**
```python
from strands import Agent
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int

agent = Agent()
user = agent.structured_output(UserProfile, "Create user profile")
```

**New API:**
```python
from strands import Agent
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int

agent = Agent()
result = agent("Create user profile", output_type=UserProfile)
user = result.get_structured_output(UserProfile)
```

### Step 2: Update Async Usage

**Old API:**
```python
user = await agent.structured_output_async(UserProfile, "Create user profile")
```

**New API:**
```python
result = await agent.async_run("Create user profile", output_type=UserProfile)
user = result.get_structured_output(UserProfile)
```

### Step 3: Set Default Output Types

**New Feature:**
```python
# Set default output type for all calls
agent = Agent(output_type=UserProfile)
result = agent("Create user profile")  # Automatically uses UserProfile
user = result.get_structured_output(UserProfile)
```

### Step 4: Use Output Modes

**New Feature:**
```python
from strands import NativeMode, ToolMode

# Use native structured output when available
agent = Agent(output_type=UserProfile, output_mode=NativeMode())

# Or explicitly use tool-based approach
agent = Agent(output_type=UserProfile, output_mode=ToolMode())
```

## Common Migration Patterns

### Pattern 1: Simple Structured Output

**Before:**
```python
def get_user_info(prompt: str) -> UserProfile:
    agent = Agent()
    return agent.structured_output(UserProfile, prompt)
```

**After:**
```python
def get_user_info(prompt: str) -> UserProfile:
    agent = Agent(output_type=UserProfile)
    result = agent(prompt)
    return result.get_structured_output(UserProfile)
```

### Pattern 2: Multiple Output Types

**Before:**
```python
def process_data(prompt: str, output_type: Type[BaseModel]):
    agent = Agent()
    return agent.structured_output(output_type, prompt)
```

**After:**
```python
def process_data(prompt: str, output_type: Type[BaseModel]):
    agent = Agent()
    result = agent(prompt, output_type=output_type)
    return result.get_structured_output(output_type)
```

### Pattern 3: Async Processing

**Before:**
```python
async def async_process(prompt: str) -> UserProfile:
    agent = Agent()
    return await agent.structured_output_async(UserProfile, prompt)
```

**After:**
```python
async def async_process(prompt: str) -> UserProfile:
    agent = Agent(output_type=UserProfile)
    result = await agent.async_run(prompt)
    return result.get_structured_output(UserProfile)
```

### Pattern 4: Error Handling

**Before:**
```python
try:
    user = agent.structured_output(UserProfile, prompt)
except Exception as e:
    print(f"Error: {e}")
```

**After:**
```python
try:
    result = agent(prompt, output_type=UserProfile)
    user = result.get_structured_output(UserProfile)
except ValueError as e:
    print(f"Structured output error: {e}")
except Exception as e:
    print(f"General error: {e}")
```

## Backward Compatibility

The old methods still work but issue deprecation warnings:

```python
import warnings

# This still works but shows a warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    user = agent.structured_output(UserProfile, "Create user")
```

## New Features Available

### 1. Output Modes

```python
from strands import NativeMode, PromptMode

# Use model's native structured output
agent = Agent(output_type=UserProfile, output_mode=NativeMode())

# Use custom prompt template
template = "Extract information: {prompt}\nFormat as JSON:"
agent = Agent(output_type=UserProfile, output_mode=PromptMode(template=template))
```

### 2. Model Capability Detection

```python
from strands.models import OpenAIModel, BedrockModel

# Automatically uses native structured output
openai_agent = Agent(model=OpenAIModel(model_id="gpt-4"), output_type=UserProfile)

# Automatically falls back to tool-based approach
bedrock_agent = Agent(model=BedrockModel(model_id="claude-3"), output_type=UserProfile)
```

### 3. Multiple Output Types

```python
from strands import OutputSchema, ToolMode

schema = OutputSchema(
    types=[UserProfile, TaskInfo],
    mode=ToolMode(),
    description="Can output user profile or task information"
)

agent = Agent(output_schema=schema)
```

### 4. Better Type Safety

```python
# Type-safe extraction with validation
result = agent("Create user", output_type=UserProfile)
user: UserProfile = result.get_structured_output(UserProfile)  # Type-checked

# Runtime type validation
try:
    task = result.get_structured_output(TaskInfo)  # Will raise ValueError
except ValueError as e:
    print("Type mismatch detected")
```

## Performance Considerations

### Old System
- Always used function calling
- No model capability detection
- Limited error handling

### New System
- Uses native structured output when available (faster)
- Automatic fallback to function calling
- Better error messages and debugging
- Caching of tool specifications

## Testing Your Migration

1. **Run both versions side by side:**
```python
# Test old vs new
old_result = agent.structured_output(UserProfile, prompt)
new_result = agent(prompt, output_type=UserProfile).get_structured_output(UserProfile)

assert old_result.name == new_result.name
assert old_result.age == new_result.age
```

2. **Check for deprecation warnings:**
```python
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    agent.structured_output(UserProfile, prompt)
    
    if w:
        print("Deprecation warnings found - migration needed")
```

3. **Test error handling:**
```python
# Ensure error handling works as expected
try:
    result = agent("Invalid input", output_type=UserProfile)
    user = result.get_structured_output(UserProfile)
except ValueError as e:
    print(f"Validation error: {e}")
```

## Troubleshooting

### Common Issues

1. **"No structured output found"**
   - Check that the model actually called the structured output tool
   - Verify your prompt is clear about what output you want

2. **Type mismatch errors**
   - Ensure you're calling `get_structured_output()` with the correct type
   - Check that the model returned the expected structure

3. **Model not supporting native output**
   - This is normal - the system automatically falls back to tool-based approach
   - You can check support with `model.supports_native_structured_output()`

### Getting Help

1. **Enable debug logging:**
```python
import logging
logging.getLogger("strands").setLevel(logging.DEBUG)
```

2. **Check model capabilities:**
```python
print(f"Native support: {agent.model.supports_native_structured_output()}")
```

3. **Inspect the result:**
```python
result = agent("Create user", output_type=UserProfile)
print(f"Has structured output: {result.structured_output is not None}")
print(f"Result type: {type(result.structured_output)}")
```

## Timeline for Migration

- **Immediate**: Start using new API for new code
- **3 months**: Migrate existing code to new API
- **6 months**: Old API will show deprecation warnings
- **12 months**: Old API may be removed (with advance notice)

The new system is fully backward compatible, so you can migrate at your own pace.
