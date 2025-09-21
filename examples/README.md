# Structured Output Examples

This directory contains comprehensive examples and documentation for the new structured output system in Strands Agents.

## Files Overview

### 📚 Documentation
- **`../docs/structured_output_guide.md`** - Complete API guide with usage patterns
- **`../docs/migration_guide.md`** - Migration guide from old to new API

### 🧪 Test Examples
- **`simple_test.py`** - Component tests that validate all system parts work
- **`structured_output_examples.py`** - Comprehensive examples (requires model setup)
- **`working_examples.py`** - Examples with mock model (currently needs fixes)

## Quick Start

### 1. Run Component Tests
```bash
cd /path/to/sdk-python
python examples/simple_test.py
```

This validates that all structured output components are working correctly without requiring model credentials.

### 2. Basic Usage Pattern

```python
from strands import Agent
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    email: str

# Method 1: Default output type
agent = Agent(output_type=UserProfile)
result = agent("Create a user profile for John, age 30, email john@example.com")
user = result.get_structured_output(UserProfile)

# Method 2: Runtime output type
agent = Agent()
result = agent("Create user profile", output_type=UserProfile)
user = result.get_structured_output(UserProfile)
```

## Key Features Demonstrated

### ✅ Output Modes
- **ToolMode** (default) - Uses function calling, works with all models
- **NativeMode** - Uses model's native structured output when available
- **PromptMode** - Uses custom prompt templates

### ✅ Model Provider Support
- **OpenAI** - Native structured output support
- **Bedrock** - Function calling approach
- **Anthropic** - Function calling approach
- **Ollama** - Native JSON schema support
- **Others** - Automatic fallback to function calling

### ✅ Advanced Features
- Multiple output types per agent
- Automatic model capability detection
- Fallback mechanisms
- Type safety and validation
- Streaming support
- Dynamic tool registration

### ✅ Backward Compatibility
- Existing `structured_output()` methods still work
- Deprecation warnings guide migration
- Same return types and behavior

## Running with Real Models

To run examples with actual models, configure your credentials:

### Bedrock (AWS)
```bash
aws configure
# or set AWS_PROFILE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Then modify examples to use real models:
```python
from strands.models import BedrockModel, OpenAIModel

# Use Bedrock
agent = Agent(
    model=BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0"),
    output_type=UserProfile
)

# Use OpenAI with native structured output
agent = Agent(
    model=OpenAIModel(model_id="gpt-4"),
    output_type=UserProfile,
    output_mode=NativeMode()
)
```

## Example Scenarios

### 1. Data Extraction
```python
class ContactInfo(BaseModel):
    name: str
    phone: str
    email: str
    company: str

agent = Agent(output_type=ContactInfo)
result = agent("Extract contact info: John Smith, 555-0123, john@acme.com, Acme Corp")
contact = result.get_structured_output(ContactInfo)
```

### 2. Task Management
```python
class Task(BaseModel):
    title: str
    priority: str
    due_date: Optional[str]
    assigned_to: Optional[str]

agent = Agent(output_type=Task)
result = agent("Create a high priority task to review code by Friday, assign to Alice")
task = result.get_structured_output(Task)
```

### 3. Content Analysis
```python
class Sentiment(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float
    key_phrases: List[str]

agent = Agent(output_type=Sentiment)
result = agent("Analyze sentiment: I love this new feature, it's amazing!")
sentiment = result.get_structured_output(Sentiment)
```

### 4. Multi-step Processing
```python
# Step 1: Extract entities
class Entities(BaseModel):
    people: List[str]
    organizations: List[str]
    locations: List[str]

# Step 2: Classify content
class Classification(BaseModel):
    category: str
    subcategory: str
    confidence: float

agent = Agent()
entities_result = agent(text, output_type=Entities)
classification_result = agent(text, output_type=Classification)

entities = entities_result.get_structured_output(Entities)
classification = classification_result.get_structured_output(Classification)
```

## Error Handling

```python
try:
    result = agent("Create user profile", output_type=UserProfile)
    user = result.get_structured_output(UserProfile)
except ValueError as e:
    print(f"Structured output error: {e}")
    # Handle missing or invalid structured output
except Exception as e:
    print(f"General error: {e}")
    # Handle other errors (model, network, etc.)
```

## Performance Tips

1. **Use NativeMode when available** - Faster for supported models
2. **Set default output types** - Avoids repeated parameter passing
3. **Cache agents** - Reuse agents for multiple calls
4. **Handle errors gracefully** - Always wrap in try/catch blocks

## Troubleshooting

### Common Issues

1. **"No structured output available"**
   - Model didn't call the structured output tool
   - Check your prompt clarity
   - Verify model supports the requested format

2. **Type mismatch errors**
   - Calling `get_structured_output()` with wrong type
   - Check the actual returned structure

3. **Model capability warnings**
   - Normal for models without native support
   - System automatically falls back to function calling

### Debug Mode

```python
import logging
logging.getLogger("strands").setLevel(logging.DEBUG)

# Now run your code to see detailed logs
result = agent("Create user", output_type=UserProfile)
```

## Next Steps

1. **Read the full guide**: `../docs/structured_output_guide.md`
2. **Check migration guide**: `../docs/migration_guide.md`
3. **Run component tests**: `python examples/simple_test.py`
4. **Try with your models**: Configure credentials and run examples
5. **Build your own**: Use the patterns in your applications

## Support

- Check the documentation in `../docs/`
- Run `simple_test.py` to validate your setup
- Enable debug logging for troubleshooting
- Review error messages for specific guidance
