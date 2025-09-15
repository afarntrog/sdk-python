# Structured Output Guide

Strands Agents supports structured output parsing that automatically converts model responses into typed Pydantic models. This feature works across all supported model providers with automatic fallback strategies for maximum reliability.

## Quick Start

```python
from strands import Agent
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    occupation: str
    active: bool = True

agent = Agent()

# Get structured output
result = agent("Create a user profile for John, age 30, software engineer", 
               output_type=UserProfile)

print(f"Text: {result}")  # Full text response
print(f"Data: {result.structured_output}")  # UserProfile(name="John", age=30, ...)
```

## Core Features

### Universal Provider Support
- **Native APIs**: OpenAI, LiteLLM (most reliable)
- **JSON Schema**: Ollama, LlamaCpp (good reliability) 
- **Tool Calling**: Bedrock, Anthropic (good reliability)
- **Prompt Engineering**: Universal fallback (works with any provider)

### Automatic Strategy Selection
The system automatically detects your model provider and uses the best available strategy:

```python
# Works with any provider - strategy selected automatically
from strands.models import BedrockModel, OpenAIModel, OllamaModel

# Bedrock uses tool calling strategy
bedrock_agent = Agent(model=BedrockModel())
result = bedrock_agent("Generate data", output_type=MyModel)

# OpenAI uses native structured output API
openai_agent = Agent(model=OpenAIModel())  
result = openai_agent("Generate data", output_type=MyModel)

# Ollama uses JSON schema formatting
ollama_agent = Agent(model=OllamaModel())
result = ollama_agent("Generate data", output_type=MyModel)
```

### Graceful Fallback
If the primary strategy fails, the system automatically falls back to prompt engineering:

```python
# Even if native/tool calling fails, prompt-based fallback ensures reliability
result = agent("Generate data", output_type=MyModel)
# Always returns a result or clear error - never silent failures
```

## Usage Patterns

### 1. Synchronous Usage

```python
from pydantic import BaseModel
from strands import Agent

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool

agent = Agent()

# Simple synchronous call
result = agent(
    "Create product info for a wireless mouse priced at $29.99", 
    output_type=ProductInfo
)

print(f"Product: {result.structured_output.name}")
print(f"Price: ${result.structured_output.price}")
print(f"Available: {result.structured_output.in_stock}")
```

### 2. Asynchronous Usage

```python
import asyncio
from pydantic import BaseModel

class WeatherReport(BaseModel):
    location: str
    temperature: int
    condition: str
    humidity: int

async def get_weather():
    agent = Agent()
    
    result = await agent.invoke_async(
        "Generate weather report for Seattle: 72°F, sunny, 45% humidity",
        output_type=WeatherReport
    )
    
    return result.structured_output

# Usage
weather = asyncio.run(get_weather())
print(f"Weather in {weather.location}: {weather.temperature}°F, {weather.condition}")
```

### 3. Streaming with Structured Output

```python
from pydantic import BaseModel

class StoryOutline(BaseModel):
    title: str
    genre: str
    main_character: str
    plot_summary: str

async def stream_story_creation():
    agent = Agent()
    
    print("Creating story outline...")
    
    async for event in agent.stream_async(
        "Create a sci-fi story outline about space exploration", 
        output_type=StoryOutline
    ):
        if "data" in event:
            # Real-time text streaming
            print(event["data"], end="", flush=True)
        elif "result" in event:
            # Final structured output
            story = event["result"].structured_output
            print(f"\n\nStructured Story Data:")
            print(f"Title: {story.title}")
            print(f"Genre: {story.genre}")
            print(f"Character: {story.main_character}")

# Usage
asyncio.run(stream_story_creation())
```

## Advanced Examples

### Complex Nested Models

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Contact(BaseModel):
    email: str
    phone: Optional[str] = None

class Person(BaseModel):
    name: str
    age: int
    address: Address
    contact: Contact
    skills: List[str]
    created_at: datetime

# Complex nested structure works seamlessly
result = agent("""
Create a person profile:
- Name: Sarah Johnson
- Age: 28  
- Address: 123 Main St, Portland, OR 97201
- Email: sarah@example.com, Phone: 555-0123
- Skills: Python, Machine Learning, Data Analysis
""", output_type=Person)

person = result.structured_output
print(f"Name: {person.name}")
print(f"City: {person.address.city}")
print(f"Skills: {', '.join(person.skills)}")
```

### Data Extraction and Analysis

```python
from pydantic import BaseModel
from typing import List

class Expense(BaseModel):
    date: str
    amount: float
    category: str
    description: str

class ExpenseReport(BaseModel):
    total_amount: float
    expenses: List[Expense]
    categories: List[str]

# Extract structured data from unstructured text
receipt_text = """
Coffee Shop - March 15 - $4.50 - Coffee and pastry
Gas Station - March 16 - $45.00 - Fuel
Restaurant - March 17 - $28.75 - Dinner with client
Office Supplies - March 18 - $12.30 - Notebooks and pens
"""

result = agent(f"""
Analyze this expense data and create a structured report:
{receipt_text}

Calculate the total and categorize each expense.
""", output_type=ExpenseReport)

report = result.structured_output
print(f"Total Expenses: ${report.total_amount}")
print(f"Categories: {', '.join(report.categories)}")
for expense in report.expenses:
    print(f"  {expense.date}: ${expense.amount} - {expense.description}")
```

### API Response Parsing

```python
from pydantic import BaseModel
from typing import List, Optional

class APIError(BaseModel):
    code: int
    message: str
    details: Optional[str] = None

class User(BaseModel):
    id: int
    username: str
    email: str
    active: bool

class APIResponse(BaseModel):
    success: bool
    data: Optional[List[User]] = None
    error: Optional[APIError] = None
    total_count: int = 0

# Parse API-like responses
api_text = """
{
  "success": true,
  "data": [
    {"id": 1, "username": "john_doe", "email": "john@example.com", "active": true},
    {"id": 2, "username": "jane_smith", "email": "jane@example.com", "active": false}
  ],
  "total_count": 2
}
"""

result = agent(f"Parse this API response: {api_text}", output_type=APIResponse)
response = result.structured_output

if response.success and response.data:
    print(f"Found {response.total_count} users:")
    for user in response.data:
        status = "Active" if user.active else "Inactive"
        print(f"  {user.username} ({user.email}) - {status}")
```

## Provider-Specific Examples

### Amazon Bedrock

```python
from strands.models import BedrockModel
from pydantic import BaseModel

class Summary(BaseModel):
    key_points: List[str]
    sentiment: str
    word_count: int

# Bedrock uses tool calling strategy automatically
agent = Agent(model=BedrockModel(model_id="us.amazon.nova-pro-v1:0"))

result = agent("""
Summarize this customer feedback:
"The product arrived quickly and works great. Setup was easy and the customer 
service team was very helpful when I had questions. Highly recommend!"
""", output_type=Summary)

summary = result.structured_output
print(f"Sentiment: {summary.sentiment}")
print(f"Key points: {summary.key_points}")
```

### OpenAI

```python
from strands.models import OpenAIModel
from pydantic import BaseModel

class CodeReview(BaseModel):
    issues_found: List[str]
    suggestions: List[str]
    overall_quality: str
    estimated_fix_time: str

# OpenAI uses native structured output API
agent = Agent(model=OpenAIModel(model_id="gpt-4"))

code = '''
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item.price
    return total
'''

result = agent(f"Review this Python code:\n{code}", output_type=CodeReview)
review = result.structured_output

print(f"Quality: {review.overall_quality}")
print(f"Issues: {review.issues_found}")
print(f"Suggestions: {review.suggestions}")
```

### Ollama (Local Models)

```python
from strands.models.ollama import OllamaModel
from pydantic import BaseModel

class Translation(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float

# Ollama uses JSON schema formatting
agent = Agent(model=OllamaModel(host="http://localhost:11434", model_id="llama3"))

result = agent("""
Translate "Hello, how are you today?" to Spanish and provide translation details.
""", output_type=Translation)

translation = result.structured_output
print(f"Original: {translation.original_text}")
print(f"Translation: {translation.translated_text}")
print(f"Confidence: {translation.confidence}")
```

## Error Handling and Debugging

### Handling Parsing Failures

```python
from pydantic import BaseModel
from strands import Agent

class DataModel(BaseModel):
    name: str
    value: int

agent = Agent()

try:
    result = agent("Generate some data", output_type=DataModel)
    
    if result.structured_output is None:
        print("Structured output parsing failed, but text is available:")
        print(str(result))
    else:
        print(f"Success: {result.structured_output}")
        
except Exception as e:
    print(f"Agent execution failed: {e}")
```

### Monitoring and Metrics

```python
# Access structured output metrics
result = agent("Generate data", output_type=MyModel)

metrics = result.metrics.get_summary()
so_metrics = metrics['structured_output']

print(f"Parsing attempts: {so_metrics['attempts']}")
print(f"Success rate: {so_metrics['success_rate']:.2%}")
print(f"Strategy used: {so_metrics['strategy_used']}")
print(f"Parsing time: {so_metrics['average_parsing_time']:.3f}s")
```

### Debugging Strategy Selection

```python
from strands.structured_output import StructuredOutputManager

# Check what strategies are available for your model
manager = StructuredOutputManager()
capabilities = manager.detect_provider_capabilities(agent.model)

print(f"Available strategies for {agent.model.__class__.__name__}:")
for strategy in capabilities:
    print(f"  - {strategy}")
```

## Best Practices

### 1. Model Design

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class WellDesignedModel(BaseModel):
    """Clear docstring describing the model purpose."""
    
    # Use descriptive field names
    user_name: str = Field(description="Full name of the user")
    age_years: int = Field(description="Age in years", ge=0, le=150)
    
    # Provide defaults for optional fields
    is_active: bool = Field(default=True, description="Whether user is active")
    
    # Use enums for constrained values
    status: str = Field(description="User status", pattern="^(active|inactive|pending)$")
    
    # Include helpful descriptions
    tags: List[str] = Field(default_factory=list, description="User tags or categories")
```

### 2. Prompt Engineering

```python
# Good: Specific, clear instructions
result = agent("""
Create a product listing with the following details:
- Name: Wireless Bluetooth Headphones
- Price: $79.99
- Category: Electronics
- In stock: Yes
- Features: Noise canceling, 20-hour battery, wireless charging

Format as a structured product record.
""", output_type=ProductInfo)

# Avoid: Vague or ambiguous prompts
# result = agent("Make some product data", output_type=ProductInfo)
```

### 3. Error Recovery

```python
async def robust_structured_output(prompt: str, model_class: type):
    """Robust structured output with fallback handling."""
    
    try:
        # Primary attempt
        result = agent(prompt, output_type=model_class)
        
        if result.structured_output is not None:
            return result.structured_output
            
        # Fallback: Try with more explicit prompt
        explicit_prompt = f"""
        {prompt}
        
        Please respond with valid JSON that matches this structure:
        {model_class.model_json_schema()}
        """
        
        result = agent(explicit_prompt, output_type=model_class)
        return result.structured_output
        
    except Exception as e:
        print(f"Structured output failed: {e}")
        return None
```

### 4. Performance Optimization

```python
# For high-throughput scenarios, reuse agent instances
agent = Agent()  # Create once

# Batch processing
async def process_batch(items: List[str]):
    results = []
    
    for item in items:
        result = await agent.invoke_async(
            f"Process: {item}", 
            output_type=MyModel
        )
        results.append(result.structured_output)
    
    return results

# Monitor performance
results = await process_batch(data_items)
metrics = agent.event_loop_metrics.get_summary()
print(f"Average parsing time: {metrics['structured_output']['average_parsing_time']:.3f}s")
```

## Troubleshooting

### Common Issues

1. **Parsing Failures**
   - Check model output format matches Pydantic schema
   - Verify field types and constraints
   - Use simpler models for testing

2. **Performance Issues**
   - Monitor parsing times in metrics
   - Consider simpler models for faster parsing
   - Use appropriate model providers for your use case

3. **Provider Compatibility**
   - All providers supported with automatic fallback
   - Check provider-specific documentation for optimal configuration
   - Test with prompt-based fallback if issues persist

### Getting Help

- Check metrics for detailed performance information
- Use debug logging to see strategy selection
- Test with simpler models to isolate issues
- Verify Pydantic model definitions are correct

## Migration Guide

### From Manual JSON Parsing

```python
# Before: Manual JSON parsing
response = agent("Generate user data")
text = str(response)
try:
    data = json.loads(text)
    user = User(**data)
except (json.JSONDecodeError, ValidationError) as e:
    # Handle parsing errors manually
    pass

# After: Structured output
result = agent("Generate user data", output_type=User)
user = result.structured_output  # Automatically parsed and validated
```

### From Provider-Specific Code

```python
# Before: Provider-specific structured output
if isinstance(model, BedrockModel):
    # Use tool calling approach
    pass
elif isinstance(model, OpenAIModel):
    # Use native API
    pass
else:
    # Manual fallback
    pass

# After: Universal structured output
result = agent("Generate data", output_type=MyModel)
# Works with any provider automatically
```

This comprehensive guide covers all aspects of using structured output with Strands Agents, from basic usage to advanced scenarios and best practices.
