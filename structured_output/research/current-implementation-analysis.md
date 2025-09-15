# Current Structured Output Implementation Analysis

## Overview

The Strands Agents SDK currently has a `structured_output` method implemented across different model providers, but it's separate from the main Agent interface and doesn't integrate with the AgentResult return object.

## Current Architecture

### Agent Class Structure
- **Location**: `/src/strands/agent/agent.py`
- **Current Methods**: 
  - `structured_output(output_model: Type[T], prompt: AgentInput = None) -> T`
  - `structured_output_async(output_model: Type[T], prompt: AgentInput = None) -> T`

### AgentResult Class
- **Location**: `/src/strands/agent/agent_result.py`
- **Current Fields**:
  - `stop_reason: StopReason`
  - `message: Message`
  - `metrics: EventLoopMetrics`
  - `state: Any`
- **Issue**: No field for structured output data

### Model Provider Implementations

#### 1. Bedrock Model (`/src/strands/models/bedrock.py`)
- **Approach**: Uses tool calling mechanism
- **Process**:
  1. Converts Pydantic model to tool spec using `convert_pydantic_to_tool_spec()`
  2. Calls `stream()` with tool_choice="any"
  3. Expects `stop_reason="tool_use"`
  4. Extracts tool input and instantiates Pydantic model
- **Issues**: Complex tool-based approach, error-prone parsing

#### 2. OpenAI Model (`/src/strands/models/openai.py`)
- **Approach**: Uses native structured output API
- **Process**:
  1. Uses `client.beta.chat.completions.parse()`
  2. Sets `response_format=output_model`
  3. Directly gets parsed response
- **Issues**: Different API pattern from other providers

#### 3. Anthropic Model (`/src/strands/models/anthropic.py`)
- **Approach**: Uses tool calling mechanism (same as Bedrock)
- **Process**: Identical to Bedrock implementation
- **Issues**: Same complexity and error-prone parsing as Bedrock

## Key Problems Identified

### 1. Inconsistent Implementation Patterns
- **OpenAI**: Uses native structured output API
- **Bedrock/Anthropic**: Uses tool calling workaround
- **Other providers**: May have different approaches or missing implementations

### 2. Separate API Surface
- `structured_output()` is a separate method from the main `__call__()` method
- No integration with the standard AgentResult return object
- Users must choose between regular conversation and structured output

### 3. Error-Prone Tool-Based Approach
- Bedrock and Anthropic rely on tool calling mechanism
- Complex parsing logic to extract structured data from tool responses
- Multiple failure points: tool spec conversion, tool execution, response parsing

### 4. Missing Integration with Agent Workflow
- No way to get structured output as part of normal agent conversation
- No support for structured output in streaming responses
- No integration with agent state management

### 5. Limited Extensibility
- Hard to add structured output support to new model providers
- No standardized interface for structured output configuration
- No fallback mechanisms for providers without native support

## Current Usage Pattern

```python
# Current separate API
agent = Agent()
result = agent.structured_output(MyModel, "Generate data")  # Returns MyModel instance

# vs. Regular usage
agent = Agent()
result = agent("Generate data")  # Returns AgentResult
```

## Missing Capabilities

1. **Unified API**: No way to specify output_type in regular agent calls
2. **AgentResult Integration**: Structured output not included in standard result object
3. **Streaming Support**: No structured output in streaming responses
4. **Fallback Handling**: No graceful degradation for unsupported providers
5. **Configuration**: No way to configure structured output behavior per provider
