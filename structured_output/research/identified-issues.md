# Identified Issues with Current Structured Output Implementation

## Core Problems

### 1. **Fragmented API Design**
- **Issue**: Two separate methods for agent interaction
  - `agent("prompt")` → Returns `AgentResult`
  - `agent.structured_output(Model, "prompt")` → Returns `Model` instance
- **Impact**: Users must choose between conversation flow and structured output
- **Root Cause**: No integration between structured output and main agent interface

### 2. **Inconsistent Model Provider Implementations**

#### OpenAI vs Others Disparity
- **OpenAI**: Uses native `client.beta.chat.completions.parse()` API
- **Bedrock/Anthropic**: Uses tool calling workaround via `convert_pydantic_to_tool_spec()`
- **Other Providers**: May have different or missing implementations

#### Tool-Based Approach Issues (Bedrock/Anthropic)
```python
# Current problematic flow:
1. Convert Pydantic model → Tool spec
2. Force tool_choice="any" 
3. Parse tool response back to Pydantic
4. Multiple failure points in parsing logic
```

### 3. **AgentResult Integration Gap**
- **Current AgentResult fields**:
  ```python
  @dataclass
  class AgentResult:
      stop_reason: StopReason
      message: Message
      metrics: EventLoopMetrics
      state: Any
      # Missing: structured_output field
  ```
- **Issue**: No way to access structured output through standard result object
- **Impact**: Inconsistent return patterns across SDK

### 4. **Error-Prone Parsing Logic**
From Bedrock/Anthropic implementations:
```python
# Fragile parsing code:
for block in content:
    if block.get("toolUse") and block["toolUse"]["name"] == tool_spec["name"]:
        output_response = block["toolUse"]["input"]  # Can fail
    else:
        continue

if output_response is None:
    raise ValueError("No valid tool use found")  # Common failure
```

### 5. **Missing Streaming Support**
- Current structured output is non-streaming only
- No way to get structured output in streaming agent responses
- Breaks consistency with main agent streaming capabilities

### 6. **No Fallback Mechanisms**
- No graceful degradation for providers without native structured output
- No prompt-based fallback for unsupported models
- Hard failures instead of best-effort attempts

### 7. **Limited Configuration Options**
- No way to configure structured output behavior per provider
- No validation settings or error handling preferences
- No way to specify output format preferences

## Specific Technical Issues

### Tool Spec Conversion Problems
- `convert_pydantic_to_tool_spec()` creates complex nested schemas
- Tool calling mechanism not designed for structured output
- Schema flattening logic can lose important type information

### Response Parsing Brittleness
- Multiple providers expect different response formats
- Tool response parsing assumes specific JSON structure
- No validation of intermediate parsing steps

### Type Safety Issues
- Generic `T` type handling inconsistent across providers
- No runtime type validation in some code paths
- Pydantic model instantiation can fail silently

## Impact on Users

### Developer Experience Issues
1. **API Confusion**: Two different patterns for similar functionality
2. **Provider Lock-in**: Code works differently across providers
3. **Error Debugging**: Complex failure modes in tool-based approach
4. **Feature Gaps**: Can't use structured output with streaming or conversation history

### Production Reliability Issues
1. **Parsing Failures**: Tool-based approach has multiple failure points
2. **Provider Differences**: Same code behaves differently across providers
3. **Error Handling**: Inconsistent error types and messages
4. **Performance**: Tool calling overhead for simple structured output

## Evidence from Tests
From test files, current expectations:
- `agent.structured_output(Model, prompt)` returns `Model` instance
- Separate from main agent conversation flow
- No integration with `AgentResult`
- Provider-specific mocking required for tests

## Root Cause Analysis
The fundamental issue is that structured output was added as an afterthought rather than being designed as a core feature of the agent interface. This led to:
1. Separate API surface instead of integrated parameter
2. Provider-specific workarounds instead of unified approach
3. Missing integration with core agent features (streaming, results, state)
