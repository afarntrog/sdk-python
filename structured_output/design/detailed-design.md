# Unified Structured Output Design Document

## Overview

This design implements a unified structured output feature for the Strands Agents SDK that integrates seamlessly with the existing agent interface while providing robust fallback strategies across all model providers.

## Requirements Summary

- **Unified API**: Add `output_type` parameter to existing `agent()` call
- **AgentResult Integration**: Add `structured_output` field to AgentResult
- **Backward Compatibility**: Maintain existing `structured_output()` method
- **Provider Abstraction**: Automatic best-approach selection per provider
- **Graceful Fallback**: Multi-tier fallback strategy (native → JSON → tool → prompt)
- **Streaming Support**: Delayed parsing with streaming text + final structured output
- **Phased Development**: Complete implementation delivered as single release

## Critical Design Update: Full Agent Loop Integration

**IMPORTANT**: The new `output_type` parameter must provide users with **identical** agent loop benefits as the current `agent()` call. This includes:

### Current Agent Loop Benefits (Must Preserve)
- **EventLoopMetrics**: Comprehensive metrics including cycle counts, durations, tool usage, traces
- **Token Usage Tracking**: Input/output tokens, cache read/write tokens, total tokens
- **Tool Execution Metrics**: Tool call counts, success rates, execution times, error tracking
- **Performance Traces**: Detailed execution traces with timing and metadata
- **Streaming Support**: Real-time event streaming with callback handlers
- **Error Handling**: Robust error recovery and reporting
- **State Management**: Conversation history and agent state tracking

### Integration Strategy: Agent Loop First

The structured output feature must be **integrated into the existing agent loop**, not bypass it. This ensures users get all current benefits plus structured output.

#### Modified Architecture Flow
```python
# Current Flow (PRESERVE THIS)
agent("prompt") → __call__ → invoke_async → stream_async → event_loop_cycle → AgentResult

# Enhanced Flow (ADD TO EXISTING)  
agent("prompt", output_type=Model) → __call__ → invoke_async → stream_async → event_loop_cycle → AgentResult + structured_output
```

#### Key Integration Points

##### 1. Event Loop Integration (Primary)
```python
# In event_loop_cycle() - ADD structured output processing to existing loop
async def event_loop_cycle(agent: "Agent", invocation_state: dict[str, Any]) -> AsyncGenerator[TypedEvent, None]:
    """Enhanced event loop with structured output support."""
    
    # Extract structured output parameters
    output_type = invocation_state.get('output_type')
    
    # EXISTING agent loop processing (unchanged)
    start_time, cycle_trace = agent.metrics.start_cycle()
    
    try:
        # Normal streaming processing with full metrics collection
        async for event in stream_messages(agent.model, messages, tool_specs, system_prompt):
            # All existing metrics, traces, tool tracking continues
            yield event
        
        # EXISTING metrics and trace completion
        agent.metrics.end_cycle(start_time, cycle_trace)
        agent.metrics.update_usage(event["usage"])
        agent.metrics.update_metrics(event["metrics"])
        
        # NEW: Add structured output processing to final event (if requested)
        if output_type is not None and "stop" in event:
            structured_output = await agent.structured_output_manager.execute_structured_output(
                model=agent.model,
                output_type=output_type,
                messages=messages,
                system_prompt=system_prompt
            )
            
            # Add structured output to existing event (no new event needed)
            event["structured_output"] = structured_output
        
        yield event
        
    except Exception as e:
        # EXISTING error handling continues to work
        agent.metrics.end_cycle(start_time, cycle_trace)
        raise e
```

##### 2. Metrics Integration (Preserve All)
```python
# EventLoopMetrics must continue to track everything + structured output
@dataclass
class EventLoopMetrics:
    # ALL existing fields preserved
    cycle_count: int = 0
    tool_metrics: Dict[str, ToolMetrics] = field(default_factory=dict)
    cycle_durations: List[float] = field(default_factory=list)
    traces: List[Trace] = field(default_factory=list)
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    
    # NEW: Add structured output metrics
    structured_output_attempts: int = 0
    structured_output_successes: int = 0
    structured_output_strategy_used: Optional[str] = None
    structured_output_parsing_time: float = 0.0
```

##### 3. Streaming Integration (Preserve All)
```python
# Streaming must continue to provide all current events + structured output in final event
async def stream_async(self, prompt, output_type=None, **kwargs):
    """Enhanced streaming with structured output support."""
    
    # ALL existing streaming functionality preserved
    callback_handler = kwargs.get("callback_handler", self.callback_handler)
    messages = self._convert_prompt_to_messages(prompt)
    
    # Pass output_type through to event loop (existing pattern)
    invocation_state = {"output_type": output_type, **kwargs}
    
    # EXISTING event loop with full metrics, traces, tool tracking
    async for event in event_loop_cycle(self, invocation_state):
        # ALL existing events continue to be yielded
        yield event
```

##### 4. Tool Usage Integration (Preserve All)
```python
# Tools must continue to work with structured output
# If user prompt triggers tool usage, all tool metrics must be preserved
# Structured output parsing happens AFTER tool execution completes
# Tool traces, timing, success rates all preserved in EventLoopMetrics
```

### Updated Component Design

#### Enhanced StructuredOutputManager
```python
class StructuredOutputManager:
    """Integrates with agent loop to provide structured output without losing benefits."""
    
    async def execute_structured_output(
        self,
        model: Model,
        output_type: Type[T],
        messages: Messages,
        system_prompt: Optional[str] = None,
        metrics: Optional[EventLoopMetrics] = None  # NEW: Accept metrics for tracking
    ) -> Optional[T]:
        """Execute structured output with metrics integration."""
        
        if metrics:
            metrics.structured_output_attempts += 1
            start_time = time.time()
        
        try:
            # Execute fallback strategy
            result = await self._execute_with_fallback(model, output_type, messages, system_prompt)
            
            if metrics:
                if result is not None:
                    metrics.structured_output_successes += 1
                metrics.structured_output_parsing_time += time.time() - start_time
                
            return result
            
        except Exception as e:
            if metrics:
                metrics.structured_output_parsing_time += time.time() - start_time
            # Don't break agent loop - log error and return None
            logger.warning(f"Structured output failed: {e}")
            return None
```

#### Backward Compatibility Guarantee
```python
# EXISTING method must continue to work identically
def structured_output(self, output_model: Type[T], prompt: AgentInput = None) -> T:
    """Existing method - unchanged behavior."""
    # Current implementation preserved exactly
    # This bypasses agent loop (as it currently does)
    # Users who want agent loop benefits should use new output_type parameter
```

### Testing Strategy Update

#### Metrics Preservation Tests
```python
class TestAgentLoopIntegration:
    """Ensure structured output preserves all agent loop benefits."""
    
    def test_metrics_preserved_with_structured_output(self):
        """Verify EventLoopMetrics identical with/without output_type."""
        # Test that cycle counts, tool metrics, traces, usage all identical
        
    def test_streaming_events_preserved(self):
        """Verify streaming events identical with/without output_type."""
        # Test that all streaming events continue to be yielded
        
    def test_tool_usage_preserved_with_structured_output(self):
        """Verify tool execution metrics preserved when using output_type."""
        # Test that tool calls, timing, success rates all tracked
        
    def test_error_handling_preserved(self):
        """Verify error handling identical with/without output_type."""
        # Test that agent loop errors handled identically
```

## Architecture Summary

The key insight is that structured output must be **additive** to the agent loop, not **alternative** to it. Users should get:

1. **All existing agent loop benefits** (metrics, streaming, tools, traces, error handling)
2. **Plus structured output** in the AgentResult when requested
3. **Zero degradation** in performance, reliability, or functionality
4. **Identical behavior** for all existing features

This ensures the new feature enhances the SDK without compromising any existing value.

### Core Components

#### 1. Agent Interface Extension
```python
class Agent:
    def __call__(
        self, 
        prompt: AgentInput = None, 
        output_type: Optional[Type[BaseModel]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """Enhanced agent call with optional structured output."""
```

#### 2. Enhanced AgentResult
```python
@dataclass
class AgentResult:
    stop_reason: StopReason
    message: Message
    metrics: EventLoopMetrics
    state: Any
    structured_output: Optional[BaseModel] = None  # New field
```

#### 3. Structured Output Manager
```python
class StructuredOutputManager:
    """Coordinates structured output across providers with fallback strategies."""
    
    async def execute_structured_output(
        self,
        model: Model,
        output_type: Type[T],
        messages: Messages,
        system_prompt: Optional[str] = None
    ) -> Optional[T]:
        """Execute structured output with automatic fallback."""
```

#### 4. Provider Strategy System
```python
class StructuredOutputStrategy(ABC):
    """Abstract base for structured output strategies."""
    
    @abstractmethod
    async def execute(self, model: Model, output_type: Type[T], messages: Messages) -> Optional[T]:
        pass

class NativeStrategy(StructuredOutputStrategy):
    """For OpenAI, LiteLLM with native support."""

class JsonSchemaStrategy(StructuredOutputStrategy):
    """For Ollama, LlamaCpp with JSON schema support."""

class ToolCallingStrategy(StructuredOutputStrategy):
    """For Bedrock, Anthropic using tool calling."""

class PromptBasedStrategy(StructuredOutputStrategy):
    """Universal fallback for all providers."""
```

## Components and Interfaces

### 1. Agent Interface Changes

#### Method Signature Extension
```python
# Current
def __call__(self, prompt: AgentInput = None, **kwargs: Any) -> AgentResult:

# Enhanced
def __call__(
    self, 
    prompt: AgentInput = None, 
    output_type: Optional[Type[BaseModel]] = None,
    **kwargs: Any
) -> AgentResult:
```

#### Parameter Flow
```python
# Flow: __call__ → invoke_async → stream_async → event_loop_cycle
# Need to pass output_type through entire chain

def __call__(self, prompt, output_type=None, **kwargs):
    kwargs['output_type'] = output_type
    return asyncio.run(self.invoke_async(prompt, **kwargs))

async def invoke_async(self, prompt, **kwargs):
    output_type = kwargs.pop('output_type', None)
    events = self.stream_async(prompt, output_type=output_type, **kwargs)
    # Process events and extract structured output
```

### 2. Structured Output Manager

#### Core Interface
```python
class StructuredOutputManager:
    def __init__(self):
        self.strategies = {
            'native': NativeStrategy(),
            'json_schema': JsonSchemaStrategy(), 
            'tool_calling': ToolCallingStrategy(),
            'prompt_based': PromptBasedStrategy()
        }
    
    def detect_provider_capabilities(self, model: Model) -> List[str]:
        """Detect which strategies a provider supports."""
        capabilities = []
        
        # Check for native support
        if self._has_native_support(model):
            capabilities.append('native')
            
        # Check for JSON schema support
        if self._has_json_schema_support(model):
            capabilities.append('json_schema')
            
        # Check for tool calling support
        if self._has_tool_calling_support(model):
            capabilities.append('tool_calling')
            
        # Prompt-based always available
        capabilities.append('prompt_based')
        
        return capabilities
    
    async def execute_structured_output(
        self,
        model: Model,
        output_type: Type[T],
        messages: Messages,
        system_prompt: Optional[str] = None
    ) -> Optional[T]:
        """Execute with automatic fallback."""
        capabilities = self.detect_provider_capabilities(model)
        
        for capability in capabilities:
            strategy = self.strategies[capability]
            try:
                result = await strategy.execute(model, output_type, messages, system_prompt)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Strategy {capability} failed: {e}")
                continue
        
        return None  # All strategies failed
```

### 3. Provider Strategy Implementations

#### Native Strategy (OpenAI, LiteLLM)
```python
class NativeStrategy(StructuredOutputStrategy):
    async def execute(self, model: Model, output_type: Type[T], messages: Messages, system_prompt: Optional[str] = None) -> Optional[T]:
        """Use provider's native structured output API."""
        # Delegate to existing model.structured_output() for native providers
        events = model.structured_output(output_type, messages, system_prompt)
        async for event in events:
            if "output" in event:
                return event["output"]
        return None
```

#### JSON Schema Strategy (Ollama, LlamaCpp)
```python
class JsonSchemaStrategy(StructuredOutputStrategy):
    async def execute(self, model: Model, output_type: Type[T], messages: Messages, system_prompt: Optional[str] = None) -> Optional[T]:
        """Use JSON schema format parameter."""
        # Similar to current Ollama implementation
        formatted_request = model.format_request(messages=messages, system_prompt=system_prompt)
        formatted_request["format"] = output_type.model_json_schema()
        
        # Execute request and parse response
        response = await model._execute_request(formatted_request)
        content = response.message.content.strip()
        
        try:
            return output_type.model_validate_json(content)
        except Exception as e:
            raise StructuredOutputError(f"JSON parsing failed: {e}") from e
```

#### Tool Calling Strategy (Bedrock, Anthropic)
```python
class ToolCallingStrategy(StructuredOutputStrategy):
    async def execute(self, model: Model, output_type: Type[T], messages: Messages, system_prompt: Optional[str] = None) -> Optional[T]:
        """Use tool calling mechanism with improved error handling."""
        # Delegate to existing model.structured_output() for tool-based providers
        # This maintains current functionality while providing consistent interface
        events = model.structured_output(output_type, messages, system_prompt)
        async for event in events:
            if "output" in event:
                return event["output"]
        return None
```

#### Prompt-Based Strategy (Universal Fallback)
```python
class PromptBasedStrategy(StructuredOutputStrategy):
    async def execute(self, model: Model, output_type: Type[T], messages: Messages, system_prompt: Optional[str] = None) -> Optional[T]:
        """Universal fallback using prompt engineering."""
        
        # Generate structured output prompt
        schema_str = self._pydantic_to_prompt_schema(output_type)
        example_str = self._generate_example_json(output_type)
        user_prompt = self._extract_user_prompt(messages)
        
        structured_prompt = STRUCTURED_OUTPUT_PROMPT.format(
            json_schema=schema_str,
            example_json=example_str,
            user_prompt=user_prompt
        )
        
        # Create new messages with structured prompt
        structured_messages = self._create_structured_messages(messages, structured_prompt)
        
        # Get response using regular streaming
        response_chunks = model.stream(structured_messages, system_prompt=system_prompt)
        full_response = await self._collect_full_response(response_chunks)
        
        # Extract and parse JSON
        json_data = self._extract_json_from_response(full_response)
        if json_data:
            try:
                return output_type(**json_data)
            except ValidationError as e:
                raise StructuredOutputError(f"Validation failed: {e}") from e
        
        return None
```

### 4. Streaming Integration

#### Event Loop Modification
```python
# In event_loop_cycle()
async def event_loop_cycle(agent: "Agent", invocation_state: dict[str, Any]) -> AsyncGenerator[TypedEvent, None]:
    """Enhanced event loop with structured output support."""
    
    # Extract structured output parameters
    output_type = invocation_state.get('output_type')
    
    # Normal streaming processing
    async for event in stream_messages(agent.model, messages, tool_specs, system_prompt):
        yield event
    
    # If structured output requested, process in final event
    if output_type is not None and "stop" in event:
        structured_output = await agent.structured_output_manager.execute_structured_output(
            model=agent.model,
            output_type=output_type,
            messages=messages,
            system_prompt=system_prompt
        )
        
        # Add structured output to final event
        event["structured_output"] = structured_output
    
    yield event
```

#### Enhanced Event Types
```python
@dataclass
class ModelStopReason:
    stop_reason: StopReason
    message: Message
    usage: Usage
    metrics: Metrics
    structured_output: Optional[BaseModel] = None  # New field
```

## Data Models

### Enhanced AgentResult
```python
@dataclass
class AgentResult:
    """Enhanced result object with structured output support."""
    stop_reason: StopReason
    message: Message
    metrics: EventLoopMetrics
    state: Any
    structured_output: Optional[BaseModel] = None
    
    def __str__(self) -> str:
        """Maintain existing string behavior."""
        content_array = self.message.get("content", [])
        result = ""
        for item in content_array:
            if isinstance(item, dict) and "text" in item:
                result += item.get("text", "") + "\n"
        return result
```

### Error Handling
```python
class StructuredOutputError(Exception):
    """Base exception for structured output failures."""
    def __init__(self, message: str, provider: str = None, strategy: str = None):
        super().__init__(message)
        self.provider = provider
        self.strategy = strategy

class StructuredOutputValidationError(StructuredOutputError):
    """Pydantic validation failed."""
    pass

class StructuredOutputParsingError(StructuredOutputError):
    """JSON parsing failed."""
    pass
```

## Testing Strategy

### Unit Tests (Mock-Based)
```python
class TestStructuredOutputIntegration:
    """Test core agent interface integration."""
    
    def test_agent_call_with_output_type(self):
        """Test agent("prompt", output_type=Model) returns AgentResult with structured_output."""
        
    def test_agent_call_without_output_type(self):
        """Test backward compatibility - no structured_output field when not requested."""
        
    def test_structured_output_method_compatibility(self):
        """Test existing agent.structured_output() method still works."""

class TestStructuredOutputManager:
    """Test fallback strategy coordination."""
    
    def test_provider_capability_detection(self):
        """Test automatic detection of provider capabilities."""
        
    def test_fallback_chain_execution(self):
        """Test fallback from native → JSON → tool → prompt."""
        
    def test_error_handling_and_recovery(self):
        """Test graceful handling of strategy failures."""

class TestProviderStrategies:
    """Test individual strategy implementations."""
    
    def test_native_strategy(self):
        """Test native API strategy for OpenAI/LiteLLM."""
        
    def test_json_schema_strategy(self):
        """Test JSON schema strategy for Ollama."""
        
    def test_tool_calling_strategy(self):
        """Test tool calling strategy for Bedrock/Anthropic."""
        
    def test_prompt_based_strategy(self):
        """Test prompt-based fallback strategy."""
```

### Integration Tests (Real Providers)
```python
class TestRealProviderIntegration:
    """Integration tests against real model providers."""
    
    @pytest.mark.integration
    def test_bedrock_structured_output(self):
        """Test structured output with real Bedrock models."""
        
    @pytest.mark.integration  
    def test_openai_structured_output(self):
        """Test structured output with real OpenAI models."""
        
    @pytest.mark.integration
    def test_anthropic_structured_output(self):
        """Test structured output with real Anthropic models."""
        
    @pytest.mark.integration
    def test_streaming_with_structured_output(self):
        """Test streaming + structured output across providers."""
        
    @pytest.mark.integration
    def test_fallback_scenarios(self):
        """Test fallback behavior with real provider failures."""
```

## Implementation Plan Summary

The implementation will be developed in three phases but delivered as a complete solution:

### Phase 1: Core Integration
- Add `output_type` parameter to Agent interface
- Extend AgentResult with `structured_output` field
- Create StructuredOutputManager with basic strategy system
- Implement streaming integration

### Phase 2: Strategy Implementation  
- Implement all four strategy types (Native, JSON Schema, Tool Calling, Prompt-based)
- Add provider capability detection
- Implement fallback chain execution
- Enhanced error handling and recovery

### Phase 3: Testing and Polish
- Comprehensive unit test suite
- Integration tests for all providers
- Performance optimization
- Documentation and examples

All phases will be completed before release to ensure users receive a fully polished, comprehensive structured output feature.
