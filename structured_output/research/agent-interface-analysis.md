# Agent Interface Analysis for output_type Parameter

## Current Agent.__call__ Method Signature

```python
def __call__(self, prompt: AgentInput = None, **kwargs: Any) -> AgentResult:
```

## Current Flow Analysis

### Synchronous Flow
1. `agent()` → `__call__()` → `invoke_async()` via ThreadPoolExecutor
2. `invoke_async()` → `stream_async()` → consumes all events → returns final `AgentResult`

### Streaming Flow  
1. `agent.stream_async()` → yields events via `event_loop_cycle()`
2. Events processed through `streaming.py` utilities
3. Final event contains `AgentResult` in `event["result"]`

## Integration Points for output_type Parameter

### 1. Method Signature Changes
```python
# Current
def __call__(self, prompt: AgentInput = None, **kwargs: Any) -> AgentResult:

# Proposed  
def __call__(self, prompt: AgentInput = None, output_type: Optional[Type[BaseModel]] = None, **kwargs: Any) -> AgentResult:
```

### 2. Parameter Flow
- `__call__()` → `invoke_async()` → `stream_async()` → `event_loop_cycle()`
- Need to pass `output_type` through this chain
- `event_loop_cycle()` needs to handle structured output logic

### 3. AgentResult Modification
```python
# Current AgentResult (in agent_result.py)
@dataclass
class AgentResult:
    stop_reason: StopReason
    message: Message
    metrics: EventLoopMetrics
    state: Any

# Proposed
@dataclass  
class AgentResult:
    stop_reason: StopReason
    message: Message
    metrics: EventLoopMetrics
    state: Any
    structured_output: Optional[BaseModel] = None  # New field
```

## Key Implementation Considerations

### 1. Backward Compatibility
- `**kwargs` pattern allows adding `output_type` without breaking existing calls
- Default `None` value maintains current behavior
- Existing `structured_output()` method remains unchanged

### 2. Event Loop Integration
- Need to detect when `output_type` is provided
- Must handle structured output parsing in final event processing
- Should integrate with existing streaming infrastructure

### 3. Error Handling Integration
- Current error handling in `event_loop_cycle()` needs extension
- Structured output errors should be handled gracefully
- Fallback strategies need integration with event loop
