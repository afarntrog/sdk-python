# Streaming Integration Analysis

## Current Streaming Architecture

Based on the code analysis, the streaming system works as follows:

### Event Flow
1. `agent.stream_async()` → `event_loop_cycle()` → `stream_messages()` → `process_stream()`
2. `process_stream()` yields individual `TypedEvent` objects during streaming
3. Final event is `ModelStopReason` containing the complete message

### Key Integration Points

#### 1. Final Event Processing
```python
# Current final event in process_stream()
yield ModelStopReason(
    stop_reason=stop_reason, 
    message=state["message"], 
    usage=usage, 
    metrics=metrics
)
```

#### 2. AgentResult Creation
```python
# In invoke_async() - consumes all events and returns final result
async for event in events:
    _ = event
return cast(AgentResult, event["result"])
```

## Streaming + Structured Output Integration Design

### Approach: Delayed Parsing in Final Event

#### 1. Detect Structured Output Request
```python
# In event_loop_cycle() or stream_messages()
if output_type is not None:
    # Set flag to enable structured output processing
    invocation_state["structured_output_requested"] = True
    invocation_state["output_type"] = output_type
```

#### 2. Stream Text Normally
- All streaming events proceed as normal
- Users see real-time text generation
- No changes to intermediate events

#### 3. Parse Structured Output in Final Event
```python
# Modified process_stream() final event
async def process_stream(chunks, output_type=None):
    # ... existing streaming logic ...
    
    # Before final yield
    structured_output = None
    if output_type is not None:
        structured_output = await parse_structured_output(
            message=state["message"],
            output_type=output_type,
            model=model  # Need to pass model reference
        )
    
    yield ModelStopReason(
        stop_reason=stop_reason,
        message=state["message"], 
        usage=usage,
        metrics=metrics,
        structured_output=structured_output  # New field
    )
```

#### 4. AgentResult Integration
```python
# Modified AgentResult creation
@dataclass
class AgentResult:
    stop_reason: StopReason
    message: Message
    metrics: EventLoopMetrics
    state: Any
    structured_output: Optional[BaseModel] = None  # New field

# In invoke_async()
final_event = None
async for event in events:
    final_event = event

# Extract structured output from final event
result = AgentResult(
    stop_reason=final_event["stop_reason"],
    message=final_event["message"],
    metrics=final_event["metrics"], 
    state=final_event["state"],
    structured_output=final_event.get("structured_output")  # New
)
```

## Implementation Challenges

### 1. Model Reference Passing
- `process_stream()` doesn't currently have access to the model instance
- Need to pass model reference through the event chain
- Required for fallback strategy execution

### 2. Error Handling in Streaming
- Structured output parsing errors shouldn't break streaming
- Need graceful degradation: stream succeeds, structured_output=None
- Error logging without disrupting user experience

### 3. Event Type Extensions
```python
# Need to extend ModelStopReason event type
@dataclass
class ModelStopReason:
    stop_reason: StopReason
    message: Message
    usage: Usage
    metrics: Metrics
    structured_output: Optional[BaseModel] = None  # New field
```

## Streaming User Experience

### Expected Behavior
```python
# User sees streaming text in real-time
async for event in agent.stream_async("Generate user data", output_type=UserModel):
    if "data" in event:
        print(event["data"], end="")  # Real-time text
    
    # Only final event has structured_output
    if event.get("complete"):
        user_data = event.get("structured_output")  # UserModel instance or None
        if user_data:
            print(f"\nParsed: {user_data}")
```

### Backward Compatibility
- Existing streaming code continues to work unchanged
- New `structured_output` field only present when requested
- No breaking changes to existing event structure

## Performance Considerations

### Parsing Overhead
- Structured output parsing happens after streaming completes
- No impact on streaming performance
- Parsing time added to total request time (acceptable tradeoff)

### Memory Usage
- Complete response text held in memory for parsing
- Same as current behavior, no additional overhead
- Structured output object adds minimal memory footprint

### Error Recovery
- If structured parsing fails, streaming result still available
- Users get text response even if structured parsing fails
- Graceful degradation maintains reliability
