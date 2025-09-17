# Structured Output Revamp Implementation Plan

## Executive Summary

This plan outlines a comprehensive revamp of the Strands SDK's structured output approach to align with PydanticAI's more intuitive developer experience. The current implementation requires users to call a separate `structured_output()` method, which bypasses the main agent loop and prevents proper metrics collection. The new approach will allow users to specify output types directly in agent invocations while maintaining backward compatibility.

**Note: This implementation will only support Pydantic models as output types for now. Support for other output formats (raw JSON, dataclasses, etc.) may be added in future iterations.**

## Current State Analysis

### Current Implementation Issues

1. **Separated from Main Agent Loop**: The current `agent.structured_output()` method bypasses the main event loop, preventing:
   - Proper metrics collection and telemetry
   - Tool usage during structured output generation
   - Event streaming and hooks integration
   - Consistent conversation management

2. **Non-intuitive Developer Experience**: Users must switch between two different patterns:
   ```python
   # Regular conversation
   result = agent("Generate a summary")

   # Structured output (different method)
   result = agent.structured_output(SummaryModel, "Generate a summary")
   ```

3. **Limited Integration**: Current approach doesn't participate in:
   - Event loop metrics
   - Callback handlers during structured output
   - Tool execution within structured output workflows

### Current Architecture

The current structured output flow:
1. `Agent.structured_output()` → `Model.structured_output()` → Direct model call
2. Bypasses event loop entirely
3. Temporary message handling without conversation persistence
4. Limited to simple prompt-to-structured-output scenarios

## PydanticAI Analysis

### Key Patterns from PydanticAI

1. **Agent-Level Output Type Definition**:
   ```python
   agent = Agent(model, output_type=MyModel)
   result = agent.run_sync("prompt")  # Returns MyModel instance
   ```

2. **Runtime Output Type Specification**:
   ```python
   result = agent.run_sync("prompt", output_type=MyModel)
   ```

3. **Multiple Output Type Support**:
   ```python
   agent = Agent(model, output_type=[ModelA, ModelB])
   ```

4. **Various Output Modes**:
   - `ToolOutput`: Function calling for structured output
   - `NativeOutput`: Native model structured output capabilities
   - `PromptedOutput`: Prompting-based approach
   - `TextOutput`: Post-processing of text output
   - `StructuredDict`: JSON schema validation

5. **Full Integration**: All approaches participate in the main agent execution loop with proper metrics, streaming, and tool integration.

## Proposed New Architecture

### 1. Core Output Type System

Create a new output type system in `src/strands/output/`:

```
src/strands/output/
├── __init__.py
├── base.py          # Abstract base classes
├── types.py         # Output type implementations
├── modes.py         # Output mode handlers
└── registry.py      # Output type registry and resolution
```

#### Base Classes (`base.py`)
```python
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Type, Union, Optional
from pydantic import BaseModel

T = TypeVar("T")

class OutputMode(ABC):
    """Base class for different structured output modes"""

    @abstractmethod
    def get_tool_specs(self, output_types: list[Type[BaseModel]]) -> list[ToolSpec]:
        """Convert output types to tool specifications"""
        pass

    @abstractmethod
    def extract_result(self, model_response: Any) -> Any:
        """Extract structured result from model response"""
        pass

class OutputSchema:
    """Container for output type information and processing mode"""

    def __init__(
        self,
        types: Union[Type[BaseModel], list[Type[BaseModel]]],  # Only Pydantic models supported
        mode: Optional[OutputMode] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.types = types if isinstance(types, list) else [types]
        self.mode = mode or ToolOutput()  # Default to tool-based approach
        self.name = name
        self.description = description
```

#### Output Mode Implementations (`modes.py`)
```python
class ToolOutput(OutputMode):
    """Use function calling for structured output (DEFAULT)

    This is the most reliable approach across all model providers and ensures
    consistent behavior regardless of model capabilities.
    """

    def get_tool_specs(self, output_types: list[Type[BaseModel]]) -> list[ToolSpec]:
        return [convert_pydantic_to_tool_spec(model) for model in output_types]

    def is_supported_by_model(self, model: Model) -> bool:
        """Tool-based output is supported by all models that support function calling"""
        return True  # All our models support function calling

class NativeOutput(OutputMode):
    """Use model's native structured output capabilities

    Only use when explicitly requested and supported by the model.
    Falls back to ToolOutput if not supported.
    """

    def get_tool_specs(self, output_types: list[Type[BaseModel]]) -> list[ToolSpec]:
        # Return empty list - will use native JSON schema instead
        return []

    def is_supported_by_model(self, model: Model) -> bool:
        """Check if model supports native structured output"""
        return model.supports_native_structured_output()

class PromptedOutput(OutputMode):
    """Use prompting to guide output format

    Only use when explicitly requested. Less reliable than tool-based approach
    but can work with models that have limited function calling support.
    """

    def __init__(self, template: Optional[str] = None):
        self.template = template or "Please respond with JSON matching this schema: {schema}"

    def get_tool_specs(self, output_types: list[Type[BaseModel]]) -> list[ToolSpec]:
        # Return empty list - will inject schema into system prompt instead
        return []

    def is_supported_by_model(self, model: Model) -> bool:
        """Prompting-based output works with all models"""
        return True
```

### 2. Enhanced AgentResult Class

Update the AgentResult class to include structured output:

```python
@dataclass
class AgentResult:
    """Represents the last result of invoking an agent with a prompt.

    Attributes:
        stop_reason: The reason why the agent's processing stopped.
        message: The last message generated by the agent.
        metrics: Performance metrics collected during processing.
        state: Additional state information from the event loop.
        structured_output: Parsed structured output when output_type was specified.
    """

    stop_reason: StopReason
    message: Message
    metrics: EventLoopMetrics
    state: Any
    structured_output: Optional[Any] = None  # NEW: Contains parsed structured output

    def get_structured_output(self, output_type: Type[T]) -> T:
        """Get structured output with type safety.

        Args:
            output_type: Expected output type for type checking

        Returns:
            Structured output cast to the expected type

        Raises:
            ValueError: If no structured output available or type mismatch
        """
        if self.structured_output is None:
            raise ValueError("No structured output available in this result")

        if not isinstance(self.structured_output, output_type):
            raise ValueError(f"Structured output type mismatch: expected {output_type}, got {type(self.structured_output)}")

        return self.structured_output
```

### 3. Enhanced Agent Interface

#### Agent Constructor Enhancement
```python
class Agent:
    def __init__(
        self,
        model: Optional[Model] = None,
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
        output_type: Optional[Union[Type[BaseModel], list[Type[BaseModel]], OutputSchema]] = None,  # Only Pydantic models
        output_mode: Optional[OutputMode] = None,
        # ... existing parameters
    ):
        # ... existing initialization
        self.default_output_schema = self._resolve_output_schema(output_type, output_mode)

    def _resolve_output_schema(self, output_type, output_mode) -> Optional[OutputSchema]:
        """Resolve output type and mode into OutputSchema with tool-based default"""
        if not output_type:
            return None

        # Default to ToolOutput if no mode specified
        resolved_mode = output_mode or ToolOutput()

        # Validate mode is supported by current model
        if not resolved_mode.is_supported_by_model(self.model):
            if isinstance(resolved_mode, NativeOutput):
                # Fallback to tool-based approach for native output
                logger.warning(
                    f"Model {self.model.__class__.__name__} does not support native structured output. "
                    "Falling back to tool-based approach."
                )
                resolved_mode = ToolOutput()
            elif isinstance(resolved_mode, PromptedOutput):
                # This shouldn't happen as all models support prompting
                raise ValueError(f"Model {self.model.__class__.__name__} does not support prompting")

        return OutputSchema(output_type, resolved_mode)
```

#### Enhanced __call__ Method
```python
def __call__(
    self,
    prompt: AgentInput = None,
    output_type: Optional[Union[Type[BaseModel], list[Type[BaseModel]], OutputSchema]] = None,  # Only Pydantic models
    output_mode: Optional[OutputMode] = None,
    **kwargs: Any
) -> AgentResult:
    """
    Enhanced agent call supporting inline output type specification

    Args:
        prompt: User input
        output_type: Output type specification (overrides agent default)
        output_mode: Output mode (overrides agent default)
        **kwargs: Additional parameters

    Returns:
        AgentResult containing response message and optionally structured output
    """
    output_schema = self._resolve_output_schema(output_type, output_mode) or self.default_output_schema

    # Always return AgentResult, but include structured output when specified
    return self._run_with_output_schema(prompt, output_schema, **kwargs)
```

### 4. Event Loop Integration

#### Enhanced Event Loop Parameters
```python
async def event_loop_cycle(
    messages: Messages,
    model: Model,
    tool_registry: ToolRegistry,
    tool_executor: ToolExecutor,
    system_prompt: Optional[str] = None,
    output_schema: Optional[OutputSchema] = None,  # NEW
    # ... existing parameters
) -> AsyncGenerator[TypedEvent, None]:
```

#### Structured Output Event Loop Flow
1. **Tool Registration**: Add output type tools to the tool registry for the session
2. **Model Invocation**: Include output type tools in model calls
3. **Response Processing**: Detect structured output tool calls and extract results
4. **Result Streaming**: Stream structured output as it's built
5. **Metrics Collection**: Track structured output metrics in event loop metrics

### 5. Model Provider Updates

#### Enhanced Model Interface
```python
class Model(abc.ABC):
    @abc.abstractmethod
    def supports_native_structured_output(self) -> bool:
        """Check if model supports native structured output"""
        pass

    @abc.abstractmethod
    def get_structured_output_config(
        self,
        output_schema: OutputSchema
    ) -> dict[str, Any]:
        """Get model-specific structured output configuration"""
        pass
```

#### Provider-Specific Implementations
All providers default to tool-based approach unless explicitly overridden:

- **Bedrock**:
  - Default: ToolOutput (function calling)
  - Native: Not supported, falls back to ToolOutput
  - Prompted: Available when explicitly requested

- **OpenAI**:
  - Default: ToolOutput (function calling)
  - Native: Available when explicitly requested (structured_outputs=True)
  - Prompted: Available when explicitly requested

- **Anthropic**:
  - Default: ToolOutput (function calling)
  - Native: Not supported, falls back to ToolOutput
  - Prompted: Available when explicitly requested

- **Others (Ollama, LiteLLM, etc.)**:
  - Default: ToolOutput (function calling)
  - Native: Model-dependent, falls back to ToolOutput if not supported
  - Prompted: Available when explicitly requested

### 6. Backward Compatibility

#### Deprecation Strategy
```python
def structured_output(self, output_model: Type[T], prompt: AgentInput = None) -> T:
    """
    Legacy structured output method (DEPRECATED)

    This method is maintained for backward compatibility.
    Consider using agent(prompt, output_type=model) instead.
    """
    warnings.warn(
        "structured_output() is deprecated. Use agent(prompt, output_type=model) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    result = self(prompt, output_type=output_model)
    return result.get_structured_output(output_model)
```

### 7. Enhanced Developer Experience

#### Type-Safe Patterns
```python
# Agent-level output type (uses ToolOutput by default)
agent = Agent(model, output_type=UserProfile)  # Uses function calling
result = agent("Extract user info from: John Doe, age 30")  # Returns AgentResult
user = result.get_structured_output(UserProfile)  # Extract UserProfile

# Runtime output type specification (uses ToolOutput by default)
result = agent("Summarize this text", output_type=Summary)  # Uses function calling
summary = result.get_structured_output(Summary)  # Extract Summary

# Explicit output mode specification
precise_agent = Agent(model, output_type=Data, output_mode=NativeOutput())  # Uses native if supported
fast_agent = Agent(model, output_type=Data, output_mode=PromptedOutput())   # Uses prompting

# Runtime mode override
result = agent("Extract data", output_type=Data, output_mode=NativeOutput())  # Uses native if supported

# Multiple output types (still uses ToolOutput by default)
agent = Agent(model, output_type=[Person, Company])  # Creates multiple function tools
result = agent("What is Apple Inc?")  # Returns AgentResult
entity = result.structured_output  # Returns Person | Company instance

# Regular text responses (no structured output)
result = agent("Hello, how are you?")  # Returns AgentResult
text = str(result)  # Get text response
# result.structured_output is None

# Fallback behavior for unsupported modes
try:
    # If model doesn't support native, automatically falls back to ToolOutput
    result = agent("Extract", output_type=Data, output_mode=NativeOutput())
except ValueError:
    # Only raises if something is fundamentally wrong
    pass
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
1. **Create Output Type System**
   - Implement base classes in `src/strands/output/base.py`
   - Create output mode implementations in `src/strands/output/modes.py`
   - Add output schema registry in `src/strands/output/registry.py`

2. **Enhance Agent Constructor**
   - Add output_type and output_mode parameters
   - Implement _resolve_output_schema method
   - Add type validation and error handling

3. **Update Type Definitions**
   - Add new type definitions to `src/strands/types/`
   - Update existing interfaces for output schema support

### Phase 2: Event Loop Integration (Week 2-3)
1. **Enhance Event Loop**
   - Add output_schema parameter to event_loop_cycle
   - Implement structured output tool registration
   - Add output extraction logic in event loop

2. **Update Tool System**
   - Enhance tool registry for dynamic output type tools
   - Update tool executor for structured output handling
   - Add output type validation in tool results

3. **Streaming Support**
   - Implement structured output streaming events
   - Add progressive result building for complex outputs
   - Update callback handlers for structured output events

### Phase 3: Model Provider Updates (Week 3-4)
1. **Enhance Model Interface**
   - Add structured output capability detection
   - Implement model-specific output configurations
   - Update all model providers for structured output support

2. **Provider-Specific Implementation**
   - All providers: Default to ToolOutput (function calling)
   - OpenAI: Add native structured output support when explicitly requested
   - Bedrock: Enhanced function calling with better schema handling
   - Anthropic: Enhanced function calling support
   - Others: Tool-based approach with prompting fallback when explicitly requested

### Phase 4: Testing and Documentation (Week 4-5)
1. **Comprehensive Testing**
   - Unit tests for all new components
   - Integration tests for end-to-end workflows
   - Performance tests comparing old vs new approaches
   - Backward compatibility tests

2. **Documentation and Examples**
   - Update API documentation
   - Create migration guide from old to new approach
   - Add comprehensive examples for all output modes
   - Update getting started guides

### Phase 5: Migration and Cleanup (Week 5-6)
1. **Deprecation Implementation**
   - Add deprecation warnings to old methods
   - Create migration utilities
   - Update internal usage to new patterns

2. **Performance Optimization**
   - Optimize output type resolution
   - Cache compiled schemas and tools
   - Minimize overhead for non-structured-output calls

3. **Final Integration**
   - Update all examples and documentation
   - Ensure full metrics and telemetry integration
   - Performance validation and optimization

## Benefits of New Architecture

### Developer Experience
- **Intuitive API**: Single method for all agent interactions
- **Type Safety**: Full TypeScript-like type inference for Python
- **Flexibility**: Multiple output modes for different use cases
- **Consistency**: Same interface for all interaction patterns

### Technical Benefits
- **Full Integration**: Structured output participates in event loop
- **Metrics Collection**: Complete telemetry for structured output
- **Tool Integration**: Tools can be used during structured output generation
- **Streaming Support**: Real-time structured output building
- **Performance**: Optimized paths for different output modes

### Maintainability
- **Modular Design**: Clear separation of concerns
- **Extensibility**: Easy to add new output modes
- **Backward Compatibility**: Smooth migration path
- **Testing**: Comprehensive test coverage for all scenarios

## Migration Guide for Users

### Simple Migration
```python
# Old approach
user_data = agent.structured_output(UserModel, "Extract user info")

# New approach (uses ToolOutput by default - most reliable)
result = agent("Extract user info", output_type=UserModel)
user_data = result.get_structured_output(UserModel)

# Or access directly (with type checking at runtime)
user_data = result.structured_output
```

### Advanced Migration
```python
# Old: Separate methods for different use cases
text_result = agent("Generate summary")
structured_result = agent.structured_output(Summary, "Generate summary")

# New: Unified interface (uses ToolOutput by default)
text_result = agent("Generate summary")
structured_result = agent("Generate summary", output_type=Summary)
summary = structured_result.get_structured_output(Summary)

# Or set default output type (uses ToolOutput by default)
summary_agent = Agent(model, output_type=Summary)
result = summary_agent("Generate summary")
summary = result.get_structured_output(Summary)

# Explicit mode selection only when needed
result = agent("Extract", output_type=Data, output_mode=NativeOutput())
# Falls back to ToolOutput if model doesn't support native
```

## Success Metrics

### Performance Metrics
- Response time for structured output scenarios
- Memory usage compared to current implementation
- Event loop overhead for non-structured calls

### Developer Experience Metrics
- API adoption rates for new vs old methods
- Developer feedback and satisfaction scores
- Documentation engagement and clarity metrics

### Technical Metrics
- Test coverage for new components
- Backward compatibility test success rates
- Integration test success across all model providers

## Risk Mitigation

### Technical Risks
- **Breaking Changes**: Comprehensive backward compatibility testing
- **Performance Regression**: Benchmarking and optimization
- **Model Provider Issues**: Fallback strategies and error handling

### Adoption Risks
- **Migration Complexity**: Clear documentation and migration tools
- **Learning Curve**: Examples and tutorials for common patterns
- **Feature Gaps**: Ensure feature parity with current implementation

## Conclusion

This implementation plan provides a comprehensive approach to revamping the Strands SDK's structured output system to match PydanticAI's intuitive developer experience while maintaining full backward compatibility. The new architecture integrates structured output into the main agent loop, enabling proper metrics collection, tool integration, and event streaming while providing a more intuitive API for developers.

The phased approach ensures systematic implementation with proper testing and validation at each stage, minimizing risks while delivering significant improvements to the developer experience and technical capabilities of the SDK.