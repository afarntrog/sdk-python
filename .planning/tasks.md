# Structured Output Revamp - Implementation Tasks

This document breaks down the structured output implementation into discrete, actionable tasks. Each task is designed to be completed independently and includes clear acceptance criteria.

## Phase 1: Core Infrastructure

### Task 1.1: Create Output Mode Base Classes
**Prompt:** Create the base output mode system in `src/strands/output/base.py`. Implement the abstract `OutputMode` class with methods for `get_tool_specs()`, `extract_result()`, and `is_supported_by_model()`. Also implement the `OutputSchema` container class that holds output types, mode, name, and description, with a default to `ToolOutput()`.

**Files to create/modify:**
- `src/strands/output/__init__.py`
- `src/strands/output/base.py`

**Acceptance criteria:**
- Abstract `OutputMode` class with required methods
- `OutputSchema` class with proper initialization
- Type hints and docstrings for all public APIs
- Default to `ToolOutput()` when no mode specified

### Task 1.2: Implement Output Mode Implementations
**Prompt:** Create the concrete output mode implementations in `src/strands/output/modes.py`. Implement `ToolOutput` (default), `NativeOutput`, and `PromptedOutput` classes. Each should implement the abstract methods from `OutputMode` and include model support detection. `ToolOutput` should always return `True` for `is_supported_by_model()`.

**Files to create/modify:**
- `src/strands/output/modes.py`

**Acceptance criteria:**
- `ToolOutput` class converts Pydantic models to tool specs
- `NativeOutput` class returns empty tool specs and checks model support
- `PromptedOutput` class with customizable template
- All classes implement `is_supported_by_model()` correctly

### Task 1.3: Create Output Registry System
**Prompt:** Create an output type registry system in `src/strands/output/registry.py`. This should handle resolution of output types, validation, and caching of tool specifications. Include utilities for converting between different output type formats and validation helpers.

**Files to create/modify:**
- `src/strands/output/registry.py`

**Acceptance criteria:**
- Registry for output type resolution
- Caching mechanism for tool specs
- Validation utilities for output schemas
- Helper functions for type conversion

### Task 1.4: Update Type Definitions
**Prompt:** Update the type definitions in `src/strands/types/` to include new output-related types. Add imports to `src/strands/output/__init__.py` to expose the public API. Ensure proper type annotations throughout.

**Files to create/modify:**
- `src/strands/types/output.py` (new)
- `src/strands/output/__init__.py`
- Update existing type files as needed

**Acceptance criteria:**
- New type definitions for output modes and schemas
- Clean public API exports
- Type compatibility with existing codebase
- No breaking changes to existing types

## Phase 2: AgentResult Enhancement

### Task 2.1: Enhance AgentResult Class
**Prompt:** Modify the `AgentResult` class in `src/strands/agent/agent_result.py` to include a `structured_output` field and add the `get_structured_output()` method with type safety. Ensure backward compatibility with existing `__str__()` method and add proper type annotations.

**Files to create/modify:**
- `src/strands/agent/agent_result.py`

**Acceptance criteria:**
- `structured_output: Optional[Any] = None` field added
- `get_structured_output(output_type: Type[T]) -> T` method implemented
- Type safety with runtime validation
- Backward compatibility maintained
- Comprehensive docstrings

### Task 2.2: Update AgentResult Tests
**Prompt:** Update the tests for `AgentResult` to cover the new structured output functionality. Add tests for `get_structured_output()` method including success cases, error cases (no structured output, type mismatch), and edge cases.

**Files to create/modify:**
- `tests/strands/agent/test_agent_result.py` (create if doesn't exist)
- Update existing agent result tests

**Acceptance criteria:**
- Tests for new `structured_output` field
- Tests for `get_structured_output()` method
- Error condition testing
- Type safety validation tests
- Backward compatibility tests

## Phase 3: Agent Interface Enhancement

### Task 3.1: Enhance Agent Constructor
**Prompt:** Modify the `Agent` class constructor in `src/strands/agent/agent.py` to accept `output_type` and `output_mode` parameters. Implement the `_resolve_output_schema()` method that defaults to `ToolOutput()` and includes model support validation with automatic fallback.

**Files to create/modify:**
- `src/strands/agent/agent.py`

**Acceptance criteria:**
- Constructor accepts `output_type` and `output_mode` parameters
- `_resolve_output_schema()` method implemented
- Default to `ToolOutput()` when no mode specified
- Model support validation with fallback logic
- Proper logging for fallback scenarios

### Task 3.2: Update Agent __call__ Method
**Prompt:** Modify the `Agent.__call__()` method to accept `output_type` and `output_mode` parameters and always return `AgentResult`. Implement `_run_with_output_schema()` method that integrates with the event loop while maintaining the existing `_run_standard()` path for non-structured calls.

**Files to create/modify:**
- `src/strands/agent/agent.py`

**Acceptance criteria:**
- `__call__()` method accepts output parameters
- Always returns `AgentResult`
- Proper parameter resolution and validation
- Integration with event loop for structured output
- Backward compatibility for existing calls

### Task 3.3: Implement Output Schema Resolution Logic
**Prompt:** Implement the complete output schema resolution logic in the `Agent` class. This includes handling runtime overrides, default schema application, model compatibility checking, and automatic fallback from unsupported modes to `ToolOutput()`.

**Files to create/modify:**
- `src/strands/agent/agent.py`

**Acceptance criteria:**
- Runtime output type override support
- Agent-level default schema application
- Model compatibility validation
- Automatic fallback to `ToolOutput()`
- Comprehensive error handling and logging

## Phase 4: Event Loop Integration

### Task 4.1: Update Event Loop Interface
**Prompt:** Modify the `event_loop_cycle()` function in `src/strands/event_loop/event_loop.py` to accept an `output_schema` parameter. Update the function signature and add handling for structured output tool registration during the event loop.

**Files to create/modify:**
- `src/strands/event_loop/event_loop.py`

**Acceptance criteria:**
- `output_schema` parameter added to event loop
- Function signature updated with proper typing
- Integration point for structured output processing
- Backward compatibility maintained

### Task 4.2: Implement Structured Output Tool Registration
**Prompt:** Implement the logic in the event loop to dynamically register structured output tools based on the `output_schema`. This should convert output types to tool specifications and add them to the tool registry for the duration of the event loop cycle.

**Files to create/modify:**
- `src/strands/event_loop/event_loop.py`

**Acceptance criteria:**
- Dynamic tool registration for output types
- Tool specs generated from Pydantic models
- Temporary registration during event loop
- Proper cleanup after event loop completion

### Task 4.3: Add Structured Output Response Processing
**Prompt:** Implement structured output response processing in the event loop. This should detect when the model calls a structured output tool, extract the result, validate it against the expected type, and populate the `structured_output` field in the final `AgentResult`.

**Files to create/modify:**
- `src/strands/event_loop/event_loop.py`

**Acceptance criteria:**
- Detection of structured output tool calls
- Result extraction and validation
- Population of `AgentResult.structured_output`
- Error handling for invalid structured output
- Metrics tracking for structured output calls

### Task 4.4: Update Event Loop Streaming
**Prompt:** Update the event loop streaming functionality to handle structured output events. Add new event types for structured output progress and ensure streaming works correctly when structured output tools are called.

**Files to create/modify:**
- `src/strands/event_loop/event_loop.py`
- `src/strands/types/_events.py`

**Acceptance criteria:**
- New event types for structured output
- Streaming support for structured output tools
- Progressive result building for complex outputs
- Integration with existing streaming infrastructure

## Phase 5: Model Provider Updates

### Task 5.1: Enhance Model Base Class
**Prompt:** Update the `Model` abstract base class in `src/strands/models/model.py` to include `supports_native_structured_output()` and `get_structured_output_config()` methods. These will be used by output modes to determine model capabilities.

**Files to create/modify:**
- `src/strands/models/model.py`

**Acceptance criteria:**
- Abstract methods for structured output support
- Clear interface for model capability detection
- Proper documentation for implementation requirements
- Type annotations for all new methods

### Task 5.2: Update OpenAI Model Provider
**Prompt:** Update the OpenAI model provider in `src/strands/models/openai.py` to implement the new structured output methods. Implement `supports_native_structured_output()` to return `True` for compatible models and `get_structured_output_config()` to return OpenAI-specific configuration for native structured output.

**Files to create/modify:**
- `src/strands/models/openai.py`

**Acceptance criteria:**
- `supports_native_structured_output()` implementation
- Model-specific capability detection
- Native structured output configuration
- Fallback to tool-based approach when needed

### Task 5.3: Update Bedrock Model Provider
**Prompt:** Update the Bedrock model provider in `src/strands/models/bedrock.py` to implement the new structured output methods. Implement `supports_native_structured_output()` to return `False` (Bedrock uses function calling) and enhance the existing function calling support for better structured output handling.

**Files to create/modify:**
- `src/strands/models/bedrock.py`

**Acceptance criteria:**
- Structured output methods implemented
- Enhanced function calling support
- Improved schema handling for structured output
- Integration with existing Bedrock functionality

### Task 5.4: Update Anthropic Model Provider
**Prompt:** Update the Anthropic model provider in `src/strands/models/anthropic.py` to implement the new structured output methods. Implement `supports_native_structured_output()` to return `False` and enhance function calling support for structured output.

**Files to create/modify:**
- `src/strands/models/anthropic.py`

**Acceptance criteria:**
- Structured output methods implemented
- Enhanced function calling for structured output
- Proper integration with Anthropic API
- Error handling for structured output scenarios

### Task 5.5: Update Remaining Model Providers
**Prompt:** Update all remaining model providers (Ollama, LiteLLM, LlamaCpp, etc.) in `src/strands/models/` to implement the new structured output methods. Most should default to tool-based approach unless they have specific native support.

**Files to create/modify:**
- `src/strands/models/ollama.py`
- `src/strands/models/litellm.py`
- `src/strands/models/llamacpp.py`
- `src/strands/models/mistral.py`
- `src/strands/models/writer.py`
- `src/strands/models/sagemaker.py`
- `src/strands/models/llamaapi.py`

**Acceptance criteria:**
- All model providers implement structured output methods
- Appropriate capability reporting for each provider
- Tool-based approach as default
- Provider-specific optimizations where applicable

## Phase 6: Tool System Integration

### Task 6.1: Enhance Tool Registry for Dynamic Output Tools
**Prompt:** Update the `ToolRegistry` in `src/strands/tools/registry.py` to support dynamic registration of structured output tools. Add methods for temporary tool registration during event loops and proper cleanup mechanisms.

**Files to create/modify:**
- `src/strands/tools/registry.py`

**Acceptance criteria:**
- Dynamic tool registration methods
- Temporary tool lifecycle management
- Proper cleanup mechanisms
- Integration with existing tool registry

### Task 6.2: Update Tool Executor for Structured Output
**Prompt:** Update the `ToolExecutor` classes in `src/strands/tools/executors/` to handle structured output tools specially. Add result validation and proper error handling for structured output tool calls.

**Files to create/modify:**
- `src/strands/tools/executors/_executor.py`
- `src/strands/tools/executors/concurrent.py`

**Acceptance criteria:**
- Special handling for structured output tools
- Result validation against expected types
- Error handling for validation failures
- Integration with existing tool execution

### Task 6.3: Enhance Structured Output Tool Conversion
**Prompt:** Enhance the `convert_pydantic_to_tool_spec()` function in `src/strands/tools/structured_output.py` to work better with the new output mode system. Add support for multiple output types and improve schema generation.

**Files to create/modify:**
- `src/strands/tools/structured_output.py`

**Acceptance criteria:**
- Improved tool spec generation
- Support for multiple output types
- Better schema handling
- Integration with new output modes

## Phase 7: Backward Compatibility

### Task 7.1: Implement Deprecated Methods
**Prompt:** Update the `Agent` class to maintain the existing `structured_output()` and `structured_output_async()` methods with deprecation warnings. These should internally use the new system but return the structured output directly for backward compatibility.

**Files to create/modify:**
- `src/strands/agent/agent.py`

**Acceptance criteria:**
- Existing methods maintained
- Deprecation warnings added
- Internal use of new system
- Same return types as before
- Comprehensive deprecation documentation

### Task 7.2: Update Import Structure
**Prompt:** Update the package imports in `src/strands/__init__.py` to expose the new output mode classes while maintaining backward compatibility. Ensure existing imports continue to work.

**Files to create/modify:**
- `src/strands/__init__.py`

**Acceptance criteria:**
- New output modes available for import
- Backward compatibility maintained
- Clean public API structure
- No breaking changes to existing imports

## Phase 8: Testing

### Task 8.1: Unit Tests for Output Mode System
**Prompt:** Create comprehensive unit tests for the new output mode system in `tests/strands/output/`. Test all output modes, schema resolution, model support detection, and error handling scenarios.

**Files to create/modify:**
- `tests/strands/output/__init__.py`
- `tests/strands/output/test_base.py`
- `tests/strands/output/test_modes.py`
- `tests/strands/output/test_registry.py`

**Acceptance criteria:**
- Complete test coverage for output modes
- Model support detection tests
- Error condition testing
- Performance tests for schema resolution

### Task 8.2: Agent Integration Tests
**Prompt:** Create integration tests for the enhanced `Agent` class in `tests/strands/agent/test_agent.py`. Test the new constructor parameters, `__call__()` method with output types, and schema resolution logic.

**Files to create/modify:**
- `tests/strands/agent/test_agent.py` (update existing)

**Acceptance criteria:**
- Tests for new constructor parameters
- `__call__()` method with output types
- Schema resolution testing
- Fallback behavior validation
- Backward compatibility tests

### Task 8.3: Event Loop Integration Tests
**Prompt:** Create integration tests for the event loop structured output functionality in `tests/strands/event_loop/`. Test tool registration, response processing, and streaming with structured output.

**Files to create/modify:**
- `tests/strands/event_loop/test_event_loop.py` (update existing)

**Acceptance criteria:**
- Event loop structured output tests
- Tool registration and cleanup tests
- Response processing validation
- Streaming functionality tests

### Task 8.4: Model Provider Tests
**Prompt:** Update the model provider tests to cover the new structured output methods. Test capability detection and configuration methods for all providers.

**Files to create/modify:**
- Update all model provider test files in `tests/strands/models/`

**Acceptance criteria:**
- Tests for all new model methods
- Capability detection validation
- Provider-specific configuration tests
- Error handling tests

### Task 8.5: End-to-End Integration Tests
**Prompt:** Create end-to-end integration tests in `tests_integ/` that test the complete structured output workflow from agent call to final result. Test with different model providers and output modes.

**Files to create/modify:**
- `tests_integ/test_structured_output_e2e.py`

**Acceptance criteria:**
- Complete workflow testing
- Multiple model provider testing
- Different output mode validation
- Performance benchmarking
- Error scenario testing

### Task 8.6: Backward Compatibility Tests
**Prompt:** Create comprehensive backward compatibility tests to ensure existing `structured_output()` methods continue to work correctly and produce the same results as before.

**Files to create/modify:**
- `tests/strands/agent/test_agent_backward_compatibility.py`

**Acceptance criteria:**
- Existing method behavior validation
- Same return types and values
- Deprecation warning testing
- Migration path validation

## Phase 9: Performance and Optimization

### Task 9.1: Performance Optimization
**Prompt:** Optimize the structured output system for performance. Focus on schema caching, tool spec generation, and minimizing overhead for non-structured-output calls.

**Files to create/modify:**
- Various files for optimization

**Acceptance criteria:**
- Schema caching implementation
- Optimized tool spec generation
- Minimal overhead for regular calls
- Performance benchmarks
- Memory usage optimization

### Task 9.2: Memory Management
**Prompt:** Implement proper memory management for the structured output system. Ensure temporary tools are properly cleaned up and schemas are cached efficiently.

**Files to create/modify:**
- Various files for memory management

**Acceptance criteria:**
- Proper cleanup mechanisms
- Efficient schema caching
- Memory leak prevention
- Resource management validation

## Phase 10: Documentation and Examples

### Task 10.1: API Documentation
**Prompt:** Create comprehensive API documentation for the new structured output system. Document all new classes, methods, and usage patterns with clear examples.

**Files to create/modify:**
- Documentation files
- Docstring updates throughout codebase

**Acceptance criteria:**
- Complete API documentation
- Usage examples for all patterns
- Migration guide from old to new
- Best practices documentation

### Task 10.2: Example Applications
**Prompt:** Create example applications that demonstrate the new structured output capabilities. Show different output modes, complex schemas, and real-world usage patterns.

**Files to create/modify:**
- Example applications and scripts

**Acceptance criteria:**
- Multiple example applications
- Different output mode demonstrations
- Real-world usage patterns
- Performance comparisons

### Task 10.3: Migration Documentation
**Prompt:** Create detailed migration documentation to help users transition from the old `structured_output()` methods to the new unified interface.

**Files to create/modify:**
- Migration guide documentation

**Acceptance criteria:**
- Step-by-step migration guide
- Before/after code examples
- Common pattern translations
- Troubleshooting guide

## Validation Tasks

### Task V.1: Performance Validation
**Prompt:** Run comprehensive performance tests comparing the new structured output system with the old implementation. Validate that performance is improved or at least equivalent.

**Acceptance criteria:**
- Performance benchmarks
- Memory usage comparison
- Latency measurements
- Throughput validation

### Task V.2: Integration Validation
**Prompt:** Validate that the new structured output system integrates properly with all existing SDK features including hooks, callbacks, telemetry, and session management.

**Acceptance criteria:**
- Hook integration testing
- Callback system validation
- Telemetry data verification
- Session management compatibility

### Task V.3: Model Provider Validation
**Prompt:** Test the new structured output system with all supported model providers to ensure consistent behavior and proper fallback mechanisms.

**Acceptance criteria:**
- All model providers tested
- Consistent behavior validation
- Fallback mechanism testing
- Error handling verification

## Deployment Tasks

### Task D.1: Pre-commit Hook Updates
**Prompt:** Update pre-commit hooks and CI/CD pipelines to include testing of the new structured output functionality.

**Acceptance criteria:**
- Updated pre-commit configuration
- CI/CD pipeline updates
- Automated testing integration

### Task D.2: Release Preparation
**Prompt:** Prepare the release including version updates, changelog creation, and release documentation.

**Acceptance criteria:**
- Version number updates
- Comprehensive changelog
- Release notes documentation
- Breaking change documentation (if any)

---

## Task Dependencies

```
Phase 1 (Core Infrastructure)
├── Task 1.1 → Task 1.2 → Task 1.3 → Task 1.4

Phase 2 (AgentResult Enhancement)
├── Task 2.1 → Task 2.2
└── Depends on: Task 1.4

Phase 3 (Agent Interface)
├── Task 3.1 → Task 3.2 → Task 3.3
└── Depends on: Task 1.4, Task 2.1

Phase 4 (Event Loop Integration)
├── Task 4.1 → Task 4.2 → Task 4.3 → Task 4.4
└── Depends on: Task 3.3

Phase 5 (Model Provider Updates)
├── Task 5.1 → Task 5.2, Task 5.3, Task 5.4, Task 5.5
└── Depends on: Task 1.4

Phase 6 (Tool System Integration)
├── Task 6.1 → Task 6.2, Task 6.3
└── Depends on: Task 4.3

Phase 7 (Backward Compatibility)
├── Task 7.1, Task 7.2
└── Depends on: Task 3.3

Phase 8 (Testing)
├── All testing tasks can run in parallel
└── Depends on: Completion of corresponding implementation tasks

Phase 9 (Performance)
├── Task 9.1, Task 9.2
└── Depends on: Phase 8 completion

Phase 10 (Documentation)
├── All documentation tasks can run in parallel
└── Depends on: Phase 9 completion
```

## Estimated Timeline

- **Phase 1-3**: 2-3 weeks (Core infrastructure and agent changes)
- **Phase 4-6**: 2-3 weeks (Event loop and model provider integration)
- **Phase 7-8**: 2-3 weeks (Backward compatibility and testing)
- **Phase 9-10**: 1-2 weeks (Optimization and documentation)

**Total estimated time**: 7-11 weeks depending on complexity and testing thoroughness.