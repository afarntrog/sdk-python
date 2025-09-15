# Implementation Prompt Plan

## Checklist
- [ ] Prompt 1: Extend AgentResult with structured_output field
- [ ] Prompt 2: Add output_type parameter to Agent interface
- [ ] Prompt 3: Create StructuredOutputManager and strategy system
- [ ] Prompt 4: Implement provider capability detection
- [ ] Prompt 5: Implement Native and JSON Schema strategies
- [ ] Prompt 6: Implement Tool Calling strategy improvements
- [ ] Prompt 7: Implement Prompt-Based fallback strategy
- [ ] Prompt 8: Integrate structured output into event loop
- [ ] Prompt 9: Add streaming integration with delayed parsing
- [ ] Prompt 10: Extend EventLoopMetrics for structured output tracking
- [ ] Prompt 11: Create comprehensive unit tests
- [ ] Prompt 12: Create integration tests for all providers

## Prompts

### Prompt 1: Extend AgentResult with structured_output field
Modify the AgentResult class to include a new structured_output field while maintaining backward compatibility. Add the field as Optional[BaseModel] with default None value. Ensure the existing __str__ method and all other functionality remains unchanged. Update any type hints and imports as needed to support BaseModel from pydantic.

Focus on minimal changes that don't break existing code. The structured_output field should only be populated when structured output is requested, otherwise it remains None.

### Prompt 2: Add output_type parameter to Agent interface
Add the output_type parameter to the Agent.__call__ method signature and ensure it flows through the entire call chain (invoke_async, stream_async, event_loop_cycle). Use the existing **kwargs pattern to maintain backward compatibility.

The parameter should be Optional[Type[BaseModel]] with default None. Ensure the parameter is properly passed through all method calls without breaking existing functionality. No logic changes yet - just parameter plumbing.

### Prompt 3: Create StructuredOutputManager and strategy system
Create the StructuredOutputManager class and abstract StructuredOutputStrategy base class. Implement the strategy pattern with four concrete strategies: NativeStrategy, JsonSchemaStrategy, ToolCallingStrategy, and PromptBasedStrategy.

Focus on the class structure and interfaces. Each strategy should have an execute method that takes (model, output_type, messages, system_prompt) and returns Optional[T]. Include proper error handling with StructuredOutputError exceptions.

### Prompt 4: Implement provider capability detection
Implement the provider capability detection logic in StructuredOutputManager. Create methods to detect which structured output approaches each model provider supports (native, JSON schema, tool calling, prompt-based).

Use introspection and provider-specific checks to determine capabilities. Create a mapping system that returns capabilities in priority order (most reliable first). This will drive the fallback strategy selection.

### Prompt 5: Implement Native and JSON Schema strategies
Implement the NativeStrategy for providers like OpenAI and LiteLLM that have native structured output support. Implement the JsonSchemaStrategy for providers like Ollama that support JSON schema formatting.

For NativeStrategy, delegate to the existing model.structured_output() method. For JsonSchemaStrategy, implement the JSON schema approach with proper error handling and validation. Both should integrate cleanly with the strategy interface.

### Prompt 6: Implement Tool Calling strategy improvements
Implement the ToolCallingStrategy that improves upon the current Bedrock/Anthropic tool calling approach. Use the existing convert_pydantic_to_tool_spec functionality but add better error handling and validation.

Focus on making the tool calling approach more robust while maintaining compatibility with current implementations. Add proper error messages and recovery logic for common failure scenarios.

### Prompt 7: Implement Prompt-Based fallback strategy
Implement the PromptBasedStrategy as a universal fallback that works with any text-generation model. Create prompt templates that inject JSON schema and examples. Implement robust JSON extraction with multiple parsing strategies.

This strategy should handle edge cases like malformed JSON, extra text, and validation errors. Include retry logic with different prompt variations. This is the most complex strategy as it needs to work reliably across all providers.

### Prompt 8: Integrate structured output into event loop
Modify the event_loop_cycle function to detect when structured output is requested and execute the StructuredOutputManager after normal event loop processing completes. Ensure all existing agent loop functionality (metrics, traces, tool usage) is preserved.

The structured output processing should happen in the final event before yielding the result. Add the structured_output field to the final event data. Ensure graceful error handling that doesn't break the agent loop if structured output fails.

### Prompt 9: Add streaming integration with delayed parsing
Implement the delayed parsing approach for streaming. Modify the streaming system to collect the complete response text and parse structured output in the final streaming event. Ensure all intermediate streaming events continue to work identically.

Users should see real-time text streaming as normal, with structured output available only in the final event. The streaming experience should be identical whether or not structured output is requested.

### Prompt 10: Extend EventLoopMetrics for structured output tracking
Add structured output metrics to the EventLoopMetrics class. Track structured output attempts, successes, strategy used, and parsing time. Integrate these metrics into the existing metrics collection and reporting system.

Ensure the new metrics appear in the metrics summary output. Add proper OpenTelemetry instrumentation for the new metrics. The metrics should help users understand structured output performance and reliability.

### Prompt 11: Create comprehensive unit tests
Create a comprehensive unit test suite covering all aspects of the structured output implementation. Test the Agent interface changes, StructuredOutputManager, all strategy implementations, event loop integration, and streaming functionality.

Use mocking to test the core logic without requiring real model providers. Test error scenarios, fallback behavior, and edge cases. Ensure backward compatibility by testing that existing functionality works identically with and without structured output.

### Prompt 12: Create integration tests for all providers
Create integration tests that validate structured output functionality against real model providers. Test each provider (Bedrock, OpenAI, Anthropic, Ollama, etc.) with actual API calls to ensure the implementation works in practice.

These tests should be marked with @pytest.mark.integration and designed to be run separately. Test both successful scenarios and error recovery. Include tests for streaming with structured output across different providers. Note: These tests will be written but not executed during development - they will be run separately for validation.
