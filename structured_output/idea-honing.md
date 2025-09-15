# Requirements Clarification

This document will capture the detailed requirements through our Q&A process.

## Question 1: API Design Approach

Should the new structured output feature be implemented as:

A) **Parameter-based approach**: Add an `output_type` parameter to the existing `agent()` call
   ```python
   result = agent("Generate user data", output_type=UserModel)
   # Returns AgentResult with structured_output field containing UserModel instance
   ```

B) **Method-based approach**: Keep separate method but integrate with AgentResult
   ```python
   result = agent.structured_output(UserModel, "Generate user data") 
   # Returns AgentResult instead of just UserModel instance
   ```

C) **Hybrid approach**: Support both patterns for different use cases
   ```python
   # Option 1: Integrated
   result = agent("Generate user data", output_type=UserModel)
   
   # Option 2: Explicit method  
   result = agent.structured_output(UserModel, "Generate user data")
   # Both return AgentResult with structured output
   ```

Which approach do you prefer and why?

**Answer**: A) Parameter-based approach: Add an `output_type` parameter to the existing `agent()` call

This approach provides a unified API where users can optionally specify structured output without learning a separate method. It maintains consistency with the existing agent interface while extending its capabilities.

## Question 2: AgentResult Integration

How should the structured output be accessible in the AgentResult object?

A) **New field**: Add a `structured_output` field to AgentResult
   ```python
   @dataclass
   class AgentResult:
       stop_reason: StopReason
       message: Message
       metrics: EventLoopMetrics
       state: Any
       structured_output: Optional[BaseModel] = None  # New field
   
   result = agent("Generate data", output_type=UserModel)
   user_data = result.structured_output  # UserModel instance or None
   ```

B) **Generic field**: Add a generic `output` field that can hold any type
   ```python
   @dataclass  
   class AgentResult:
       # ... existing fields
       output: Optional[Any] = None  # Could be BaseModel or None
   ```

C) **Method accessor**: Keep AgentResult unchanged, add method to access structured output
   ```python
   result = agent("Generate data", output_type=UserModel)
   user_data = result.get_structured_output()  # Returns UserModel or None
   ```

Which approach do you prefer for accessing the structured output from AgentResult?

**Answer**: A) New field: Add a `structured_output` field to AgentResult

This provides clear, explicit access to structured output with proper typing. The field name makes it obvious what it contains and when it's available.

## Question 3: Backward Compatibility

How should we handle the existing `agent.structured_output()` method?

A) **Deprecate gradually**: Keep the existing method but mark it as deprecated
   ```python
   # New way (preferred)
   result = agent("prompt", output_type=UserModel)
   
   # Old way (deprecated but still works)
   user_data = agent.structured_output(UserModel, "prompt")  # Issues deprecation warning
   ```

B) **Remove immediately**: Delete the existing method in this implementation
   ```python
   # Only the new way works
   result = agent("prompt", output_type=UserModel)
   ```

C) **Keep both**: Maintain both APIs indefinitely
   ```python
   # Both ways work permanently
   result = agent("prompt", output_type=UserModel)
   user_data = agent.structured_output(UserModel, "prompt")
   ```

Which approach do you prefer for handling the existing structured_output method?

**Answer**: C) Keep both: Maintain both APIs indefinitely

This ensures no breaking changes for existing users while providing the new unified interface. Both APIs can coexist and serve different use cases.

## Question 4: Provider Implementation Strategy

How should we handle the different structured output capabilities across model providers?

A) **Unified abstraction**: Create a single interface that automatically chooses the best approach per provider
   ```python
   # Same code works for all providers, implementation varies internally:
   # - OpenAI: Uses native API
   # - Bedrock: Uses tool calling  
   # - Ollama: Uses JSON schema
   # - Others: Uses prompt-based fallback
   ```

B) **Explicit configuration**: Let users choose the implementation strategy
   ```python
   agent = Agent(structured_output_strategy="native")  # or "tool", "json", "prompt"
   result = agent("prompt", output_type=UserModel)
   ```

C) **Provider-specific**: Keep current provider-specific implementations as-is
   ```python
   # Each provider continues with its current approach
   # No standardization across providers
   ```

Which approach do you prefer for handling provider differences?

**Answer**: A) Unified abstraction: Create a single interface that automatically chooses the best approach per provider

This provides the best developer experience by hiding implementation complexity while ensuring optimal performance for each provider. Users get consistent behavior regardless of the underlying model.

## Question 5: Error Handling Strategy

How should structured output failures be handled?

A) **Graceful fallback**: Attempt multiple strategies if the primary approach fails
   ```python
   # Try: native API → JSON schema → prompt-based → raise error
   result = agent("prompt", output_type=UserModel)
   # Always returns AgentResult, structured_output=None if all methods fail
   ```

B) **Fail fast**: Raise an exception immediately if structured output fails
   ```python
   result = agent("prompt", output_type=UserModel)
   # Raises StructuredOutputError if parsing fails
   ```

C) **Best effort**: Return partial results when possible
   ```python
   result = agent("prompt", output_type=UserModel)
   # structured_output might be None, but message contains raw response
   # User can manually parse if needed
   ```

Which error handling approach do you prefer?

**Answer**: A) Graceful fallback: Attempt multiple strategies if the primary approach fails

This provides maximum reliability by trying multiple approaches before giving up. Users get the best chance of success while maintaining predictable return types.

## Question 6: Streaming Support

How should structured output work with streaming responses?

A) **No streaming**: Structured output always requires complete response before parsing
   ```python
   # Streaming disabled when output_type is specified
   result = agent("prompt", output_type=UserModel, streaming=True)  # Ignores streaming=True
   ```

B) **Delayed parsing**: Stream the text response, parse structured output at the end
   ```python
   async for chunk in agent.stream("prompt", output_type=UserModel):
       print(chunk.message)  # Stream text as usual
   # chunk.structured_output available only in final chunk
   ```

C) **Partial parsing**: Attempt to parse structured output as it streams in
   ```python
   async for chunk in agent.stream("prompt", output_type=UserModel):
       if chunk.structured_output:  # Available when parsing succeeds
           print("Got structured data:", chunk.structured_output)
   ```

Which approach do you prefer for streaming with structured output?

**Answer**: B) Delayed parsing: Stream the text response, parse structured output at the end

This provides the best balance of user experience (streaming text) and reliability (complete response parsing). Users get immediate feedback while ensuring accurate structured output parsing.

## Question 7: Type Safety and Validation

How should the implementation handle type safety for the structured_output field?

A) **Generic typing**: Use Union type to handle any Pydantic model
   ```python
   @dataclass
   class AgentResult:
       # ... existing fields
       structured_output: Optional[BaseModel] = None
   
   result = agent("prompt", output_type=UserModel)
   user_data = result.structured_output  # Type: Optional[BaseModel], needs casting
   ```

B) **Generic class**: Make AgentResult generic based on output_type
   ```python
   @dataclass
   class AgentResult[T]:
       # ... existing fields  
       structured_output: Optional[T] = None
   
   result = agent("prompt", output_type=UserModel)  # Returns AgentResult[UserModel]
   user_data = result.structured_output  # Type: Optional[UserModel], properly typed
   ```

C) **Runtime typing**: Keep simple typing, rely on runtime validation
   ```python
   @dataclass
   class AgentResult:
       # ... existing fields
       structured_output: Optional[Any] = None  # Simple but less type-safe
   ```

Which typing approach do you prefer for type safety?

**Answer**: A) Generic typing: Use Union type to handle any Pydantic model

This approach maintains simplicity while providing reasonable type safety. Users can cast to their specific model type when needed, and it avoids the complexity of generic classes.

## Question 8: Implementation Priority

Which aspects should be prioritized in the initial implementation?

A) **Core functionality first**: Focus on basic parameter integration and AgentResult changes
   ```python
   # Phase 1: Basic integration
   result = agent("prompt", output_type=UserModel)  # Works with existing provider implementations
   
   # Phase 2: Provider optimization and fallback strategies
   # Phase 3: Streaming integration and advanced features
   ```

B) **Provider standardization first**: Fix all provider implementations before API changes
   ```python
   # Phase 1: Standardize all model provider structured_output methods
   # Phase 2: Add output_type parameter to agent
   # Phase 3: Streaming and advanced features
   ```

C) **Complete solution**: Implement all features together in one comprehensive update
   ```python
   # Single phase: All features including fallbacks, streaming, provider optimization
   ```

Which implementation approach do you prefer for managing complexity and delivery?

**Answer**: Phased development with complete delivery - Develop in phases for manageable complexity, but release all phases together as a complete solution to users.

This approach provides the best of both worlds: structured development process while ensuring users get a fully polished, comprehensive feature set.

## Question 9: Testing Strategy

How should we ensure the new structured output implementation works reliably across all providers?

A) **Mock-based testing**: Test the unified interface with mocked provider responses
   ```python
   # Test agent("prompt", output_type=Model) with mocked model.structured_output()
   # Fast, predictable, but doesn't catch provider-specific issues
   ```

B) **Integration testing**: Test against real model providers with actual API calls
   ```python
   # Test each provider (Bedrock, OpenAI, Anthropic, etc.) with real models
   # Slower, requires credentials, but catches real-world issues
   ```

C) **Hybrid approach**: Mock tests for core logic + integration tests for provider validation
   ```python
   # Unit tests: Mock-based for agent interface and fallback logic
   # Integration tests: Real providers for validation and compatibility
   ```

Which testing approach do you think will give us the most confidence in the implementation?

**Answer**: C) Hybrid approach: Mock tests for core logic + integration tests for provider validation

Mock tests for fast feedback on core functionality, integration tests for real-world validation. Integration tests will be written but not executed during development - you will run them separately for validation.

## Requirements Summary

Based on our Q&A session, here are the finalized requirements:

### Core API Design
- **Unified Interface**: Add `output_type` parameter to existing `agent()` call
- **AgentResult Integration**: Add `structured_output: Optional[BaseModel]` field to AgentResult
- **Backward Compatibility**: Keep existing `agent.structured_output()` method indefinitely

### Implementation Strategy  
- **Provider Abstraction**: Unified interface that automatically chooses best approach per provider
- **Error Handling**: Graceful fallback through multiple strategies (native → JSON → prompt → fail)
- **Streaming Support**: Delayed parsing - stream text, parse structured output in final chunk
- **Type Safety**: Use `Optional[BaseModel]` typing with runtime validation

### Development Approach
- **Phased Development**: Build in phases for manageable complexity
- **Complete Delivery**: Release all phases together as comprehensive solution
- **Testing Strategy**: Mock tests for core logic + integration tests for provider validation (integration tests written but not executed during development)

The requirements clarification is now complete. Ready to proceed to the next step?
