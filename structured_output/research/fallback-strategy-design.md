# Fallback Strategy Design Research

## Provider Capability Detection

Based on the current implementations, I need to design a system that can detect and utilize the best structured output approach for each provider.

## Current Provider Capabilities Analysis

### Tier 1: Native Structured Output
- **OpenAI**: `client.beta.chat.completions.parse(response_format=model)`
- **LiteLLM**: `litellm.acompletion(response_format=model)` 
- **Characteristics**: Most reliable, fastest, built-in validation

### Tier 2: JSON Schema Support  
- **Ollama**: `format=model.model_json_schema()`
- **LlamaCpp**: Likely similar JSON schema approach
- **Characteristics**: Good reliability, moderate performance

### Tier 3: Tool Calling Workaround
- **Bedrock**: `convert_pydantic_to_tool_spec()` + forced tool usage
- **Anthropic**: Same as Bedrock
- **Characteristics**: Complex, multiple failure points, but functional

### Tier 4: Prompt-Based Fallback
- **Writer, Mistral, SageMaker**: Likely need prompt-based approach
- **Characteristics**: Least reliable, requires robust parsing

## Fallback Chain Design

### Strategy Selection Logic
```python
def get_structured_output_strategy(provider: Model) -> StructuredOutputStrategy:
    # 1. Check for native support
    if hasattr(provider, 'supports_native_structured_output'):
        return NativeStrategy()
    
    # 2. Check for JSON schema support  
    if hasattr(provider, 'supports_json_schema'):
        return JsonSchemaStrategy()
        
    # 3. Check for tool calling support
    if hasattr(provider, 'stream') and can_use_tools(provider):
        return ToolCallingStrategy()
        
    # 4. Fallback to prompt-based
    return PromptBasedStrategy()
```

### Error Recovery Chain
```python
async def execute_with_fallback(strategies: List[Strategy], model: Model, output_type: Type[T]) -> Optional[T]:
    for strategy in strategies:
        try:
            result = await strategy.execute(model, output_type)
            return result
        except StructuredOutputError as e:
            logger.warning(f"Strategy {strategy} failed: {e}")
            continue
    return None  # All strategies failed
```
