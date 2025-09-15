"""Integration test demonstrating the complete structured output feature."""

from pydantic import BaseModel
from typing import Optional

class UserProfile(BaseModel):
    """Example Pydantic model for structured output."""
    name: str
    age: int
    occupation: str
    active: bool = True

def test_agent_interface():
    """Test that Agent class has the correct interface for structured output."""
    
    # Import here to avoid circular imports during testing
    from ..agent.agent import Agent
    from ..agent.agent_result import AgentResult
    
    print("🧪 Testing Agent interface for structured output...")
    
    # Test 1: Check that Agent methods have output_type parameter
    import inspect
    
    # Check __call__ method (synchronous)
    call_sig = inspect.signature(Agent.__call__)
    assert 'output_type' in call_sig.parameters, "Agent.__call__ missing output_type parameter"
    assert call_sig.parameters['output_type'].annotation == Optional[type[BaseModel]], "Wrong type annotation"
    print("✅ Agent.__call__ has correct output_type parameter")
    
    # Check invoke_async method
    invoke_sig = inspect.signature(Agent.invoke_async)
    assert 'output_type' in invoke_sig.parameters, "Agent.invoke_async missing output_type parameter"
    print("✅ Agent.invoke_async has correct output_type parameter")
    
    # Check stream_async method
    stream_sig = inspect.signature(Agent.stream_async)
    assert 'output_type' in stream_sig.parameters, "Agent.stream_async missing output_type parameter"
    print("✅ Agent.stream_async has correct output_type parameter")
    
    # Test 2: Check AgentResult has structured_output field
    result_fields = AgentResult.__dataclass_fields__
    assert 'structured_output' in result_fields, "AgentResult missing structured_output field"
    print("✅ AgentResult has structured_output field")
    
    # Test 3: Test AgentResult creation with structured_output
    from ..telemetry.metrics import EventLoopMetrics
    
    metrics = EventLoopMetrics()
    message = {'role': 'assistant', 'content': [{'text': 'Test message'}]}
    test_profile = UserProfile(name="Alice", age=25, occupation="Engineer")
    
    result = AgentResult(
        stop_reason='end_turn',
        message=message,
        metrics=metrics,
        state={},
        structured_output=test_profile
    )
    
    assert result.structured_output == test_profile, "Structured output not preserved"
    assert isinstance(result.structured_output, UserProfile), "Wrong structured output type"
    print("✅ AgentResult correctly stores structured output")
    
    # Test 4: Test AgentResult string representation still works
    text_content = str(result)
    assert "Test message" in text_content, "Text content not preserved"
    print("✅ AgentResult text representation works with structured output")
    
    print("\n🎉 All Agent interface tests passed!")
    print("📋 Summary:")
    print("   • Agent.__call__ supports output_type parameter")
    print("   • Agent.invoke_async supports output_type parameter") 
    print("   • Agent.stream_async supports output_type parameter")
    print("   • AgentResult includes structured_output field")
    print("   • Text and structured output coexist properly")

if __name__ == "__main__":
    test_agent_interface()
