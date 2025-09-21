"""
Test structured output tool registration architecture.

This module tests the new architecture where structured output tools are registered
as real tools in Agent._execute_event_loop_cycle() with proper lifecycle management.
"""

import pytest
import unittest.mock
from pydantic import BaseModel
from typing import List

import strands
from strands import Agent
from strands.output.modes import ToolMode
from strands.types.output import OutputSchema
from strands.tools.structured_output_tool import StructuredOutputTool
from tests.fixtures.mocked_model_provider import MockedModelProvider


class UserModel(BaseModel):
    """Test model for structured output testing."""
    name: str
    age: int
    email: str


class TaskModel(BaseModel):
    """Another test model for structured output testing."""
    title: str
    priority: int
    completed: bool = False


@pytest.fixture
def test_user():
    return UserModel(name="Alice Smith", age=28, email="alice@example.com")


@pytest.fixture
def test_task():
    return TaskModel(title="Complete project", priority=1)


@pytest.fixture
def mock_successful_structured_output_response():
    """Mock LLM response that calls structured output tool successfully."""
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_1",
                        "name": "UserModel",
                        "input": {
                            "name": "Alice Smith",
                            "age": 28,
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        }
    ]


@pytest.fixture
def mock_failed_structured_output_response():
    """Mock LLM response that calls structured output tool with invalid data."""
    return [
        {
            "role": "assistant", 
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_1",
                        "name": "UserModel",
                        "input": {
                            "name": "Alice Smith",
                            "age": "invalid_age",  # Should be int
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        }
    ]


@pytest.fixture
def mock_retry_structured_output_response():
    """Mock LLM response sequence: first invalid, then valid after retry."""
    return [
        # First response with invalid data
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_1", 
                        "name": "UserModel",
                        "input": {
                            "name": "Alice Smith",
                            "age": "invalid_age",  # Should be int
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        },
        # Second response with valid data after seeing error
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_2",
                        "name": "UserModel", 
                        "input": {
                            "name": "Alice Smith",
                            "age": 28,  # Now correct
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        }
    ]


def test_structured_output_tool_registration_lifecycle():
    """Test that structured output tools are registered and deregistered properly."""
    model = MockedModelProvider([
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_1",
                        "name": "UserModel",
                        "input": {
                            "name": "Alice Smith", 
                            "age": 28,
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        }
    ])
    
    agent = Agent(model=model)
    output_schema = OutputSchema(type=UserModel, mode=ToolMode())
    
    # Verify tool is not registered initially
    assert agent.tool_registry.get_tool("UserModel") is None
    
    # Make a call with structured output
    result = agent("Generate user data", output_schema=output_schema)
    
    # Verify tool is deregistered after the call
    assert agent.tool_registry.get_tool("UserModel") is None
    
    # Verify we got a successful result
    assert result.stop_reason is not None


def test_multiple_sequential_structured_output_calls():
    """Test multiple sequential structured output calls work without conflicts."""
    model = MockedModelProvider([
        # First call response
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_1",
                        "name": "UserModel",
                        "input": {
                            "name": "Alice Smith",
                            "age": 28, 
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        },
        # Second call response
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_2",
                        "name": "TaskModel",
                        "input": {
                            "title": "Complete project",
                            "priority": 1,
                            "completed": False
                        }
                    }
                }
            ]
        }
    ])
    
    agent = Agent(model=model)
    
    # First call with UserModel schema
    output_schema_1 = OutputSchema(type=UserModel, mode=ToolMode())
    result_1 = agent("Generate user", output_schema=output_schema_1)
    
    # Verify first call worked and tool is cleaned up
    assert result_1.stop_reason is not None
    assert agent.tool_registry.get_tool("UserModel") is None
    
    # Second call with different schema
    output_schema_2 = OutputSchema(type=TaskModel, mode=ToolMode())
    result_2 = agent("Generate task", output_schema=output_schema_2)
    
    # Verify second call worked and tool is cleaned up
    assert result_2.stop_reason is not None
    assert agent.tool_registry.get_tool("TaskModel") is None
    assert agent.tool_registry.get_tool("UserModel") is None


def test_structured_output_validation_error_retry():
    """Test that validation errors are sent back to LLM for retry."""
    model = MockedModelProvider([
        # First response with invalid data
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_1",
                        "name": "UserModel", 
                        "input": {
                            "name": "Alice Smith",
                            "age": "invalid_age",  # Should be int
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        },
        # Second response with valid data after seeing error
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_2",
                        "name": "UserModel",
                        "input": {
                            "name": "Alice Smith", 
                            "age": 28,  # Now correct
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        }
    ])
    
    agent = Agent(model=model)
    output_schema = OutputSchema(type=UserModel, mode=ToolMode())
    
    result = agent("Generate user", output_schema=output_schema)
    
    # Verify we eventually got a successful result after retry
    assert result.stop_reason is not None
    
    # Verify the agent received error feedback and retried
    # The messages should show: user prompt -> assistant invalid call -> user error -> assistant valid call
    assert len(agent.messages) >= 4
    
    # Check that there was an error message in the conversation
    error_message_found = False
    for message in agent.messages:
        if message.get("role") == "user":
            for content in message.get("content", []):
                if isinstance(content, dict) and content.get("toolResult", {}).get("status") == "error":
                    error_content = content["toolResult"]["content"][0]["text"]
                    assert "Validation failed for UserModel" in error_content
                    assert "Field 'age'" in error_content
                    error_message_found = True
                    break
    
    assert error_message_found, "Expected to find error message in conversation history"


def test_structured_output_tool_cleanup_on_exception():
    """Test that tools are cleaned up even when an exception occurs."""
    model = MockedModelProvider([])
    
    # Mock the model to raise an exception
    original_stream = model.stream
    
    def mock_stream_with_exception(*args, **kwargs):
        # Register the tool first (simulating normal flow)
        async def failing_stream():
            # This will cause the event loop to fail after tool registration
            raise ValueError("Simulated model failure")
            yield  # This line won't be reached
        return failing_stream()
    
    model.stream = mock_stream_with_exception
    
    agent = Agent(model=model)
    output_schema = OutputSchema(type=UserModel, mode=ToolMode())
    
    # Verify tool starts unregistered
    assert agent.tool_registry.get_tool("UserModel") is None
    
    # The call should raise an exception, but tools should still be cleaned up
    with pytest.raises(ValueError, match="Simulated model failure"):
        agent("Generate user", output_schema=output_schema)
    
    # Verify tool is still cleaned up despite the exception
    assert agent.tool_registry.get_tool("UserModel") is None


def test_structured_output_no_duplicate_registration():
    """Test that duplicate tool registration is prevented."""
    # Create a tool registry with a mock tool already registered
    model = MockedModelProvider([
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_1",
                        "name": "UserModel",
                        "input": {
                            "name": "Alice Smith",
                            "age": 28,
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        }
    ])
    
    agent = Agent(model=model)
    
    # Manually register a tool with the same name to simulate duplicate scenario
    existing_tool = StructuredOutputTool(UserModel)
    agent.tool_registry.register_dynamic_tool(existing_tool)
    
    # Verify the tool is registered
    assert agent.tool_registry.get_tool("UserModel") is not None
    
    output_schema = OutputSchema(type=UserModel, mode=ToolMode())
    
    # This should not raise a "tool already defined" error
    # The architecture should check for existing tools before registering
    result = agent("Generate user", output_schema=output_schema)
    
    # Should complete successfully 
    assert result.stop_reason is not None
    
    # The original tool should remain (not re-registered)
    remaining_tool = agent.tool_registry.get_tool("UserModel")
    assert remaining_tool is existing_tool


def test_structured_output_tool_specs_still_provided_to_llm():
    """Test that tool specs are still provided to LLM even with new architecture."""
    model_mock = unittest.mock.Mock()
    
    async def mock_stream(*args, **kwargs):
        # Capture the tool_specs parameter that's passed to the model
        tool_specs = args[1] if len(args) > 1 else kwargs.get("tool_specs", [])
        
        # Verify structured output tool spec is included
        structured_tool_found = False
        for spec in tool_specs:
            if spec.get("name") == "UserModel":
                structured_tool_found = True
                # Verify the spec has the expected structure
                assert "description" in spec
                assert "inputSchema" in spec
                assert spec["description"] == "Test model for structured output testing."
                break
        
        assert structured_tool_found, f"UserModel tool spec not found in: {[spec.get('name') for spec in tool_specs]}"
        
        # Return a valid response
        yield {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "toolUseId": "structured_output_1",
                        "name": "UserModel"
                    }
                }
            }
        }
        yield {
            "contentBlockDelta": {
                "delta": {
                    "toolUse": {
                        "input": '{"name": "Alice Smith", "age": 28, "email": "alice@example.com"}'
                    }
                }
            }
        }
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        
    model_mock.stream.side_effect = mock_stream
    
    agent = Agent(model=model_mock)
    output_schema = OutputSchema(type=UserModel, mode=ToolMode())
    
    result = agent("Generate user", output_schema=output_schema)
    
    # Verify the model was called (our mock would assert if tool specs weren't correct)
    model_mock.stream.assert_called()
    
    # Verify successful completion
    assert result.stop_reason is not None


@pytest.mark.asyncio
async def test_structured_output_async_tool_registration():
    """Test that async agent calls also properly manage tool registration."""
    model = MockedModelProvider([
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_1", 
                        "name": "UserModel",
                        "input": {
                            "name": "Alice Smith",
                            "age": 28,
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        }
    ])
    
    agent = Agent(model=model)
    output_schema = OutputSchema(type=UserModel, mode=ToolMode())
    
    # Verify tool is not registered initially
    assert agent.tool_registry.get_tool("UserModel") is None
    
    # Make an async call with structured output  
    result = await agent.invoke_async("Generate user", output_schema=output_schema)
    
    # Verify tool is deregistered after the call
    assert agent.tool_registry.get_tool("UserModel") is None
    
    # Verify we got a successful result
    assert result.stop_reason is not None


def test_structured_output_with_existing_regular_tools():
    """Test that structured output tools work alongside regular tools."""
    
    @strands.tool
    def regular_tool(text: str) -> str:
        """A regular tool for testing."""
        return f"Processed: {text}"
    
    model = MockedModelProvider([
        # First call regular tool
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "regular_1",
                        "name": "regular_tool",
                        "input": {"text": "test"}
                    }
                }
            ]
        },
        # Then call structured output
        {
            "role": "assistant", 
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "structured_output_1",
                        "name": "UserModel",
                        "input": {
                            "name": "Alice Smith",
                            "age": 28,
                            "email": "alice@example.com"
                        }
                    }
                }
            ]
        }
    ])
    
    agent = Agent(model=model, tools=[regular_tool])
    output_schema = OutputSchema(type=UserModel, mode=ToolMode())
    
    # Verify regular tool is registered, structured output tool is not
    assert agent.tool_registry.get_tool("regular_tool") is not None
    assert agent.tool_registry.get_tool("UserModel") is None
    
    result = agent("Process data and generate user", output_schema=output_schema)
    
    # Verify both tools worked and only structured output tool was cleaned up
    assert agent.tool_registry.get_tool("regular_tool") is not None  # Regular tool remains
    assert agent.tool_registry.get_tool("UserModel") is None  # Structured output tool cleaned up
    assert result.stop_reason is not None
