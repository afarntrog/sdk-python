import unittest.mock
from typing import cast

import pytest
from pydantic import BaseModel

from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.types.content import Message
from strands.types.streaming import StopReason


class UserModel(BaseModel):
    name: str
    age: int


class DocumentModel(BaseModel):
    title: str


@pytest.fixture
def mock_metrics():
    return unittest.mock.Mock(spec=EventLoopMetrics)


@pytest.fixture
def simple_message():
    return {"role": "assistant", "content": [{"text": "Hello world!"}]}


@pytest.fixture
def complex_message():
    return {
        "role": "assistant",
        "content": [
            {"text": "First paragraph"},
            {"text": "Second paragraph"},
            {"non_text_content": "This should be ignored"},
            {"text": "Third paragraph"},
        ],
    }


@pytest.fixture
def empty_message():
    return {"role": "assistant", "content": []}


def test__init__(mock_metrics, simple_message: Message):
    """Test that AgentResult can be properly initialized with all required fields."""
    stop_reason: StopReason = "end_turn"
    state = {"key": "value"}

    result = AgentResult(stop_reason=stop_reason, message=simple_message, metrics=mock_metrics, state=state)

    assert result.stop_reason == stop_reason
    assert result.message == simple_message
    assert result.metrics == mock_metrics
    assert result.state == state
    assert result.structured_output is None


def test__init__with_structured_output(mock_metrics, simple_message: Message):
    """Test that AgentResult can be initialized with structured output."""
    test_output = UserModel(name="John", age=30)
    
    result = AgentResult(
        stop_reason="end_turn", 
        message=simple_message, 
        metrics=mock_metrics, 
        state={},
        structured_output=test_output
    )

    assert result.structured_output == test_output


def test_get_structured_output_success(mock_metrics, simple_message: Message):
    """Test successful retrieval of structured output with correct type."""
    test_output = UserModel(name="John", age=30)
    
    result = AgentResult(
        stop_reason="end_turn", 
        message=simple_message, 
        metrics=mock_metrics, 
        state={},
        structured_output=test_output
    )

    retrieved = result.get_structured_output(UserModel)
    assert retrieved == test_output
    assert isinstance(retrieved, UserModel)


def test_get_structured_output_no_output(mock_metrics, simple_message: Message):
    """Test error when no structured output is available."""
    result = AgentResult(
        stop_reason="end_turn", 
        message=simple_message, 
        metrics=mock_metrics, 
        state={}
    )

    with pytest.raises(ValueError, match="No structured output available"):
        result.get_structured_output(UserModel)


def test_get_structured_output_type_mismatch(mock_metrics, simple_message: Message):
    """Test error when structured output type doesn't match expected type."""
    test_output = UserModel(name="John", age=30)
    
    result = AgentResult(
        stop_reason="end_turn", 
        message=simple_message, 
        metrics=mock_metrics, 
        state={},
        structured_output=test_output
    )

    with pytest.raises(ValueError, match="Structured output type mismatch"):
        result.get_structured_output(DocumentModel)


def test__str__simple(mock_metrics, simple_message: Message):
    """Test that str() works with a simple message."""
    result = AgentResult(stop_reason="end_turn", message=simple_message, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == "Hello world!\n"


def test__str__complex(mock_metrics, complex_message: Message):
    """Test that str() works with a complex message with multiple text blocks."""
    result = AgentResult(stop_reason="end_turn", message=complex_message, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == "First paragraph\nSecond paragraph\nThird paragraph\n"


def test__str__empty(mock_metrics, empty_message: Message):
    """Test that str() works with an empty message."""
    result = AgentResult(stop_reason="end_turn", message=empty_message, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == ""


def test__str__no_content(mock_metrics):
    """Test that str() works with a message that has no content field."""
    message_without_content = cast(Message, {"role": "assistant"})

    result = AgentResult(stop_reason="end_turn", message=message_without_content, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == ""


def test__str__non_dict_content(mock_metrics):
    """Test that str() handles non-dictionary content items gracefully."""
    message_with_non_dict = cast(
        Message,
        {"role": "assistant", "content": [{"text": "Valid text"}, "Not a dictionary", {"text": "More valid text"}]},
    )

    result = AgentResult(stop_reason="end_turn", message=message_with_non_dict, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == "Valid text\nMore valid text\n"


def test__str__with_structured_output(mock_metrics, simple_message: Message):
    """Test that str() works normally even when structured output is present."""
    test_output = UserModel(name="John", age=30)
    
    result = AgentResult(
        stop_reason="end_turn", 
        message=simple_message, 
        metrics=mock_metrics, 
        state={},
        structured_output=test_output
    )

    message_string = str(result)
    assert message_string == "Hello world!\n"
