import unittest.mock
from typing import cast

import pytest
from pydantic import BaseModel

from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.types.content import Message
from strands.types.streaming import StopReason


class TestUser(BaseModel):
    """Test model for structured output testing."""
    name: str
    age: int
    email: str


class TestProduct(BaseModel):
    """Another test model for structured output testing."""
    name: str
    price: float
    category: str


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


@pytest.fixture
def test_user():
    return TestUser(name="John Doe", age=30, email="john@example.com")


@pytest.fixture
def test_product():
    return TestProduct(name="Laptop", price=999.99, category="Electronics")


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
    assert not result.has_structured_output


def test__init__with_structured_output(mock_metrics, simple_message: Message, test_user):
    """Test that AgentResult can be initialized with structured output."""
    stop_reason: StopReason = "end_turn"
    state = {"key": "value"}

    result = AgentResult(
        stop_reason=stop_reason,
        message=simple_message,
        metrics=mock_metrics,
        state=state,
        structured_output=test_user
    )

    assert result.stop_reason == stop_reason
    assert result.message == simple_message
    assert result.metrics == mock_metrics
    assert result.state == state
    assert result.structured_output == test_user
    assert result.has_structured_output


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


# Structured Output Tests

def test_get_structured_output_success(mock_metrics, simple_message: Message, test_user):
    """Test successful retrieval of structured output with correct type."""
    result = AgentResult(
        stop_reason="end_turn",
        message=simple_message,
        metrics=mock_metrics,
        state={},
        structured_output=test_user
    )

    retrieved_user = result.get_structured_output(TestUser)
    assert retrieved_user == test_user
    assert isinstance(retrieved_user, TestUser)


def test_get_structured_output_no_output(mock_metrics, simple_message: Message):
    """Test get_structured_output raises error when no structured output available."""
    result = AgentResult(
        stop_reason="end_turn",
        message=simple_message,
        metrics=mock_metrics,
        state={}
    )

    with pytest.raises(ValueError, match="No structured output available in this result"):
        result.get_structured_output(TestUser)


def test_get_structured_output_type_mismatch(mock_metrics, simple_message: Message, test_user):
    """Test get_structured_output raises error on type mismatch."""
    result = AgentResult(
        stop_reason="end_turn",
        message=simple_message,
        metrics=mock_metrics,
        state={},
        structured_output=test_user  # TestUser instance
    )

    with pytest.raises(ValueError, match="Structured output type mismatch: expected TestProduct, got TestUser"):
        result.get_structured_output(TestProduct)


def test_has_structured_output_true(mock_metrics, simple_message: Message, test_user):
    """Test has_structured_output property returns True when output is present."""
    result = AgentResult(
        stop_reason="end_turn",
        message=simple_message,
        metrics=mock_metrics,
        state={},
        structured_output=test_user
    )

    assert result.has_structured_output is True


def test_has_structured_output_false(mock_metrics, simple_message: Message):
    """Test has_structured_output property returns False when no output."""
    result = AgentResult(
        stop_reason="end_turn",
        message=simple_message,
        metrics=mock_metrics,
        state={}
    )

    assert result.has_structured_output is False


def test_structured_output_with_different_types(mock_metrics, simple_message: Message, test_product):
    """Test structured output works with different model types."""
    result = AgentResult(
        stop_reason="end_turn",
        message=simple_message,
        metrics=mock_metrics,
        state={},
        structured_output=test_product
    )

    retrieved_product = result.get_structured_output(TestProduct)
    assert retrieved_product == test_product
    assert isinstance(retrieved_product, TestProduct)
    assert retrieved_product.name == "Laptop"
    assert retrieved_product.price == 999.99
    assert retrieved_product.category == "Electronics"


def test_str_unchanged_with_structured_output(mock_metrics, simple_message: Message, test_user):
    """Test that __str__ method behavior is unchanged when structured output is present."""
    result_without = AgentResult(
        stop_reason="end_turn",
        message=simple_message,
        metrics=mock_metrics,
        state={}
    )

    result_with = AgentResult(
        stop_reason="end_turn",
        message=simple_message,
        metrics=mock_metrics,
        state={},
        structured_output=test_user
    )

    # String representation should be the same regardless of structured output
    assert str(result_without) == str(result_with)
    assert str(result_with) == "Hello world!\n"
