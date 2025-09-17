"""Tests for output base classes."""

import pytest
from pydantic import BaseModel

from strands.output.base import OutputMode, OutputSchema
from strands.output.modes import ToolOutput


class TestOutputModel(BaseModel):
    name: str
    value: int


class MockOutputMode(OutputMode):
    """Mock output mode for testing."""
    
    def get_tool_specs(self, output_type):
        return [{"name": "test_tool", "type": output_type.__name__}]
    
    def extract_result(self, response, output_type):
        return output_type(name="test", value=42)
    
    def is_supported_by_model(self, model):
        return True


def test_output_schema_creation():
    """Test OutputSchema creation with defaults."""
    schema = OutputSchema()
    
    assert schema.output_type is None
    assert isinstance(schema.mode, ToolOutput)
    assert schema.name is None
    assert schema.description is None


def test_output_schema_with_parameters():
    """Test OutputSchema creation with parameters."""
    mode = MockOutputMode()
    schema = OutputSchema(
        output_type=TestOutputModel,
        mode=mode,
        name="test_schema",
        description="Test schema description"
    )
    
    assert schema.output_type == TestOutputModel
    assert schema.mode == mode
    assert schema.name == "test_schema"
    assert schema.description == "Test schema description"


def test_output_schema_repr():
    """Test OutputSchema string representation."""
    schema = OutputSchema(
        output_type=TestOutputModel,
        mode=MockOutputMode()
    )
    
    repr_str = repr(schema)
    assert "OutputSchema" in repr_str
    assert "TestOutputModel" in repr_str
    assert "MockOutputMode" in repr_str
