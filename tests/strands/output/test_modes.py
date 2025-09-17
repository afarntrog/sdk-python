"""Tests for output mode implementations."""

import pytest
from pydantic import BaseModel

from strands.output.modes import ToolOutput, NativeOutput, PromptedOutput


class TestOutputModel(BaseModel):
    name: str
    value: int


class MockModel:
    """Mock model for testing."""
    
    def supports_native_structured_output(self):
        return True


class MockModelNoSupport:
    """Mock model without native structured output support."""
    
    def supports_native_structured_output(self):
        return False


def test_tool_output_mode():
    """Test ToolOutput mode functionality."""
    mode = ToolOutput()
    
    # Test tool spec generation
    tool_specs = mode.get_tool_specs(TestOutputModel)
    assert len(tool_specs) == 1
    assert tool_specs[0]["name"] == "TestOutputModel"
    
    # Test model support (should always be True)
    assert mode.is_supported_by_model(MockModel()) is True
    assert mode.is_supported_by_model(MockModelNoSupport()) is True


def test_native_output_mode():
    """Test NativeOutput mode functionality."""
    mode = NativeOutput()
    
    # Test tool spec generation (should be empty)
    tool_specs = mode.get_tool_specs(TestOutputModel)
    assert len(tool_specs) == 0
    
    # Test model support detection
    assert mode.is_supported_by_model(MockModel()) is True
    assert mode.is_supported_by_model(MockModelNoSupport()) is False
    
    # Test model without method
    class ModelWithoutMethod:
        pass
    
    assert mode.is_supported_by_model(ModelWithoutMethod()) is False


def test_prompted_output_mode():
    """Test PromptedOutput mode functionality."""
    mode = PromptedOutput()
    
    # Test tool spec generation (should be empty)
    tool_specs = mode.get_tool_specs(TestOutputModel)
    assert len(tool_specs) == 0
    
    # Test model support (should always be True)
    assert mode.is_supported_by_model(MockModel()) is True
    assert mode.is_supported_by_model(MockModelNoSupport()) is True


def test_prompted_output_custom_template():
    """Test PromptedOutput with custom template."""
    custom_template = "Custom template: {schema}"
    mode = PromptedOutput(template=custom_template)
    
    # Test template retrieval
    template = mode.get_prompt_template(TestOutputModel)
    assert "Custom template:" in template
    assert "TestOutputModel" in template or "name" in template  # Schema should be included


def test_prompted_output_default_template():
    """Test PromptedOutput with default template."""
    mode = PromptedOutput()
    
    # Test template retrieval
    template = mode.get_prompt_template(TestOutputModel)
    assert "JSON" in template
    assert "schema" in template.lower()
