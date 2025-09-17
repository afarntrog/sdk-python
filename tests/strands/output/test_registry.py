"""Tests for output registry system."""

import pytest
from pydantic import BaseModel

from strands.output.registry import OutputRegistry, get_global_registry, convert_type_to_schema
from strands.output.base import OutputSchema
from strands.output.modes import ToolOutput, NativeOutput


class TestOutputModel(BaseModel):
    name: str
    value: int


def test_output_registry_creation():
    """Test OutputRegistry creation."""
    registry = OutputRegistry()
    
    assert len(registry._schemas) == 0
    assert len(registry._tool_spec_cache) == 0


def test_register_and_get_schema():
    """Test schema registration and retrieval."""
    registry = OutputRegistry()
    schema = OutputSchema(
        output_type=TestOutputModel,
        mode=ToolOutput(),
        name="test_schema"
    )
    
    registry.register_schema("test", schema)
    retrieved = registry.get_schema("test")
    
    assert retrieved == schema
    assert retrieved.name == "test_schema"


def test_resolve_output_schema_with_name():
    """Test output schema resolution with registered name."""
    registry = OutputRegistry()
    original_schema = OutputSchema(
        output_type=TestOutputModel,
        mode=ToolOutput(),
        name="original"
    )
    
    registry.register_schema("test", original_schema)
    
    # Resolve with name should use registered schema
    resolved = registry.resolve_output_schema(name="test")
    assert resolved.output_type == TestOutputModel
    assert resolved.name == "test"


def test_resolve_output_schema_new():
    """Test output schema resolution for new schema."""
    registry = OutputRegistry()
    
    resolved = registry.resolve_output_schema(
        output_type=TestOutputModel,
        output_mode=NativeOutput(),
        name="new_schema"
    )
    
    assert resolved.output_type == TestOutputModel
    assert isinstance(resolved.mode, NativeOutput)
    assert resolved.name == "new_schema"


def test_resolve_output_schema_list_type():
    """Test output schema resolution with list of types."""
    registry = OutputRegistry()
    
    resolved = registry.resolve_output_schema(
        output_type=[TestOutputModel, str],  # Should use first type
        output_mode=ToolOutput()
    )
    
    assert resolved.output_type == TestOutputModel
    assert isinstance(resolved.mode, ToolOutput)


def test_resolve_output_schema_existing_schema():
    """Test output schema resolution with existing OutputSchema."""
    registry = OutputRegistry()
    existing_schema = OutputSchema(
        output_type=TestOutputModel,
        mode=NativeOutput()
    )
    
    resolved = registry.resolve_output_schema(output_type=existing_schema)
    
    assert resolved == existing_schema


def test_get_tool_specs():
    """Test tool spec generation and caching."""
    registry = OutputRegistry()
    schema = OutputSchema(
        output_type=TestOutputModel,
        mode=ToolOutput(),
        name="test"
    )
    
    # First call should generate and cache
    specs1 = registry.get_tool_specs(schema)
    assert len(specs1) == 1
    
    # Second call should use cache
    specs2 = registry.get_tool_specs(schema)
    assert specs1 == specs2


def test_validate_schema():
    """Test schema validation."""
    registry = OutputRegistry()
    
    # Valid schema
    valid_schema = OutputSchema(
        output_type=TestOutputModel,
        mode=ToolOutput()
    )
    assert registry.validate_schema(valid_schema) is True
    
    # Invalid schema (not OutputSchema instance)
    assert registry.validate_schema("not a schema") is False


def test_global_registry():
    """Test global registry access."""
    registry = get_global_registry()
    assert isinstance(registry, OutputRegistry)
    
    # Should return same instance
    registry2 = get_global_registry()
    assert registry is registry2


def test_convert_type_to_schema():
    """Test type to schema conversion utility."""
    schema = convert_type_to_schema(
        TestOutputModel,
        mode=NativeOutput(),
        name="converted"
    )
    
    assert isinstance(schema, OutputSchema)
    assert schema.output_type == TestOutputModel
    assert isinstance(schema.mode, NativeOutput)
    assert schema.name == "converted"
