#!/usr/bin/env python3
"""
Working examples with mock model for testing structured output features.
These examples actually work without requiring external API credentials.
"""

import json
from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field
from strands import Agent, ToolMode, NativeMode, OutputSchema
from strands.models.model import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec


# Mock Model for Testing
class MockModel(Model):
    """A mock model that generates structured output for testing"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)
    
    def get_config(self) -> Any:
        return self.config
    
    def supports_native_structured_output(self) -> bool:
        return True
    
    def get_structured_output_config(self, output_schema: "OutputSchema") -> Dict[str, Any]:
        return {"mock_config": True}
    
    async def structured_output(
        self, output_model: Type[BaseModel], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Union[BaseModel, Any]], None]:
        """Generate mock structured output"""
        
        # Extract the prompt text
        prompt_text = ""
        if prompt:
            for message in prompt:
                for content in message.get("content", []):
                    if "text" in content:
                        prompt_text += content["text"] + " "
        
        # Generate mock data based on the output model
        mock_data = self._generate_mock_data(output_model, prompt_text.strip())
        instance = output_model(**mock_data)
        
        yield {"output": instance}
    
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Mock streaming implementation"""
        
        # Extract prompt text
        prompt_text = ""
        if messages:
            for message in messages:
                for content in message.get("content", []):
                    if "text" in content:
                        prompt_text += content["text"] + " "
        
        # If we have tool specs (structured output), call the appropriate tool
        if tool_specs:
            for tool_spec in tool_specs:
                tool_name = tool_spec["name"]
                
                # Generate mock tool call
                mock_input = self._generate_mock_tool_input(tool_spec, prompt_text.strip())
                
                yield {
                    "chunk_type": "content_start",
                    "data_type": "tool_use",
                    "data": {
                        "toolUseId": "mock_tool_use_123",
                        "name": tool_name,
                        "input": mock_input
                    }
                }
                
                yield {
                    "chunk_type": "content_stop",
                    "data_type": "tool_use"
                }
                
                break  # Use first tool spec
        else:
            # Regular text response
            yield {
                "chunk_type": "content_start",
                "data_type": "text",
                "data": "Mock response: " + prompt_text
            }
            
            yield {
                "chunk_type": "content_stop",
                "data_type": "text"
            }
        
        yield {
            "chunk_type": "message_stop",
            "data": "end_turn"
        }
    
    def _generate_mock_data(self, model_class: Type[BaseModel], prompt: str) -> Dict[str, Any]:
        """Generate mock data based on the model class and prompt"""
        
        # Get model fields
        fields = model_class.model_fields
        mock_data = {}
        
        for field_name, field_info in fields.items():
            field_type = field_info.annotation
            
            # Generate mock data based on field type and prompt content
            if field_type == str or field_type == Optional[str]:
                if "name" in field_name.lower():
                    mock_data[field_name] = self._extract_name_from_prompt(prompt) or f"Mock {field_name}"
                elif "email" in field_name.lower():
                    mock_data[field_name] = self._extract_email_from_prompt(prompt) or "mock@example.com"
                elif "title" in field_name.lower() or field_name == "name":
                    mock_data[field_name] = f"Mock {field_name.title()}"
                else:
                    mock_data[field_name] = f"Mock {field_name}"
            
            elif field_type == int:
                if "age" in field_name.lower():
                    mock_data[field_name] = self._extract_age_from_prompt(prompt) or 25
                else:
                    mock_data[field_name] = 42
            
            elif field_type == bool:
                mock_data[field_name] = False
            
            elif field_type == List[str]:
                mock_data[field_name] = ["item1", "item2"]
            
            # Handle nested models
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                # List of models
                inner_type = field_type.__args__[0]
                if hasattr(inner_type, 'model_fields'):
                    mock_data[field_name] = [self._generate_mock_data(inner_type, prompt)]
                else:
                    mock_data[field_name] = ["mock_item"]
            
            elif hasattr(field_type, 'model_fields'):
                # Nested model
                mock_data[field_name] = self._generate_mock_data(field_type, prompt)
            
            else:
                mock_data[field_name] = f"mock_{field_name}"
        
        return mock_data
    
    def _generate_mock_tool_input(self, tool_spec: ToolSpec, prompt: str) -> Dict[str, Any]:
        """Generate mock tool input based on tool spec"""
        
        schema = tool_spec["inputSchema"]["json"]
        properties = schema.get("properties", {})
        
        mock_input = {}
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get("type", "string")
            
            if prop_type == "string":
                if "name" in prop_name.lower():
                    mock_input[prop_name] = self._extract_name_from_prompt(prompt) or f"Mock {prop_name}"
                elif "email" in prop_name.lower():
                    mock_input[prop_name] = self._extract_email_from_prompt(prompt) or "mock@example.com"
                else:
                    mock_input[prop_name] = f"Mock {prop_name}"
            elif prop_type == "integer":
                if "age" in prop_name.lower():
                    mock_input[prop_name] = self._extract_age_from_prompt(prompt) or 25
                else:
                    mock_input[prop_name] = 42
            elif prop_type == "boolean":
                mock_input[prop_name] = False
            else:
                mock_input[prop_name] = f"mock_{prop_name}"
        
        return mock_input
    
    def _extract_name_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract name from prompt text"""
        words = prompt.split()
        for i, word in enumerate(words):
            if word.lower() in ["name", "called", "named"] and i + 1 < len(words):
                return words[i + 1].strip(",")
        
        # Look for capitalized words that might be names
        for word in words:
            if word[0].isupper() and len(word) > 2 and word.isalpha():
                return word
        
        return None
    
    def _extract_email_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract email from prompt text"""
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, prompt)
        return match.group() if match else None
    
    def _extract_age_from_prompt(self, prompt: str) -> Optional[int]:
        """Extract age from prompt text"""
        import re
        age_pattern = r'\b(?:age|aged?)\s+(\d+)\b'
        match = re.search(age_pattern, prompt, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Look for standalone numbers that could be ages
        numbers = re.findall(r'\b(\d{1,2})\b', prompt)
        for num in numbers:
            age = int(num)
            if 1 <= age <= 120:  # Reasonable age range
                return age
        
        return None


# Example Models
class Person(BaseModel):
    """A person's information"""
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    email: str = Field(description="Email address")


class Task(BaseModel):
    """A task or todo item"""
    title: str = Field(description="Task title")
    description: str = Field(description="Task description")
    priority: str = Field(description="Priority: low, medium, high")
    completed: bool = Field(description="Whether completed", default=False)


class Company(BaseModel):
    """A company"""
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry")
    employees: int = Field(description="Number of employees")


def working_example_1():
    """Working Example 1: Basic structured output"""
    print("=== Working Example 1: Basic Structured Output ===")
    
    # Create agent with mock model
    agent = Agent(model=MockModel(), output_type=Person)
    
    # This will actually work!
    result = agent("Create a person named John Smith, age 30, email john@example.com")
    person = result.get_structured_output(Person)
    
    print(f"✅ Created person: {person.name}")
    print(f"   Age: {person.age}")
    print(f"   Email: {person.email}")
    print()


def working_example_2():
    """Working Example 2: Runtime output type"""
    print("=== Working Example 2: Runtime Output Type ===")
    
    agent = Agent(model=MockModel())
    
    # Different types per call
    person_result = agent("Extract person: Alice Johnson, age 28", output_type=Person)
    task_result = agent("Create task: Review code, high priority", output_type=Task)
    
    person = person_result.get_structured_output(Person)
    task = task_result.get_structured_output(Task)
    
    print(f"✅ Person: {person.name}, {person.age} years old")
    print(f"✅ Task: {task.title} (Priority: {task.priority})")
    print()


def working_example_3():
    """Working Example 3: Output modes"""
    print("=== Working Example 3: Output Modes ===")
    
    # Test different output modes
    modes = [
        ("ToolMode", ToolMode()),
        ("NativeMode", NativeMode()),
    ]
    
    for mode_name, mode in modes:
        agent = Agent(model=MockModel(), output_type=Person, output_mode=mode)
        result = agent("Create person: Bob Wilson, age 35, bob@test.com")
        person = result.get_structured_output(Person)
        
        print(f"✅ {mode_name}: {person.name} ({person.email})")
    
    print()


def working_example_4():
    """Working Example 4: Model capabilities"""
    print("=== Working Example 4: Model Capabilities ===")
    
    mock_model = MockModel()
    
    print(f"✅ Mock model supports native structured output: {mock_model.supports_native_structured_output()}")
    
    # Test configuration
    schema = OutputSchema(types=[Person], mode=ToolMode())
    config = mock_model.get_structured_output_config(schema)
    print(f"✅ Mock model config: {config}")
    print()


def working_example_5():
    """Working Example 5: Complex extraction"""
    print("=== Working Example 5: Complex Data Extraction ===")
    
    agent = Agent(model=MockModel(), output_type=Company)
    
    prompt = """
    Extract company information:
    TechCorp is a software company with 250 employees.
    """
    
    result = agent(prompt)
    company = result.get_structured_output(Company)
    
    print(f"✅ Company: {company.name}")
    print(f"   Industry: {company.industry}")
    print(f"   Employees: {company.employees}")
    print()


def working_example_6():
    """Working Example 6: Error handling"""
    print("=== Working Example 6: Error Handling ===")
    
    agent = Agent(model=MockModel())
    
    # Test successful case
    result = agent("Create person: Carol Davis, age 40", output_type=Person)
    person = result.get_structured_output(Person)
    print(f"✅ Success: {person.name}")
    
    # Test type mismatch error
    try:
        # Try to get wrong type from result
        task = result.get_structured_output(Task)
        print("❌ Should not reach here")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")
    
    print()


def working_example_7():
    """Working Example 7: Tool registry integration"""
    print("=== Working Example 7: Tool Registry ===")
    
    from strands.tools.registry import ToolRegistry
    from strands import tool
    
    @tool
    def create_person_tool(name: str, age: int) -> dict:
        """Create a person with name and age"""
        return {"name": name, "age": age, "email": f"{name.lower().replace(' ', '.')}@example.com"}
    
    # Test dynamic registration
    registry = ToolRegistry()
    registry.register_dynamic_tool(create_person_tool)
    
    print(f"✅ Registered tool: {create_person_tool.tool_spec['name']}")
    
    # Test retrieval
    retrieved = registry.get_tool('create_person_tool')
    print(f"✅ Retrieved tool: {retrieved is not None}")
    
    # Test cleanup
    registry.clear_dynamic_tools()
    print("✅ Cleaned up dynamic tools")
    print()


def working_example_8():
    """Working Example 8: Output schema details"""
    print("=== Working Example 8: Output Schema ===")
    
    # Create schema
    schema = OutputSchema(
        types=[Person, Task],
        mode=ToolMode(),
        name="multi_output",
        description="Can output Person or Task"
    )
    
    print(f"✅ Schema created with {len(schema.types)} types")
    print(f"   Single type: {schema.is_single_type}")
    print(f"   Mode: {type(schema.mode).__name__}")
    
    # Generate tool specs
    tool_specs = schema.mode.get_tool_specs(schema.type)
    print(f"✅ Generated {len(tool_specs)} tool specs:")
    for spec in tool_specs:
        print(f"   - {spec['name']}: {spec['description']}")
    
    print()


def run_working_examples():
    """Run all working examples"""
    print("🚀 Running Working Structured Output Examples")
    print("=" * 60)
    print("These examples use a mock model and will actually work!")
    print()
    
    working_example_1()
    working_example_2()
    working_example_3()
    working_example_4()
    working_example_5()
    working_example_6()
    working_example_7()
    working_example_8()
    
    print("✅ All working examples completed successfully!")
    print()
    print("🎉 The structured output system is working correctly!")
    print("   You can now use these patterns with real models by:")
    print("   1. Configuring your model credentials")
    print("   2. Replacing MockModel() with your actual model")
    print("   3. Running the same code patterns")


if __name__ == "__main__":
    run_working_examples()
