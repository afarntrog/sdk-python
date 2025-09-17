#!/usr/bin/env python3
"""
Comprehensive examples for the new structured output system.
Run this file to test all structured output features.
"""

import asyncio
import warnings
from typing import List, Optional
from pydantic import BaseModel, Field
from strands import Agent, ToolOutput, NativeOutput, PromptedOutput, OutputSchema


# Example Models
class Address(BaseModel):
    """A physical address"""
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State or province")
    zip_code: str = Field(description="Postal code")
    country: str = Field(description="Country name")


class Person(BaseModel):
    """A person's basic information"""
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years", ge=0, le=150)
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number", default=None)


class Company(BaseModel):
    """A company or organization"""
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    address: Address = Field(description="Company address")
    employees: int = Field(description="Number of employees", ge=1)


class Task(BaseModel):
    """A task or todo item"""
    title: str = Field(description="Task title")
    description: str = Field(description="Detailed description")
    priority: str = Field(description="Priority level: low, medium, high")
    completed: bool = Field(description="Whether task is completed", default=False)


class Project(BaseModel):
    """A project containing multiple tasks"""
    name: str = Field(description="Project name")
    description: str = Field(description="Project description")
    tasks: List[Task] = Field(description="List of tasks in the project")
    deadline: Optional[str] = Field(description="Project deadline", default=None)


def example_1_basic_usage():
    """Example 1: Basic structured output usage"""
    print("=== Example 1: Basic Usage ===")
    
    # Create agent with default output type
    agent = Agent(output_type=Person)
    
    try:
        result = agent("Create a person profile for Sarah Johnson, age 28, email sarah@example.com, phone 555-0123")
        person = result.get_structured_output(Person)
        print(f"✓ Created person: {person.name}, {person.age} years old")
        print(f"  Email: {person.email}, Phone: {person.phone}")
    except Exception as e:
        print(f"Expected error (no model configured): {e}")
    
    print()


def example_2_runtime_output_type():
    """Example 2: Runtime output type specification"""
    print("=== Example 2: Runtime Output Type ===")
    
    agent = Agent()
    
    try:
        # Different output types per call
        person_result = agent("Extract person: John Doe, 35, john@test.com", output_type=Person)
        task_result = agent("Create task: Review code, high priority, not completed", output_type=Task)
        
        person = person_result.get_structured_output(Person)
        task = task_result.get_structured_output(Task)
        
        print(f"✓ Person: {person.name}")
        print(f"✓ Task: {task.title} (Priority: {task.priority})")
    except Exception as e:
        print(f"Expected error (no model configured): {e}")
    
    print()


def example_3_output_modes():
    """Example 3: Different output modes"""
    print("=== Example 3: Output Modes ===")
    
    # Test different output modes
    modes = [
        ("ToolOutput (default)", ToolOutput()),
        ("NativeOutput", NativeOutput()),
        ("PromptedOutput", PromptedOutput(template="Extract info: {prompt}\nFormat as JSON:")),
    ]
    
    for mode_name, mode in modes:
        try:
            agent = Agent(output_type=Person, output_mode=mode)
            print(f"✓ Created agent with {mode_name}")
        except Exception as e:
            print(f"Error with {mode_name}: {e}")
    
    print()


def example_4_complex_nested_models():
    """Example 4: Complex nested models"""
    print("=== Example 4: Complex Nested Models ===")
    
    agent = Agent(output_type=Company)
    
    try:
        prompt = """
        Create a company profile:
        TechCorp Inc, Software industry, 500 employees
        Located at 123 Tech Street, San Francisco, CA, 94105, USA
        """
        
        result = agent(prompt)
        company = result.get_structured_output(Company)
        
        print(f"✓ Company: {company.name}")
        print(f"  Industry: {company.industry}")
        print(f"  Employees: {company.employees}")
        print(f"  Address: {company.address.street}, {company.address.city}")
    except Exception as e:
        print(f"Expected error (no model configured): {e}")
    
    print()


def example_5_multiple_output_types():
    """Example 5: Multiple output types in sequence"""
    print("=== Example 5: Multiple Output Types ===")
    
    agent = Agent()
    
    try:
        # Create tasks
        task1_result = agent("Task: Write documentation, medium priority", output_type=Task)
        task2_result = agent("Task: Fix bug #123, high priority", output_type=Task)
        
        # Create project with tasks
        project_prompt = """
        Create a project called 'Q1 Release' with description 'First quarter product release'
        Include the previously created tasks and set deadline to March 31st
        """
        project_result = agent(project_prompt, output_type=Project)
        
        project = project_result.get_structured_output(Project)
        print(f"✓ Project: {project.name}")
        print(f"  Tasks: {len(project.tasks)} tasks")
        print(f"  Deadline: {project.deadline}")
    except Exception as e:
        print(f"Expected error (no model configured): {e}")
    
    print()


def example_6_output_schema():
    """Example 6: Using OutputSchema directly"""
    print("=== Example 6: Output Schema ===")
    
    # Create custom output schema
    schema = OutputSchema(
        types=[Person, Task],
        mode=ToolOutput(),
        name="multi_type_output",
        description="Can output either Person or Task"
    )
    
    print(f"✓ Created schema with {len(schema.types)} types")
    print(f"  Single type: {schema.is_single_type}")
    print(f"  Mode: {type(schema.mode).__name__}")
    
    # Test tool spec generation
    tool_specs = schema.mode.get_tool_specs(schema.type)
    print(f"  Generated {len(tool_specs)} tool specs")
    for spec in tool_specs:
        print(f"    - {spec['name']}: {spec['description']}")
    
    print()


def example_7_model_capabilities():
    """Example 7: Model provider capabilities"""
    print("=== Example 7: Model Capabilities ===")
    
    from strands.models import BedrockModel
    
    try:
        # Test different model providers
        bedrock_model = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        
        print(f"✓ Bedrock native support: {bedrock_model.supports_native_structured_output()}")
        
        # Test configuration generation
        schema = OutputSchema(types=[Person], mode=ToolOutput())
        config = bedrock_model.get_structured_output_config(schema)
        print(f"  Bedrock config: {config}")
        
    except Exception as e:
        print(f"Model capability test: {e}")
    
    print()


def example_8_backward_compatibility():
    """Example 8: Backward compatibility"""
    print("=== Example 8: Backward Compatibility ===")
    
    agent = Agent()
    
    # Test deprecated methods with warning capture
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # This should issue a deprecation warning
            person = agent.structured_output(Person, "Create person: Alice, 30, alice@test.com")
            
            # Check for deprecation warning
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            if deprecation_warnings:
                print("✓ Deprecation warning properly issued")
                print(f"  Warning: {deprecation_warnings[0].message}")
            else:
                print("✗ No deprecation warning found")
                
        except Exception as e:
            print(f"Expected error (no model configured): {e}")
            # Still check for warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            if deprecation_warnings:
                print("✓ Deprecation warning properly issued")
    
    print()


def example_9_error_handling():
    """Example 9: Error handling"""
    print("=== Example 9: Error Handling ===")
    
    agent = Agent()
    
    try:
        result = agent("Create user profile", output_type=Person)
        person = result.get_structured_output(Person)
        print(f"✓ Got person: {person.name}")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    except Exception as e:
        print(f"Expected error (no model configured): {e}")
    
    # Test type mismatch
    try:
        result = agent("Create user profile", output_type=Person)
        # Try to get wrong type
        task = result.get_structured_output(Task)  # This should fail
        print(f"✗ Should not reach here")
    except ValueError as e:
        print(f"✓ Caught type mismatch error: {e}")
    except Exception as e:
        print(f"Expected error (no model configured): {e}")
    
    print()


async def example_10_async_usage():
    """Example 10: Async usage"""
    print("=== Example 10: Async Usage ===")
    
    agent = Agent(output_type=Person)
    
    try:
        # Test async structured output
        result = await agent.structured_output_async(Person, "Create person: Bob, 25, bob@test.com")
        print(f"✓ Async structured output: {result.name}")
    except Exception as e:
        print(f"Expected error (no model configured): {e}")
    
    print()


def example_11_tool_registry():
    """Example 11: Tool registry integration"""
    print("=== Example 11: Tool Registry Integration ===")
    
    from strands.tools.registry import ToolRegistry
    from strands import tool
    
    @tool
    def test_structured_tool(name: str, age: int) -> Person:
        """Create a person with given name and age"""
        return Person(name=name, age=age, email=f"{name.lower()}@example.com")
    
    # Test dynamic tool registration
    registry = ToolRegistry()
    registry.register_dynamic_tool(test_structured_tool)
    
    print(f"✓ Registered dynamic tool: {test_structured_tool.tool_spec['name']}")
    
    # Test retrieval
    retrieved = registry.get_tool('test_structured_tool')
    print(f"✓ Retrieved tool: {retrieved is not None}")
    
    # Test cleanup
    registry.clear_dynamic_tools()
    print("✓ Cleaned up dynamic tools")
    
    print()


def run_all_examples():
    """Run all examples"""
    print("🚀 Running Comprehensive Structured Output Examples")
    print("=" * 60)
    
    # Run synchronous examples
    example_1_basic_usage()
    example_2_runtime_output_type()
    example_3_output_modes()
    example_4_complex_nested_models()
    example_5_multiple_output_types()
    example_6_output_schema()
    example_7_model_capabilities()
    example_8_backward_compatibility()
    example_9_error_handling()
    example_11_tool_registry()
    
    # Run async example
    asyncio.run(example_10_async_usage())
    
    print("✅ All examples completed!")
    print("\nNote: Most examples show 'Expected error (no model configured)' because")
    print("no actual model credentials are configured. This is normal for testing.")
    print("\nTo run with real models, configure your model credentials:")
    print("- For Bedrock: Configure AWS credentials")
    print("- For OpenAI: Set OPENAI_API_KEY environment variable")
    print("- For other providers: See documentation for setup")


if __name__ == "__main__":
    run_all_examples()
