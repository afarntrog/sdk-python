"""Practical examples demonstrating structured output with Strands Agents.

This file contains working examples that demonstrate various use cases
for structured output across different scenarios and model providers.
"""

import asyncio
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

# Example models for different use cases

class UserProfile(BaseModel):
    """Basic user profile model."""
    name: str
    age: int
    occupation: str
    active: bool = True

class Address(BaseModel):
    """Address information."""
    street: str
    city: str
    state: str
    zip_code: str

class Contact(BaseModel):
    """Contact information."""
    email: str
    phone: Optional[str] = None
    preferred_method: str = "email"

class Employee(BaseModel):
    """Complex nested employee model."""
    name: str
    employee_id: int
    department: str
    address: Address
    contact: Contact
    skills: List[str]
    hire_date: str
    salary_range: str

class ProductReview(BaseModel):
    """Product review analysis."""
    product_name: str
    rating: int = Field(ge=1, le=5, description="Rating from 1-5 stars")
    sentiment: str = Field(pattern="^(positive|negative|neutral)$")
    key_points: List[str]
    would_recommend: bool

class WeatherForecast(BaseModel):
    """Weather forecast data."""
    location: str
    temperature: int
    condition: str
    humidity: int
    wind_speed: int
    forecast_date: str

class TaskList(BaseModel):
    """Task management structure."""
    project_name: str
    tasks: List[str]
    priority: str = Field(pattern="^(high|medium|low)$")
    due_date: str
    estimated_hours: int

# Example functions demonstrating different usage patterns

def example_basic_usage():
    """Basic structured output example."""
    print("🔹 Basic Usage Example")
    print("=" * 50)
    
    # Note: This is a demonstration of the API
    # In real usage, you would initialize with your preferred model
    from strands import Agent
    
    agent = Agent()
    
    # Simple structured output
    result = agent(
        "Create a user profile for Sarah Johnson, age 28, software engineer, currently active",
        output_type=UserProfile
    )
    
    print(f"Text response: {str(result).strip()}")
    if result.structured_output:
        user = result.structured_output
        print(f"Structured data:")
        print(f"  Name: {user.name}")
        print(f"  Age: {user.age}")
        print(f"  Occupation: {user.occupation}")
        print(f"  Active: {user.active}")
    
    print()

async def example_async_usage():
    """Asynchronous structured output example."""
    print("🔹 Async Usage Example")
    print("=" * 50)
    
    from strands import Agent
    
    agent = Agent()
    
    # Async structured output
    result = await agent.invoke_async(
        """
        Analyze this product review:
        "This wireless mouse is fantastic! Great battery life, smooth tracking, 
        and the ergonomic design is perfect for long work sessions. The price 
        is reasonable too. I'd definitely buy it again and recommend it to others.
        Rating: 5 stars"
        """,
        output_type=ProductReview
    )
    
    print(f"Text response: {str(result).strip()}")
    if result.structured_output:
        review = result.structured_output
        print(f"Structured analysis:")
        print(f"  Product: {review.product_name}")
        print(f"  Rating: {review.rating}/5 stars")
        print(f"  Sentiment: {review.sentiment}")
        print(f"  Key points: {', '.join(review.key_points)}")
        print(f"  Recommend: {review.would_recommend}")
    
    print()

async def example_streaming_usage():
    """Streaming with structured output example."""
    print("🔹 Streaming Usage Example")
    print("=" * 50)
    
    from strands import Agent
    
    agent = Agent()
    
    print("Streaming weather forecast generation...")
    print("Real-time text: ", end="", flush=True)
    
    async for event in agent.stream_async(
        "Generate a weather forecast for Seattle: 68°F, partly cloudy, 55% humidity, 8 mph winds, for tomorrow",
        output_type=WeatherForecast
    ):
        if "data" in event:
            # Real-time text streaming
            print(event["data"], end="", flush=True)
        elif "result" in event:
            # Final structured output
            print("\n\nStructured forecast:")
            if event["result"].structured_output:
                forecast = event["result"].structured_output
                print(f"  Location: {forecast.location}")
                print(f"  Temperature: {forecast.temperature}°F")
                print(f"  Condition: {forecast.condition}")
                print(f"  Humidity: {forecast.humidity}%")
                print(f"  Wind: {forecast.wind_speed} mph")
    
    print()

def example_complex_nested_model():
    """Complex nested model example."""
    print("🔹 Complex Nested Model Example")
    print("=" * 50)
    
    from strands import Agent
    
    agent = Agent()
    
    employee_data = """
    Create an employee record:
    - Name: Michael Chen
    - Employee ID: 12345
    - Department: Engineering
    - Address: 456 Tech Street, San Francisco, CA 94105
    - Email: michael.chen@company.com
    - Phone: 555-0199
    - Preferred contact: email
    - Skills: Python, Machine Learning, Cloud Architecture, Team Leadership
    - Hire date: January 15, 2020
    - Salary range: $120,000 - $150,000
    """
    
    result = agent(employee_data, output_type=Employee)
    
    print(f"Text response: {str(result).strip()}")
    if result.structured_output:
        emp = result.structured_output
        print(f"Structured employee data:")
        print(f"  Name: {emp.name} (ID: {emp.employee_id})")
        print(f"  Department: {emp.department}")
        print(f"  Address: {emp.address.street}, {emp.address.city}, {emp.address.state}")
        print(f"  Contact: {emp.contact.email} ({emp.contact.preferred_method})")
        print(f"  Skills: {', '.join(emp.skills)}")
        print(f"  Hired: {emp.hire_date}")
        print(f"  Salary: {emp.salary_range}")
    
    print()

def example_data_extraction():
    """Data extraction from unstructured text."""
    print("🔹 Data Extraction Example")
    print("=" * 50)
    
    from strands import Agent
    
    agent = Agent()
    
    # Extract structured task list from meeting notes
    meeting_notes = """
    Project Alpha Meeting Notes - March 20, 2024
    
    We discussed the upcoming website redesign project. Here are the key tasks:
    - Design new homepage layout (high priority, due March 30)
    - Update user authentication system 
    - Implement responsive mobile design
    - Conduct user testing sessions
    - Deploy to staging environment
    
    Estimated total effort: 40 hours
    This is a high priority project that needs to be completed by March 30th.
    """
    
    result = agent(
        f"Extract a structured task list from these meeting notes:\n{meeting_notes}",
        output_type=TaskList
    )
    
    print(f"Text response: {str(result).strip()}")
    if result.structured_output:
        tasks = result.structured_output
        print(f"Extracted task list:")
        print(f"  Project: {tasks.project_name}")
        print(f"  Priority: {tasks.priority}")
        print(f"  Due date: {tasks.due_date}")
        print(f"  Estimated hours: {tasks.estimated_hours}")
        print(f"  Tasks:")
        for i, task in enumerate(tasks.tasks, 1):
            print(f"    {i}. {task}")
    
    print()

def example_error_handling():
    """Error handling and fallback strategies."""
    print("🔹 Error Handling Example")
    print("=" * 50)
    
    from strands import Agent
    
    agent = Agent()
    
    try:
        # Attempt structured output with potentially challenging input
        result = agent(
            "Generate some random data that might not fit the model perfectly",
            output_type=UserProfile
        )
        
        if result.structured_output is not None:
            print("✅ Structured output successful:")
            user = result.structured_output
            print(f"  Name: {user.name}")
            print(f"  Age: {user.age}")
            print(f"  Occupation: {user.occupation}")
        else:
            print("⚠️ Structured output failed, but text is available:")
            print(f"  Text: {str(result).strip()}")
            
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
    
    print()

def example_metrics_monitoring():
    """Monitoring structured output performance."""
    print("🔹 Metrics Monitoring Example")
    print("=" * 50)
    
    from strands import Agent
    
    agent = Agent()
    
    # Perform several structured output operations
    for i in range(3):
        result = agent(
            f"Create user profile #{i+1} with random data",
            output_type=UserProfile
        )
        
        if result.structured_output:
            print(f"✅ Profile {i+1}: {result.structured_output.name}")
        else:
            print(f"❌ Profile {i+1}: Failed to parse")
    
    # Check metrics
    metrics = agent.event_loop_metrics.get_summary()
    so_metrics = metrics.get('structured_output', {})
    
    print(f"\nStructured Output Metrics:")
    print(f"  Attempts: {so_metrics.get('attempts', 0)}")
    print(f"  Successes: {so_metrics.get('successes', 0)}")
    print(f"  Success rate: {so_metrics.get('success_rate', 0):.1%}")
    print(f"  Strategy used: {so_metrics.get('strategy_used', 'unknown')}")
    print(f"  Avg parsing time: {so_metrics.get('average_parsing_time', 0):.3f}s")
    
    print()

def example_provider_comparison():
    """Compare different model providers."""
    print("🔹 Provider Comparison Example")
    print("=" * 50)
    
    # This example shows how different providers can be used
    # In practice, you would configure with your available models
    
    from strands import Agent
    from strands.structured_output import StructuredOutputManager
    
    # Default agent (uses configured model)
    agent = Agent()
    
    # Check what strategies are available
    manager = StructuredOutputManager()
    capabilities = manager.detect_provider_capabilities(agent.model)
    
    print(f"Model: {agent.model.__class__.__name__}")
    print(f"Available strategies: {capabilities}")
    
    # Test structured output
    result = agent(
        "Create a weather forecast for New York: 75°F, sunny, 40% humidity, 5 mph winds",
        output_type=WeatherForecast
    )
    
    if result.structured_output:
        print(f"✅ Structured output successful with {agent.model.__class__.__name__}")
        forecast = result.structured_output
        print(f"  Location: {forecast.location}")
        print(f"  Temperature: {forecast.temperature}°F")
    else:
        print(f"❌ Structured output failed with {agent.model.__class__.__name__}")
    
    print()

# Main execution
async def run_all_examples():
    """Run all examples in sequence."""
    print("🚀 Strands Agents Structured Output Examples")
    print("=" * 60)
    print()
    
    # Basic examples
    example_basic_usage()
    await example_async_usage()
    await example_streaming_usage()
    
    # Advanced examples
    example_complex_nested_model()
    example_data_extraction()
    
    # Operational examples
    example_error_handling()
    example_metrics_monitoring()
    example_provider_comparison()
    
    print("🎉 All examples completed!")
    print("\nNext steps:")
    print("• Try modifying the examples with your own models")
    print("• Experiment with different model providers")
    print("• Check the documentation for advanced features")

if __name__ == "__main__":
    # Note: These examples demonstrate the API structure
    # To run them, you'll need to configure your preferred model provider
    print("📚 Structured Output Examples")
    print("=" * 40)
    print()
    print("This file contains example code demonstrating structured output usage.")
    print("To run the examples, configure your model provider and uncomment the line below:")
    print()
    print("# asyncio.run(run_all_examples())")
    print()
    print("Available examples:")
    print("• Basic synchronous usage")
    print("• Asynchronous processing")
    print("• Streaming with structured output")
    print("• Complex nested models")
    print("• Data extraction from text")
    print("• Error handling strategies")
    print("• Performance monitoring")
    print("• Provider comparison")
