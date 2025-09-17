#!/usr/bin/env python3
"""
Basic Weather Report Example

This example demonstrates the fundamental usage of structured output with the Strands SDK.
It shows how to define a Pydantic model and get structured data from the agent.
"""

import asyncio
from typing import Optional

from pydantic import BaseModel, Field

from strands import Agent, ToolOutput


class WeatherReport(BaseModel):
    """A structured weather report."""

    location: str = Field(description="The location (city, state/country)")
    temperature: float = Field(description="Temperature in Fahrenheit")
    conditions: str = Field(description="Brief description of weather conditions")
    humidity: int = Field(description="Humidity percentage (0-100)")
    wind_speed: Optional[float] = Field(default=None, description="Wind speed in mph")


def main():
    """Demonstrate basic structured output usage."""
    print("🌤️ Basic Weather Report Example")
    print("=" * 50)

    # Create agent with default output type
    agent = Agent(
        model_id="gpt-4o",
        output_type=WeatherReport,  # Set default output type
        output_mode=ToolOutput()    # Use tool-based approach (most reliable)
    )

    # Get weather report with structured output
    print("Getting weather report for San Francisco...")
    result = agent("What's the current weather like in San Francisco, California?")

    # Extract structured output
    weather = result.get_structured_output(WeatherReport)

    # Display results
    print("\n📊 Weather Report:")
    print(f"📍 Location: {weather.location}")
    print(f"🌡️  Temperature: {weather.temperature}°F")
    print(f"☁️  Conditions: {weather.conditions}")
    print(f"💧 Humidity: {weather.humidity}%")
    if weather.wind_speed:
        print(f"💨 Wind Speed: {weather.wind_speed} mph")

    # Show how to override output type for a different call
    print("\n" + "=" * 50)
    print("🌍 Getting weather for a different location...")

    result2 = agent(
        "What's the weather in Tokyo, Japan?",
        output_type=WeatherReport  # Can override even if agent has default
    )

    weather2 = result2.get_structured_output(WeatherReport)
    print(f"📍 {weather2.location}: {weather2.temperature}°F, {weather2.conditions}")


async def async_example():
    """Demonstrate async structured output with streaming."""
    print("\n🔄 Async Streaming Example")
    print("=" * 50)

    from strands.output import OutputSchema

    agent = Agent(model_id="gpt-4o")

    # Use async streaming with structured output
    output_schema = OutputSchema([WeatherReport])
    events = agent.stream_async(
        "Give me a detailed weather report for New York City",
        output_schema=output_schema
    )

    print("Streaming weather data...")
    async for event in events:
        # Handle structured output events
        if hasattr(event, 'get') and event.get('structured_output'):
            output = event['structured_output']
            output_type = event['output_type']
            print(f"✨ Received structured output: {output_type}")
            print(f"   Data: {output}")

        # Handle final result
        elif hasattr(event, 'get') and event.get('result'):
            result = event['result']
            if result.structured_output:
                weather = result.get_structured_output(WeatherReport)
                print("\n🎯 Final Result:")
                print(f"   {weather.location}: {weather.temperature}°F")
                break


if __name__ == "__main__":
    # Run sync example
    main()

    # Run async example
    print("\n" + "=" * 70)
    asyncio.run(async_example())

    print("\n✅ Example completed!")
    print("\nKey takeaways:")
    print("• Define Pydantic models for your structured data")
    print("• Pass output_type to agent calls or set as default")
    print("• Use result.get_structured_output(YourModel) to extract data")
    print("• Structured output works with both sync and async interfaces")