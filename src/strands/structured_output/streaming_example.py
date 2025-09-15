"""Example demonstrating streaming integration with structured output.

This example shows how structured output works with streaming:
1. Intermediate streaming events contain real-time text chunks
2. Final AgentResultEvent contains both text and structured output
3. Users get immediate streaming feedback + structured data at the end
"""

from pydantic import BaseModel
from typing import AsyncGenerator

class UserProfile(BaseModel):
    """Example Pydantic model for structured output."""
    name: str
    age: int
    occupation: str

async def streaming_with_structured_output_example():
    """Example of how streaming works with structured output."""
    
    # Simulated streaming events that would come from agent.stream_async()
    streaming_events = [
        # Intermediate streaming events - real-time text chunks
        {"event": {"data": "Let me create a user profile for you.\n\n"}},
        {"event": {"data": "Name: John Smith\n"}},
        {"event": {"data": "Age: 30 years old\n"}},
        {"event": {"data": "Occupation: Software Engineer\n"}},
        
        # Final event - contains both text and structured output
        {
            "result": {
                "stop_reason": "end_turn",
                "message": {
                    "role": "assistant", 
                    "content": [{"text": "Let me create a user profile for you.\n\nName: John Smith\nAge: 30 years old\nOccupation: Software Engineer\n"}]
                },
                "structured_output": UserProfile(name="John Smith", age=30, occupation="Software Engineer")
            }
        }
    ]
    
    print("🔄 Streaming with structured output:")
    print("=" * 50)
    
    for i, event in enumerate(streaming_events):
        if "event" in event:
            # Intermediate streaming event - show real-time text
            text_chunk = event["event"]["data"]
            print(f"📡 Stream chunk {i+1}: {repr(text_chunk)}")
            
        elif "result" in event:
            # Final event - show both text and structured output
            result = event["result"]
            text_content = result["message"]["content"][0]["text"]
            structured_data = result["structured_output"]
            
            print(f"\n✅ Final result:")
            print(f"📄 Full text: {repr(text_content)}")
            print(f"🏗️  Structured output: {structured_data}")
            print(f"   - Name: {structured_data.name}")
            print(f"   - Age: {structured_data.age}")
            print(f"   - Occupation: {structured_data.occupation}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(streaming_with_structured_output_example())
