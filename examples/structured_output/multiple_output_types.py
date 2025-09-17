#!/usr/bin/env python3
"""
Multiple Output Types Example

This example demonstrates how to use multiple output types in a single schema,
allowing the agent to choose the most appropriate response type based on the input.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from strands import Agent
from strands.output import OutputSchema


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Task(BaseModel):
    """A single task item."""
    title: str = Field(description="Brief title of the task")
    description: str = Field(description="Detailed description")
    priority: TaskPriority = Field(description="Priority level")
    estimated_hours: Optional[float] = Field(default=None, description="Estimated hours to complete")


class TaskList(BaseModel):
    """A list of tasks for a project."""
    project_name: str = Field(description="Name of the project")
    tasks: List[Task] = Field(description="List of tasks")
    total_estimated_hours: Optional[float] = Field(default=None, description="Total estimated hours")


class QuestionResponse(BaseModel):
    """A response to a question about project management."""
    answer: str = Field(description="Direct answer to the question")
    confidence: float = Field(description="Confidence level (0.0 to 1.0)")
    related_topics: List[str] = Field(description="Related topics or suggestions")


class ErrorResponse(BaseModel):
    """An error response when the request cannot be fulfilled."""
    error_type: str = Field(description="Type of error")
    message: str = Field(description="Human-readable error message")
    suggestions: List[str] = Field(description="Suggestions for resolving the issue")


def main():
    """Demonstrate multiple output types with project management scenarios."""
    print("📋 Multiple Output Types Example")
    print("=" * 60)

    # Create agent with multiple possible output types
    agent = Agent(model_id="gpt-4o")

    # Define schema with multiple output types
    # The agent will choose the most appropriate one based on the input
    output_schema = OutputSchema(
        types=[TaskList, QuestionResponse, ErrorResponse],
        name="Project Management Assistant",
        description="Helps with project planning, task management, and answering questions"
    )

    # Test Case 1: Request for task breakdown (should return TaskList)
    print("🎯 Test Case 1: Requesting a task breakdown")
    print("Prompt: 'Create a task list for building a simple web application'")

    result1 = agent(
        "Create a task list for building a simple web application with user authentication",
        output_schema=output_schema
    )

    print(f"Response type: {type(result1.structured_output).__name__}")

    if isinstance(result1.structured_output, TaskList):
        task_list = result1.get_structured_output(TaskList)
        print(f"\n📊 Project: {task_list.project_name}")
        print(f"Total estimated hours: {task_list.total_estimated_hours or 'Not specified'}")
        print("\n📝 Tasks:")
        for i, task in enumerate(task_list.tasks, 1):
            hours = f" ({task.estimated_hours}h)" if task.estimated_hours else ""
            print(f"  {i}. [{task.priority.upper()}] {task.title}{hours}")
            print(f"     {task.description}")

    # Test Case 2: Ask a question (should return QuestionResponse)
    print("\n" + "=" * 60)
    print("🎯 Test Case 2: Asking a project management question")
    print("Prompt: 'What are the best practices for code review in a team?'")

    result2 = agent(
        "What are the best practices for code review in a team?",
        output_schema=output_schema
    )

    print(f"Response type: {type(result2.structured_output).__name__}")

    if isinstance(result2.structured_output, QuestionResponse):
        response = result2.get_structured_output(QuestionResponse)
        print(f"\n💡 Answer (Confidence: {response.confidence:.1%}):")
        print(f"   {response.answer}")
        print("\n🔗 Related topics:")
        for topic in response.related_topics:
            print(f"   • {topic}")

    # Test Case 3: Invalid/unclear request (should return ErrorResponse)
    print("\n" + "=" * 60)
    print("🎯 Test Case 3: Ambiguous or invalid request")
    print("Prompt: 'Purple elephant dancing Tuesday'")

    result3 = agent(
        "Purple elephant dancing Tuesday",
        output_schema=output_schema
    )

    print(f"Response type: {type(result3.structured_output).__name__}")

    if isinstance(result3.structured_output, ErrorResponse):
        error = result3.get_structured_output(ErrorResponse)
        print(f"\n❌ Error ({error.error_type}):")
        print(f"   {error.message}")
        print("\n💡 Suggestions:")
        for suggestion in error.suggestions:
            print(f"   • {suggestion}")


def demonstrate_conditional_logic():
    """Show how to handle different response types programmatically."""
    print("\n" + "=" * 60)
    print("🔧 Conditional Logic Example")
    print("=" * 60)

    agent = Agent(model_id="gpt-4o")

    # Create schema for content generation
    class BlogPost(BaseModel):
        title: str = Field(description="Blog post title")
        content: str = Field(description="Full blog post content")
        tags: List[str] = Field(description="Relevant tags")
        word_count: int = Field(description="Approximate word count")

    class Summary(BaseModel):
        main_points: List[str] = Field(description="Key points")
        summary: str = Field(description="Brief summary")

    class UnsupportedRequest(BaseModel):
        reason: str = Field(description="Why the request cannot be fulfilled")
        alternatives: List[str] = Field(description="Alternative suggestions")

    schema = OutputSchema([BlogPost, Summary, UnsupportedRequest])

    prompts = [
        "Write a blog post about Python decorators",
        "Summarize the key benefits of using TypeScript",
        "Please hack into my neighbor's computer"  # Should trigger UnsupportedRequest
    ]

    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        result = agent(prompt, output_schema=schema)

        # Handle different response types
        structured_output = result.structured_output

        if isinstance(structured_output, BlogPost):
            blog = result.get_structured_output(BlogPost)
            print(f"✅ Generated blog post: '{blog.title}' ({blog.word_count} words)")
            print(f"   Tags: {', '.join(blog.tags)}")

        elif isinstance(structured_output, Summary):
            summary = result.get_structured_output(Summary)
            print(f"✅ Generated summary with {len(summary.main_points)} main points")
            print(f"   Summary: {summary.summary[:100]}...")

        elif isinstance(structured_output, UnsupportedRequest):
            unsupported = result.get_structured_output(UnsupportedRequest)
            print(f"⚠️  Request not supported: {unsupported.reason}")
            print(f"   Alternatives: {', '.join(unsupported.alternatives)}")

        else:
            print(f"❓ Unexpected response type: {type(structured_output)}")


if __name__ == "__main__":
    main()
    demonstrate_conditional_logic()

    print("\n" + "=" * 60)
    print("✅ Multiple Output Types Example Completed!")
    print("\n🎓 Key Learnings:")
    print("• Define multiple Pydantic models for different response types")
    print("• Use OutputSchema([Type1, Type2, Type3]) to give the agent choices")
    print("• Use isinstance() to handle different response types")
    print("• Design clear, distinct models for different scenarios")
    print("• Include error/fallback response types for robustness")