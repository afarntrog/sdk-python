#!/usr/bin/env python3
"""
Output Modes Comparison Example

This example demonstrates the three different output modes available in the Strands SDK:
1. ToolOutput (default) - Uses function calling
2. NativeOutput - Uses model's native structured output capabilities
3. PromptedOutput - Uses carefully crafted prompts

This helps you understand when to use each mode and their trade-offs.
"""

import time
from typing import List, Optional

from pydantic import BaseModel, Field

from strands import Agent
from strands.output import NativeOutput, PromptedOutput, ToolOutput


class ProductReview(BaseModel):
    """A structured product review."""

    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(description="Rating from 1 to 5 stars", ge=1, le=5)
    review_text: str = Field(description="Detailed review text")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    recommended: bool = Field(description="Whether the reviewer recommends the product")
    price_value: Optional[str] = Field(default=None, description="Assessment of price vs value")


def test_output_mode(agent: Agent, mode_name: str, prompt: str) -> tuple[ProductReview, float]:
    """Test a specific output mode and measure performance."""
    print(f"\n🔄 Testing {mode_name}...")

    start_time = time.time()
    result = agent(prompt, output_type=ProductReview)
    end_time = time.time()

    duration = end_time - start_time
    review = result.get_structured_output(ProductReview)

    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"📦 Product: {review.product_name}")
    print(f"⭐ Rating: {review.rating}/5")
    print(f"👍 Recommended: {'Yes' if review.recommended else 'No'}")

    return review, duration


def main():
    """Compare different output modes with the same prompt."""
    print("🔄 Output Modes Comparison Example")
    print("=" * 70)

    # The prompt we'll use for all tests
    test_prompt = """
    Write a review for the iPhone 15 Pro Max based on typical user feedback.
    Consider aspects like camera quality, battery life, build quality, and value for money.
    """

    print(f"📝 Test prompt: {test_prompt.strip()}")
    print("\n" + "=" * 70)

    # Store results for comparison
    results = {}

    # 1. ToolOutput Mode (Default)
    print("\n1️⃣ TOOL OUTPUT MODE (Default)")
    print("-" * 40)
    print("How it works: Converts your Pydantic model to a function schema")
    print("              and uses the model's function calling capability")
    print("Advantages:   Most reliable, works with all models, best validation")
    print("Disadvantages: Slight overhead from function call mechanism")

    agent_tool = Agent(
        model_id="gpt-4o",
        output_mode=ToolOutput()
    )

    tool_review, tool_duration = test_output_mode(agent_tool, "ToolOutput", test_prompt)
    results['ToolOutput'] = (tool_review, tool_duration)

    # 2. NativeOutput Mode
    print("\n2️⃣ NATIVE OUTPUT MODE")
    print("-" * 40)
    print("How it works: Uses the model's built-in structured output features")
    print("              (e.g., OpenAI's response_format with strict mode)")
    print("Advantages:   Potentially faster, direct model integration")
    print("Disadvantages: Limited model support, falls back to ToolOutput if unsupported")

    agent_native = Agent(
        model_id="gpt-4o",
        output_mode=NativeOutput()
    )

    try:
        native_review, native_duration = test_output_mode(agent_native, "NativeOutput", test_prompt)
        results['NativeOutput'] = (native_review, native_duration)
    except Exception as e:
        print(f"❌ NativeOutput failed (falling back to ToolOutput): {e}")
        # This would normally happen automatically, but we're simulating here
        results['NativeOutput'] = (None, 0)

    # 3. PromptedOutput Mode
    print("\n3️⃣ PROMPTED OUTPUT MODE")
    print("-" * 40)
    print("How it works: Adds instructions to your prompt requesting JSON output")
    print("              matching your Pydantic schema")
    print("Advantages:   Works with any model, even those without function calling")
    print("Disadvantages: Less reliable, depends on model following instructions")

    agent_prompted = Agent(
        model_id="gpt-4o",
        output_mode=PromptedOutput(
            template="""
Please respond with a valid JSON object that matches this exact schema:
{schema}

Your response should be ONLY the JSON object, nothing else.
            """.strip()
        )
    )

    try:
        prompted_review, prompted_duration = test_output_mode(agent_prompted, "PromptedOutput", test_prompt)
        results['PromptedOutput'] = (prompted_review, prompted_duration)
    except Exception as e:
        print(f"❌ PromptedOutput failed: {e}")
        results['PromptedOutput'] = (None, 0)

    # Comparison Summary
    print("\n" + "=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)

    print(f"{'Mode':<15} {'Duration':<10} {'Success':<8} {'Notes'}")
    print("-" * 60)

    for mode_name, (review, duration) in results.items():
        success = "✅ Yes" if review else "❌ No"
        duration_str = f"{duration:.2f}s" if duration > 0 else "N/A"
        notes = ""

        if mode_name == "ToolOutput":
            notes = "Most reliable"
        elif mode_name == "NativeOutput":
            notes = "Model-dependent" if review else "Fell back to ToolOutput"
        elif mode_name == "PromptedOutput":
            notes = "Prompt-dependent"

        print(f"{mode_name:<15} {duration_str:<10} {success:<8} {notes}")


def demonstrate_fallback_behavior():
    """Show how automatic fallback works when a mode isn't supported."""
    print("\n" + "=" * 70)
    print("🔄 AUTOMATIC FALLBACK DEMONSTRATION")
    print("=" * 70)

    print("When you specify NativeOutput but the model doesn't support it,")
    print("the system automatically falls back to ToolOutput.")

    # Simulate using a model that doesn't support native structured output
    # (In reality, this would be detected automatically)
    print("\nTesting with a model that doesn't support native structured output...")

    agent = Agent(
        model_id="claude-3-sonnet",  # Anthropic models don't support native structured output
        output_mode=NativeOutput()   # This will automatically fall back to ToolOutput
    )

    print("✅ Agent created with NativeOutput mode")
    print("🔄 Making request... (should automatically use ToolOutput as fallback)")

    result = agent("Create a simple product review for AirPods Pro", output_type=ProductReview)
    review = result.get_structured_output(ProductReview)

    print(f"✅ Successfully got structured output: {review.product_name}")
    print("💡 The system automatically used ToolOutput as fallback!")


def choosing_the_right_mode():
    """Provide guidance on when to use each mode."""
    print("\n" + "=" * 70)
    print("🎯 CHOOSING THE RIGHT OUTPUT MODE")
    print("=" * 70)

    guidance = {
        "ToolOutput": {
            "use_when": [
                "You want maximum reliability and compatibility",
                "You're working with complex nested data structures",
                "You need strong validation guarantees",
                "You're not sure about model capabilities"
            ],
            "avoid_when": [
                "You need absolute maximum performance (rare)",
                "You're working with models that don't support function calling"
            ]
        },
        "NativeOutput": {
            "use_when": [
                "You're using OpenAI GPT-4 or other supporting models",
                "You need maximum performance and speed",
                "You want the strictest possible schema adherence"
            ],
            "avoid_when": [
                "You need to support multiple model providers",
                "You're not sure if your model supports it",
                "You're working with complex schemas that might not be supported"
            ]
        },
        "PromptedOutput": {
            "use_when": [
                "You're working with models that don't support function calling",
                "You want to customize the prompting strategy",
                "You're working with legacy models",
                "You want to experiment with different prompt templates"
            ],
            "avoid_when": [
                "You need guaranteed structured output reliability",
                "You're working with very complex schemas",
                "The model tends to ignore instructions"
            ]
        }
    }

    for mode, info in guidance.items():
        print(f"\n🔧 {mode}:")
        print("   ✅ Use when:")
        for use_case in info["use_when"]:
            print(f"      • {use_case}")
        print("   ❌ Avoid when:")
        for avoid_case in info["avoid_when"]:
            print(f"      • {avoid_case}")


if __name__ == "__main__":
    main()
    demonstrate_fallback_behavior()
    choosing_the_right_mode()

    print("\n" + "=" * 70)
    print("✅ Output Modes Comparison Complete!")
    print("\n🎓 Key Takeaways:")
    print("• ToolOutput is the safe default choice for most use cases")
    print("• NativeOutput offers performance benefits when supported")
    print("• PromptedOutput provides flexibility for special cases")
    print("• The system automatically handles fallbacks for you")
    print("• Choose based on your reliability vs performance needs")