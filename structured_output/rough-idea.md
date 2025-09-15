# Structured Output Implementation - Rough Idea

You must read the entire #codebase and then come up with an implementation plan to update the Agent so that it accepts an output_type parameter that takes in a Pydantic model. The agent will then return the result converted to the provided pydantic model as part of the AgentResult return object. This is what we call Structured Output. It is currently implemented differently in this codebase which causes lots of issues. You should understand how we implement it now (for each model provider) and then you can understand how to fix it.

## Context
- This is for the Strands Agents SDK
- Current implementation has issues across different model providers
- Need to standardize structured output handling
- Should integrate with existing AgentResult return object
- Must support Pydantic models as output types
