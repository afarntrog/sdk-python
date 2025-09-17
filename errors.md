print("📇 Contact Information Extraction")
print("=" * 50)

# agent = Agent(output_mode=ToolOutput())
agent = Agent()

# Sample unstructured text with contact information
contact_text = """
Hi there! My name is Sarah Johnson and I'm the Marketing Director at TechCorp Solutions.
You can reach me at sarah.johnson@techcorp.com or call me at (555) 123-4567.
Our office is located at 123 Business Ave, Suite 456, San Francisco, CA 94105.
I'd love to discuss potential collaboration opportunities!
"""

print("📄 Input text:")
print(contact_text.strip())

result = agent(
    f"Extract contact information from this text: {contact_text}",
    output_type=ContactInfo
)

contact = result.get_structured_output(ContactInfo)

<ERROR>
: PydanticDeprecatedSince20: Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.11/migration/
  @validator('email')
📇 Contact Information Extraction
==================================================
📄 Input text:
Hi there! My name is Sarah Johnson and I'm the Marketing Director at TechCorp Solutions.
You can reach me at sarah.johnson@techcorp.com or call me at (555) 123-4567.
Our office is located at 123 Business Ave, Suite 456, San Francisco, CA 94105.
I'd love to discuss potential collaboration opportunities!
I'll extract the contact information from the text for you.
Tool #1: ContactInfo
tool_name=<ContactInfo>, available_tools=<[]> | tool not found in registry
Failed to extract structured output from tool result: Response validation failed for ContactInfo: 1 validation error for ContactInfo
name
  Field required [type=missing, input_value={'text': 'Unknown tool: ContactInfo'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/missing
I apologize for the error. Let me extract the contact information using the correct function:
Tool #2: ContactInfo
tool_name=<ContactInfo>, available_tools=<[]> | tool not found in registry
Failed to extract structured output from tool result: Response validation failed for ContactInfo: 1 validation error for ContactInfo
name
  Field required [type=missing, input_value={'text': 'Unknown tool: ContactInfo'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/missing
I see the issue - the function definition shows the name but it's not being recognized. Based on the text you provided, here's the extracted contact information:

**Name:** Sarah Johnson  
**Title:** Marketing Director  
**Company:** TechCorp Solutions  
**Email:** sarah.johnson@techcorp.com  
**Phone:** (555) 123-4567  
**Address:** 123 Business Ave, Suite 456, San Francisco, CA 94105

The text contains complete contact information including personal details, professional information, and multiple ways to reach Sarah Johnson for potential collaboration opportunities.
</ERROR>