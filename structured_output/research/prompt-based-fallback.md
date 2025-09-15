# Prompt-Based Fallback Strategy Research

## Prompt Engineering for Structured Output

When native structured output and JSON schema approaches fail, we need a robust prompt-based fallback that can work with any text-generation model.

## Prompt Template Design

### Basic Template Structure
```python
STRUCTURED_OUTPUT_PROMPT = """
You must respond with valid JSON that matches this exact schema:

{json_schema}

Requirements:
- Response must be valid JSON only
- No additional text before or after the JSON
- All required fields must be present
- Follow the exact field names and types specified

User request: {user_prompt}

JSON Response:
"""
```

### Enhanced Template with Examples
```python
STRUCTURED_OUTPUT_PROMPT_WITH_EXAMPLES = """
You must respond with valid JSON matching this schema:

Schema:
{json_schema}

Example valid response:
{example_json}

Rules:
1. Return ONLY valid JSON, no other text
2. Include all required fields
3. Use exact field names from schema
4. Follow data types specified

User request: {user_prompt}

JSON:
"""
```

## JSON Parsing Strategy

### Robust JSON Extraction
```python
def extract_json_from_response(response_text: str) -> Optional[dict]:
    """Extract JSON from model response with multiple fallback strategies."""
    
    # Strategy 1: Direct JSON parsing
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find JSON block markers
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # ```json {...} ```
        r'```\s*(\{.*?\})\s*```',      # ``` {...} ```
        r'(\{.*\})',                   # First {...} block
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Clean and retry
    cleaned = clean_json_response(response_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def clean_json_response(text: str) -> str:
    """Clean common JSON formatting issues."""
    # Remove common prefixes/suffixes
    text = re.sub(r'^[^{]*', '', text)  # Remove text before first {
    text = re.sub(r'[^}]*$', '', text)  # Remove text after last }
    
    # Fix common issues
    text = text.replace('```json', '').replace('```', '')
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    return text.strip()
```

## Schema Generation for Prompts

### Convert Pydantic to Human-Readable Schema
```python
def pydantic_to_prompt_schema(model: Type[BaseModel]) -> str:
    """Convert Pydantic model to human-readable schema for prompts."""
    schema = model.model_json_schema()
    
    # Simplify schema for prompt clarity
    simplified = {
        "type": "object",
        "properties": {},
        "required": schema.get("required", [])
    }
    
    for prop_name, prop_def in schema.get("properties", {}).items():
        simplified["properties"][prop_name] = {
            "type": prop_def.get("type", "string"),
            "description": prop_def.get("description", "")
        }
    
    return json.dumps(simplified, indent=2)

def generate_example_json(model: Type[BaseModel]) -> str:
    """Generate example JSON for the prompt."""
    # Create instance with example data
    example_data = {}
    schema = model.model_json_schema()
    
    for prop_name, prop_def in schema.get("properties", {}).items():
        prop_type = prop_def.get("type", "string")
        if prop_type == "string":
            example_data[prop_name] = f"example_{prop_name}"
        elif prop_type == "integer":
            example_data[prop_name] = 42
        elif prop_type == "boolean":
            example_data[prop_name] = True
        # Add more type handlers as needed
    
    return json.dumps(example_data, indent=2)
```

## Integration with Existing Providers

### Fallback Integration Pattern
```python
async def prompt_based_structured_output(
    model: Model, 
    output_model: Type[T], 
    messages: Messages,
    system_prompt: Optional[str] = None
) -> Optional[T]:
    """Implement prompt-based structured output as fallback."""
    
    # Generate prompt components
    schema_str = pydantic_to_prompt_schema(output_model)
    example_str = generate_example_json(output_model)
    
    # Get user's last message
    user_prompt = extract_user_prompt(messages)
    
    # Create structured output prompt
    structured_prompt = STRUCTURED_OUTPUT_PROMPT_WITH_EXAMPLES.format(
        json_schema=schema_str,
        example_json=example_str,
        user_prompt=user_prompt
    )
    
    # Create new messages with structured prompt
    structured_messages = create_structured_messages(messages, structured_prompt)
    
    # Get response from model
    response = model.stream(structured_messages, system_prompt=system_prompt)
    full_response = await collect_full_response(response)
    
    # Extract and parse JSON
    json_data = extract_json_from_response(full_response)
    if json_data:
        try:
            return output_model(**json_data)
        except ValidationError as e:
            logger.warning(f"Pydantic validation failed: {e}")
            return None
    
    return None
```

## Error Recovery Strategies

### Multiple Prompt Attempts
```python
PROMPT_VARIATIONS = [
    # Variation 1: Strict format
    "Return only valid JSON matching this schema: {schema}",
    
    # Variation 2: With examples  
    "Generate JSON like this example: {example}. Schema: {schema}",
    
    # Variation 3: Step by step
    "First understand the schema: {schema}. Then create JSON for: {prompt}",
]

async def prompt_with_retries(model: Model, output_model: Type[T], max_retries: int = 3) -> Optional[T]:
    """Try multiple prompt variations if first attempt fails."""
    for i, prompt_template in enumerate(PROMPT_VARIATIONS[:max_retries]):
        try:
            result = await prompt_based_structured_output(model, output_model, prompt_template)
            if result:
                return result
        except Exception as e:
            logger.warning(f"Prompt attempt {i+1} failed: {e}")
            continue
    return None
```
