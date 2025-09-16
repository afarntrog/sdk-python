https://ai.pydantic.dev/output/#structured-output
My testing:
```python

from pydantic_ai import Agent
import os
os.environ['GOOGLE_API_KEY'] = 'put_key_here'

class CityLocation(BaseModel):
    city: str
    country: str

agent = Agent('google-gla:gemini-1.5-flash', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')

```

`>>> result.all_messages_json()`
```
[
    {
        "parts": [
            {
                "content": "Where were the olympics held in 2012?",
                "timestamp": "2025-09-15T00:42:43.633897Z",
                "part_kind": "user-prompt"
            }
        ],
        "instructions": null,
        "kind": "request"
    },
    {
        "parts": [
            {
                "tool_name": "final_result",
                "args": {
                    "city": "London",
                    "country": "UK"
                },
                "tool_call_id": "pyd_ai_15aaa0edc36e46fe97c04b3ee3f7efcc",
                "part_kind": "tool-call"
            }
        ],
        "usage": {
            "input_tokens": 30,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 7,
            "input_audio_tokens": 0,
            "cache_audio_read_tokens": 0,
            "output_audio_tokens": 0,
            "details": {
                "text_prompt_tokens": 30,
                "text_candidates_tokens": 7
            }
        },
        "model_name": "gemini-1.5-flash",
        "timestamp": "2025-09-15T00:42:49.589814Z",
        "kind": "response",
        "provider_name": "google-gla",
        "provider_details": {
            "finish_reason": "STOP"
        },
        "provider_response_id": "CWHHaM_NDY7O_uMPsuKe6Aw"
    },
    {
        "parts": [
            {
                "tool_name": "final_result",
                "content": "Final result processed.",
                "tool_call_id": "pyd_ai_15aaa0edc36e46fe97c04b3ee3f7efcc",
                "metadata": null,
                "timestamp": "2025-09-15T00:42:49.591310Z",
                "part_kind": "tool-return"
            }
        ],
        "instructions": null,
        "kind": "request"
    }
]
```

`>>> result.all_messages()`
```
[ModelRequest(parts=[UserPromptPart(content='Where were the olympics held in 2012?', timestamp=datetime.datetime(2025, 9, 16, 19, 42, 37, 93895, tzinfo=datetime.timezone.utc))]),
 ModelResponse(parts=[ToolCallPart(tool_name='final_result', args={'country': 'United Kingdom', 'city': 'London'}, tool_call_id='pyd_ai_26c6b49a1bec42f2a992a77616ce2e90')], usage=RequestUsage(input_tokens=30, output_tokens=8, details={'text_prompt_tokens': 30, 'text_candidates_tokens': 8}), model_name='gemini-1.5-flash', timestamp=datetime.datetime(2025, 9, 16, 19, 42, 37, 721805, tzinfo=datetime.timezone.utc), provider_name='google-gla', provider_details={'finish_reason': 'STOP'}, provider_response_id='rb3JaKOOEviBjMcP6byk-Qw'),
 ModelRequest(parts=[ToolReturnPart(tool_name='final_result', content='Final result processed.', tool_call_id='pyd_ai_26c6b49a1bec42f2a992a77616ce2e90', timestamp=datetime.datetime(2025, 9, 16, 19, 52, 23, 753725, tzinfo=datetime.timezone.utc))])]
 ```

## Native Output:
```python
from pydantic_ai import Agent, NativeOutput
import os

os.environ['OPENAI_API_KEY'] = 'put key here'

class Fruit(BaseModel):
    name: str
    color: str


class Vehicle(BaseModel):
    name: str
    wheels: int



agent = Agent(
    'openai:gpt-4o',
    output_type=NativeOutput(
        [Fruit, Vehicle], 
        name='Fruit_or_vehicle',
        description='Return a fruit or vehicle.'
    ),
)
result = agent.run_sync('What is a Ford Explorer?')
print(repr(result.output))
#> Vehicle(name='Ford Explorer', wheels=4)
```


Result:

`result.all_messages_json()`
```
[
    {
        "parts": [
            {
                "content": "What is a Ford Explorer?",
                "timestamp": "2025-09-16T21:12:33.120727Z",
                "part_kind": "user-prompt"
            }
        ],
        "instructions": null,
        "kind": "request"
    },
    {
        "parts": [
            {
                "content": "{\\"result\\":{\\"kind\\":\\"Vehicle\\",\\"data\\":{\\"name\\":\\"Ford Explorer\\",\\"wheels\\":4}}}",
                "part_kind": "text"
            }
        ],
        "usage": {
            "input_tokens": 127,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 20,
            "input_audio_tokens": 0,
            "cache_audio_read_tokens": 0,
            "output_audio_tokens": 0,
            "details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        },
        "model_name": "gpt-4o-2024-08-06",
        "timestamp": "2025-09-16T21:12:34Z",
        "kind": "response",
        "provider_name": "openai",
        "provider_details": null,
        "provider_response_id": "chatcmpl-CGXGUmce3ZuxcHLK7qDiMFnT1OlGJ"
    }
]
```


## Structured Output with regular tools (and SO tool)
```python
import random

from pydantic_ai._run_context import RunContext
import pydantic
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps

class DiceRoll(pydantic.BaseModel):
    """A dice roll."""

    player_name: str
    result: str


agent_a = Agent(
    'google-gla:gemini-1.5-flash',
    output_type=DiceRoll,
    tools=[roll_dice, get_player_name],  
)

result = agent_a.run_sync('roll dice?')
result.all_messages_json()

```

Result:

```
[
    {
        "parts": [
            {
                "content": "roll dice?",
                "timestamp": "2025-09-16T21:44:39.499147Z",
                "part_kind": "user-prompt"
            }
        ],
        "instructions": null,
        "kind": "request"
    },
    {
        "parts": [
            {
                "tool_name": "roll_dice",
                "args": {},
                "tool_call_id": "pyd_ai_6c8e6f11c43d49b09d5f3720fd4d5a7e",
                "part_kind": "tool-call"
            }
        ],
        "usage": {
            "input_tokens": 49,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 3,
            "input_audio_tokens": 0,
            "cache_audio_read_tokens": 0,
            "output_audio_tokens": 0,
            "details": {
                "text_prompt_tokens": 49,
                "text_candidates_tokens": 3
            }
        },
        "model_name": "gemini-1.5-flash",
        "timestamp": "2025-09-16T21:44:40.144280Z",
        "kind": "response",
        "provider_name": "google-gla",
        "provider_details": {
            "finish_reason": "STOP"
        },
        "provider_response_id": "R9rJaM-kLYCOjMcP7_SR6Aw"
    },
    {
        "parts": [
            {
                "tool_name": "roll_dice",
                "content": "2",
                "tool_call_id": "pyd_ai_6c8e6f11c43d49b09d5f3720fd4d5a7e",
                "metadata": null,
                "timestamp": "2025-09-16T21:44:40.148527Z",
                "part_kind": "tool-return"
            }
        ],
        "instructions": null,
        "kind": "request"
    },
    {
        "parts": [
            {
                "tool_name": "get_player_name",
                "args": {},
                "tool_call_id": "pyd_ai_4f3022da34c24d5b8d34fb7ba49a9ffb",
                "part_kind": "tool-call"
            }
        ],
        "usage": {
            "input_tokens": 59,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 5,
            "input_audio_tokens": 0,
            "cache_audio_read_tokens": 0,
            "output_audio_tokens": 0,
            "details": {
                "text_prompt_tokens": 59,
                "text_candidates_tokens": 5
            }
        },
        "model_name": "gemini-1.5-flash",
        "timestamp": "2025-09-16T21:44:40.689195Z",
        "kind": "response",
        "provider_name": "google-gla",
        "provider_details": {
            "finish_reason": "STOP"
        },
        "provider_response_id": "SNrJaLfpEMC1jMcPi4jq-QE"
    },
    {
        "parts": [
            {
                "tool_name": "get_player_name",
                "content": null,
                "tool_call_id": "pyd_ai_4f3022da34c24d5b8d34fb7ba49a9ffb",
                "metadata": null,
                "timestamp": "2025-09-16T21:44:40.690393Z",
                "part_kind": "tool-return"
            }
        ],
        "instructions": null,
        "kind": "request"
    },
    {
        "parts": [
            {
                "tool_name": "final_result",
                "args": {
                    "player_name": "Mickey",
                    "result": "2"
                },
                "tool_call_id": "pyd_ai_fffb29090909493a86bbfe4dc12d65fd",
                "part_kind": "tool-call"
            }
        ],
        "usage": {
            "input_tokens": 73,
            "cache_write_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 9,
            "input_audio_tokens": 0,
            "cache_audio_read_tokens": 0,
            "output_audio_tokens": 0,
            "details": {
                "text_prompt_tokens": 73,
                "text_candidates_tokens": 9
            }
        },
        "model_name": "gemini-1.5-flash",
        "timestamp": "2025-09-16T21:44:41.298016Z",
        "kind": "response",
        "provider_name": "google-gla",
        "provider_details": {
            "finish_reason": "STOP"
        },
        "provider_response_id": "SNrJaOCQMa7fjMcPtKqQ0Aw"
    },
    {
        "parts": [
            {
                "tool_name": "final_result",
                "content": "Final result processed.",
                "tool_call_id": "pyd_ai_fffb29090909493a86bbfe4dc12d65fd",
                "metadata": null,
                "timestamp": "2025-09-16T21:44:41.298280Z",
                "part_kind": "tool-return"
            }
        ],
        "instructions": null,
        "kind": "request"
    }
]
```


---
---
# llama_index
---
---


llama_index does not include the tool calls as part of the request object
```python

from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field

llm = OpenAI(model="gpt-4.1")


## define structured output format  and tools
class MathResult(BaseModel):
    operation: str = Field(description="the performed operation")
    result: int = Field(description="the result of the operation")



## define agent
agent = FunctionAgent(
    name="calculator",
    system_prompt="You are a calculator agent who can multiply two numbers using the `multiply` tool.",
    output_cls=MathResult,
    llm=llm,
)

response = await agent.run("What is 3415 * 43144?")
```

`>>> response.structured_response`
```
{'operation': '3415 * 43144', 'result': 147367960}
```

`>>> response.get_pydantic_model(MathResult)`
```
MathResult(operation='3415 * 43144', result=147367960)
```

`>>> response.tool_calls`

[]


`>>> response`
```
AgentOutput(response=ChatMessage(role=<MessageRole.ASSISTANT: 'assistant'>, additional_kwargs={}, blocks=[TextBlock(block_type='text', text='To find the product of 3415 and 43144, I will multiply the two numbers.\n\nmultiply(3415, 43144)')]), structured_response={'operation': '3415 * 43144', 'result': 147367960}, current_agent_name='calculator', raw={'id': 'chatcmpl-CGWbdqUGQ85WS5oScFZLK34ZIRKVB', 'choices': [{'delta': {'content': None, 'function_call': None, 'refusal': None, 'role': None, 'tool_calls': None}, 'finish_reason': 'stop', 'index': 0, 'logprobs': None}], 'created': 1758054621, 'model': 'gpt-4.1-2025-04-14', 'object': 'chat.completion.chunk', 'service_tier': 'default', 'system_fingerprint': 'fp_3502f4eb73', 'usage': None, 'obfuscation': 'vZLG0a'}, tool_calls=[], retry_messages=[])
```

`>>> response.json()`
```
{
    "response": {
        "role": "assistant",
        "additional_kwargs": {},
        "blocks": [
            {
                "block_type": "text",
                "text": "To find the product of 3415 and 43144, I will multiply the two numbers.\\n\\nmultiply(3415, 43144)",
            }
        ],
    },
    "structured_response": {"operation": "3415 * 43144", "result": 147367960},
    "current_agent_name": "calculator",
    "tool_calls": [],
    "retry_messages": [],
}
```
