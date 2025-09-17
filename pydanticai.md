Directory structure:
└── pydantic_ai/
    ├── __init__.py
    ├── __main__.py
    ├── _a2a.py
    ├── _cli.py
    ├── _function_schema.py
    ├── _griffe.py
    ├── _mcp.py
    ├── _otel_messages.py
    ├── _output.py
    ├── _parts_manager.py
    ├── _run_context.py
    ├── _system_prompt.py
    ├── _thinking_part.py
    ├── _tool_manager.py
    ├── _utils.py
    ├── ag_ui.py
    ├── builtin_tools.py
    ├── direct.py
    ├── exceptions.py
    ├── format_prompt.py
    ├── mcp.py
    ├── output.py
    ├── py.typed
    ├── result.py
    ├── retries.py
    ├── run.py
    ├── settings.py
    ├── tools.py
    ├── usage.py
    ├── agent/
    │   ├── abstract.py
    │   └── wrapper.py
    ├── common_tools/
    │   ├── __init__.py
    │   ├── duckduckgo.py
    │   └── tavily.py
    ├── durable_exec/
    │   ├── __init__.py
    │   ├── dbos/
    │   │   ├── __init__.py
    │   │   ├── _agent.py
    │   │   ├── _mcp_server.py
    │   │   ├── _model.py
    │   │   └── _utils.py
    │   └── temporal/
    │       ├── __init__.py
    │       ├── _agent.py
    │       ├── _function_toolset.py
    │       ├── _logfire.py
    │       ├── _mcp_server.py
    │       ├── _model.py
    │       ├── _run_context.py
    │       └── _toolset.py
    ├── ext/
    │   ├── __init__.py
    │   ├── aci.py
    │   └── langchain.py
    ├── models/
    │   ├── __init__.py
    │   ├── anthropic.py
    │   ├── bedrock.py
    │   ├── cohere.py
    │   ├── fallback.py
    │   ├── function.py
    │   ├── gemini.py
    │   ├── google.py
    │   ├── groq.py
    │   ├── huggingface.py
    │   ├── instrumented.py
    │   ├── mcp_sampling.py
    │   ├── mistral.py
    │   ├── test.py
    │   └── wrapper.py
    ├── profiles/
    │   ├── __init__.py
    │   ├── _json_schema.py
    │   ├── amazon.py
    │   ├── anthropic.py
    │   ├── cohere.py
    │   ├── deepseek.py
    │   ├── google.py
    │   ├── grok.py
    │   ├── groq.py
    │   ├── harmony.py
    │   ├── meta.py
    │   ├── mistral.py
    │   ├── moonshotai.py
    │   ├── openai.py
    │   └── qwen.py
    ├── providers/
    │   ├── __init__.py
    │   ├── anthropic.py
    │   ├── azure.py
    │   ├── bedrock.py
    │   ├── cerebras.py
    │   ├── cohere.py
    │   ├── deepseek.py
    │   ├── fireworks.py
    │   ├── gateway.py
    │   ├── github.py
    │   ├── google.py
    │   ├── google_gla.py
    │   ├── google_vertex.py
    │   ├── grok.py
    │   ├── groq.py
    │   ├── heroku.py
    │   ├── huggingface.py
    │   ├── litellm.py
    │   ├── mistral.py
    │   ├── moonshotai.py
    │   ├── ollama.py
    │   ├── openai.py
    │   ├── openrouter.py
    │   ├── together.py
    │   └── vercel.py
    └── toolsets/
        ├── __init__.py
        ├── _dynamic.py
        ├── abstract.py
        ├── approval_required.py
        ├── combined.py
        ├── external.py
        ├── filtered.py
        ├── function.py
        ├── prefixed.py
        ├── prepared.py
        ├── renamed.py
        └── wrapper.py


Files Content:

(Files content cropped to 300k characters, download full ingest to see more)
================================================
FILE: pydantic_ai_slim/pydantic_ai/__init__.py
================================================
from importlib.metadata import version as _metadata_version

from .agent import (
    Agent,
    CallToolsNode,
    EndStrategy,
    InstrumentationSettings,
    ModelRequestNode,
    UserPromptNode,
    capture_run_messages,
)
from .builtin_tools import CodeExecutionTool, UrlContextTool, WebSearchTool, WebSearchUserLocation
from .exceptions import (
    AgentRunError,
    ApprovalRequired,
    CallDeferred,
    FallbackExceptionGroup,
    ModelHTTPError,
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from .format_prompt import format_as_xml
from .messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, VideoUrl
from .output import NativeOutput, PromptedOutput, StructuredDict, TextOutput, ToolOutput
from .settings import ModelSettings
from .tools import DeferredToolRequests, DeferredToolResults, RunContext, Tool, ToolApproved, ToolDefinition, ToolDenied
from .usage import RequestUsage, RunUsage, UsageLimits

__all__ = (
    '__version__',
    # agent
    'Agent',
    'EndStrategy',
    'CallToolsNode',
    'ModelRequestNode',
    'UserPromptNode',
    'capture_run_messages',
    'InstrumentationSettings',
    # exceptions
    'AgentRunError',
    'CallDeferred',
    'ApprovalRequired',
    'ModelRetry',
    'ModelHTTPError',
    'FallbackExceptionGroup',
    'UnexpectedModelBehavior',
    'UsageLimitExceeded',
    'UserError',
    # messages
    'ImageUrl',
    'AudioUrl',
    'VideoUrl',
    'DocumentUrl',
    'BinaryContent',
    # tools
    'Tool',
    'ToolDefinition',
    'RunContext',
    'DeferredToolRequests',
    'DeferredToolResults',
    'ToolApproved',
    'ToolDenied',
    # builtin_tools
    'WebSearchTool',
    'WebSearchUserLocation',
    'UrlContextTool',
    'CodeExecutionTool',
    # output
    'ToolOutput',
    'NativeOutput',
    'PromptedOutput',
    'TextOutput',
    'StructuredDict',
    # format_prompt
    'format_as_xml',
    # settings
    'ModelSettings',
    # usage
    'RunUsage',
    'RequestUsage',
    'UsageLimits',
)
__version__ = _metadata_version('pydantic_ai_slim')



================================================
FILE: pydantic_ai_slim/pydantic_ai/__main__.py
================================================
"""This means `python -m pydantic_ai` should run the CLI."""

from ._cli import cli_exit

if __name__ == '__main__':
    cli_exit()



================================================
FILE: pydantic_ai_slim/pydantic_ai/_a2a.py
================================================
from __future__ import annotations, annotations as _annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter
from typing_extensions import assert_never

from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
    VideoUrl,
)

from .agent import AbstractAgent, AgentDepsT, OutputDataT

# AgentWorker output type needs to be invariant for use in both parameter and return positions
WorkerOutputT = TypeVar('WorkerOutputT')

try:
    from fasta2a.applications import FastA2A
    from fasta2a.broker import Broker, InMemoryBroker
    from fasta2a.schema import (
        AgentProvider,
        Artifact,
        DataPart,
        Message,
        Part,
        Skill,
        TaskIdParams,
        TaskSendParams,
        TextPart as A2ATextPart,
    )
    from fasta2a.storage import InMemoryStorage, Storage
    from fasta2a.worker import Worker
    from starlette.middleware import Middleware
    from starlette.routing import Route
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as _import_error:
    raise ImportError(
        'Please install the `fasta2a` package to use `Agent.to_a2a()` method, '
        'you can use the `a2a` optional group — `pip install "pydantic-ai-slim[a2a]"`'
    ) from _import_error


@asynccontextmanager
async def worker_lifespan(
    app: FastA2A, worker: Worker, agent: AbstractAgent[AgentDepsT, OutputDataT]
) -> AsyncIterator[None]:
    """Custom lifespan that runs the worker during application startup.

    This ensures the worker is started and ready to process tasks as soon as the application starts.
    """
    async with app.task_manager, agent:
        async with worker.run():
            yield


def agent_to_a2a(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    storage: Storage | None = None,
    broker: Broker | None = None,
    # Agent card
    name: str | None = None,
    url: str = 'http://localhost:8000',
    version: str = '1.0.0',
    description: str | None = None,
    provider: AgentProvider | None = None,
    skills: list[Skill] | None = None,
    # Starlette
    debug: bool = False,
    routes: Sequence[Route] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: dict[Any, ExceptionHandler] | None = None,
    lifespan: Lifespan[FastA2A] | None = None,
) -> FastA2A:
    """Create a FastA2A server from an agent."""
    storage = storage or InMemoryStorage()
    broker = broker or InMemoryBroker()
    worker = AgentWorker(agent=agent, broker=broker, storage=storage)

    lifespan = lifespan or partial(worker_lifespan, worker=worker, agent=agent)

    return FastA2A(
        storage=storage,
        broker=broker,
        name=name or agent.name,
        url=url,
        version=version,
        description=description,
        provider=provider,
        skills=skills,
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        lifespan=lifespan,
    )


@dataclass
class AgentWorker(Worker[list[ModelMessage]], Generic[WorkerOutputT, AgentDepsT]):
    """A worker that uses an agent to execute tasks."""

    agent: AbstractAgent[AgentDepsT, WorkerOutputT]

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'])
        if task is None:
            raise ValueError(f'Task {params["id"]} not found')  # pragma: no cover

        # TODO(Marcelo): Should we lock `run_task` on the `context_id`?
        # Ensure this task hasn't been run before
        if task['status']['state'] != 'submitted':
            raise ValueError(  # pragma: no cover
                f'Task {params["id"]} has already been processed (state: {task["status"]["state"]})'
            )

        await self.storage.update_task(task['id'], state='working')

        # Load context - contains pydantic-ai message history from previous tasks in this conversation
        message_history = await self.storage.load_context(task['context_id']) or []
        message_history.extend(self.build_message_history(task.get('history', [])))

        try:
            result = await self.agent.run(message_history=message_history)  # type: ignore

            await self.storage.update_context(task['context_id'], result.all_messages())

            # Convert new messages to A2A format for task history
            a2a_messages: list[Message] = []

            for message in result.new_messages():
                if isinstance(message, ModelRequest):
                    # Skip user prompts - they're already in task history
                    continue
                else:
                    # Convert response parts to A2A format
                    a2a_parts = self._response_parts_to_a2a(message.parts)
                    if a2a_parts:  # Add if there are visible parts (text/thinking)
                        a2a_messages.append(
                            Message(role='agent', parts=a2a_parts, kind='message', message_id=str(uuid.uuid4()))
                        )

            artifacts = self.build_artifacts(result.output)
        except Exception:
            await self.storage.update_task(task['id'], state='failed')
            raise
        else:
            await self.storage.update_task(
                task['id'], state='completed', new_artifacts=artifacts, new_messages=a2a_messages
            )

    async def cancel_task(self, params: TaskIdParams) -> None:
        pass

    def build_artifacts(self, result: WorkerOutputT) -> list[Artifact]:
        """Build artifacts from agent result.

        All agent outputs become artifacts to mark them as durable task outputs.
        For string results, we use TextPart. For structured data, we use DataPart.
        Metadata is included to preserve type information.
        """
        artifact_id = str(uuid.uuid4())
        part = self._convert_result_to_part(result)
        return [Artifact(artifact_id=artifact_id, name='result', parts=[part])]

    def _convert_result_to_part(self, result: WorkerOutputT) -> Part:
        """Convert agent result to a Part (TextPart or DataPart).

        For string results, returns a TextPart.
        For structured data, returns a DataPart with properly serialized data.
        """
        if isinstance(result, str):
            return A2ATextPart(kind='text', text=result)
        else:
            output_type = type(result)
            type_adapter = TypeAdapter(output_type)
            data = type_adapter.dump_python(result, mode='json')
            json_schema = type_adapter.json_schema(mode='serialization')
            return DataPart(kind='data', data={'result': data}, metadata={'json_schema': json_schema})

    def build_message_history(self, history: list[Message]) -> list[ModelMessage]:
        model_messages: list[ModelMessage] = []
        for message in history:
            if message['role'] == 'user':
                model_messages.append(ModelRequest(parts=self._request_parts_from_a2a(message['parts'])))
            else:
                model_messages.append(ModelResponse(parts=self._response_parts_from_a2a(message['parts'])))
        return model_messages

    def _request_parts_from_a2a(self, parts: list[Part]) -> list[ModelRequestPart]:
        """Convert A2A Part objects to pydantic-ai ModelRequestPart objects.

        This handles the conversion from A2A protocol parts (text, file, data) to
        pydantic-ai's internal request parts (UserPromptPart with various content types).

        Args:
            parts: List of A2A Part objects from incoming messages

        Returns:
            List of ModelRequestPart objects for the pydantic-ai agent
        """
        model_parts: list[ModelRequestPart] = []
        for part in parts:
            if part['kind'] == 'text':
                model_parts.append(UserPromptPart(content=part['text']))
            elif part['kind'] == 'file':
                file_content = part['file']
                if 'bytes' in file_content:
                    data = file_content['bytes'].encode('utf-8')
                    mime_type = file_content.get('mime_type', 'application/octet-stream')
                    content = BinaryContent(data=data, media_type=mime_type)
                    model_parts.append(UserPromptPart(content=[content]))
                else:
                    url = file_content['uri']
                    for url_cls in (DocumentUrl, AudioUrl, ImageUrl, VideoUrl):
                        content = url_cls(url=url)
                        try:
                            content.media_type
                        except ValueError:  # pragma: no cover
                            continue
                        else:
                            break
                    else:
                        raise ValueError(f'Unsupported file type: {url}')  # pragma: no cover
                    model_parts.append(UserPromptPart(content=[content]))
            elif part['kind'] == 'data':
                raise NotImplementedError('Data parts are not supported yet.')
            else:
                assert_never(part)
        return model_parts

    def _response_parts_from_a2a(self, parts: list[Part]) -> list[ModelResponsePart]:
        """Convert A2A Part objects to pydantic-ai ModelResponsePart objects.

        This handles the conversion from A2A protocol parts (text, file, data) to
        pydantic-ai's internal response parts. Currently only supports text parts
        as agent responses in A2A are expected to be text-based.

        Args:
            parts: List of A2A Part objects from stored agent messages

        Returns:
            List of ModelResponsePart objects for message history
        """
        model_parts: list[ModelResponsePart] = []
        for part in parts:
            if part['kind'] == 'text':
                model_parts.append(TextPart(content=part['text']))
            elif part['kind'] == 'file':  # pragma: no cover
                raise NotImplementedError('File parts are not supported yet.')
            elif part['kind'] == 'data':  # pragma: no cover
                raise NotImplementedError('Data parts are not supported yet.')
            else:  # pragma: no cover
                assert_never(part)
        return model_parts

    def _response_parts_to_a2a(self, parts: Sequence[ModelResponsePart]) -> list[Part]:
        """Convert pydantic-ai ModelResponsePart objects to A2A Part objects.

        This handles the conversion from pydantic-ai's internal response parts to
        A2A protocol parts. Different part types are handled as follows:
        - TextPart: Converted directly to A2A TextPart
        - ThinkingPart: Converted to TextPart with metadata indicating it's thinking
        - ToolCallPart: Skipped (internal to agent execution)

        Args:
            parts: List of ModelResponsePart objects from agent response

        Returns:
            List of A2A Part objects suitable for sending via A2A protocol
        """
        a2a_parts: list[Part] = []
        for part in parts:
            if isinstance(part, TextPart):
                a2a_parts.append(A2ATextPart(kind='text', text=part.content))
            elif isinstance(part, ThinkingPart):
                # Convert thinking to text with metadata
                a2a_parts.append(
                    A2ATextPart(
                        kind='text',
                        text=part.content,
                        metadata={'type': 'thinking', 'thinking_id': part.id, 'signature': part.signature},
                    )
                )
            elif isinstance(part, ToolCallPart):
                # Skip tool calls - they're internal to agent execution
                pass
        return a2a_parts



================================================
FILE: pydantic_ai_slim/pydantic_ai/_cli.py
================================================
from __future__ import annotations as _annotations

import argparse
import asyncio
import importlib
import os
import sys
from asyncio import CancelledError
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from typing_inspection.introspection import get_literal_values

from . import __version__
from ._run_context import AgentDepsT
from .agent import AbstractAgent, Agent
from .exceptions import UserError
from .messages import ModelMessage, TextPart
from .models import KnownModelName, infer_model
from .output import OutputDataT

try:
    import argcomplete
    import pyperclip
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import FileHistory
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.live import Live
    from rich.markdown import CodeBlock, Heading, Markdown
    from rich.status import Status
    from rich.style import Style
    from rich.syntax import Syntax
    from rich.text import Text
except ImportError as _import_error:
    raise ImportError(
        'Please install `rich`, `prompt-toolkit`, `pyperclip` and `argcomplete` to use the Pydantic AI CLI, '
        'you can use the `cli` optional group — `pip install "pydantic-ai-slim[cli]"`'
    ) from _import_error


__all__ = 'cli', 'cli_exit'


PYDANTIC_AI_HOME = Path.home() / '.pydantic-ai'
"""The home directory for Pydantic AI CLI.

This folder is used to store the prompt history and configuration.
"""

PROMPT_HISTORY_FILENAME = 'prompt-history.txt'


class SimpleCodeBlock(CodeBlock):
    """Customized code blocks in markdown.

    This avoids a background color which messes up copy-pasting and sets the language name as dim prefix and suffix.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style='dim')
        yield Syntax(code, self.lexer_name, theme=self.theme, background_color='default', word_wrap=True)
        yield Text(f'/{self.lexer_name}', style='dim')


class LeftHeading(Heading):
    """Customized headings in markdown to stop centering and prepend markdown style hashes."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # note we use `Style(bold=True)` not `self.style_name` here to disable underlining which is ugly IMHO
        yield Text(f'{"#" * int(self.tag[1:])} {self.text.plain}', style=Style(bold=True))


Markdown.elements.update(
    fence=SimpleCodeBlock,
    heading_open=LeftHeading,
)


cli_agent = Agent()


@cli_agent.system_prompt
def cli_system_prompt() -> str:
    now_utc = datetime.now(timezone.utc)
    tzinfo = now_utc.astimezone().tzinfo
    tzname = tzinfo.tzname(now_utc) if tzinfo else ''
    return f"""\
Help the user by responding to their request, the output should be concise and always written in markdown.
The current date and time is {datetime.now()} {tzname}.
The user is running {sys.platform}."""


def cli_exit(prog_name: str = 'pai'):  # pragma: no cover
    """Run the CLI and exit."""
    sys.exit(cli(prog_name=prog_name))


def cli(  # noqa: C901
    args_list: Sequence[str] | None = None, *, prog_name: str = 'pai', default_model: str = 'openai:gpt-4.1'
) -> int:
    """Run the CLI and return the exit code for the process."""
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=f"""\
Pydantic AI CLI v{__version__}\n\n

Special prompts:
* `/exit` - exit the interactive mode (ctrl-c and ctrl-d also work)
* `/markdown` - show the last markdown output of the last question
* `/multiline` - toggle multiline mode
* `/cp` - copy the last response to clipboard
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('prompt', nargs='?', help='AI Prompt, if omitted fall into interactive mode')
    arg = parser.add_argument(
        '-m',
        '--model',
        nargs='?',
        help=f'Model to use, in format "<provider>:<model>" e.g. "openai:gpt-4.1" or "anthropic:claude-sonnet-4-0". Defaults to "{default_model}".',
    )
    # we don't want to autocomplete or list models that don't include the provider,
    # e.g. we want to show `openai:gpt-4o` but not `gpt-4o`
    qualified_model_names = [n for n in get_literal_values(KnownModelName.__value__) if ':' in n]
    arg.completer = argcomplete.ChoicesCompleter(qualified_model_names)  # type: ignore[reportPrivateUsage]
    parser.add_argument(
        '-a',
        '--agent',
        help='Custom Agent to use, in format "module:variable", e.g. "mymodule.submodule:my_agent"',
    )
    parser.add_argument(
        '-l',
        '--list-models',
        action='store_true',
        help='List all available models and exit',
    )
    parser.add_argument(
        '-t',
        '--code-theme',
        nargs='?',
        help='Which colors to use for code, can be "dark", "light" or any theme from pygments.org/styles/. Defaults to "dark" which works well on dark terminals.',
        default='dark',
    )
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming from the model')
    parser.add_argument('--version', action='store_true', help='Show version and exit')

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args_list)

    console = Console()
    name_version = f'[green]{prog_name} - Pydantic AI CLI v{__version__}[/green]'
    if args.version:
        console.print(name_version, highlight=False)
        return 0
    if args.list_models:
        console.print(f'{name_version}\n\n[green]Available models:[/green]')
        for model in qualified_model_names:
            console.print(f'  {model}', highlight=False)
        return 0

    agent: Agent[None, str] = cli_agent
    if args.agent:
        sys.path.append(os.getcwd())
        try:
            module_path, variable_name = args.agent.split(':')
        except ValueError:
            console.print('[red]Error: Agent must be specified in "module:variable" format[/red]')
            return 1

        module = importlib.import_module(module_path)
        agent = getattr(module, variable_name)
        if not isinstance(agent, Agent):
            console.print(f'[red]Error: {args.agent} is not an Agent instance[/red]')
            return 1

    model_arg_set = args.model is not None
    if agent.model is None or model_arg_set:
        try:
            agent.model = infer_model(args.model or default_model)
        except UserError as e:
            console.print(f'Error initializing [magenta]{args.model}[/magenta]:\n[red]{e}[/red]')
            return 1

    model_name = agent.model if isinstance(agent.model, str) else f'{agent.model.system}:{agent.model.model_name}'
    if args.agent and model_arg_set:
        console.print(
            f'{name_version} using custom agent [magenta]{args.agent}[/magenta] with [magenta]{model_name}[/magenta]',
            highlight=False,
        )
    elif args.agent:
        console.print(f'{name_version} using custom agent [magenta]{args.agent}[/magenta]', highlight=False)
    else:
        console.print(f'{name_version} with [magenta]{model_name}[/magenta]', highlight=False)

    stream = not args.no_stream
    if args.code_theme == 'light':
        code_theme = 'default'
    elif args.code_theme == 'dark':
        code_theme = 'monokai'
    else:
        code_theme = args.code_theme  # pragma: no cover

    if prompt := cast(str, args.prompt):
        try:
            asyncio.run(ask_agent(agent, prompt, stream, console, code_theme))
        except KeyboardInterrupt:
            pass
        return 0

    try:
        return asyncio.run(run_chat(stream, agent, console, code_theme, prog_name))
    except KeyboardInterrupt:  # pragma: no cover
        return 0


async def run_chat(
    stream: bool,
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    console: Console,
    code_theme: str,
    prog_name: str,
    config_dir: Path | None = None,
    deps: AgentDepsT = None,
    message_history: list[ModelMessage] | None = None,
) -> int:
    prompt_history_path = (config_dir or PYDANTIC_AI_HOME) / PROMPT_HISTORY_FILENAME
    prompt_history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_history_path.touch(exist_ok=True)
    session: PromptSession[Any] = PromptSession(history=FileHistory(str(prompt_history_path)))

    multiline = False
    messages: list[ModelMessage] = message_history[:] if message_history else []

    while True:
        try:
            auto_suggest = CustomAutoSuggest(['/markdown', '/multiline', '/exit', '/cp'])
            text = await session.prompt_async(f'{prog_name} ➤ ', auto_suggest=auto_suggest, multiline=multiline)
        except (KeyboardInterrupt, EOFError):  # pragma: no cover
            return 0

        if not text.strip():
            continue

        ident_prompt = text.lower().strip().replace(' ', '-')
        if ident_prompt.startswith('/'):
            exit_value, multiline = handle_slash_command(ident_prompt, messages, multiline, console, code_theme)
            if exit_value is not None:
                return exit_value
        else:
            try:
                messages = await ask_agent(agent, text, stream, console, code_theme, deps, messages)
            except CancelledError:  # pragma: no cover
                console.print('[dim]Interrupted[/dim]')
            except Exception as e:  # pragma: no cover
                cause = getattr(e, '__cause__', None)
                console.print(f'\n[red]{type(e).__name__}:[/red] {e}')
                if cause:
                    console.print(f'[dim]Caused by: {cause}[/dim]')


async def ask_agent(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    prompt: str,
    stream: bool,
    console: Console,
    code_theme: str,
    deps: AgentDepsT = None,
    messages: list[ModelMessage] | None = None,
) -> list[ModelMessage]:
    status = Status('[dim]Working on it…[/dim]', console=console)

    if not stream:
        with status:
            result = await agent.run(prompt, message_history=messages, deps=deps)
        content = str(result.output)
        console.print(Markdown(content, code_theme=code_theme))
        return result.all_messages()

    with status, ExitStack() as stack:
        async with agent.iter(prompt, message_history=messages, deps=deps) as agent_run:
            live = Live('', refresh_per_second=15, console=console, vertical_overflow='ellipsis')
            async for node in agent_run:
                if Agent.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as handle_stream:
                        status.stop()  # stopping multiple times is idempotent
                        stack.enter_context(live)  # entering multiple times is idempotent

                        async for content in handle_stream.stream_output(debounce_by=None):
                            live.update(Markdown(str(content), code_theme=code_theme))

        assert agent_run.result is not None
        return agent_run.result.all_messages()


class CustomAutoSuggest(AutoSuggestFromHistory):
    def __init__(self, special_suggestions: list[str] | None = None):
        super().__init__()
        self.special_suggestions = special_suggestions or []

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:  # pragma: no cover
        # Get the suggestion from history
        suggestion = super().get_suggestion(buffer, document)

        # Check for custom suggestions
        text = document.text_before_cursor.strip()
        for special in self.special_suggestions:
            if special.startswith(text):
                return Suggestion(special[len(text) :])
        return suggestion


def handle_slash_command(
    ident_prompt: str, messages: list[ModelMessage], multiline: bool, console: Console, code_theme: str
) -> tuple[int | None, bool]:
    if ident_prompt == '/markdown':
        try:
            parts = messages[-1].parts
        except IndexError:
            console.print('[dim]No markdown output available.[/dim]')
        else:
            console.print('[dim]Markdown output of last question:[/dim]\n')
            for part in parts:
                if part.part_kind == 'text':
                    console.print(
                        Syntax(
                            part.content,
                            lexer='markdown',
                            theme=code_theme,
                            word_wrap=True,
                            background_color='default',
                        )
                    )

    elif ident_prompt == '/multiline':
        multiline = not multiline
        if multiline:
            console.print(
                'Enabling multiline mode. [dim]Press [Meta+Enter] or [Esc] followed by [Enter] to accept input.[/dim]'
            )
        else:
            console.print('Disabling multiline mode.')
        return None, multiline
    elif ident_prompt == '/exit':
        console.print('[dim]Exiting…[/dim]')
        return 0, multiline
    elif ident_prompt == '/cp':
        try:
            parts = messages[-1].parts
        except IndexError:
            console.print('[dim]No output available to copy.[/dim]')
        else:
            text_to_copy = '\n\n'.join(part.content for part in parts if isinstance(part, TextPart))
            text_to_copy = text_to_copy.strip()
            if text_to_copy:
                pyperclip.copy(text_to_copy)
                console.print('[dim]Copied last output to clipboard.[/dim]')
            else:
                console.print('[dim]No text content to copy.[/dim]')
    else:
        console.print(f'[red]Unknown command[/red] [magenta]`{ident_prompt}`[/magenta]')
    return None, multiline



================================================
FILE: pydantic_ai_slim/pydantic_ai/_function_schema.py
================================================
"""Used to build pydantic validators and JSON schemas from functions.

This module has to use numerous internal Pydantic APIs and is therefore brittle to changes in Pydantic.
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Concatenate, cast, get_origin

from pydantic import ConfigDict
from pydantic._internal import _decorators, _generate_schema, _typing_extra
from pydantic._internal._config import ConfigWrapper
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic.plugin._schema_validator import create_schema_validator
from pydantic_core import SchemaValidator, core_schema
from typing_extensions import ParamSpec, TypeIs, TypeVar

from ._griffe import doc_descriptions
from ._run_context import RunContext
from ._utils import check_object_json_schema, is_async_callable, is_model_like, run_in_executor

if TYPE_CHECKING:
    from .tools import DocstringFormat, ObjectJsonSchema


__all__ = ('function_schema',)


@dataclass(kw_only=True)
class FunctionSchema:
    """Internal information about a function schema."""

    function: Callable[..., Any]
    description: str | None
    validator: SchemaValidator
    json_schema: ObjectJsonSchema
    # if not None, the function takes a single by that name (besides potentially `info`)
    takes_ctx: bool
    is_async: bool
    single_arg_name: str | None = None
    positional_fields: list[str] = field(default_factory=list)
    var_positional_field: str | None = None

    async def call(self, args_dict: dict[str, Any], ctx: RunContext[Any]) -> Any:
        args, kwargs = self._call_args(args_dict, ctx)
        if self.is_async:
            function = cast(Callable[[Any], Awaitable[str]], self.function)
            return await function(*args, **kwargs)
        else:
            function = cast(Callable[[Any], str], self.function)
            return await run_in_executor(function, *args, **kwargs)

    def _call_args(
        self,
        args_dict: dict[str, Any],
        ctx: RunContext[Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        if self.single_arg_name:
            args_dict = {self.single_arg_name: args_dict}

        args = [ctx] if self.takes_ctx else []
        for positional_field in self.positional_fields:
            args.append(args_dict.pop(positional_field))  # pragma: no cover
        if self.var_positional_field:
            args.extend(args_dict.pop(self.var_positional_field))

        return args, args_dict


def function_schema(  # noqa: C901
    function: Callable[..., Any],
    schema_generator: type[GenerateJsonSchema],
    takes_ctx: bool | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
) -> FunctionSchema:
    """Build a Pydantic validator and JSON schema from a tool function.

    Args:
        function: The function to build a validator and JSON schema for.
        takes_ctx: Whether the function takes a `RunContext` first argument.
        docstring_format: The docstring format to use.
        require_parameter_descriptions: Whether to require descriptions for all tool function parameters.
        schema_generator: The JSON schema generator class to use.

    Returns:
        A `FunctionSchema` instance.
    """
    if takes_ctx is None:
        takes_ctx = _takes_ctx(function)

    config = ConfigDict(title=function.__name__, use_attribute_docstrings=True)
    config_wrapper = ConfigWrapper(config)
    gen_schema = _generate_schema.GenerateSchema(config_wrapper)
    errors: list[str] = []

    try:
        sig = signature(function)
    except ValueError as e:
        errors.append(str(e))
        sig = signature(lambda: None)

    type_hints = _typing_extra.get_function_type_hints(function)

    var_kwargs_schema: core_schema.CoreSchema | None = None
    fields: dict[str, core_schema.TypedDictField] = {}
    positional_fields: list[str] = []
    var_positional_field: str | None = None
    decorators = _decorators.DecoratorInfos()

    description, field_descriptions = doc_descriptions(function, sig, docstring_format=docstring_format)

    if require_parameter_descriptions:
        if takes_ctx:
            parameters_without_ctx = set(
                name for name in sig.parameters if not _is_call_ctx(sig.parameters[name].annotation)
            )
            missing_params = parameters_without_ctx - set(field_descriptions)
        else:
            missing_params = set(sig.parameters) - set(field_descriptions)

        if missing_params:
            errors.append(f'Missing parameter descriptions for {", ".join(missing_params)}')

    for index, (name, p) in enumerate(sig.parameters.items()):
        if p.annotation is sig.empty:
            if takes_ctx and index == 0:
                # should be the `context` argument, skip
                continue
            # TODO warn?
            annotation = Any
        else:
            annotation = type_hints[name]

            if index == 0 and takes_ctx:
                if not _is_call_ctx(annotation):
                    errors.append('First parameter of tools that take context must be annotated with RunContext[...]')
                continue
            elif not takes_ctx and _is_call_ctx(annotation):
                errors.append('RunContext annotations can only be used with tools that take context')
                continue
            elif index != 0 and _is_call_ctx(annotation):
                errors.append('RunContext annotations can only be used as the first argument')
                continue

        field_name = p.name
        if p.kind == Parameter.VAR_KEYWORD:
            var_kwargs_schema = gen_schema.generate_schema(annotation)
        else:
            if p.kind == Parameter.VAR_POSITIONAL:
                annotation = list[annotation]

            required = p.default is Parameter.empty
            # FieldInfo.from_annotated_attribute expects a type, `annotation` is Any
            annotation = cast(type[Any], annotation)
            if required:
                field_info = FieldInfo.from_annotation(annotation)
            else:
                field_info = FieldInfo.from_annotated_attribute(annotation, p.default)
            if field_info.description is None:
                field_info.description = field_descriptions.get(field_name)

            fields[field_name] = td_schema = gen_schema._generate_td_field_schema(  # pyright: ignore[reportPrivateUsage]
                field_name,
                field_info,
                decorators,
                required=required,
            )
            # noinspection PyTypeChecker
            td_schema.setdefault('metadata', {})['is_model_like'] = is_model_like(annotation)

            if p.kind == Parameter.POSITIONAL_ONLY:
                positional_fields.append(field_name)
            elif p.kind == Parameter.VAR_POSITIONAL:
                var_positional_field = field_name

    if errors:
        from .exceptions import UserError

        error_details = '\n  '.join(errors)
        raise UserError(f'Error generating schema for {function.__qualname__}:\n  {error_details}')

    core_config = config_wrapper.core_config(None)
    # noinspection PyTypedDict
    core_config['extra_fields_behavior'] = 'allow' if var_kwargs_schema else 'forbid'

    schema, single_arg_name = _build_schema(fields, var_kwargs_schema, gen_schema, core_config)
    schema = gen_schema.clean_schema(schema)
    # noinspection PyUnresolvedReferences
    schema_validator = create_schema_validator(
        schema,
        function,
        function.__module__,
        function.__qualname__,
        'validate_call',
        core_config,
        config_wrapper.plugin_settings,
    )
    # PluggableSchemaValidator is api compatible with SchemaValidator
    schema_validator = cast(SchemaValidator, schema_validator)
    json_schema = schema_generator().generate(schema)

    # workaround for https://github.com/pydantic/pydantic/issues/10785
    # if we build a custom TypedDict schema (matches when `single_arg_name is None`), we manually set
    # `additionalProperties` in the JSON Schema
    if single_arg_name is not None and not description:
        # if the tool description is not set, and we have a single parameter, take the description from that
        # and set it on the tool
        description = json_schema.pop('description', None)

    return FunctionSchema(
        description=description,
        validator=schema_validator,
        json_schema=check_object_json_schema(json_schema),
        single_arg_name=single_arg_name,
        positional_fields=positional_fields,
        var_positional_field=var_positional_field,
        takes_ctx=takes_ctx,
        is_async=is_async_callable(function),
        function=function,
    )


P = ParamSpec('P')
R = TypeVar('R')


WithCtx = Callable[Concatenate[RunContext[Any], P], R]
WithoutCtx = Callable[P, R]
TargetFunc = WithCtx[P, R] | WithoutCtx[P, R]


def _takes_ctx(function: TargetFunc[P, R]) -> TypeIs[WithCtx[P, R]]:
    """Check if a function takes a `RunContext` first argument.

    Args:
        function: The function to check.

    Returns:
        `True` if the function takes a `RunContext` as first argument, `False` otherwise.
    """
    try:
        sig = signature(function)
    except ValueError:  # pragma: no cover
        return False  # pragma: no cover
    try:
        first_param_name = next(iter(sig.parameters.keys()))
    except StopIteration:
        return False
    else:
        type_hints = _typing_extra.get_function_type_hints(function)
        annotation = type_hints.get(first_param_name)
        if annotation is None:
            return False  # pragma: no cover
        return True is not sig.empty and _is_call_ctx(annotation)


def _build_schema(
    fields: dict[str, core_schema.TypedDictField],
    var_kwargs_schema: core_schema.CoreSchema | None,
    gen_schema: _generate_schema.GenerateSchema,
    core_config: core_schema.CoreConfig,
) -> tuple[core_schema.CoreSchema, str | None]:
    """Generate a typed dict schema for function parameters.

    Args:
        fields: The fields to generate a typed dict schema for.
        var_kwargs_schema: The variable keyword arguments schema.
        gen_schema: The `GenerateSchema` instance.
        core_config: The core configuration.

    Returns:
        tuple of (generated core schema, single arg name).
    """
    if len(fields) == 1 and var_kwargs_schema is None:
        name = next(iter(fields))
        td_field = fields[name]
        if td_field['metadata']['is_model_like']:  # type: ignore
            return td_field['schema'], name

    td_schema = core_schema.typed_dict_schema(
        fields,
        config=core_config,
        extras_schema=gen_schema.generate_schema(var_kwargs_schema) if var_kwargs_schema else None,
    )
    return td_schema, None


def _is_call_ctx(annotation: Any) -> bool:
    """Return whether the annotation is the `RunContext` class, parameterized or not."""
    return annotation is RunContext or get_origin(annotation) is RunContext



================================================
FILE: pydantic_ai_slim/pydantic_ai/_griffe.py
================================================
from __future__ import annotations as _annotations

import logging
import re
from collections.abc import Callable
from contextlib import contextmanager
from inspect import Signature
from typing import TYPE_CHECKING, Any, Literal, cast

from griffe import Docstring, DocstringSectionKind, Object as GriffeObject

if TYPE_CHECKING:
    from .tools import DocstringFormat

DocstringStyle = Literal['google', 'numpy', 'sphinx']


def doc_descriptions(
    func: Callable[..., Any],
    sig: Signature,
    *,
    docstring_format: DocstringFormat,
) -> tuple[str | None, dict[str, str]]:
    """Extract the function description and parameter descriptions from a function's docstring.

    The function parses the docstring using the specified format (or infers it if 'auto')
    and extracts both the main description and parameter descriptions. If a returns section
    is present in the docstring, the main description will be formatted as XML.

    Returns:
        A tuple containing:
        - str: Main description string, which may be either:
            * Plain text if no returns section is present
            * XML-formatted if returns section exists, including <summary> and <returns> tags
        - dict[str, str]: Dictionary mapping parameter names to their descriptions
    """
    doc = func.__doc__
    if doc is None:
        return None, {}

    # see https://github.com/mkdocstrings/griffe/issues/293
    parent = cast(GriffeObject, sig)

    docstring_style = _infer_docstring_style(doc) if docstring_format == 'auto' else docstring_format
    docstring = Docstring(
        doc,
        lineno=1,
        parser=docstring_style,
        parent=parent,
        # https://mkdocstrings.github.io/griffe/reference/docstrings/#google-options
        parser_options={'returns_named_value': False, 'returns_multiple_items': False},
    )
    with _disable_griffe_logging():
        sections = docstring.parse()

    params = {}
    if parameters := next((p for p in sections if p.kind == DocstringSectionKind.parameters), None):
        params = {p.name: p.description for p in parameters.value}

    main_desc = ''
    if main := next((p for p in sections if p.kind == DocstringSectionKind.text), None):
        main_desc = main.value

    if return_ := next((p for p in sections if p.kind == DocstringSectionKind.returns), None):
        return_statement = return_.value[0]
        return_desc = return_statement.description
        return_type = return_statement.annotation
        type_tag = f'<type>{return_type}</type>\n' if return_type else ''
        return_xml = f'<returns>\n{type_tag}<description>{return_desc}</description>\n</returns>'

        if main_desc:
            main_desc = f'<summary>{main_desc}</summary>\n{return_xml}'
        else:
            main_desc = return_xml

    return main_desc, params


def _infer_docstring_style(doc: str) -> DocstringStyle:
    """Simplistic docstring style inference."""
    for pattern, replacements, style in _docstring_style_patterns:
        matches = (
            re.search(pattern.format(replacement), doc, re.IGNORECASE | re.MULTILINE) for replacement in replacements
        )
        if any(matches):
            return style
    # fallback to google style
    return 'google'


# See https://github.com/mkdocstrings/griffe/issues/329#issuecomment-2425017804
_docstring_style_patterns: list[tuple[str, list[str], DocstringStyle]] = [
    (
        r'\n[ \t]*:{0}([ \t]+\w+)*:([ \t]+.+)?\n',
        [
            'param',
            'parameter',
            'arg',
            'argument',
            'key',
            'keyword',
            'type',
            'var',
            'ivar',
            'cvar',
            'vartype',
            'returns',
            'return',
            'rtype',
            'raises',
            'raise',
            'except',
            'exception',
        ],
        'sphinx',
    ),
    (
        r'\n[ \t]*{0}:([ \t]+.+)?\n[ \t]+.+',
        [
            'args',
            'arguments',
            'params',
            'parameters',
            'keyword args',
            'keyword arguments',
            'other args',
            'other arguments',
            'other params',
            'other parameters',
            'raises',
            'exceptions',
            'returns',
            'yields',
            'receives',
            'examples',
            'attributes',
            'functions',
            'methods',
            'classes',
            'modules',
            'warns',
            'warnings',
        ],
        'google',
    ),
    (
        r'\n[ \t]*{0}\n[ \t]*---+\n',
        [
            'deprecated',
            'parameters',
            'other parameters',
            'returns',
            'yields',
            'receives',
            'raises',
            'warns',
            'attributes',
            'functions',
            'methods',
            'classes',
            'modules',
        ],
        'numpy',
    ),
]


@contextmanager
def _disable_griffe_logging():
    # Hacky, but suggested here: https://github.com/mkdocstrings/griffe/issues/293#issuecomment-2167668117
    old_level = logging.root.getEffectiveLevel()
    logging.root.setLevel(logging.ERROR)
    yield
    logging.root.setLevel(old_level)



================================================
FILE: pydantic_ai_slim/pydantic_ai/_mcp.py
================================================
import base64
from collections.abc import Sequence
from typing import Literal

from . import exceptions, messages

try:
    from mcp import types as mcp_types
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use the MCP server, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error


def map_from_mcp_params(params: mcp_types.CreateMessageRequestParams) -> list[messages.ModelMessage]:
    """Convert from MCP create message request parameters to pydantic-ai messages."""
    pai_messages: list[messages.ModelMessage] = []
    request_parts: list[messages.ModelRequestPart] = []
    if params.systemPrompt:
        request_parts.append(messages.SystemPromptPart(content=params.systemPrompt))
    response_parts: list[messages.ModelResponsePart] = []
    for msg in params.messages:
        content = msg.content
        if msg.role == 'user':
            # if there are any response parts, add a response message wrapping them
            if response_parts:
                pai_messages.append(messages.ModelResponse(parts=response_parts))
                response_parts = []

            # TODO(Marcelo): We can reuse the `_map_tool_result_part` from the mcp module here.
            if isinstance(content, mcp_types.TextContent):
                user_part_content: str | Sequence[messages.UserContent] = content.text
            else:
                # image content
                user_part_content = [
                    messages.BinaryContent(data=base64.b64decode(content.data), media_type=content.mimeType)
                ]

            request_parts.append(messages.UserPromptPart(content=user_part_content))
        else:
            # role is assistant
            # if there are any request parts, add a request message wrapping them
            if request_parts:
                pai_messages.append(messages.ModelRequest(parts=request_parts))
                request_parts = []

            response_parts.append(map_from_sampling_content(content))

    if response_parts:
        pai_messages.append(messages.ModelResponse(parts=response_parts))
    if request_parts:
        pai_messages.append(messages.ModelRequest(parts=request_parts))
    return pai_messages


def map_from_pai_messages(pai_messages: list[messages.ModelMessage]) -> tuple[str, list[mcp_types.SamplingMessage]]:
    """Convert from pydantic-ai messages to MCP sampling messages.

    Returns:
        A tuple containing the system prompt and a list of sampling messages.
    """
    sampling_msgs: list[mcp_types.SamplingMessage] = []

    def add_msg(
        role: Literal['user', 'assistant'],
        content: mcp_types.TextContent | mcp_types.ImageContent | mcp_types.AudioContent,
    ):
        sampling_msgs.append(mcp_types.SamplingMessage(role=role, content=content))

    system_prompt: list[str] = []
    for pai_message in pai_messages:
        if isinstance(pai_message, messages.ModelRequest):
            if pai_message.instructions is not None:
                system_prompt.append(pai_message.instructions)

            for part in pai_message.parts:
                if isinstance(part, messages.SystemPromptPart):
                    system_prompt.append(part.content)
                if isinstance(part, messages.UserPromptPart):
                    if isinstance(part.content, str):
                        add_msg('user', mcp_types.TextContent(type='text', text=part.content))
                    else:
                        for chunk in part.content:
                            if isinstance(chunk, str):
                                add_msg('user', mcp_types.TextContent(type='text', text=chunk))
                            elif isinstance(chunk, messages.BinaryContent) and chunk.is_image:
                                add_msg(
                                    'user',
                                    mcp_types.ImageContent(
                                        type='image',
                                        data=base64.b64decode(chunk.data).decode(),
                                        mimeType=chunk.media_type,
                                    ),
                                )
                            # TODO(Marcelo): Add support for audio content.
                            else:
                                raise NotImplementedError(f'Unsupported content type: {type(chunk)}')
        else:
            add_msg('assistant', map_from_model_response(pai_message))
    return ''.join(system_prompt), sampling_msgs


def map_from_model_response(model_response: messages.ModelResponse) -> mcp_types.TextContent:
    """Convert from a model response to MCP text content."""
    text_parts: list[str] = []
    for part in model_response.parts:
        if isinstance(part, messages.TextPart):
            text_parts.append(part.content)
        # TODO(Marcelo): We should ignore ThinkingPart here.
        else:
            raise exceptions.UnexpectedModelBehavior(f'Unexpected part type: {type(part).__name__}, expected TextPart')
    return mcp_types.TextContent(type='text', text=''.join(text_parts))


def map_from_sampling_content(
    content: mcp_types.TextContent | mcp_types.ImageContent | mcp_types.AudioContent,
) -> messages.TextPart:
    """Convert from sampling content to a pydantic-ai text part."""
    if isinstance(content, mcp_types.TextContent):  # pragma: no branch
        return messages.TextPart(content=content.text)
    else:
        raise NotImplementedError('Image and Audio responses in sampling are not yet supported')



================================================
FILE: pydantic_ai_slim/pydantic_ai/_otel_messages.py
================================================
"""Type definitions of OpenTelemetry GenAI spec message parts.

Based on https://github.com/lmolkova/semantic-conventions/blob/eccd1f806e426a32c98271c3ce77585492d26de2/docs/gen-ai/non-normative/models.ipynb
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import JsonValue
from typing_extensions import NotRequired, TypedDict


class TextPart(TypedDict):
    type: Literal['text']
    content: NotRequired[str]


class ToolCallPart(TypedDict):
    type: Literal['tool_call']
    id: str
    name: str
    arguments: NotRequired[JsonValue]


class ToolCallResponsePart(TypedDict):
    type: Literal['tool_call_response']
    id: str
    name: str
    result: NotRequired[JsonValue]


class MediaUrlPart(TypedDict):
    type: Literal['image-url', 'audio-url', 'video-url', 'document-url']
    url: NotRequired[str]


class BinaryDataPart(TypedDict):
    type: Literal['binary']
    media_type: str
    content: NotRequired[str]


class ThinkingPart(TypedDict):
    type: Literal['thinking']
    content: NotRequired[str]


MessagePart: TypeAlias = 'TextPart | ToolCallPart | ToolCallResponsePart | MediaUrlPart | BinaryDataPart | ThinkingPart'


Role = Literal['system', 'user', 'assistant']


class ChatMessage(TypedDict):
    role: Role
    parts: list[MessagePart]


InputMessages: TypeAlias = list[ChatMessage]


class OutputMessage(ChatMessage):
    finish_reason: NotRequired[str]


OutputMessages: TypeAlias = list[OutputMessage]



================================================
FILE: pydantic_ai_slim/pydantic_ai/_output.py
================================================
from __future__ import annotations as _annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, overload

from pydantic import Json, TypeAdapter, ValidationError
from pydantic_core import SchemaValidator, to_json
from typing_extensions import Self, TypedDict, TypeVar, assert_never

from . import _function_schema, _utils, messages as _messages
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, ToolRetryError, UserError
from .output import (
    DeferredToolRequests,
    NativeOutput,
    OutputDataT,
    OutputMode,
    OutputSpec,
    OutputTypeOrFunction,
    PromptedOutput,
    StructuredOutputMode,
    TextOutput,
    TextOutputFunc,
    ToolOutput,
    _OutputSpecItem,  # type: ignore[reportPrivateUsage]
)
from .tools import GenerateToolJsonSchema, ObjectJsonSchema, ToolDefinition
from .toolsets.abstract import AbstractToolset, ToolsetTool

if TYPE_CHECKING:
    from .profiles import ModelProfile

T = TypeVar('T')
"""An invariant TypeVar."""
OutputDataT_inv = TypeVar('OutputDataT_inv', default=str)
"""
An invariant type variable for the result data of a model.

We need to use an invariant typevar for `OutputValidator` and `OutputValidatorFunc` because the output data type is used
in both the input and output of a `OutputValidatorFunc`. This can theoretically lead to some issues assuming that types
possessing OutputValidator's are covariant in the result data type, but in practice this is rarely an issue, and
changing it would have negative consequences for the ergonomics of the library.

At some point, it may make sense to change the input to OutputValidatorFunc to be `Any` or `object` as doing that would
resolve these potential variance issues.
"""

OutputValidatorFunc = (
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], OutputDataT_inv]
    | Callable[[RunContext[AgentDepsT], OutputDataT_inv], Awaitable[OutputDataT_inv]]
    | Callable[[OutputDataT_inv], OutputDataT_inv]
    | Callable[[OutputDataT_inv], Awaitable[OutputDataT_inv]]
)
"""
A function that always takes and returns the same type of data (which is the result type of an agent run), and:

* may or may not take [`RunContext`][pydantic_ai.tools.RunContext] as a first argument
* may or may not be async

Usage `OutputValidatorFunc[AgentDepsT, T]`.
"""


DEFAULT_OUTPUT_TOOL_NAME = 'final_result'
DEFAULT_OUTPUT_TOOL_DESCRIPTION = 'The final response which ends this conversation'


async def execute_traced_output_function(
    function_schema: _function_schema.FunctionSchema,
    run_context: RunContext[AgentDepsT],
    args: dict[str, Any] | Any,
    wrap_validation_errors: bool = True,
) -> Any:
    """Execute an output function within a traced span with error handling.

    This function executes the output function within an OpenTelemetry span for observability,
    automatically records the function response, and handles ModelRetry exceptions by converting
    them to ToolRetryError when wrap_validation_errors is True.

    Args:
        function_schema: The function schema containing the function to execute
        run_context: The current run context containing tracing and tool information
        args: Arguments to pass to the function
        wrap_validation_errors: If True, wrap ModelRetry exceptions in ToolRetryError

    Returns:
        The result of the function execution

    Raises:
        ToolRetryError: When wrap_validation_errors is True and a ModelRetry is caught
        ModelRetry: When wrap_validation_errors is False and a ModelRetry occurs
    """
    # Set up span attributes
    tool_name = run_context.tool_name or getattr(function_schema.function, '__name__', 'output_function')
    attributes = {
        'gen_ai.tool.name': tool_name,
        'logfire.msg': f'running output function: {tool_name}',
    }
    if run_context.tool_call_id:
        attributes['gen_ai.tool.call.id'] = run_context.tool_call_id
    if run_context.trace_include_content:
        attributes['tool_arguments'] = to_json(args).decode()
        attributes['logfire.json_schema'] = json.dumps(
            {
                'type': 'object',
                'properties': {
                    'tool_arguments': {'type': 'object'},
                    'tool_response': {'type': 'object'},
                },
            }
        )

    with run_context.tracer.start_as_current_span('running output function', attributes=attributes) as span:
        try:
            output = await function_schema.call(args, run_context)
        except ModelRetry as r:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=r.message,
                    tool_name=run_context.tool_name,
                )
                if run_context.tool_call_id:
                    m.tool_call_id = run_context.tool_call_id  # pragma: no cover
                raise ToolRetryError(m) from r
            else:
                raise

        # Record response if content inclusion is enabled
        if run_context.trace_include_content and span.is_recording():
            from .models.instrumented import InstrumentedModel

            span.set_attribute(
                'tool_response',
                output if isinstance(output, str) else json.dumps(InstrumentedModel.serialize_any(output)),
            )

        return output


@dataclass
class OutputValidator(Generic[AgentDepsT, OutputDataT_inv]):
    function: OutputValidatorFunc[AgentDepsT, OutputDataT_inv]
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 1
        self._is_async = _utils.is_async_callable(self.function)

    async def validate(
        self,
        result: T,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> T:
        """Validate a result but calling the function.

        Args:
            result: The result data after Pydantic validation the message content.
            run_context: The current run context.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Result of either the validated result data (ok) or a retry message (Err).
        """
        if self._takes_ctx:
            args = run_context, result
        else:
            args = (result,)

        try:
            if self._is_async:
                function = cast(Callable[[Any], Awaitable[T]], self.function)
                result_data = await function(*args)
            else:
                function = cast(Callable[[Any], T], self.function)
                result_data = await _utils.run_in_executor(function, *args)
        except ModelRetry as r:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=r.message,
                    tool_name=run_context.tool_name,
                )
                if run_context.tool_call_id:  # pragma: no cover
                    m.tool_call_id = run_context.tool_call_id
                raise ToolRetryError(m) from r
            else:
                raise r
        else:
            return result_data


@dataclass
class BaseOutputSchema(ABC, Generic[OutputDataT]):
    allows_deferred_tools: bool

    @abstractmethod
    def with_default_mode(self, mode: StructuredOutputMode) -> OutputSchema[OutputDataT]:
        raise NotImplementedError()

    @property
    def toolset(self) -> OutputToolset[Any] | None:
        """Get the toolset for this output schema."""
        return None


@dataclass(init=False)
class OutputSchema(BaseOutputSchema[OutputDataT], ABC):
    """Model the final output from an agent run."""

    @classmethod
    @overload
    def build(
        cls,
        output_spec: OutputSpec[OutputDataT],
        *,
        default_mode: StructuredOutputMode,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> OutputSchema[OutputDataT]: ...

    @classmethod
    @overload
    def build(
        cls,
        output_spec: OutputSpec[OutputDataT],
        *,
        default_mode: None = None,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> BaseOutputSchema[OutputDataT]: ...

    @classmethod
    def build(  # noqa: C901
        cls,
        output_spec: OutputSpec[OutputDataT],
        *,
        default_mode: StructuredOutputMode | None = None,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> BaseOutputSchema[OutputDataT]:
        """Build an OutputSchema dataclass from an output type."""
        raw_outputs = _flatten_output_spec(output_spec)

        outputs = [output for output in raw_outputs if output is not DeferredToolRequests]
        allows_deferred_tools = len(outputs) < len(raw_outputs)
        if len(outputs) == 0 and allows_deferred_tools:
            raise UserError('At least one output type must be provided other than `DeferredToolRequests`.')

        if output := next((output for output in outputs if isinstance(output, NativeOutput)), None):
            if len(outputs) > 1:
                raise UserError('`NativeOutput` must be the only output type.')  # pragma: no cover

            return NativeOutputSchema(
                processor=cls._build_processor(
                    _flatten_output_spec(output.outputs),
                    name=output.name,
                    description=output.description,
                    strict=output.strict,
                ),
                allows_deferred_tools=allows_deferred_tools,
            )
        elif output := next((output for output in outputs if isinstance(output, PromptedOutput)), None):
            if len(outputs) > 1:
                raise UserError('`PromptedOutput` must be the only output type.')  # pragma: no cover

            return PromptedOutputSchema(
                processor=cls._build_processor(
                    _flatten_output_spec(output.outputs),
                    name=output.name,
                    description=output.description,
                ),
                template=output.template,
                allows_deferred_tools=allows_deferred_tools,
            )

        text_outputs: Sequence[type[str] | TextOutput[OutputDataT]] = []
        tool_outputs: Sequence[ToolOutput[OutputDataT]] = []
        other_outputs: Sequence[OutputTypeOrFunction[OutputDataT]] = []
        for output in outputs:
            if output is str:
                text_outputs.append(cast(type[str], output))
            elif isinstance(output, TextOutput):
                text_outputs.append(output)
            elif isinstance(output, ToolOutput):
                tool_outputs.append(output)
            elif isinstance(output, NativeOutput):
                # We can never get here because this is checked for above.
                raise UserError('`NativeOutput` must be the only output type.')  # pragma: no cover
            elif isinstance(output, PromptedOutput):
                # We can never get here because this is checked for above.
                raise UserError('`PromptedOutput` must be the only output type.')  # pragma: no cover
            else:
                other_outputs.append(output)

        toolset = OutputToolset.build(tool_outputs + other_outputs, name=name, description=description, strict=strict)

        if len(text_outputs) > 0:
            if len(text_outputs) > 1:
                raise UserError('Only one `str` or `TextOutput` is allowed.')
            text_output = text_outputs[0]

            text_output_schema = None
            if isinstance(text_output, TextOutput):
                text_output_schema = PlainTextOutputProcessor(text_output.output_function)

            if toolset:
                return ToolOrTextOutputSchema(
                    processor=text_output_schema,
                    toolset=toolset,
                    allows_deferred_tools=allows_deferred_tools,
                )
            else:
                return PlainTextOutputSchema(processor=text_output_schema, allows_deferred_tools=allows_deferred_tools)

        if len(tool_outputs) > 0:
            return ToolOutputSchema(toolset=toolset, allows_deferred_tools=allows_deferred_tools)

        if len(other_outputs) > 0:
            schema = OutputSchemaWithoutMode(
                processor=cls._build_processor(other_outputs, name=name, description=description, strict=strict),
                toolset=toolset,
                allows_deferred_tools=allows_deferred_tools,
            )
            if default_mode:
                schema = schema.with_default_mode(default_mode)
            return schema

        raise UserError('At least one output type must be provided.')

    @staticmethod
    def _build_processor(
        outputs: Sequence[OutputTypeOrFunction[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> ObjectOutputProcessor[OutputDataT] | UnionOutputProcessor[OutputDataT]:
        outputs = _flatten_output_spec(outputs)
        if len(outputs) == 1:
            return ObjectOutputProcessor(output=outputs[0], name=name, description=description, strict=strict)

        return UnionOutputProcessor(outputs=outputs, strict=strict, name=name, description=description)

    @property
    @abstractmethod
    def mode(self) -> OutputMode:
        raise NotImplementedError()

    @abstractmethod
    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        raise NotImplementedError()

    def with_default_mode(self, mode: StructuredOutputMode) -> OutputSchema[OutputDataT]:
        return self


@dataclass(init=False)
class OutputSchemaWithoutMode(BaseOutputSchema[OutputDataT]):
    processor: ObjectOutputProcessor[OutputDataT] | UnionOutputProcessor[OutputDataT]
    _toolset: OutputToolset[Any] | None

    def __init__(
        self,
        processor: ObjectOutputProcessor[OutputDataT] | UnionOutputProcessor[OutputDataT],
        toolset: OutputToolset[Any] | None,
        allows_deferred_tools: bool,
    ):
        super().__init__(allows_deferred_tools)
        self.processor = processor
        self._toolset = toolset

    def with_default_mode(self, mode: StructuredOutputMode) -> OutputSchema[OutputDataT]:
        if mode == 'native':
            return NativeOutputSchema(processor=self.processor, allows_deferred_tools=self.allows_deferred_tools)
        elif mode == 'prompted':
            return PromptedOutputSchema(processor=self.processor, allows_deferred_tools=self.allows_deferred_tools)
        elif mode == 'tool':
            return ToolOutputSchema(toolset=self.toolset, allows_deferred_tools=self.allows_deferred_tools)
        else:
            assert_never(mode)

    @property
    def toolset(self) -> OutputToolset[Any] | None:
        """Get the toolset for this output schema."""
        # We return a toolset here as they're checked for name conflicts with other toolsets in the Agent constructor.
        # At that point we may not know yet what output mode we're going to use if no model was provided or it was deferred until agent.run time,
        # but we cover ourselves just in case we end up using the tool output mode.
        return self._toolset


class TextOutputSchema(OutputSchema[OutputDataT], ABC):
    @abstractmethod
    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        raise NotImplementedError()


@dataclass
class PlainTextOutputSchema(TextOutputSchema[OutputDataT]):
    processor: PlainTextOutputProcessor[OutputDataT] | None = None

    @property
    def mode(self) -> OutputMode:
        return 'text'

    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        pass

    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            text: The output text to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        if self.processor is None:
            return cast(OutputDataT, text)

        return await self.processor.process(
            text, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass
class StructuredTextOutputSchema(TextOutputSchema[OutputDataT], ABC):
    processor: ObjectOutputProcessor[OutputDataT] | UnionOutputProcessor[OutputDataT]

    @property
    def object_def(self) -> OutputObjectDefinition:
        return self.processor.object_def


@dataclass
class NativeOutputSchema(StructuredTextOutputSchema[OutputDataT]):
    @property
    def mode(self) -> OutputMode:
        return 'native'

    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        if not profile.supports_json_schema_output:
            raise UserError('Native structured output is not supported by the model.')

    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            text: The output text to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        return await self.processor.process(
            text, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass
class PromptedOutputSchema(StructuredTextOutputSchema[OutputDataT]):
    template: str | None = None

    @property
    def mode(self) -> OutputMode:
        return 'prompted'

    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        pass

    def instructions(self, default_template: str) -> str:
        """Get instructions to tell model to output JSON matching the schema."""
        template = self.template or default_template

        if '{schema}' not in template:
            template = '\n\n'.join([template, '{schema}'])

        object_def = self.object_def
        schema = object_def.json_schema.copy()
        if object_def.name:
            schema['title'] = object_def.name
        if object_def.description:
            schema['description'] = object_def.description

        return template.format(schema=json.dumps(schema))

    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            text: The output text to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        text = _utils.strip_markdown_fences(text)

        return await self.processor.process(
            text, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass(init=False)
class ToolOutputSchema(OutputSchema[OutputDataT]):
    _toolset: OutputToolset[Any] | None

    def __init__(self, toolset: OutputToolset[Any] | None, allows_deferred_tools: bool):
        super().__init__(allows_deferred_tools)
        self._toolset = toolset

    @property
    def mode(self) -> OutputMode:
        return 'tool'

    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        if not profile.supports_tools:
            raise UserError('Output tools are not supported by the model.')

    @property
    def toolset(self) -> OutputToolset[Any] | None:
        """Get the toolset for this output schema."""
        return self._toolset


@dataclass(init=False)
class ToolOrTextOutputSchema(ToolOutputSchema[OutputDataT], PlainTextOutputSchema[OutputDataT]):
    def __init__(
        self,
        processor: PlainTextOutputProcessor[OutputDataT] | None,
        toolset: OutputToolset[Any] | None,
        allows_deferred_tools: bool,
    ):
        super().__init__(toolset=toolset, allows_deferred_tools=allows_deferred_tools)
        self.processor = processor

    @property
    def mode(self) -> OutputMode:
        return 'tool_or_text'


@dataclass
class OutputObjectDefinition:
    json_schema: ObjectJsonSchema
    name: str | None = None
    description: str | None = None
    strict: bool | None = None


@dataclass(init=False)
class BaseOutputProcessor(ABC, Generic[OutputDataT]):
    @abstractmethod
    async def process(
        self,
        data: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message, performing validation and (if necessary) calling the output function."""
        raise NotImplementedError()


@dataclass(init=False)
class ObjectOutputProcessor(BaseOutputProcessor[OutputDataT]):
    object_def: OutputObjectDefinition
    outer_typed_dict_key: str | None = None
    validator: SchemaValidator
    _function_schema: _function_schema.FunctionSchema | None = None

    def __init__(
        self,
        output: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        if inspect.isfunction(output) or inspect.ismethod(output):
            self._function_schema = _function_schema.function_schema(output, GenerateToolJsonSchema)
            self.validator = self._function_schema.validator
            json_schema = self._function_schema.json_schema
            json_schema['description'] = self._function_schema.description
        else:
            json_schema_type_adapter: TypeAdapter[Any]
            validation_type_adapter: TypeAdapter[Any]
            if _utils.is_model_like(output):
                json_schema_type_adapter = validation_type_adapter = TypeAdapter(output)
            else:
                self.outer_typed_dict_key = 'response'
                output_type: type[OutputDataT] = cast(type[OutputDataT], output)

                response_data_typed_dict = TypedDict(  # noqa: UP013
                    'response_data_typed_dict',
                    {'response': output_type},  # pyright: ignore[reportInvalidTypeForm]
                )
                json_schema_type_adapter = TypeAdapter(response_data_typed_dict)

                # More lenient validator: allow either the native type or a JSON string containing it
                # i.e. `response: OutputDataT | Json[OutputDataT]`, as some models don't follow the schema correctly,
                # e.g. `BedrockConverseModel('us.meta.llama3-2-11b-instruct-v1:0')`
                response_validation_typed_dict = TypedDict(  # noqa: UP013
                    'response_validation_typed_dict',
                    {'response': output_type | Json[output_type]},  # pyright: ignore[reportInvalidTypeForm]
                )
                validation_type_adapter = TypeAdapter(response_validation_typed_dict)

            # Really a PluggableSchemaValidator, but it's API-compatible
            self.validator = cast(SchemaValidator, validation_type_adapter.validator)
            json_schema = _utils.check_object_json_schema(
                json_schema_type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
            )

            if self.outer_typed_dict_key:
                # including `response_data_typed_dict` as a title here doesn't add anything and could confuse the LLM
                json_schema.pop('title')

        if name is None and (json_schema_title := json_schema.get('title', None)):
            name = json_schema_title

        if json_schema_description := json_schema.pop('description', None):
            if description is None:
                description = json_schema_description
            else:
                description = f'{description}. {json_schema_description}'

        self.object_def = OutputObjectDefinition(
            name=name or getattr(output, '__name__', None),
            description=description,
            json_schema=json_schema,
            strict=strict,
        )

    async def process(
        self,
        data: str | dict[str, Any] | None,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message, performing validation and (if necessary) calling the output function.

        Args:
            data: The output data to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            output = self.validate(data, allow_partial)
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=e.errors(include_url=False),
                )
                raise ToolRetryError(m) from e
            else:
                raise

        output = await self.call(output, run_context, wrap_validation_errors)

        return output

    def validate(
        self,
        data: str | dict[str, Any] | None,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
        if isinstance(data, str):
            return self.validator.validate_json(data or '{}', allow_partial=pyd_allow_partial)
        else:
            return self.validator.validate_python(data or {}, allow_partial=pyd_allow_partial)

    async def call(
        self,
        output: Any,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ):
        if k := self.outer_typed_dict_key:
            output = output[k]

        if self._function_schema:
            output = await execute_traced_output_function(
                self._function_schema, run_context, output, wrap_validation_errors
            )

        return output


@dataclass
class UnionOutputResult:
    kind: str
    data: ObjectJsonSchema


@dataclass
class UnionOutputModel:
    result: UnionOutputResult


@dataclass(init=False)
class UnionOutputProcessor(BaseOutputProcessor[OutputDataT]):
    object_def: OutputObjectDefinition
    _union_processor: ObjectOutputProcessor[UnionOutputModel]
    _processors: dict[str, ObjectOutputProcessor[OutputDataT]]

    def __init__(
        self,
        outputs: Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        self._union_processor = ObjectOutputProcessor(output=UnionOutputModel)

        json_schemas: list[ObjectJsonSchema] = []
        self._processors = {}
        for output in outputs:
            processor = ObjectOutputProcessor(output=output, strict=strict)
            object_def = processor.object_def

            object_key = object_def.name or output.__name__
            i = 1
            original_key = object_key
            while object_key in self._processors:
                i += 1
                object_key = f'{original_key}_{i}'

            self._processors[object_key] = processor

            json_schema = object_def.json_schema
            if object_def.name:  # pragma: no branch
                json_schema['title'] = object_def.name
            if object_def.description:
                json_schema['description'] = object_def.description

            json_schemas.append(json_schema)

        json_schemas, all_defs = _utils.merge_json_schema_defs(json_schemas)

        discriminated_json_schemas: list[ObjectJsonSchema] = []
        for object_key, json_schema in zip(self._processors.keys(), json_schemas):
            title = json_schema.pop('title', None)
            description = json_schema.pop('description', None)

            discriminated_json_schema = {
                'type': 'object',
                'properties': {
                    'kind': {
                        'type': 'string',
                        'const': object_key,
                    },
                    'data': json_schema,
                },
                'required': ['kind', 'data'],
                'additionalProperties': False,
            }
            if title:  # pragma: no branch
                discriminated_json_schema['title'] = title
            if description:
                discriminated_json_schema['description'] = description

            discriminated_json_schemas.append(discriminated_json_schema)

        json_schema = {
            'type': 'object',
            'properties': {
                'result': {
                    'anyOf': discriminated_json_schemas,
                }
            },
            'required': ['result'],
            'additionalProperties': False,
        }
        if all_defs:
            json_schema['$defs'] = all_defs

        self.object_def = OutputObjectDefinition(
            json_schema=json_schema,
            strict=strict,
            name=name,
            description=description,
        )

    async def process(
        self,
        data: str | dict[str, Any] | None,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        union_object = await self._union_processor.process(
            data, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )

        result = union_object.result
        kind = result.kind
        data = result.data
        try:
            processor = self._processors[kind]
        except KeyError as e:  # pragma: no cover
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(content=f'Invalid kind: {kind}')
                raise ToolRetryError(m) from e
            else:
                raise

        return await processor.process(
            data, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass(init=False)
class PlainTextOutputProcessor(BaseOutputProcessor[OutputDataT]):
    _function_schema: _function_schema.FunctionSchema
    _str_argument_name: str

    def __init__(
        self,
        output_function: TextOutputFunc[OutputDataT],
    ):
        self._function_schema = _function_schema.function_schema(output_function, GenerateToolJsonSchema)

        arguments_schema = self._function_schema.json_schema.get('properties', {})
        argument_name = next(iter(arguments_schema.keys()), None)
        if argument_name and arguments_schema.get(argument_name, {}).get('type') == 'string':
            self._str_argument_name = argument_name
            return

        raise UserError('TextOutput must take a function taking a `str`')

    @property
    def object_def(self) -> None:
        return None  # pragma: no cover

    async def process(
        self,
        data: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        args = {self._str_argument_name: data}
        output = await execute_traced_output_function(self._function_schema, run_context, args, wrap_validation_errors)

        return cast(OutputDataT, output)


@dataclass(init=False)
class OutputToolset(AbstractToolset[AgentDepsT]):
    """A toolset that contains contains output tools for agent output types."""

    _tool_defs: list[ToolDefinition]
    """The tool definitions for the output tools in this toolset."""
    processors: dict[str, ObjectOutputProcessor[Any]]
    """The processors for the output tools in this toolset."""
    max_retries: int
    output_validators: list[OutputValidator[AgentDepsT, Any]]

    @classmethod
    def build(
        cls,
        outputs: list[OutputTypeOrFunction[OutputDataT] | ToolOutput[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> Self | None:
        if len(outputs) == 0:
            return None

        processors: dict[str, ObjectOutputProcessor[Any]] = {}
        tool_defs: list[ToolDefinition] = []

        default_name = name or DEFAULT_OUTPUT_TOOL_NAME
        default_description = description
        default_strict = strict

        multiple = len(outputs) > 1
        for output in outputs:
            name = None
            description = None
            strict = None
            if isinstance(output, ToolOutput):
                # do we need to error on conflicts here? (DavidM): If this is internal maybe doesn't matter, if public, use overloads
                name = output.name
                description = output.description
                strict = output.strict

                output = output.output

            description = description or default_description
            if strict is None:
                strict = default_strict

            processor = ObjectOutputProcessor(output=output, description=description, strict=strict)
            object_def = processor.object_def

            if name is None:
                name = default_name
                if multiple:
                    name += f'_{object_def.name}'

            i = 1
            original_name = name
            while name in processors:
                i += 1
                name = f'{original_name}_{i}'

            description = object_def.description
            if not description:
                description = DEFAULT_OUTPUT_TOOL_DESCRIPTION
                if multiple:
                    description = f'{object_def.name}: {description}'

            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters_json_schema=object_def.json_schema,
                strict=object_def.strict,
                outer_typed_dict_key=processor.outer_typed_dict_key,
                kind='output',
            )
            processors[name] = processor
            tool_defs.append(tool_def)

        return cls(processors=processors, tool_defs=tool_defs)

    def __init__(
        self,
        tool_defs: list[ToolDefinition],
        processors: dict[str, ObjectOutputProcessor[Any]],
        max_retries: int = 1,
        output_validators: list[OutputValidator[AgentDepsT, Any]] | None = None,
    ):
        self.processors = processors
        self._tool_defs = tool_defs
        self.max_retries = max_retries
        self.output_validators = output_validators or []

    @property
    def id(self) -> str | None:
        return '<output>'  # pragma: no cover

    @property
    def label(self) -> str:
        return "the agent's output tools"

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self.max_retries,
                args_validator=self.processors[tool_def.name].validator,
            )
            for tool_def in self._tool_defs
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        output = await self.processors[name].call(tool_args, ctx, wrap_validation_errors=False)
        for validator in self.output_validators:
            output = await validator.validate(output, ctx, wrap_validation_errors=False)
        return output


@overload
def _flatten_output_spec(
    output_spec: OutputTypeOrFunction[T] | Sequence[OutputTypeOrFunction[T]],
) -> Sequence[OutputTypeOrFunction[T]]: ...


@overload
def _flatten_output_spec(output_spec: OutputSpec[T]) -> Sequence[_OutputSpecItem[T]]: ...


def _flatten_output_spec(output_spec: OutputSpec[T]) -> Sequence[_OutputSpecItem[T]]:
    outputs: Sequence[OutputSpec[T]]
    if isinstance(output_spec, Sequence):
        outputs = output_spec
    else:
        outputs = (output_spec,)

    outputs_flat: list[_OutputSpecItem[T]] = []
    for output in outputs:
        if isinstance(output, Sequence):
            outputs_flat.extend(_flatten_output_spec(cast(OutputSpec[T], output)))
        elif union_types := _utils.get_union_args(output):
            outputs_flat.extend(union_types)
        else:
            outputs_flat.append(cast(_OutputSpecItem[T], output))
    return outputs_flat



================================================
FILE: pydantic_ai_slim/pydantic_ai/_parts_manager.py
================================================
"""This module provides functionality to manage and update parts of a model's streamed response.

The manager tracks which parts (in particular, text and tool calls) correspond to which
vendor-specific identifiers (e.g., `index`, `tool_call_id`, etc., as appropriate for a given model),
and produces Pydantic AI-format events as appropriate for consumers of the streaming APIs.

The "vendor-specific identifiers" to use depend on the semantics of the responses of the responses from the vendor,
and are tightly coupled to the specific model being used, and the Pydantic AI Model subclass implementation.

This `ModelResponsePartsManager` is used in each of the subclasses of `StreamedResponse` as a way to consolidate
event-emitting logic.
"""

from __future__ import annotations as _annotations

from collections.abc import Hashable
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

from ._utils import generate_tool_call_id as _generate_tool_call_id

VendorId = Hashable
"""
Type alias for a vendor identifier, which can be any hashable type (e.g., a string, UUID, etc.)
"""

ManagedPart = ModelResponsePart | ToolCallPartDelta
"""
A union of types that are managed by the ModelResponsePartsManager.
Because many vendors have streaming APIs that may produce not-fully-formed tool calls,
this includes ToolCallPartDelta's in addition to the more fully-formed ModelResponsePart's.
"""


@dataclass
class ModelResponsePartsManager:
    """Manages a sequence of parts that make up a model's streamed response.

    Parts are generally added and/or updated by providing deltas, which are tracked by vendor-specific IDs.
    """

    _parts: list[ManagedPart] = field(default_factory=list, init=False)
    """A list of parts (text or tool calls) that make up the current state of the model's response."""
    _vendor_id_to_part_index: dict[VendorId, int] = field(default_factory=dict, init=False)
    """Maps a vendor's "part" ID (if provided) to the index in `_parts` where that part resides."""

    def get_parts(self) -> list[ModelResponsePart]:
        """Return only model response parts that are complete (i.e., not ToolCallPartDelta's).

        Returns:
            A list of ModelResponsePart objects. ToolCallPartDelta objects are excluded.
        """
        return [p for p in self._parts if not isinstance(p, ToolCallPartDelta)]

    def handle_text_delta(
        self,
        *,
        vendor_part_id: VendorId | None,
        content: str,
        id: str | None = None,
        thinking_tags: tuple[str, str] | None = None,
        ignore_leading_whitespace: bool = False,
    ) -> ModelResponseStreamEvent | None:
        """Handle incoming text content, creating or updating a TextPart in the manager as appropriate.

        When `vendor_part_id` is None, the latest part is updated if it exists and is a TextPart;
        otherwise, a new TextPart is created. When a non-None ID is specified, the TextPart corresponding
        to that vendor ID is either created or updated.

        Args:
            vendor_part_id: The ID the vendor uses to identify this piece
                of text. If None, a new part will be created unless the latest part is already
                a TextPart.
            content: The text content to append to the appropriate TextPart.
            id: An optional id for the text part.
            thinking_tags: If provided, will handle content between the thinking tags as thinking parts.
            ignore_leading_whitespace: If True, will ignore leading whitespace in the content.

        Returns:
            - A `PartStartEvent` if a new part was created.
            - A `PartDeltaEvent` if an existing part was updated.
            - `None` if no new event is emitted (e.g., the first text part was all whitespace).

        Raises:
            UnexpectedModelBehavior: If attempting to apply text content to a part that is not a TextPart.
        """
        existing_text_part_and_index: tuple[TextPart, int] | None = None

        if vendor_part_id is None:
            # If the vendor_part_id is None, check if the latest part is a TextPart to update
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, TextPart):
                    existing_text_part_and_index = latest_part, part_index
        else:
            # Otherwise, attempt to look up an existing TextPart by vendor_part_id
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]

                if thinking_tags and isinstance(existing_part, ThinkingPart):
                    # We may be building a thinking part instead of a text part if we had previously seen a thinking tag
                    if content == thinking_tags[1]:
                        # When we see the thinking end tag, we're done with the thinking part and the next text delta will need a new part
                        self._vendor_id_to_part_index.pop(vendor_part_id)
                        return None
                    else:
                        return self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=content)
                elif isinstance(existing_part, TextPart):
                    existing_text_part_and_index = existing_part, part_index
                else:
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {existing_part=}')

        if thinking_tags and content == thinking_tags[0]:
            # When we see a thinking start tag (which is a single token), we'll build a new thinking part instead
            self._vendor_id_to_part_index.pop(vendor_part_id, None)
            return self.handle_thinking_delta(vendor_part_id=vendor_part_id, content='')

        if existing_text_part_and_index is None:
            # This is a workaround for models that emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
            # which we don't want to end up treating as a final result when using `run_stream` with `str` a valid `output_type`.
            if ignore_leading_whitespace and (len(content) == 0 or content.isspace()):
                return None

            # There is no existing text part that should be updated, so create a new one
            new_part_index = len(self._parts)
            part = TextPart(content=content, id=id)
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = new_part_index
            self._parts.append(part)
            return PartStartEvent(index=new_part_index, part=part)
        else:
            # Update the existing TextPart with the new content delta
            existing_text_part, part_index = existing_text_part_and_index
            part_delta = TextPartDelta(content_delta=content)
            self._parts[part_index] = part_delta.apply(existing_text_part)
            return PartDeltaEvent(index=part_index, delta=part_delta)

    def handle_thinking_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        content: str | None = None,
        id: str | None = None,
        signature: str | None = None,
        provider_name: str | None = None,
    ) -> ModelResponseStreamEvent:
        """Handle incoming thinking content, creating or updating a ThinkingPart in the manager as appropriate.

        When `vendor_part_id` is None, the latest part is updated if it exists and is a ThinkingPart;
        otherwise, a new ThinkingPart is created. When a non-None ID is specified, the ThinkingPart corresponding
        to that vendor ID is either created or updated.

        Args:
            vendor_part_id: The ID the vendor uses to identify this piece
                of thinking. If None, a new part will be created unless the latest part is already
                a ThinkingPart.
            content: The thinking content to append to the appropriate ThinkingPart.
            id: An optional id for the thinking part.
            signature: An optional signature for the thinking content.
            provider_name: An optional provider name for the thinking part.

        Returns:
            A `PartStartEvent` if a new part was created, or a `PartDeltaEvent` if an existing part was updated.

        Raises:
            UnexpectedModelBehavior: If attempting to apply a thinking delta to a part that is not a ThinkingPart.
        """
        existing_thinking_part_and_index: tuple[ThinkingPart, int] | None = None

        if vendor_part_id is None:
            # If the vendor_part_id is None, check if the latest part is a ThinkingPart to update
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ThinkingPart):  # pragma: no branch
                    existing_thinking_part_and_index = latest_part, part_index
        else:
            # Otherwise, attempt to look up an existing ThinkingPart by vendor_part_id
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, ThinkingPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a thinking delta to {existing_part=}')
                existing_thinking_part_and_index = existing_part, part_index

        if existing_thinking_part_and_index is None:
            if content is not None or signature is not None:
                # There is no existing thinking part that should be updated, so create a new one
                new_part_index = len(self._parts)
                part = ThinkingPart(content=content or '', id=id, signature=signature, provider_name=provider_name)
                if vendor_part_id is not None:  # pragma: no branch
                    self._vendor_id_to_part_index[vendor_part_id] = new_part_index
                self._parts.append(part)
                return PartStartEvent(index=new_part_index, part=part)
            else:
                raise UnexpectedModelBehavior('Cannot create a ThinkingPart with no content or signature')
        else:
            if content is not None or signature is not None:
                # Update the existing ThinkingPart with the new content and/or signature delta
                existing_thinking_part, part_index = existing_thinking_part_and_index
                part_delta = ThinkingPartDelta(
                    content_delta=content, signature_delta=signature, provider_name=provider_name
                )
                self._parts[part_index] = part_delta.apply(existing_thinking_part)
                return PartDeltaEvent(index=part_index, delta=part_delta)
            else:
                raise UnexpectedModelBehavior('Cannot update a ThinkingPart with no content or signature')

    def handle_tool_call_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str | None,
        args: str | dict[str, Any] | None,
        tool_call_id: str | None,
    ) -> ModelResponseStreamEvent | None:
        """Handle or update a tool call, creating or updating a `ToolCallPart` or `ToolCallPartDelta`.

        Managed items remain as `ToolCallPartDelta`s until they have at least a tool_name, at which
        point they are upgraded to `ToolCallPart`s.

        If `vendor_part_id` is None, updates the latest matching ToolCallPart (or ToolCallPartDelta)
        if any. Otherwise, a new part (or delta) may be created.

        Args:
            vendor_part_id: The ID the vendor uses for this tool call.
                If None, the latest matching tool call may be updated.
            tool_name: The name of the tool. If None, the manager does not enforce
                a name match when `vendor_part_id` is None.
            args: Arguments for the tool call, either as a string, a dictionary of key-value pairs, or None.
            tool_call_id: An optional string representing an identifier for this tool call.

        Returns:
            - A `PartStartEvent` if a new ToolCallPart is created.
            - A `PartDeltaEvent` if an existing part is updated.
            - `None` if no new event is emitted (e.g., the part is still incomplete).

        Raises:
            UnexpectedModelBehavior: If attempting to apply a tool call delta to a part that is not
                a ToolCallPart or ToolCallPartDelta.
        """
        existing_matching_part_and_index: tuple[ToolCallPartDelta | ToolCallPart, int] | None = None

        if vendor_part_id is None:
            # vendor_part_id is None, so check if the latest part is a matching tool call or delta to update
            # When the vendor_part_id is None, if the tool_name is _not_ None, assume this should be a new part rather
            # than a delta on an existing one. We can change this behavior in the future if necessary for some model.
            if tool_name is None and self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ToolCallPart | ToolCallPartDelta):  # pragma: no branch
                    existing_matching_part_and_index = latest_part, part_index
        else:
            # vendor_part_id is provided, so look up the corresponding part or delta
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, ToolCallPartDelta | ToolCallPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a tool call delta to {existing_part=}')
                existing_matching_part_and_index = existing_part, part_index

        if existing_matching_part_and_index is None:
            # No matching part/delta was found, so create a new ToolCallPartDelta (or ToolCallPart if fully formed)
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            part = delta.as_part() or delta
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = len(self._parts)
            new_part_index = len(self._parts)
            self._parts.append(part)
            # Only emit a PartStartEvent if we have enough information to produce a full ToolCallPart
            if isinstance(part, ToolCallPart):
                return PartStartEvent(index=new_part_index, part=part)
        else:
            # Update the existing part or delta with the new information
            existing_part, part_index = existing_matching_part_and_index
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            updated_part = delta.apply(existing_part)
            self._parts[part_index] = updated_part
            if isinstance(updated_part, ToolCallPart):
                if isinstance(existing_part, ToolCallPartDelta):
                    # We just upgraded a delta to a full part, so emit a PartStartEvent
                    return PartStartEvent(index=part_index, part=updated_part)
                else:
                    # We updated an existing part, so emit a PartDeltaEvent
                    if updated_part.tool_call_id and not delta.tool_call_id:
                        delta = replace(delta, tool_call_id=updated_part.tool_call_id)
                    return PartDeltaEvent(index=part_index, delta=delta)

    def handle_tool_call_part(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str,
        args: str | dict[str, Any] | None,
        tool_call_id: str | None = None,
    ) -> ModelResponseStreamEvent:
        """Immediately create or fully-overwrite a ToolCallPart with the given information.

        This does not apply a delta; it directly sets the tool call part contents.

        Args:
            vendor_part_id: The vendor's ID for this tool call part. If not
                None and an existing part is found, that part is overwritten.
            tool_name: The name of the tool being invoked.
            args: The arguments for the tool call, either as a string, a dictionary, or None.
            tool_call_id: An optional string identifier for this tool call.

        Returns:
            ModelResponseStreamEvent: A `PartStartEvent` indicating that a new tool call part
            has been added to the manager, or replaced an existing part.
        """
        new_part = ToolCallPart(
            tool_name=tool_name,
            args=args,
            tool_call_id=tool_call_id or _generate_tool_call_id(),
        )
        if vendor_part_id is None:
            # vendor_part_id is None, so we unconditionally append a new ToolCallPart to the end of the list
            new_part_index = len(self._parts)
            self._parts.append(new_part)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new ToolCallPart.
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None:
                new_part_index = maybe_part_index
                self._parts[new_part_index] = new_part
            else:
                new_part_index = len(self._parts)
                self._parts.append(new_part)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        return PartStartEvent(index=new_part_index, part=new_part)



================================================
FILE: pydantic_ai_slim/pydantic_ai/_run_context.py
================================================
from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING, Generic

from opentelemetry.trace import NoOpTracer, Tracer
from typing_extensions import TypeVar

from . import _utils, messages as _messages

if TYPE_CHECKING:
    from .models import Model
    from .result import RunUsage

AgentDepsT = TypeVar('AgentDepsT', default=None, contravariant=True)
"""Type variable for agent dependencies."""


@dataclasses.dataclass(repr=False, kw_only=True)
class RunContext(Generic[AgentDepsT]):
    """Information about the current call."""

    deps: AgentDepsT
    """Dependencies for the agent."""
    model: Model
    """The model used in this run."""
    usage: RunUsage
    """LLM usage associated with the run."""
    prompt: str | Sequence[_messages.UserContent] | None = None
    """The original user prompt passed to the run."""
    messages: list[_messages.ModelMessage] = field(default_factory=list)
    """Messages exchanged in the conversation so far."""
    tracer: Tracer = field(default_factory=NoOpTracer)
    """The tracer to use for tracing the run."""
    trace_include_content: bool = False
    """Whether to include the content of the messages in the trace."""
    retries: dict[str, int] = field(default_factory=dict)
    """Number of retries for each tool so far."""
    tool_call_id: str | None = None
    """The ID of the tool call."""
    tool_name: str | None = None
    """Name of the tool being called."""
    retry: int = 0
    """Number of retries so far."""
    run_step: int = 0
    """The current step in the run."""
    tool_call_approved: bool = False
    """Whether a tool call that required approval has now been approved."""

    __repr__ = _utils.dataclasses_no_defaults_repr



================================================
FILE: pydantic_ai_slim/pydantic_ai/_system_prompt.py
================================================
from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, cast

from . import _utils
from ._run_context import AgentDepsT, RunContext
from .tools import SystemPromptFunc


@dataclass
class SystemPromptRunner(Generic[AgentDepsT]):
    function: SystemPromptFunc[AgentDepsT]
    dynamic: bool = False
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 0
        self._is_async = _utils.is_async_callable(self.function)

    async def run(self, run_context: RunContext[AgentDepsT]) -> str:
        if self._takes_ctx:
            args = (run_context,)
        else:
            args = ()

        if self._is_async:
            function = cast(Callable[[Any], Awaitable[str]], self.function)
            return await function(*args)
        else:
            function = cast(Callable[[Any], str], self.function)
            return await _utils.run_in_executor(function, *args)



================================================
FILE: pydantic_ai_slim/pydantic_ai/_thinking_part.py
================================================
from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart


def split_content_into_text_and_thinking(content: str, thinking_tags: tuple[str, str]) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.
    """
    start_tag, end_tag = thinking_tags
    parts: list[ThinkingPart | TextPart] = []

    start_index = content.find(start_tag)
    while start_index >= 0:
        before_think, content = content[:start_index], content[start_index + len(start_tag) :]
        if before_think:
            parts.append(TextPart(content=before_think))
        end_index = content.find(end_tag)
        if end_index >= 0:
            think_content, content = content[:end_index], content[end_index + len(end_tag) :]
            parts.append(ThinkingPart(content=think_content))
        else:
            # We lose the `<think>` tag, but it shouldn't matter.
            parts.append(TextPart(content=content))
            content = ''
        start_index = content.find(start_tag)
    if content:
        parts.append(TextPart(content=content))
    return parts



================================================
FILE: pydantic_ai_slim/pydantic_ai/_tool_manager.py
================================================
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Generic

from opentelemetry.trace import Tracer
from pydantic import ValidationError
from typing_extensions import assert_never

from . import messages as _messages
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior
from .messages import ToolCallPart
from .tools import ToolDefinition
from .toolsets.abstract import AbstractToolset, ToolsetTool
from .usage import UsageLimits

_sequential_tool_calls_ctx_var: ContextVar[bool] = ContextVar('sequential_tool_calls', default=False)


@dataclass
class ToolManager(Generic[AgentDepsT]):
    """Manages tools for an agent run step. It caches the agent run's toolset's tool definitions and handles calling tools and retries."""

    toolset: AbstractToolset[AgentDepsT]
    """The toolset that provides the tools for this run step."""
    ctx: RunContext[AgentDepsT] | None = None
    """The agent run context for a specific run step."""
    tools: dict[str, ToolsetTool[AgentDepsT]] | None = None
    """The cached tools for this run step."""
    failed_tools: set[str] = field(default_factory=set)
    """Names of tools that failed in this run step."""

    @classmethod
    @contextmanager
    def sequential_tool_calls(cls) -> Iterator[None]:
        """Run tool calls sequentially during the context."""
        token = _sequential_tool_calls_ctx_var.set(True)
        try:
            yield
        finally:
            _sequential_tool_calls_ctx_var.reset(token)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> ToolManager[AgentDepsT]:
        """Build a new tool manager for the next run step, carrying over the retries from the current run step."""
        if self.ctx is not None:
            if ctx.run_step == self.ctx.run_step:
                return self

            retries = {
                failed_tool_name: self.ctx.retries.get(failed_tool_name, 0) + 1
                for failed_tool_name in self.failed_tools
            }
            ctx = replace(ctx, retries=retries)

        return self.__class__(
            toolset=self.toolset,
            ctx=ctx,
            tools=await self.toolset.get_tools(ctx),
        )

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        """The tool definitions for the tools in this tool manager."""
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        return [tool.tool_def for tool in self.tools.values()]

    def should_call_sequentially(self, calls: list[ToolCallPart]) -> bool:
        """Whether to require sequential tool calls for a list of tool calls."""
        return _sequential_tool_calls_ctx_var.get() or any(
            tool_def.sequential for call in calls if (tool_def := self.get_tool_def(call.tool_name))
        )

    def get_tool_def(self, name: str) -> ToolDefinition | None:
        """Get the tool definition for a given tool name, or `None` if the tool is unknown."""
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        try:
            return self.tools[name].tool_def
        except KeyError:
            return None

    async def handle_call(
        self,
        call: ToolCallPart,
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
        usage_limits: UsageLimits | None = None,
    ) -> Any:
        """Handle a tool call by validating the arguments, calling the tool, and handling retries.

        Args:
            call: The tool call part to handle.
            allow_partial: Whether to allow partial validation of the tool arguments.
            wrap_validation_errors: Whether to wrap validation errors in a retry prompt part.
            usage_limits: Optional usage limits to check before executing tools.
        """
        if self.tools is None or self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        if (tool := self.tools.get(call.tool_name)) and tool.tool_def.kind == 'output':
            # Output tool calls are not traced and not counted
            return await self._call_tool(call, allow_partial, wrap_validation_errors, count_tool_usage=False)
        else:
            return await self._call_tool_traced(
                call,
                allow_partial,
                wrap_validation_errors,
                self.ctx.tracer,
                self.ctx.trace_include_content,
                usage_limits,
            )

    async def _call_tool(
        self,
        call: ToolCallPart,
        allow_partial: bool,
        wrap_validation_errors: bool,
        usage_limits: UsageLimits | None = None,
        count_tool_usage: bool = True,
    ) -> Any:
        if self.tools is None or self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        name = call.tool_name
        tool = self.tools.get(name)
        try:
            if tool is None:
                if self.tools:
                    msg = f'Available tools: {", ".join(f"{name!r}" for name in self.tools.keys())}'
                else:
                    msg = 'No tools available.'
                raise ModelRetry(f'Unknown tool name: {name!r}. {msg}')

            if tool.tool_def.defer:
                raise RuntimeError('Deferred tools cannot be called')

            ctx = replace(
                self.ctx,
                tool_name=name,
                tool_call_id=call.tool_call_id,
                retry=self.ctx.retries.get(name, 0),
            )

            pyd_allow_partial = 'trailing-strings' if allow_partial else 'off'
            validator = tool.args_validator
            if isinstance(call.args, str):
                args_dict = validator.validate_json(call.args or '{}', allow_partial=pyd_allow_partial)
            else:
                args_dict = validator.validate_python(call.args or {}, allow_partial=pyd_allow_partial)

            if usage_limits is not None and count_tool_usage:
                usage_limits.check_before_tool_call(self.ctx.usage)

            result = await self.toolset.call_tool(name, args_dict, ctx, tool)

            if count_tool_usage:
                self.ctx.usage.tool_calls += 1

            return result
        except (ValidationError, ModelRetry) as e:
            max_retries = tool.max_retries if tool is not None else 1
            current_retry = self.ctx.retries.get(name, 0)

            if current_retry == max_retries:
                raise UnexpectedModelBehavior(f'Tool {name!r} exceeded max retries count of {max_retries}') from e
            else:
                if wrap_validation_errors:
                    if isinstance(e, ValidationError):
                        m = _messages.RetryPromptPart(
                            tool_name=name,
                            content=e.errors(include_url=False, include_context=False),
                            tool_call_id=call.tool_call_id,
                        )
                        e = ToolRetryError(m)
                    elif isinstance(e, ModelRetry):
                        m = _messages.RetryPromptPart(
                            tool_name=name,
                            content=e.message,
                            tool_call_id=call.tool_call_id,
                        )
                        e = ToolRetryError(m)
                    else:
                        assert_never(e)

                if not allow_partial:
                    # If we're validating partial arguments, we don't want to count this as a failed tool as it may still succeed once the full arguments are received.
                    self.failed_tools.add(name)

                raise e

    async def _call_tool_traced(
        self,
        call: ToolCallPart,
        allow_partial: bool,
        wrap_validation_errors: bool,
        tracer: Tracer,
        include_content: bool = False,
        usage_limits: UsageLimits | None = None,
    ) -> Any:
        """See <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span>."""
        span_attributes = {
            'gen_ai.tool.name': call.tool_name,
            # NOTE: this means `gen_ai.tool.call.id` will be included even if it was generated by pydantic-ai
            'gen_ai.tool.call.id': call.tool_call_id,
            **({'tool_arguments': call.args_as_json_str()} if include_content else {}),
            'logfire.msg': f'running tool: {call.tool_name}',
            # add the JSON schema so these attributes are formatted nicely in Logfire
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {
                        **(
                            {
                                'tool_arguments': {'type': 'object'},
                                'tool_response': {'type': 'object'},
                            }
                            if include_content
                            else {}
                        ),
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                }
            ),
        }
        with tracer.start_as_current_span('running tool', attributes=span_attributes) as span:
            try:
                tool_result = await self._call_tool(call, allow_partial, wrap_validation_errors, usage_limits)
            except ToolRetryError as e:
                part = e.tool_retry
                if include_content and span.is_recording():
                    span.set_attribute('tool_response', part.model_response())
                raise e

            if include_content and span.is_recording():
                span.set_attribute(
                    'tool_response',
                    tool_result
                    if isinstance(tool_result, str)
                    else _messages.tool_return_ta.dump_json(tool_result).decode(),
                )

        return tool_result



================================================
FILE: pydantic_ai_slim/pydantic_ai/_utils.py
================================================
from __future__ import annotations as _annotations

import asyncio
import functools
import inspect
import re
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from functools import partial
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeGuard, TypeVar, get_args, get_origin, overload

from anyio.to_thread import run_sync
from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import (
    ParamSpec,
    TypeIs,
    is_typeddict,
)
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

from pydantic_graph._utils import AbstractSpan

from . import exceptions

AbstractSpan = AbstractSpan

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRun, AgentRunResult
    from pydantic_graph import GraphRun, GraphRunResult

    from . import messages as _messages
    from .tools import ObjectJsonSchema

_P = ParamSpec('_P')
_R = TypeVar('_R')


async def run_in_executor(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    wrapped_func = partial(func, *args, **kwargs)
    return await run_sync(wrapped_func)


def is_model_like(type_: Any) -> bool:
    """Check if something is a pydantic model, dataclass or typedict.

    These should all generate a JSON Schema with `{"type": "object"}` and therefore be usable directly as
    function parameters.
    """
    return (
        isinstance(type_, type)
        and not isinstance(type_, GenericAlias)
        and (
            issubclass(type_, BaseModel)
            or is_dataclass(type_)  # pyright: ignore[reportUnknownArgumentType]
            or is_typeddict(type_)  # pyright: ignore[reportUnknownArgumentType]
            or getattr(type_, '__is_model_like__', False)  # pyright: ignore[reportUnknownArgumentType]
        )
    )


def check_object_json_schema(schema: JsonSchemaValue) -> ObjectJsonSchema:
    from .exceptions import UserError

    if schema.get('type') == 'object':
        return schema
    elif schema.get('$ref') is not None:
        maybe_result = schema.get('$defs', {}).get(schema['$ref'][8:])  # This removes the initial "#/$defs/".

        if "'$ref': '#/$defs/" in str(maybe_result):
            return schema  # We can't remove the $defs because the schema contains other references
        return maybe_result
    else:
        raise UserError('Schema must be an object')


T = TypeVar('T')


@dataclass
class Some(Generic[T]):
    """Analogous to Rust's `Option::Some` type."""

    value: T


Option: TypeAlias = Some[T] | None
"""Analogous to Rust's `Option` type, usage: `Option[Thing]` is equivalent to `Some[Thing] | None`."""


class Unset:
    """A singleton to represent an unset value."""

    pass


UNSET = Unset()


def is_set(t_or_unset: T | Unset) -> TypeGuard[T]:
    return t_or_unset is not UNSET


@asynccontextmanager
async def group_by_temporal(
    aiterable: AsyncIterable[T], soft_max_interval: float | None
) -> AsyncIterator[AsyncIterable[list[T]]]:
    """Group items from an async iterable into lists based on time interval between them.

    Effectively, this debounces the iterator.

    This returns a context manager usable as an iterator so any pending tasks can be cancelled if an error occurs
    during iteration.

    Usage:

    ```python
    async with group_by_temporal(yield_groups(), 0.1) as groups_iter:
        async for groups in groups_iter:
            print(groups)
    ```

    Args:
        aiterable: The async iterable to group.
        soft_max_interval: Maximum interval over which to group items, this should avoid a trickle of items causing
            a group to never be yielded. It's a soft max in the sense that once we're over this time, we yield items
            as soon as `aiter.__anext__()` returns. If `None`, no grouping/debouncing is performed

    Returns:
        A context manager usable as an async iterable of lists of items produced by the input async iterable.
    """
    if soft_max_interval is None:

        async def async_iter_groups_noop() -> AsyncIterator[list[T]]:
            async for item in aiterable:
                yield [item]

        yield async_iter_groups_noop()
        return

    # we might wait for the next item more than once, so we store the task to await next time
    task: asyncio.Task[T] | None = None

    async def async_iter_groups() -> AsyncIterator[list[T]]:
        nonlocal task

        assert soft_max_interval is not None and soft_max_interval >= 0, 'soft_max_interval must be a positive number'
        buffer: list[T] = []
        group_start_time = time.monotonic()

        aiterator = aiterable.__aiter__()
        while True:
            if group_start_time is None:
                # group hasn't started, we just wait for the maximum interval
                wait_time = soft_max_interval
            else:
                # wait for the time remaining in the group
                wait_time = soft_max_interval - (time.monotonic() - group_start_time)

            # if there's no current task, we get the next one
            if task is None:
                # aiter.__anext__() returns an Awaitable[T], not a Coroutine which asyncio.create_task expects
                # so far, this doesn't seem to be a problem
                task = asyncio.create_task(aiterator.__anext__())  # pyright: ignore[reportArgumentType]

            # we use asyncio.wait to avoid cancelling the coroutine if it's not done
            done, _ = await asyncio.wait((task,), timeout=wait_time)

            if done:
                # the one task we waited for completed
                try:
                    item = done.pop().result()
                except StopAsyncIteration:
                    # if the task raised StopAsyncIteration, we're done iterating
                    if buffer:
                        yield buffer
                    task = None
                    break
                else:
                    # we got an item, add it to the buffer and set task to None to get the next item
                    buffer.append(item)
                    task = None
                    # if this is the first item in the group, set the group start time
                    if group_start_time is None:
                        group_start_time = time.monotonic()
            elif buffer:
                # otherwise if the task timeout expired and we have items in the buffer, yield the buffer
                yield buffer
                # clear the buffer and reset the group start time ready for the next group
                buffer = []
                group_start_time = None

    try:
        yield async_iter_groups()
    finally:  # pragma: no cover
        # after iteration if a tasks still exists, cancel it, this will only happen if an error occurred
        if task:
            task.cancel('Cancelling due to error in iterator')
            with suppress(asyncio.CancelledError):
                await task


def sync_anext(iterator: Iterator[T]) -> T:
    """Get the next item from a sync iterator, raising `StopAsyncIteration` if it's exhausted.

    Useful when iterating over a sync iterator in an async context.
    """
    try:
        return next(iterator)
    except StopIteration as e:
        raise StopAsyncIteration() from e


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def guard_tool_call_id(
    t: _messages.ToolCallPart
    | _messages.ToolReturnPart
    | _messages.RetryPromptPart
    | _messages.BuiltinToolCallPart
    | _messages.BuiltinToolReturnPart,
) -> str:
    """Type guard that either returns the tool call id or generates a new one if it's None."""
    return t.tool_call_id or generate_tool_call_id()


def generate_tool_call_id() -> str:
    """Generate a tool call id.

    Ensure that the tool call id is unique.
    """
    return f'pyd_ai_{uuid.uuid4().hex}'


class PeekableAsyncStream(Generic[T]):
    """Wraps an async iterable of type T and allows peeking at the *next* item without consuming it.

    We only buffer one item at a time (the next item). Once that item is yielded, it is discarded.
    This is a single-pass stream.
    """

    def __init__(self, source: AsyncIterable[T]):
        self._source = source
        self._source_iter: AsyncIterator[T] | None = None
        self._buffer: T | Unset = UNSET
        self._exhausted = False

    async def peek(self) -> T | Unset:
        """Returns the next item that would be yielded without consuming it.

        Returns None if the stream is exhausted.
        """
        if self._exhausted:
            return UNSET

        # If we already have a buffered item, just return it.
        if not isinstance(self._buffer, Unset):
            return self._buffer

        # Otherwise, we need to fetch the next item from the underlying iterator.
        if self._source_iter is None:
            self._source_iter = self._source.__aiter__()

        try:
            self._buffer = await self._source_iter.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            return UNSET

        return self._buffer

    async def is_exhausted(self) -> bool:
        """Returns True if the stream is exhausted, False otherwise."""
        return isinstance(await self.peek(), Unset)

    def __aiter__(self) -> AsyncIterator[T]:
        # For a single-pass iteration, we can return self as the iterator.
        return self

    async def __anext__(self) -> T:
        """Yields the buffered item if present, otherwise fetches the next item from the underlying source.

        Raises StopAsyncIteration if the stream is exhausted.
        """
        if self._exhausted:
            raise StopAsyncIteration

        # If we have a buffered item, yield it.
        if not isinstance(self._buffer, Unset):
            item = self._buffer
            self._buffer = UNSET
            return item

        # Otherwise, fetch the next item from the source.
        if self._source_iter is None:
            self._source_iter = self._source.__aiter__()

        try:
            return await self._source_iter.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            raise


def get_traceparent(x: AgentRun | AgentRunResult | GraphRun | GraphRunResult) -> str:
    return x._traceparent(required=False) or ''  # type: ignore[reportPrivateUsage]


def dataclasses_no_defaults_repr(self: Any) -> str:
    """Exclude fields with values equal to the field default."""
    kv_pairs = (
        f'{f.name}={getattr(self, f.name)!r}' for f in fields(self) if f.repr and getattr(self, f.name) != f.default
    )
    return f'{self.__class__.__qualname__}({", ".join(kv_pairs)})'


_datetime_ta = TypeAdapter(datetime)


def number_to_datetime(x: int | float) -> datetime:
    return _datetime_ta.validate_python(x)


AwaitableCallable = Callable[..., Awaitable[T]]


@overload
def is_async_callable(obj: AwaitableCallable[T]) -> TypeIs[AwaitableCallable[T]]: ...


@overload
def is_async_callable(obj: Any) -> TypeIs[AwaitableCallable[Any]]: ...


def is_async_callable(obj: Any) -> Any:
    """Correctly check if a callable is async.

    This function was copied from Starlette:
    https://github.com/encode/starlette/blob/78da9b9e218ab289117df7d62aee200ed4c59617/starlette/_utils.py#L36-L40
    """
    while isinstance(obj, functools.partial):
        obj = obj.func

    return inspect.iscoroutinefunction(obj) or (callable(obj) and inspect.iscoroutinefunction(obj.__call__))  # type: ignore


def _update_mapped_json_schema_refs(s: dict[str, Any], name_mapping: dict[str, str]) -> None:
    """Update $refs in a schema to use the new names from name_mapping."""
    if '$ref' in s:
        ref = s['$ref']
        if ref.startswith('#/$defs/'):  # pragma: no branch
            original_name = ref[8:]  # Remove '#/$defs/'
            new_name = name_mapping.get(original_name, original_name)
            s['$ref'] = f'#/$defs/{new_name}'

    # Recursively update refs in properties
    if 'properties' in s:
        props: dict[str, dict[str, Any]] = s['properties']
        for prop in props.values():
            _update_mapped_json_schema_refs(prop, name_mapping)

    # Handle arrays
    if 'items' in s and isinstance(s['items'], dict):
        items: dict[str, Any] = s['items']
        _update_mapped_json_schema_refs(items, name_mapping)
    if 'prefixItems' in s:
        prefix_items: list[dict[str, Any]] = s['prefixItems']
        for item in prefix_items:
            _update_mapped_json_schema_refs(item, name_mapping)

    # Handle unions
    for union_type in ['anyOf', 'oneOf']:
        if union_type in s:
            union_items: list[dict[str, Any]] = s[union_type]
            for item in union_items:
                _update_mapped_json_schema_refs(item, name_mapping)


def merge_json_schema_defs(schemas: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Merges the `$defs` from different JSON schemas into a single deduplicated `$defs`, handling name collisions of `$defs` that are not the same, and rewrites `$ref`s to point to the new `$defs`.

    Returns a tuple of the rewritten schemas and a dictionary of the new `$defs`.
    """
    all_defs: dict[str, dict[str, Any]] = {}
    rewritten_schemas: list[dict[str, Any]] = []

    for schema in schemas:
        if '$defs' not in schema:
            rewritten_schemas.append(schema)
            continue

        schema = schema.copy()
        defs = schema.pop('$defs', None)
        schema_name_mapping: dict[str, str] = {}

        # Process definitions and build mapping
        for name, def_schema in defs.items():
            if name not in all_defs:
                all_defs[name] = def_schema
                schema_name_mapping[name] = name
            elif def_schema != all_defs[name]:
                new_name = name
                if title := schema.get('title'):
                    new_name = f'{title}_{name}'

                i = 1
                original_new_name = new_name
                new_name = f'{new_name}_{i}'
                while new_name in all_defs:
                    i += 1
                    new_name = f'{original_new_name}_{i}'

                all_defs[new_name] = def_schema
                schema_name_mapping[name] = new_name

        _update_mapped_json_schema_refs(schema, schema_name_mapping)
        rewritten_schemas.append(schema)

    return rewritten_schemas, all_defs


def validate_empty_kwargs(_kwargs: dict[str, Any]) -> None:
    """Validate that no unknown kwargs remain after processing.

    Args:
        _kwargs: Dictionary of remaining kwargs after specific ones have been processed.

    Raises:
        UserError: If any unknown kwargs remain.
    """
    if _kwargs:
        unknown_kwargs = ', '.join(f'`{k}`' for k in _kwargs.keys())
        raise exceptions.UserError(f'Unknown keyword arguments: {unknown_kwargs}')


def strip_markdown_fences(text: str) -> str:
    if text.startswith('{'):
        return text

    regex = r'```(?:\w+)?\n(\{.*\})\n```'
    match = re.search(regex, text, re.DOTALL)
    if match:
        return match.group(1)

    return text


def _unwrap_annotated(tp: Any) -> Any:
    origin = get_origin(tp)
    while typing_objects.is_annotated(origin):
        tp = tp.__origin__
        origin = get_origin(tp)
    return tp


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `tp` is a union, otherwise return an empty tuple."""
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    tp = _unwrap_annotated(tp)
    origin = get_origin(tp)
    if is_union_origin(origin):
        return tuple(_unwrap_annotated(arg) for arg in get_args(tp))
    else:
        return ()



================================================
FILE: pydantic_ai_slim/pydantic_ai/ag_ui.py
================================================
"""Provides an AG-UI protocol adapter for the Pydantic AI agent.

This package provides seamless integration between pydantic-ai agents and ag-ui
for building interactive AI applications with streaming event-based communication.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import Field, dataclass, replace
from http import HTTPStatus
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ValidationError
from typing_extensions import assert_never

from . import _utils
from ._agent_graph import CallToolsNode, ModelRequestNode
from .agent import AbstractAgent, AgentRun, AgentRunResult
from .exceptions import UserError
from .messages import (
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from .models import KnownModelName, Model
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import AgentDepsT, DeferredToolRequests, ToolDefinition
from .toolsets import AbstractToolset
from .toolsets.external import ExternalToolset
from .usage import RunUsage, UsageLimits

try:
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
        DeveloperMessage,
        EventType,
        Message,
        RunAgentInput,
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        State,
        SystemMessage,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ThinkingEndEvent,
        ThinkingStartEvent,
        ThinkingTextMessageContentEvent,
        ThinkingTextMessageEndEvent,
        ThinkingTextMessageStartEvent,
        Tool as AGUITool,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
        ToolMessage,
        UserMessage,
    )
    from ag_ui.encoder import EventEncoder
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import BaseRoute
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e


__all__ = [
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'AGUIApp',
    'OnCompleteFunc',
    'handle_ag_ui_request',
    'run_ag_ui',
]

SSE_CONTENT_TYPE: Final[str] = 'text/event-stream'
"""Content type header value for Server-Sent Events (SSE)."""

OnCompleteFunc: TypeAlias = Callable[[AgentRunResult[Any]], None] | Callable[[AgentRunResult[Any]], Awaitable[None]]
"""Callback function type that receives the `AgentRunResult` of the completed run. Can be sync or async."""


class AGUIApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """ASGI application for running Pydantic AI agents with AG-UI protocol support."""

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        # Agent.iter parameters.
        output_type: OutputSpec[Any] | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        # Starlette parameters.
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[AGUIApp[AgentDepsT, OutputDataT]] | None = None,
    ) -> None:
        """An ASGI application that handles every AG-UI request by running the agent.

        Note that the `deps` will be the same for each request, with the exception of the AG-UI state that's
        injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ag_ui.StateHandler] protocol.
        To provide different `deps` for each request (e.g. based on the authenticated user),
        use [`pydantic_ai.ag_ui.run_ag_ui`][pydantic_ai.ag_ui.run_ag_ui] or
        [`pydantic_ai.ag_ui.handle_ag_ui_request`][pydantic_ai.ag_ui.handle_ag_ui_request] instead.

        Args:
            agent: The agent to run.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request. A starlette application will always
                automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
                outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
                `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
                exception cases occurring in the routing or endpoints.
            exception_handlers: A mapping of either integer status codes, or exception class types onto
                callables which handle the exceptions. Exception handler callables should be of the form
                `handler(request, exc) -> response` and may be either standard functions, or async functions.
            on_startup: A list of callables to run on application startup. Startup handler callables do not
                take any arguments, and may be either standard functions, or async functions.
            on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
                not take any arguments, and may be either standard functions, or async functions.
            lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
                This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
                the other, not both.
        """
        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )

        async def endpoint(request: Request) -> Response:
            """Endpoint to run the agent with the provided input data."""
            return await handle_ag_ui_request(
                agent,
                request,
                output_type=output_type,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            )

        self.router.add_route('/', endpoint, methods=['POST'], name='run_agent')


async def handle_ag_ui_request(
    agent: AbstractAgent[AgentDepsT, Any],
    request: Request,
    *,
    output_type: OutputSpec[Any] | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    on_complete: OnCompleteFunc | None = None,
) -> Response:
    """Handle an AG-UI request by running the agent and returning a streaming response.

    Args:
        agent: The agent to run.
        request: The Starlette request (e.g. from FastAPI) containing the AG-UI run input.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

    Returns:
        A streaming Starlette response with AG-UI protocol events.
    """
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        input_data = RunAgentInput.model_validate(await request.json())
    except ValidationError as e:  # pragma: no cover
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    return StreamingResponse(
        run_ag_ui(
            agent,
            input_data,
            accept,
            output_type=output_type,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            on_complete=on_complete,
        ),
        media_type=accept,
    )


async def run_ag_ui(
    agent: AbstractAgent[AgentDepsT, Any],
    run_input: RunAgentInput,
    accept: str = SSE_CONTENT_TYPE,
    *,
    output_type: OutputSpec[Any] | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    on_complete: OnCompleteFunc | None = None,
) -> AsyncIterator[str]:
    """Run the agent with the AG-UI run input and stream AG-UI protocol events.

    Args:
        agent: The agent to run.
        run_input: The AG-UI run input containing thread_id, run_id, messages, etc.
        accept: The accept header value for the run.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

    Yields:
        Streaming event chunks encoded as strings according to the accept header value.
    """
    encoder = EventEncoder(accept=accept)
    if run_input.tools:
        # AG-UI tools can't be prefixed as that would result in a mismatch between the tool names in the
        # Pydantic AI events and actual AG-UI tool names, preventing the tool from being called. If any
        # conflicts arise, the AG-UI tool should be renamed or a `PrefixedToolset` used for local toolsets.
        toolset = _AGUIFrontendToolset[AgentDepsT](run_input.tools)
        toolsets = [*toolsets, toolset] if toolsets else [toolset]

    try:
        yield encoder.encode(
            RunStartedEvent(
                thread_id=run_input.thread_id,
                run_id=run_input.run_id,
            ),
        )

        if not run_input.messages:
            raise _NoMessagesError

        raw_state: dict[str, Any] = run_input.state or {}
        if isinstance(deps, StateHandler):
            if isinstance(deps.state, BaseModel):
                try:
                    state = type(deps.state).model_validate(raw_state)
                except ValidationError as e:  # pragma: no cover
                    raise _InvalidStateError from e
            else:
                state = raw_state

            deps = replace(deps, state=state)
        elif raw_state:
            raise UserError(
                f'AG-UI state is provided but `deps` of type `{type(deps).__name__}` does not implement the `StateHandler` protocol: it needs to be a dataclass with a non-optional `state` field.'
            )
        else:
            # `deps` not being a `StateHandler` is OK if there is no state.
            pass

        messages = _messages_from_ag_ui(run_input.messages)

        async with agent.iter(
            user_prompt=None,
            output_type=[output_type or agent.output_type, DeferredToolRequests],
            message_history=messages,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
        ) as run:
            async for event in _agent_stream(run):
                yield encoder.encode(event)

        if on_complete is not None and run.result is not None:
            if _utils.is_async_callable(on_complete):
                await on_complete(run.result)
            else:
                await _utils.run_in_executor(on_complete, run.result)
    except _RunError as e:
        yield encoder.encode(
            RunErrorEvent(message=e.message, code=e.code),
        )
    except Exception as e:
        yield encoder.encode(
            RunErrorEvent(message=str(e)),
        )
        raise e
    else:
        yield encoder.encode(
            RunFinishedEvent(
                thread_id=run_input.thread_id,
                run_id=run_input.run_id,
            ),
        )


async def _agent_stream(run: AgentRun[AgentDepsT, Any]) -> AsyncIterator[BaseEvent]:
    """Run the agent streaming responses using AG-UI protocol events.

    Args:
        run: The agent run to process.

    Yields:
        AG-UI Server-Sent Events (SSE).
    """
    async for node in run:
        stream_ctx = _RequestStreamContext()
        if isinstance(node, ModelRequestNode):
            async with node.stream(run.ctx) as request_stream:
                async for agent_event in request_stream:
                    async for msg in _handle_model_request_event(stream_ctx, agent_event):
                        yield msg

                if stream_ctx.part_end:  # pragma: no branch
                    yield stream_ctx.part_end
                    stream_ctx.part_end = None
                if stream_ctx.thinking:
                    yield ThinkingEndEvent(
                        type=EventType.THINKING_END,
                    )
                    stream_ctx.thinking = False
        elif isinstance(node, CallToolsNode):
            async with node.stream(run.ctx) as handle_stream:
                async for event in handle_stream:
                    if isinstance(event, FunctionToolResultEvent):
                        async for msg in _handle_tool_result_event(stream_ctx, event):
                            yield msg


async def _handle_model_request_event(  # noqa: C901
    stream_ctx: _RequestStreamContext,
    agent_event: ModelResponseStreamEvent,
) -> AsyncIterator[BaseEvent]:
    """Handle an agent event and yield AG-UI protocol events.

    Args:
        stream_ctx: The request stream context to manage state.
        agent_event: The agent event to process.

    Yields:
        AG-UI Server-Sent Events (SSE) based on the agent event.
    """
    if isinstance(agent_event, PartStartEvent):
        if stream_ctx.part_end:
            # End the previous part.
            yield stream_ctx.part_end
            stream_ctx.part_end = None

        part = agent_event.part
        if isinstance(part, ThinkingPart):  # pragma: no branch
            if not stream_ctx.thinking:
                yield ThinkingStartEvent(
                    type=EventType.THINKING_START,
                )
                stream_ctx.thinking = True

            if part.content:
                yield ThinkingTextMessageStartEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_START,
                )
                yield ThinkingTextMessageContentEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                    delta=part.content,
                )
                stream_ctx.part_end = ThinkingTextMessageEndEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_END,
                )
        else:
            if stream_ctx.thinking:
                yield ThinkingEndEvent(
                    type=EventType.THINKING_END,
                )
                stream_ctx.thinking = False

            if isinstance(part, TextPart):
                message_id = stream_ctx.new_message_id()
                yield TextMessageStartEvent(
                    message_id=message_id,
                )
                if part.content:  # pragma: no branch
                    yield TextMessageContentEvent(
                        message_id=message_id,
                        delta=part.content,
                    )
                stream_ctx.part_end = TextMessageEndEvent(
                    message_id=message_id,
                )
            elif isinstance(part, ToolCallPart):  # pragma: no branch
                message_id = stream_ctx.message_id or stream_ctx.new_message_id()
                yield ToolCallStartEvent(
                    tool_call_id=part.tool_call_id,
                    tool_call_name=part.tool_name,
                    parent_message_id=message_id,
                )
                if part.args:
                    yield ToolCallArgsEvent(
                        tool_call_id=part.tool_call_id,
                        delta=part.args if isinstance(part.args, str) else json.dumps(part.args),
                    )
                stream_ctx.part_end = ToolCallEndEvent(
                    tool_call_id=part.tool_call_id,
                )

    elif isinstance(agent_event, PartDeltaEvent):
        delta = agent_event.delta
        if isinstance(delta, TextPartDelta):
            if delta.content_delta:  # pragma: no branch
                yield TextMessageContentEvent(
                    message_id=stream_ctx.message_id,
                    delta=delta.content_delta,
                )
        elif isinstance(delta, ToolCallPartDelta):  # pragma: no branch
            assert delta.tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
            yield ToolCallArgsEvent(
                tool_call_id=delta.tool_call_id,
                delta=delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta),
            )
        elif isinstance(delta, ThinkingPartDelta):  # pragma: no branch
            if delta.content_delta:  # pragma: no branch
                if not isinstance(stream_ctx.part_end, ThinkingTextMessageEndEvent):
                    yield ThinkingTextMessageStartEvent(
                        type=EventType.THINKING_TEXT_MESSAGE_START,
                    )
                    stream_ctx.part_end = ThinkingTextMessageEndEvent(
                        type=EventType.THINKING_TEXT_MESSAGE_END,
                    )

                yield ThinkingTextMessageContentEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                    delta=delta.content_delta,
                )


async def _handle_tool_result_event(
    stream_ctx: _RequestStreamContext,
    event: FunctionToolResultEvent,
) -> AsyncIterator[BaseEvent]:
    """Convert a tool call result to AG-UI events.

    Args:
        stream_ctx: The request stream context to manage state.
        event: The tool call result event to process.

    Yields:
        AG-UI Server-Sent Events (SSE).
    """
    result = event.result
    if not isinstance(result, ToolReturnPart):
        return

    message_id = stream_ctx.new_message_id()
    yield ToolCallResultEvent(
        message_id=message_id,
        type=EventType.TOOL_CALL_RESULT,
        role='tool',
        tool_call_id=result.tool_call_id,
        content=result.model_response_str(),
    )

    # Now check for AG-UI events returned by the tool calls.
    possible_event = result.metadata or result.content
    if isinstance(possible_event, BaseEvent):
        yield possible_event
    elif isinstance(possible_event, str | bytes):  # pragma: no branch
        # Avoid iterable check for strings and bytes.
        pass
    elif isinstance(possible_event, Iterable):  # pragma: no branch
        for item in possible_event:  # type: ignore[reportUnknownMemberType]
            if isinstance(item, BaseEvent):  # pragma: no branch
                yield item


def _messages_from_ag_ui(messages: list[Message]) -> list[ModelMessage]:
    """Convert a AG-UI history to a Pydantic AI one."""
    result: list[ModelMessage] = []
    tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
    request_parts: list[ModelRequestPart] | None = None
    response_parts: list[ModelResponsePart] | None = None
    for msg in messages:
        if isinstance(msg, UserMessage | SystemMessage | DeveloperMessage | ToolMessage):
            if request_parts is None:
                request_parts = []
                result.append(ModelRequest(parts=request_parts))
                response_parts = None

            if isinstance(msg, UserMessage):
                request_parts.append(UserPromptPart(content=msg.content))
            elif isinstance(msg, SystemMessage | DeveloperMessage):
                request_parts.append(SystemPromptPart(content=msg.content))
            elif isinstance(msg, ToolMessage):
                tool_name = tool_calls.get(msg.tool_call_id)
                if tool_name is None:  # pragma: no cover
                    raise _ToolCallNotFoundError(tool_call_id=msg.tool_call_id)

                request_parts.append(
                    ToolReturnPart(
                        tool_name=tool_name,
                        content=msg.content,
                        tool_call_id=msg.tool_call_id,
                    )
                )
            else:
                assert_never(msg)

        elif isinstance(msg, AssistantMessage):
            if response_parts is None:
                response_parts = []
                result.append(ModelResponse(parts=response_parts))
                request_parts = None

            if msg.content:
                response_parts.append(TextPart(content=msg.content))

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls[tool_call.id] = tool_call.function.name

                response_parts.extend(
                    ToolCallPart(
                        tool_name=tool_call.function.name,
                        tool_call_id=tool_call.id,
                        args=tool_call.function.arguments,
                    )
                    for tool_call in msg.tool_calls
                )
        else:
            assert_never(msg)

    return result


@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs. Requires the class to be a dataclass with a `state` field."""

    # Has to be a dataclass so we can use `replace` to update the state.
    # From https://github.com/python/typeshed/blob/9ab7fde0a0cd24ed7a72837fcb21093b811b80d8/stdlib/_typeshed/__init__.pyi#L352
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def state(self) -> State:
        """Get the current state of the agent run."""
        ...

    @state.setter
    def state(self, state: State) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Args:
            state: The run state.

        Raises:
            InvalidStateError: If `state` does not match the expected model.
        """
        ...


StateT = TypeVar('StateT', bound=BaseModel)
"""Type variable for the state type, which must be a subclass of `BaseModel`."""


@dataclass
class StateDeps(Generic[StateT]):
    """Provides AG-UI state management.

    This class is used to manage the state of an agent run. It allows setting
    the state of the agent run with a specific type of state model, which must
    be a subclass of `BaseModel`.

    The state is set using the `state` setter by the `Adapter` when the run starts.

    Implements the `StateHandler` protocol.
    """

    state: StateT


@dataclass(repr=False)
class _RequestStreamContext:
    """Data class to hold request stream context."""

    message_id: str = ''
    part_end: BaseEvent | None = None
    thinking: bool = False

    def new_message_id(self) -> str:
        """Generate a new message ID for the request stream.

        Assigns a new UUID to the `message_id` and returns it.

        Returns:
            A new message ID.
        """
        self.message_id = str(uuid.uuid4())
        return self.message_id


@dataclass
class _RunError(Exception):
    """Exception raised for errors during agent runs."""

    message: str
    code: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


@dataclass
class _NoMessagesError(_RunError):
    """Exception raised when no messages are found in the input."""

    message: str = 'no messages found in the input'
    code: str = 'no_messages'


@dataclass
class _InvalidStateError(_RunError, ValidationError):
    """Exception raised when an invalid state is provided."""

    message: str = 'invalid state provided'
    code: str = 'invalid_state'


class _ToolCallNotFoundError(_RunError, ValueError):
    """Exception raised when an tool result is present without a matching call."""

    def __init__(self, tool_call_id: str) -> None:
        """Initialize the exception with the tool call ID."""
        super().__init__(  # pragma: no cover
            message=f'Tool call with ID {tool_call_id} not found in the history.',
            code='tool_call_not_found',
        )


class _AGUIFrontendToolset(ExternalToolset[AgentDepsT]):
    def __init__(self, tools: list[AGUITool]):
        super().__init__(
            [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters,
                )
                for tool in tools
            ]
        )

    @property
    def label(self) -> str:
        return 'the AG-UI frontend tools'  # pragma: no cover



================================================
FILE: pydantic_ai_slim/pydantic_ai/builtin_tools.py
================================================
from __future__ import annotations as _annotations

from abc import ABC
from dataclasses import dataclass
from typing import Literal

from typing_extensions import TypedDict

__all__ = ('AbstractBuiltinTool', 'WebSearchTool', 'WebSearchUserLocation', 'CodeExecutionTool', 'UrlContextTool')


@dataclass(kw_only=True)
class AbstractBuiltinTool(ABC):
    """A builtin tool that can be used by an agent.

    This class is abstract and cannot be instantiated directly.

    The builtin tools are passed to the model as part of the `ModelRequestParameters`.
    """


@dataclass(kw_only=True)
class WebSearchTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to search the web for information.

    The parameters that PydanticAI passes depend on the model, as some parameters may not be supported by certain models.

    Supported by:

    * Anthropic
    * OpenAI Responses
    * Groq
    * Google
    """

    search_context_size: Literal['low', 'medium', 'high'] = 'medium'
    """The `search_context_size` parameter controls how much context is retrieved from the web to help the tool formulate a response.

    Supported by:

    * OpenAI Responses
    """

    user_location: WebSearchUserLocation | None = None
    """The `user_location` parameter allows you to localize search results based on a user's location.

    Supported by:

    * Anthropic
    * OpenAI Responses
    """

    blocked_domains: list[str] | None = None
    """If provided, these domains will never appear in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
    * Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>
    """

    allowed_domains: list[str] | None = None
    """If provided, only these domains will be included in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
    * Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>
    """

    max_uses: int | None = None
    """If provided, the tool will stop searching the web after the given number of uses.

    Supported by:

    * Anthropic
    """


class WebSearchUserLocation(TypedDict, total=False):
    """Allows you to localize search results based on a user's location.

    Supported by:

    * Anthropic
    * OpenAI Responses
    """

    city: str
    """The city where the user is located."""

    country: str
    """The country where the user is located. For OpenAI, this must be a 2-letter country code (e.g., 'US', 'GB')."""

    region: str
    """The region or state where the user is located."""

    timezone: str
    """The timezone of the user's location."""


class CodeExecutionTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to execute code.

    Supported by:

    * Anthropic
    * OpenAI Responses
    * Google
    """


class UrlContextTool(AbstractBuiltinTool):
    """Allows your agent to access contents from URLs.

    Supported by:

    * Google
    """



================================================
FILE: pydantic_ai_slim/pydantic_ai/direct.py
================================================
"""Methods for making imperative requests to language models with minimal abstraction.

These methods allow you to make requests to LLMs where the only abstraction is input and output schema
translation so you can use all models with the same API.

These methods are thin wrappers around [`Model`][pydantic_ai.models.Model] implementations.
"""

from __future__ import annotations as _annotations

import queue
import threading
from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType

from pydantic_ai.usage import RequestUsage
from pydantic_graph._utils import get_event_loop as _get_event_loop

from . import agent, messages, models, settings
from .models import StreamedResponse, instrumented as instrumented_models

__all__ = (
    'model_request',
    'model_request_sync',
    'model_request_stream',
    'model_request_stream_sync',
    'StreamedResponseSync',
)

STREAM_INITIALIZATION_TIMEOUT = 30


async def model_request(
    model: models.Model | models.KnownModelName | str,
    messages: list[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> messages.ModelResponse:
    """Make a non-streamed request to a model.

    ```py title="model_request_example.py"
    from pydantic_ai.direct import model_request
    from pydantic_ai.messages import ModelRequest


    async def main():
        model_response = await model_request(
            'anthropic:claude-3-5-haiku-latest',
            [ModelRequest.user_text_prompt('What is the capital of France?')]  # (1)!
        )
        print(model_response)
        '''
        ModelResponse(
            parts=[TextPart(content='The capital of France is Paris.')],
            usage=RequestUsage(input_tokens=56, output_tokens=7),
            model_name='claude-3-5-haiku-latest',
            timestamp=datetime.datetime(...),
        )
        '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        The model response and token usage associated with the request.
    """
    model_instance = _prepare_model(model, instrument)
    return await model_instance.request(
        messages,
        model_settings,
        model_instance.customize_request_parameters(model_request_parameters or models.ModelRequestParameters()),
    )


def model_request_sync(
    model: models.Model | models.KnownModelName | str,
    messages: list[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> messages.ModelResponse:
    """Make a Synchronous, non-streamed request to a model.

    This is a convenience method that wraps [`model_request`][pydantic_ai.direct.model_request] with
    `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

    ```py title="model_request_sync_example.py"
    from pydantic_ai.direct import model_request_sync
    from pydantic_ai.messages import ModelRequest

    model_response = model_request_sync(
        'anthropic:claude-3-5-haiku-latest',
        [ModelRequest.user_text_prompt('What is the capital of France?')]  # (1)!
    )
    print(model_response)
    '''
    ModelResponse(
        parts=[TextPart(content='The capital of France is Paris.')],
        usage=RequestUsage(input_tokens=56, output_tokens=7),
        model_name='claude-3-5-haiku-latest',
        timestamp=datetime.datetime(...),
    )
    '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        The model response and token usage associated with the request.
    """
    return _get_event_loop().run_until_complete(
        model_request(
            model,
            messages,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
            instrument=instrument,
        )
    )


def model_request_stream(
    model: models.Model | models.KnownModelName | str,
    messages: list[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> AbstractAsyncContextManager[models.StreamedResponse]:
    """Make a streamed async request to a model.

    ```py {title="model_request_stream_example.py"}

    from pydantic_ai.direct import model_request_stream
    from pydantic_ai.messages import ModelRequest


    async def main():
        messages = [ModelRequest.user_text_prompt('Who was Albert Einstein?')]  # (1)!
        async with model_request_stream('openai:gpt-4.1-mini', messages) as stream:
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            print(chunks)
            '''
            [
                PartStartEvent(index=0, part=TextPart(content='Albert Einstein was ')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(
                    index=0, delta=TextPartDelta(content_delta='a German-born theoretical ')
                ),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='physicist.')),
            ]
            '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        A [stream response][pydantic_ai.models.StreamedResponse] async context manager.
    """
    model_instance = _prepare_model(model, instrument)
    return model_instance.request_stream(
        messages,
        model_settings,
        model_instance.customize_request_parameters(model_request_parameters or models.ModelRequestParameters()),
    )


def model_request_stream_sync(
    model: models.Model | models.KnownModelName | str,
    messages: list[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> StreamedResponseSync:
    """Make a streamed synchronous request to a model.

    This is the synchronous version of [`model_request_stream`][pydantic_ai.direct.model_request_stream].
    It uses threading to run the asynchronous stream in the background while providing a synchronous iterator interface.

    ```py {title="model_request_stream_sync_example.py"}

    from pydantic_ai.direct import model_request_stream_sync
    from pydantic_ai.messages import ModelRequest

    messages = [ModelRequest.user_text_prompt('Who was Albert Einstein?')]
    with model_request_stream_sync('openai:gpt-4.1-mini', messages) as stream:
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        print(chunks)
        '''
        [
            PartStartEvent(index=0, part=TextPart(content='Albert Einstein was ')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=0, delta=TextPartDelta(content_delta='a German-born theoretical ')
            ),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='physicist.')),
        ]
        '''
    ```

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        A [sync stream response][pydantic_ai.direct.StreamedResponseSync] context manager.
    """
    async_stream_cm = model_request_stream(
        model=model,
        messages=messages,
        model_settings=model_settings,
        model_request_parameters=model_request_parameters,
        instrument=instrument,
    )

    return StreamedResponseSync(async_stream_cm)


def _prepare_model(
    model: models.Model | models.KnownModelName | str,
    instrument: instrumented_models.InstrumentationSettings | bool | None,
) -> models.Model:
    model_instance = models.infer_model(model)

    if instrument is None:
        instrument = agent.Agent._instrument_default  # pyright: ignore[reportPrivateUsage]

    return instrumented_models.instrument_model(model_instance, instrument)


@dataclass
class StreamedResponseSync:
    """Synchronous wrapper to async streaming responses by running the async producer in a background thread and providing a synchronous iterator.

    This class must be used as a context manager with the `with` statement.
    """

    _async_stream_cm: AbstractAsyncContextManager[StreamedResponse]
    _queue: queue.Queue[messages.ModelResponseStreamEvent | Exception | None] = field(
        default_factory=queue.Queue, init=False
    )
    _thread: threading.Thread | None = field(default=None, init=False)
    _stream_response: StreamedResponse | None = field(default=None, init=False)
    _exception: Exception | None = field(default=None, init=False)
    _context_entered: bool = field(default=False, init=False)
    _stream_ready: threading.Event = field(default_factory=threading.Event, init=False)

    def __enter__(self) -> StreamedResponseSync:
        self._context_entered = True
        self._start_producer()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        self._cleanup()

    def __iter__(self) -> Iterator[messages.ModelResponseStreamEvent]:
        """Stream the response as an iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
        self._check_context_manager_usage()

        while True:
            item = self._queue.get()
            if item is None:  # End of stream
                break
            elif isinstance(item, Exception):
                raise item
            else:
                yield item

    def __repr__(self) -> str:
        if self._stream_response:
            return repr(self._stream_response)
        else:
            return f'{self.__class__.__name__}(context_entered={self._context_entered})'

    __str__ = __repr__

    def _check_context_manager_usage(self) -> None:
        if not self._context_entered:
            raise RuntimeError(
                'StreamedResponseSync must be used as a context manager. '
                'Use: `with model_request_stream_sync(...) as stream:`'
            )

    def _ensure_stream_ready(self) -> StreamedResponse:
        self._check_context_manager_usage()

        if self._stream_response is None:
            # Wait for the background thread to signal that the stream is ready
            if not self._stream_ready.wait(timeout=STREAM_INITIALIZATION_TIMEOUT):
                raise RuntimeError('Stream failed to initialize within timeout')

            if self._stream_response is None:  # pragma: no cover
                raise RuntimeError('Stream failed to initialize')

        return self._stream_response

    def _start_producer(self):
        self._thread = threading.Thread(target=self._async_producer, daemon=True)
        self._thread.start()

    def _async_producer(self):
        async def _consume_async_stream():
            try:
                async with self._async_stream_cm as stream:
                    self._stream_response = stream
                    # Signal that the stream is ready
                    self._stream_ready.set()
                    async for event in stream:
                        self._queue.put(event)
            except Exception as e:
                # Signal ready even on error so waiting threads don't hang
                self._stream_ready.set()
                self._queue.put(e)
            finally:
                self._queue.put(None)  # Signal end

        _get_event_loop().run_until_complete(_consume_async_stream())

    def _cleanup(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def get(self) -> messages.ModelResponse:
        """Build a ModelResponse from the data received from the stream so far."""
        return self._ensure_stream_ready().get()

    def usage(self) -> RequestUsage:
        """Get the usage of the response so far."""
        return self._ensure_stream_ready().usage()

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._ensure_stream_ready().model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._ensure_stream_ready().timestamp



================================================
FILE: pydantic_ai_slim/pydantic_ai/exceptions.py
================================================
from __future__ import annotations as _annotations

import json
import sys
from typing import TYPE_CHECKING, Any

from pydantic_core import core_schema

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup as ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover

if TYPE_CHECKING:
    from .messages import RetryPromptPart

__all__ = (
    'ModelRetry',
    'CallDeferred',
    'ApprovalRequired',
    'UserError',
    'AgentRunError',
    'UnexpectedModelBehavior',
    'UsageLimitExceeded',
    'ModelHTTPError',
    'FallbackExceptionGroup',
)


class ModelRetry(Exception):
    """Exception to raise when a tool function should be retried.

    The agent will return the message to the model and ask it to try calling the function/tool again.
    """

    message: str
    """The message to return to the model."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and other.message == self.message

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> core_schema.CoreSchema:
        """Pydantic core schema to allow `ModelRetry` to be (de)serialized."""
        schema = core_schema.typed_dict_schema(
            {
                'message': core_schema.typed_dict_field(core_schema.str_schema()),
                'kind': core_schema.typed_dict_field(core_schema.literal_schema(['model-retry'])),
            }
        )
        return core_schema.no_info_after_validator_function(
            lambda dct: ModelRetry(dct['message']),
            schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: {'message': x.message, 'kind': 'model-retry'},
                return_schema=schema,
            ),
        )


class CallDeferred(Exception):
    """Exception to raise when a tool call should be deferred.

    See [tools docs](../deferred-tools.md#deferred-tools) for more information.
    """

    pass


class ApprovalRequired(Exception):
    """Exception to raise when a tool call requires human-in-the-loop approval.

    See [tools docs](../deferred-tools.md#human-in-the-loop-tool-approval) for more information.
    """

    pass


class UserError(RuntimeError):
    """Error caused by a usage mistake by the application developer — You!"""

    message: str
    """Description of the mistake."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class AgentRunError(RuntimeError):
    """Base class for errors occurring during an agent run."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class UsageLimitExceeded(AgentRunError):
    """Error raised when a Model's usage exceeds the specified limits."""


class UnexpectedModelBehavior(AgentRunError):
    """Error caused by unexpected Model behavior, e.g. an unexpected response code."""

    message: str
    """Description of the unexpected behavior."""
    body: str | None
    """The body of the response, if available."""

    def __init__(self, message: str, body: str | None = None):
        self.message = message
        if body is None:
            self.body: str | None = None
        else:
            try:
                self.body = json.dumps(json.loads(body), indent=2)
            except ValueError:
                self.body = body
        super().__init__(message)

    def __str__(self) -> str:
        if self.body:
            return f'{self.message}, body:\n{self.body}'
        else:
            return self.message


class ModelHTTPError(AgentRunError):
    """Raised when an model provider response has a status code of 4xx or 5xx."""

    status_code: int
    """The HTTP status code returned by the API."""

    model_name: str
    """The name of the model associated with the error."""

    body: object | None
    """The body of the response, if available."""

    message: str
    """The error message with the status code and response body, if available."""

    def __init__(self, status_code: int, model_name: str, body: object | None = None):
        self.status_code = status_code
        self.model_name = model_name
        self.body = body
        message = f'status_code: {status_code}, model_name: {model_name}, body: {body}'
        super().__init__(message)


class FallbackExceptionGroup(ExceptionGroup):
    """A group of exceptions that can be raised when all fallback models fail."""


class ToolRetryError(Exception):
    """Exception used to signal a `ToolRetry` message should be returned to the LLM."""

    def __init__(self, tool_retry: RetryPromptPart):
        self.tool_retry = tool_retry
        super().__init__()



================================================
FILE: pydantic_ai_slim/pydantic_ai/format_prompt.py
================================================
from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ('format_as_xml',)


def format_as_xml(
    obj: Any,
    root_tag: str | None = None,
    item_tag: str = 'item',
    none_str: str = 'null',
    indent: str | None = '  ',
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
    `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai import format_as_xml

    print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
    '''
    <user>
      <name>John</name>
      <height>6</height>
      <weight>200</weight>
    </user>
    '''
    ```
    """
    el = _ToXml(item_tag=item_tag, none_str=none_str).to_xml(obj, root_tag)
    if root_tag is None and el.text is None:
        join = '' if indent is None else '\n'
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')


@dataclass
class _ToXml:
    item_tag: str
    none_str: str

    def to_xml(self, value: Any, tag: str | None) -> ElementTree.Element:
        element = ElementTree.Element(self.item_tag if tag is None else tag)
        if value is None:
            element.text = self.none_str
        elif isinstance(value, str):
            element.text = value
        elif isinstance(value, bytes | bytearray):
            element.text = value.decode(errors='ignore')
        elif isinstance(value, bool | int | float):
            element.text = str(value)
        elif isinstance(value, date):
            element.text = value.isoformat()
        elif isinstance(value, Mapping):
            self._mapping_to_xml(element, value)  # pyright: ignore[reportUnknownArgumentType]
        elif is_dataclass(value) and not isinstance(value, type):
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            dc_dict = asdict(value)
            self._mapping_to_xml(element, dc_dict)
        elif isinstance(value, BaseModel):
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            self._mapping_to_xml(element, value.model_dump(mode='python'))
        elif isinstance(value, Iterable):
            for item in value:  # pyright: ignore[reportUnknownVariableType]
                item_el = self.to_xml(item, None)
                element.append(item_el)
        else:
            raise TypeError(f'Unsupported type for XML formatting: {type(value)}')
        return element

    def _mapping_to_xml(self, element: ElementTree.Element, mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')



================================================
FILE: pydantic_ai_slim/pydantic_ai/mcp.py
================================================
from __future__ import annotations

import base64
import functools
import warnings
from abc import ABC, abstractmethod
from asyncio import Lock
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import field, replace
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any

import anyio
import httpx
import pydantic_core
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, Discriminator, Field, Tag
from pydantic_core import CoreSchema, core_schema
from typing_extensions import Self, assert_never, deprecated

from pydantic_ai.tools import RunContext, ToolDefinition

from .direct import model_request
from .toolsets.abstract import AbstractToolset, ToolsetTool

try:
    from mcp import types as mcp_types
    from mcp.client.session import ClientSession, ElicitationFnT, LoggingFnT
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
    from mcp.shared.context import RequestContext
    from mcp.shared.exceptions import McpError
    from mcp.shared.message import SessionMessage
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use the MCP server, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error

# after mcp imports so any import error maps to this file, not _mcp.py
from . import _mcp, _utils, exceptions, messages, models

__all__ = 'MCPServer', 'MCPServerStdio', 'MCPServerHTTP', 'MCPServerSSE', 'MCPServerStreamableHTTP', 'load_mcp_servers'

TOOL_SCHEMA_VALIDATOR = pydantic_core.SchemaValidator(
    schema=pydantic_core.core_schema.dict_schema(
        pydantic_core.core_schema.str_schema(), pydantic_core.core_schema.any_schema()
    )
)


class MCPServer(AbstractToolset[Any], ABC):
    """Base class for attaching agents to MCP servers.

    See <https://modelcontextprotocol.io> for more information.
    """

    tool_prefix: str | None
    """A prefix to add to all tools that are registered with the server.

    If not empty, will include a trailing underscore(`_`).

    e.g. if `tool_prefix='foo'`, then a tool named `bar` will be registered as `foo_bar`
    """

    log_level: mcp_types.LoggingLevel | None
    """The log level to set when connecting to the server, if any.

    See <https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/logging#logging> for more details.

    If `None`, no log level will be set.
    """

    log_handler: LoggingFnT | None
    """A handler for logging messages from the server."""

    timeout: float
    """The timeout in seconds to wait for the client to initialize."""

    read_timeout: float
    """Maximum time in seconds to wait for new messages before timing out.

    This timeout applies to the long-lived connection after it's established.
    If no new messages are received within this time, the connection will be considered stale
    and may be closed. Defaults to 5 minutes (300 seconds).
    """

    process_tool_call: ProcessToolCallback | None
    """Hook to customize tool calling and optionally pass extra metadata."""

    allow_sampling: bool
    """Whether to allow MCP sampling through this client."""

    sampling_model: models.Model | None
    """The model to use for sampling."""

    max_retries: int
    """The maximum number of times to retry a tool call."""

    elicitation_callback: ElicitationFnT | None = None
    """Callback function to handle elicitation requests from the server."""

    _id: str | None

    _enter_lock: Lock = field(compare=False)
    _running_count: int
    _exit_stack: AsyncExitStack | None

    _client: ClientSession
    _read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    _write_stream: MemoryObjectSendStream[SessionMessage]

    def __init__(
        self,
        tool_prefix: str | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        log_handler: LoggingFnT | None = None,
        timeout: float = 5,
        read_timeout: float = 5 * 60,
        process_tool_call: ProcessToolCallback | None = None,
        allow_sampling: bool = True,
        sampling_model: models.Model | None = None,
        max_retries: int = 1,
        elicitation_callback: ElicitationFnT | None = None,
        *,
        id: str | None = None,
    ):
        self.tool_prefix = tool_prefix
        self.log_level = log_level
        self.log_handler = log_handler
        self.timeout = timeout
        self.read_timeout = read_timeout
        self.process_tool_call = process_tool_call
        self.allow_sampling = allow_sampling
        self.sampling_model = sampling_model
        self.max_retries = max_retries
        self.elicitation_callback = elicitation_callback

        self._id = id or tool_prefix

        self.__post_init__()

    def __post_init__(self):
        self._enter_lock = Lock()
        self._running_count = 0
        self._exit_stack = None

    @abstractmethod
    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        """Create the streams for the MCP server."""
        raise NotImplementedError('MCP Server subclasses must implement this method.')
        yield

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def label(self) -> str:
        if self.id:
            return super().label  # pragma: no cover
        else:
            return repr(self)

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Set the `tool_prefix` attribute to avoid name conflicts.'

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Retrieve tools that are currently active on the server.

        Note:
        - We don't cache tools as they might change.
        - We also don't subscribe to the server to avoid complexity.
        """
        async with self:  # Ensure server is running
            result = await self._client.list_tools()
        return result.tools

    async def direct_call_tool(
        self,
        name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Call a tool on the server.

        Args:
            name: The name of the tool to call.
            args: The arguments to pass to the tool.
            metadata: Request-level metadata (optional)

        Returns:
            The result of the tool call.

        Raises:
            ModelRetry: If the tool call fails.
        """
        async with self:  # Ensure server is running
            try:
                result = await self._client.send_request(
                    mcp_types.ClientRequest(
                        mcp_types.CallToolRequest(
                            method='tools/call',
                            params=mcp_types.CallToolRequestParams(
                                name=name,
                                arguments=args,
                                _meta=mcp_types.RequestParams.Meta(**metadata) if metadata else None,
                            ),
                        )
                    ),
                    mcp_types.CallToolResult,
                )
            except McpError as e:
                raise exceptions.ModelRetry(e.error.message)

        content = [await self._map_tool_result_part(part) for part in result.content]

        if result.isError:
            text = '\n'.join(str(part) for part in content)
            raise exceptions.ModelRetry(text)
        else:
            return content[0] if len(content) == 1 else content

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> ToolResult:
        if self.tool_prefix:
            name = name.removeprefix(f'{self.tool_prefix}_')
            ctx = replace(ctx, tool_name=name)

        if self.process_tool_call is not None:
            return await self.process_tool_call(ctx, self.direct_call_tool, name, tool_args)
        else:
            return await self.direct_call_tool(name, tool_args)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        return {
            name: self.tool_for_tool_def(
                ToolDefinition(
                    name=name,
                    description=mcp_tool.description,
                    parameters_json_schema=mcp_tool.inputSchema,
                    metadata={
                        'meta': mcp_tool.meta,
                        'annotations': mcp_tool.annotations.model_dump() if mcp_tool.annotations else None,
                        'output_schema': mcp_tool.outputSchema or None,
                    },
                ),
            )
            for mcp_tool in await self.list_tools()
            if (name := f'{self.tool_prefix}_{mcp_tool.name}' if self.tool_prefix else mcp_tool.name)
        }

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[Any]:
        return ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=self.max_retries,
            args_validator=TOOL_SCHEMA_VALIDATOR,
        )

    async def __aenter__(self) -> Self:
        """Enter the MCP server context.

        This will initialize the connection to the server.
        If this server is an [`MCPServerStdio`][pydantic_ai.mcp.MCPServerStdio], the server will first be started as a subprocess.

        This is a no-op if the MCP server has already been entered.
        """
        async with self._enter_lock:
            if self._running_count == 0:
                async with AsyncExitStack() as exit_stack:
                    self._read_stream, self._write_stream = await exit_stack.enter_async_context(self.client_streams())
                    client = ClientSession(
                        read_stream=self._read_stream,
                        write_stream=self._write_stream,
                        sampling_callback=self._sampling_callback if self.allow_sampling else None,
                        elicitation_callback=self.elicitation_callback,
                        logging_callback=self.log_handler,
                        read_timeout_seconds=timedelta(seconds=self.read_timeout),
                    )
                    self._client = await exit_stack.enter_async_context(client)

                    with anyio.fail_after(self.timeout):
                        await self._client.initialize()

                        if log_level := self.log_level:
                            await self._client.set_logging_level(log_level)

                    self._exit_stack = exit_stack.pop_all()
            self._running_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if self._running_count == 0:
            raise ValueError('MCPServer.__aexit__ called more times than __aenter__')
        async with self._enter_lock:
            self._running_count -= 1
            if self._running_count == 0 and self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None

    @property
    def is_running(self) -> bool:
        """Check if the MCP server is running."""
        return bool(self._running_count)

    async def _sampling_callback(
        self, context: RequestContext[ClientSession, Any], params: mcp_types.CreateMessageRequestParams
    ) -> mcp_types.CreateMessageResult | mcp_types.ErrorData:
        """MCP sampling callback."""
        if self.sampling_model is None:
            raise ValueError('Sampling model is not set')  # pragma: no cover

        pai_messages = _mcp.map_from_mcp_params(params)
        model_settings = models.ModelSettings()
        if max_tokens := params.maxTokens:  # pragma: no branch
            model_settings['max_tokens'] = max_tokens
        if temperature := params.temperature:  # pragma: no branch
            model_settings['temperature'] = temperature
        if stop_sequences := params.stopSequences:  # pragma: no branch
            model_settings['stop_sequences'] = stop_sequences

        model_response = await model_request(self.sampling_model, pai_messages, model_settings=model_settings)
        return mcp_types.CreateMessageResult(
            role='assistant',
            content=_mcp.map_from_model_response(model_response),
            model=self.sampling_model.model_name,
        )

    async def _map_tool_result_part(
        self, part: mcp_types.ContentBlock
    ) -> str | messages.BinaryContent | dict[str, Any] | list[Any]:
        # See https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#return-values

        if isinstance(part, mcp_types.TextContent):
            text = part.text
            if text.startswith(('[', '{')):
                try:
                    return pydantic_core.from_json(text)
                except ValueError:
                    pass
            return text
        elif isinstance(part, mcp_types.ImageContent):
            return messages.BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)
        elif isinstance(part, mcp_types.AudioContent):
            # NOTE: The FastMCP server doesn't support audio content.
            # See <https://github.com/modelcontextprotocol/python-sdk/issues/952> for more details.
            return messages.BinaryContent(
                data=base64.b64decode(part.data), media_type=part.mimeType
            )  # pragma: no cover
        elif isinstance(part, mcp_types.EmbeddedResource):
            resource = part.resource
            return self._get_content(resource)
        elif isinstance(part, mcp_types.ResourceLink):
            resource_result: mcp_types.ReadResourceResult = await self._client.read_resource(part.uri)
            return (
                self._get_content(resource_result.contents[0])
                if len(resource_result.contents) == 1
                else [self._get_content(resource) for resource in resource_result.contents]
            )
        else:
            assert_never(part)

    def _get_content(
        self, resource: mcp_types.TextResourceContents | mcp_types.BlobResourceContents
    ) -> str | messages.BinaryContent:
        if isinstance(resource, mcp_types.TextResourceContents):
            return resource.text
        elif isinstance(resource, mcp_types.BlobResourceContents):
            return messages.BinaryContent(
                data=base64.b64decode(resource.blob), media_type=resource.mimeType or 'application/octet-stream'
            )
        else:
            assert_never(resource)


class MCPServerStdio(MCPServer):
    """Runs an MCP server in a subprocess and communicates with it over stdin/stdout.

    This class implements the stdio transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio> for more information.

    !!! note
        Using this class as an async context manager will start the server as a subprocess when entering the context,
        and stop it when exiting the context.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio

    server = MCPServerStdio(  # (1)!
        'uv', args=['run', 'mcp-run-python', 'stdio'], timeout=10
    )
    agent = Agent('openai:gpt-4o', toolsets=[server])

    async def main():
        async with agent:  # (2)!
            ...
    ```

    1. See [MCP Run Python](https://github.com/pydantic/mcp-run-python) for more information.
    2. This will start the server as a subprocess and connect to it.
    """

    command: str
    """The command to run."""

    args: Sequence[str]
    """The arguments to pass to the command."""

    env: dict[str, str] | None
    """The environment variables the CLI server will have access to.

    By default the subprocess will not inherit any environment variables from the parent process.
    If you want to inherit the environment variables from the parent process, use `env=os.environ`.
    """

    cwd: str | Path | None
    """The working directory to use when spawning the process."""

    # last fields are re-defined from the parent class so they appear as fields
    tool_prefix: str | None
    log_level: mcp_types.LoggingLevel | None
    log_handler: LoggingFnT | None
    timeout: float
    read_timeout: float
    process_tool_call: ProcessToolCallback | None
    allow_sampling: bool
    sampling_model: models.Model | None
    max_retries: int
    elicitation_callback: ElicitationFnT | None = None

    def __init__(
        self,
        command: str,
        args: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        tool_prefix: str | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        log_handler: LoggingFnT | None = None,
        timeout: float = 5,
        read_timeout: float = 5 * 60,
        process_tool_call: ProcessToolCallback | None = None,
        allow_sampling: bool = True,
        sampling_model: models.Model | None = None,
        max_retries: int = 1,
        elicitation_callback: ElicitationFnT | None = None,
        id: str | None = None,
    ):
        """Build a new MCP server.

        Args:
            command: The command to run.
            args: The arguments to pass to the command.
            env: The environment variables to set in the subprocess.
            cwd: The working directory to use when spawning the process.
            tool_prefix: A prefix to add to all tools that are registered with the server.
            log_level: The log level to set when connecting to the server, if any.
            log_handler: A handler for logging messages from the server.
            timeout: The timeout in seconds to wait for the client to initialize.
            read_timeout: Maximum time in seconds to wait for new messages before timing out.
            process_tool_call: Hook to customize tool calling and optionally pass extra metadata.
            allow_sampling: Whether to allow MCP sampling through this client.
            sampling_model: The model to use for sampling.
            max_retries: The maximum number of times to retry a tool call.
            elicitation_callback: Callback function to handle elicitation requests from the server.
            id: An optional unique ID for the MCP server. An MCP server needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the server's activities within the workflow.
        """
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd

        super().__init__(
            tool_prefix,
            log_level,
            log_handler,
            timeout,
            read_timeout,
            process_tool_call,
            allow_sampling,
            sampling_model,
            max_retries,
            elicitation_callback,
            id=id,
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerStdio(**dct),
            core_schema.typed_dict_schema(
                {
                    'command': core_schema.typed_dict_field(core_schema.str_schema()),
                    'args': core_schema.typed_dict_field(core_schema.list_schema(core_schema.str_schema())),
                    'env': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()),
                        required=False,
                    ),
                }
            ),
        )

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        server = StdioServerParameters(command=self.command, args=list(self.args), env=self.env, cwd=self.cwd)
        async with stdio_client(server=server) as (read_stream, write_stream):
            yield read_stream, write_stream

    def __repr__(self) -> str:
        repr_args = [
            f'command={self.command!r}',
            f'args={self.args!r}',
        ]
        if self.id:
            repr_args.append(f'id={self.id!r}')
        return f'{self.__class__.__name__}({", ".join(repr_args)})'

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, MCPServerStdio):
            return False  # pragma: no cover
        return (
            self.command == value.command
            and self.args == value.args
            and self.env == value.env
            and self.cwd == value.cwd
        )


class _MCPServerHTTP(MCPServer):
    url: str
    """The URL of the endpoint on the MCP server."""

    headers: dict[str, Any] | None
    """Optional HTTP headers to be sent with each request to the endpoint.

    These headers will be passed directly to the underlying `httpx.AsyncClient`.
    Useful for authentication, custom headers, or other HTTP-specific configurations.

    !!! note
        You can either pass `headers` or `http_client`, but not both.

        See [`MCPServerHTTP.http_client`][pydantic_ai.mcp.MCPServerHTTP.http_client] for more information.
    """

    http_client: httpx.AsyncClient | None
    """An `httpx.AsyncClient` to use with the endpoint.

    This client may be configured to use customized connection parameters like self-signed certificates.

    !!! note
        You can either pass `headers` or `http_client`, but not both.

        If you want to use both, you can pass the headers to the `http_client` instead.

        ```python {py="3.10" test="skip"}
        import httpx

        from pydantic_ai.mcp import MCPServerSSE

        http_client = httpx.AsyncClient(headers={'Authorization': 'Bearer ...'})
        server = MCPServerSSE('http://localhost:3001/sse', http_client=http_client)
        ```
    """

    # last fields are re-defined from the parent class so they appear as fields
    tool_prefix: str | None
    log_level: mcp_types.LoggingLevel | None
    log_handler: LoggingFnT | None
    timeout: float
    read_timeout: float
    process_tool_call: ProcessToolCallback | None
    allow_sampling: bool
    sampling_model: models.Model | None
    max_retries: int
    elicitation_callback: ElicitationFnT | None = None

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        id: str | None = None,
        tool_prefix: str | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        log_handler: LoggingFnT | None = None,
        timeout: float = 5,
        read_timeout: float | None = None,
        process_tool_call: ProcessToolCallback | None = None,
        allow_sampling: bool = True,
        sampling_model: models.Model | None = None,
        max_retries: int = 1,
        elicitation_callback: ElicitationFnT | None = None,
        **_deprecated_kwargs: Any,
    ):
        """Build a new MCP server.

        Args:
            url: The URL of the endpoint on the MCP server.
            headers: Optional HTTP headers to be sent with each request to the endpoint.
            http_client: An `httpx.AsyncClient` to use with the endpoint.
            id: An optional unique ID for the MCP server. An MCP server needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the server's activities within the workflow.
            tool_prefix: A prefix to add to all tools that are registered with the server.
            log_level: The log level to set when connecting to the server, if any.
            log_handler: A handler for logging messages from the server.
            timeout: The timeout in seconds to wait for the client to initialize.
            read_timeout: Maximum time in seconds to wait for new messages before timing out.
            process_tool_call: Hook to customize tool calling and optionally pass extra metadata.
            allow_sampling: Whether to allow MCP sampling through this client.
            sampling_model: The model to use for sampling.
            max_retries: The maximum number of times to retry a tool call.
            elicitation_callback: Callback function to handle elicitation requests from the server.
        """
        if 'sse_read_timeout' in _deprecated_kwargs:
            if read_timeout is not None:
                raise TypeError("'read_timeout' and 'sse_read_timeout' cannot be set at the same time.")

            warnings.warn(
                "'sse_read_timeout' is deprecated, use 'read_timeout' instead.", DeprecationWarning, stacklevel=2
            )
            read_timeout = _deprecated_kwargs.pop('sse_read_timeout')

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        if read_timeout is None:
            read_timeout = 5 * 60

        self.url = url
        self.headers = headers
        self.http_client = http_client

        super().__init__(
            tool_prefix,
            log_level,
            log_handler,
            timeout,
            read_timeout,
            process_tool_call,
            allow_sampling,
            sampling_model,
            max_retries,
            elicitation_callback,
            id=id,
        )

    @property
    @abstractmethod
    def _transport_client(
        self,
    ) -> Callable[
        ...,
        AbstractAsyncContextManager[
            tuple[
                MemoryObjectReceiveStream[SessionMessage | Exception],
                MemoryObjectSendStream[SessionMessage],
                GetSessionIdCallback,
            ],
        ]
        | AbstractAsyncContextManager[
            tuple[
                MemoryObjectReceiveStream[SessionMessage | Exception],
                MemoryObjectSendStream[SessionMessage],
            ]
        ],
    ]: ...

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:  # pragma: no cover
        if self.http_client and self.headers:
            raise ValueError('`http_client` is mutually exclusive with `headers`.')

        transport_client_partial = functools.partial(
            self._transport_client,
            url=self.url,
            timeout=self.timeout,
            sse_read_timeout=self.read_timeout,
        )

        if self.http_client is not None:

            def httpx_client_factory(
                headers: dict[str, str] | None = None,
                timeout: httpx.Timeout | None = None,
                auth: httpx.Auth | None = None,
            ) -> httpx.AsyncClient:
                assert self.http_client is not None
                return self.http_client

            async with transport_client_partial(httpx_client_factory=httpx_client_factory) as (
                read_stream,
                write_stream,
                *_,
            ):
                yield read_stream, write_stream
        else:
            async with transport_client_partial(headers=self.headers) as (read_stream, write_stream, *_):
                yield read_stream, write_stream

    def __repr__(self) -> str:  # pragma: no cover
        repr_args = [
            f'url={self.url!r}',
        ]
        if self.id:
            repr_args.append(f'id={self.id!r}')
        return f'{self.__class__.__name__}({", ".join(repr_args)})'


class MCPServerSSE(_MCPServerHTTP):
    """An MCP server that connects over streamable HTTP connections.

    This class implements the SSE transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerSSE

    server = MCPServerSSE('http://localhost:3001/sse')
    agent = Agent('openai:gpt-4o', toolsets=[server])

    async def main():
        async with agent:  # (1)!
            ...
    ```

    1. This will connect to a server running on `localhost:3001`.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerSSE(**dct),
            core_schema.typed_dict_schema(
                {
                    'url': core_schema.typed_dict_field(core_schema.str_schema()),
                    'headers': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()), required=False
                    ),
                }
            ),
        )

    @property
    def _transport_client(self):
        return sse_client  # pragma: no cover

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, MCPServerSSE):
            return False  # pragma: no cover
        return self.url == value.url


@deprecated('The `MCPServerHTTP` class is deprecated, use `MCPServerSSE` instead.')
class MCPServerHTTP(MCPServerSSE):
    """An MCP server that connects over HTTP using the old SSE transport.

    This class implements the SSE transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10" test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerHTTP

    server = MCPServerHTTP('http://localhost:3001/sse')
    agent = Agent('openai:gpt-4o', toolsets=[server])

    async def main():
        async with agent:  # (2)!
            ...
    ```

    1. This will connect to a server running on `localhost:3001`.
    """


class MCPServerStreamableHTTP(_MCPServerHTTP):
    """An MCP server that connects over HTTP using the Streamable HTTP transport.

    This class implements the Streamable HTTP transport from the MCP specification.
    See <https://modelcontextprotocol.io/introduction#streamable-http> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStreamableHTTP

    server = MCPServerStreamableHTTP('http://localhost:8000/mcp')  # (1)!
    agent = Agent('openai:gpt-4o', toolsets=[server])

    async def main():
        async with agent:  # (2)!
            ...
    ```
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerStreamableHTTP(**dct),
            core_schema.typed_dict_schema(
                {
                    'url': core_schema.typed_dict_field(core_schema.str_schema()),
                    'headers': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()), required=False
                    ),
                }
            ),
        )

    @property
    def _transport_client(self):
        return streamablehttp_client  # pragma: no cover

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, MCPServerStreamableHTTP):
            return False  # pragma: no cover
        return self.url == value.url


ToolResult = (
    str
    | messages.BinaryContent
    | dict[str, Any]
    | list[Any]
    | Sequence[str | messages.BinaryContent | dict[str, Any] | list[Any]]
)
"""The result type of an MCP tool call."""

CallToolFunc = Callable[[str, dict[str, Any], dict[str, Any] | None], Awaitable[ToolResult]]
"""A function type that represents a tool call."""

ProcessToolCallback = Callable[
    [
        RunContext[Any],
        CallToolFunc,
        str,
        dict[str, Any],
    ],
    Awaitable[ToolResult],
]
"""A process tool callback.

It accepts a run context, the original tool call function, a tool name, and arguments.

Allows wrapping an MCP server tool call to customize it, including adding extra request
metadata.
"""


def _mcp_server_discriminator(value: dict[str, Any]) -> str | None:
    if 'url' in value:
        if value['url'].endswith('/sse'):
            return 'sse'
        return 'streamable-http'
    return 'stdio'


class MCPServerConfig(BaseModel):
    """Configuration for MCP servers."""

    mcp_servers: Annotated[
        dict[
            str,
            Annotated[
                Annotated[MCPServerStdio, Tag('stdio')]
                | Annotated[MCPServerStreamableHTTP, Tag('streamable-http')]
                | Annotated[MCPServerSSE, Tag('sse')],
                Discriminator(_mcp_server_discriminator),
            ],
        ],
        Field(alias='mcpServers'),
    ]


def load_mcp_servers(config_path: str | Path) -> list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE]:
    """Load MCP servers from a configuration file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A list of MCP servers.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValidationError: If the configuration file does not match the schema.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f'Config file {config_path} not found')

    config = MCPServerConfig.model_validate_json(config_path.read_bytes())
    return list(config.mcp_servers.values())



================================================
FILE: pydantic_ai_slim/pydantic_ai/output.py
================================================
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import TypeAliasType, TypeVar, deprecated

from . import _utils
from .messages import ToolCallPart
from .tools import DeferredToolRequests, RunContext, ToolDefinition

__all__ = (
    # classes
    'ToolOutput',
    'NativeOutput',
    'PromptedOutput',
    'TextOutput',
    'StructuredDict',
    # types
    'OutputDataT',
    'OutputMode',
    'StructuredOutputMode',
    'OutputSpec',
    'OutputTypeOrFunction',
    'TextOutputFunc',
)

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
"""Covariant type variable for the output data type of a run."""

OutputMode = Literal['text', 'tool', 'native', 'prompted', 'tool_or_text']
"""All output modes."""
StructuredOutputMode = Literal['tool', 'native', 'prompted']
"""Output modes that can be used for structured output. Used by ModelProfile.default_structured_output_mode"""


OutputTypeOrFunction = TypeAliasType(
    'OutputTypeOrFunction', type[T_co] | Callable[..., Awaitable[T_co] | T_co], type_params=(T_co,)
)
"""Definition of an output type or function.

You should not need to import or use this type directly.

See [output docs](../output.md) for more information.
"""


TextOutputFunc = TypeAliasType(
    'TextOutputFunc',
    Callable[[RunContext, str], Awaitable[T_co] | T_co] | Callable[[str], Awaitable[T_co] | T_co],
    type_params=(T_co,),
)
"""Definition of a function that will be called to process the model's plain text output. The function must take a single string argument.

You should not need to import or use this type directly.

See [text output docs](../output.md#text-output) for more information.
"""


@dataclass(init=False)
class ToolOutput(Generic[OutputDataT]):
    """Marker class to use a tool for output and optionally customize the tool.

    Example:
    ```python {title="tool_output.py"}
    from pydantic import BaseModel

    from pydantic_ai import Agent, ToolOutput


    class Fruit(BaseModel):
        name: str
        color: str


    class Vehicle(BaseModel):
        name: str
        wheels: int


    agent = Agent(
        'openai:gpt-4o',
        output_type=[
            ToolOutput(Fruit, name='return_fruit'),
            ToolOutput(Vehicle, name='return_vehicle'),
        ],
    )
    result = agent.run_sync('What is a banana?')
    print(repr(result.output))
    #> Fruit(name='banana', color='yellow')
    ```
    """

    output: OutputTypeOrFunction[OutputDataT]
    """An output type or function."""
    name: str | None
    """The name of the tool that will be passed to the model. If not specified and only one output is provided, `final_result` will be used. If multiple outputs are provided, the name of the output type or function will be added to the tool name."""
    description: str | None
    """The description of the tool that will be passed to the model. If not specified, the docstring of the output type or function will be used."""
    max_retries: int | None
    """The maximum number of retries for the tool."""
    strict: bool | None
    """Whether to use strict mode for the tool."""

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        max_retries: int | None = None,
        strict: bool | None = None,
    ):
        self.output = type_
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.strict = strict


@dataclass(init=False)
class NativeOutput(Generic[OutputDataT]):
    """Marker class to use the model's native structured outputs functionality for outputs and optionally customize the name and description.

    Example:
    ```python {title="native_output.py" requires="tool_output.py"}
    from pydantic_ai import Agent, NativeOutput

    from tool_output import Fruit, Vehicle

    agent = Agent(
        'openai:gpt-4o',
        output_type=NativeOutput(
            [Fruit, Vehicle],
            name='Fruit or vehicle',
            description='Return a fruit or vehicle.'
        ),
    )
    result = agent.run_sync('What is a Ford Explorer?')
    print(repr(result.output))
    #> Vehicle(name='Ford Explorer', wheels=4)
    ```
    """

    outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]]
    """The output types or functions."""
    name: str | None
    """The name of the structured output that will be passed to the model. If not specified and only one output is provided, the name of the output type or function will be used."""
    description: str | None
    """The description of the structured output that will be passed to the model. If not specified and only one output is provided, the docstring of the output type or function will be used."""
    strict: bool | None
    """Whether to use strict mode for the output, if the model supports it."""

    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        self.outputs = outputs
        self.name = name
        self.description = description
        self.strict = strict


@dataclass(init=False)
class PromptedOutput(Generic[OutputDataT]):
    """Marker class to use a prompt to tell the model what to output and optionally customize the prompt.

    Example:
    ```python {title="prompted_output.py" requires="tool_output.py"}
    from pydantic import BaseModel

    from pydantic_ai import Agent, PromptedOutput

    from tool_output import Vehicle


    class Device(BaseModel):
        name: str
        kind: str


    agent = Agent(
        'openai:gpt-4o',
        output_type=PromptedOutput(
            [Vehicle, Device],
            name='Vehicle or device',
            description='Return a vehicle or device.'
        ),
    )
    result = agent.run_sync('What is a MacBook?')
    print(repr(result.output))
    #> Device(name='MacBook', kind='laptop')

    agent = Agent(
        'openai:gpt-4o',
        output_type=PromptedOutput(
            [Vehicle, Device],
            template='Gimme some JSON: {schema}'
        ),
    )
    result = agent.run_sync('What is a Ford Explorer?')
    print(repr(result.output))
    #> Vehicle(name='Ford Explorer', wheels=4)
    ```
    """

    outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]]
    """The output types or functions."""
    name: str | None
    """The name of the structured output that will be passed to the model. If not specified and only one output is provided, the name of the output type or function will be used."""
    description: str | None
    """The description that will be passed to the model. If not specified and only one output is provided, the docstring of the output type or function will be used."""
    template: str | None
    """Template for the prompt passed to the model.
    The '{schema}' placeholder will be replaced with the output JSON schema.
    If not specified, the default template specified on the model's profile will be used.
    """

    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        template: str | None = None,
    ):
        self.outputs = outputs
        self.name = name
        self.description = description
        self.template = template


@dataclass
class TextOutput(Generic[OutputDataT]):
    """Marker class to use text output for an output function taking a string argument.

    Example:
    ```python
    from pydantic_ai import Agent, TextOutput


    def split_into_words(text: str) -> list[str]:
        return text.split()


    agent = Agent(
        'openai:gpt-4o',
        output_type=TextOutput(split_into_words),
    )
    result = agent.run_sync('Who was Albert Einstein?')
    print(result.output)
    #> ['Albert', 'Einstein', 'was', 'a', 'German-born', 'theoretical', 'physicist.']
    ```
    """

    output_function: TextOutputFunc[OutputDataT]
    """The function that will be called to process the model's plain text output. The function must take a single string argument."""


def StructuredDict(
    json_schema: JsonSchemaValue, name: str | None = None, description: str | None = None
) -> type[JsonSchemaValue]:
    """Returns a `dict[str, Any]` subclass with a JSON schema attached that will be used for structured output.

    Args:
        json_schema: A JSON schema of type `object` defining the structure of the dictionary content.
        name: Optional name of the structured output. If not provided, the `title` field of the JSON schema will be used if it's present.
        description: Optional description of the structured output. If not provided, the `description` field of the JSON schema will be used if it's present.

    Example:
    ```python {title="structured_dict.py"}
    from pydantic_ai import Agent, StructuredDict

    schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer'}
        },
        'required': ['name', 'age']
    }

    agent = Agent('openai:gpt-4o', output_type=StructuredDict(schema))
    result = agent.run_sync('Create a person')
    print(result.output)
    #> {'name': 'John Doe', 'age': 30}
    ```
    """
    json_schema = _utils.check_object_json_schema(json_schema)

    if name:
        json_schema['title'] = name

    if description:
        json_schema['description'] = description

    class _StructuredDict(JsonSchemaValue):
        __is_model_like__ = True

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.dict_schema(
                keys_schema=core_schema.str_schema(),
                values_schema=core_schema.any_schema(),
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            return json_schema

    return _StructuredDict


_OutputSpecItem = TypeAliasType(
    '_OutputSpecItem',
    OutputTypeOrFunction[T_co] | ToolOutput[T_co] | NativeOutput[T_co] | PromptedOutput[T_co] | TextOutput[T_co],
    type_params=(T_co,),
)

OutputSpec = TypeAliasType(
    'OutputSpec',
    _OutputSpecItem[T_co] | Sequence['OutputSpec[T_co]'],
    type_params=(T_co,),
)
"""Specification of the agent's output data.

This can be a single type, a function, a sequence of types and/or functions, or an instance of one of the output mode marker classes:
- [`ToolOutput`][pydantic_ai.output.ToolOutput]
- [`NativeOutput`][pydantic_ai.output.NativeOutput]
- [`PromptedOutput`][pydantic_ai.output.PromptedOutput]
- [`TextOutput`][pydantic_ai.output.TextOutput]

You should not need to import or use this type directly.

See [output docs](../output.md) for more information.
"""


@deprecated('`DeferredToolCalls` is deprecated, use `DeferredToolRequests` instead')
class DeferredToolCalls(DeferredToolRequests):  # pragma: no cover
    @property
    @deprecated('`DeferredToolCalls.tool_calls` is deprecated, use `DeferredToolRequests.calls` instead')
    def tool_calls(self) -> list[ToolCallPart]:
        return self.calls

    @property
    @deprecated('`DeferredToolCalls.tool_defs` is deprecated')
    def tool_defs(self) -> dict[str, ToolDefinition]:
        return {}



================================================
FILE: pydantic_ai_slim/pydantic_ai/py.typed
================================================
[Empty file]


================================================
FILE: pydantic_ai_slim/pydantic_ai/result.py
================================================
from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, cast, overload

from pydantic import ValidationError
from typing_extensions import TypeVar, deprecated

from . import _utils, exceptions, messages as _messages, models
from ._output import (
    OutputDataT_inv,
    OutputSchema,
    OutputValidator,
    OutputValidatorFunc,
    PlainTextOutputSchema,
    TextOutputSchema,
    ToolOutputSchema,
)
from ._run_context import AgentDepsT, RunContext
from ._tool_manager import ToolManager
from .messages import ModelResponseStreamEvent
from .output import (
    DeferredToolRequests,
    OutputDataT,
    ToolOutput,
)
from .run import AgentRunResult
from .usage import RunUsage, UsageLimits

__all__ = (
    'OutputDataT',
    'OutputDataT_inv',
    'ToolOutput',
    'OutputValidatorFunc',
)


T = TypeVar('T')
"""An invariant TypeVar."""


@dataclass(kw_only=True)
class AgentStream(Generic[AgentDepsT, OutputDataT]):
    _raw_stream_response: models.StreamedResponse
    _output_schema: OutputSchema[OutputDataT]
    _model_request_parameters: models.ModelRequestParameters
    _output_validators: list[OutputValidator[AgentDepsT, OutputDataT]]
    _run_ctx: RunContext[AgentDepsT]
    _usage_limits: UsageLimits | None
    _tool_manager: ToolManager[AgentDepsT]

    _agent_stream_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)
    _initial_run_ctx_usage: RunUsage = field(init=False)

    def __post_init__(self):
        self._initial_run_ctx_usage = deepcopy(self._run_ctx.usage)

    async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        """Asynchronously stream the (validated) agent outputs."""
        async for response in self.stream_responses(debounce_by=debounce_by):
            if self._raw_stream_response.final_result_event is not None:
                try:
                    yield await self.validate_response_output(response, allow_partial=True)
                except ValidationError:
                    pass
        if self._raw_stream_response.final_result_event is not None:  # pragma: no branch
            yield await self.validate_response_output(self._raw_stream_response.get())

    async def stream_responses(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[_messages.ModelResponse]:
        """Asynchronously stream the (unvalidated) model responses for the agent."""
        # if the message currently has any parts with content, yield before streaming
        msg = self._raw_stream_response.get()
        for part in msg.parts:
            if part.has_content():
                yield msg
                break

        async with _utils.group_by_temporal(self, debounce_by) as group_iter:
            async for _items in group_iter:
                yield self._raw_stream_response.get()  # current state of the response

    async def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]:
        """Stream the text result as an async iterable.

        !!! note
            Result validators will NOT be called on the text result if `delta=True`.

        Args:
            delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
                up to the current point.
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.
        """
        if not isinstance(self._output_schema, PlainTextOutputSchema):
            raise exceptions.UserError('stream_text() can only be used with text responses')

        if delta:
            async for text in self._stream_response_text(delta=True, debounce_by=debounce_by):
                yield text
        else:
            async for text in self._stream_response_text(delta=False, debounce_by=debounce_by):
                for validator in self._output_validators:
                    text = await validator.validate(text, self._run_ctx)  # pragma: no cover
                yield text

    def get(self) -> _messages.ModelResponse:
        """Get the current state of the response."""
        return self._raw_stream_response.get()

    def usage(self) -> RunUsage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        return self._initial_run_ctx_usage + self._raw_stream_response.usage()

    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._raw_stream_response.timestamp

    async def get_output(self) -> OutputDataT:
        """Stream the whole response, validate the output and return it."""
        async for _ in self:
            pass

        return await self.validate_response_output(self._raw_stream_response.get())

    async def validate_response_output(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        """Validate a structured result message."""
        final_result_event = self._raw_stream_response.final_result_event
        if final_result_event is None:
            raise exceptions.UnexpectedModelBehavior('Invalid response, unable to find output')  # pragma: no cover

        output_tool_name = final_result_event.tool_name

        if isinstance(self._output_schema, ToolOutputSchema) and output_tool_name is not None:
            tool_call = next(
                (
                    part
                    for part in message.parts
                    if isinstance(part, _messages.ToolCallPart) and part.tool_name == output_tool_name
                ),
                None,
            )
            if tool_call is None:
                raise exceptions.UnexpectedModelBehavior(  # pragma: no cover
                    f'Invalid response, unable to find tool call for {output_tool_name!r}'
                )
            return await self._tool_manager.handle_call(
                tool_call, allow_partial=allow_partial, wrap_validation_errors=False
            )
        elif deferred_tool_requests := _get_deferred_tool_requests(message.parts, self._tool_manager):
            if not self._output_schema.allows_deferred_tools:
                raise exceptions.UserError(
                    'A deferred tool call was present, but `DeferredToolRequests` is not among output types. To resolve this, add `DeferredToolRequests` to the list of output types for this agent.'
                )
            return cast(OutputDataT, deferred_tool_requests)
        elif isinstance(self._output_schema, TextOutputSchema):
            text = '\n\n'.join(x.content for x in message.parts if isinstance(x, _messages.TextPart))

            result_data = await self._output_schema.process(
                text, self._run_ctx, allow_partial=allow_partial, wrap_validation_errors=False
            )
            for validator in self._output_validators:
                result_data = await validator.validate(result_data, self._run_ctx)
            return result_data
        else:
            raise exceptions.UnexpectedModelBehavior(  # pragma: no cover
                'Invalid response, unable to process text output'
            )

    async def _stream_response_text(
        self, *, delta: bool = False, debounce_by: float | None = 0.1
    ) -> AsyncIterator[str]:
        """Stream the response as an async iterable of text."""

        # Define a "merged" version of the iterator that will yield items that have already been retrieved
        # and items that we receive while streaming. We define a dedicated async iterator for this so we can
        # pass the combined stream to the group_by_temporal function within `_stream_text_deltas` below.
        async def _stream_text_deltas_ungrouped() -> AsyncIterator[tuple[str, int]]:
            # yields tuples of (text_content, part_index)
            # we don't currently make use of the part_index, but in principle this may be useful
            # so we retain it here for now to make possible future refactors simpler
            msg = self._raw_stream_response.get()
            for i, part in enumerate(msg.parts):
                if isinstance(part, _messages.TextPart) and part.content:
                    yield part.content, i

            async for event in self._raw_stream_response:
                if (
                    isinstance(event, _messages.PartStartEvent)
                    and isinstance(event.part, _messages.TextPart)
                    and event.part.content
                ):
                    yield event.part.content, event.index  # pragma: no cover
                elif (  # pragma: no branch
                    isinstance(event, _messages.PartDeltaEvent)
                    and isinstance(event.delta, _messages.TextPartDelta)
                    and event.delta.content_delta
                ):
                    yield event.delta.content_delta, event.index

        async def _stream_text_deltas() -> AsyncIterator[str]:
            async with _utils.group_by_temporal(_stream_text_deltas_ungrouped(), debounce_by) as group_iter:
                async for items in group_iter:
                    # Note: we are currently just dropping the part index on the group here
                    yield ''.join([content for content, _ in items])

        if delta:
            async for text in _stream_text_deltas():
                yield text
        else:
            # a quick benchmark shows it's faster to build up a string with concat when we're
            # yielding at each step
            deltas: list[str] = []
            async for text in _stream_text_deltas():
                deltas.append(text)
                yield ''.join(deltas)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
        if self._agent_stream_iterator is None:
            self._agent_stream_iterator = _get_usage_checking_stream_response(
                self._raw_stream_response, self._usage_limits, self.usage
            )

        return self._agent_stream_iterator


@dataclass(init=False)
class StreamedRunResult(Generic[AgentDepsT, OutputDataT]):
    """Result of a streamed run that returns structured data via a tool call."""

    _all_messages: list[_messages.ModelMessage]
    _new_message_index: int

    _stream_response: AgentStream[AgentDepsT, OutputDataT] | None = None
    _on_complete: Callable[[], Awaitable[None]] | None = None

    _run_result: AgentRunResult[OutputDataT] | None = None

    is_complete: bool = field(default=False, init=False)
    """Whether the stream has all been received.

    This is set to `True` when one of
    [`stream_output`][pydantic_ai.result.StreamedRunResult.stream_output],
    [`stream_text`][pydantic_ai.result.StreamedRunResult.stream_text],
    [`stream_responses`][pydantic_ai.result.StreamedRunResult.stream_responses] or
    [`get_output`][pydantic_ai.result.StreamedRunResult.get_output] completes.
    """

    @overload
    def __init__(
        self,
        all_messages: list[_messages.ModelMessage],
        new_message_index: int,
        stream_response: AgentStream[AgentDepsT, OutputDataT] | None,
        on_complete: Callable[[], Awaitable[None]] | None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        all_messages: list[_messages.ModelMessage],
        new_message_index: int,
        *,
        run_result: AgentRunResult[OutputDataT],
    ) -> None: ...

    def __init__(
        self,
        all_messages: list[_messages.ModelMessage],
        new_message_index: int,
        stream_response: AgentStream[AgentDepsT, OutputDataT] | None = None,
        on_complete: Callable[[], Awaitable[None]] | None = None,
        run_result: AgentRunResult[OutputDataT] | None = None,
    ) -> None:
        self._all_messages = all_messages
        self._new_message_index = new_message_index

        self._stream_response = stream_response
        self._on_complete = on_complete
        self._run_result = run_result

    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        # this is a method to be consistent with the other methods
        if output_tool_return_content is not None:
            raise NotImplementedError('Setting output tool return content is not supported for this result type.')
        return self._all_messages

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResult.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(output_tool_return_content=output_tool_return_content)
        )

    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self.all_messages(output_tool_return_content=output_tool_return_content)[self._new_message_index :]

    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResult.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.new_messages(output_tool_return_content=output_tool_return_content)
        )

    @deprecated('`StreamedRunResult.stream` is deprecated, use `stream_output` instead.')
    async def stream(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        async for output in self.stream_output(debounce_by=debounce_by):
            yield output

    async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        """Stream the output as an async iterable.

        The pydantic validator for structured data will be called in
        [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
        on each iteration.

        Args:
            debounce_by: by how much (if at all) to debounce/group the output chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured outputs to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An async iterable of the response data.
        """
        if self._run_result is not None:
            yield self._run_result.output
            await self._marked_completed()
        elif self._stream_response is not None:
            async for output in self._stream_response.stream_output(debounce_by=debounce_by):
                yield output
            await self._marked_completed(self._stream_response.get())
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    async def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]:
        """Stream the text result as an async iterable.

        !!! note
            Result validators will NOT be called on the text result if `delta=True`.

        Args:
            delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
                up to the current point.
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.
        """
        if self._run_result is not None:  # pragma: no cover
            # We can't really get here, as `_run_result` is only set in `run_stream` when `CallToolsNode` produces `DeferredToolRequests` output
            # as a result of a tool function raising `CallDeferred` or `ApprovalRequired`.
            # That'll change if we ever support something like `raise EndRun(output: OutputT)` where `OutputT` could be `str`.
            if not isinstance(self._run_result.output, str):
                raise exceptions.UserError('stream_text() can only be used with text responses')
            yield self._run_result.output
            await self._marked_completed()
        elif self._stream_response is not None:
            async for text in self._stream_response.stream_text(delta=delta, debounce_by=debounce_by):
                yield text
            await self._marked_completed(self._stream_response.get())
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @deprecated('`StreamedRunResult.stream_structured` is deprecated, use `stream_responses` instead.')
    async def stream_structured(
        self, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
        async for msg, last in self.stream_responses(debounce_by=debounce_by):
            yield msg, last

    async def stream_responses(
        self, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
        """Stream the response as an async iterable of Structured LLM Messages.

        Args:
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An async iterable of the structured response message and whether that is the last message.
        """
        if self._run_result is not None:
            model_response = cast(_messages.ModelResponse, self.all_messages()[-1])
            yield model_response, True
            await self._marked_completed()
        elif self._stream_response is not None:
            # if the message currently has any parts with content, yield before streaming
            async for msg in self._stream_response.stream_responses(debounce_by=debounce_by):
                yield msg, False

            msg = self._stream_response.get()
            yield msg, True

            await self._marked_completed(msg)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    async def get_output(self) -> OutputDataT:
        """Stream the whole response, validate and return it."""
        if self._run_result is not None:
            output = self._run_result.output
            await self._marked_completed()
            return output
        elif self._stream_response is not None:
            output = await self._stream_response.get_output()
            await self._marked_completed(self._stream_response.get())
            return output
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    def usage(self) -> RunUsage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        if self._run_result is not None:
            return self._run_result.usage()
        elif self._stream_response is not None:
            return self._stream_response.usage()
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        if self._run_result is not None:
            return self._run_result.timestamp()
        elif self._stream_response is not None:
            return self._stream_response.timestamp()
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @deprecated('`validate_structured_output` is deprecated, use `validate_response_output` instead.')
    async def validate_structured_output(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        return await self.validate_response_output(message, allow_partial=allow_partial)

    async def validate_response_output(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        """Validate a structured result message."""
        if self._run_result is not None:
            return self._run_result.output
        elif self._stream_response is not None:
            return await self._stream_response.validate_response_output(message, allow_partial=allow_partial)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    async def _marked_completed(self, message: _messages.ModelResponse | None = None) -> None:
        self.is_complete = True
        if message is not None:
            self._all_messages.append(message)
        if self._on_complete is not None:
            await self._on_complete()


@dataclass(repr=False)
class FinalResult(Generic[OutputDataT]):
    """Marker class storing the final output of an agent run and associated metadata."""

    output: OutputDataT
    """The final result data."""

    tool_name: str | None = None
    """Name of the final output tool; `None` if the output came from unstructured text content."""

    tool_call_id: str | None = None
    """ID of the tool call that produced the final output; `None` if the output came from unstructured text content."""

    __repr__ = _utils.dataclasses_no_defaults_repr


def _get_usage_checking_stream_response(
    stream_response: models.StreamedResponse,
    limits: UsageLimits | None,
    get_usage: Callable[[], RunUsage],
) -> AsyncIterator[ModelResponseStreamEvent]:
    if limits is not None and limits.has_token_limits():

        async def _usage_checking_iterator():
            async for item in stream_response:
                limits.check_tokens(get_usage())
                yield item

        return _usage_checking_iterator()
    else:
        return aiter(stream_response)


def _get_deferred_tool_requests(
    parts: Iterable[_messages.ModelResponsePart], tool_manager: ToolManager[AgentDepsT]
) -> DeferredToolRequests | None:
    """Get the deferred tool requests from the model response parts."""
    approvals: list[_messages.ToolCallPart] = []
    calls: list[_messages.ToolCallPart] = []

    for part in parts:
        if isinstance(part, _messages.ToolCallPart):
            tool_def = tool_manager.get_tool_def(part.tool_name)
            if tool_def is not None:  # pragma: no branch
                if tool_def.kind == 'unapproved':
                    approvals.append(part)
                elif tool_def.kind == 'external':
                    calls.append(part)

    if not calls and not approvals:
        return None

    return DeferredToolRequests(calls=calls, approvals=approvals)



================================================
FILE: pydantic_ai_slim/pydantic_ai/retries.py
================================================
"""Retries utilities based on tenacity, especially for HTTP requests.

This module provides HTTP transport wrappers and wait strategies that integrate with
the tenacity library to add retry capabilities to HTTP requests. The transports can be
used with HTTP clients that support custom transports (such as httpx), while the wait
strategies can be used with any tenacity retry decorator.

The module includes:
- TenacityTransport: Synchronous HTTP transport with retry capabilities
- AsyncTenacityTransport: Asynchronous HTTP transport with retry capabilities
- wait_retry_after: Wait strategy that respects HTTP Retry-After headers
"""

from __future__ import annotations

from types import TracebackType

from httpx import (
    AsyncBaseTransport,
    AsyncHTTPTransport,
    BaseTransport,
    HTTPStatusError,
    HTTPTransport,
    Request,
    Response,
)

try:
    from tenacity import RetryCallState, RetryError, retry, wait_exponential
except ImportError as _import_error:
    raise ImportError(
        'Please install `tenacity` to use the retries utilities, '
        'you can use the `retries` optional group — `pip install "pydantic-ai-slim[retries]"`'
    ) from _import_error

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from tenacity.asyncio.retry import RetryBaseT
    from tenacity.retry import RetryBaseT as SyncRetryBaseT
    from tenacity.stop import StopBaseT
    from tenacity.wait import WaitBaseT

__all__ = ['RetryConfig', 'TenacityTransport', 'AsyncTenacityTransport', 'wait_retry_after']


class RetryConfig(TypedDict, total=False):
    """The configuration for tenacity-based retrying.

    These are precisely the arguments to the tenacity `retry` decorator, and they are generally
    used internally by passing them to that decorator via `@retry(**config)` or similar.

    All fields are optional, and if not provided, the default values from the `tenacity.retry` decorator will be used.
    """

    sleep: Callable[[int | float], None | Awaitable[None]]
    """A sleep strategy to use for sleeping between retries.

    Tenacity's default for this argument is `tenacity.nap.sleep`."""

    stop: StopBaseT
    """
    A stop strategy to determine when to stop retrying.

    Tenacity's default for this argument is `tenacity.stop.stop_never`."""

    wait: WaitBaseT
    """
    A wait strategy to determine how long to wait between retries.

    Tenacity's default for this argument is `tenacity.wait.wait_none`."""

    retry: SyncRetryBaseT | RetryBaseT
    """A retry strategy to determine which exceptions should trigger a retry.

    Tenacity's default for this argument is `tenacity.retry.retry_if_exception_type()`."""

    before: Callable[[RetryCallState], None | Awaitable[None]]
    """
    A callable that is called before each retry attempt.

    Tenacity's default for this argument is `tenacity.before.before_nothing`."""

    after: Callable[[RetryCallState], None | Awaitable[None]]
    """
    A callable that is called after each retry attempt.

    Tenacity's default for this argument is `tenacity.after.after_nothing`."""

    before_sleep: Callable[[RetryCallState], None | Awaitable[None]] | None
    """
    An optional callable that is called before sleeping between retries.

    Tenacity's default for this argument is `None`."""

    reraise: bool
    """Whether to reraise the last exception if the retry attempts are exhausted, or raise a RetryError instead.

    Tenacity's default for this argument is `False`."""

    retry_error_cls: type[RetryError]
    """The exception class to raise when the retry attempts are exhausted and `reraise` is False.

    Tenacity's default for this argument is `tenacity.RetryError`."""

    retry_error_callback: Callable[[RetryCallState], Any | Awaitable[Any]] | None
    """An optional callable that is called when the retry attempts are exhausted and `reraise` is False.

    Tenacity's default for this argument is `None`."""


class TenacityTransport(BaseTransport):
    """Synchronous HTTP transport with tenacity-based retry functionality.

    This transport wraps another BaseTransport and adds retry capabilities using the tenacity library.
    It can be configured to retry requests based on various conditions such as specific exception types,
    response status codes, or custom validation logic.

    The transport works by intercepting HTTP requests and responses, allowing the tenacity controller
    to determine when and how to retry failed requests. The validate_response function can be used
    to convert HTTP responses into exceptions that trigger retries.

    Args:
        wrapped: The underlying transport to wrap and add retry functionality to.
        config: The arguments to use for the tenacity `retry` decorator, including retry conditions,
            wait strategy, stop conditions, etc. See the tenacity docs for more info.
        validate_response: Optional callable that takes a Response and can raise an exception
            to be handled by the controller if the response should trigger a retry.
            Common use case is to raise exceptions for certain HTTP status codes.
            If None, no response validation is performed.

    Example:
        ```python
        from httpx import Client, HTTPStatusError, HTTPTransport
        from tenacity import retry_if_exception_type, stop_after_attempt

        from pydantic_ai.retries import RetryConfig, TenacityTransport, wait_retry_after

        transport = TenacityTransport(
            RetryConfig(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=300),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            HTTPTransport(),
            validate_response=lambda r: r.raise_for_status()
        )
        client = Client(transport=transport)
        ```
    """

    def __init__(
        self,
        config: RetryConfig,
        wrapped: BaseTransport | None = None,
        validate_response: Callable[[Response], Any] | None = None,
    ):
        self.config = config
        self.wrapped = wrapped or HTTPTransport()
        self.validate_response = validate_response

    def handle_request(self, request: Request) -> Response:
        """Handle an HTTP request with retry logic.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.

        Raises:
            RuntimeError: If the retry controller did not make any attempts.
            Exception: Any exception raised by the wrapped transport or validation function.
        """

        @retry(**self.config)
        def handle_request(req: Request) -> Response:
            response = self.wrapped.handle_request(req)

            # this is normally set by httpx _after_ calling this function, but we want the request in the validator:
            response.request = req

            if self.validate_response:
                try:
                    self.validate_response(response)
                except Exception:
                    response.close()
                    raise
            return response

        return handle_request(request)

    def __enter__(self) -> TenacityTransport:
        self.wrapped.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        self.wrapped.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        self.wrapped.close()  # pragma: no cover


class AsyncTenacityTransport(AsyncBaseTransport):
    """Asynchronous HTTP transport with tenacity-based retry functionality.

    This transport wraps another AsyncBaseTransport and adds retry capabilities using the tenacity library.
    It can be configured to retry requests based on various conditions such as specific exception types,
    response status codes, or custom validation logic.

    The transport works by intercepting HTTP requests and responses, allowing the tenacity controller
    to determine when and how to retry failed requests. The validate_response function can be used
    to convert HTTP responses into exceptions that trigger retries.

    Args:
        wrapped: The underlying async transport to wrap and add retry functionality to.
        config: The arguments to use for the tenacity `retry` decorator, including retry conditions,
            wait strategy, stop conditions, etc. See the tenacity docs for more info.
        validate_response: Optional callable that takes a Response and can raise an exception
            to be handled by the controller if the response should trigger a retry.
            Common use case is to raise exceptions for certain HTTP status codes.
            If None, no response validation is performed.

    Example:
        ```python
        from httpx import AsyncClient, HTTPStatusError
        from tenacity import retry_if_exception_type, stop_after_attempt

        from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

        transport = AsyncTenacityTransport(
            RetryConfig(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=300),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            validate_response=lambda r: r.raise_for_status()
        )
        client = AsyncClient(transport=transport)
        ```
    """

    def __init__(
        self,
        config: RetryConfig,
        wrapped: AsyncBaseTransport | None = None,
        validate_response: Callable[[Response], Any] | None = None,
    ):
        self.config = config
        self.wrapped = wrapped or AsyncHTTPTransport()
        self.validate_response = validate_response

    async def handle_async_request(self, request: Request) -> Response:
        """Handle an async HTTP request with retry logic.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.

        Raises:
            RuntimeError: If the retry controller did not make any attempts.
            Exception: Any exception raised by the wrapped transport or validation function.
        """

        @retry(**self.config)
        async def handle_async_request(req: Request) -> Response:
            response = await self.wrapped.handle_async_request(req)

            # this is normally set by httpx _after_ calling this function, but we want the request in the validator:
            response.request = req

            if self.validate_response:
                try:
                    self.validate_response(response)
                except Exception:
                    await response.aclose()
                    raise
            return response

        return await handle_async_request(request)

    async def __aenter__(self) -> AsyncTenacityTransport:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        await self.wrapped.__aexit__(exc_type, exc_value, traceback)

    async def aclose(self) -> None:
        await self.wrapped.aclose()


def wait_retry_after(
    fallback_strategy: Callable[[RetryCallState], float] | None = None, max_wait: float = 300
) -> Callable[[RetryCallState], float]:
    """Create a tenacity-compatible wait strategy that respects HTTP Retry-After headers.

    This wait strategy checks if the exception contains an HTTPStatusError with a
    Retry-After header, and if so, waits for the time specified in the header.
    If no header is present or parsing fails, it falls back to the provided strategy.

    The Retry-After header can be in two formats:
    - An integer representing seconds to wait
    - An HTTP date string representing when to retry

    Args:
        fallback_strategy: Wait strategy to use when no Retry-After header is present
                          or parsing fails. Defaults to exponential backoff with max 60s.
        max_wait: Maximum time to wait in seconds, regardless of header value.
                 Defaults to 300 (5 minutes).

    Returns:
        A wait function that can be used with tenacity retry decorators.

    Example:
        ```python
        from httpx import AsyncClient, HTTPStatusError
        from tenacity import retry_if_exception_type, stop_after_attempt

        from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

        transport = AsyncTenacityTransport(
            RetryConfig(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=120),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            validate_response=lambda r: r.raise_for_status()
        )
        client = AsyncClient(transport=transport)
        ```
    """
    if fallback_strategy is None:
        fallback_strategy = wait_exponential(multiplier=1, max=60)

    def wait_func(state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, HTTPStatusError):
            retry_after = exc.response.headers.get('retry-after')
            if retry_after:
                try:
                    # Try parsing as seconds first
                    wait_seconds = int(retry_after)
                    return min(float(wait_seconds), max_wait)
                except ValueError:
                    # Try parsing as HTTP date
                    try:
                        retry_time = cast(datetime, parsedate_to_datetime(retry_after))
                        assert isinstance(retry_time, datetime)
                        now = datetime.now(timezone.utc)
                        wait_seconds = (retry_time - now).total_seconds()

                        if wait_seconds > 0:
                            return min(wait_seconds, max_wait)
                    except (ValueError, TypeError, AssertionError):
                        # If date parsing fails, fall back to fallback strategy
                        pass

        # Use fallback strategy
        return fallback_strategy(state)

    return wait_func



================================================
FILE: pydantic_ai_slim/pydantic_ai/run.py
================================================
from __future__ import annotations as _annotations

import dataclasses
from collections.abc import AsyncIterator
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Literal, overload

from pydantic_graph import End, GraphRun, GraphRunContext

from . import (
    _agent_graph,
    exceptions,
    messages as _messages,
    usage as _usage,
)
from .output import OutputDataT
from .tools import AgentDepsT

if TYPE_CHECKING:
    from .result import FinalResult


@dataclasses.dataclass(repr=False)
class AgentRun(Generic[AgentDepsT, OutputDataT]):
    """A stateful, async-iterable run of an [`Agent`][pydantic_ai.agent.Agent].

    You generally obtain an `AgentRun` instance by calling `async with my_agent.iter(...) as agent_run:`.

    Once you have an instance, you can use it to iterate through the run's nodes as they execute. When an
    [`End`][pydantic_graph.nodes.End] is reached, the run finishes and [`result`][pydantic_ai.agent.AgentRun.result]
    becomes available.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
        # Iterate through the run, recording each node along the way:
        async with agent.iter('What is the capital of France?') as agent_run:
            async for node in agent_run:
                nodes.append(node)
        print(nodes)
        '''
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                instructions_functions=[],
                system_prompts=(),
                system_prompt_functions=[],
                system_prompt_dynamic_functions={},
            ),
            ModelRequestNode(
                request=ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=datetime.datetime(...),
                        )
                    ]
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='The capital of France is Paris.')],
                    usage=RequestUsage(input_tokens=56, output_tokens=7),
                    model_name='gpt-4o',
                    timestamp=datetime.datetime(...),
                )
            ),
            End(data=FinalResult(output='The capital of France is Paris.')),
        ]
        '''
        print(agent_run.result.output)
        #> The capital of France is Paris.
    ```

    You can also manually drive the iteration using the [`next`][pydantic_ai.agent.AgentRun.next] method for
    more granular control.
    """

    _graph_run: GraphRun[
        _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[OutputDataT]
    ]

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        traceparent = self._graph_run._traceparent(required=False)  # type: ignore[reportPrivateUsage]
        if traceparent is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return traceparent

    @property
    def ctx(self) -> GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]]:
        """The current context of the agent run."""
        return GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]](
            state=self._graph_run.state, deps=self._graph_run.deps
        )

    @property
    def next_node(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """The next node that will be run in the agent graph.

        This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
        """
        next_node = self._graph_run.next_node
        if isinstance(next_node, End):
            return next_node
        if _agent_graph.is_agent_node(next_node):
            return next_node
        raise exceptions.AgentRunError(f'Unexpected node type: {type(next_node)}')  # pragma: no cover

    @property
    def result(self) -> AgentRunResult[OutputDataT] | None:
        """The final result of the run if it has ended, otherwise `None`.

        Once the run returns an [`End`][pydantic_graph.nodes.End] node, `result` is populated
        with an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
        """
        graph_run_result = self._graph_run.result
        if graph_run_result is None:
            return None
        return AgentRunResult(
            graph_run_result.output.output,
            graph_run_result.output.tool_name,
            graph_run_result.state,
            self._graph_run.deps.new_message_index,
            self._traceparent(required=False),
        )

    def __aiter__(
        self,
    ) -> AsyncIterator[_agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]]:
        """Provide async-iteration over the nodes in the agent run."""
        return self

    async def __anext__(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Advance to the next node automatically based on the last returned node."""
        next_node = await self._graph_run.__anext__()
        if _agent_graph.is_agent_node(node=next_node):
            return next_node
        assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
        return next_node

    async def next(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, OutputDataT],
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Manually drive the agent run by passing in the node you want to run next.

        This lets you inspect or mutate the node before continuing execution, or skip certain nodes
        under dynamic conditions. The agent run should be stopped when you return an [`End`][pydantic_graph.nodes.End]
        node.

        Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_graph import End

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.iter('What is the capital of France?') as agent_run:
                next_node = agent_run.next_node  # start with the first node
                nodes = [next_node]
                while not isinstance(next_node, End):
                    next_node = await agent_run.next(next_node)
                    nodes.append(next_node)
                # Once `next_node` is an End, we've finished:
                print(nodes)
                '''
                [
                    UserPromptNode(
                        user_prompt='What is the capital of France?',
                        instructions_functions=[],
                        system_prompts=(),
                        system_prompt_functions=[],
                        system_prompt_dynamic_functions={},
                    ),
                    ModelRequestNode(
                        request=ModelRequest(
                            parts=[
                                UserPromptPart(
                                    content='What is the capital of France?',
                                    timestamp=datetime.datetime(...),
                                )
                            ]
                        )
                    ),
                    CallToolsNode(
                        model_response=ModelResponse(
                            parts=[TextPart(content='The capital of France is Paris.')],
                            usage=RequestUsage(input_tokens=56, output_tokens=7),
                            model_name='gpt-4o',
                            timestamp=datetime.datetime(...),
                        )
                    ),
                    End(data=FinalResult(output='The capital of France is Paris.')),
                ]
                '''
                print('Final result:', agent_run.result.output)
                #> Final result: The capital of France is Paris.
        ```

        Args:
            node: The node to run next in the graph.

        Returns:
            The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
            the run has completed.
        """
        # Note: It might be nice to expose a synchronous interface for iteration, but we shouldn't do it
        # on this class, or else IDEs won't warn you if you accidentally use `for` instead of `async for` to iterate.
        next_node = await self._graph_run.next(node)
        if _agent_graph.is_agent_node(next_node):
            return next_node
        assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
        return next_node

    def usage(self) -> _usage.RunUsage:
        """Get usage statistics for the run so far, including token usage, model requests, and so on."""
        return self._graph_run.state.usage

    def __repr__(self) -> str:  # pragma: no cover
        result = self._graph_run.result
        result_repr = '<run not finished>' if result is None else repr(result.output)
        return f'<{type(self).__name__} result={result_repr} usage={self.usage()}>'


@dataclasses.dataclass
class AgentRunResult(Generic[OutputDataT]):
    """The final result of an agent run."""

    output: OutputDataT
    """The output data from the agent run."""

    _output_tool_name: str | None = dataclasses.field(repr=False)
    _state: _agent_graph.GraphAgentState = dataclasses.field(repr=False)
    _new_message_index: int = dataclasses.field(repr=False)
    _traceparent_value: str | None = dataclasses.field(repr=False)

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        if self._traceparent_value is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return self._traceparent_value

    def _set_output_tool_return(self, return_content: str) -> list[_messages.ModelMessage]:
        """Set return content for the output tool.

        Useful if you want to continue the conversation and want to set the response to the output tool call.
        """
        if not self._output_tool_name:
            raise ValueError('Cannot set output tool return content when the return type is `str`.')

        messages = self._state.message_history
        last_message = messages[-1]
        for idx, part in enumerate(last_message.parts):
            if isinstance(part, _messages.ToolReturnPart) and part.tool_name == self._output_tool_name:
                # Only do deepcopy when we have to modify
                copied_messages = list(messages)
                copied_last = deepcopy(last_message)
                copied_last.parts[idx].content = return_content  # type: ignore[misc]
                copied_messages[-1] = copied_last
                return copied_messages

        raise LookupError(f'No tool call found with tool name {self._output_tool_name!r}.')

    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        if output_tool_return_content is not None:
            return self._set_output_tool_return(output_tool_return_content)
        else:
            return self._state.message_history

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRunResult.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(output_tool_return_content=output_tool_return_content)
        )

    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                Th