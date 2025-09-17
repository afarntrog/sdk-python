"""This module implements the central event loop.

The event loop allows agents to:

1. Process conversation messages
2. Execute tools based on model requests
3. Handle errors and recovery strategies
4. Manage recursive execution cycles
"""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from opentelemetry import trace as trace_api

from ..experimental.hooks import (
    AfterModelInvocationEvent,
    BeforeModelInvocationEvent,
)
from ..hooks import (
    MessageAddedEvent,
)
from ..telemetry.metrics import Trace
from ..telemetry.tracer import get_tracer
from ..tools._validator import validate_and_prepare_tools
from ..types._events import (
    EventLoopStopEvent,
    EventLoopThrottleEvent,
    ForceStopEvent,
    ModelMessageEvent,
    ModelStopReason,
    StartEvent,
    StartEventLoopEvent,
    StructuredOutputEvent,
    ToolResultMessageEvent,
    TypedEvent,
)
from ..types.content import Message
from ..types.exceptions import (
    ContextWindowOverflowException,
    EventLoopException,
    MaxTokensReachedException,
    ModelThrottledException,
)
from ..types.streaming import Metrics, StopReason
from ..types.tools import ToolResult, ToolUse
from ._recover_message_on_max_tokens_reached import recover_message_on_max_tokens_reached
from .streaming import stream_messages

if TYPE_CHECKING:
    from ..agent import Agent
    from ..output.base import OutputSchema

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
INITIAL_DELAY = 4
MAX_DELAY = 240  # 4 minutes


async def event_loop_cycle(
    agent: "Agent", invocation_state: dict[str, Any], output_schema: Optional["OutputSchema"] = None
) -> AsyncGenerator[TypedEvent, None]:
    """Execute a single cycle of the event loop.

    This core function processes a single conversation turn, handling model inference, tool execution, and error
    recovery. It manages the entire lifecycle of a conversation turn, including:

    1. Initializing cycle state and metrics
    2. Checking execution limits
    3. Processing messages with the model
    4. Handling tool execution requests
    5. Managing recursive calls for multi-turn tool interactions
    6. Collecting and reporting metrics
    7. Error handling and recovery

    Args:
        agent: The agent for which the cycle is being executed.
        invocation_state: Additional arguments including:

            - request_state: State maintained across cycles
            - event_loop_cycle_id: Unique ID for this cycle
            - event_loop_cycle_span: Current tracing Span for this cycle
        output_schema: Optional output schema for structured output processing.
            When provided, the event loop will register structured output tools
            and handle response processing.

    Yields:
        Model and tool stream events. The last event is a tuple containing:

            - StopReason: Reason the model stopped generating (e.g., "tool_use")
            - Message: The generated message from the model
            - EventLoopMetrics: Updated metrics for the event loop
            - Any: Updated request state

    Raises:
        EventLoopException: If an error occurs during execution
        ContextWindowOverflowException: If the input is too large for the model
    """
    # Initialize cycle state
    invocation_state["event_loop_cycle_id"] = uuid.uuid4()

    # Initialize state and get cycle trace
    if "request_state" not in invocation_state:
        invocation_state["request_state"] = {}
    attributes = {"event_loop_cycle_id": str(invocation_state.get("event_loop_cycle_id"))}
    cycle_start_time, cycle_trace = agent.event_loop_metrics.start_cycle(attributes=attributes)
    invocation_state["event_loop_cycle_trace"] = cycle_trace

    yield StartEvent()
    yield StartEventLoopEvent()

    # Create tracer span for this event loop cycle
    tracer = get_tracer()
    cycle_span = tracer.start_event_loop_cycle_span(
        invocation_state=invocation_state, messages=agent.messages, parent_span=agent.trace_span
    )
    invocation_state["event_loop_cycle_span"] = cycle_span

    # Create a trace for the stream_messages call
    stream_trace = Trace("stream_messages", parent_id=cycle_trace.id)
    cycle_trace.add_child(stream_trace)

    # Process messages with exponential backoff for throttling
    message: Message
    stop_reason: StopReason
    usage: Any
    metrics: Metrics

    # Retry loop for handling throttling exceptions
    current_delay = INITIAL_DELAY
    for attempt in range(MAX_ATTEMPTS):
        model_id = agent.model.config.get("model_id") if hasattr(agent.model, "config") else None
        model_invoke_span = tracer.start_model_invoke_span(
            messages=agent.messages,
            parent_span=cycle_span,
            model_id=model_id,
        )
        with trace_api.use_span(model_invoke_span):
            agent.hooks.invoke_callbacks(
                BeforeModelInvocationEvent(
                    agent=agent,
                )
            )

            # Register structured output tools if output schema is provided
            structured_output_tools = []
            structured_output_tool_names = set()
            if output_schema:
                from ..output import get_global_registry

                registry = get_global_registry()
                try:
                    structured_output_tools = registry.get_tool_specs(output_schema)
                    structured_output_tool_names = {tool.get('name') for tool in structured_output_tools}
                    logger.debug(f"Registered {len(structured_output_tools)} structured output tools: {structured_output_tool_names}")
                except Exception as e:
                    logger.error(f"Failed to generate structured output tools: {e}")
                    # Continue without structured output tools rather than failing completely
                    structured_output_tools = []

            # Store structured output context in invocation state for later use
            invocation_state["output_schema"] = output_schema
            invocation_state["structured_output_tool_names"] = structured_output_tool_names

            # Get all regular tools plus structured output tools
            tool_specs = agent.tool_registry.get_all_tool_specs() + structured_output_tools

            try:
                async for event in stream_messages(agent.model, agent.system_prompt, agent.messages, tool_specs):
                    if not isinstance(event, ModelStopReason):
                        yield event

                stop_reason, message, usage, metrics = event["stop"]
                invocation_state.setdefault("request_state", {})

                agent.hooks.invoke_callbacks(
                    AfterModelInvocationEvent(
                        agent=agent,
                        stop_response=AfterModelInvocationEvent.ModelStopResponse(
                            stop_reason=stop_reason,
                            message=message,
                        ),
                    )
                )

                if stop_reason == "max_tokens":
                    message = recover_message_on_max_tokens_reached(message)

                if model_invoke_span:
                    tracer.end_model_invoke_span(model_invoke_span, message, usage, stop_reason)
                break  # Success! Break out of retry loop

            except Exception as e:
                if model_invoke_span:
                    tracer.end_span_with_error(model_invoke_span, str(e), e)

                agent.hooks.invoke_callbacks(
                    AfterModelInvocationEvent(
                        agent=agent,
                        exception=e,
                    )
                )

                if isinstance(e, ModelThrottledException):
                    if attempt + 1 == MAX_ATTEMPTS:
                        yield ForceStopEvent(reason=e)
                        raise e

                    logger.debug(
                        "retry_delay_seconds=<%s>, max_attempts=<%s>, current_attempt=<%s> "
                        "| throttling exception encountered "
                        "| delaying before next retry",
                        current_delay,
                        MAX_ATTEMPTS,
                        attempt + 1,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * 2, MAX_DELAY)

                    yield EventLoopThrottleEvent(delay=current_delay)
                else:
                    raise e

    try:
        # Add message in trace and mark the end of the stream messages trace
        stream_trace.add_message(message)
        stream_trace.end()

        # Add the response message to the conversation
        agent.messages.append(message)
        agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=message))
        yield ModelMessageEvent(message=message)

        # Update metrics
        agent.event_loop_metrics.update_usage(usage)
        agent.event_loop_metrics.update_metrics(metrics)

        if stop_reason == "max_tokens":
            """
            Handle max_tokens limit reached by the model.

            When the model reaches its maximum token limit, this represents a potentially unrecoverable
            state where the model's response was truncated. By default, Strands fails hard with an
            MaxTokensReachedException to maintain consistency with other failure types.
            """
            raise MaxTokensReachedException(
                message=(
                    "Agent has reached an unrecoverable state due to max_tokens limit. "
                    "For more information see: "
                    "https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/#maxtokensreachedexception"
                )
            )

        # If the model is requesting to use tools
        if stop_reason == "tool_use":
            # Handle tool execution
            events = _handle_tool_execution(
                stop_reason,
                message,
                agent=agent,
                cycle_trace=cycle_trace,
                cycle_span=cycle_span,
                cycle_start_time=cycle_start_time,
                invocation_state=invocation_state,
            )
            async for typed_event in events:
                yield typed_event

            return

        # End the cycle and return results
        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace, attributes)
        if cycle_span:
            tracer.end_event_loop_cycle_span(
                span=cycle_span,
                message=message,
            )
    except EventLoopException as e:
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)

        # Don't yield or log the exception - we already did it when we
        # raised the exception and we don't need that duplication.
        raise
    except (ContextWindowOverflowException, MaxTokensReachedException) as e:
        # Special cased exceptions which we want to bubble up rather than get wrapped in an EventLoopException
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)
        raise e
    except Exception as e:
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)

        # Handle any other exceptions
        yield ForceStopEvent(reason=e)
        logger.exception("cycle failed")
        raise EventLoopException(e, invocation_state["request_state"]) from e

    yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])


async def recurse_event_loop(agent: "Agent", invocation_state: dict[str, Any]) -> AsyncGenerator[TypedEvent, None]:
    """Make a recursive call to event_loop_cycle with the current state.

    This function is used when the event loop needs to continue processing after tool execution.

    Args:
        agent: Agent for which the recursive call is being made.
        invocation_state: Arguments to pass through event_loop_cycle


    Yields:
        Results from event_loop_cycle where the last result contains:

            - StopReason: Reason the model stopped generating
            - Message: The generated message from the model
            - EventLoopMetrics: Updated metrics for the event loop
            - Any: Updated request state
    """
    cycle_trace = invocation_state["event_loop_cycle_trace"]

    # Recursive call trace
    recursive_trace = Trace("Recursive call", parent_id=cycle_trace.id)
    cycle_trace.add_child(recursive_trace)

    yield StartEvent()

    # Retrieve the output_schema from invocation_state to preserve it in recursive calls
    output_schema = invocation_state.get("output_schema")
    events = event_loop_cycle(agent=agent, invocation_state=invocation_state, output_schema=output_schema)
    async for event in events:
        yield event

    recursive_trace.end()


async def _handle_tool_execution(
    stop_reason: StopReason,
    message: Message,
    agent: "Agent",
    cycle_trace: Trace,
    cycle_span: Any,
    cycle_start_time: float,
    invocation_state: dict[str, Any],
) -> AsyncGenerator[TypedEvent, None]:
    """Handles the execution of tools requested by the model during an event loop cycle.

    Args:
        stop_reason: The reason the model stopped generating.
        message: The message from the model that may contain tool use requests.
        agent: Agent for which tools are being executed.
        cycle_trace: Trace object for the current event loop cycle.
        cycle_span: Span object for tracing the cycle (type may vary).
        cycle_start_time: Start time of the current cycle.
        invocation_state: Additional keyword arguments, including request state.

    Yields:
        Tool stream events along with events yielded from a recursive call to the event loop. The last event is a tuple
        containing:
            - The stop reason,
            - The updated message,
            - The updated event loop metrics,
            - The updated request state.
    """
    tool_uses: list[ToolUse] = []
    tool_results: list[ToolResult] = []
    invalid_tool_use_ids: list[str] = []

    validate_and_prepare_tools(message, tool_uses, tool_results, invalid_tool_use_ids)
    tool_uses = [tool_use for tool_use in tool_uses if tool_use.get("toolUseId") not in invalid_tool_use_ids]
    if not tool_uses:
        yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])
        return

    tool_events = agent.tool_executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state
    )
    async for tool_event in tool_events:
        yield tool_event

    # Process structured output if any tools were structured output tools
    structured_output_result = _extract_structured_output(
        tool_uses, tool_results, invocation_state
    )
    if structured_output_result is not None:
        structured_output, output_type = structured_output_result
        invocation_state["request_state"]["structured_output"] = structured_output
        logger.debug(f"Extracted structured output: {type(structured_output).__name__}")

        # Emit structured output event for streaming
        yield StructuredOutputEvent(structured_output, output_type)

    # Store parent cycle ID for the next cycle
    invocation_state["event_loop_parent_cycle_id"] = invocation_state["event_loop_cycle_id"]

    tool_result_message: Message = {
        "role": "user",
        "content": [{"toolResult": result} for result in tool_results],
    }

    agent.messages.append(tool_result_message)
    agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=tool_result_message))
    yield ToolResultMessageEvent(message=tool_result_message)

    if cycle_span:
        tracer = get_tracer()
        tracer.end_event_loop_cycle_span(span=cycle_span, message=message, tool_result_message=tool_result_message)

    if invocation_state["request_state"].get("stop_event_loop", False):
        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace)
        yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])
        return

    events = recurse_event_loop(agent=agent, invocation_state=invocation_state)
    async for event in events:
        yield event


def _extract_structured_output(
    tool_uses: list[ToolUse],
    tool_results: list[ToolResult],
    invocation_state: dict[str, Any]
) -> tuple[Any, type] | None:
    """Extract structured output from tool results if any structured output tools were called.

    Args:
        tool_uses: List of tool use requests from the model
        tool_results: List of tool execution results
        invocation_state: Current invocation state containing structured output context

    Returns:
        Tuple of (parsed structured output instance, output type), or None if no structured output tools were called
    """
    output_schema = invocation_state.get("output_schema")
    structured_output_tool_names = invocation_state.get("structured_output_tool_names", set())

    if not output_schema or not structured_output_tool_names:
        return None

    # Find the first structured output tool that was called
    structured_output_tool_use = None
    structured_output_result = None

    for tool_use in tool_uses:
        tool_name = tool_use.get("name")
        if tool_name in structured_output_tool_names:
            # Find the corresponding result
            tool_use_id = tool_use.get("toolUseId")
            for result in tool_results:
                if result.get("toolUseId") == tool_use_id:
                    structured_output_tool_use = tool_use
                    structured_output_result = result
                    break
            break

    if not structured_output_tool_use or not structured_output_result:
        return None

    # Extract the structured output using the output mode
    try:
        # Get the expected type for this tool
        tool_name = structured_output_tool_use.get("name")
        expected_type = None

        # Find the expected type by matching tool name to output types
        for output_type in output_schema.types:
            if output_type.__name__ == tool_name:
                expected_type = output_type
                break

        if not expected_type:
            logger.warning(f"Could not find expected type for structured output tool: {tool_name}")
            return None

        # Extract the result content
        result_content = structured_output_result.get("content")
        if isinstance(result_content, list) and len(result_content) > 0:
            # Tool results are typically wrapped in a list
            result_data = result_content[0]
        else:
            result_data = result_content

        # Use the output mode to extract the structured result
        structured_output = output_schema.mode.extract_result(result_data, expected_type)

        logger.debug(f"Successfully extracted structured output of type {expected_type.__name__}")
        return (structured_output, expected_type)

    except Exception as e:
        logger.error(f"Failed to extract structured output from tool result: {e}")
        return None
