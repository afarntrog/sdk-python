"""This module implements the central event loop.

The event loop allows agents to:

1. Process conversation messages
2. Execute tools based on model requests
3. Handle errors and recovery strategies
4. Manage recursive execution cycles
"""

import asyncio
import logging
import time
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
from ..output.modes import NativeMode
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
    from ..agent.agent import Agent
    from ..output.base import OutputSchema


if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
INITIAL_DELAY = 4
MAX_DELAY = 240  # 4 minutes
_MAX_STRUCTURED_OUTPUT_ATTEMPTS = 3


async def event_loop_cycle(
    agent: "Agent", 
    invocation_state: dict[str, Any]
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
    output_schema: OutputSchema = invocation_state.get("output_schema")

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

            if invocation_state.get("structured_output_only"):
                tool_specs = output_schema.mode.get_tool_specs(output_schema.type) if output_schema else []
            else:
                tool_specs = agent.tool_registry.get_all_tool_specs()

            tool_choice = invocation_state.get("tool_choice")

            try:
                async for event in stream_messages(agent.model, agent.system_prompt, agent.messages, tool_specs, tool_choice):
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
        # stop_reason = 'TEST_TODO_AFARN' # TODO remove this line... for testing only.
        # popped_msg = agent.messages.pop() # TODO remove this line... for testing only.
        # print(popped_msg)
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
    finally:
        # Ensure cleanup of any remaining structured outputs to prevent memory leaks
        from ..tools.structured_output_tool import _cleanup_all_structured_outputs
        _cleanup_all_structured_outputs(invocation_state)

    # Force structured output tool call if LLM didn't use it automatically
    if output_schema and stop_reason != "tool_use":
        if "structured_output_attempts" not in invocation_state:
            invocation_state["structured_output_attempts"] = 0

        
        if invocation_state["structured_output_attempts"] >= _MAX_STRUCTURED_OUTPUT_ATTEMPTS:
            logger.warning(f"Structured output forcing exceeded maximum attempts ({_MAX_STRUCTURED_OUTPUT_ATTEMPTS}), returning without structured output")
            yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"], None)
            return

        invocation_state["structured_output_attempts"] += 1

        logger.debug(f"Forcing structured output tool, attempt {invocation_state['structured_output_attempts']}/{_MAX_STRUCTURED_OUTPUT_ATTEMPTS}")

        # Create new invocation  If this event loop cycle is called again, it would still have these settings TODO may not be necessary.
        forced_invocation_state = invocation_state.copy()
        forced_invocation_state["tool_choice"] = {"any": {}}
        forced_invocation_state["structured_output_only"] = True

        events = recurse_event_loop(agent=agent, invocation_state=forced_invocation_state)
        async for typed_event in events:
            yield typed_event
        return

    yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"], None)


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

    events = event_loop_cycle(agent=agent, invocation_state=invocation_state)
    async for event in events:
        yield event

    recursive_trace.end()


def _prepare_tools_for_execution(
    message: Message,
) -> tuple[list[ToolUse], list[ToolResult], list[str]]:
    """Prepare and validate tools from a message for execution.
    
    Args:
        message: The message from the model that may contain tool use requests.
        
    Returns:
        A tuple containing:
            - List of valid tool uses
            - List of tool results for invalid tools
            - List of invalid tool use IDs
    """
    tool_uses: list[ToolUse] = []
    tool_results: list[ToolResult] = []
    invalid_tool_use_ids: list[str] = []

    validate_and_prepare_tools(message, tool_uses, tool_results, invalid_tool_use_ids)
    valid_tool_uses = [tool_use for tool_use in tool_uses if tool_use.get("toolUseId") not in invalid_tool_use_ids]
    
    return valid_tool_uses, tool_results, invalid_tool_use_ids


def _create_tool_result_message(tool_results: list[ToolResult]) -> Message:
    """Create a tool result message for conversation history.
    
    Args:
        tool_results: List of tool results to include in the message.
        
    Returns:
        A formatted message containing the tool results.
    """
    return {
        "role": "user",
        "content": [{"toolResult": result} for result in tool_results],
    }


def _process_tool_result_message(
    agent: "Agent",
    tool_result_message: Message,
) -> None:
    """Process and add a tool result message to the agent's conversation history.
    
    Args:
        agent: The agent to add the message to.
        tool_result_message: The tool result message to process.
    """
    agent.messages.append(tool_result_message)
    agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=tool_result_message))


def _cleanup_event_loop_cycle(
    cycle_span: Any,
    cycle_start_time: float,
    cycle_trace: Trace,
    agent: "Agent",
    message: Message,
    tool_result_message: Message,
) -> None:
    """Clean up tracing and metrics for an event loop cycle.
    
    Args:
        cycle_span: Span object for tracing the cycle.
        cycle_start_time: Start time of the current cycle.
        cycle_trace: Trace object for the current event loop cycle.
        agent: The agent for which the cycle is being cleaned up.
        message: The original message from the model.
        tool_result_message: The tool result message.
    """
    if cycle_span:
        tracer = get_tracer()
        tracer.end_event_loop_cycle_span(
            span=cycle_span, 
            message=message, 
            tool_result_message=tool_result_message
        )


async def _handle_structured_output(
    output_schema: "OutputSchema",
    invocation_state: dict[str, Any],
    tool_uses: list[ToolUse],
    tool_results: list[ToolResult],
    stop_reason: StopReason,
    message: Message,
    agent: "Agent",
    cycle_start_time: float,
    cycle_trace: Trace,
    cycle_span: Any,
) -> AsyncGenerator[tuple[TypedEvent, bool], None]:
    """Handle structured output processing and emit appropriate events.
    
    Args:
        output_schema: The output schema to process.
        invocation_state: Current invocation state.
        tool_uses: List of tool uses.
        tool_results: List of tool results.
        stop_reason: The reason the model stopped generating.
        message: The original message from the model.
        agent: The agent processing the structured output.
        cycle_start_time: Start time of the current cycle.
        cycle_trace: Trace object for the current event loop cycle.
        cycle_span: Span object for tracing the cycle.
        
    Yields:
        A tuple of (event, should_stop) where should_stop indicates if processing should halt.
    """
    from ..tools.structured_output_tool import _extract_structured_output_from_state, _cleanup_structured_outputs
    
    structured_output_result = _extract_structured_output_from_state(
        invocation_state, tool_uses, output_schema.type.__name__
    )
    
    # Cleanup any remaining structured outputs
    tool_use_ids = [str(tool_use.get("toolUseId", "")) for tool_use in tool_uses]
    _cleanup_structured_outputs(invocation_state, tool_use_ids)
    
    if structured_output_result is not None:
        yield StructuredOutputEvent(structured_output=structured_output_result), False
        
        # Create and process tool result message
        tool_result_message = _create_tool_result_message(tool_results)
        _process_tool_result_message(agent, tool_result_message)
        yield ToolResultMessageEvent(message=tool_result_message), False

        # End the event loop cycle with structured output
        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace, {})
        _cleanup_event_loop_cycle(cycle_span, cycle_start_time, cycle_trace, agent, message, tool_result_message)

        yield EventLoopStopEvent(
            stop_reason, 
            message, 
            agent.event_loop_metrics, 
            invocation_state["request_state"],
            structured_output=structured_output_result
        ), True


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
    # Prepare and validate tools
    tool_uses, tool_results, invalid_tool_use_ids = _prepare_tools_for_execution(message)
    
    if not tool_uses:
        yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"], None)
        return

    # Execute all tools through normal pipeline
    tool_events = agent.tool_executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state
    )
    async for tool_event in tool_events:
        yield tool_event

    # Handle structured output if present
    output_schema: "OutputSchema" = invocation_state.get("output_schema")
    if output_schema:
        structured_output_tool_names = {output_schema.type.__name__}
        if structured_output_tool_names:  # Only proceed if we have structured output tool names
            async for event, should_stop in _handle_structured_output(
                output_schema, invocation_state, tool_uses, tool_results,
                stop_reason, message, agent, cycle_start_time, cycle_trace, cycle_span
            ):
                yield event
                # Check if we should stop processing
                if should_stop:
                    return

    # Store parent cycle ID for the next cycle
    invocation_state["event_loop_parent_cycle_id"] = invocation_state["event_loop_cycle_id"]

    # Create and process tool result message
    tool_result_message = _create_tool_result_message(tool_results)
    _process_tool_result_message(agent, tool_result_message)
    yield ToolResultMessageEvent(message=tool_result_message)

    # Cleanup tracing
    _cleanup_event_loop_cycle(cycle_span, cycle_start_time, cycle_trace, agent, message, tool_result_message)

    # Check if we should stop the event loop
    if invocation_state["request_state"].get("stop_event_loop", False):
        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace)
        yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"], None)
        return

    # Continue with recursive event loop processing
    events = recurse_event_loop(agent=agent, invocation_state=invocation_state)
    async for event in events:
        yield event
