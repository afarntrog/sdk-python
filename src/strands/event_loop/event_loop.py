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
from ..output.modes import NativeOutput
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


async def event_loop_cycle(
    agent: "Agent", 
    invocation_state: dict[str, Any],
    output_schema: "Optional[OutputSchema]" = None
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

            # Handle tool specifications based on invocation state
            if invocation_state.get("structured_output_only", False):
                # Only use structured output tools when forcing invocation
                if output_schema:
                    tool_specs = output_schema.mode.get_tool_specs(output_schema.type)
                else:
                    tool_specs = []
            else:
                # Normal operation: use all available tools
                tool_specs = agent.tool_registry.get_all_tool_specs()
                
                # Add structured output tools if output_schema is specified
                if output_schema:
                    structured_output_tools = output_schema.mode.get_tool_specs(output_schema.type)
                    tool_specs.extend(structured_output_tools)

            # Get tool_choice parameter if specified
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

        # If the model is requesting to use tools TODO over here we EXIT the loop if the stop_reason is tool_use so we never get to check if the structured output wasn't called
        # stop_reason = 'TEST_TODO_AFARN'
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

    # Force structured output tool call if LLM didn't use it automatically
    if output_schema and stop_reason != "tool_use":
        # Check if we're using NativeOutput mode
        if isinstance(output_schema.mode, NativeOutput):
            logger.debug("LLM didn't call structured output tool, trying native structured output")

            # Use model's native structured output instead of forcing tools
            try:
                events = agent.model.structured_output(
                    output_schema.type,
                    agent.messages,
                    system_prompt=agent.system_prompt
                )

                # Process events from native structured output
                async for event in events:
                    if "output" in event:
                        # Emit structured output event
                        yield StructuredOutputEvent(structured_output=event["output"])

                        # Emit stop event with structured output to properly terminate
                        yield EventLoopStopEvent(
                            stop_reason=stop_reason,  # Use the actual stop reason from the original model call
                            message=message,
                            metrics=agent.event_loop_metrics,
                            request_state=invocation_state["request_state"],
                            structured_output=event["output"]
                        )
                        return

            except Exception as e:
                logger.warning(f"Native structured output failed: {e}, falling back to tool forcing")
                # Fall through to tool forcing logic below

        logger.debug("LLM didn't call structured output tool, forcing invocation via recursive event loop")

        # Add forcing message to conversation
        # TODO I don't think we need this. I think we can just pass in the data with the tool with toolChoice
        # force_message: Message = {
        #     "role": "user",
        #     "content": [{"text": f"Use the {output_schema.type.__name__} tool."}]
        # }
        # agent.messages.append(force_message)
        # agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=force_message))

        # TODO testing only; remove the tool_use message (this can never happen in prod because we check in the begining if)

        # Create new invocation state for forced call with tool choice
        forced_invocation_state = invocation_state.copy()
        forced_invocation_state["tool_choice"] = {"tool": {"name": output_schema.type.__name__}}
        forced_invocation_state["structured_output_only"] = True

        # Recursively call event loop with constraints
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

    events = event_loop_cycle(agent=agent, invocation_state=invocation_state, output_schema=invocation_state.get("output_schema"))
    async for event in events:
        yield event

    recursive_trace.end()


async def _handle_structured_output_execution(
    tool_uses: list[ToolUse],
    output_schema: "OutputSchema",
    stop_reason: StopReason,
    message: Message,
    agent: "Agent",
    cycle_trace: Trace,
    cycle_span: Any,
    cycle_start_time: float,
    invocation_state: dict[str, Any],
) -> AsyncGenerator[TypedEvent, None]:
    """Handle structured output tool execution.
    
    This function processes tool uses that match structured output schemas,
    creating appropriate tool results and managing tracing/metrics.
    
    Args:
        tool_uses: List of tool use requests from the model
        output_schema: Output schema defining expected structured output
        stop_reason: The reason the model stopped generating
        message: The message from the model
        agent: Agent for which tools are being executed
        cycle_trace: Trace object for the current event loop cycle
        cycle_span: Span object for tracing the cycle
        cycle_start_time: Start time of the current cycle
        invocation_state: State maintained across cycles
        
    Yields:
        Events related to structured output execution
    """
    structured_output_result = _extract_structured_output(tool_uses, output_schema)
    if structured_output_result is None:
        return
        
    # Get expected tool names for structured output
    expected_tool_names = {output_schema.type.__name__}
    
    # Create tool results and traces for structured output tools
    structured_output_tool_results = []
    for tool_use in tool_uses:
        tool_name = tool_use.get("name", "")
        
        if tool_name in expected_tool_names:
            # Create a tool result for this structured output tool. we are here because the LLM decided to populate the SO tool: `tool_use`
            tool_result: ToolResult = {
                "toolUseId": tool_use["toolUseId"],
                "content": [{"text": f"Successfully extracted {tool_name} data"}],
                "status": "success"
            }
            structured_output_tool_results.append(tool_result)
            
            # Create trace and metrics using existing infrastructure pattern
            tracer = get_tracer()
            tool_call_span = tracer.start_tool_call_span(tool_use, cycle_span)
            tool_trace = Trace(f"Tool: {tool_name}", parent_id=cycle_trace.id, raw_name=tool_name)
            tool_start_time = time.time()
            
            # Simulate successful tool execution
            tool_success = True
            tool_duration = time.time() - tool_start_time
            result_message = Message(role="user", content=[{"toolResult": tool_result}])
            agent.event_loop_metrics.add_tool_usage(tool_use, tool_duration, tool_trace, tool_success, result_message)
            cycle_trace.add_child(tool_trace)
            
            tracer.end_tool_call_span(tool_call_span, tool_result)
    
    # Emit structured output event
    yield StructuredOutputEvent(structured_output_result)
    
    # Add tool result message to conversation history to maintain proper flow
    if structured_output_tool_results:
        tool_result_message: Message = {
            "role": "user",
            "content": [{"toolResult": result} for result in structured_output_tool_results],
        }
        agent.messages.append(tool_result_message)
        agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=tool_result_message))
        yield ToolResultMessageEvent(message=tool_result_message)
    
    # End the event loop with the structured output result
    agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace, {})
    if cycle_span:
        tracer = get_tracer()
        tracer.end_event_loop_cycle_span(span=cycle_span, message=message)
    
    yield EventLoopStopEvent(
        stop_reason, 
        message, 
        agent.event_loop_metrics, 
        invocation_state["request_state"],
        structured_output=structured_output_result
    )


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
        yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"], None)
        return

    # Check for structured output tool calls
    output_schema = invocation_state.get("output_schema")
    if output_schema:
        structured_output_result = _extract_structured_output(tool_uses, output_schema)
        if structured_output_result is not None:
            # Delegate to structured output handling
            async for event in _handle_structured_output_execution(
                tool_uses, output_schema, stop_reason, message, agent,
                cycle_trace, cycle_span, cycle_start_time, invocation_state
            ):
                yield event
            return

    # TODO over here we take the `tool_uses` from earlier and then actually execute them. We don't execute the structured output tool but I'm thinking we should put it in a tool so that the LLM can retry if it needs to. Example "tool_uses": `[{'toolUseId': 'tooluse_PPkLaDZSR3CtHJy6xtVz5w', 'name': 'get_user_location', 'input': {}}]`
    tool_events = agent.tool_executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state
    )
    async for tool_event in tool_events:
        yield tool_event

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
        yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"], None)
        return

    events = recurse_event_loop(agent=agent, invocation_state=invocation_state)
    async for event in events:
        yield event


def _extract_structured_output(tool_uses: list[ToolUse], output_schema: "OutputSchema") -> Any:
    """Extract structured output from tool uses if any match the output schema.
    
    Args:
        tool_uses: List of tool use requests from the model
        output_schema: Output schema defining expected structured output
        
    Returns:
        Structured output instance if found, None otherwise
    """
    # Get the expected tool names for structured output
    expected_tool_names = set()
    # Tool names are the exact class name as generated by convert_pydantic_to_tool_spec
    expected_tool_names.add(output_schema.type.__name__)
    
    # Look for matching tool calls
    for tool_use in tool_uses:
        tool_name = tool_use.get("name", "")
        if tool_name in expected_tool_names:
            # Found a structured output tool call
            tool_input = tool_use.get("input", {})
            
            # Find the matching output type
            if tool_name == output_schema.type.__name__:
                try:
                    # Create instance from tool input
                    return output_schema.type(**tool_input)
                except Exception as e:
                    logger.warning(f"Failed to create {output_schema.type.__name__} from tool input: {e}")
                    continue
    
    return None
