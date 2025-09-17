ValidationException                       Traceback (most recent call last)
File /Volumes/workplace/dev/structured_output/v3/sdk-python/src/strands/event_loop/event_loop.py:269, in event_loop_cycle(agent, invocation_state, output_schema)
    260 events = _handle_tool_execution(
    261     stop_reason,
    262     message,
   (...)
    267     invocation_state=invocation_state,
    268 )
--> 269 async for typed_event in events:
    270     yield typed_event

File /Volumes/workplace/dev/structured_output/v3/sdk-python/src/strands/event_loop/event_loop.py:416, in _handle_tool_execution(stop_reason, message, agent, cycle_trace, cycle_span, cycle_start_time, invocation_state)
    415 events = recurse_event_loop(agent=agent, invocation_state=invocation_state)
--> 416 async for event in events:
    417     yield event

File /Volumes/workplace/dev/structured_output/v3/sdk-python/src/strands/event_loop/event_loop.py:332, in recurse_event_loop(agent, invocation_state)
    331 events = event_loop_cycle(agent=agent, invocation_state=invocation_state)
--> 332 async for event in events:
    333     yield event

File /Volumes/workplace/dev/structured_output/v3/sdk-python/src/strands/event_loop/event_loop.py:225, in event_loop_cycle(agent, invocation_state, output_schema)
    224             else:
--> 225                 raise e
...
    299     logger.exception("cycle failed")
--> 300     raise EventLoopException(e, invocation_state["request_state"]) from e
    302 yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])

EventLoopException: An error occurred (ValidationException) when calling the ConverseStream operation: The toolConfig field must be defined when using toolUse and toolResult content blocks.