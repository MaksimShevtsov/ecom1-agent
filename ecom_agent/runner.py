"""Run one ECOM task end to end: prologue -> agent -> answer."""

from __future__ import annotations

import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, field

import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.usage import UsageLimits

from .config import Settings
from .models import ERR_COMPLETION, TaskCompletion
from .prompts import task_prompt
from .runtime import EcomDeps, InjectionHit, prologue, submit_answer

GREEN, RED, BLUE, YELLOW, DIM, CLR = '\x1b[32m', '\x1b[31m', '\x1b[34m', '\x1b[33m', '\x1b[2m', '\x1b[0m'


@dataclass
class TaskRun:
    task_id: str
    completion: TaskCompletion
    error: str | None = None
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_sec: float = 0.0
    mutations: list[str] = field(default_factory=list)
    injections: list[InjectionHit] = field(default_factory=list)


def describe_error(exc: BaseException) -> str:
    """Include sub-exceptions (e.g. every model tried by ModelFallback) in the message."""
    text = f'{type(exc).__name__}: {exc}'
    if isinstance(exc, BaseExceptionGroup):
        text += ' [' + '; '.join(describe_error(sub) for sub in exc.exceptions) + ']'
    return text


def _short(text: object, limit: int = 300) -> str:
    s = str(text).replace('\n', ' | ')
    return s if len(s) <= limit else s[:limit] + '…'


def console_handler(task_id: str):
    async def handler(ctx: RunContext[EcomDeps], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            if isinstance(event, FunctionToolCallEvent):
                print(f'{DIM}[{task_id}]{CLR} {BLUE}→ {event.part.tool_name}{CLR} {_short(event.part.args, 400)}')
            elif isinstance(event, FunctionToolResultEvent):
                print(f'{DIM}[{task_id}]{CLR} {GREEN}←{CLR} {_short(getattr(event.part, "content", event.part))}')

    return handler


async def run_task(
    agent: Agent[EcomDeps, TaskCompletion],
    deps: EcomDeps,
    settings: Settings,
    *,
    verbose: bool = True,
) -> TaskRun:
    started = time.monotonic()
    with logfire.span('ecom task {task_id}', task_id=deps.task_id, config=settings.describe()) as span:
        error: str | None = None
        usage = None
        try:
            context = await prologue(deps.vm)
            result = await agent.run(
                task_prompt(context, deps.instruction),
                deps=deps,
                usage_limits=UsageLimits(request_limit=settings.max_steps),
                # A stream handler switches the run to streaming requests; only attach it when printing.
                event_stream_handler=console_handler(deps.task_id) if verbose else None,
            )
            completion = result.output
            usage = result.usage
        except Exception as exc:  # any failure must still produce an answer
            error = describe_error(exc)
            completion = ERR_COMPLETION.model_copy(update={'message': f'{ERR_COMPLETION.message} ({error})'})
            print(f'{RED}[{deps.task_id}] agent error: {error}{CLR}')

        try:
            await submit_answer(deps, completion)
        except Exception as exc:
            error = (error + '; ' if error else '') + f'answer failed: {exc}'
            print(f'{RED}[{deps.task_id}] answer failed: {exc}{CLR}')

        run = TaskRun(
            task_id=deps.task_id,
            completion=completion,
            error=error,
            requests=usage.requests if usage else 0,
            tool_calls=usage.tool_calls if usage else 0,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            duration_sec=time.monotonic() - started,
            mutations=list(deps.mutations),
            injections=list(deps.injections),
        )
        span.set_attributes(
            {
                'outcome': completion.outcome,
                'requests': run.requests,
                'error': error or '',
                'injections': len(run.injections),
                'injections_blocked': sum(h.blocked for h in run.injections),
            }
        )

    color = GREEN if completion.outcome == 'OUTCOME_OK' else YELLOW
    print(
        f'{color}[{deps.task_id}] {completion.outcome}{CLR}: {completion.message}\n'
        f'  refs={completion.grounding_refs} mutations={run.mutations} '
        f'requests={run.requests} tokens={run.input_tokens}/{run.output_tokens} {run.duration_sec:.0f}s'
    )
    for hit in run.injections:
        action = 'BLOCKED' if hit.blocked else 'audit'
        print(f'  {RED}injection {action}{CLR} {hit.risk} in {hit.tool}({hit.args}) {hit.detections}')
    return run
