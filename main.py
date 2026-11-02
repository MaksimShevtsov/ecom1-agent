"""Bench runner: `uv run python main.py [t01 t04 ...] [--flags reminders,planning] [--name NAME]`."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses

from connectrpc.errors import ConnectError

from ecom_agent.agent import build_agent
from ecom_agent.bitgn_client import Bitgn, TrialScore, configure_logfire
from ecom_agent.config import Settings, parse_flags
from ecom_agent.runner import BLUE, CLR, GREEN, RED, TaskRun, run_task
from ecom_agent.runtime import EcomDeps, connect_vm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tasks', nargs='*', help='task ids to run (default: all)')
    p.add_argument('--flags', default=None, help='override AGENT_FLAGS (A/B capabilities)')
    p.add_argument('--name', default='ecom1-agent (pydantic-ai-harness, z.ai)')
    p.add_argument('--quiet', action='store_true', help='do not print every tool call')
    return p.parse_args()


async def run_trial(
    bitgn: Bitgn, agent, settings: Settings, trial_id: str, *, verbose: bool, task_filter: list[str]
) -> tuple[TaskRun, TrialScore] | None:
    trial = await bitgn.start_trial(trial_id)
    if task_filter and trial.task_id not in task_filter:
        return None  # only reachable when GetRun did not expose task ids up front
    print(f'{"=" * 30} Starting task: {trial.task_id} {"=" * 30}\n{BLUE}{trial.instruction}{CLR}\n{"-" * 80}')
    deps = EcomDeps(vm=connect_vm(trial.harness_url), task_id=trial.task_id, instruction=trial.instruction)
    run = await run_task(agent, deps, settings, verbose=verbose)
    score = await bitgn.end_trial(trial.task_id, trial.trial_id)
    if score.score is not None:
        style = GREEN if score.score == 1 else RED
        print(f'{style}[{trial.task_id}] score {score.score:0.2f}{CLR} ' + '; '.join(score.detail))
    return run, score


async def main() -> None:
    args = parse_args()
    settings = Settings()
    if args.flags is not None:
        settings = dataclasses.replace(settings, flags=parse_flags(args.flags))
    configure_logfire()
    agent = build_agent(settings, 'bench')
    bitgn = Bitgn(settings)

    try:
        await bitgn.describe_benchmark()
        run_id, trials = await bitgn.start_run(args.name)
        selected = [tid for task, tid in trials.items() if not args.tasks or task in args.tasks or task.startswith('?')]
        sem = asyncio.Semaphore(max(1, settings.concurrency))

        async def guarded(trial_id: str):
            async with sem:
                return await run_trial(
                    bitgn, agent, settings, trial_id, verbose=not args.quiet, task_filter=args.tasks
                )

        try:
            results = await asyncio.gather(*(guarded(t) for t in selected), return_exceptions=True)
            for res in results:
                if isinstance(res, BaseException):
                    print(f'{RED}trial failed: {type(res).__name__}: {res}{CLR}')
        finally:
            await bitgn.submit_run(run_id)
    except ConnectError as exc:
        print(f'{exc.code}: {exc.message}')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f'{RED}Interrupted{CLR}')
