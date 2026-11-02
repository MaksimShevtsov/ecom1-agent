"""Web chat over one BitGN task: `uv run python chat.py t01 [--port 7932] [--submit]`.

BitGN no longer supports playgrounds, so this starts a normal run and plays only the trial for the
task. It collects the same workspace context the bench agent gets and serves the Pydantic AI web UI
on localhost. The final answer goes through the `report_completion` tool, which you approve in the
UI. On exit the trial is ended (DEV returns its score); the run is not submitted unless `--submit`.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses

import uvicorn

from ecom_agent.agent import build_agent
from ecom_agent.bitgn_client import Bitgn, configure_logfire
from ecom_agent.config import Settings, parse_flags
from ecom_agent.prompts import task_prompt
from ecom_agent.runner import BLUE, CLR, GREEN
from ecom_agent.runtime import EcomDeps, connect_vm, prologue


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task', help='task id, e.g. t01')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=7932)
    p.add_argument('--flags', default=None, help='override AGENT_FLAGS')
    p.add_argument('--submit', action='store_true', help='submit the run on exit')
    args = p.parse_args()

    settings = Settings()
    if args.flags is not None:
        settings = dataclasses.replace(settings, flags=parse_flags(args.flags))
    configure_logfire()

    bitgn = Bitgn(settings)
    run_id, trial = await bitgn.start_task_trial(f'chat {args.task}', args.task)
    print(f'Run {run_id}, trial {trial.trial_id} for {trial.task_id}\n{BLUE}{trial.instruction}{CLR}')

    deps = EcomDeps(vm=connect_vm(trial.harness_url), task_id=trial.task_id, instruction=trial.instruction)
    context = await prologue(deps.vm)

    agent = build_agent(settings, 'chat')
    app = agent.to_web(deps=deps, instructions=task_prompt(context, trial.instruction))

    print(f'{GREEN}Chat UI: http://{args.host}:{args.port}{CLR}  (Ctrl+C to stop)')
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level='warning'))
    try:
        await server.serve()
    finally:
        score = await bitgn.end_trial(trial.task_id, trial.trial_id)
        print(f'Trial ended. answered={deps.answered is not None} score={score.score} {score.detail}')
        if args.submit:
            await bitgn.submit_run(run_id)


if __name__ == '__main__':
    asyncio.run(main())
