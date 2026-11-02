"""BitGN harness helpers: runs, trials, scoring."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import logfire
from bitgn.harness_connect import HarnessServiceClient
from bitgn.harness_pb2 import (
    TRIAL_STATE_DONE,
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    GetRunRequest,
    StartRunRequest,
    StartTrialRequest,
    StatusRequest,
    SubmitRunRequest,
)

from .config import Settings
from .runner import CLR, GREEN, RED

def configure_logfire() -> None:
    logfire.configure(send_to_logfire='if-token-present', service_name='ecom1-agent', console=False)
    logfire.instrument_pydantic_ai()


@dataclass
class TrialScore:
    task_id: str
    trial_id: str
    score: float | None
    detail: list[str]


class Bitgn:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = HarnessServiceClient(settings.bitgn_host)

    async def describe_benchmark(self):
        print('Connecting to BitGN', await self.client.status(StatusRequest()))
        bench = await self.client.get_benchmark(GetBenchmarkRequest(benchmark_id=self.settings.benchmark_id))
        print(
            f'{EvalPolicy.Name(bench.policy)} benchmark: {bench.benchmark_id} with {len(bench.tasks)} tasks.\n'
            f'{GREEN}{bench.description}{CLR}'
        )
        return bench

    async def start_run(self, name: str) -> tuple[str, dict[str, str]]:
        """Start a run; return (run_id, {task_id: trial_id}) without starting any trial yet."""
        run = await self.client.start_run(
            StartRunRequest(name=name, benchmark_id=self.settings.benchmark_id, api_key=self.settings.bitgn_api_key)
        )
        info = await self.client.get_run(GetRunRequest(run_id=run.run_id))
        trials = {t.task_id: t.trial_id for t in info.trials}
        if not trials:  # fall back to the ids from StartRun (task ids unknown until started)
            trials = {f'?{i}': tid for i, tid in enumerate(run.trial_ids)}
        return run.run_id, trials

    async def start_trial(self, trial_id: str):
        return await self.client.start_trial(StartTrialRequest(trial_id=trial_id))

    async def start_task_trial(self, run_name: str, task_id: str):
        """Start a fresh run and only the trial for `task_id` (BitGN no longer supports playgrounds)."""
        run_id, trials = await self.start_run(run_name)
        if task_id in trials:
            return run_id, await self.start_trial(trials[task_id])
        for trial_id in trials.values():  # task ids not exposed up front: probe trials
            trial = await self.start_trial(trial_id)
            if trial.task_id == task_id:
                return run_id, trial
        raise ValueError(f'task {task_id!r} not found in {self.settings.benchmark_id}')

    async def end_trial(self, task_id: str, trial_id: str) -> TrialScore:
        res = await self.client.end_trial(EndTrialRequest(trial_id=trial_id))
        return TrialScore(
            task_id=task_id,
            trial_id=trial_id,
            score=res.score if res.score_available else None,
            detail=list(res.score_detail),
        )

    async def submit_run(self, run_id: str):
        print(f'\n{GREEN}>>>> Submitting run {run_id}... <<<<{CLR}')
        result = await self.client.submit_run(SubmitRunRequest(run_id=run_id, force=True))
        if not result.score_available:
            print(f'{RED}Score is not available. Results are sealed and will be revealed later{CLR}')
            return result
        print(f'FINAL SCORE: {result.score:0.2f}')
        incomplete = 0
        for t in result.trials:
            if t.state != TRIAL_STATE_DONE:
                incomplete += 1
                continue
            style = GREEN if t.score == 1 else RED
            detail = textwrap.indent('\n'.join(t.score_detail), '  ')
            print(f'- {t.task_id}: {style}Score: {t.score:0.2f}{CLR}\n{detail}'.rstrip())
        if incomplete:
            print(f'{RED}incomplete trials: {incomplete}{CLR}')
        return result
