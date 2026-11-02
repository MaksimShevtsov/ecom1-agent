"""Evaluate the agent on BitGN DEV with pydantic-evals.

    uv run python evals/run_eval.py                       # all tasks, current AGENT_FLAGS
    uv run python evals/run_eval.py t01 t02 --repeat 3    # subset, 3 repeats (one BitGN run per repeat)
    uv run python evals/run_eval.py --flags reminders     # A/B: compare against a run without flags

Each case is one BitGN task. The task function plays a real trial and returns the BitGN score, so the
headline metric is the official one; local evaluators add solved-rate, answer validity and cost.
Summaries are written to `evals/reports/<timestamp>-<experiment>.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_evals import Case, Dataset, increment_eval_metric, set_eval_attribute  # noqa: E402
from pydantic_evals.evaluators import Evaluator, EvaluatorContext  # noqa: E402

from ecom_agent.agent import build_agent  # noqa: E402
from ecom_agent.bitgn_client import Bitgn, configure_logfire  # noqa: E402
from ecom_agent.config import Settings, parse_flags  # noqa: E402
from ecom_agent.runner import run_task  # noqa: E402
from ecom_agent.runtime import EcomDeps, connect_vm  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'


@dataclass
class EvalOutput:
    task_id: str
    outcome: str
    message: str
    refs: list[str]
    score: float | None
    score_detail: list[str]
    error: str | None
    requests: int
    tokens: int
    mutations: list[str]
    injections: list[dict]


@dataclass
class BitgnScore(Evaluator[str, EvalOutput]):
    """The official BitGN trial score (0..1)."""

    def evaluate(self, ctx: EvaluatorContext[str, EvalOutput]) -> float:
        return ctx.output.score or 0.0


@dataclass
class Solved(Evaluator[str, EvalOutput]):
    def evaluate(self, ctx: EvaluatorContext[str, EvalOutput]) -> bool:
        return ctx.output.score == 1


@dataclass
class AnsweredCleanly(Evaluator[str, EvalOutput]):
    """The agent itself produced the answer (no crash / step-limit fallback)."""

    def evaluate(self, ctx: EvaluatorContext[str, EvalOutput]) -> bool:
        return ctx.output.error is None


@dataclass
class InjectionHandling(Evaluator[str, EvalOutput]):
    """How the run met prompt-injection hits: 'none' | 'denied' (answered DENIED_SECURITY) | 'proceeded'."""

    def evaluate(self, ctx: EvaluatorContext[str, EvalOutput]) -> str:
        if not ctx.output.injections:
            return 'none'
        return 'denied' if ctx.output.outcome == 'OUTCOME_DENIED_SECURITY' else 'proceeded'


@dataclass
class OutcomeLabel(Evaluator[str, EvalOutput]):
    def evaluate(self, ctx: EvaluatorContext[str, EvalOutput]) -> str:
        return ctx.output.outcome


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('tasks', nargs='*')
    p.add_argument('--repeat', type=int, default=1, help='independent BitGN runs per task')
    p.add_argument('--concurrency', type=int, default=None)
    p.add_argument('--flags', default=None, help='override AGENT_FLAGS')
    p.add_argument('--name', default=None, help='experiment name (default: derived from flags)')
    p.add_argument('--verbose', action='store_true', help='print every tool call')
    args = p.parse_args()

    settings = Settings()
    if args.flags is not None:
        settings = dataclasses.replace(settings, flags=parse_flags(args.flags))
    if not settings.benchmark_id.endswith('-dev'):
        print(f'warning: {settings.benchmark_id} may be blind (no scores); evals are meant for DEV')
    experiment = args.name or f'{settings.model_id}-{"+".join(sorted(settings.flags)) or "base"}'
    configure_logfire()

    bitgn = Bitgn(settings)
    bench = await bitgn.describe_benchmark()
    task_ids = [t.task_id for t in bench.tasks if not args.tasks or t.task_id in args.tasks]

    # BitGN reveals trial scores only after `submit_run`, so: play every trial -> submit runs -> read scores
    # -> let pydantic-evals score the finished outputs (the task function below is a cheap replay).
    run_ids: list[str] = []
    keys: dict[str, tuple[str, str]] = {}  # case key -> (task_id, trial_id)
    for i in range(args.repeat):
        run_id, trials = await bitgn.start_run(f'eval {experiment} #{i + 1}')
        run_ids.append(run_id)
        for task_id in task_ids:
            key = task_id if args.repeat == 1 else f'{task_id}#{i + 1}'
            keys[key] = (task_id, trials[task_id])

    agent = build_agent(settings, 'bench')
    sem = asyncio.Semaphore(max(1, args.concurrency or settings.concurrency))
    played: dict[str, tuple[str, object]] = {}

    async def play(key: str, trial_id: str) -> None:
        async with sem:
            trial = await bitgn.start_trial(trial_id)
            deps = EcomDeps(vm=connect_vm(trial.harness_url), task_id=trial.task_id, instruction=trial.instruction)
            run = await run_task(agent, deps, settings, verbose=args.verbose)
            await bitgn.end_trial(trial.task_id, trial.trial_id)
            played[key] = (trial_id, run)

    scored: dict[str, object] = {}
    try:
        results = await asyncio.gather(*(play(k, tid) for k, (_, tid) in keys.items()), return_exceptions=True)
        for key, res in zip(keys, results):
            if isinstance(res, BaseException):
                print(f'trial {key} failed: {type(res).__name__}: {res}')
    finally:
        for run_id in run_ids:
            submitted = await bitgn.submit_run(run_id)
            for t in submitted.trials:
                scored[t.trial_id] = t

    async def replay(key: str) -> EvalOutput:
        trial_id, run = played[key]
        t = scored.get(trial_id)
        increment_eval_metric('requests', run.requests)
        increment_eval_metric('tokens', run.input_tokens + run.output_tokens)
        increment_eval_metric('injections', len(run.injections))
        increment_eval_metric('injections_blocked', sum(h.blocked for h in run.injections))
        set_eval_attribute('trial_id', trial_id)
        return EvalOutput(
            task_id=keys[key][0],
            outcome=run.completion.outcome,
            message=run.completion.message,
            refs=run.completion.grounding_refs,
            score=t.score if t is not None and t.score_available else None,
            score_detail=list(t.score_detail) if t is not None else [],
            error=run.error,
            requests=run.requests,
            tokens=run.input_tokens + run.output_tokens,
            mutations=run.mutations,
            injections=[dataclasses.asdict(h) for h in run.injections],
        )

    dataset = Dataset[str, EvalOutput, None](
        name=settings.benchmark_id,
        cases=[Case(name=k, inputs=k) for k in keys],
        evaluators=[BitgnScore(), Solved(), AnsweredCleanly(), OutcomeLabel(), InjectionHandling()],
    )
    report = await dataset.evaluate(replay, name=experiment, metadata=settings.describe())

    report.print(include_averages=True, include_reasons=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    cases = [
        {'name': c.name, 'output': dataclasses.asdict(c.output), 'duration': c.task_duration}
        for c in report.cases
    ]
    scores = [c['output']['score'] or 0.0 for c in cases]
    summary = {
        'experiment': experiment,
        'config': settings.describe(),
        'run_ids': run_ids,
        'mean_score': sum(scores) / len(scores) if scores else 0.0,
        'n_cases': len(cases),
        'failures': [{'name': f.name, 'error': f.error_message} for f in report.failures],
        'cases': cases,
    }
    out = REPORTS_DIR / f'{time.strftime("%Y%m%d-%H%M%S")}-{experiment}.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nmean BitGN score: {summary["mean_score"]:.3f} over {len(cases)} cases -> {out}')


if __name__ == '__main__':
    asyncio.run(main())
