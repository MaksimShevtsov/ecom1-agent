# ECOM1 agent: Pydantic AI + pydantic-ai-harness + z.ai

An agent for the [BitGN ECOM1](https://bitgn.com/start/ecom) benchmark, built on `pydantic-ai-slim 2.64` and
`pydantic-ai-harness 0.64`. It runs `glm-5.3` through the z.ai GLM Coding Plan
(`https://api.z.ai/api/coding/paas/v4`). It ships with an eval harness (`pydantic-evals`), a web chat
(`agent.to_web`) and optional Logfire tracing.

## Results on BitGN DEV (55 tasks)

| Configuration | Mean score | Solved |
|---|---|---|
| Baseline prompt | 0.602 | 28 |
| + answer rules (refs, formats, policy docs) | 0.815 | 44 |
| + refined refs / refusal rules (`prompt4`) | 0.877 | 47 |
| + deterministic answer validators, higher `run_code` retries (3 passes) | 0.871 | 46-47 per pass |
| + archive-row ref fix, ambiguity rule (2 passes) | 0.894 | 46 / 49 |

Single passes are noisy (about +/-0.02 between identical runs), and task data is regenerated for every BitGN
run, so compare averages over several repeats, not individual tasks. DEV scores do not guarantee PROD performance:
parts of the prompt were derived from DEV failures.

A/B of the optional capabilities on top of `prompt4` (one pass each): `planning` 0.867, `reminders` 0.821,
`defender` (blocking) 0.793, versus 0.877 without flags. None of them helped, so `AGENT_FLAGS` is empty by default.

## Setup

```powershell
uv sync                      # Python 3.14; the BitGN SDK is pulled from buf.build
copy .env.example .env       # fill in ZAI_API_KEY and BITGN_API_KEY (+ LOGFIRE_TOKEN, optional)
uv run pytest -q             # offline tests (fake VM + FunctionModel), no keys needed
```

## Usage

| What | Command |
|---|---|
| Smoke test, one task | `uv run python main.py t01` |
| Whole DEV benchmark | `uv run python main.py --quiet` |
| PROD (blind) | `$env:BENCHMARK_ID='bitgn/ecom1-prod'; uv run python main.py --quiet` |
| Eval (pydantic-evals) | `uv run python evals/run_eval.py t01 t02 --repeat 3` |
| A/B a capability | `uv run python evals/run_eval.py --flags reminders,planning` |
| Web chat for one task | `uv run python chat.py t07` then open http://127.0.0.1:7932 |

## Architecture

```
main.py              benchmark runner: start_run -> trials (CONCURRENCY) -> end_trial -> submit_run
chat.py              one-task run + agent.to_web(deps=...); the final answer is approved in the UI
evals/run_eval.py    Dataset from DEV tasks; BitGN score + Solved/AnsweredCleanly/Outcome; reports in evals/reports/
ecom_agent/
  agent.py           build_agent(settings, 'bench'|'chat'): models, capabilities, answer validator
  tools.py           explore (read-only -> CodeMode) / mutate (explicit calls) / chat (report_completion)
  validation.py      deterministic answer clean-up (URLs, <YES>/<NO>, private refs in refusals)
  runtime.py         EcomDeps, shell-style formatting of VM responses, prologue (tree, AGENTS.MD, date, id), answer
  runner.py          one task: prologue -> agent.run -> answer (always sent, even after a crash)
  bitgn_client.py    HarnessService wrapper, Logfire setup
  prompts.py         system prompt and task framing
```

### How the eval works

BitGN reveals trial scores only after `submit_run`. The eval therefore plays every trial, submits the runs, reads the
scores, and then lets `pydantic-evals` score the finished outputs. Each `--repeat` is an independent BitGN run.
Reports are written to `evals/reports/<timestamp>-<experiment>.json` (git-ignored).

### Capabilities

Always on:

- `CodeMode(tools=read-only, max_retries=8)`: `tree/list_dir/find/search/read_file/stat/sql` are available inside
  `run_code`; `write_file/delete_file/exec` stay explicit tool calls.
- `Thinking(effort=THINKING_EFFORT)`: maps to `reasoning_effort` for GLM-5.3.
- `ToolOutputLimits`: head+tail truncation of outputs over 12k characters.
- `RepairToolArguments`: repairs invalid tool-argument JSON.
- `DetectLoops(action='nudge')`: breaks repeating action loops.
- `WarnNearLimits(max_iterations=MAX_STEPS)`: lets the model answer before the step limit.
- `ModelFallback(FALLBACK_MODEL_ID)`: switches to `glm-5.2` on API errors.
- `PromptInjectionDefender` in **audit** mode: classifies results of the data tools, records hits in
  `deps.injections` (console, Logfire, eval metrics) and never hides content from the model. `run_code` as a whole is
  not classified: CodeMode routes every nested call through the defender, so only a poisoned call is affected.
- Answer validator: `grounding_refs` are checked with `stat` (archive `#row=` and `:line` suffixes are stripped), one
  retry, then missing refs are dropped. Afterwards deterministic clean-up runs (see below).

#### Deterministic answer clean-up (`ecom_agent/validation.py`)

- URLs that are not in the task text are replaced with `[link removed]` (phishing links found in data).
- If the task asks for `<YES>` / `<NO>`, the message is reduced to that token.
- For `OUTCOME_DENIED_SECURITY`, refs to employee records, other customers' records and other customers' baskets
  are dropped (the requester comes from `/bin/id`); policy documents and the requester's own records stay.

### Prompt-injection defender modes

| flags | behaviour |
|---|---|
| (none) | audit: records hits, content still reaches the model |
| `defender` | high/critical results are replaced with a stub that hints at `OUTCOME_DENIED_SECURITY` |
| `defender,defender_ml` | + local ONNX classifier, requires `uv sync --extra defender-ml` |

The eval reports `injections` / `injections_blocked` metrics and an `InjectionHandling` label (`none` / `denied` /
`proceeded`).

### A/B flags (`AGENT_FLAGS` / `--flags`)

| flag | capability |
|---|---|
| `reminders` | `SystemReminders` + `GoalReanchor` (policies / original task every 6 steps) |
| `planning` | `Planning` |
| `tool_judge` | `ToolCallJudge` (JUDGE_MODEL_ID) on `write_file/delete_file/exec` |
| `defender` / `defender_ml` | blocking mode of `PromptInjectionDefender` / + ML (see above) |
| `trajectory` | `TrajectoryJudge(every=8)` |
| `sgr` | Schema-Guided Reasoning: structured final report (cascade of checks, discriminated answer branches). Experimental: 0.866 over 3 passes vs 0.894 without, so off by default |

## Configuration

See `.env.example`: z.ai key and endpoint, `MODEL_ID`, `FALLBACK_MODEL_ID`, `THINKING_EFFORT`, `JUDGE_MODEL_ID`,
BitGN key / host / `BENCHMARK_ID`, `MAX_STEPS`, `CONCURRENCY`, `AGENT_FLAGS`, `LOGFIRE_TOKEN`. Never commit `.env`.

## Notes

- `connectrpc` is pinned to `<0.11`: the generated BitGN SDK is incompatible with 0.13 (`MethodInfo.stream_type`).
- BitGN's playground/sandbox is disabled ("use normal runs"), so the chat creates a normal run and plays a single
  task; the run is not submitted unless you pass `--submit`.
- Evals are only meaningful on DEV: PROD scores are sealed.
- Known weak spots: the payment-fraud tasks (t38-t40, t48) recover only part of the incident or flag too many
  payments. A heuristic anomaly tool was tried and made results worse, because the model anchored on its clusters.

## License

[MIT](LICENSE)
