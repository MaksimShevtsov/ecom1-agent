"""Thin async layer over the BitGN ECOM runtime: deps, shell-shaped formatting, prologue."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Protocol

from bitgn.vm.ecom.ecom_pb2 import (
    AnswerRequest,
    ExecRequest,
    NodeKind,
    ReadRequest,
    TreeRequest,
)
from connectrpc.errors import ConnectError

from .models import OUTCOME_BY_NAME, TaskCompletion


class EcomVm(Protocol):
    """The subset of `EcomRuntimeClient` the agent uses (a fake implements it in tests)."""

    async def tree(self, request: Any) -> Any: ...
    async def find(self, request: Any) -> Any: ...
    async def search(self, request: Any) -> Any: ...
    async def list(self, request: Any) -> Any: ...
    async def read(self, request: Any) -> Any: ...
    async def write(self, request: Any) -> Any: ...
    async def delete(self, request: Any) -> Any: ...
    async def stat(self, request: Any) -> Any: ...
    async def exec(self, request: Any) -> Any: ...
    async def answer(self, request: Any) -> Any: ...


@dataclass
class InjectionHit:
    """One prompt-injection verdict from `PromptInjectionDefender` on a tool result."""

    tool: str
    args: str
    risk: str
    detections: list[str]
    tier2_score: float | None
    blocked: bool


@dataclass
class EcomDeps:
    vm: EcomVm
    task_id: str = ''
    instruction: str = ''
    # Set once `answer` was sent, so we never answer twice (chat mode can call it as a tool).
    answered: TaskCompletion | None = None
    # Paths written/deleted during the run, for console summary and eval metrics.
    mutations: list[str] = field(default_factory=list)
    # Prompt-injection verdicts on tool results (audit and blocking modes).
    injections: list[InjectionHit] = field(default_factory=list)


def connect_vm(harness_url: str) -> EcomVm:
    from bitgn.vm.ecom.ecom_connect import EcomRuntimeClient

    return EcomRuntimeClient(harness_url)


def format_error(exc: ConnectError) -> str:
    return f'ERROR {exc.code}: {exc.message}'


def _truncated(result: Any, body: str, hint: str) -> str:
    if not getattr(result, 'truncated', False):
        return body
    marker = f'[TRUNCATED: {hint}]'
    return f'{body}\n{marker}' if body else marker


def _tree_lines(entry: Any, prefix: str = '', is_last: bool = True) -> list[str]:
    lines = [f'{prefix}{"`-- " if is_last else "|-- "}{entry.name}']
    child_prefix = f'{prefix}{"    " if is_last else "|   "}'
    children = list(entry.children)
    for i, child in enumerate(children):
        lines.extend(_tree_lines(child, child_prefix, i == len(children) - 1))
    return lines


def fmt_tree(root_arg: str, level: int, result: Any) -> str:
    root = result.root
    if not root.name:
        body = '.'
    else:
        lines = [root.name]
        children = list(root.children)
        for i, child in enumerate(children):
            lines.extend(_tree_lines(child, '', i == len(children) - 1))
        body = '\n'.join(lines)
    body = _truncated(result, body, 'tree hit a limit; use a narrower root or search')
    level_arg = f' -L {level}' if level > 0 else ''
    return f'$ tree{level_arg} {root_arg or "/"}\n{body}'


def fmt_list(path: str, result: Any) -> str:
    body = (
        '\n'.join(f'{e.name}/' if e.kind == NodeKind.NODE_KIND_DIR else e.name for e in result.entries)
        or '.'
    )
    return f'$ ls {path}\n{body}'


def fmt_read(path: str, start_line: int, end_line: int, number: bool, result: Any) -> str:
    if start_line > 0 or end_line > 0:
        cmd = f"sed -n '{start_line or 1},{end_line or '$'}p' {path}"
    elif number:
        cmd = f'cat -n {path}'
    else:
        cmd = f'cat {path}'
    body = _truncated(result, result.content, 'file hit a limit; read a smaller start_line/end_line range')
    return f'$ {cmd}\n{body}'


def fmt_search(pattern: str, root: str, result: Any) -> str:
    body = '\n'.join(f'{m.path}:{m.line}:{m.line_text}' for m in result.matches) or '(no matches)'
    body = _truncated(result, body, 'match limit reached; narrow pattern/root or raise limit')
    return f'$ rg -n -e {shlex.quote(pattern)} {shlex.quote(root or "/")}\n{body}'


def fmt_find(name: str, root: str, result: Any) -> str:
    body = '\n'.join(result.paths) or '(nothing found)'
    body = _truncated(result, body, 'result limit reached; narrow name/root or raise limit')
    return f'$ find {shlex.quote(root or "/")} -name {shlex.quote(name)}\n{body}'


def fmt_exec(path: str, args: list[str], stdin: str, result: Any) -> str:
    cmd = ' '.join([shlex.quote(path), *(shlex.quote(a) for a in args)])
    if stdin:
        label = 'SQL' if path == '/bin/sql' else 'STDIN'
        cmd = f"{cmd} <<'{label}'\n{stdin.rstrip()}\n{label}"
    parts = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(f'stderr:\n{result.stderr.rstrip()}')
    if result.exit_code:
        parts.append(f'[exit {result.exit_code}]')
    return f'$ {cmd}\n' + ('\n'.join(parts) or '.')


async def prologue(vm: EcomVm) -> str:
    """Context every task starts from (same as the BitGN sample): tree, AGENTS.MD, date, id."""
    sections: list[str] = []
    try:
        sections.append(fmt_tree('/', 2, await vm.tree(TreeRequest(root='/', level=2))))
    except ConnectError as exc:
        sections.append(f'$ tree -L 2 /\n{format_error(exc)}')
    try:
        sections.append(fmt_read('/AGENTS.MD', 0, 0, False, await vm.read(ReadRequest(path='/AGENTS.MD'))))
    except ConnectError as exc:
        sections.append(f'$ cat /AGENTS.MD\n{format_error(exc)}')
    for binary in ('/bin/date', '/bin/id'):
        try:
            sections.append(fmt_exec(binary, [], '', await vm.exec(ExecRequest(path=binary))))
        except ConnectError as exc:
            sections.append(f'$ {binary}\n{format_error(exc)}')
    return '\n\n'.join(sections)


async def submit_answer(deps: EcomDeps, completion: TaskCompletion) -> None:
    """Send the final answer exactly once per trial."""
    if deps.answered is not None:
        return
    await deps.vm.answer(
        AnswerRequest(
            message=completion.message,
            outcome=OUTCOME_BY_NAME[completion.outcome],
            refs=completion.grounding_refs,
        )
    )
    deps.answered = completion
