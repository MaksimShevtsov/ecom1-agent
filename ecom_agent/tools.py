"""Agent tools over the ECOM runtime.

Two toolsets:
- `explore_tools` is read-only and tagged `code_mode=True`, so `CodeMode` moves it behind `run_code`
  (the model can batch SQL/read/search in one sandboxed script).
- `mutate_tools` changes state and stays as explicit tool calls, so every write is visible in the trace
  and can be gated by a tool-call judge.
"""

from __future__ import annotations

from typing import Annotated, Literal

from bitgn.vm.ecom.ecom_pb2 import (
    DeleteRequest,
    ExecRequest,
    FindRequest,
    ListRequest,
    NodeKind,
    ReadRequest,
    SearchRequest,
    StatRequest,
    TreeRequest,
    WriteRequest,
)
from connectrpc.errors import ConnectError
from google.protobuf.json_format import MessageToDict
from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from .models import TaskCompletion
from .runtime import (
    EcomDeps,
    fmt_exec,
    fmt_find,
    fmt_list,
    fmt_read,
    fmt_search,
    fmt_tree,
    format_error,
    submit_answer,
)

MUTATING_TOOLS = ('write_file', 'delete_file', 'exec')

Limit = Annotated[int, Field(ge=1, le=20)]

explore_tools: FunctionToolset[EcomDeps] = FunctionToolset()
mutate_tools: FunctionToolset[EcomDeps] = FunctionToolset()


@explore_tools.tool
async def tree(ctx: RunContext[EcomDeps], root: str = '/', level: int = 2) -> str:
    """Show the directory tree under `root`, `level` deep (0 = unlimited)."""
    try:
        return fmt_tree(root, level, await ctx.deps.vm.tree(TreeRequest(root=root, level=level)))
    except ConnectError as exc:
        return format_error(exc)


@explore_tools.tool
async def list_dir(ctx: RunContext[EcomDeps], path: str = '/') -> str:
    """List entries of one directory (directories end with `/`)."""
    try:
        return fmt_list(path, await ctx.deps.vm.list(ListRequest(path=path)))
    except ConnectError as exc:
        return format_error(exc)


@explore_tools.tool
async def find(
    ctx: RunContext[EcomDeps],
    name: str,
    root: str = '/',
    kind: Literal['all', 'files', 'dirs'] = 'all',
    limit: Limit = 10,
) -> str:
    """Find files/directories by name under `root`."""
    node_kind = {
        'all': NodeKind.NODE_KIND_UNSPECIFIED,
        'files': NodeKind.NODE_KIND_FILE,
        'dirs': NodeKind.NODE_KIND_DIR,
    }[kind]
    try:
        result = await ctx.deps.vm.find(FindRequest(root=root, name=name, kind=node_kind, limit=limit))
        return fmt_find(name, root, result)
    except ConnectError as exc:
        return format_error(exc)


@explore_tools.tool
async def search(ctx: RunContext[EcomDeps], pattern: str, root: str = '/', limit: Limit = 10) -> str:
    """Search file contents with a regex `pattern` under `root`; returns `path:line:text` matches."""
    try:
        result = await ctx.deps.vm.search(SearchRequest(root=root, pattern=pattern, limit=limit))
        return fmt_search(pattern, root, result)
    except ConnectError as exc:
        return format_error(exc)


@explore_tools.tool
async def read_file(
    ctx: RunContext[EcomDeps],
    path: str,
    start_line: Annotated[int, Field(ge=0)] = 0,
    end_line: Annotated[int, Field(ge=0)] = 0,
    number: bool = False,
) -> str:
    """Read a file. `start_line`/`end_line` are 1-based inclusive (0 = from start / to end); `number` adds line numbers."""
    try:
        result = await ctx.deps.vm.read(
            ReadRequest(path=path, number=number, start_line=start_line, end_line=end_line)
        )
        return fmt_read(path, start_line, end_line, number, result)
    except ConnectError as exc:
        return format_error(exc)


@explore_tools.tool
async def stat(ctx: RunContext[EcomDeps], path: str) -> str:
    """Show metadata of a path: kind, content type, whether it is writable."""
    try:
        return str(MessageToDict(await ctx.deps.vm.stat(StatRequest(path=path))))
    except ConnectError as exc:
        return format_error(exc)


@explore_tools.tool
async def sql(ctx: RunContext[EcomDeps], query: str) -> str:
    """Run a SQL query against the catalogue via `/bin/sql`. Prefer it over reading many files."""
    try:
        result = await ctx.deps.vm.exec(ExecRequest(path='/bin/sql', stdin=query))
        return fmt_exec('/bin/sql', [], query, result)
    except ConnectError as exc:
        return format_error(exc)


@mutate_tools.tool
async def write_file(ctx: RunContext[EcomDeps], path: str, content: str) -> str:
    """Create or overwrite a file with the full `content`. Keep edits small and targeted."""
    try:
        await ctx.deps.vm.write(WriteRequest(path=path, content=content))
    except ConnectError as exc:
        return format_error(exc)
    ctx.deps.mutations.append(f'write {path}')
    return f'wrote {path} ({len(content)} chars)'


@mutate_tools.tool
async def delete_file(ctx: RunContext[EcomDeps], path: str) -> str:
    """Delete a file."""
    try:
        await ctx.deps.vm.delete(DeleteRequest(path=path))
    except ConnectError as exc:
        return format_error(exc)
    ctx.deps.mutations.append(f'delete {path}')
    return f'deleted {path}'


@mutate_tools.tool
async def exec(
    ctx: RunContext[EcomDeps], path: str, args: list[str] | None = None, stdin: str = ''
) -> str:
    """Run a runtime binary (e.g. under `/bin`) with `args` and optional `stdin`. May change state."""
    args = args or []
    try:
        result = await ctx.deps.vm.exec(ExecRequest(path=path, args=args, stdin=stdin))
    except ConnectError as exc:
        return format_error(exc)
    ctx.deps.mutations.append(f'exec {path} {" ".join(args)}'.strip())
    return fmt_exec(path, args, stdin, result)


# Chat mode only: the human approves the final answer in the web UI before it is sent.
chat_tools: FunctionToolset[EcomDeps] = FunctionToolset()


@chat_tools.tool(requires_approval=True)
async def report_completion(ctx: RunContext[EcomDeps], completion: TaskCompletion) -> str:
    """Submit the final answer for the task. This ends the task; it can be sent only once."""
    if ctx.deps.answered is not None:
        return f'Already answered with {ctx.deps.answered.outcome}.'
    try:
        await submit_answer(ctx.deps, completion)
    except ConnectError as exc:
        return format_error(exc)
    return f'Answer submitted: {completion.outcome}.'
