"""In-memory stand-in for the ECOM runtime, returning real protobuf responses."""

from __future__ import annotations

from bitgn.vm.ecom import ecom_pb2 as pb
from connectrpc.code import Code
from connectrpc.errors import ConnectError


class FakeVm:
    def __init__(self, files: dict[str, str] | None = None, sql_stdout: str = 'sku|stock\nA1|3\n'):
        self.files = dict(files or {})
        self.sql_stdout = sql_stdout
        self.answers: list[pb.AnswerRequest] = []
        self.execs: list[pb.ExecRequest] = []

    def _missing(self, path: str) -> ConnectError:
        return ConnectError(Code.NOT_FOUND, f'{path}: not found')

    async def tree(self, request):
        root = pb.TreeResponse()
        root.root.name = '/'
        for path in sorted(self.files):
            root.root.children.add(name=path.lstrip('/'))
        return root

    async def list(self, request):
        res = pb.ListResponse(path=request.path)
        for path in sorted(self.files):
            res.entries.add(name=path.rsplit('/', 1)[-1], kind=pb.NodeKind.NODE_KIND_FILE)
        return res

    async def find(self, request):
        return pb.FindResponse(paths=[p for p in self.files if request.name in p])

    async def search(self, request):
        res = pb.SearchResponse()
        for path, text in self.files.items():
            for i, line in enumerate(text.splitlines(), 1):
                if request.pattern in line:
                    res.matches.add(path=path, line=i, line_text=line)
        return res

    async def read(self, request):
        if request.path not in self.files:
            raise self._missing(request.path)
        return pb.ReadResponse(path=request.path, content=self.files[request.path])

    async def stat(self, request):
        if request.path not in self.files:
            raise self._missing(request.path)
        return pb.StatResponse(path=request.path, kind=pb.NodeKind.NODE_KIND_FILE, writable=True)

    async def write(self, request):
        self.files[request.path] = request.content
        return pb.WriteResponse(path=request.path)

    async def delete(self, request):
        if self.files.pop(request.path, None) is None:
            raise self._missing(request.path)
        return pb.DeleteResponse()

    async def exec(self, request):
        self.execs.append(request)
        if request.path == '/bin/sql':
            return pb.ExecResponse(stdout=self.sql_stdout)
        if request.path == '/bin/date':
            return pb.ExecResponse(stdout='2026-11-01T00:00:00Z')
        if request.path == '/bin/id':
            return pb.ExecResponse(stdout='uid=ops role=operator')
        return pb.ExecResponse(stderr=f'{request.path}: not found', exit_code=127)

    async def answer(self, request):
        self.answers.append(request)
        return pb.AnswerResponse()
