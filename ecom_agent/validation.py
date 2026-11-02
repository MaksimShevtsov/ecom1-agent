"""Deterministic post-processing of the final answer (the grader checks these things exactly)."""

from __future__ import annotations

import json
import re

from bitgn.vm.ecom.ecom_pb2 import ExecRequest, ReadRequest
from connectrpc.errors import ConnectError

from .runtime import EcomVm

URL_RE = re.compile(r'https?://[^\s<>"\')\]]+')
TOKEN_RE = re.compile(r'<(?:YES|NO)>')
IDENTITY_RE = re.compile(r'\b((?:cust|emp)_\w+)\b')
PRIVATE_RE = re.compile(r'^/proc/(employees|customers|baskets)/([^/]+?)(?:\.json)?$')


def clean_message(message: str, instruction: str) -> str:
    """Drop URLs the task did not mention (phishing links from data) and reduce <YES>/<NO> answers to the token."""
    def drop(m: re.Match[str]) -> str:
        return m.group(0) if m.group(0) in instruction else '[link removed]'

    message = URL_RE.sub(drop, message)
    if '<YES>' in instruction and '<NO>' in instruction:
        token = TOKEN_RE.search(message)
        if token:
            return token.group(0)
    return message


async def resolve_identity(vm: EcomVm) -> str | None:
    try:
        out = (await vm.exec(ExecRequest(path='/bin/id'))).stdout
    except ConnectError:
        return None
    m = IDENTITY_RE.search(out)
    return m.group(1) if m else None


async def drop_private_refs(vm: EcomVm, refs: list[str], identity: str | None) -> list[str]:
    """For refusals: never cite other customers' baskets/records or employee records."""
    kept: list[str] = []
    for ref in refs:
        m = PRIVATE_RE.match(ref)
        if not m:
            kept.append(ref)
            continue
        kind, name = m.groups()
        if kind == 'employees':
            if name == identity:
                kept.append(ref)
        elif kind == 'customers':
            if identity is None or name == identity:
                kept.append(ref)
        else:  # baskets: keep only the requester's own
            if identity is None:
                kept.append(ref)
                continue
            try:
                data = json.loads((await vm.read(ReadRequest(path=ref))).content)
            except (ConnectError, ValueError):
                continue
            if isinstance(data, dict) and data.get('customer_id') == identity:
                kept.append(ref)
    return kept
