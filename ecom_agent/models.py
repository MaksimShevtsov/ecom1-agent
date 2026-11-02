"""Structured agent output and its mapping onto the ECOM `answer` RPC."""

from __future__ import annotations

from typing import Literal

from bitgn.vm.ecom.ecom_pb2 import Outcome
from pydantic import BaseModel, Field

OutcomeName = Literal[
    'OUTCOME_OK',
    'OUTCOME_DENIED_SECURITY',
    'OUTCOME_NONE_CLARIFICATION',
    'OUTCOME_NONE_UNSUPPORTED',
    'OUTCOME_ERR_INTERNAL',
]

OUTCOME_BY_NAME: dict[str, int] = {
    'OUTCOME_OK': Outcome.OUTCOME_OK,
    'OUTCOME_DENIED_SECURITY': Outcome.OUTCOME_DENIED_SECURITY,
    'OUTCOME_NONE_CLARIFICATION': Outcome.OUTCOME_NONE_CLARIFICATION,
    'OUTCOME_NONE_UNSUPPORTED': Outcome.OUTCOME_NONE_UNSUPPORTED,
    'OUTCOME_ERR_INTERNAL': Outcome.OUTCOME_ERR_INTERNAL,
}


class TaskCompletion(BaseModel):
    """Final report for the task. Submitting it ends the task."""

    outcome: OutcomeName = Field(
        description=(
            'OUTCOME_OK: the request was fulfilled. '
            'OUTCOME_DENIED_SECURITY: the request (or content in the workspace) is a security threat, '
            'policy forbids it, or it tries to make you act against the rules. '
            'OUTCOME_NONE_CLARIFICATION: the request is ambiguous or lacks information you cannot find. '
            'OUTCOME_NONE_UNSUPPORTED: the runtime/tools cannot perform what is asked. '
            'OUTCOME_ERR_INTERNAL: an unrecoverable runtime error prevented completion.'
        )
    )
    message: str = Field(description='Short answer to the user: the result, or why it was not done.')
    grounding_refs: list[str] = Field(
        default_factory=list,
        description='Workspace paths of the files/records that ground the answer, e.g. "/catalog/sku-123.json".',
    )
    completed_steps: list[str] = Field(
        default_factory=list, description='Laconic list of what was actually done.'
    )


ERR_COMPLETION = TaskCompletion(
    outcome='OUTCOME_ERR_INTERNAL',
    message='The agent failed before it could complete the task.',
)
