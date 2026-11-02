"""Settings from environment variables (and an optional `.env`)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

ZAI_CODING_PLAN_URL = 'https://api.z.ai/api/coding/paas/v4'

# A/B capabilities that can be switched on via AGENT_FLAGS / --flags.
KNOWN_FLAGS = frozenset({'reminders', 'planning', 'tool_judge', 'defender', 'defender_ml', 'trajectory'})


def parse_flags(raw: str | None) -> frozenset[str]:
    flags = frozenset(f.strip() for f in (raw or '').split(',') if f.strip())
    unknown = flags - KNOWN_FLAGS
    if unknown:
        raise ValueError(f'Unknown AGENT_FLAGS: {sorted(unknown)}; known: {sorted(KNOWN_FLAGS)}')
    return flags


@dataclass(frozen=True)
class Settings:
    zai_api_key: str = field(default_factory=lambda: os.getenv('ZAI_API_KEY', ''))
    zai_base_url: str = field(default_factory=lambda: os.getenv('ZAI_BASE_URL') or ZAI_CODING_PLAN_URL)
    model_id: str = field(default_factory=lambda: os.getenv('MODEL_ID') or 'glm-5.3')
    fallback_model_id: str = field(default_factory=lambda: os.getenv('FALLBACK_MODEL_ID', 'glm-5.2'))
    judge_model_id: str = field(default_factory=lambda: os.getenv('JUDGE_MODEL_ID') or 'glm-5.3-flash')
    thinking_effort: str = field(default_factory=lambda: os.getenv('THINKING_EFFORT') or 'high')

    bitgn_api_key: str = field(default_factory=lambda: os.getenv('BITGN_API_KEY', ''))
    bitgn_host: str = field(default_factory=lambda: os.getenv('BITGN_HOST') or 'https://api.bitgn.com')
    benchmark_id: str = field(
        default_factory=lambda: os.getenv('BENCHMARK_ID') or os.getenv('BENCH_ID') or 'bitgn/ecom1-dev'
    )

    max_steps: int = field(default_factory=lambda: int(os.getenv('MAX_STEPS') or 40))
    concurrency: int = field(default_factory=lambda: int(os.getenv('CONCURRENCY') or 1))
    flags: frozenset[str] = field(default_factory=lambda: parse_flags(os.getenv('AGENT_FLAGS')))

    def describe(self) -> dict[str, object]:
        """Non-secret config snapshot, attached to eval reports and trace spans."""
        return {
            'model_id': self.model_id,
            'fallback_model_id': self.fallback_model_id,
            'thinking_effort': self.thinking_effort,
            'max_steps': self.max_steps,
            'flags': sorted(self.flags),
            'benchmark_id': self.benchmark_id,
        }
