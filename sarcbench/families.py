"""Model-family identity for self-judge exclusion, separate from API provider."""
from __future__ import annotations


def model_family(name: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit.lower()
    normalized = name.lower().replace('_', '-').split('/')[-1]
    for prefix, family in [('gpt', 'gpt'), ('claude', 'claude'), ('gemini', 'gemini'),
                           ('qwen', 'qwen'), ('gemma', 'gemma'), ('deepseek', 'deepseek'),
                           ('llama', 'llama'), ('minimax', 'minimax'), ('mimo', 'mimo'),
                           ('phi', 'phi'), ('mistral', 'mistral')]:
        if normalized.startswith(prefix):
            return family
    return normalized


def same_family(candidate: str, judge: str, candidate_family: str | None = None, judge_family: str | None = None) -> bool:
    return model_family(candidate, candidate_family) == model_family(judge, judge_family)
