from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompt_templates(prompts_dir: Path) -> dict[str, str]:
    return {
        "task1": (prompts_dir / "task1.txt").read_text(encoding="utf-8"),
        "task2": (prompts_dir / "task2.txt").read_text(encoding="utf-8"),
        "task3": (prompts_dir / "task3.txt").read_text(encoding="utf-8"),
        "task2_judge": (prompts_dir / "task2_judge.txt").read_text(encoding="utf-8"),
        "task3_judge": (prompts_dir / "task3_judge.txt").read_text(encoding="utf-8"),
    }


def render_candidate_prompt(task: str, template: str, record: dict[str, Any]) -> str:
    if task == "task1":
        payload = {
            "comments": record["comments"],
            "IntentOptions": record["IntentOptions"],
            "TargetOptions": record["TargetOptions"],
        }
        return template.replace("__TASK_INPUT_JSON__", json.dumps(payload, ensure_ascii=False, indent=2))
    if task == "task2":
        return (
            template.replace("__THREAD_TEXT__", record["comments"])
            .replace("__RESPONSE_STYLE__", str(record["gold"].get("reply_style_label", "")))
        )
    if task == "task3":
        return (
            template.replace("__THREAD_TEXT__", record["comments"])
            .replace("__INTENDED_MEANING__", record["intended_meaning"])
            .replace("__MECHANISMS__", json.dumps(record["gold"].get("preferred_mechanisms", []), ensure_ascii=False))
        )
    raise ValueError(f"Unknown task: {task}")


def render_task2_judge_prompt(template: str, record: dict[str, Any], candidate_reply: str) -> str:
    return (
        template.replace("__THREAD_TEXT__", record["comments"])
        .replace("__RESPONSE_STYLE__", str(record["gold"].get("reply_style_label", "")))
        .replace("__REFERENCE_REPLY__", record["gold"]["reference_reply"])
        .replace("__CANDIDATE_REPLY__", candidate_reply.strip())
    )


def render_task3_judge_prompt(template: str, record: dict[str, Any], candidate_comment: str) -> str:
    return (
        template.replace("__THREAD_TEXT__", record["comments"])
        .replace("__INTENDED_MEANING__", record["intended_meaning"])
        .replace("__TARGET__", record["gold"]["target"])
        .replace("__MECHANISMS__", json.dumps(record["gold"].get("preferred_mechanisms", []), ensure_ascii=False))
        .replace("__REFERENCE_COMMENT__", record["gold"]["reference_sarcastic_comment"])
        .replace("__CANDIDATE_COMMENT__", candidate_comment.strip())
    )
