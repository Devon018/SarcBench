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


def render_candidate_prompt(
    task: str,
    template: str,
    record: dict[str, Any],
    *,
    requested_style: str | None = None,
) -> str:
    if task == "task1":
        payload = {
            "comments": record["comments"],
            "IntentOptions": record["IntentOptions"],
            "TargetOptions": record["TargetOptions"],
        }
        return template.replace("__TASK_INPUT_JSON__", json.dumps(payload, ensure_ascii=False, indent=2))
    if task == "task2":
        style = requested_style or str(record["gold"].get("reply_style_label", ""))
        return (
            template.replace("__THREAD_TEXT__", record["comments"])
            .replace("__RESPONSE_STYLE__", style)
        )
    if task == "task3":
        return (
            template.replace("__THREAD_TEXT__", record["comments"])
            .replace("__INTENDED_MEANING__", record["intended_meaning"])
            .replace("__MECHANISMS__", json.dumps(record["gold"].get("preferred_mechanisms", []), ensure_ascii=False))
        )
    raise ValueError(f"Unknown task: {task}")


def task2_reference(record: dict[str, Any], requested_style: str) -> tuple[str, str]:
    gold = record["gold"]
    if "reference_replies" in gold:
        replies = gold["reference_replies"]
        if requested_style not in replies or not isinstance(replies[requested_style], str) or not replies[requested_style].strip():
            raise ValueError(f"Missing reference for requested style {requested_style}")
        return replies[requested_style], requested_style
    # Legacy single-reference files remain usable as cross-style content anchors.
    # They cannot supply same-style human ceilings for the other styles.
    return str(gold["reference_reply"]), str(gold.get("reply_style_label", "unknown"))


def render_task2_judge_prompt(
    template: str,
    record: dict[str, Any],
    candidate_reply: str,
    *,
    requested_style: str | None = None,
) -> str:
    style = requested_style or str(record["gold"].get("reply_style_label", ""))
    reference, source_style = task2_reference(record, style)
    return (
        template.replace("__THREAD_TEXT__", record["comments"])
        .replace("__RESPONSE_STYLE__", style)
        .replace("__REFERENCE_STYLE__", source_style)
        .replace("__REFERENCE_REPLY__", reference)
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
