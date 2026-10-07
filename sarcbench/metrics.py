from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable

from sarcbench.families import same_family


MECHANISM_ALIASES = {
    "polarity reversal": "polarity_reversal",
    "polarity_reversal": "polarity_reversal",
    "reversal": "polarity_reversal",
    "inversion": "polarity_reversal",
    "exaggeration": "exaggeration",
    "understatement": "understatement",
    "rhetorical question": "rhetorical_question",
    "rhetorical_question": "rhetorical_question",
    "contrast with context": "contrast_with_context",
    "contrast_with_context": "contrast_with_context",
    "contrast": "contrast_with_context",
    "sarcasm marker": "sarcasm_marker",
    "sarcasm_marker": "sarcasm_marker",
    "marker": "sarcasm_marker",
    "other": "other",
}


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model response")
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"failed to parse JSON object from response prefix: {stripped[:240]!r}")


def score_task1(record: dict[str, Any], response_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_json_object(response_text)
    prediction = {
        "Intent": "".join(sorted(normalize_letters(parsed.get("Intent")))),
        "Target": "".join(sorted(normalize_letters(parsed.get("Target")))),
        "Mechanism": sorted(normalize_mechanisms(parsed.get("Mechanism"))),
    }
    gold = record["gold"]
    gold_intent = "".join(sorted(normalize_letters(gold["Intent"])))
    gold_target = "".join(sorted(normalize_letters(gold["Target"])))
    gold_mechanism = sorted(normalize_mechanisms(gold["Mechanism"]))

    intent_exact = prediction["Intent"] == gold_intent
    target_exact = prediction["Target"] == gold_target
    mechanism_exact = prediction["Mechanism"] == gold_mechanism
    metrics = {
        "intent_exact": intent_exact,
        "target_exact": target_exact,
        "mechanism_exact": mechanism_exact,
        "mechanism_f1": set_f1(set(prediction["Mechanism"]), set(gold_mechanism)),
        "joint_exact": intent_exact and target_exact and mechanism_exact,
        "task1_macro_accuracy": (float(intent_exact) + float(target_exact) + float(mechanism_exact)) / 3.0,
    }
    return prediction, metrics


def normalize_letters(value: Any) -> set[str]:
    if isinstance(value, list):
        text = "".join(str(item) for item in value)
    else:
        text = str(value or "")
    return set(re.findall(r"[A-H]", text.upper()))


def normalize_mechanisms(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw_parts = re.split(r"[,;/|]+", value)
    elif isinstance(value, list):
        raw_parts = [str(item) for item in value]
    else:
        raw_parts = [str(value)]
    normalized = set()
    for part in raw_parts:
        key = part.strip().lower().replace("-", "_")
        key = key.replace("_", " ")
        alias = MECHANISM_ALIASES.get(key) or MECHANISM_ALIASES.get(key.replace(" ", "_"))
        if alias:
            normalized.add(alias)
    return normalized


def set_f1(prediction: set[str], gold: set[str]) -> float:
    if not prediction and not gold:
        return 1.0
    if not prediction or not gold:
        return 0.0
    overlap = len(prediction & gold)
    precision = overlap / len(prediction)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def aggregate_task1(metric_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(metric_rows)
    keys = ["intent_exact", "target_exact", "mechanism_exact", "mechanism_f1", "joint_exact", "task1_macro_accuracy"]
    payload: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        payload[key] = mean(float(row.get(key, 0.0)) for row in rows) if rows else None
        payload[f"{key}_pct"] = None if payload[key] is None else 100.0 * payload[key]
    return payload


def bool_judgment_score(judgment: dict[str, Any]) -> dict[str, float]:
    return {key: 1.0 if value else 0.0 for key, value in judgment.items() if isinstance(value, bool)}


SCORED_CRITERIA = {
    "task2": ("conversational_fit", "addresses_last_comment", "avoids_literal_trap", "style_match", "socially_appropriate", "natural_style"),
    "task3": ("meaning_preserved", "sarcastic_tone", "context_fit", "target_consistency", "mechanism_alignment", "natural_style"),
}
SCORE_VERSION = "six-criteria-nonself-v1"


def filter_judgment_rows(rows: list[dict[str, Any]], *, exclude_self_family: bool = True) -> list[dict[str, Any]]:
    if not exclude_self_family:
        return list(rows)
    return [row for row in rows if not same_family(
        str(row["candidate_model"]), str(row["judge_model"]),
        row.get("candidate_family"), row.get("judge_family"),
    )]


def judgment_score(judgment: dict[str, Any], task: str) -> float:
    criteria = SCORED_CRITERIA[task]
    invalid = [key for key in criteria if not isinstance(judgment.get(key), bool)]
    if invalid:
        raise ValueError(f"Missing/non-boolean scored criteria: {invalid}")
    # overall_pass is diagnostic only. Its value cannot affect the paper score.
    return sum(float(judgment[key]) for key in criteria) / len(criteria)


def judgment_instance_key(row: dict[str, Any], task: str) -> str:
    if task == "task2":
        style = row.get("requested_style") or row.get("reply_style_label")
        if not style:
            raise ValueError("A Response judgment must identify its requested style.")
        return f"{row['sample_id']}||{style}"
    return str(row["sample_id"])


def summarize_judgment_rows(rows: list[dict[str, Any]], task: str, *, exclude_self_family: bool = True) -> dict[str, Any]:
    retained = filter_judgment_rows(rows, exclude_self_family=exclude_self_family)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retained:
        by_sample[judgment_instance_key(row, task)].append(row)
    sample_scores = []
    for _, sample_rows in sorted(by_sample.items()):
        scores = [judgment_score(row["judgment"], task) for row in sample_rows]
        criteria = {key: mean(float(row["judgment"][key]) for row in sample_rows) for key in SCORED_CRITERIA[task]}
        sample_summary = {
            "sample_id": sample_rows[0]["sample_id"],
            "num_judgments": len(sample_rows),
            "score": mean(scores),
            "criteria": criteria,
            "overall_pass_rate": mean(float(row["judgment"].get("overall_pass", False)) for row in sample_rows),
        }
        if task == "task2":
            sample_summary["requested_style"] = sample_rows[0].get("requested_style") or sample_rows[0].get("reply_style_label")
            sample_summary["reply_style_label"] = sample_summary["requested_style"]
        sample_scores.append(sample_summary)

    def optional_mean(values):
        values = list(values)
        return mean(values) if values else None

    score = optional_mean(sample["score"] for sample in sample_scores)
    payload: dict[str, Any] = {
        "score_version": SCORE_VERSION,
        "exclude_self_family": exclude_self_family,
        "num_raw_judgments": len(rows),
        "num_self_judgments_excluded": len(rows) - len(retained),
        "num_samples": len(sample_scores),
        "num_judgments": len(retained),
        "mean_score": score,
        "mean_score_pct": None if score is None else 100.0 * score,
        "failure_policy": "exclude_and_report; no valid non-self judgments yields null, not zero",
    }
    # Task-level criteria use the same equal-instance weighting as mean_score.
    payload["criteria"] = {key: optional_mean(sample["criteria"][key] for sample in sample_scores) for key in SCORED_CRITERIA[task]}
    payload["criteria_pct"] = {key: None if value is None else 100.0 * value for key, value in payload["criteria"].items()}
    payload["overall_pass_rate"] = optional_mean(sample["overall_pass_rate"] for sample in sample_scores)
    by_judge: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in retained:
        by_judge[row["judge_model"]][judgment_instance_key(row, task)].append(judgment_score(row["judgment"], task))
    payload["by_judge"] = {
        judge: {
            "num_judgments": sum(len(values) for values in instances.values()),
            "num_samples": len(instances),
            "mean_score": mean(mean(values) for values in instances.values()),
            "mean_score_pct": 100.0 * mean(mean(values) for values in instances.values()),
        } for judge, instances in sorted(by_judge.items())
    }
    if task == "task2":
        by_style: dict[str, list[float]] = defaultdict(list)
        for sample in sample_scores:
            by_style[str(sample["requested_style"])].append(sample["score"])
        payload["by_style"] = {
            style: {"num_samples": len(values), "mean_score": mean(values), "mean_score_pct": 100.0 * mean(values)}
            for style, values in sorted(by_style.items())
        }
    return payload


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return sum(rows) / len(rows)
