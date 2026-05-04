from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable


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
        payload[key] = mean(float(row.get(key, 0.0)) for row in rows)
        payload[f"{key}_pct"] = 100.0 * payload[key]
    return payload


def bool_judgment_score(judgment: dict[str, Any]) -> dict[str, float]:
    return {key: 1.0 if value else 0.0 for key, value in judgment.items() if isinstance(value, bool)}


def summarize_judgment_rows(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if task == "task2":
            requested_style = str(row.get("requested_style") or row.get("reply_style_label") or "unknown")
            key = f"{row['sample_id']}||{requested_style}"
        else:
            key = row["sample_id"]
        by_sample[key].append(row)

    sample_scores = []
    for sample_id, sample_rows in by_sample.items():
        criterion_totals: dict[str, list[float]] = defaultdict(list)
        for row in sample_rows:
            for key, value in bool_judgment_score(row["judgment"]).items():
                criterion_totals[key].append(value)
        sample_summary = {
            "sample_id": sample_rows[0]["sample_id"],
            "num_judgments": len(sample_rows),
            "score": mean(criterion_totals.get("overall_pass", [])),
            "criteria": {key: mean(values) for key, values in sorted(criterion_totals.items())},
        }
        if task == "task2":
            sample_summary["requested_style"] = sample_rows[0].get("requested_style") or sample_rows[0].get("reply_style_label")
            sample_summary["reply_style_label"] = sample_summary["requested_style"]
        sample_scores.append(sample_summary)

    payload: dict[str, Any] = {
        "num_samples": len(sample_scores),
        "num_judgments": len(rows),
        "mean_score": mean(row["score"] for row in sample_scores),
    }
    payload["mean_score_pct"] = 100.0 * payload["mean_score"]

    all_criteria: dict[str, list[float]] = defaultdict(list)
    by_judge: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        scores = bool_judgment_score(row["judgment"])
        for key, value in scores.items():
            all_criteria[key].append(value)
        by_judge[row["judge_model"]].append(scores.get("overall_pass", 0.0))
    payload["criteria"] = {key: mean(values) for key, values in sorted(all_criteria.items())}
    payload["criteria_pct"] = {key: 100.0 * value for key, value in payload["criteria"].items()}
    payload["by_judge"] = {
        judge: {"num_judgments": len(values), "mean_score": mean(values), "mean_score_pct": 100.0 * mean(values)}
        for judge, values in sorted(by_judge.items())
    }

    if task == "task2":
        by_style: dict[str, list[float]] = defaultdict(list)
        for sample in sample_scores:
            by_style[str(sample.get("requested_style") or "unknown")].append(sample["score"])
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
