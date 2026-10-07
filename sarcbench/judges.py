from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sarcbench.candidates import add_usage, parse_tasks, print_plan
from sarcbench.config import ModelSpec, RunConfig
from sarcbench.http_client import call_text_model
from sarcbench.io import append_jsonl, iter_jsonl, load_completed_ids, load_jsonl, task_path, write_json
from sarcbench.metrics import parse_json_object, summarize_judgment_rows, filter_judgment_rows, judgment_instance_key
from sarcbench.families import same_family
from sarcbench.prompts import load_prompt_templates, render_task2_judge_prompt, render_task3_judge_prompt
from sarcbench.styles import task2_instance_id


JUDGE_SCHEMA = {
    "task2": {
        "criteria": [
            "conversational_fit",
            "addresses_last_comment",
            "avoids_literal_trap",
            "style_match",
            "socially_appropriate",
            "natural_style",
            "overall_pass",
        ],
        "explanation": [
            "conversational_fit",
            "addresses_last_comment",
            "avoids_literal_trap",
            "style_match",
            "socially_appropriate",
            "natural_style",
            "overall",
        ],
    },
    "task3": {
        "criteria": [
            "meaning_preserved",
            "sarcastic_tone",
            "context_fit",
            "target_consistency",
            "mechanism_alignment",
            "natural_style",
            "overall_pass",
        ],
        "explanation": [
            "meaning_preserved",
            "sarcastic_tone",
            "context_fit",
            "target_consistency",
            "mechanism_alignment",
            "natural_style",
            "overall",
        ],
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SarcBench paper LLM-as-a-judge evaluation.")
    parser.add_argument("--config", default="configs/paper_models.json", help="Path to the paper model roster config.")
    parser.add_argument("--candidates", default="all", help="Candidate aliases to judge, comma-separated, or 'all'.")
    parser.add_argument("--judges", default="all", help="Judge aliases to run, comma-separated, or 'all'.")
    parser.add_argument("--tasks", default="task2,task3", help="Open-ended tasks to judge.")
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix limit over candidate prediction rows.")
    parser.add_argument("--reset", action="store_true", help="Delete existing judge outputs for selected tasks.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned work without API calls.")
    parser.add_argument("--verbose", action="store_true", help="Print per-judgment progress messages.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RunConfig.load(Path(args.config))
    candidates = config.select_candidates(args.candidates)
    judges = config.select_judges(args.judges)
    tasks = [task for task in parse_tasks(args.tasks) if task in {"task2", "task3"}]
    prompts = load_prompt_templates(config.prompts_dir)

    if args.dry_run:
        print_plan(config, judges, tasks, args.limit, stage="judges")
        print("candidate_aliases=" + ",".join(model.alias for model in candidates))
        return 0

    for candidate in candidates:
        run_judges_for_candidate(
            config,
            candidate,
            judges,
            tasks,
            prompts,
            limit=args.limit,
            reset=args.reset,
            verbose=args.verbose,
        )
    return 0


def run_judges_for_candidate(
    config: RunConfig,
    candidate: ModelSpec,
    judges: list[ModelSpec],
    tasks: list[str],
    prompts: dict[str, str],
    *,
    limit: int | None,
    reset: bool,
    verbose: bool,
) -> dict[str, Any]:
    output_dir = config.output_dir / "judgments" / candidate.alias
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "candidate_model": candidate.public_dict(),
        "judge_models": [judge.public_dict() for judge in judges],
        "judge_repeats": config.protocol.judge_repeats,
        "tasks": {},
        "started_at": now(),
    }
    for task in tasks:
        summary["tasks"][task] = run_judge_task(
            config,
            candidate,
            judges,
            task,
            prompts,
            output_dir,
            limit=limit,
            reset=reset,
            verbose=verbose,
        )
    summary["finished_at"] = now()
    write_json(output_dir / "judge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def run_judge_task(
    config: RunConfig,
    candidate: ModelSpec,
    judges: list[ModelSpec],
    task: str,
    prompts: dict[str, str],
    output_dir: Path,
    *,
    limit: int | None,
    reset: bool,
    verbose: bool,
) -> dict[str, Any]:
    judgments_path = output_dir / f"{task}_judgments.jsonl"
    errors_path = output_dir / f"{task}_judge_errors.jsonl"
    sample_scores_path = output_dir / f"{task}_scores_by_sample.jsonl"
    metrics_path = output_dir / f"{task}_judge_metrics.json"
    if reset:
        for path in [judgments_path, errors_path, sample_scores_path, metrics_path]:
            path.unlink(missing_ok=True)

    prediction_path = config.output_dir / "candidates" / candidate.alias / f"{task}_predictions.jsonl"
    prediction_rows = load_jsonl(prediction_path, limit=limit)
    benchmark_records = {row["sample_id"]: row for row in iter_jsonl(task_path(config.data_dir, task))}
    if task == "task2":
        completed = load_completed_ids(
            judgments_path,
            key_fields=("sample_id", "requested_style", "judge_model", "judge_repeat"),
        )
        completed |= load_completed_ids(
            errors_path,
            key_fields=("sample_id", "requested_style", "judge_model", "judge_repeat"),
        )
    else:
        completed = load_completed_ids(judgments_path, key_fields=("sample_id", "judge_model", "judge_repeat"))
        completed |= load_completed_ids(errors_path, key_fields=("sample_id", "judge_model", "judge_repeat"))

    eligible_judges = [judge for judge in judges if not config.protocol.exclude_self_family or not same_family(candidate.alias, judge.alias, candidate.family, judge.family)]
    if not eligible_judges:
        raise ValueError(f"No non-self judges remain for {candidate.alias}")
    pending = []
    for row in prediction_rows:
        for judge in eligible_judges:
            for repeat in range(1, config.protocol.judge_repeats + 1):
                key = "||".join([row["sample_id"], judge.alias, str(repeat)])
                if task == "task2":
                    key = "||".join([row["sample_id"], str(row.get("requested_style", "")), judge.alias, str(repeat)])
                if key not in completed:
                    pending.append((row, judge, repeat))

    usage_totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    with ThreadPoolExecutor(max_workers=max(config.protocol.judge_workers, 1)) as executor:
        future_to_payload = {
            executor.submit(call_judge_once, config, task, prompts, benchmark_records[row["sample_id"]], row, judge, repeat): (
                row,
                judge,
                repeat,
            )
            for row, judge, repeat in pending
        }
        for index, future in enumerate(as_completed(future_to_payload), start=1):
            row, judge, repeat = future_to_payload[future]
            try:
                result = future.result()
                usage = result.get("usage") or {}
                for key in usage_totals:
                    usage_totals[key] += int(usage.get(key, 0) or 0)
                append_jsonl(judgments_path, result)
            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "sample_id": row["sample_id"],
                        "task": task,
                        "candidate_model": candidate.alias,
                        "judge_model": judge.alias,
                        "judge_repeat": repeat,
                        "requested_style": row.get("requested_style"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "created_at": now(),
                    },
                )
            if verbose or index % 100 == 0:
                print(f"[judge] candidate={candidate.alias} task={task} completed={index}/{len(pending)}", flush=True)

    requested_keys = {judgment_instance_key(row, task) for row in prediction_rows}
    judgment_rows = [row for row in load_jsonl(judgments_path) if judgment_instance_key(row, task) in requested_keys]
    for row in judgment_rows:
        row["candidate_family"] = candidate.family
        row["judge_family"] = next((judge.family for judge in judges if judge.alias == row["judge_model"]), row.get("judge_family"))
    metrics = summarize_judgment_rows(judgment_rows, task, exclude_self_family=config.protocol.exclude_self_family)
    retained_rows = filter_judgment_rows(judgment_rows, exclude_self_family=config.protocol.exclude_self_family)
    valid_keys = {judgment_instance_key(row, task) for row in retained_rows}
    metrics.update(
        {
            "candidate_model": candidate.alias,
            "judgment_file": str(judgments_path),
            "sample_score_file": str(sample_scores_path),
            "error_file": str(errors_path),
            "num_errors": sum(1 for _ in iter_jsonl(errors_path)),
            "raw_judgments_requested_per_sample": len(judges) * config.protocol.judge_repeats,
            "expected_judgments_per_sample": len(eligible_judges) * config.protocol.judge_repeats,
            "num_predictions_attempted": len(prediction_rows),
            "num_predictions_without_valid_judgments": len(requested_keys - valid_keys),
            "num_expected_nonself_calls": len(prediction_rows) * len(eligible_judges) * config.protocol.judge_repeats,
            "num_missing_nonself_calls": len(prediction_rows) * len(eligible_judges) * config.protocol.judge_repeats - len(retained_rows),
            "usage": usage_totals,
            "finished_at": now(),
        }
    )
    write_sample_scores(sample_scores_path, retained_rows, task, exclude_self_family=config.protocol.exclude_self_family)
    write_json(metrics_path, metrics)
    return metrics


def call_judge_once(
    config: RunConfig,
    task: str,
    prompts: dict[str, str],
    benchmark_record: dict[str, Any],
    prediction_row: dict[str, Any],
    judge: ModelSpec,
    repeat: int,
) -> dict[str, Any]:
    candidate_text = prediction_row["prediction"]
    if task == "task2":
        requested_style = str(
            prediction_row.get("requested_style")
            or prediction_row.get("reply_style_label")
            or benchmark_record.get("gold", {}).get("reply_style_label", "")
        )
        prompt = render_task2_judge_prompt(
            prompts["task2_judge"],
            benchmark_record,
            candidate_text,
            requested_style=requested_style,
        )
    elif task == "task3":
        prompt = render_task3_judge_prompt(prompts["task3_judge"], benchmark_record, candidate_text)
    else:
        raise ValueError(f"Unsupported judge task: {task}")
    attempts = max(config.protocol.max_attempts, 1)
    retry_errors: list[dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response_text, usage, raw_response = call_text_model(
                judge,
                prompt,
                temperature=config.protocol.judge_temperature,
                timeout=judge.request_timeout or config.protocol.request_timeout,
                max_attempts=1,
            )
            add_usage(usage_total, usage)
            judgment = parse_json_object(response_text)
            validate_judgment_schema(task, judgment)
            result: dict[str, Any] = {
                "sample_id": benchmark_record["sample_id"],
                "task": task,
                "candidate_model": prediction_row["candidate_model"],
                "judge_model": judge.alias,
                "judge_family": judge.family,
                "candidate_family": prediction_row.get("candidate_family"),
                "judge_spec": judge.public_dict(),
                "judge_repeat": str(repeat),
                "prediction": candidate_text,
                "gold": benchmark_record.get("gold"),
                "judgment": judgment,
                "usage": usage_total,
                "attempts": attempt,
                "created_at": now(),
            }
            if task == "task2":
                result["requested_style"] = requested_style
                result["reply_style_label"] = requested_style
                result["gold_reply_style_label"] = requested_style if "reference_replies" in benchmark_record.get("gold", {}) else benchmark_record.get("gold", {}).get("reply_style_label")
                result["task_instance_id"] = task2_instance_id(benchmark_record["sample_id"], requested_style)
            if retry_errors:
                result["retry_errors"] = retry_errors
            if raw_response:
                result["raw_response"] = raw_response
            return result
        except Exception as exc:
            last_error = exc
            retry_errors.append({"attempt": attempt, "error_type": type(exc).__name__, "error": str(exc)})

    raise ValueError(f"failed after {attempts} attempts: {retry_errors}") from last_error


def validate_judgment_schema(task: str, judgment: dict[str, Any]) -> None:
    schema = JUDGE_SCHEMA[task]
    missing_criteria = [key for key in schema["criteria"] if key not in judgment]
    non_boolean = [key for key in schema["criteria"] if key in judgment and not isinstance(judgment[key], bool)]
    if missing_criteria or non_boolean:
        raise ValueError(
            "invalid judge schema: "
            f"missing boolean criteria={missing_criteria}; non_boolean criteria={non_boolean}"
        )

    explanation = judgment.get("explanation")
    if not isinstance(explanation, dict):
        raise ValueError("invalid judge schema: explanation must be an object")
    missing_explanations = [key for key in schema["explanation"] if not str(explanation.get(key, "")).strip()]
    if missing_explanations:
        raise ValueError(f"invalid judge schema: missing explanations={missing_explanations}")


def write_sample_scores(path: Path, judgment_rows: list[dict[str, Any]], task: str, *, exclude_self_family: bool = True) -> None:
    path.unlink(missing_ok=True)
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in filter_judgment_rows(judgment_rows, exclude_self_family=exclude_self_family):
        if task == "task2":
            key = task2_instance_id(row["sample_id"], str(row.get("requested_style") or row.get("reply_style_label") or ""))
        else:
            key = row["sample_id"]
        by_sample.setdefault(key, []).append(row)
    for _, rows in sorted(by_sample.items()):
        summary = summarize_judgment_rows(rows, task, exclude_self_family=exclude_self_family)
        payload = {
            "sample_id": rows[0]["sample_id"],
            "task": task,
            "candidate_model": rows[0]["candidate_model"],
            "num_judgments": summary["num_judgments"],
            "score": summary["mean_score"],
            "score_pct": summary["mean_score_pct"],
            "score_version": summary["score_version"],
            "overall_pass_rate": summary["overall_pass_rate"],
            "criteria": summary["criteria"],
            "explanations": [
                {
                    "judge_model": row["judge_model"],
                    "judge_repeat": row["judge_repeat"],
                    "explanation": row.get("judgment", {}).get("explanation", {}),
                }
                for row in sorted(rows, key=lambda item: (item["judge_model"], item["judge_repeat"]))
            ],
        }
        if task == "task2":
            payload["requested_style"] = rows[0].get("requested_style") or rows[0].get("reply_style_label")
            payload["reply_style_label"] = payload["requested_style"]
            payload["gold_reply_style_label"] = rows[0].get("gold_reply_style_label")
            payload["task_instance_id"] = rows[0].get("task_instance_id")
        append_jsonl(path, payload)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
