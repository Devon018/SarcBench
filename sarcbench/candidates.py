from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sarcbench.config import ModelSpec, RunConfig
from sarcbench.http_client import call_text_model
from sarcbench.io import TASK_FILES, append_jsonl, iter_jsonl, load_completed_ids, task_path, write_json
from sarcbench.metrics import aggregate_task1, score_task1
from sarcbench.prompts import load_prompt_templates, render_candidate_prompt
from sarcbench.styles import TASK2_PAPER_STYLES, parse_task2_styles, task2_instance_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SarcBench paper candidate-model inference.")
    parser.add_argument("--config", default="configs/paper_models.json", help="Path to the paper model roster config.")
    parser.add_argument("--models", default="all", help="Candidate aliases to run, comma-separated, or 'all'.")
    parser.add_argument("--tasks", default="task1,task2,task3", help="Tasks to run, comma-separated.")
    parser.add_argument(
        "--task2-styles",
        default="paper",
        help="Task 2 styles to run, comma-separated. Defaults to the paper styles: teasing,pushback,neutral.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix limit for controlled partial runs.")
    parser.add_argument("--reset", action="store_true", help="Delete existing candidate outputs for selected tasks.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned work without API calls.")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample progress messages.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RunConfig.load(Path(args.config))
    candidates = config.select_candidates(args.models)
    tasks = parse_tasks(args.tasks)
    task2_styles = parse_task2_styles(args.task2_styles)
    prompts = load_prompt_templates(config.prompts_dir)

    if args.dry_run:
        print_plan(config, candidates, tasks, args.limit, stage="candidates", task2_styles=task2_styles)
        return 0

    for model in candidates:
        run_candidate_model(
            config,
            model,
            tasks,
            prompts,
            task2_styles=task2_styles,
            limit=args.limit,
            reset=args.reset,
            verbose=args.verbose,
        )
    return 0


def parse_tasks(raw: str) -> list[str]:
    tasks = [task.strip() for task in raw.split(",") if task.strip()]
    unknown = [task for task in tasks if task not in TASK_FILES]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}; expected {sorted(TASK_FILES)}")
    return tasks


def run_candidate_model(
    config: RunConfig,
    model: ModelSpec,
    tasks: list[str],
    prompts: dict[str, str],
    *,
    task2_styles: list[str],
    limit: int | None,
    reset: bool,
    verbose: bool,
) -> dict[str, Any]:
    output_dir = config.output_dir / "candidates" / model.alias
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": model.public_dict(),
        "tasks": {},
        "started_at": now(),
        "protocol": {
            "candidate_temperature": config.protocol.candidate_temperature,
            "request_timeout": config.protocol.request_timeout,
            "max_attempts": config.protocol.max_attempts,
        },
    }
    for task in tasks:
        summary["tasks"][task] = run_candidate_task(
            config,
            model,
            task,
            prompts[task],
            output_dir,
            task2_styles=task2_styles,
            limit=limit,
            reset=reset,
            verbose=verbose,
        )
    summary["finished_at"] = now()
    write_json(output_dir / "candidate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def run_candidate_task(
    config: RunConfig,
    model: ModelSpec,
    task: str,
    prompt_template: str,
    output_dir: Path,
    *,
    task2_styles: list[str],
    limit: int | None,
    reset: bool,
    verbose: bool,
) -> dict[str, Any]:
    predictions_path = output_dir / f"{task}_predictions.jsonl"
    errors_path = output_dir / f"{task}_errors.jsonl"
    metrics_path = output_dir / f"{task}_metrics.json"
    if reset:
        for path in [predictions_path, errors_path, metrics_path]:
            path.unlink(missing_ok=True)

    if task == "task2":
        completed = load_completed_ids(predictions_path, key_fields=("sample_id", "requested_style"))
        completed |= load_completed_ids(errors_path, key_fields=("sample_id", "requested_style"))
    else:
        completed = load_completed_ids(predictions_path) | load_completed_ids(errors_path)
    records = [row for row in iter_jsonl(task_path(config.data_dir, task), limit=limit)]
    if task == "task2":
        pending = [
            (row, style)
            for row in records
            for style in task2_styles
            if "||".join([row["sample_id"], style]) not in completed
        ]
    else:
        pending = [(row, None) for row in records if row["sample_id"] not in completed]
    started_at = now()
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    task1_metric_rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(config.protocol.max_workers, 1)) as executor:
        future_to_record = {
            executor.submit(
                call_candidate_once,
                config,
                model,
                task,
                prompt_template,
                record,
                requested_style=requested_style,
            ): (record, requested_style)
            for record, requested_style in pending
        }
        for index, future in enumerate(as_completed(future_to_record), start=1):
            record, requested_style = future_to_record[future]
            try:
                result = future.result()
                usage = result.get("usage") or {}
                for key in usage_totals:
                    usage_totals[key] += int(usage.get(key, 0) or 0)
                append_jsonl(predictions_path, result)
                if task == "task1":
                    task1_metric_rows.append(result["metrics"])
            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "sample_id": record["sample_id"],
                        "task": task,
                        "candidate_model": model.alias,
                        "requested_style": requested_style,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "created_at": now(),
                    },
                )
            if verbose or index % 100 == 0:
                print(f"[candidate] model={model.alias} task={task} completed={index}/{len(pending)}", flush=True)

    if task == "task1":
        existing_metrics = [row["metrics"] for row in iter_jsonl(predictions_path)]
        metrics = aggregate_task1(existing_metrics)
    else:
        metrics = {
            "n": sum(1 for _ in iter_jsonl(predictions_path)),
        }
        if task == "task2":
            metrics["num_base_samples"] = len(records)
            metrics["requested_styles"] = list(task2_styles)
    metrics.update(
        {
            "num_errors": sum(1 for _ in iter_jsonl(errors_path)),
            "prediction_file": str(predictions_path),
            "error_file": str(errors_path),
            "usage": usage_totals,
            "started_at": started_at,
            "finished_at": now(),
        }
    )
    write_json(metrics_path, metrics)
    return metrics


def call_candidate_once(
    config: RunConfig,
    model: ModelSpec,
    task: str,
    prompt_template: str,
    record: dict[str, Any],
    *,
    requested_style: str | None = None,
) -> dict[str, Any]:
    prompt = render_candidate_prompt(task, prompt_template, record, requested_style=requested_style)
    attempts = max(config.protocol.max_attempts, 1)
    retry_errors: list[dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response_text, usage, raw_response = call_text_model(
                model,
                prompt,
                temperature=config.protocol.candidate_temperature,
                timeout=config.protocol.request_timeout,
                max_attempts=1,
            )
            add_usage(usage_total, usage)
            prediction = response_text.strip()
            if not prediction:
                raise ValueError("empty model response")
            result: dict[str, Any] = {
                "sample_id": record["sample_id"],
                "task": task,
                "candidate_model": model.alias,
                "prediction": prediction,
                "gold": record.get("gold"),
                "usage": usage_total,
                "attempts": attempt,
                "created_at": now(),
            }
            if task == "task2":
                if requested_style is None:
                    raise ValueError("Task 2 requires a requested_style")
                result["task_instance_id"] = task2_instance_id(record["sample_id"], requested_style)
                result["requested_style"] = requested_style
                result["reply_style_label"] = requested_style
                result["gold_reply_style_label"] = record.get("gold", {}).get("reply_style_label")
            if task == "task1":
                parsed_prediction, metrics = score_task1(record, response_text)
                result["parsed_prediction"] = parsed_prediction
                result["metrics"] = metrics
            if retry_errors:
                result["retry_errors"] = retry_errors
            if raw_response:
                result["raw_response"] = raw_response
            return result
        except Exception as exc:
            last_error = exc
            retry_errors.append({"attempt": attempt, "error_type": type(exc).__name__, "error": str(exc)})

    raise ValueError(f"failed after {attempts} attempts: {retry_errors}") from last_error


def add_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in total:
        total[key] += int(usage.get(key, 0) or 0)


def print_plan(
    config: RunConfig,
    models: list[ModelSpec],
    tasks: list[str],
    limit: int | None,
    *,
    stage: str,
    task2_styles: list[str] | None = None,
) -> None:
    payload = {
        "stage": stage,
        "config": str(config.config_path),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "prompts_dir": str(config.prompts_dir),
        "tasks": tasks,
        "task2_styles": task2_styles or list(TASK2_PAPER_STYLES),
        "limit": limit,
        "models": [model.public_dict() for model in models],
        "protocol": config.protocol.__dict__,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
