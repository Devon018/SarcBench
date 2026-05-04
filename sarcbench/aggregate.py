from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from sarcbench.config import RunConfig
from sarcbench.io import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SarcBench main-experiment metrics.")
    parser.add_argument("--config", default="configs/paper_models.json", help="Path to the paper model roster config.")
    parser.add_argument("--models", default="all", help="Candidate aliases to aggregate, comma-separated, or 'all'.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RunConfig.load(Path(args.config))
    candidates = config.select_candidates(args.models)
    rows = [aggregate_candidate(config, candidate.alias, candidate.display_name) for candidate in candidates]
    output_dir = config.output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "main_results.json", {"models": rows})
    write_csv(output_dir / "main_results.csv", rows)
    print(json.dumps({"output_dir": str(output_dir), "models": rows}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def aggregate_candidate(config: RunConfig, alias: str, display_name: str) -> dict[str, Any]:
    candidate_dir = config.output_dir / "candidates" / alias
    judge_dir = config.output_dir / "judgments" / alias
    task1 = load_json(candidate_dir / "task1_metrics.json")
    task2 = load_json(judge_dir / "task2_judge_metrics.json")
    task3 = load_json(judge_dir / "task3_judge_metrics.json")

    task1_overall = task1.get("task1_macro_accuracy_pct", 0.0)
    task2_overall = task2.get("mean_score_pct", 0.0)
    task3_overall = task3.get("mean_score_pct", 0.0)
    task2_by_style = task2.get("by_style", {})
    return {
        "model_alias": alias,
        "model": display_name,
        "task1_intent": task1.get("intent_exact_pct", 0.0),
        "task1_target": task1.get("target_exact_pct", 0.0),
        "task1_mechanism": task1.get("mechanism_exact_pct", 0.0),
        "task1_overall": task1_overall,
        "task2_teasing": style_score(task2_by_style, "teasing"),
        "task2_pushback": style_score(task2_by_style, "pushback"),
        "task2_neutral": style_score(task2_by_style, "neutral"),
        "task2_overall": task2_overall,
        "task2_by_style": task2_by_style,
        "task3_overall": task3_overall,
        "overall": (task1_overall + task2_overall + task3_overall) / 3.0,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required metric file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def style_score(by_style: dict[str, Any], style: str) -> float:
    payload = by_style.get(style) or {}
    return float(payload.get("mean_score_pct", 0.0) or 0.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_alias",
        "model",
        "task1_intent",
        "task1_target",
        "task1_mechanism",
        "task1_overall",
        "task2_teasing",
        "task2_pushback",
        "task2_neutral",
        "task2_overall",
        "task3_overall",
        "overall",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
