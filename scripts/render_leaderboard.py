#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    ("Rank", "rank"),
    ("Model", "model"),
    ("Task1 Intent", "task1_intent"),
    ("Task1 Target", "task1_target"),
    ("Task1 Mechanism", "task1_mechanism"),
    ("Task1 Overall", "task1_overall"),
    ("Task2 Overall", "task2_overall"),
    ("Task3 Overall", "task3_overall"),
    ("Overall", "overall"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a paper-style leaderboard from SarcBench aggregate results.")
    parser.add_argument("--input", type=Path, default=Path("outputs/paper_main/results/main_results.json"))
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = sorted(payload["models"], key=lambda row: row.get("overall", 0.0), reverse=True)
    ranked = []
    for index, row in enumerate(rows, start=1):
        ranked.append({"rank": index, **row})

    output_md = args.output_md or args.input.with_name("leaderboard.md")
    output_csv = args.output_csv or args.input.with_name("leaderboard.csv")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(output_md, ranked)
    write_csv(output_csv, ranked)
    print(json.dumps({"input": str(args.input), "output_md": str(output_md), "output_csv": str(output_csv), "rows": len(ranked)}, ensure_ascii=False, indent=2))
    return 0


def write_markdown(path: Path, rows: list[dict]) -> None:
    header = "| " + " | ".join(label for label, _ in FIELDS) + " |"
    divider = "| " + " | ".join("---" for _ in FIELDS) + " |"
    lines = [header, divider]
    for row in rows:
        values = []
        for _, key in FIELDS:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for _, key in FIELDS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for _, key in FIELDS})


if __name__ == "__main__":
    raise SystemExit(main())
