# SarcBench

<h3 align="center">A Bilingual Benchmark for Contextual Sarcasm Understanding, Response, and Generation</h3>

<p align="center">
  <a href="docs/evaluation_protocol.md">Evaluation Protocol</a> |
  <a href="docs/data_schema.md">Data Schema</a> |
  <a href="docs/configuration.md">Configuration</a> |
  <a href="evaluation/prompts">Prompts</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E=3.10-blue" alt="Python >=3.10">
  <img src="https://img.shields.io/badge/dependencies-standard%20library-green" alt="Standard-library runner">
  <img src="https://img.shields.io/badge/data-JSONL%20%2B%20Git%20LFS-orange" alt="JSONL data with Git LFS">
  <img src="https://img.shields.io/badge/tasks-understanding%20%7C%20response%20%7C%20generation-purple" alt="Three sarcasm tasks">
</p>

SarcBench is a bilingual benchmark for evaluating whether large language models
can interpret sarcasm in context, respond to sarcastic comments with controlled
social stance, and generate sarcastic comments under explicit pragmatic
constraints. This repository contains the NeurIPS 2026 protocol alignment: it contains
the released data, prompts, main-experiment evaluation code, model/judge
configuration, and protocol documentation. It does not include the paper source
or generated result files.

## Contents

- [Overview](#overview)
- [Access SarcBench](#access-sarcbench)
- [Data Format](#data-format)
- [Experiment and Evaluation](#experiment-and-evaluation)
- [Model Roster](#model-roster)
- [Outputs](#outputs)
- [Repository Layout](#repository-layout)
- [Notes](#notes)

## Overview

SarcBench focuses on naturally occurring video-comment threads from English and
Chinese social media. Each benchmark instance keeps the conversational context
needed to reason about sarcasm: video metadata, the parent comment, preceding
replies, and the target sarcastic reply or controlled generation condition.

| Item | Description |
| --- | --- |
| Languages | English and Chinese |
| Benchmark scale | 30.1K sarcasm-focused instances in the paper benchmark |
| Task 1 | Sarcasm understanding: recover intent, target, and mechanism labels |
| Task 2 | Sarcasm response: write a next-turn reply in a requested style |
| Task 3 | Sarcasm generation: write a sarcastic comment from context and constraints |
| Candidate models | 12 paper models across provider APIs and local open-source checkpoints |
| Open-ended scoring | LLM-as-a-judge for Task 2 and Task 3 with three judge families |

The release contains four JSONL files:

- `data/benchmark_master.jsonl`: full aligned sarcasm events.
- `data/task1_benchmark.jsonl`: direct-label understanding task.
- `data/task2_benchmark.jsonl`: response-generation base samples.
- `data/task3_benchmark.jsonl`: controlled sarcasm-generation samples.

## Access SarcBench

The released JSONL files are tracked with Git LFS. Install and pull LFS objects
before running evaluation; otherwise the files under `data/` will be pointer
files rather than benchmark records.

```bash
git clone git@github.com:Devon018/SarcBench.git
cd SarcBench

git lfs install
git lfs pull
```

Create an environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The evaluation runner itself uses only the Python standard library. Provider
SDKs are not required because API calls are made through HTTP endpoints.

## Data Format

Each file is newline-delimited JSON. The task projections expose only the fields
needed by the corresponding evaluation stage.

### Task 1: Understanding

Each row contains a comment thread, answer options, and gold labels. A model
must return JSON with:

```json
{
  "Intent": "A",
  "Target": "B",
  "Mechanism": ["polarity_reversal", "exaggeration"]
}
```

Task 1 is scored directly against gold intent, target, and mechanism labels.

### Task 2: Response

Each base row contains the context and a reference reply. During paper
evaluation, every base sample is expanded into three requested styles:

```text
teasing, pushback, neutral
```

The judge sees the full thread, requested style, reference reply, and candidate
reply. Style matching is controlled by the requested style, not by the reference
reply alone.

### Task 3: Generation

Each row contains the context, intended meaning, target, preferred mechanisms,
and a reference sarcastic comment. The model writes a new sarcastic comment that
preserves the intended meaning and follows the requested mechanisms.

See [docs/data_schema.md](docs/data_schema.md) for the complete field-level
schema.

## Experiment and Evaluation

The default config in `configs/paper_models.json` implements the NeurIPS manuscript protocol. Original per-sample logs are needed to verify the author-reported table values; protocol alignment alone is not a reproduction claim.

Set the API keys and endpoint base URLs referenced in
[docs/configuration.md](docs/configuration.md), then run:

```bash
python3 scripts/run_candidates.py --config configs/paper_models.json --models all --reset
python3 scripts/run_judges.py --config configs/paper_models.json --candidates all --reset
python3 scripts/aggregate_results.py --config configs/paper_models.json --models all
python3 scripts/render_leaderboard.py --input outputs/paper_main/results/main_results.json
```

For a single candidate:

```bash
python3 scripts/run_candidates.py --models gpt-5.4 --reset
python3 scripts/run_judges.py --candidates gpt-5.4 --reset
python3 scripts/aggregate_results.py --models gpt-5.4
python3 scripts/render_leaderboard.py --input outputs/paper_main/results/main_results.json
```

Use `--dry-run` to inspect configured models, paths, and tasks without sending
requests:

```bash
python3 scripts/run_candidates.py --models all --tasks task1 --limit 1 --dry-run
python3 scripts/run_judges.py --candidates gpt-5.4 --judges all --tasks task2 --limit 1 --dry-run
```

## Model Roster

| Role | Models |
| --- | --- |
| Closed-source candidates | GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6, Qwen 3.6 Plus, DeepSeek V3.2, MiniMax 2.7, MiMo V2 Pro |
| Open-source candidates | Qwen 3.6 35B, Gemma 4 31B, Phi-4, Llama 4 Scout, Mistral 7B Instruct |
| Judge families | GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6 |

Open-source models are served through OpenAI-compatible local endpoints. The
paper config records the served model identifiers, default ports, and endpoint
override environment variables. See
[configs/paper_models.json](configs/paper_models.json) for the exact roster.

## Outputs

All generated files are written under `outputs/paper_main/`.

| Path | Description |
| --- | --- |
| `candidates/<model>/task*_predictions.jsonl` | Raw candidate predictions |
| `candidates/<model>/task1_metrics.json` | Direct Task 1 metrics |
| `judgments/<model>/task2_judgments.jsonl` | Raw Task 2 judge decisions |
| `judgments/<model>/task3_judgments.jsonl` | Raw Task 3 judge decisions |
| `judgments/<model>/task*_scores_by_sample.jsonl` | Per-sample open-ended scores |
| `judgments/<model>/task*_judge_metrics.json` | Aggregated judge metrics |
| `results/main_results.json` | Main leaderboard payload |
| `results/main_results.csv` | Main leaderboard table |

Judge outputs include boolean rubric decisions and concise explanations. The
aggregate metrics use the boolean fields, while raw judgments and per-sample
score files preserve explanations for inspection.

## Repository Layout

```text
configs/paper_models.json        Paper model roster and protocol settings
data/                            Released benchmark JSONL files
docs/                            Protocol, schema, and configuration docs
evaluation/prompts/              Candidate and judge prompts
sarcbench/                       Evaluation package
scripts/run_candidates.py        Candidate inference entrypoint
scripts/run_judges.py            LLM-as-a-judge entrypoint
scripts/aggregate_results.py     Main-result aggregation entrypoint
scripts/render_leaderboard.py    Ranked-table export entrypoint
```

## Notes

- Candidate prompts are single-turn prompts with no retrieval, tools,
  self-consistency sampling, or chain-of-thought request.
- Task 2 and Task 3 use deterministic judge decoding at temperature `0.0`.
- Failed API calls are written to `*_errors.jsonl` instead of being silently
  hidden by long retry loops.
- The paper source and generated result artifacts are intentionally excluded
  from this release repository.

## NeurIPS release verification

The hosted task files are pinned at dataset revision
`5b1539d8ba1d62d2281796be44f8fa14b8da3962`. Response has three style-specific
references and all task views use opaque numeric sample IDs. The corrected
scorer averages the six rubric criteria, excludes same-family judgments, and
reports missing/failed denominators. It rejects old overall-pass metric files.

```sh
python3 -m unittest discover -s tests -v
python3 scripts/audit_dataset.py
python3 scripts/reaggregate_judgments.py --config configs/paper_models.json
```

Original calibration examples are preserved; corrected examples use
`configs/paper_models_corrected_prompts.json` and require new judgments.
The previous exploratory roster is in `configs/latest_models.json`.
See [the protocol documentation](docs/evaluation_protocol.md) for exact weighting,
failure treatment, provenance, and remaining historical-verification requirements.
