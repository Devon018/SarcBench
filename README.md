# SarcBench

SarcBench is a bilingual benchmark for contextual sarcasm understanding,
response, and controlled sarcasm generation. This repository is the clean
reviewer-facing release: it contains the released data, the main-experiment
evaluation code, model/judge configuration, prompts, and documentation. It does
not contain the paper source or generated result files.

## Repository Layout

```text
configs/paper_models.json        Paper model roster and protocol settings
data/                            Released benchmark JSONL files
docs/                            Protocol and schema documentation
evaluation/prompts/              Candidate and judge prompts
sarcbench/                       Evaluation package
scripts/run_candidates.py        Candidate inference entrypoint
scripts/run_judges.py            LLM-as-a-judge entrypoint
scripts/aggregate_results.py     Main-result aggregation entrypoint
scripts/render_leaderboard.py    Ranked-table export entrypoint
```

## Paper-Equivalent Main Experiment

The default config follows the paper protocol:

- 12 candidate models: GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6, Qwen 3.6 Plus,
  DeepSeek V3.2, MiniMax 2.7, MiMo V2 Pro, Qwen 3.6 35B, Gemma 4 31B, Phi-4,
  Llama 4 Scout, and Mistral 7B Instruct v0.3.
- Task 1 is scored directly against gold intent, target, and mechanism labels.
- Task 2 and Task 3 are scored by GPT-5.4, Gemini 3.1 Pro, and Claude Opus 4.6.
- Each judge scores each response three times; final open-ended scores are
  averaged across judge families and repeats.
- Candidate generation uses temperature `0.7`; judge calls use temperature `0.0`.

## Run

Set the API keys and endpoint base URLs referenced in
`configs/paper_models.json`, then run the three evaluation stages:

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

Outputs are written under `outputs/paper_main/`:

- `candidates/<model>/task*_predictions.jsonl`
- `candidates/<model>/task1_metrics.json`
- `judgments/<model>/task2_judgments.jsonl`
- `judgments/<model>/task3_judgments.jsonl`
- `judgments/<model>/task*_scores_by_sample.jsonl`
- `judgments/<model>/task*_judge_metrics.json`
- `results/main_results.json`
- `results/main_results.csv`

See [docs/evaluation_protocol.md](docs/evaluation_protocol.md) for the complete
protocol and [docs/data_schema.md](docs/data_schema.md) for file schemas.

Judge outputs include both boolean rubric decisions and concise explanations.
The aggregate metrics use only the boolean fields, while raw judgments and
per-sample score files preserve the explanations for inspection.
