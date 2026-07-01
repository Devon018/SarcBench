# Evaluation Protocol

This repository implements the main experiment described in the paper. It does
not implement downstream research-question analysis, plotting, or manuscript
generation.

## Candidate Models

The default roster in `configs/paper_models.json` contains the 12 paper models:

1. GPT-5.5
2. Gemini 3.1 Pro
3. Claude Opus 4.8
4. Qwen 3.7 Plus
5. DeepSeek V4
6. MiniMax 3
7. MiMo V2 Pro
8. Qwen 3.6 35B
9. Gemma 4 31B
10. Phi-4
11. Llama 4 Scout
12. Mistral 7B Instruct

Closed-source models use provider APIs. Open-source models use the
OpenAI-compatible chat endpoints configured in `configs/paper_models.json`.
Reviewers can override endpoint URLs through the corresponding `*_BASE_URL`
environment variables while preserving the paper model aliases.

## Tasks

Task 1, Intent Recognition, asks a model to recover:

- one or more intended-meaning options
- one target option
- up to three sarcasm mechanism labels

Task 1 is scored directly against gold labels. The main Task 1 score is the
macro average of intent exact match, target exact match, and mechanism exact
match.

Task 2, Sarcasm Response, asks a model to write a next-turn reply to the
sarcastic last comment under an explicit requested response style. The paper
experiment expands every released base sample into three style-conditioned
instances: `teasing`, `pushback`, and `neutral`.

Task 3, Sarcasm Generation, asks a model to write a sarcastic comment conditioned
on the preceding thread, intended meaning, and preferred mechanism labels.

## Judge Protocol

Task 2 and Task 3 are open-ended, so they use LLM-as-a-judge:

- judges: GPT-5.5, Gemini 3.1 Pro, Claude Opus 4.8
- repeats: 3 independent judge calls per judge family
- decoding: deterministic judge temperature `0.0`
- final sample score: mean of `overall_pass` across all judge/repeat judgments
- final task score: mean of sample scores

The judge prompt is designed as a cross-model evaluation protocol rather than a
single vague preference question. It instructs each judge to:

- ignore candidate model identity;
- avoid length and verbosity bias;
- treat the reference answer as a quality anchor, not an exact target string;
- decompose quality into independent yes/no criteria;
- reject fluent but off-topic, literal, wrong-style, or wrong-target answers;
- use few-shot calibration examples before judging the actual item.
- output one concise explanation for each criterion plus an overall explanation.

Task 2 criteria are conversational fit, addressing the sarcastic last comment,
avoiding literal-trap errors, requested-style match, social appropriateness,
natural style, and overall pass. The judge uses the `requested_style` on each
prediction row as the source of truth for style matching. The released
`reference_reply` is kept as a content anchor for the same sarcasm event, but it
does not override the requested style.

Task 3 criteria are meaning preservation, sarcastic tone, context fit, target
consistency, mechanism alignment, natural style, and overall pass.

The judge prompts are stored in:

- `evaluation/prompts/task2_judge.txt`
- `evaluation/prompts/task3_judge.txt`

Raw judge judgments are preserved as JSONL so reviewers can recompute metrics.
The per-sample score files also keep the judge explanations for every
judge/repeat judgment, making failures inspectable without rerunning the judges.

## Decoding

The default config uses:

- candidate temperature: `0.7`
- judge temperature: `0.0`
- one model query per Task 1 or Task 3 sample
- three Task 2 queries per base sample, one each for `teasing`, `pushback`, and
  `neutral`
- no chain-of-thought prompting
- no retrieval augmentation
- no tools

Network retries default to one attempt so failed API calls are visible in
`*_errors.jsonl` rather than silently hidden by retry loops.

## Commands

Candidate inference:

```bash
python3 scripts/run_candidates.py --config configs/paper_models.json --models all --reset
```

Judging:

```bash
python3 scripts/run_judges.py --config configs/paper_models.json --candidates all --reset
```

Aggregation:

```bash
python3 scripts/aggregate_results.py --config configs/paper_models.json --models all
```

Use `--dry-run` on candidate or judge commands to inspect the configured model
roster and paths without sending provider requests.
