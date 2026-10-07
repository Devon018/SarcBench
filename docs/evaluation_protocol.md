# NeurIPS 2026 evaluation protocol

`configs/paper_models.json` implements the protocol described by the NeurIPS
manuscript: GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6, Qwen 3.6 Plus, DeepSeek V3.2,
MiniMax 2.7, MiMo V2 Pro, Qwen 3.6 35B, Gemma 4 31B, Phi-4, Llama 4 Scout, and
Mistral 7B Instruct. These groups distinguish API-hosted from locally served
models, not weight availability. DeepSeek V3.2 has public weights.

The previous default roster remains in `configs/latest_models.json` as a separate
exploratory configuration. It is not the NeurIPS roster.

## Scoring

Understanding uses intent/target/mechanism exact match, their macro average,
and strict joint exact match. Mechanisms are unordered sets. The six canonical
mechanisms and the `other` fallback are all retained for exact-match scoring;
canonical distribution statistics explicitly exclude `other`.

For Response and Generation, each valid non-self judgment receives one point
for each of the six task-specific Boolean criteria. The normalized judgment
score is their sum divided by six. `overall_pass` is a separate diagnostic and
never contributes to this score. An instance's score is the mean of its valid
non-self calls. A task's score is the unweighted mean of valid instance scores.
Criterion summaries use the same equal-instance weighting. The scorer records
`score_version=six-criteria-nonself-v1`; aggregation rejects legacy overall-pass
metric files instead of silently combining incompatible metrics.

Family identity is independent of transport/provider: GPT judges cannot judge
GPT candidates, including a different GPT version; Claude excludes Claude and
Gemini excludes Gemini. Gemma is a different family from Gemini. Same-family
calls are not dispatched, and saved legacy self-judgments are filtered again
when reaggregated. Three judge families and three repeats imply nine nominal
raw slots, or six eligible non-self slots for candidates in one judge family.
Other candidates have nine eligible slots, before failed calls.

The 30,083 events project into 90,249 base task records. Expanding each Response
record into teasing, pushback, and neutral yields 150,415 full-context queries
per model. Response rows are grouped by `(sample_id, requested_style)`, not ID
alone. The pinned hosted release provides three style-specific references;
the judge uses the reference matching the requested style. Legacy single-reference
files remain readable as cross-style anchors, with their source style explicit,
but do not establish same-style human ceilings.

## Failure accounting

Candidate and judge errors are saved separately. Failed candidates are not
fabricated as empty successful predictions. Judgments with invalid Boolean
criteria/explanations are rejected. A prediction with zero valid non-self
judgments has no task score; an entirely missing task has `mean_score=null`,
not an inferred zero. The runner reports attempted/valid/error counts, expected
non-self calls, missing calls, and predictions without valid judgments.
Leaderboard aggregation rejects a missing task. Available-case scoring must
be interpreted together with these effective denominators.

This is the released runner's policy. Without the original per-sample logs,
it does not prove which implementation or failure denominators produced the
manuscript's author-reported tables.

## Request settings

All NeurIPS-profile calls use OpenAI-compatible `/v1/chat/completions`. Candidate
temperature is 0.0 except for both Qwen models (0.7); judges use 0.0. No output
limit is sent when `max_tokens=null`. Qwen thinking is disabled through the
endpoint-specific option. Exact compatible endpoint URLs and alias mappings
must be supplied by the operator; unsupported aliases are never silently
substituted with newer models. Native Anthropic/Gemini adapters remain available
for other configurations and are not selected by this profile.

Candidate prompts have no demonstrations. The original judge prompts have
three fixed calibration examples. They remain unchanged under
`evaluation/prompts/paper_2026_original`. The corrected calibration templates
are in `evaluation/prompts`, selected by
`configs/paper_models_corrected_prompts.json`. The corrected Response example
identifies its teasing reference and does not equate neutral advice with a
style failure. The corrected Generation example requests exaggeration rather
than understatement. Corrected prompts also address the residual `other` label.
These changes require new judgments and cannot retroactively fix old scores.

## Offline verification

```sh
python3 -m unittest discover -s tests -v
python3 scripts/audit_dataset.py
python3 scripts/reaggregate_judgments.py --config configs/paper_models.json
```

Reaggregation uses saved records only and makes no model calls. It raises an
error if original judgment files are missing. It cannot reconstruct individual
judgments or audit denominators from published means.

`data/release_manifest.json` pins the hosted dataset revision and hashes the
released task files and public master. `docs/dataset_statistics.json` records
exact descriptive counts and their denominators. These checks verify the
released artifact and scoring behavior, not independent replication of the
reported experiments.
