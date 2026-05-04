# Data Schema

The release contains four JSONL files under `data/`.

## `benchmark_master.jsonl`

The master file stores full aligned sarcasm events. It includes video metadata,
parent comment, original and reconstructed replies, target sarcastic reply,
core annotations, and the three task projections.

Important fields include:

- `sample_id`
- `platform`
- `prompt_language`
- `video_title`
- `video_description`
- `parent_comment`
- `reconstructed_replies`
- `target_sarcastic_reply`
- `core_annotation`
- `task1`
- `task2`
- `task3`

## `task1_benchmark.jsonl`

Task 1 is the direct-label task. Each row contains:

- `sample_id`
- `platform`
- `comments`
- `IntentOptions`
- `TargetOptions`
- `gold.Intent`
- `gold.Target`
- `gold.Mechanism`

The model must return JSON with `Intent`, `Target`, and `Mechanism`.

## `task2_benchmark.jsonl`

Task 2 is response generation. Each row contains:

- `sample_id`
- `platform`
- `comments`
- `gold.reference_reply`
- `gold.reply_style_label`

The model writes a natural reply to the sarcastic last comment. The judge sees
the full thread, response-style label, reference reply, and candidate reply.

## `task3_benchmark.jsonl`

Task 3 is controlled sarcasm generation. Each row contains:

- `sample_id`
- `platform`
- `comments`
- `intended_meaning`
- `gold.reference_sarcastic_comment`
- `gold.target`
- `gold.preferred_mechanisms`

The model writes a sarcastic comment conditioned on the thread, intended meaning,
and preferred mechanisms. The judge also receives the target and reference
sarcastic comment.
