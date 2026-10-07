# Configuration

The NeurIPS profile is `configs/paper_models.json`. Original and corrected judge
calibration prompts are separate profiles. `configs/latest_models.json` preserves
the older exploratory roster; do not relabel its results as NeurIPS results.

Models have distinct `alias`/`display_name`, `family`, and `model`/`request_model`
fields. Family determines self-judge exclusion; the provider controls transport.
The family must not be inferred from a shared gateway vendor. The request name
must resolve to the stated model at the configured endpoint. In particular,
`claude-opus-4.6` and `minimax-2.7` are compatible-profile aliases, not claims that
those are the native vendor IDs (`claude-opus-4-6` and `MiniMax-M2.7`). Record the
actual gateway and mappings with any run; do not assume an arbitrary endpoint
accepts the paper aliases. A `request_model` override provides an explicit mapping.

The paper profile intentionally requires compatible endpoints for API-hosted
models. Set `OPENAI_BASE_URL`, `CLAUDE_COMPATIBLE_BASE_URL`,
`GEMINI_COMPATIBLE_BASE_URL`, `DEEPSEEK_BASE_URL`, `MINIMAX_BASE_URL`, and
`MIMO_BASE_URL` to their `/v1`-style API bases and the corresponding key variables
in the JSON. The public DashScope Qwen compatible base is supplied by default.
No endpoint secrets belong in the repository. Locally served endpoints can be
overridden with each model's `*_BASE_URL`; model weights must use the recorded
revision SHA.

Per-model `temperature` overrides the candidate default. `max_tokens=null`
means omit the field, rather than send 512/256. Per-model `request_timeout`
overrides the protocol timeout. `exclude_self_family=true` applies at dispatch
and aggregation. The old native Anthropic transport requires its API's explicit
output limit; it is not used by the paper profile.

Qwen's current official model card recommends vLLM >=0.19.0:
<https://huggingface.co/Qwen/Qwen3.6-35B-A3B>. This is a supported setup
recommendation, not evidence of the historical experiment's serving version.
The original environment lockfile/backport commit is still needed to verify
that version. Capture the actual vLLM/Transformers/CUDA versions, weight
revisions, endpoint alias mappings, and resolved decoding settings with runs.
