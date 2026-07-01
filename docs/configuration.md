# Configuration

The paper protocol is encoded in `configs/paper_models.json`.

## Paths

- `paths.data_dir`: benchmark JSONL directory
- `paths.output_dir`: output root
- `paths.prompts_dir`: prompt templates

## Protocol

- `candidate_temperature`: candidate decoding temperature, default `0.7`
- `judge_temperature`: judge decoding temperature, default `0.0`
- `judge_repeats`: number of repeated calls per judge family, default `3`
- `request_timeout`: per-request timeout in seconds
- `max_attempts`: API attempts per sample; default `1` to expose failures
- `max_workers`: candidate inference concurrency
- `judge_workers`: judge concurrency

## Provider Models

Provider-backed models use one of these providers:

- `openai`
- `anthropic`
- `gemini`
- `openai_compatible`

API keys are read from the `api_key_env` field. Base URLs can be fixed with
`base_url` or overridden by `base_url_env`.

## OpenAI-Compatible Endpoint Models

Open-source paper models are evaluated through OpenAI-compatible chat endpoints.
A typical endpoint configuration is:

```json
{
  "provider": "openai_compatible",
  "deployment": "openai_compatible_endpoint",
  "model": "Qwen/Qwen3.6-35B-A3B",
  "base_url": "http://127.0.0.1:8010/v1",
  "base_url_env": "QWEN_35B_BASE_URL",
  "api_key": "EMPTY"
}
```

`model` is the model name sent to the chat endpoint. `base_url_env` lets a
reviewer override the default endpoint without editing the config. Keep the
alias and display name unchanged so output files remain aligned with the paper
roster.
