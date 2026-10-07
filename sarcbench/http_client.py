from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from sarcbench.config import ModelSpec


class ModelRequestError(RuntimeError):
    pass


def call_text_model(
    spec: ModelSpec,
    prompt: str,
    *,
    temperature: float,
    timeout: float,
    max_attempts: int,
) -> tuple[str, dict[str, int], dict[str, Any]]:
    if spec.provider in {"openai", "openai_compatible"}:
        return call_openai_chat(spec, prompt, temperature=temperature, timeout=timeout, max_attempts=max_attempts)
    if spec.provider == "anthropic":
        return call_anthropic_messages(spec, prompt, temperature=temperature, timeout=timeout, max_attempts=max_attempts)
    if spec.provider == "gemini":
        return call_gemini_generate(spec, prompt, temperature=temperature, timeout=timeout, max_attempts=max_attempts)
    raise ValueError(f"Unsupported provider {spec.provider!r} for model {spec.alias}")


def call_openai_chat(
    spec: ModelSpec,
    prompt: str,
    *,
    temperature: float,
    timeout: float,
    max_attempts: int,
) -> tuple[str, dict[str, int], dict[str, Any]]:
    base_url = spec.resolved_base_url()
    if not base_url:
        if spec.provider == "openai":
            base_url = "https://api.openai.com/v1"
        else:
            raise ModelRequestError(f"Missing compatible endpoint for {spec.alias}; set {spec.base_url_env}")
    api_key = spec.resolved_api_key()
    if not api_key and spec.deployment != "openai_compatible_endpoint":
        raise ModelRequestError(f"Missing API key for {spec.alias}; set {spec.api_key_env}")
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": spec.request_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }
    if spec.max_tokens is not None:
        payload["max_tokens"] = spec.max_tokens
    payload.update(spec.extra_body)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or 'EMPTY'}",
    }
    headers.update(spec.extra_headers)
    data = request_json(url, payload, headers, timeout=timeout, max_attempts=max_attempts)
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    return text, normalize_openai_usage(data.get("usage") or {}), data


def call_anthropic_messages(
    spec: ModelSpec,
    prompt: str,
    *,
    temperature: float,
    timeout: float,
    max_attempts: int,
) -> tuple[str, dict[str, int], dict[str, Any]]:
    api_key = spec.resolved_api_key()
    if not api_key:
        raise ModelRequestError(f"Missing API key for {spec.alias}; set {spec.api_key_env}")
    base_url = spec.resolved_base_url() or "https://api.anthropic.com"
    url = base_url.rstrip("/") + "/v1/messages"
    if spec.max_tokens is None:
        raise ModelRequestError("Native Anthropic messages require an explicit max_tokens; use the compatible paper endpoint for the paper protocol.")
    payload: dict[str, Any] = {
        "model": spec.request_name,
        "max_tokens": spec.max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(spec.extra_body)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": spec.extra_headers.get("anthropic-version", "2023-06-01"),
    }
    headers.update({key: value for key, value in spec.extra_headers.items() if key != "anthropic-version"})
    data = request_json(url, payload, headers, timeout=timeout, max_attempts=max_attempts)
    text_parts = []
    for item in data.get("content") or []:
        if item.get("type") == "text":
            text_parts.append(item.get("text") or "")
    usage = data.get("usage") or {}
    return "".join(text_parts), {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cached_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
    }, data


def call_gemini_generate(
    spec: ModelSpec,
    prompt: str,
    *,
    temperature: float,
    timeout: float,
    max_attempts: int,
) -> tuple[str, dict[str, int], dict[str, Any]]:
    api_key = spec.resolved_api_key()
    if not api_key:
        raise ModelRequestError(f"Missing API key for {spec.alias}; set {spec.api_key_env}")
    base_url = spec.resolved_base_url() or "https://generativelanguage.googleapis.com/v1beta"
    model_name = urllib.parse.quote(spec.request_name, safe="")
    url = f"{base_url.rstrip('/')}/models/{model_name}:generateContent?key={urllib.parse.quote(api_key)}"
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
        },
    }
    if spec.max_tokens is not None:
        payload["generationConfig"]["maxOutputTokens"] = spec.max_tokens
    payload.update(spec.extra_body)
    data = request_json(
        url,
        payload,
        {"Content-Type": "application/json", **spec.extra_headers},
        timeout=timeout,
        max_attempts=max_attempts,
    )
    text_parts = []
    candidates = data.get("candidates") or []
    if candidates:
        content = candidates[0].get("content") or {}
        for part in content.get("parts") or []:
            text_parts.append(part.get("text") or "")
    usage = data.get("usageMetadata") or {}
    return "".join(text_parts), {
        "input_tokens": int(usage.get("promptTokenCount", 0) or 0),
        "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
        "cached_tokens": int(usage.get("cachedContentTokenCount", 0) or 0),
    }, data


def request_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    max_attempts: int,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    attempts = max(max_attempts, 1)
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")
            if attempt >= attempts or not is_retryable_status(exc.code):
                raise ModelRequestError(f"HTTP {exc.code} from {url}: {response_text}") from exc
        except urllib.error.URLError as exc:
            if attempt >= attempts:
                raise ModelRequestError(f"Request failed for {url}: {exc}") from exc
        time.sleep(min(2 ** (attempt - 1), 8))
    raise ModelRequestError(f"Request failed for {url}")


def is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def normalize_openai_usage(usage: dict[str, Any]) -> dict[str, int]:
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens", 0) or 0),
    }
