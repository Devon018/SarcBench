from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    display_name: str
    provider: str
    deployment: str
    model: str
    request_model: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    max_tokens: int | None = None
    family: str | None = None
    temperature: float | None = None
    request_timeout: float | None = None
    revision: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelSpec":
        return cls(
            alias=str(payload["alias"]),
            display_name=str(payload.get("display_name") or payload["alias"]),
            provider=str(payload["provider"]),
            deployment=str(payload.get("deployment") or "api"),
            model=str(payload["model"]),
            request_model=payload.get("request_model"),
            base_url=payload.get("base_url"),
            base_url_env=payload.get("base_url_env"),
            api_key=payload.get("api_key"),
            api_key_env=payload.get("api_key_env"),
            max_tokens=None if payload.get("max_tokens") is None else int(payload["max_tokens"]),
            family=payload.get("family"),
            temperature=None if payload.get("temperature") is None else float(payload["temperature"]),
            request_timeout=None if payload.get("request_timeout") is None else float(payload["request_timeout"]),
            revision=payload.get("revision"),
            extra_headers=dict(payload.get("extra_headers") or {}),
            extra_body=dict(payload.get("extra_body") or {}),
            notes=payload.get("notes"),
        )

    @property
    def request_name(self) -> str:
        return self.request_model or self.model

    def resolved_base_url(self) -> str | None:
        if self.base_url_env and os.environ.get(self.base_url_env):
            return os.environ[self.base_url_env]
        return self.base_url

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env and os.environ.get(self.api_key_env):
            return os.environ[self.api_key_env]
        return None

    def public_dict(self) -> dict[str, Any]:
        payload = {
            "alias": self.alias,
            "display_name": self.display_name,
            "provider": self.provider,
            "deployment": self.deployment,
            "model": self.model,
            "request_model": self.request_model,
            "base_url": self.base_url,
            "base_url_env": self.base_url_env,
            "api_key_env": self.api_key_env,
            "max_tokens": self.max_tokens,
            "family": self.family,
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            "revision": self.revision,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
            "notes": self.notes,
        }
        if self.api_key:
            payload["api_key"] = "***"
        return payload


@dataclass(frozen=True)
class ProtocolConfig:
    candidate_temperature: float
    judge_temperature: float
    judge_repeats: int
    request_timeout: float
    max_attempts: int
    max_workers: int
    judge_workers: int
    exclude_self_family: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProtocolConfig":
        return cls(
            candidate_temperature=float(payload.get("candidate_temperature", 0.7)),
            judge_temperature=float(payload.get("judge_temperature", 0.0)),
            judge_repeats=int(payload.get("judge_repeats", 3)),
            request_timeout=float(payload.get("request_timeout", 120.0)),
            max_attempts=int(payload.get("max_attempts", 1)),
            max_workers=int(payload.get("max_workers", 8)),
            judge_workers=int(payload.get("judge_workers", 8)),
            exclude_self_family=bool(payload.get("exclude_self_family", True)),
        )


@dataclass(frozen=True)
class RunConfig:
    config_path: Path
    data_dir: Path
    output_dir: Path
    prompts_dir: Path
    protocol: ProtocolConfig
    candidate_models: list[ModelSpec]
    judge_models: list[ModelSpec]

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        paths = payload.get("paths") or {}
        return cls(
            config_path=path,
            data_dir=Path(paths.get("data_dir", "data")),
            output_dir=Path(paths.get("output_dir", "outputs/paper_main")),
            prompts_dir=Path(paths.get("prompts_dir", "evaluation/prompts")),
            protocol=ProtocolConfig.from_dict(payload.get("protocol") or {}),
            candidate_models=[ModelSpec.from_dict(item) for item in payload["candidate_models"]],
            judge_models=[ModelSpec.from_dict(item) for item in payload["judge_models"]],
        )

    def select_candidates(self, aliases: str | None) -> list[ModelSpec]:
        return select_models(self.candidate_models, aliases)

    def select_judges(self, aliases: str | None) -> list[ModelSpec]:
        return select_models(self.judge_models, aliases)


def select_models(models: list[ModelSpec], aliases: str | None) -> list[ModelSpec]:
    if aliases is None or aliases == "all":
        return models
    wanted = [item.strip() for item in aliases.split(",") if item.strip()]
    by_alias = {model.alias: model for model in models}
    missing = [alias for alias in wanted if alias not in by_alias]
    if missing:
        available = ", ".join(sorted(by_alias))
        raise ValueError(f"Unknown model aliases {missing}; available aliases: {available}")
    return [by_alias[alias] for alias in wanted]
