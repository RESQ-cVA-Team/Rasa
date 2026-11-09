# pyright: reportMissingTypeStubs=false, reportMissingModuleSource=false
"""Minimal OpenAI-chat-completions-compatible HTTP client.

Deliberately not a copy of Action's langchain-based provider abstraction --
Rasa and Action are separate deployables with separate requirements.txt, and
pulling in the full langchain dependency chain here just to make one
short-lived classification call per uncertain message isn't worth it.

Instead this covers the same ground with plain `requests`, since four of
Action's five providers (openai, openai-compatible, vllm, sglang) already
speak the identical OpenAI wire protocol, and Ollama exposes the same shape
at `{base_url}/v1`. Env var names intentionally match Action's
(LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_KWARGS_JSON) so the
two services stay operationally consistent without sharing code.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

import requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMConfigError(RuntimeError):
    pass


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_base_url(provider: str) -> str:
    configured = _env("LLM_BASE_URL")
    if provider == "openai":
        return configured or _DEFAULT_OPENAI_BASE_URL
    if provider == "ollama":
        base = configured or _DEFAULT_OLLAMA_BASE_URL
        return base.rstrip("/") + "/v1"
    # openai-compatible, vllm, sglang: no sane default, must be configured.
    if not configured:
        raise LLMConfigError(f"LLM_BASE_URL is required for provider '{provider}'")
    return configured.rstrip("/")


def _resolve_api_key(provider: str) -> Optional[str]:
    api_key = _env("LLM_API_KEY") or _env("OPENAI_API_KEY")
    if api_key:
        return api_key
    if provider == "ollama":
        return None
    raise LLMConfigError(f"LLM_API_KEY (or OPENAI_API_KEY) is required for provider '{provider}'")


def _load_extra_kwargs() -> Dict[str, Any]:
    raw = _env("LLM_KWARGS_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMConfigError("LLM_KWARGS_JSON must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMConfigError("LLM_KWARGS_JSON must decode to a JSON object")
    return parsed


class OpenAICompatibleLLMClient:
    """Calls one `/chat/completions` endpoint. Returns None on any failure --
    callers are expected to treat that as "no answer" and keep their own
    fallback behavior, never crash on a missing/misbehaving LLM."""

    def __init__(self, timeout_seconds: float = 8.0) -> None:
        self._timeout = timeout_seconds
        self._provider = (_env("LLM_PROVIDER") or "").lower()
        self._model = _env("LLM_MODEL")
        self._configured = False
        self._base_url = ""
        self._api_key: Optional[str] = None
        self._extra_kwargs: Dict[str, Any] = {}

        if not self._provider or not self._model:
            logger.info("LLM_PROVIDER/LLM_MODEL not set -- LLM client disabled")
            return

        try:
            self._base_url = _resolve_base_url(self._provider)
            self._api_key = _resolve_api_key(self._provider)
            self._extra_kwargs = _load_extra_kwargs()
            self._configured = True
        except LLMConfigError as exc:
            logger.warning(f"LLM client misconfigured, disabling: {exc}")

    @property
    def enabled(self) -> bool:
        return self._configured

    def complete(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 20,
        temperature: float = 0.0,
    ) -> Optional[str]:
        if not self._configured:
            return None

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            **self._extra_kwargs,
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content")
            return content.strip() if isinstance(content, str) else None
        except Exception as exc:  # noqa: BLE001 -- any failure here just means "no answer"
            logger.warning(f"LLM completion call failed: {exc}")
            return None


def build_default_client(timeout_seconds: float = 8.0) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(timeout_seconds=timeout_seconds)
