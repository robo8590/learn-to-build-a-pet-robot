"""OpenAI-compatible HTTP provider (happy path).

This is the first slice of the HTTP-only migration. It intentionally skips
retries, SSE streaming, and readiness polling: those are follow-up tasks in
``TASKS.md`` (A4/A5/A6). What it does provide is a working chat *and* vision
path against any server that implements ``POST /v1/chat/completions``.
"""

from __future__ import annotations

import json
import ssl
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sparky.config.runtime import RuntimeConfig, auth_headers
from sparky.llm.backends.base import Capabilities, HealthStatus, LLMProvider

_ERROR_BODY_CHARS = 200


class ProviderError(RuntimeError):
    """Raised when the LLM server cannot complete a request."""


class RemoteHTTPProvider(LLMProvider):
    name = "remote-http"
    capabilities = Capabilities(supports_vision=True, supports_streaming=False)

    def __init__(self, config: RuntimeConfig):
        self.config = config

    # -- transport ---------------------------------------------------------
    def _request(self, path: str, payload: Optional[dict] = None) -> dict:
        url = f"{self.config.url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        method = "GET" if payload is None else "POST"
        request = Request(url, data=data, headers=auth_headers(self.config), method=method)
        context = None if self.config.verify_tls else ssl._create_unverified_context()

        try:
            with urlopen(request, timeout=self.config.timeout, context=context) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:_ERROR_BODY_CHARS]
            raise ProviderError(
                f"LLM server returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise ProviderError(
                f"Could not reach the LLM server at {self.config.url}: {exc.reason}"
            ) from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError("LLM server returned a non-JSON response.") from exc

    def _complete(
        self,
        messages,
        *,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload.update({key: value for key, value in kwargs.items() if value is not None})

        result = self._request("/chat/completions", payload)
        try:
            content = result["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("LLM server response did not contain a message.") from exc
        return (content or "").strip()

    # -- LLMProvider interface --------------------------------------------
    def chat(self, messages, *, max_tokens: int, temperature: float = 0.7, **kwargs) -> str:
        return self._complete(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    def stream_chat(
        self, messages, *, max_tokens: int, temperature: float = 0.7, **kwargs
    ) -> Iterator[str]:
        # SSE is a follow-up task; yield the full reply as one chunk for now.
        yield self.chat(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        )

    def vision_chat(
        self, messages, *, max_tokens: int, temperature: float = 0.25, **kwargs
    ) -> str:
        return self._complete(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    def health(self) -> HealthStatus:
        try:
            self._request("/models")
        except ProviderError as exc:
            return HealthStatus(False, str(exc))
        return HealthStatus(True, "ok")

    # -- legacy call shape -------------------------------------------------
    def create_chat_completion(
        self,
        messages,
        *,
        temperature: float = 0.7,
        max_tokens: int = 96,
        **kwargs,
    ) -> dict:
        """llama.cpp-compatible response shape used by the current callers.

        Removed in A4.5 once ``text_llm`` / ``vision_llm`` call the provider
        interface directly.
        """
        content = self._complete(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}
