"""Provider interface shared by every LLM backend.

See ``ARCHITECTURE.md`` section 6. Callers are moving to ``chat`` /
``stream_chat`` / ``vision_chat``; until then ``RemoteHTTPProvider`` also keeps
the llama.cpp ``create_chat_completion`` shape so the existing call sites and
tests keep working.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Capabilities:
    supports_vision: bool = True
    supports_streaming: bool = False


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    detail: str = ""


class LLMProvider(ABC):
    name: str
    capabilities: Capabilities

    @abstractmethod
    def chat(self, messages, *, max_tokens: int, temperature: float = 0.7, **kwargs) -> str:
        """Return a complete text reply."""

    @abstractmethod
    def stream_chat(
        self, messages, *, max_tokens: int, temperature: float = 0.7, **kwargs
    ) -> Iterator[str]:
        """Yield reply chunks."""

    @abstractmethod
    def vision_chat(
        self, messages, *, max_tokens: int, temperature: float = 0.25, **kwargs
    ) -> str:
        """Return a complete reply for a message containing an image."""

    @abstractmethod
    def health(self) -> HealthStatus:
        """Report whether the backend is reachable."""

    def close(self) -> None:
        return None
