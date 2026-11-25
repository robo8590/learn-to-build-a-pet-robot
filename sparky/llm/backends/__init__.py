"""LLM provider implementations.

The HTTP-only design keeps ``LLMProvider`` as the extension seam; today there
is a single ``RemoteHTTPProvider`` that talks to an OpenAI-compatible server.
"""

from sparky.llm.backends.base import Capabilities, HealthStatus, LLMProvider
from sparky.llm.backends.remote_http import ProviderError, RemoteHTTPProvider

__all__ = [
    "Capabilities",
    "HealthStatus",
    "LLMProvider",
    "ProviderError",
    "RemoteHTTPProvider",
]
