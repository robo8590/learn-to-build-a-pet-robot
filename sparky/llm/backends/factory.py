"""Build the single LLM provider from :class:`RuntimeConfig`."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from sparky.config.runtime import RuntimeConfig, resolve_runtime_config
from sparky.llm.backends.base import LLMProvider
from sparky.llm.backends.remote_http import RemoteHTTPProvider


def build_provider(config: RuntimeConfig) -> LLMProvider:
    return RemoteHTTPProvider(config)


def initialize_llm(args: Any = None) -> Tuple[LLMProvider, bool]:
    """Resolve runtime config and return ``(provider, vision_enabled)``.

    Kept under the old ``initialize_llm`` name so existing imports and test
    patches do not change; renamed in A4.5. Vision support is assumed for the
    happy path and confirmed by ``--check-vision``.
    """
    config = resolve_runtime_config(args)
    provider = build_provider(config)
    print(f"Using LLM server {config.url} (model: {config.model})")
    return provider, provider.capabilities.supports_vision
