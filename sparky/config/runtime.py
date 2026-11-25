"""Resolve Sparky's runtime configuration for the HTTP-only LLM path.

Precedence is **CLI flag → environment variable → built-in default**. There are
no device profiles and no backend switch: ``ROBOT_REMOTE_*`` means "over HTTP",
including the Jetson's loopback server.

Secrets are never included in ``repr``, error messages, or the redacted view
used by diagnostics.
"""

from __future__ import annotations

import ipaddress
import json
import os
import ssl
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

DEFAULT_REMOTE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_VERIFY_TLS = True
DEFAULT_MAX_TOKENS = 96
DEFAULT_VISION_MAX_TOKENS = 256
DEFAULT_MEMORY_MAX_TURNS = 8
DEFAULT_IMAGE_MAX_DIM = 768

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_MODEL_FETCHER = Callable[["RuntimeConfig"], list[str]]


class ConfigError(ValueError):
    """Raised when runtime settings are missing, invalid, or unsafe."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Fully resolved runtime settings. Construct via ``resolve_runtime_config``."""

    url: str
    model: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    verify_tls: bool = DEFAULT_VERIFY_TLS
    max_tokens: int = DEFAULT_MAX_TOKENS
    vision_max_tokens: int = DEFAULT_VISION_MAX_TOKENS
    memory_max_turns: int = DEFAULT_MEMORY_MAX_TURNS
    image_max_dim: int = DEFAULT_IMAGE_MAX_DIM

    def __post_init__(self):
        object.__setattr__(self, "url", _normalize_url(self.url))
        if self.model is not None:
            object.__setattr__(self, "model", str(self.model).strip() or None)
        if self.api_key is not None:
            object.__setattr__(self, "api_key", str(self.api_key).strip() or None)

        if self.timeout <= 0:
            raise ConfigError("ROBOT_REMOTE_TIMEOUT must be greater than zero.")
        if self.max_retries < 0:
            raise ConfigError("ROBOT_REMOTE_MAX_RETRIES cannot be negative.")
        if self.max_tokens <= 0:
            raise ConfigError("ROBOT_LLM_MAX_TOKENS must be greater than zero.")
        if self.vision_max_tokens <= 0:
            raise ConfigError("ROBOT_VISION_MAX_TOKENS must be greater than zero.")
        if self.memory_max_turns < 1:
            raise ConfigError("ROBOT_MEMORY_MAX_TURNS must be at least 1.")
        if self.image_max_dim <= 0:
            raise ConfigError("ROBOT_IMAGE_MAX_DIM must be greater than zero.")

        if not self.is_loopback:
            if _url_scheme(self.url) != "https":
                raise ConfigError(
                    "ROBOT_REMOTE_URL must use HTTPS for a non-loopback server. "
                    "Use a reverse proxy or a trusted certificate; see README."
                )
            if not self.api_key:
                raise ConfigError(
                    "ROBOT_REMOTE_API_KEY is required for a non-loopback server."
                )

    @property
    def is_loopback(self) -> bool:
        """True when the LLM server is on this machine (HTTP is then fine)."""
        return _is_loopback_host(_url_host(self.url))

    def redacted(self) -> dict[str, Any]:
        """Return a safe mapping for diagnostics; the API key is masked."""
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "api_key"
        }
        values["api_key"] = "***" if self.api_key else None
        return values


def _url_parts(url: str, name: str = "ROBOT_REMOTE_URL"):
    if not isinstance(url, str) or not url.strip():
        raise ConfigError(f"{name} must be a non-empty URL.")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise ConfigError(f"{name} must start with http:// or https://.")
    if not parts.hostname:
        raise ConfigError(f"{name} must include a host.")
    if parts.username or parts.password:
        raise ConfigError(f"{name} must not embed credentials; use {name.rsplit('_', 1)[0]}_API_KEY.")
    if parts.query or parts.fragment:
        raise ConfigError(f"{name} must not contain a query string or fragment.")
    return parts


def _url_scheme(url: str) -> str:
    return urlsplit(url).scheme


def _url_host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _normalize_url(url: str) -> str:
    parts = _url_parts(url)
    path = parts.path.rstrip("/") or "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _parse_text(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ConfigError(f"{name} must not be empty.")
    return text


def _parse_int(value: Any, name: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer.") from exc


def _parse_float(value: Any, name: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number.") from exc


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be true/false (or 1/0, yes/no, on/off).")


def _cli_value(cli: Any, attribute: str) -> Any:
    value = getattr(cli, attribute, None) if cli is not None else None
    return None if value is None or str(value).strip() == "" else value


def auth_headers(config: RuntimeConfig) -> dict[str, str]:
    """Headers for HTTP requests to the LLM server (never logged)."""
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def fetch_model_ids(config: RuntimeConfig) -> list[str]:
    """Read the model ids advertised at ``GET {url}/models`` (OpenAI shape)."""
    request = Request(f"{config.url}/models", headers=auth_headers(config))
    context = None if config.verify_tls else ssl._create_unverified_context()
    with urlopen(request, timeout=config.timeout, context=context) as response:
        payload = json.load(response)
    ids = []
    for item in payload.get("data", []):
        model_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id.strip())
    return ids


def resolve_runtime_config(
    cli: Any = None,
    env: Optional[Mapping[str, str]] = None,
    *,
    fetch_model: bool = True,
    model_fetcher: Optional[_MODEL_FETCHER] = None,
) -> RuntimeConfig:
    """Build a validated RuntimeConfig from CLI, environment, and defaults.

    When the model is unset, query ``/v1/models`` and adopt the id only if the
    server advertises exactly one. Any other outcome fails fast with a clear
    error that never includes credentials.
    """
    environ: Mapping[str, str] = os.environ if env is None else env

    def pick(attribute: str, env_name: str, parser, default):
        value = _cli_value(cli, attribute)
        if value is not None:
            return parser(value, env_name)
        value = environ.get(env_name)
        if value is not None and str(value).strip():
            return parser(value, env_name)
        return default

    url = pick("remote_url", "ROBOT_REMOTE_URL", _parse_text, DEFAULT_REMOTE_URL)
    api_key = pick("remote_api_key", "ROBOT_REMOTE_API_KEY", _parse_text, None)

    # CLI wins over env: --remote-model, then the legacy --model alias.
    model = _cli_value(cli, "remote_model")
    if model is None:
        model = _cli_value(cli, "model")
    if model is not None:
        model = _parse_text(model, "--model/--remote-model")
    else:
        model_value = environ.get("ROBOT_REMOTE_MODEL")
        if model_value is not None and str(model_value).strip():
            model = _parse_text(model_value, "ROBOT_REMOTE_MODEL")

    config = RuntimeConfig(
        url=url,
        model=model,
        api_key=api_key,
        timeout=pick("remote_timeout", "ROBOT_REMOTE_TIMEOUT", _parse_float, DEFAULT_TIMEOUT),
        max_retries=pick(
            "remote_max_retries", "ROBOT_REMOTE_MAX_RETRIES", _parse_int, DEFAULT_MAX_RETRIES
        ),
        verify_tls=pick(
            "remote_verify_tls", "ROBOT_REMOTE_VERIFY_TLS", _parse_bool, DEFAULT_VERIFY_TLS
        ),
        max_tokens=pick("max_tokens", "ROBOT_LLM_MAX_TOKENS", _parse_int, DEFAULT_MAX_TOKENS),
        vision_max_tokens=pick(
            "vision_max_tokens", "ROBOT_VISION_MAX_TOKENS", _parse_int, DEFAULT_VISION_MAX_TOKENS
        ),
        memory_max_turns=pick(
            "memory_max_turns", "ROBOT_MEMORY_MAX_TURNS", _parse_int, DEFAULT_MEMORY_MAX_TURNS
        ),
        image_max_dim=pick(
            "image_max_dim", "ROBOT_IMAGE_MAX_DIM", _parse_int, DEFAULT_IMAGE_MAX_DIM
        ),
    )

    if config.model is None:
        if not fetch_model:
            raise ConfigError(
                "ROBOT_REMOTE_MODEL is required (model auto-detection is disabled)."
            )
        fetcher = model_fetcher or fetch_model_ids
        try:
            ids = fetcher(config)
        except Exception as exc:
            raise ConfigError(
                f"ROBOT_REMOTE_MODEL is unset and the server at {config.url} could "
                "not be queried for models. Start the server or set "
                "ROBOT_REMOTE_MODEL explicitly."
            ) from exc
        if len(ids) != 1:
            raise ConfigError(
                f"ROBOT_REMOTE_MODEL is unset and the server at {config.url} "
                f"advertises {len(ids)} models; set ROBOT_REMOTE_MODEL to one of: "
                f"{', '.join(ids) if ids else '(none)'}."
            )
        config = replace(config, model=ids[0])

    return config
