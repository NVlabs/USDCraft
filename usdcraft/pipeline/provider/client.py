# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import http.client
import json
import math
import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Event
from typing import Any, Literal, Protocol, TypeGuard

import aiohttp
from dotenv import dotenv_values

from usdcraft.pipeline.provider.telemetry import decode_usage
from usdcraft.values import normalize_reasoning_level

# The OpenAI API is the default endpoint: `OPENAI_API_KEY` in `.env` selects it, and it is also
# the identity used when no key is configured at all.
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-6-astra"
# Alternative endpoint: `NVIDIA_API_KEY` alone (no `OPENAI_API_KEY`) selects the NVIDIA inference
# API through the same Responses client.
DEFAULT_NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "azure/openai/gpt-6-astra"
DEFAULT_NVIDIA_REASONING_EFFORT = "low"
# Per-HTTP-attempt deadline. It is a transport hang guard, not a run clock: a timed-out attempt is
# retried from scratch (same request, up to `max_retries` more attempts) and the run fails only
# when every attempt times out. Because one attempt is one whole model turn, the value must exceed
# the longest plausible turn (several minutes), so it is 900 s; `None` disables the deadline.
DEFAULT_NVIDIA_REQUEST_TIMEOUT_SECONDS: float | None = 900.0
DEFAULT_NVIDIA_MAX_REQUEST_BYTES = 64 * 1024 * 1024
# `.env` names for the per-request byte limit: the public name first, then the legacy alias that
# older `.env` files still use. The first one present wins.
MAX_REQUEST_BYTES_SETTING_NAMES: tuple[str, ...] = (
    "USDCRAFT_MAX_REQUEST_BYTES",
    "NVIDIA_MAX_REQUEST_BYTES",
)
# The Provider arm's endpoint is also its record/run identity (`provider` field).
ProviderEndpoint = Literal["nvidia", "openai"]
PROVIDER_ENDPOINTS: tuple[ProviderEndpoint, ...] = ("nvidia", "openai")
DEFAULT_PROVIDER_ENDPOINT: ProviderEndpoint = "openai"

_REQUEST_ERROR_CATEGORIES = frozenset(
    {
        "request_failed",
        "request_cancelled",
        "request_size_limit",
        "transport_timeout",
        "transport_http_protocol",
        "transport_error",
        "http_request_timeout",
        "http_request_too_large",
        "http_rate_limited",
        "http_bad_request",
        "http_client_error",
        "http_server_error",
        "http_unexpected_status",
    }
)


class NvidiaProviderError(RuntimeError):
    """Base error for the NVIDIA-only Generation transport."""


class NvidiaRequestError(NvidiaProviderError):
    """An HTTP or network request failed after the configured retries."""

    def __init__(
        self,
        message: str,
        *,
        error_category: str = "request_failed",
        http_status: int | None = None,
        request_bytes: int | None = None,
        max_request_bytes: int | None = None,
    ) -> None:
        if error_category not in _REQUEST_ERROR_CATEGORIES:
            raise ValueError(f"Unsupported NVIDIA request error category: {error_category!r}.")
        if http_status is not None and not 100 <= http_status <= 599:
            raise ValueError("Provider HTTP status metadata must be between 100 and 599.")
        for label, value in (
            ("request_bytes", request_bytes),
            ("max_request_bytes", max_request_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"NVIDIA {label} metadata must be a non-negative integer.")
        super().__init__(message)
        self.error_category = error_category
        self.http_status = http_status
        self.request_bytes = request_bytes
        self.max_request_bytes = max_request_bytes

    def safe_diagnostics(self) -> dict[str, int | str]:
        """Return only bounded local metadata that is safe to retain in traces."""

        category = (
            self.error_category
            if self.error_category in _REQUEST_ERROR_CATEGORIES
            else "request_failed"
        )
        diagnostics: dict[str, int | str] = {"error_category": category}
        if (
            isinstance(self.http_status, int)
            and not isinstance(self.http_status, bool)
            and 100 <= self.http_status <= 599
        ):
            diagnostics["http_status"] = self.http_status
        if _is_safe_byte_count(self.request_bytes):
            diagnostics["request_bytes"] = self.request_bytes
        if _is_safe_byte_count(self.max_request_bytes):
            diagnostics["max_request_bytes"] = self.max_request_bytes
        return diagnostics


class NvidiaProtocolError(NvidiaProviderError):
    """The NVIDIA endpoint returned a malformed Responses payload."""


class _NvidiaTransportCancelled(Exception):
    """Private control signal used to unwind the default async HTTP transport."""


def _is_safe_byte_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**63 - 1


@dataclass(slots=True, frozen=True)
class NvidiaRuntimeIdentity:
    """Resolved public runtime identity, independent of Provider credentials."""

    model: str = DEFAULT_OPENAI_MODEL
    reasoning_effort: str = DEFAULT_NVIDIA_REASONING_EFFORT

    def __post_init__(self) -> None:
        normalized_model = str(self.model or "").strip()
        normalized_effort = normalize_reasoning_level(str(self.reasoning_effort or ""))
        if not normalized_model:
            raise ValueError("NVIDIA model is required.")
        object.__setattr__(self, "model", normalized_model)
        object.__setattr__(self, "reasoning_effort", normalized_effort)


def _uses_openai_endpoint(values: Mapping[str, object]) -> bool:
    """The OpenAI API is the default; only NVIDIA_API_KEY without OPENAI_API_KEY selects NVIDIA."""

    openai_key = str(values.get("OPENAI_API_KEY") or "").strip()
    nvidia_key = str(values.get("NVIDIA_API_KEY") or "").strip()
    return bool(openai_key) or not nvidia_key


def resolve_provider_endpoint(repo_root: Path) -> ProviderEndpoint:
    """Resolve the endpoint ``from_repo`` selects from ``repo_root/.env``, without credentials.

    The value names the Provider-arm identity recorded in records and run summaries.
    """

    root = Path(repo_root).expanduser().resolve()
    values = dotenv_values(root / ".env", interpolate=False)
    return "openai" if _uses_openai_endpoint(values) else "nvidia"


def _resolve_nvidia_runtime_identity(
    values: Mapping[str, object],
    *,
    model: str | None,
    reasoning_effort: str | None,
) -> NvidiaRuntimeIdentity:
    if _uses_openai_endpoint(values):
        default_model = values.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    else:
        default_model = values.get("NVIDIA_MODEL") or DEFAULT_NVIDIA_MODEL
    return NvidiaRuntimeIdentity(
        model=str(model or default_model),
        reasoning_effort=str(reasoning_effort or DEFAULT_NVIDIA_REASONING_EFFORT),
    )


def resolve_nvidia_runtime_identity(
    repo_root: Path,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> NvidiaRuntimeIdentity:
    """Resolve model and effort without requiring credentials or mutating process state."""

    root = Path(repo_root).expanduser().resolve()
    values = dotenv_values(root / ".env", interpolate=False)
    return _resolve_nvidia_runtime_identity(
        values,
        model=model,
        reasoning_effort=reasoning_effort,
    )


@dataclass(slots=True, frozen=True)
class NvidiaProviderConfig:
    """Explicit NVIDIA Responses configuration with a non-printable API key."""

    api_key: str = field(repr=False)
    model: str = DEFAULT_OPENAI_MODEL
    reasoning_effort: str = DEFAULT_NVIDIA_REASONING_EFFORT
    base_url: str = DEFAULT_OPENAI_BASE_URL
    # Optional wall-clock deadline for one HTTP attempt; None means the attempt is never cut off.
    timeout_seconds: float | None = DEFAULT_NVIDIA_REQUEST_TIMEOUT_SECONDS
    max_request_bytes: int = DEFAULT_NVIDIA_MAX_REQUEST_BYTES
    max_retries: int = 3
    retry_base_seconds: float = 1.0
    max_retry_delay_seconds: float = 60.0
    api_style: str = "responses"
    # "openai" for the OpenAI API (default), "nvidia" for the NVIDIA inference API; it only names
    # the endpoint.
    endpoint: ProviderEndpoint = DEFAULT_PROVIDER_ENDPOINT

    def __post_init__(self) -> None:
        if self.endpoint not in PROVIDER_ENDPOINTS:
            raise ValueError("Provider endpoint must be nvidia or openai.")
        if self.api_style not in {"responses", "chat-completions"}:
            raise ValueError("NVIDIA api_style must be responses or chat-completions.")
        normalized_key = str(self.api_key or "").strip()
        normalized_model = str(self.model or "").strip()
        normalized_effort = normalize_reasoning_level(str(self.reasoning_effort or ""))
        normalized_base_url = str(self.base_url or "").strip().rstrip("/")
        if not normalized_key:
            raise ValueError("NVIDIA API key is required.")
        if not normalized_model:
            raise ValueError("NVIDIA model is required.")
        if not normalized_effort:
            raise ValueError("NVIDIA reasoning effort is required.")
        if not normalized_base_url.startswith(("https://", "http://")):
            raise ValueError("NVIDIA base URL must use HTTP or HTTPS.")
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValueError("NVIDIA request timeout must be finite and positive when set.")
        if isinstance(self.max_request_bytes, bool) or not isinstance(self.max_request_bytes, int):
            raise ValueError("NVIDIA max_request_bytes must be a positive integer.")
        if self.max_request_bytes <= 0:
            raise ValueError("NVIDIA max_request_bytes must be a positive integer.")
        if self.max_retries < 0:
            raise ValueError("NVIDIA max_retries must be non-negative.")
        if not math.isfinite(self.retry_base_seconds) or self.retry_base_seconds < 0:
            raise ValueError("NVIDIA retry_base_seconds must be finite and non-negative.")
        if not math.isfinite(self.max_retry_delay_seconds) or self.max_retry_delay_seconds < 0:
            raise ValueError("NVIDIA max_retry_delay_seconds must be finite and non-negative.")
        object.__setattr__(self, "api_key", normalized_key)
        object.__setattr__(self, "model", normalized_model)
        object.__setattr__(self, "reasoning_effort", normalized_effort)
        object.__setattr__(self, "base_url", normalized_base_url)

    @classmethod
    def from_repo(
        cls,
        repo_root: Path,
        *,
        api_key: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = DEFAULT_NVIDIA_REQUEST_TIMEOUT_SECONDS,
        max_request_bytes: int | None = None,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        max_retry_delay_seconds: float = 60.0,
        api_style: str = "responses",
    ) -> NvidiaProviderConfig:
        """Read only Provider values from ``repo_root/.env`` without mutating the process env.

        ``OPENAI_API_KEY`` selects the OpenAI API (default) with its optional ``OPENAI_MODEL`` and
        ``OPENAI_BASE_URL``. ``NVIDIA_API_KEY`` alone selects the NVIDIA inference API with
        ``NVIDIA_MODEL`` and ``NVIDIA_BASE_URL``; when both keys are set, ``OPENAI_API_KEY`` wins.
        """

        root = Path(repo_root).expanduser().resolve()
        # Keep model/base metadata from expanding a credential into a trace or record field.
        values = dotenv_values(root / ".env", interpolate=False)
        openai = api_key is None and _uses_openai_endpoint(values)
        key_name = "OPENAI_API_KEY" if openai else "NVIDIA_API_KEY"
        resolved_key = str(api_key or values.get(key_name) or "").strip()
        if not resolved_key:
            raise ValueError(
                "No API key found in the explicit override or repository .env. "
                "Set OPENAI_API_KEY (OpenAI API) or NVIDIA_API_KEY (NVIDIA inference API)."
            )
        if openai:
            default_base_url = values.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        else:
            default_base_url = values.get("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
        runtime_identity = _resolve_nvidia_runtime_identity(
            values,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        limit_name = next(
            (name for name in MAX_REQUEST_BYTES_SETTING_NAMES if name in values), None
        )
        if max_request_bytes is not None:
            resolved_max_request_bytes = max_request_bytes
        elif limit_name is not None:
            resolved_max_request_bytes = _parse_positive_decimal_setting(
                values.get(limit_name),
                name=limit_name,
            )
        else:
            resolved_max_request_bytes = DEFAULT_NVIDIA_MAX_REQUEST_BYTES
        return cls(
            api_key=resolved_key,
            model=runtime_identity.model,
            reasoning_effort=runtime_identity.reasoning_effort,
            base_url=str(base_url or default_base_url),
            timeout_seconds=timeout_seconds,
            max_request_bytes=resolved_max_request_bytes,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            max_retry_delay_seconds=max_retry_delay_seconds,
            api_style=api_style,
            endpoint="openai" if openai else "nvidia",
        )

    @property
    def runtime_source(self) -> str:
        """Recorded endpoint and protocol, e.g. ``openai-responses-api``."""

        return f"{self.endpoint}-{self.api_style}-api"


def _parse_positive_decimal_setting(raw: object, *, name: str) -> int:
    normalized = str(raw or "").strip()
    if not normalized.isascii() or not normalized.isdecimal():
        raise ValueError(f"{name} must be a positive base-10 integer.")
    value = int(normalized)
    if value <= 0:
        raise ValueError(f"{name} must be a positive base-10 integer.")
    return value


@dataclass(slots=True, frozen=True)
class NvidiaHttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes


class NvidiaHttpTransport(Protocol):
    def __call__(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float | None,
    ) -> NvidiaHttpResponse: ...


def _aiohttp_transport(
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float | None,
    cancellation_event: Event | None = None,
) -> NvidiaHttpResponse:
    """Execute one request in this calling thread, under an aiohttp total deadline only when set.

    ``timeout_seconds=None`` disables aiohttp's default five-minute total so a long reasoning turn
    is never cut off; the cancellation event remains the user's way to stop such a request.
    """

    async def post() -> NvidiaHttpResponse:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(url, headers=dict(headers), data=body) as response:
                response_body = await response.read()
                return NvidiaHttpResponse(
                    status_code=int(response.status),
                    headers={str(key): str(value) for key, value in response.headers.items()},
                    body=response_body,
                )

    async def post_with_cancellation() -> NvidiaHttpResponse:
        if cancellation_event is None:
            return await post()
        request_task = asyncio.create_task(post())
        try:
            while True:
                if cancellation_event.is_set():
                    request_task.cancel()
                    with suppress(BaseException):
                        await request_task
                    raise _NvidiaTransportCancelled
                done, _pending = await asyncio.wait((request_task,), timeout=0.05)
                if done:
                    return request_task.result()
        finally:
            if not request_task.done():
                request_task.cancel()
                with suppress(BaseException):
                    await request_task

    try:
        return asyncio.run(post_with_cancellation())
    except TimeoutError as exc:
        raise TimeoutError("Provider HTTP request exceeded its wall-clock deadline.") from exc
    except aiohttp.ClientError as exc:
        # Provider-controlled payloads are intentionally excluded from this transport diagnostic.
        raise ConnectionError(f"Provider HTTP transport failed ({type(exc).__name__}).") from exc


@dataclass(slots=True, frozen=True)
class NvidiaResponse:
    response_id: str | None
    status: str | None
    output: tuple[dict[str, Any], ...]
    usage: Mapping[str, int]


class NvidiaResponsesClient:
    """Small synchronous client for NVIDIA's OpenAI-compatible Responses endpoint.

    Conversation state intentionally does not live here or at the provider. The harness passes the
    complete canonical Responses ``input`` history on every call, which keeps memory deterministic
    for a single asset run and prevents state from leaking between batch rows.
    """

    def __init__(
        self,
        config: NvidiaProviderConfig,
        *,
        transport: NvidiaHttpTransport = _aiohttp_transport,
        sleep: Callable[[float], None] = time.sleep,
        cancellation_event: Event | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._sleep = sleep
        self._cancellation_event = cancellation_event

    @property
    def responses_url(self) -> str:
        if self.config.api_style == "chat-completions":
            if self.config.base_url.endswith("/chat/completions"):
                return self.config.base_url
            return f"{self.config.base_url}/chat/completions"
        if self.config.base_url.endswith("/responses"):
            return self.config.base_url
        return f"{self.config.base_url}/responses"

    def create_response(
        self,
        *,
        input_items: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        instructions: str | None = None,
    ) -> NvidiaResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": list(input_items),
            "tools": list(tools),
            "parallel_tool_calls": False,
            "reasoning": {
                "effort": self.config.reasoning_effort,
                "summary": "auto",
            },
            "include": ["reasoning.encrypted_content"],
            "store": False,
        }
        if instructions is not None and instructions.strip():
            payload["instructions"] = instructions
        if self.config.reasoning_effort == "provider-default":
            payload.pop("reasoning")
            payload.pop("include")
        if self.config.api_style == "chat-completions":
            from usdcraft.pipeline.provider.chat import chat_request

            payload = chat_request(payload)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._raise_if_cancelled(request_bytes=len(encoded))
        if len(encoded) > self.config.max_request_bytes:
            raise NvidiaRequestError(
                "NVIDIA Responses request exceeds the configured serialized request size limit "
                f"({len(encoded)} bytes > {self.config.max_request_bytes} bytes).",
                error_category="request_size_limit",
                request_bytes=len(encoded),
                max_request_bytes=self.config.max_request_bytes,
            )
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response: NvidiaHttpResponse | None = None
        last_transport_error: BaseException | None = None
        for attempt in range(self.config.max_retries + 1):
            self._raise_if_cancelled(request_bytes=len(encoded))
            try:
                if self._transport is _aiohttp_transport:
                    response = _aiohttp_transport(
                        self.responses_url,
                        headers=headers,
                        body=encoded,
                        timeout_seconds=self.config.timeout_seconds,
                        cancellation_event=self._cancellation_event,
                    )
                else:
                    response = self._transport(
                        self.responses_url,
                        headers=headers,
                        body=encoded,
                        timeout_seconds=self.config.timeout_seconds,
                    )
                last_transport_error = None
            except _NvidiaTransportCancelled as exc:
                raise self._cancelled_error(request_bytes=len(encoded)) from exc
            except (
                TimeoutError,
                urllib.error.URLError,
                ConnectionError,
                OSError,
                http.client.HTTPException,
            ) as exc:
                last_transport_error = exc
                if attempt >= self.config.max_retries:
                    break
                self._retry_sleep(
                    self._retry_delay(attempt=attempt, headers={}),
                    request_bytes=len(encoded),
                )
                continue

            self._raise_if_cancelled(request_bytes=len(encoded))
            if 200 <= response.status_code < 300:
                return self._decode_response(response.body)
            if not _is_retryable_status(response.status_code) or attempt >= self.config.max_retries:
                raise self._http_error(response, request_bytes=len(encoded))
            self._retry_sleep(
                self._retry_delay(attempt=attempt, headers=response.headers),
                request_bytes=len(encoded),
            )

        if last_transport_error is not None:
            error_name = type(last_transport_error).__name__
            raise NvidiaRequestError(
                f"NVIDIA Responses request failed after retries ({error_name}).",
                error_category=_transport_error_category(last_transport_error),
            ) from last_transport_error
        raise NvidiaRequestError("NVIDIA Responses request failed after retries.")

    def _raise_if_cancelled(self, *, request_bytes: int) -> None:
        if self._cancellation_event is not None and self._cancellation_event.is_set():
            raise self._cancelled_error(request_bytes=request_bytes)

    def _cancelled_error(self, *, request_bytes: int) -> NvidiaRequestError:
        return NvidiaRequestError(
            "NVIDIA Responses request was cancelled by the local Batch coordinator.",
            error_category="request_cancelled",
            request_bytes=request_bytes,
            max_request_bytes=self.config.max_request_bytes,
        )

    def _retry_sleep(self, delay: float, *, request_bytes: int) -> None:
        if self._cancellation_event is None:
            self._sleep(delay)
            return
        if self._cancellation_event.wait(delay):
            raise self._cancelled_error(request_bytes=request_bytes)

    def _decode_response(self, body: bytes) -> NvidiaResponse:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NvidiaProtocolError("NVIDIA Responses endpoint returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise NvidiaProtocolError("NVIDIA Responses payload must be a JSON object.")
        if self.config.api_style == "chat-completions":
            from usdcraft.pipeline.provider.chat import chat_response

            try:
                payload = chat_response(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise NvidiaProtocolError("Malformed Chat Completions response.") from exc
        raw_output = payload.get("output")
        if not isinstance(raw_output, list):
            raise NvidiaProtocolError("NVIDIA Responses payload is missing an output list.")
        output: list[dict[str, Any]] = []
        for index, item in enumerate(raw_output):
            if not isinstance(item, dict):
                raise NvidiaProtocolError(
                    f"NVIDIA Responses output item {index} must be a JSON object."
                )
            output.append(item)
        usage = decode_usage(payload.get("usage"))
        response_id = payload.get("id")
        status = payload.get("status")
        return NvidiaResponse(
            response_id=response_id if isinstance(response_id, str) else None,
            status=status if isinstance(status, str) else None,
            output=tuple(output),
            usage=usage,
        )

    def _retry_delay(self, *, attempt: int, headers: Mapping[str, str]) -> float:
        retry_after = _retry_after_seconds(headers)
        if retry_after is None:
            retry_after = self.config.retry_base_seconds * (2**attempt)
        return min(max(0.0, retry_after), self.config.max_retry_delay_seconds)

    def _http_error(
        self,
        response: NvidiaHttpResponse,
        *,
        request_bytes: int,
    ) -> NvidiaRequestError:
        # Error bodies are provider-controlled and may echo request text or image data. Keep them
        # out of exceptions because harness failures are retained as local diagnostics.
        status_code = response.status_code
        if status_code == 413:
            message = "NVIDIA Responses request returned HTTP 413 (request too large)."
        elif status_code == 400:
            message = (
                "NVIDIA Responses request returned HTTP 400 (bad request; verify the model, "
                "reasoning effort, and serialized input size)."
            )
        else:
            message = f"NVIDIA Responses request returned HTTP {status_code}."
        return NvidiaRequestError(
            message,
            error_category=_http_error_category(status_code),
            http_status=status_code,
            request_bytes=request_bytes,
            max_request_bytes=self.config.max_request_bytes,
        )


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 429} or 500 <= status_code <= 599


def _http_error_category(status_code: int) -> str:
    if status_code == 408:
        return "http_request_timeout"
    if status_code == 413:
        return "http_request_too_large"
    if status_code == 429:
        return "http_rate_limited"
    if status_code == 400:
        return "http_bad_request"
    if 400 <= status_code <= 499:
        return "http_client_error"
    if 500 <= status_code <= 599:
        return "http_server_error"
    return "http_unexpected_status"


def _transport_error_category(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "transport_timeout"
    if isinstance(exc, http.client.HTTPException):
        return "transport_http_protocol"
    return "transport_error"


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    raw = next(
        (str(value).strip() for key, value in headers.items() if key.lower() == "retry-after"),
        "",
    )
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


__all__ = [
    "DEFAULT_NVIDIA_BASE_URL",
    "DEFAULT_NVIDIA_MAX_REQUEST_BYTES",
    "DEFAULT_NVIDIA_MODEL",
    "DEFAULT_NVIDIA_REASONING_EFFORT",
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_PROVIDER_ENDPOINT",
    "DEFAULT_NVIDIA_REQUEST_TIMEOUT_SECONDS",
    "MAX_REQUEST_BYTES_SETTING_NAMES",
    "NvidiaHttpResponse",
    "NvidiaHttpTransport",
    "NvidiaProtocolError",
    "NvidiaProviderConfig",
    "NvidiaProviderError",
    "NvidiaRequestError",
    "NvidiaResponse",
    "NvidiaResponsesClient",
    "NvidiaRuntimeIdentity",
    "PROVIDER_ENDPOINTS",
    "ProviderEndpoint",
    "resolve_nvidia_runtime_identity",
    "resolve_provider_endpoint",
]
