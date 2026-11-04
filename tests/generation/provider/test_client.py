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

import http.client
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from usdcraft.pipeline.provider import (
    DEFAULT_NVIDIA_BASE_URL,
    DEFAULT_NVIDIA_MAX_REQUEST_BYTES,
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_NVIDIA_REASONING_EFFORT,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    PROVIDER_ENDPOINTS,
    NvidiaHttpResponse,
    NvidiaProviderConfig,
    NvidiaRequestError,
    NvidiaResponsesClient,
    resolve_nvidia_runtime_identity,
    resolve_provider_endpoint,
)


def _success_response() -> NvidiaHttpResponse:
    return NvidiaHttpResponse(
        status_code=200,
        headers={},
        body=json.dumps(
            {
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path":"model.py"}',
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            }
        ).encode(),
    )


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (
            {
                "input_tokens_details": {"cached_tokens": 8000, "cache_write_tokens": 500},
                "output_tokens_details": {"reasoning_tokens": 600},
            },
            {
                "cached_input_tokens": 8000,
                "cache_write_input_tokens": 500,
                "reasoning_output_tokens": 600,
            },
        ),
        (
            {
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            {"cached_input_tokens": 0, "cache_write_input_tokens": 0, "reasoning_output_tokens": 0},
        ),
        ({}, {}),
        ({"input_tokens_details": None, "output_tokens_details": []}, {}),
        (
            {
                "input_tokens_details": {"cached_tokens": True, "cache_write_tokens": -1},
                "output_tokens_details": {"reasoning_tokens": "600"},
            },
            {},
        ),
    ],
)
def test_client_preserves_only_reported_valid_usage_details(details, expected) -> None:
    payload = json.loads(_success_response().body)
    payload["usage"] = {
        "input_tokens": 10000,
        "output_tokens": 1000,
        "total_tokens": 11000,
        **details,
    }
    body = json.dumps(payload).encode()
    client = NvidiaResponsesClient(
        NvidiaProviderConfig(api_key="test"),
        transport=lambda *_args, **_kwargs: NvidiaHttpResponse(200, {}, body),
    )
    response = client.create_response(input_items=[], tools=[])
    assert response.usage == {
        "input_tokens": 10000,
        "output_tokens": 1000,
        "total_tokens": 11000,
        **expected,
    }


@contextmanager
def _http_server(
    handler: type[BaseHTTPRequestHandler],
) -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_config_reads_repo_env_without_mutating_process_and_normalizes_reasoning(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=repo-secret\n"
        "NVIDIA_MODEL=azure/openai/test-model\n"
        "NVIDIA_BASE_URL=https://nvidia.invalid/v1\n",
        encoding="utf-8",
    )
    before = dict(os.environ)

    config = NvidiaProviderConfig.from_repo(tmp_path, reasoning_effort=" med ")

    assert config.model == "azure/openai/test-model"
    assert config.reasoning_effort == "medium"
    assert config.base_url == "https://nvidia.invalid/v1"
    assert "repo-secret" not in repr(config)
    assert os.environ == before


def test_config_without_an_api_key_names_both_supported_keys(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("NVIDIA_MODEL=some/model\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Set OPENAI_API_KEY .* or NVIDIA_API_KEY"):
        NvidiaProviderConfig.from_repo(tmp_path)


def test_openai_api_key_selects_the_openai_endpoint_and_defaults(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-openai\n", encoding="utf-8")

    config = NvidiaProviderConfig.from_repo(tmp_path)

    assert config.api_key == "sk-openai"
    assert config.endpoint == "openai"
    assert config.base_url == DEFAULT_OPENAI_BASE_URL
    assert config.model == DEFAULT_OPENAI_MODEL
    assert config.reasoning_effort == DEFAULT_NVIDIA_REASONING_EFFORT
    assert config.runtime_source == "openai-responses-api"
    assert resolve_nvidia_runtime_identity(tmp_path).model == DEFAULT_OPENAI_MODEL


def test_openai_settings_and_nvidia_precedence(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-openai\n"
        "OPENAI_MODEL=gpt-test\n"
        "OPENAI_BASE_URL=https://proxy.example/v1/\n",
        encoding="utf-8",
    )
    config = NvidiaProviderConfig.from_repo(tmp_path, api_style="chat-completions")
    assert (config.model, config.base_url) == ("gpt-test", "https://proxy.example/v1")
    assert config.runtime_source == "openai-chat-completions-api"

    # OPENAI_API_KEY wins when both keys are present.
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=nv-key\nOPENAI_API_KEY=sk-openai\nOPENAI_MODEL=gpt-test\n",
        encoding="utf-8",
    )
    config = NvidiaProviderConfig.from_repo(tmp_path)
    assert (config.api_key, config.endpoint) == ("sk-openai", "openai")
    assert (config.base_url, config.model) == (DEFAULT_OPENAI_BASE_URL, "gpt-test")
    assert config.runtime_source == "openai-responses-api"

    # NVIDIA_API_KEY alone selects the NVIDIA inference API.
    (tmp_path / ".env").write_text("NVIDIA_API_KEY=nv-key\n", encoding="utf-8")
    config = NvidiaProviderConfig.from_repo(tmp_path)
    assert (config.api_key, config.endpoint) == ("nv-key", "nvidia")
    assert (config.base_url, config.model) == (DEFAULT_NVIDIA_BASE_URL, DEFAULT_NVIDIA_MODEL)
    assert config.runtime_source == "nvidia-responses-api"


def test_provider_endpoint_resolves_without_credentials_like_from_repo(tmp_path: Path) -> None:
    assert PROVIDER_ENDPOINTS == ("nvidia", "openai")
    # No `.env` at all: the OpenAI endpoint is the default identity.
    assert resolve_provider_endpoint(tmp_path) == "openai"
    (tmp_path / ".env").write_text("NVIDIA_API_KEY=nv-key\n", encoding="utf-8")
    assert resolve_provider_endpoint(tmp_path) == "nvidia"
    assert NvidiaProviderConfig.from_repo(tmp_path).endpoint == "nvidia"
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=nv-key\nOPENAI_API_KEY=sk-openai\n", encoding="utf-8"
    )
    assert resolve_provider_endpoint(tmp_path) == "openai"
    assert NvidiaProviderConfig.from_repo(tmp_path).endpoint == "openai"


def test_config_resolves_request_byte_limit_from_repo_env_with_explicit_precedence(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=repo-secret\nNVIDIA_MAX_REQUEST_BYTES=4096\n",
        encoding="utf-8",
    )

    repo_config = NvidiaProviderConfig.from_repo(tmp_path)
    explicit_config = NvidiaProviderConfig.from_repo(tmp_path, max_request_bytes=8192)

    assert repo_config.max_request_bytes == 4096
    assert explicit_config.max_request_bytes == 8192


def test_config_prefers_the_public_request_byte_limit_name_over_the_legacy_alias(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-openai\nUSDCRAFT_MAX_REQUEST_BYTES=2048\n", encoding="utf-8"
    )
    assert NvidiaProviderConfig.from_repo(tmp_path).max_request_bytes == 2048

    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-openai\n"
        "USDCRAFT_MAX_REQUEST_BYTES=2048\n"
        "NVIDIA_MAX_REQUEST_BYTES=4096\n",
        encoding="utf-8",
    )
    assert NvidiaProviderConfig.from_repo(tmp_path).max_request_bytes == 2048

    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-openai\nUSDCRAFT_MAX_REQUEST_BYTES=0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="USDCRAFT_MAX_REQUEST_BYTES"):
        NvidiaProviderConfig.from_repo(tmp_path)


@pytest.mark.parametrize(
    "raw_limit",
    ["", "0", "-1", "+1", "1.5", "1_024", "${NVIDIA_API_KEY}"],
)
def test_config_rejects_invalid_repo_request_byte_limit(
    tmp_path: Path,
    raw_limit: str,
) -> None:
    (tmp_path / ".env").write_text(
        f"NVIDIA_API_KEY=repo-secret\nNVIDIA_MAX_REQUEST_BYTES={raw_limit}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="NVIDIA_MAX_REQUEST_BYTES"):
        NvidiaProviderConfig.from_repo(tmp_path)


def test_runtime_identity_resolves_without_credentials_and_matches_config_rules(
    tmp_path: Path,
) -> None:
    # NVIDIA_API_KEY alone selects the NVIDIA endpoint; NVIDIA_MODEL then names its model.
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=repo-secret\nNVIDIA_MODEL=azure/openai/repo-model\n",
        encoding="utf-8",
    )
    before = dict(os.environ)

    repo_identity = resolve_nvidia_runtime_identity(tmp_path, reasoning_effort=" med ")
    explicit_identity = resolve_nvidia_runtime_identity(
        tmp_path,
        model=" azure/openai/explicit-model ",
        reasoning_effort=" xhigh ",
    )

    assert repo_identity.model == "azure/openai/repo-model"
    assert repo_identity.reasoning_effort == "medium"
    assert explicit_identity.model == "azure/openai/explicit-model"
    assert explicit_identity.reasoning_effort == "xhigh"
    assert os.environ == before


def test_runtime_identity_uses_provider_defaults_without_repo_env(tmp_path: Path) -> None:
    identity = resolve_nvidia_runtime_identity(tmp_path)

    # Without any `.env`, the OpenAI endpoint and its default model are the identity.
    assert identity.model == DEFAULT_OPENAI_MODEL
    assert identity.reasoning_effort == DEFAULT_NVIDIA_REASONING_EFFORT


def test_runtime_identity_and_config_share_non_interpolating_resolution(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=repo-secret\nNVIDIA_MODEL=${NVIDIA_API_KEY}\n",
        encoding="utf-8",
    )

    identity = resolve_nvidia_runtime_identity(tmp_path)
    config = NvidiaProviderConfig.from_repo(tmp_path)

    assert identity.model == "${NVIDIA_API_KEY}"
    assert identity.reasoning_effort == DEFAULT_NVIDIA_REASONING_EFFORT
    assert (config.model, config.reasoning_effort) == (
        identity.model,
        identity.reasoning_effort,
    )


def test_config_does_not_interpolate_api_key_into_provider_metadata(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "NVIDIA_API_KEY=repo-secret\n"
        "NVIDIA_MODEL=${NVIDIA_API_KEY}\n"
        "NVIDIA_BASE_URL=https://nvidia.invalid/${NVIDIA_API_KEY}\n",
        encoding="utf-8",
    )

    config = NvidiaProviderConfig.from_repo(tmp_path)

    assert config.api_key == "repo-secret"
    assert config.model == "${NVIDIA_API_KEY}"
    assert config.base_url == "https://nvidia.invalid/${NVIDIA_API_KEY}"
    assert "repo-secret" not in repr(config)


def test_client_sends_full_input_and_honors_retry_after_and_timeout() -> None:
    calls: list[dict[str, object]] = []
    responses = [
        NvidiaHttpResponse(status_code=429, headers={"Retry-After": "2.5"}, body=b"busy"),
        _success_response(),
    ]
    sleeps: list[float] = []

    def transport(
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> NvidiaHttpResponse:
        calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": json.loads(body),
                "timeout": timeout_seconds,
            }
        )
        return responses.pop(0)

    config = NvidiaProviderConfig(
        api_key="super-secret",
        model="azure/openai/test-model",
        reasoning_effort="med",
        base_url="https://nvidia.invalid/v1",
        timeout_seconds=7.5,
        max_retries=1,
        max_retry_delay_seconds=10,
    )
    client = NvidiaResponsesClient(config, transport=transport, sleep=sleeps.append)
    history = [
        {"role": "user", "content": [{"type": "input_text", "text": "task"}]},
        {
            "type": "function_call",
            "call_id": "prior",
            "name": "read_file",
            "arguments": '{"path":"model.py"}',
        },
        {"type": "function_call_output", "call_id": "prior", "output": '{"ok":true}'},
    ]

    response = client.create_response(input_items=history, tools=[])

    assert response.response_id == "resp_1"
    assert response.usage == {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
    assert sleeps == [2.5]
    assert len(calls) == 2
    payload = calls[-1]["payload"]
    assert isinstance(payload, dict)
    assert payload["input"] == history
    assert "previous_response_id" not in payload
    assert payload["parallel_tool_calls"] is False
    assert payload["store"] is False
    assert payload["reasoning"]["effort"] == "medium"
    assert calls[-1]["timeout"] == 7.5
    assert calls[-1]["headers"]["Authorization"] == "Bearer super-secret"


def test_client_retries_transport_timeout_and_redacts_key_from_terminal_error() -> None:
    attempts = 0

    def flaky_transport(*args: object, **kwargs: object) -> NvidiaHttpResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary timeout")
        return NvidiaHttpResponse(
            status_code=400,
            headers={},
            body=b'{"error":"credential leaked-secret rejected"}',
        )

    client = NvidiaResponsesClient(
        NvidiaProviderConfig(
            api_key="leaked-secret",
            max_retries=1,
            retry_base_seconds=0,
        ),
        transport=flaky_transport,
        sleep=lambda _delay: None,
    )

    with pytest.raises(NvidiaRequestError) as caught:
        client.create_response(input_items=[], tools=[])

    assert attempts == 2
    assert "leaked-secret" not in str(caught.value)
    assert "credential" not in str(caught.value)
    assert "HTTP 400" in str(caught.value)


def test_client_retries_http_client_protocol_exception() -> None:
    attempts = 0

    def flaky_transport(*args: object, **kwargs: object) -> NvidiaHttpResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http.client.IncompleteRead(b"provider-controlled-partial-body")
        return _success_response()

    client = NvidiaResponsesClient(
        NvidiaProviderConfig(
            api_key="test-key",
            max_retries=1,
            retry_base_seconds=0,
        ),
        transport=flaky_transport,
        sleep=lambda _delay: None,
    )

    response = client.create_response(input_items=[], tools=[])

    assert attempts == 2
    assert response.response_id == "resp_1"


def test_client_classifies_terminal_http_protocol_exception_without_retaining_body() -> None:
    def broken_transport(*args: object, **kwargs: object) -> NvidiaHttpResponse:
        raise http.client.BadStatusLine("private prompt and provider body")

    client = NvidiaResponsesClient(
        NvidiaProviderConfig(api_key="test-key", max_retries=0),
        transport=broken_transport,
    )

    with pytest.raises(NvidiaRequestError) as caught:
        client.create_response(input_items=[], tools=[])

    assert caught.value.error_category == "transport_http_protocol"
    assert "private prompt" not in str(caught.value)
    assert "provider body" not in str(caught.value)


def test_client_rejects_serialized_full_history_above_local_byte_limit() -> None:
    transport_called = False

    def unexpected_transport(*args: object, **kwargs: object) -> NvidiaHttpResponse:
        nonlocal transport_called
        transport_called = True
        return _success_response()

    client = NvidiaResponsesClient(
        NvidiaProviderConfig(api_key="test-key", max_request_bytes=128),
        transport=unexpected_transport,
    )
    history = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "private prompt " * 20}],
        },
        {
            "type": "function_call_output",
            "call_id": "prior",
            "output": json.dumps({"ok": True, "content": "private tool output " * 20}),
        },
    ]

    with pytest.raises(NvidiaRequestError) as caught:
        client.create_response(input_items=history, tools=[])

    assert transport_called is False
    assert caught.value.error_category == "request_size_limit"
    assert caught.value.request_bytes is not None
    assert caught.value.request_bytes > 128
    assert caught.value.max_request_bytes == 128
    assert "private prompt" not in str(caught.value)
    assert "private tool output" not in str(caught.value)


@pytest.mark.parametrize(
    ("status_code", "expected_category"),
    [(400, "http_bad_request"), (413, "http_request_too_large")],
)
def test_client_classifies_http_request_errors_without_retaining_provider_body(
    status_code: int,
    expected_category: str,
) -> None:
    def transport(*args: object, **kwargs: object) -> NvidiaHttpResponse:
        return NvidiaHttpResponse(
            status_code=status_code,
            headers={},
            body=b"private prompt and data:image/png;base64,secret",
        )

    client = NvidiaResponsesClient(
        NvidiaProviderConfig(api_key="test-key", max_retries=0),
        transport=transport,
    )

    with pytest.raises(NvidiaRequestError) as caught:
        client.create_response(input_items=[], tools=[])

    assert caught.value.error_category == expected_category
    assert caught.value.http_status == status_code
    assert caught.value.request_bytes is not None
    assert caught.value.max_request_bytes == DEFAULT_NVIDIA_MAX_REQUEST_BYTES
    assert "private prompt" not in str(caught.value)
    assert "data:image" not in str(caught.value)


@pytest.mark.parametrize("status_code", [200, 413])
def test_default_transport_total_deadline_covers_slow_response_body(status_code: int) -> None:
    # Either body trickles out for over 10 s (>= 227 bytes at 0.05 s each), far beyond the
    # elapsed bound below, so only the total request deadline can end the read that soon.
    response_body = (
        _success_response().body
        if status_code == 200
        else b"private provider error body" + b"." * 200
    )

    class SlowBodyHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            try:
                for byte in response_body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    with _http_server(SlowBodyHandler) as server:
        client = NvidiaResponsesClient(
            NvidiaProviderConfig(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                timeout_seconds=0.12,
                max_retries=0,
            )
        )
        started = time.monotonic()

        with pytest.raises(NvidiaRequestError) as caught:
            client.create_response(input_items=[], tools=[])

        elapsed = time.monotonic() - started

    assert elapsed < 2
    assert caught.value.error_category == "transport_timeout"
    assert "private provider error body" not in str(caught.value)


def test_default_transport_cooperatively_cancels_an_in_flight_response() -> None:
    # The body trickles out for over 10 s (231 bytes at 0.05 s each), far beyond the elapsed bound.
    response_body = _success_response().body

    class SlowBodyHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            try:
                for byte in response_body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    cancellation_event = threading.Event()
    with _http_server(SlowBodyHandler) as server:
        client = NvidiaResponsesClient(
            NvidiaProviderConfig(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                timeout_seconds=5,
                max_retries=0,
            ),
            cancellation_event=cancellation_event,
        )
        timer = threading.Timer(0.1, cancellation_event.set)
        timer.start()
        started = time.monotonic()
        try:
            with pytest.raises(NvidiaRequestError) as caught:
                client.create_response(input_items=[], tools=[])
        finally:
            timer.join(timeout=1)
        elapsed = time.monotonic() - started

    assert elapsed < 2
    assert caught.value.error_category == "request_cancelled"
    assert caught.value.request_bytes is not None
    assert caught.value.max_request_bytes == DEFAULT_NVIDIA_MAX_REQUEST_BYTES


def test_default_transport_honors_process_http_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_paths: list[str] = []

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requested_paths.append(self.path)
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            response_body = _success_response().body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    with _http_server(ProxyHandler) as server:
        proxy_url = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("http_proxy", proxy_url)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        client = NvidiaResponsesClient(
            NvidiaProviderConfig(
                api_key="test-key",
                base_url="http://provider.invalid/v1",
                timeout_seconds=2,
                max_retries=0,
            )
        )

        response = client.create_response(input_items=[], tools=[])

    assert response.response_id == "resp_1"
    assert requested_paths == ["http://provider.invalid/v1/responses"]


def test_provider_default_request_timeout_covers_the_longest_observed_turn() -> None:
    # One HTTP attempt is one whole model turn; the deadline is a transport hang guard above the
    # longest plausible turn, and None disables it.
    config = NvidiaProviderConfig(api_key="test-key")

    assert config.timeout_seconds == 900.0
    assert config.max_request_bytes == DEFAULT_NVIDIA_MAX_REQUEST_BYTES

    seen: list[float | None] = []

    def transport(url, *, headers, body, timeout_seconds):  # type: ignore[no-untyped-def]
        seen.append(timeout_seconds)
        return _success_response()

    NvidiaResponsesClient(config, transport=transport).create_response(input_items=[], tools=[])
    assert seen == [900.0]

    unbounded = NvidiaProviderConfig(api_key="test-key", timeout_seconds=None)
    seen.clear()
    NvidiaResponsesClient(unbounded, transport=transport).create_response(input_items=[], tools=[])
    assert seen == [None]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_provider_config_rejects_an_invalid_explicit_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive when set"):
        NvidiaProviderConfig(api_key="test-key", timeout_seconds=timeout)
