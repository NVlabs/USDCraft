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

import base64
import hashlib
import json
import mimetypes
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Protocol

from usdcraft.pipeline.provider.client import (
    NvidiaRequestError,
    NvidiaResponse,
    NvidiaResponsesClient,
)
from usdcraft.pipeline.provider.progress import (
    ProviderProgressCallback,
    emit_provider_progress,
)
from usdcraft.pipeline.provider.telemetry import request_composition
from usdcraft.pipeline.tools import (
    ProviderAuthorWorkspace,
    ProviderCompletionStatus,
    ProviderToolExecution,
    ProviderToolExecutor,
    build_completion_tool_schema,
    build_provider_tool_schemas,
)

_SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_REFERENCE_READ_TOOLS = frozenset({"read_file", "search_docs", "find_examples"})


def _tool_reminder(finish_tool_name: str) -> str:
    return (
        "Continue authoring with the available function tools. Do not finish with an ordinary text "
        f"answer: call {finish_tool_name} with status complete or blocked when the work truly "
        "reaches that state."
    )


DEFAULT_PROVIDER_MAX_TURNS = 64
# The last N tool results of a run carry ``turns_remaining`` and a Host note, so the Author
# knows the budget is closing and finishes a passing model instead of refining it further.
TURN_BUDGET_NOTICE_TURNS = 8


def turn_budget_note(remaining: int, finish_tool_name: str) -> str:
    """The Host note attached to a tool result when ``remaining`` turns are left after this one."""

    tail = (
        f"A run that reaches its turn limit without an accepted {finish_tool_name} call fails "
        "and discards the model. Once model checks and compilation pass for the current model "
        f"revision, call {finish_tool_name} with status complete instead of refining further."
    )
    if remaining <= 0:
        return "This was the final turn of the run."
    if remaining == 1:
        return f"1 turn remains after this one: your next response must call {finish_tool_name}. {tail}"
    return f"{remaining} turns remain after this one. {tail}"


class ProviderHarnessError(RuntimeError):
    """Base error for one NVIDIA Generation-agent run."""


class ProviderHarnessExhaustedError(ProviderHarnessError):
    """The agent did not finish within the explicit turn budget."""


class ProviderHarnessCancelledError(ProviderHarnessError):
    """The local Batch coordinator cancelled this Provider item."""


class ProviderResponseClient(Protocol):
    def create_response(
        self,
        *,
        input_items: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        instructions: str | None = None,
    ) -> NvidiaResponse: ...


@dataclass(slots=True, frozen=True)
class ProviderHarnessResult:
    status: ProviderCompletionStatus
    summary: str
    provider: str
    model: str
    reasoning_effort: str
    turn_count: int
    tool_call_count: int
    elapsed_seconds: float
    usage: Mapping[str, int]
    trace_dir: Path
    usage_reported_turns: Mapping[str, int] = field(default_factory=dict)

    def runtime_report(self) -> dict[str, Any]:
        return {
            "returncode": 0,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "final_status": self.status,
            "final_summary": self.summary,
            "turn_count": self.turn_count,
            "tool_call_count": self.tool_call_count,
            "usage": dict(self.usage),
            "usage_reported_turns": dict(self.usage_reported_turns),
        }


class ProviderHarness:
    """Run one stateful NVIDIA agent over a product-supplied Author workspace."""

    def __init__(
        self,
        *,
        workspace: ProviderAuthorWorkspace,
        client: ProviderResponseClient,
        tool_executor: ProviderToolExecutor,
        max_turns: int = DEFAULT_PROVIDER_MAX_TURNS,
        progress: ProviderProgressCallback | None = None,
        cancellation_event: Event | None = None,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
        progress_label: str,
    ) -> None:
        if tool_executor.workspace != workspace:
            raise ValueError("Provider tool executor and harness use different workspaces.")
        if max_turns < 1:
            raise ValueError("Provider harness max_turns must be positive.")
        self.workspace = workspace
        self.client = client
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.progress = progress
        self.cancellation_event = cancellation_event
        self.label = str(progress_label).strip() or "Provider Author"
        self._history: list[dict[str, Any]] = []
        self._tools = (
            (
                *build_provider_tool_schemas(),
                build_completion_tool_schema(tool_executor.finish_tool_name),
            )
            if tool_schemas is None
            else tuple(json.loads(json.dumps(dict(schema))) for schema in tool_schemas)
        )
        self._recorder: _ProviderTraceRecorder | None = None
        self._last_result: ProviderHarnessResult | None = None
        self._model = ""
        self._effort = ""
        self._started = 0.0
        self._turns_used = 0
        self._tool_call_count = 0
        self._usage: dict[str, int] = {}
        self._usage_reported_turns: dict[str, int] = {}
        self._response_count = 0
        self._previous_input_count = 0

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        """Return a defensive copy of the in-memory canonical Responses history."""

        return tuple(json.loads(json.dumps(item)) for item in self._history)

    @property
    def last_result(self) -> ProviderHarnessResult | None:
        return self._last_result

    def run(
        self,
        *,
        prompt: str,
        image_path: Path | None = None,
        image_paths: Sequence[Path] | None = None,
        cache_prefix: str | None = None,
    ) -> ProviderHarnessResult:
        """Start the Author on ``prompt`` with the workspace's staged images attached.

        ``image_path`` sends one staged image (Generation's inspiration image); ``image_paths``
        sends several in order (Reconstruction's semantic image and declared renders). Every path
        must be one of the images the workspace staged for its Author.
        """

        normalized_prompt = str(prompt or "").strip()
        if not normalized_prompt:
            raise ValueError(f"{self.label} prompt must be non-empty.")
        if self._recorder is not None:
            raise ProviderHarnessError("A Provider harness runs one Author.")
        if image_path is not None and image_paths is not None:
            raise ValueError("Pass either image_path or image_paths, not both.")
        staged = _staged_images(self.workspace)
        if image_paths is not None:
            requested = tuple(Path(path).expanduser().resolve() for path in image_paths)
        elif image_path is not None:
            requested = (Path(image_path).expanduser().resolve(),)
        else:
            requested = staged
        for path in requested:
            if path not in staged:
                raise ValueError("Provider harness may send only the workspace's staged images.")
        staged_image: Path | None = requested[0] if requested else None

        self._model, self._effort = _client_runtime(self.client)
        self._recorder = _ProviderTraceRecorder(
            self.workspace,
            model=self._model,
            reasoning_effort=self._effort,
        )
        self._started = time.monotonic()
        self.workspace.write_task(normalized_prompt)
        initial_content = _user_message_content(
            normalized_prompt, image_paths=requested, cache_prefix=cache_prefix
        )
        self._history = [
            {
                "role": "user",
                "content": initial_content,
            }
        ]
        self._recorder.event(
            "provider.run.started",
            turn=0,
            model=self._model,
            reasoning_effort=self._effort,
            input_mode="text_image" if staged_image is not None else "text",
            image_count=len(requested),
        )
        return self._run_turns(max_turns=self.max_turns)

    def _record_request_composition(self, turn: int) -> None:
        recorder = self._recorder
        assert recorder is not None
        try:
            composition = request_composition(
                self._history, self._tools, previous_item_count=self._previous_input_count
            )
            recorder.event(
                "provider.request.composition",
                turn=turn,
                revision=self.tool_executor.revision,
                model_sha256=hashlib.sha256(self.workspace.model_path.read_bytes()).hexdigest(),
                **composition,
            )
        except Exception as exc:
            # Accounting must not stop an otherwise valid Author request or expose its body.
            recorder.event(
                "provider.request.composition_unavailable", turn=turn, error_type=type(exc).__name__
            )
        self._previous_input_count = len(self._history)

    def _run_turns(self, *, max_turns: int) -> ProviderHarnessResult:
        recorder = self._recorder
        assert recorder is not None
        first_turn = self._turns_used + 1
        last_turn = self._turns_used + max_turns
        try:
            for turn in range(first_turn, last_turn + 1):
                self._turns_used = turn
                self._raise_if_cancelled()
                self._record_request_composition(turn)
                request_started = time.monotonic()
                recorder.timeline("provider.request.started", turn=turn)
                response = self.client.create_response(
                    input_items=self._history,
                    tools=self._tools,
                )
                self._raise_if_cancelled()
                self._response_count += 1
                for key, value in response.usage.items():
                    self._usage[key] = self._usage.get(key, 0) + value
                    self._usage_reported_turns[key] = self._usage_reported_turns.get(key, 0) + 1
                self._history.extend(json.loads(json.dumps(item)) for item in response.output)
                calls = _extract_function_calls(response.output)
                request_elapsed = time.monotonic() - request_started
                recorder.event(
                    "provider.response.completed",
                    turn=turn,
                    response_id=response.response_id,
                    response_status=response.status,
                    elapsed_seconds=round(request_elapsed, 6),
                    output_types=[str(item.get("type") or "") for item in response.output],
                    usage=dict(response.usage),
                    tool_call_count=len(calls),
                )
                recorder.timeline(
                    "provider.request.completed",
                    turn=turn,
                    elapsed_seconds=round(request_elapsed, 6),
                )
                tool_names = ",".join(call.name for call in calls) or "none"
                tool_elapsed_total = 0.0
                tool_failures = 0

                def emit_turn_line(status: str) -> None:
                    # One progress line per turn: the model response time, the tool(s) it called,
                    # their outcome and the resulting revision. Per-call detail stays in the trace.
                    emit_provider_progress(
                        self.progress,
                        f"{self.label} | turn={turn}/{last_turn} | "
                        f'activity="turn completed" | duration={request_elapsed:.2f}s | '
                        f"tool={json.dumps(tool_names, ensure_ascii=False)} | status={status} | "
                        f"tool_duration={tool_elapsed_total:.2f}s | "
                        f"revision={self.tool_executor.revision} | "
                        f"tokens={_format_usage(response.usage)}",
                    )

                if not calls:
                    self._history.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": _tool_reminder(self.tool_executor.finish_tool_name),
                                }
                            ],
                        }
                    )
                    recorder.event("provider.tool_reminder.appended", turn=turn)
                    emit_turn_line("reminder")
                    continue

                tool_images: list[tuple[str, tuple[Path, ...]]] = []
                for call in calls:
                    self._raise_if_cancelled()
                    self._tool_call_count += 1
                    tool_started = time.monotonic()
                    execution = self._execute_call(call)
                    self._raise_if_cancelled()
                    tool_elapsed = time.monotonic() - tool_started
                    tool_elapsed_total += tool_elapsed
                    if execution.payload.get("ok") is not True:
                        tool_failures += 1
                    payload = dict(execution.payload)
                    turns_remaining = last_turn - turn
                    if turns_remaining <= TURN_BUDGET_NOTICE_TURNS:
                        payload["turns_remaining"] = turns_remaining
                        payload["host_note"] = turn_budget_note(
                            turns_remaining, self.tool_executor.finish_tool_name
                        )
                    output_item = {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                    self._history.append(output_item)
                    if execution.image_paths:
                        tool_images.append((call.name, execution.image_paths))
                    recorder.event(
                        "provider.tool.completed",
                        turn=turn,
                        call_id=call.call_id,
                        tool_name=call.name,
                        ok=execution.payload.get("ok") is True,
                        revision=self.tool_executor.revision,
                        elapsed_seconds=round(tool_elapsed, 6),
                        **_traced_tool_fields(call, execution),
                    )
                    if execution.completion_status is None:
                        continue
                    assert execution.completion_summary is not None
                    emit_turn_line("ok" if tool_failures == 0 else "failed")
                    elapsed = time.monotonic() - self._started
                    result = ProviderHarnessResult(
                        status=execution.completion_status,
                        summary=execution.completion_summary,
                        provider=_client_endpoint(self.client),
                        model=self._model,
                        reasoning_effort=self._effort,
                        turn_count=turn,
                        tool_call_count=self._tool_call_count,
                        elapsed_seconds=elapsed,
                        usage={
                            key: value
                            for key, value in self._usage.items()
                            if self._usage_reported_turns[key] == self._response_count
                        },
                        trace_dir=recorder.trace_dir,
                        usage_reported_turns=dict(self._usage_reported_turns),
                    )
                    recorder.finish(result)
                    self._last_result = result
                    emit_provider_progress(
                        self.progress,
                        f'{self.label} | activity="agent completed" | '
                        f"status={result.status} | turns={result.turn_count} | "
                        f"tool_calls={result.tool_call_count} | elapsed={elapsed:.2f}s",
                    )
                    return result
                for tool_name, images in tool_images:
                    self._history.append(
                        {
                            "role": "user",
                            "content": _user_message_content(
                                f"Requested {tool_name} feedback for the tool result above.",
                                image_paths=images,
                            ),
                        }
                    )
                    recorder.event(
                        "provider.images.appended",
                        turn=turn,
                        source=tool_name,
                        image_paths=[str(path) for path in images],
                    )
                emit_turn_line("ok" if tool_failures == 0 else "failed")

            raise ProviderHarnessExhaustedError(
                f"NVIDIA agent did not call an accepted {self.tool_executor.finish_tool_name} "
                f"within {max_turns} turns."
            )
        except BaseException as exc:
            elapsed = time.monotonic() - self._started
            recorder.error(exc, elapsed_seconds=elapsed)
            diagnostic_suffix = _failure_progress_suffix(exc)
            emit_provider_progress(
                self.progress,
                f'{self.label} | activity="agent failed" | '
                f"error_type={type(exc).__name__} | elapsed={elapsed:.2f}s"
                f"{diagnostic_suffix}",
            )
            raise

    def _raise_if_cancelled(self) -> None:
        if self.cancellation_event is not None and self.cancellation_event.is_set():
            raise ProviderHarnessCancelledError(
                f"{self.label} was cancelled by the local Batch coordinator."
            )

    def _execute_call(self, call: _FunctionCall) -> ProviderToolExecution:
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            return ProviderToolExecution(
                {
                    "ok": False,
                    "revision": self.tool_executor.revision,
                    "error": f"Tool arguments are invalid JSON: {exc}",
                }
            )
        if not isinstance(arguments, dict):
            return ProviderToolExecution(
                {
                    "ok": False,
                    "revision": self.tool_executor.revision,
                    "error": "Tool arguments must decode to a JSON object.",
                }
            )
        return self.tool_executor.execute(call.name, arguments)


@dataclass(slots=True, frozen=True)
class _FunctionCall:
    call_id: str
    name: str
    arguments: str


def _extract_function_calls(output: Sequence[Mapping[str, Any]]) -> tuple[_FunctionCall, ...]:
    calls: list[_FunctionCall] = []
    for index, item in enumerate(output):
        if item.get("type") != "function_call":
            continue
        call_id = item.get("call_id")
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise ProviderHarnessError(
                f"NVIDIA function_call output item {index} is missing call_id."
            )
        if not isinstance(name, str) or not name:
            raise ProviderHarnessError(f"NVIDIA function_call {call_id!r} is missing name.")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        if not isinstance(arguments, str):
            raise ProviderHarnessError(
                f"NVIDIA function_call {call_id!r} arguments must be JSON text."
            )
        calls.append(_FunctionCall(call_id=call_id, name=name, arguments=arguments))
    return tuple(calls)


def _traced_tool_fields(call: _FunctionCall, execution: ProviderToolExecution) -> dict[str, Any]:
    """Model-authored arguments plus retained local result for one traced tool call.

    Successful reference reads omit their result body: the returned document/example text is
    reproducible from the repository plus the recorded arguments. Authoring tools and every failed
    call retain the complete local payload so the run's repair loop can be replayed from the trace
    alone. Provider transport error bodies never enter tool payloads, so this retains only locally
    generated content.
    """

    try:
        arguments: Any = json.loads(call.arguments)
    except json.JSONDecodeError:
        arguments = call.arguments
    fields: dict[str, Any] = {"arguments": arguments}
    if call.name in _REFERENCE_READ_TOOLS and execution.payload.get("ok") is True:
        return fields
    fields["output"] = dict(
        execution.trace_payload if execution.trace_payload is not None else execution.payload
    )
    return fields


def _format_usage(usage: Mapping[str, int]) -> str:
    return "/".join(
        str(int(usage.get(key, 0))) for key in ("input_tokens", "output_tokens", "total_tokens")
    )


def _user_message_content(
    text: str, *, image_paths: Sequence[Path], cache_prefix: str | None = None
) -> list[dict[str, Any]]:
    """One user message body: the text followed by any images as high-detail inputs.

    NVIDIA function results are JSON text only, so every image the Author sees — the staged
    inspiration image on the first turn, tool feedback or review composites — travels in a user
    message.
    """

    content: list[dict[str, Any]] = []
    if cache_prefix is not None:
        if not cache_prefix or not text.startswith(cache_prefix):
            raise ValueError("cache_prefix must be a nonempty exact prefix of the task")
        content.append(
            {
                "type": "input_text",
                "text": cache_prefix,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        )
        text = text[len(cache_prefix) :]
    if text:
        content.append({"type": "input_text", "text": text})
    if image_paths:
        # Keep a reusable prefix before images on initial, tool-feedback and resumed turns.
        content[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    for path in image_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": image_path_to_data_url(path),
                "detail": "high",
            }
        )
    return content


def image_path_to_data_url(image_path: Path) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Provider image file not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in _SUPPORTED_IMAGE_MIME_TYPES:
        raise ValueError(f"Unsupported NVIDIA image type: {path.name} ({mime_type or 'unknown'}).")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _staged_images(workspace: ProviderAuthorWorkspace) -> tuple[Path, ...]:
    """Every image the workspace staged for its Author, in attachment order."""

    attached = getattr(workspace, "attached_images", None)
    if attached is not None:
        return tuple(Path(path).expanduser().resolve() for path in attached)
    single = getattr(workspace, "input_image_path", None)
    return (Path(single).expanduser().resolve(),) if single is not None else ()


def _client_endpoint(client: ProviderResponseClient) -> str:
    """The Provider endpoint serving this client; a configless test fake reports ``openai``."""

    config = getattr(client, "config", None)
    endpoint = getattr(config, "endpoint", None)
    return str(endpoint) if isinstance(endpoint, str) and endpoint else "openai"


def _client_runtime(client: ProviderResponseClient) -> tuple[str, str]:
    if isinstance(client, NvidiaResponsesClient):
        return client.config.model, client.config.reasoning_effort
    model = str(getattr(client, "model", "fake-nvidia-model") or "fake-nvidia-model")
    effort = str(getattr(client, "reasoning_effort", "test") or "test")
    return model, effort


class _ProviderTraceRecorder:
    """Fail-open lifecycle trace.

    Retains model-authored tool arguments and locally generated authoring diagnostics; omits raw
    prompts, images, reference-read result bodies, and arbitrary provider error text.
    """

    def __init__(
        self,
        workspace: ProviderAuthorWorkspace,
        *,
        model: str,
        reasoning_effort: str,
    ) -> None:
        self.trace_dir = workspace.root / ".provider-run"
        self.trace_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.trace_dir / "events.jsonl"
        self.final_path = self.trace_dir / "final.json"
        self.stderr_path = self.trace_dir / "stderr.log"
        self.timeline_path = self.trace_dir / "timeline.jsonl"
        self.trace_errors_path = self.trace_dir / "trace_errors.log"
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.stderr_path.write_text("", encoding="utf-8")
        self.trace_errors_path.write_text("", encoding="utf-8")

    def event(self, record_type: str, **fields: Any) -> None:
        self._append_json(self.events_path, record_type, fields)

    def timeline(self, record_type: str, **fields: Any) -> None:
        self._append_json(self.timeline_path, record_type, fields)

    def finish(self, result: ProviderHarnessResult) -> None:
        payload = {
            "status": result.status,
            "summary": result.summary,
            "provider": result.provider,
            "model": result.model,
            "reasoning_effort": result.reasoning_effort,
            "turn_count": result.turn_count,
            "tool_call_count": result.tool_call_count,
            "elapsed_seconds": round(result.elapsed_seconds, 6),
            "usage": dict(result.usage),
            "usage_reported_turns": dict(result.usage_reported_turns),
        }
        self._write_json(self.final_path, payload)
        self.event(
            "provider.run.completed",
            status=result.status,
            turn=result.turn_count,
            tool_call_count=result.tool_call_count,
            elapsed_seconds=round(result.elapsed_seconds, 6),
        )

    def error(self, exc: BaseException, *, elapsed_seconds: float) -> None:
        # Arbitrary provider/protocol error text can echo inputs. Retain only its local type and
        # validated numeric/enumerated request diagnostics.
        self._safe_write(self.stderr_path, f"{type(exc).__name__}\n")
        diagnostics = _safe_request_diagnostics(exc)
        self.event(
            "provider.run.failed",
            error_type=type(exc).__name__,
            elapsed_seconds=round(elapsed_seconds, 6),
            **diagnostics,
        )

    def _append_json(self, path: Path, record_type: str, fields: Mapping[str, Any]) -> None:
        payload = {
            "record_type": record_type,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **dict(fields),
        }
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception as exc:
            self._record_trace_error(path, exc)

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        self._safe_write(path, json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n")

    def _safe_write(self, path: Path, content: str) -> None:
        try:
            path.write_text(content, encoding="utf-8")
        except Exception as exc:
            self._record_trace_error(path, exc)

    def _record_trace_error(self, path: Path, exc: BaseException) -> None:
        try:
            with self.trace_errors_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{path.name}: {type(exc).__name__}\n")
        except Exception:
            pass


def _safe_request_diagnostics(exc: BaseException) -> dict[str, int | str]:
    if not isinstance(exc, NvidiaRequestError):
        return {}
    try:
        return exc.safe_diagnostics()
    except Exception:
        return {}


def _failure_progress_suffix(exc: BaseException) -> str:
    diagnostics = _safe_request_diagnostics(exc)
    fields = [
        f"{key}={value}"
        for key in ("error_category", "http_status", "request_bytes", "max_request_bytes")
        if (value := diagnostics.get(key)) is not None
    ]
    return "" if not fields else " | " + " | ".join(fields)


__all__ = [
    "DEFAULT_PROVIDER_MAX_TURNS",
    "TURN_BUDGET_NOTICE_TURNS",
    "turn_budget_note",
    "ProviderHarness",
    "ProviderHarnessError",
    "ProviderHarnessCancelledError",
    "ProviderHarnessExhaustedError",
    "ProviderHarnessResult",
    "ProviderResponseClient",
    "image_path_to_data_url",
]
