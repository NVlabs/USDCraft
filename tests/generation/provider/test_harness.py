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

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from usdcraft.generation.contract import GenerationRequest
from usdcraft.generation.workspace import GenerationWorkspace, create_generation_workspace
from usdcraft.pipeline.provider import (
    DEFAULT_NVIDIA_MAX_REQUEST_BYTES,
    NvidiaHttpResponse,
    NvidiaProviderConfig,
    NvidiaRequestError,
    NvidiaResponse,
    NvidiaResponsesClient,
    ProviderHarness,
    ProviderHarnessCancelledError,
    ProviderToolExecutor,
    ProviderToolResult,
    image_path_to_data_url,
)
from usdcraft.pipeline.tools import ProviderToolExecution


def run_provider_harness(*, prompt, image_path=None, **kwargs):  # type: ignore[no-untyped-def]
    return ProviderHarness(**kwargs).run(prompt=prompt, image_path=image_path)


def _workspace(tmp_path: Path, *, with_image: bool = False) -> GenerationWorkspace:
    repo = tmp_path / "repo"
    (repo / "sdk" / "_docs").mkdir(parents=True)
    (repo / "sdk" / "_examples").mkdir(parents=True)
    (repo / "scaffold.py").write_text("value = 1\n", encoding="utf-8")
    image = None
    if with_image:
        image = tmp_path / "reference.png"
        image.write_bytes(b"not-decoded-by-transport")
    request = GenerationRequest(prompt="test asset", repo_root=repo, image_path=image)
    return create_generation_workspace(
        request,
        base_dir=tmp_path / "runs",
        run_id="provider_harness",
    )


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


class _ScriptedClient:
    model = "azure/openai/fake"
    reasoning_effort = "high"

    def __init__(self, outputs: list[list[dict[str, Any]]]) -> None:
        self.outputs = outputs
        self.inputs: list[list[dict[str, Any]]] = []
        self.tools: list[list[dict[str, Any]]] = []

    def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None = None,
    ) -> NvidiaResponse:
        assert instructions is None
        self.inputs.append(json.loads(json.dumps(input_items)))
        self.tools.append(json.loads(json.dumps(tools)))
        output = self.outputs.pop(0)
        turn = len(self.inputs)
        return NvidiaResponse(
            response_id=f"resp_{turn}",
            status="completed",
            output=tuple(output),
            usage={"input_tokens": turn, "output_tokens": 2, "total_tokens": turn + 2},
        )


@pytest.mark.parametrize("endpoint", [None, "nvidia", "openai"])
def test_harness_result_provider_is_the_client_endpoint(
    tmp_path: Path, endpoint: str | None
) -> None:
    workspace = _workspace(tmp_path)
    client = _ScriptedClient(
        [[_call("call_finish", "finish_generation", {"status": "blocked", "summary": "ok"})]]
    )
    if endpoint is not None:
        client.config = SimpleNamespace(endpoint=endpoint)
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok("checks passed"),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok({}),
    )

    result = run_provider_harness(
        progress_label="Provider Generation",
        prompt="shared generation prompt",
        workspace=workspace,
        client=client,
        tool_executor=executor,
    )

    # A configless test fake reports the default endpoint.
    expected = endpoint or "openai"
    assert result.provider == expected
    assert json.loads((result.trace_dir / "final.json").read_text())["provider"] == expected


def test_harness_preserves_full_canonical_history_and_fresh_tool_loop(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    patch = "*** Begin Patch\n*** Update File: model.py\n@@\n-value = 1\n+value = 2\n*** End Patch"
    reasoning_item = {
        "type": "reasoning",
        "id": "reason_1",
        "encrypted_content": "opaque-provider-memory",
        "summary": [{"type": "summary_text", "text": "inspect model"}],
    }
    client = _ScriptedClient(
        [
            [reasoning_item, _call("call_read", "read_file", {"path": "model.py"})],
            [_call("call_patch", "apply_patch", {"patch": patch})],
            [_call("call_check", "check_model", {})],
            [_call("call_compile", "compile_model", {})],
            [
                _call(
                    "call_finish",
                    "finish_generation",
                    {"status": "complete", "summary": "model ready"},
                )
            ],
        ]
    )
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok("checks passed"),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(
            {"compiled": True, "render": render}
        ),
    )
    progress: list[str] = []

    result = run_provider_harness(
        progress_label="Provider Generation",
        prompt="shared generation prompt",
        workspace=workspace,
        client=client,
        tool_executor=executor,
        progress=progress.append,
    )

    assert result.status == "complete"
    assert result.summary == "model ready"
    assert result.turn_count == 5
    assert result.tool_call_count == 5
    assert result.usage == {"input_tokens": 15, "output_tokens": 10, "total_tokens": 25}
    assert result.runtime_report()["usage"] == dict(result.usage)
    assert workspace.model_path.read_text() == "value = 2\n"

    second_input = client.inputs[1]
    assert "prompt_cache_breakpoint" not in second_input[0]["content"][0]
    assert second_input[1] == reasoning_item
    assert second_input[2]["call_id"] == "call_read"
    assert second_input[3]["type"] == "function_call_output"
    assert second_input[3]["call_id"] == "call_read"
    assert all("previous_response_id" not in item for turn in client.inputs for item in turn)

    trace_dir = workspace.root / ".provider-run"
    assert result.trace_dir == trace_dir
    assert json.loads((trace_dir / "final.json").read_text())["status"] == "complete"
    assert (trace_dir / "events.jsonl").is_file()
    tool_events = {
        event["tool_name"]: event
        for event in map(json.loads, (trace_dir / "events.jsonl").read_text().splitlines())
        if event["record_type"] == "provider.tool.completed"
    }
    assert tool_events["read_file"]["arguments"] == {"path": "model.py"}
    assert "output" not in tool_events["read_file"]
    assert tool_events["apply_patch"]["arguments"] == {"patch": patch}
    assert tool_events["apply_patch"]["output"]["output"]["changed"] is True
    assert tool_events["check_model"]["output"]["output"] == "checks passed"
    assert tool_events["compile_model"]["output"]["output"] == {"compiled": True, "render": False}
    assert tool_events["finish_generation"]["output"]["status"] == "complete"
    assert all(event["elapsed_seconds"] >= 0 for event in tool_events.values())
    assert (trace_dir / "timeline.jsonl").is_file()
    assert (trace_dir / "stderr.log").read_text() == ""
    assert (trace_dir / "trace_errors.log").read_text() == ""
    assert workspace.task_path.read_text() == "shared generation prompt\n"
    # One progress line per turn: response time, tool, outcome, revision and tokens.
    assert progress[0].startswith(
        'Provider Generation | turn=1/64 | activity="turn completed" | duration='
    )
    assert 'tool="read_file" | status=ok | tool_duration=' in progress[0]
    assert "revision=0 | tokens=1/2/3" in progress[0]
    assert 'activity="turn completed"' in progress[1]
    assert 'tool="apply_patch" | status=ok' in progress[1]
    assert "revision=1" in progress[1]
    assert len(progress) == 6
    assert progress[-1].startswith(
        'Provider Generation | activity="agent completed" | status=complete | turns=5'
    )
    rendered_progress = "\n".join(progress)
    assert "shared generation prompt" not in rendered_progress
    assert "opaque-provider-memory" not in rendered_progress
    assert "*** Begin Patch" not in rendered_progress

    compositions = [
        event
        for event in map(json.loads, (trace_dir / "events.jsonl").read_text().splitlines())
        if event["record_type"] == "provider.request.composition"
    ]
    assert len(compositions) == len(client.inputs)
    previous_count = 0
    for event, sent, schemas in zip(compositions, client.inputs, client.tools, strict=True):

        def encoded(value):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()

        assert event["input_json_bytes"] == len(encoded(sent))
        assert event["tools_json_bytes"] == len(encoded(schemas))
        assert event["partitions"]["replayed"].get("item_count", 0) == previous_count
        assert [item["index"] for item in event["new_items"]] == list(
            range(previous_count, len(sent))
        )
        previous_count = len(sent)
    assert compositions[1]["groups"]["reasoning"]["encrypted_reasoning_bytes"] == len(
        "opaque-provider-memory"
    )
    assert compositions[1]["groups"]["tool_result:read_file"]["text_bytes"] == len(
        second_input[3]["output"].encode()
    )
    assert compositions[1]["model_sha256"] != compositions[2]["model_sha256"]
    assert compositions[2]["model_sha256"] == compositions[4]["model_sha256"]
    assert [event["revision"] for event in compositions] == [0, 0, 1, 1, 1]
    metadata = json.dumps(compositions)
    for private in ("shared generation prompt", "opaque-provider-memory", "value = 1", patch):
        assert private not in metadata
    assert result.usage_reported_turns == {"input_tokens": 5, "output_tokens": 5, "total_tokens": 5}


@pytest.mark.parametrize("second_cache", [None, 0, 20])
def test_cache_totals_require_complete_coverage(tmp_path: Path, second_cache: int | None) -> None:
    workspace = _workspace(tmp_path)

    class UsageClient(_ScriptedClient):
        def create_response(self, **kwargs):
            response = super().create_response(**kwargs)
            cache = 0 if len(self.inputs) == 1 else second_cache
            usage = dict(response.usage)
            if cache is not None:
                usage["cached_input_tokens"] = cache
            return replace(response, usage=usage)

    client = UsageClient(
        [
            [_call("read", "read_file", {"path": "model.py"})],
            [_call("finish", "finish_generation", {"status": "blocked", "summary": "unsupported"})],
        ]
    )
    result = run_provider_harness(
        progress_label="Provider Generation",
        prompt="test",
        workspace=workspace,
        client=client,
        tool_executor=ProviderToolExecutor(
            finish_tool_name="finish_generation",
            workspace=workspace,
            check_runner=lambda _path: ProviderToolResult.ok(),
            compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
        ),
    )
    assert result.usage_reported_turns["cached_input_tokens"] == (1 if second_cache is None else 2)
    if second_cache is None:
        assert "cached_input_tokens" not in result.usage
    else:
        assert result.usage["cached_input_tokens"] == second_cache
    assert "cache_write_input_tokens" not in result.usage
    final = json.loads((result.trace_dir / "final.json").read_text())
    assert final["usage_reported_turns"] == result.runtime_report()["usage_reported_turns"]


def test_composition_failure_does_not_change_author_request(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    client = _ScriptedClient(
        [[_call("finish", "finish_generation", {"status": "blocked", "summary": "unsupported"})]]
    )

    def fail(*args, **kwargs):
        raise RuntimeError("secret body must not appear")

    monkeypatch.setattr("usdcraft.pipeline.provider.harness.request_composition", fail)
    result = run_provider_harness(
        progress_label="Provider Generation",
        prompt="test",
        workspace=workspace,
        client=client,
        tool_executor=ProviderToolExecutor(
            finish_tool_name="finish_generation",
            workspace=workspace,
            check_runner=lambda _path: ProviderToolResult.ok(),
            compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
        ),
    )
    assert result.status == "blocked"
    assert client.inputs == [
        [{"role": "user", "content": [{"type": "input_text", "text": "test"}]}]
    ]
    events = (result.trace_dir / "events.jsonl").read_text()
    assert "provider.request.composition_unavailable" in events
    assert "secret body" not in events


def test_composition_counts_conversation_feedback_not_full_local_diagnostic(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    class CompactExecutor(ProviderToolExecutor):
        def execute(self, name, arguments):
            if name == "check_model":
                return ProviderToolExecution(
                    payload={"ok": True, "output": "short feedback"},
                    trace_payload={"ok": True, "output": "full local diagnostic " * 1000},
                )
            return super().execute(name, arguments)

    client = _ScriptedClient(
        [
            [_call("check", "check_model", {})],
            [_call("finish", "finish_generation", {"status": "blocked", "summary": "unsupported"})],
        ]
    )
    result = run_provider_harness(
        progress_label="Provider Generation",
        prompt="test",
        workspace=workspace,
        client=client,
        tool_executor=CompactExecutor(
            workspace,
            finish_tool_name="finish_generation",
            check_runner=lambda _path: ProviderToolResult.ok(),
            compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
        ),
    )
    events = list(map(json.loads, (result.trace_dir / "events.jsonl").read_text().splitlines()))
    composition = next(
        e for e in events if e["record_type"] == "provider.request.composition" and e["turn"] == 2
    )
    feedback = client.inputs[1][2]["output"]
    assert composition["groups"]["tool_result:check_model"]["text_bytes"] == len(feedback.encode())
    assert "short feedback" in feedback and "full local diagnostic" not in feedback
    check_event = next(e for e in events if e["record_type"] == "provider.tool.completed")
    assert len(check_event["output"]["output"]) > 10000


def test_tool_trace_retains_failed_reference_reads_and_check_diagnostics(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    client = _ScriptedClient(
        [
            [_call("call_read", "read_file", {"path": "missing_doc.md"})],
            [_call("call_check", "check_model", {})],
            [
                _call(
                    "call_finish",
                    "finish_generation",
                    {"status": "blocked", "summary": "diagnostics unresolved"},
                )
            ],
        ]
    )
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.error(
            {"failures": [{"test": "expect_gap", "message": "parts overlap"}]}
        ),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok({"render": render}),
    )

    result = run_provider_harness(
        progress_label="Provider Generation",
        prompt="shared generation prompt",
        workspace=workspace,
        client=client,
        tool_executor=executor,
    )

    assert result.status == "blocked"
    events = [
        json.loads(line)
        for line in (workspace.root / ".provider-run" / "events.jsonl").read_text().splitlines()
    ]
    tool_events = {
        event["tool_name"]: event
        for event in events
        if event["record_type"] == "provider.tool.completed"
    }
    read_event = tool_events["read_file"]
    assert read_event["ok"] is False
    assert read_event["arguments"] == {"path": "missing_doc.md"}
    assert "read_file may access only" in read_event["output"]["error"]
    check_event = tool_events["check_model"]
    assert check_event["ok"] is False
    assert check_event["output"]["output"]["failures"][0]["message"] == "parts overlap"


def test_harness_sends_only_staged_image_as_data_url(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, with_image=True)
    assert workspace.input_image_path is not None
    client = _ScriptedClient(
        [
            [_call("read", "read_file", {"path": "model.py"})],
            [
                _call(
                    "finish",
                    "finish_generation",
                    {"status": "blocked", "summary": "Concrete unsupported request."},
                )
            ],
        ]
    )
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
    )

    result = run_provider_harness(
        progress_label="Provider Generation",
        prompt="image task",
        workspace=workspace,
        client=client,
        tool_executor=executor,
    )

    assert result.status == "blocked"
    assert client.inputs[0][0]["content"][0] == {
        "type": "input_text",
        "text": "image task",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }
    assert client.inputs[1][0] == client.inputs[0][0]
    assert workspace.task_path.read_text() == "image task\n"
    image_part = client.inputs[0][0]["content"][1]
    assert image_part["type"] == "input_image"
    assert image_part["image_url"] == image_path_to_data_url(workspace.input_image_path)
    assert image_part["image_url"].startswith("data:image/png;base64,")
    event = next(
        event
        for event in map(json.loads, (result.trace_dir / "events.jsonl").read_text().splitlines())
        if event["record_type"] == "provider.request.composition"
    )
    assert event["groups"]["initial_task"]["image_count"] == 1
    assert event["groups"]["initial_task"]["image_url_bytes"] == len(
        image_part["image_url"].encode()
    )
    assert event["groups"]["initial_task"]["text_bytes"] == len("image task")
    assert image_part["image_url"] not in json.dumps(event)


def test_failure_trace_omits_arbitrary_provider_error_text(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    progress: list[str] = []

    class FailingClient:
        model = "azure/openai/fake"
        reasoning_effort = "high"

        def create_response(self, **_kwargs: object) -> NvidiaResponse:
            raise NvidiaRequestError("private prompt and data:image/png;base64,secret")

    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
    )

    try:
        run_provider_harness(
            progress_label="Provider Generation",
            prompt="private prompt",
            workspace=workspace,
            client=FailingClient(),
            tool_executor=executor,
            progress=progress.append,
        )
    except NvidiaRequestError:
        pass
    else:
        raise AssertionError("expected NvidiaRequestError")

    stderr = (workspace.root / ".provider-run" / "stderr.log").read_text(encoding="utf-8")
    assert stderr == "NvidiaRequestError\n"
    assert "private prompt" not in stderr
    assert "data:image" not in stderr
    assert progress[-1].startswith(
        'Provider Generation | activity="agent failed" | error_type=NvidiaRequestError'
    )
    assert "private prompt" not in "\n".join(progress)
    assert "data:image" not in "\n".join(progress)


def test_failure_trace_retains_safe_http_request_classification_only(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    progress: list[str] = []

    def transport(*args: object, **kwargs: object) -> NvidiaHttpResponse:
        return NvidiaHttpResponse(
            status_code=413,
            headers={},
            body=b"private prompt and data:image/png;base64,secret",
        )

    client = NvidiaResponsesClient(
        NvidiaProviderConfig(api_key="test-key", max_retries=0),
        transport=transport,
    )
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
    )

    try:
        run_provider_harness(
            progress_label="Provider Generation",
            prompt="private prompt",
            workspace=workspace,
            client=client,
            tool_executor=executor,
            progress=progress.append,
        )
    except NvidiaRequestError as exc:
        assert exc.error_category == "http_request_too_large"
    else:
        raise AssertionError("expected NvidiaRequestError")

    trace_dir = workspace.root / ".provider-run"
    failed_event = json.loads((trace_dir / "events.jsonl").read_text().splitlines()[-1])
    assert failed_event["record_type"] == "provider.run.failed"
    assert failed_event["error_category"] == "http_request_too_large"
    assert failed_event["http_status"] == 413
    assert failed_event["request_bytes"] > 0
    assert failed_event["max_request_bytes"] == DEFAULT_NVIDIA_MAX_REQUEST_BYTES
    retained_trace = "\n".join(
        path.read_text(encoding="utf-8") for path in trace_dir.iterdir() if path.is_file()
    )
    assert "private prompt" not in retained_trace
    assert "data:image" not in retained_trace
    assert "private prompt" not in "\n".join(progress)
    assert "data:image" not in "\n".join(progress)
    assert "error_category=http_request_too_large" in progress[-1]
    assert "http_status=413" in progress[-1]
    assert "request_bytes=" in progress[-1]
    assert f"max_request_bytes={DEFAULT_NVIDIA_MAX_REQUEST_BYTES}" in progress[-1]


def test_full_history_size_limit_is_traced_without_truncating_local_history(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    transport_called = False

    def transport(*args: object, **kwargs: object) -> NvidiaHttpResponse:
        nonlocal transport_called
        transport_called = True
        raise AssertionError("oversized request must not reach transport")

    client = NvidiaResponsesClient(
        NvidiaProviderConfig(api_key="test-key", max_request_bytes=128),
        transport=transport,
    )
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
    )
    harness = ProviderHarness(
        progress_label="Provider Generation",
        workspace=workspace,
        client=client,
        tool_executor=executor,
    )
    prompt = "private prompt retained only in canonical local history"

    try:
        harness.run(prompt=prompt)
    except NvidiaRequestError as exc:
        assert exc.error_category == "request_size_limit"
    else:
        raise AssertionError("expected NvidiaRequestError")

    assert transport_called is False
    assert harness.history[0]["content"][0]["text"] == prompt
    failed_event = json.loads(
        (workspace.root / ".provider-run" / "events.jsonl").read_text().splitlines()[-1]
    )
    assert failed_event["error_category"] == "request_size_limit"
    assert failed_event["request_bytes"] > failed_event["max_request_bytes"] == 128
    retained_trace = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (workspace.root / ".provider-run").iterdir()
        if path.is_file()
    )
    assert prompt not in retained_trace


def test_harness_stops_before_the_next_provider_turn_when_batch_cancels(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    client = _ScriptedClient([[]])
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
    )
    cancellation_event = threading.Event()
    cancellation_event.set()

    with pytest.raises(ProviderHarnessCancelledError):
        run_provider_harness(
            progress_label="Provider Generation",
            prompt="test asset",
            workspace=workspace,
            client=client,
            tool_executor=executor,
            cancellation_event=cancellation_event,
        )

    assert client.inputs == []
    failed_event = json.loads(
        (workspace.root / ".provider-run" / "events.jsonl").read_text().splitlines()[-1]
    )
    assert failed_event["record_type"] == "provider.run.failed"
    assert failed_event["error_type"] == "ProviderHarnessCancelledError"


def test_progress_callback_failure_does_not_change_generation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    client = _ScriptedClient(
        [
            [
                _call(
                    "finish",
                    "finish_generation",
                    {"status": "blocked", "summary": "Concrete unsupported request."},
                )
            ]
        ]
    )
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
    )

    def fail_progress(_message: str) -> None:
        raise RuntimeError("display unavailable")

    result = run_provider_harness(
        progress_label="Provider Generation",
        prompt="test asset",
        workspace=workspace,
        client=client,
        tool_executor=executor,
        progress=fail_progress,
    )

    assert result.status == "blocked"


def test_last_tool_results_carry_turn_budget_notes(tmp_path: Path) -> None:
    """The final TURN_BUDGET_NOTICE_TURNS tool results tell the Author how many turns remain."""

    from usdcraft.pipeline.provider.harness import (
        TURN_BUDGET_NOTICE_TURNS,
        ProviderHarnessExhaustedError,
        turn_budget_note,
    )

    workspace = _workspace(tmp_path)
    client = _ScriptedClient(
        [
            [_call("call_read", "read_file", {"path": "model.py"})],
            [_call("call_check", "check_model", {})],
            [_call("call_compile", "compile_model", {})],
            [_call("call_read2", "read_file", {"path": "model.py"})],
        ]
    )
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok("checks passed"),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok({"ok": True}),
    )
    assert TURN_BUDGET_NOTICE_TURNS >= 3
    with pytest.raises(ProviderHarnessExhaustedError):
        run_provider_harness(
            progress_label="Provider Generation",
            prompt="shared generation prompt",
            workspace=workspace,
            client=client,
            tool_executor=executor,
            max_turns=4,
        )
    outputs = [
        json.loads(item["output"])
        for item in client.inputs[-1]
        if item.get("type") == "function_call_output"
    ]
    # With a 4-turn budget every result falls inside the notice window: 3, 2, 1 turns remain.
    assert [out["turns_remaining"] for out in outputs] == [3, 2, 1]
    assert outputs[0]["host_note"] == turn_budget_note(3, "finish_generation")
    assert "3 turns remain after this one" in outputs[0]["host_note"]
    assert (
        "call finish_generation with status complete instead of refining further"
        in (outputs[0]["host_note"])
    )
    assert outputs[2]["host_note"].startswith(
        "1 turn remains after this one: your next response must call finish_generation."
    )
    assert turn_budget_note(0, "finish_generation") == "This was the final turn of the run."
    assert "finish_reconstruction" in turn_budget_note(2, "finish_reconstruction")


def test_early_tool_results_carry_no_turn_budget_note(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    client = _ScriptedClient(
        [
            [_call("call_read", "read_file", {"path": "model.py"})],
            [
                _call(
                    "call_patch",
                    "apply_patch",
                    {
                        "patch": (
                            "*** Begin Patch\n*** Update File: model.py\n@@\n-value = 1\n"
                            "+value = 2\n*** End Patch"
                        )
                    },
                )
            ],
            [_call("call_check", "check_model", {})],
            [_call("call_compile", "compile_model", {})],
            [
                _call(
                    "call_finish",
                    "finish_generation",
                    {"status": "complete", "summary": "model ready"},
                )
            ],
        ]
    )
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok("checks passed"),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok({"ok": True}),
    )
    run_provider_harness(
        progress_label="Provider Generation",
        prompt="shared generation prompt",
        workspace=workspace,
        client=client,
        tool_executor=executor,
        max_turns=64,
    )
    first_output = json.loads(
        next(item for item in client.inputs[1] if item.get("type") == "function_call_output")[
            "output"
        ]
    )
    assert "turns_remaining" not in first_output and "host_note" not in first_output
