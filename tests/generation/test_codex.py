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

import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from usdcraft.generation.codex import run_codex
from usdcraft.generation.contract import GenerationRequest
from usdcraft.generation.workspace import GenerationWorkspace, create_generation_workspace
from usdcraft.pipeline.codex import (
    CLEAN_CONTEXT_DISABLED_CODEX_FEATURES,
    CODEX_FINAL_SCHEMA,
    CodexExecutionError,
    CodexUnavailableError,
    ProcessResult,
    _run_streaming_process,
    build_codex_task_command,
)
from usdcraft.pipeline.contracts import CodexCompletionStatus
from usdcraft.pipeline.trace import CodexTraceRecorder


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    return repo


def _workspace(
    tmp_path: Path,
    *,
    with_image: bool = False,
    model: str | None = "gpt-test",
) -> tuple[GenerationRequest, GenerationWorkspace]:
    repo = _repo(tmp_path)
    image: Path | None = None
    if with_image:
        image = tmp_path / "source.png"
        image.write_bytes(b"image")
    request = GenerationRequest(
        prompt="a functional lamp",
        repo_root=repo,
        image_path=image,
        model=model,
        reasoning_effort="xhigh",
        runtime_source="explicit_generation_request",
        selection_source="explicit_generation_request",
    )
    workspace = create_generation_workspace(
        request,
        base_dir=tmp_path / "runs",
        run_id="codex_case",
    )
    return request, workspace


def build_codex_command(request, workspace, **kwargs):  # type: ignore[no-untyped-def]
    image = workspace.input_image_path
    return build_codex_task_command(
        request, workspace, attached_images=(image,) if image is not None else (), **kwargs
    )


def _write_fake_codex(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "fake_codex.py"
    executable.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    executable.chmod(0o755)
    return executable


def test_codex_command_is_one_direct_workspace_write_process(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path, with_image=True, model="gpt-codex")

    command = build_codex_command(request, workspace, executable="codex-test")

    assert command[:2] == ("codex-test", "exec")
    assert command.count("exec") == 1
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("-C") + 1] == str(workspace.root)
    assert command[command.index("--model") + 1] == "gpt-codex"
    assert command[command.index("--image") + 1] == str(workspace.input_image_path)
    assert str(request.image_path) not in command
    assert 'model_reasoning_effort="xhigh"' in command
    assert [command[index + 1] for index, value in enumerate(command) if value == "-c"].count(
        "project_doc_max_bytes=0"
    ) == 1
    assert command[-1] == "-"


def test_text_only_command_has_no_image_flag(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path)
    command = build_codex_command(request, workspace)

    assert "--image" not in command
    assert command[command.index("--model") + 1] == "gpt-test"


def test_codex_command_requires_resolved_model(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path, model=None)

    with pytest.raises(ValueError, match="resolved model"):
        build_codex_command(request, workspace)


def test_codex_command_quotes_future_reasoning_effort(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path)
    object.__setattr__(request, "reasoning_effort", 'future"tier')

    command = build_codex_command(request, workspace)

    assert 'model_reasoning_effort="future\\"tier"' in command


def test_run_codex_writes_contract_task_and_logs(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path, with_image=True)
    captured: dict[str, object] = {}

    def fake_runner(
        command: Sequence[str],
        *,
        stdin_text: str,
        cwd: Path,
        timeout_seconds: float,
        env: Mapping[str, str],
    ) -> ProcessResult:
        assert (workspace.root / ".candidate_render/active.json").is_file()
        captured.update(
            command=tuple(command),
            stdin_text=stdin_text,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            env=dict(env),
        )
        workspace.codex_last_message_path.write_text(
            json.dumps({"status": "complete", "summary": "model ready"}),
            encoding="utf-8",
        )
        return ProcessResult(returncode=0, stdout='{"type":"turn.completed"}\n', stderr="")

    result = run_codex(
        request,
        workspace,
        prompt="direct generation task",
        executable="fake-codex",
        env_overrides={"EXAMPLE": "set"},
        process_runner=fake_runner,
    )

    assert not (workspace.root / ".candidate_render/active.json").exists()
    assert result.final_message.status is CodexCompletionStatus.COMPLETE
    assert result.final_message.summary == "model ready"
    assert result.command[0] == "fake-codex"
    assert captured["stdin_text"] == "direct generation task"
    assert captured["cwd"] == workspace.root
    # The Codex arm has no time limit.
    assert captured["timeout_seconds"] is None
    # Like the Reconstruction Author, the Generation Author ignores the user's personal Codex
    # configuration and runs with the shared clean-context feature set.
    command = result.command
    assert "--ignore-user-config" in command
    disabled = [command[index + 1] for index, value in enumerate(command) if value == "--disable"]
    assert tuple(disabled) == CLEAN_CONTEXT_DISABLED_CODEX_FEATURES
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["EXAMPLE"] == "set"
    assert env["TMPDIR"] == str(workspace.temp_dir)
    assert env["UV_CACHE_DIR"] == str(workspace.cache_dir / "uv")
    assert str(request.repo_root) == env["PYTHONPATH"].split(os.pathsep)[0]
    assert workspace.task_path.read_text(encoding="utf-8") == "direct generation task\n"
    assert json.loads(workspace.codex_schema_path.read_text(encoding="utf-8")) == CODEX_FINAL_SCHEMA
    assert workspace.codex_stdout_path.read_text(encoding="utf-8").startswith("{")
    assert workspace.codex_stderr_path.read_text(encoding="utf-8") == ""
    timeline = [json.loads(line) for line in workspace.codex_timeline_path.read_text().splitlines()]
    assert timeline[0]["record_type"] == "process.launch_requested"
    assert timeline[0]["capture_mode"] == "buffered_fallback"
    assert timeline[-1]["record_type"] == "process.completed"


def test_run_codex_preserves_nonzero_output_for_debugging(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path)

    def failed_runner(*args: object, **kwargs: object) -> ProcessResult:
        return ProcessResult(returncode=2, stdout="event log", stderr="concrete failure")

    with pytest.raises(CodexExecutionError, match="status 2.*concrete failure"):
        run_codex(request, workspace, prompt="task", process_runner=failed_runner)

    assert workspace.codex_stdout_path.read_text(encoding="utf-8") == "event log"
    assert workspace.codex_stderr_path.read_text(encoding="utf-8") == "concrete failure"


def test_run_codex_requires_valid_final_json(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path)

    def no_final_runner(*args: object, **kwargs: object) -> ProcessResult:
        return ProcessResult(returncode=0, stdout="", stderr="")

    with pytest.raises(CodexExecutionError, match="did not write"):
        run_codex(request, workspace, prompt="task", process_runner=no_final_runner)


def test_run_codex_reports_missing_binary(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path)

    def missing_runner(*args: object, **kwargs: object) -> ProcessResult:
        raise FileNotFoundError("codex")

    with pytest.raises(CodexUnavailableError, match="requires the `codex` executable"):
        run_codex(request, workspace, prompt="task", process_runner=missing_runner)


def test_default_runner_streams_command_waits_and_file_writes(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path)
    executable = _write_fake_codex(
        tmp_path,
        """\
import json
import sys
import time
from pathlib import Path

arguments = sys.argv[1:]
final_path = Path(arguments[arguments.index("--output-last-message") + 1])
sys.stdin.read()

def emit(payload):
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)

emit({"type": "thread.started", "thread_id": "thread-test"})
emit({"type": "turn.started"})
emit({
    "type": "item.started",
    "item": {
        "id": "command-1",
        "type": "command_execution",
        "command": "python -c 'rewrite model.py'",
        "aggregated_output": "",
        "exit_code": None,
        "status": "in_progress",
    },
})
time.sleep(0.04)
Path("model.py").write_text("command_version = 'café'\\n", encoding="utf-8")
sys.stderr.write("command diagnostic\\n")
sys.stderr.flush()
emit({
    "type": "item.completed",
    "item": {
        "id": "command-1",
        "type": "command_execution",
        "command": "python -c 'rewrite model.py'",
        "aggregated_output": "read content α\\n",
        "exit_code": 0,
        "status": "completed",
    },
})
changed_path = str(Path.cwd() / "model.py")
emit({
    "type": "item.started",
    "item": {
        "id": "change-1",
        "type": "file_change",
        "changes": [{"path": changed_path, "kind": "update"}],
        "status": "in_progress",
    },
})
time.sleep(0.04)
Path("model.py").write_text("file_change_version = 'naïve'\\n", encoding="utf-8")
emit({
    "type": "item.completed",
    "item": {
        "id": "change-1",
        "type": "file_change",
        "changes": [{"path": changed_path, "kind": "update"}],
        "status": "completed",
    },
})
final_path.write_text(
    json.dumps({"status": "complete", "summary": "streamed"}),
    encoding="utf-8",
)
emit({"type": "turn.completed"})
""",
    )

    result = run_codex(
        request,
        workspace,
        prompt="full streamed instruction",
        executable=str(executable),
    )

    assert result.final_message.summary == "streamed"
    raw_lines = workspace.codex_stdout_path.read_text(encoding="utf-8").splitlines(keepends=True)
    timeline = [
        json.loads(line)
        for line in workspace.codex_timeline_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["sequence"] for entry in timeline] == list(range(1, len(timeline) + 1))
    observed = [
        datetime.fromisoformat(entry["observed_at_utc"].replace("Z", "+00:00"))
        for entry in timeline
    ]
    assert observed == sorted(observed)
    assert timeline[0]["record_type"] == "process.launch_requested"
    task_trace = timeline[0]["process"]["task"]
    assert task_trace["stdin"]["text"] == "full streamed instruction"
    assert task_trace["workspace_file"]["text"] == "full streamed instruction\n"
    assert (
        task_trace["workspace_file"]["sha256"]
        == hashlib.sha256(workspace.task_path.read_bytes()).hexdigest()
    )
    assert any(entry["record_type"] == "process.spawned" for entry in timeline)
    assert timeline[-1]["record_type"] == "process.completed"
    assert timeline[-1]["process"]["outcome"] == "exited"

    stdout_events = [entry for entry in timeline if entry["record_type"] == "codex.stdout"]
    assert len(stdout_events) == len(raw_lines)
    assert [entry["stream"]["offset_bytes"] for entry in stdout_events] == [
        sum(len(previous.encode("utf-8")) for previous in raw_lines[:index])
        for index in range(len(raw_lines))
    ]
    assert [entry["stream"]["sha256"] for entry in stdout_events] == [
        hashlib.sha256(line.encode("utf-8")).hexdigest() for line in raw_lines
    ]

    command_completed = next(
        entry
        for entry in stdout_events
        if entry.get("event_type") == "item.completed"
        and entry.get("item", {}).get("id") == "command-1"
    )
    assert command_completed["item_timing"]["source"] == "host_stream_observation"
    assert command_completed["item_timing"]["duration_seconds"] >= 0.03
    assert command_completed["wait_since_previous_observation"]["active_items"] == [
        {
            "item_id": "command-1",
            "item_type": "command_execution",
            "command": "python -c 'rewrite model.py'",
            "waiting_for": "item.completed",
        }
    ]
    assert command_completed["raw_event"]["item"]["aggregated_output"] == "read content α\n"
    command_model_change = next(
        change
        for change in command_completed["workspace_changes_since_previous_activity"]
        if change["path"] == "model.py"
    )
    assert command_model_change["before"]["text"] == "object_model = object()\n"
    assert command_model_change["after"]["text"] == "command_version = 'café'\n"

    file_completed = next(
        entry
        for entry in stdout_events
        if entry.get("event_type") == "item.completed"
        and entry.get("item", {}).get("id") == "change-1"
    )
    file_evidence = file_completed["file_changes"][0]
    assert file_evidence["workspace_relative_path"] == "model.py"
    assert file_evidence["before"]["text"] == "command_version = 'café'\n"
    assert file_evidence["after"]["text"] == "file_change_version = 'naïve'\n"
    assert "-command_version" in file_evidence["unified_diff"]
    assert "+file_change_version" in file_evidence["unified_diff"]
    stderr_event = next(entry for entry in timeline if entry["record_type"] == "codex.stderr")
    assert stderr_event["stream"]["text"] == "command diagnostic\n"


def test_trace_write_failure_is_non_blocking(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path)
    object.__setattr__(workspace, "codex_timeline_path", workspace.codex_dir)

    def fake_runner(
        command: Sequence[str],
        *,
        stdin_text: str,
        cwd: Path,
        timeout_seconds: float,
        env: Mapping[str, str],
    ) -> ProcessResult:
        del command, stdin_text, cwd, timeout_seconds, env
        workspace.codex_last_message_path.write_text(
            json.dumps({"status": "complete", "summary": "trace failed open"}),
            encoding="utf-8",
        )
        return ProcessResult(returncode=0, stdout='{"type":"turn.completed"}\n', stderr="")

    result = run_codex(request, workspace, prompt="task", process_runner=fake_runner)

    assert result.final_message.summary == "trace failed open"
    assert workspace.codex_trace_errors_path.is_file()
    assert "IsADirectoryError" in workspace.codex_trace_errors_path.read_text(encoding="utf-8")


def test_trace_records_launch_without_per_process_timeout(tmp_path: Path) -> None:
    _, workspace = _workspace(tmp_path)
    recorder = CodexTraceRecorder(workspace, capture_mode="streaming")
    recorder.start_process(
        command=("codex", "exec"), cwd=workspace.root, timeout_seconds=None, task_text="task"
    )
    event = json.loads(workspace.codex_timeline_path.read_text().splitlines()[0])
    assert event["record_type"] == "process.launch_requested"
    assert event["process"]["timeout_seconds"] is None
    assert event["process"]["command"] == ["codex", "exec"]
    assert not workspace.codex_trace_errors_path.exists()


def test_trace_never_reads_reported_file_outside_generation_workspace(tmp_path: Path) -> None:
    _request, workspace = _workspace(tmp_path)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("must-not-enter-trace", encoding="utf-8")
    recorder = CodexTraceRecorder(workspace, capture_mode="streaming")
    recorder.start_process(
        command=("codex", "exec"),
        cwd=workspace.root,
        timeout_seconds=1.0,
        task_text="task",
    )
    started = {
        "type": "item.started",
        "item": {
            "id": "outside-change",
            "type": "file_change",
            "changes": [{"path": str(outside), "kind": "update"}],
            "status": "in_progress",
        },
    }
    completed = {
        **started,
        "type": "item.completed",
        "item": {**started["item"], "status": "completed"},
    }
    recorder.observe_stdout_line(json.dumps(started) + "\n")
    recorder.observe_stdout_line(json.dumps(completed) + "\n")
    recorder.finish_process(outcome="exited", returncode=0, elapsed_seconds=0.1)

    timeline_text = workspace.codex_timeline_path.read_text(encoding="utf-8")
    assert "must-not-enter-trace" not in timeline_text
    timeline = [json.loads(line) for line in timeline_text.splitlines()]
    completed_event = next(
        entry
        for entry in timeline
        if entry.get("event_type") == "item.completed"
        and entry.get("item", {}).get("id") == "outside-change"
    )
    evidence = completed_event["file_changes"][0]
    assert evidence["capture_status"] == "outside_generation_workspace_not_read"
    assert evidence["workspace_relative_path"] is None
    assert evidence["before"] is None
    assert evidence["after"] is None


def test_default_runner_concurrently_drains_large_stderr(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path)
    executable = _write_fake_codex(
        tmp_path,
        """\
import json
import sys
from pathlib import Path

arguments = sys.argv[1:]
final_path = Path(arguments[arguments.index("--output-last-message") + 1])
sys.stdin.read()
sys.stderr.write("x" * 262144 + "\\n")
sys.stderr.flush()
final_path.write_text(
    json.dumps({"status": "complete", "summary": "drained"}),
    encoding="utf-8",
)
print(json.dumps({"type": "turn.completed"}), flush=True)
""",
    )

    result = run_codex(request, workspace, prompt="drain task", executable=str(executable))

    assert result.final_message.summary == "drained"
    assert result.stderr == "x" * 262144 + "\n"
    assert workspace.codex_stderr_path.read_text(encoding="utf-8") == result.stderr


def test_slow_trace_observer_cannot_truncate_raw_stdout(tmp_path: Path) -> None:
    release_observer = threading.Event()
    observer_started = threading.Event()
    observer_timed_out = threading.Event()

    def slow_observer(line: str, observed_monotonic: float, observed_at_utc: datetime) -> None:
        del line, observed_monotonic, observed_at_utc
        observer_started.set()
        release_observer.wait(timeout=10.0)

    result = _run_streaming_process(
        (
            sys.executable,
            "-c",
            "for index in range(300): print(f'event-{index}', flush=True)",
        ),
        stdin_text="",
        cwd=tmp_path,
        timeout_seconds=3.0,
        env=os.environ,
        stdout_observer=slow_observer,
        observer_timeout_callback=observer_timed_out.set,
        observer_drain_timeout_seconds=0.2,
    )
    release_observer.set()

    assert observer_started.is_set()
    assert observer_timed_out.is_set()
    assert result.stdout.splitlines() == [f"event-{index}" for index in range(300)]


def test_background_descendant_holding_pipes_cannot_hang_runner(tmp_path: Path) -> None:
    request, workspace = _workspace(tmp_path)
    executable = _write_fake_codex(
        tmp_path,
        """\
import json
import subprocess
import sys
from pathlib import Path

arguments = sys.argv[1:]
final_path = Path(arguments[arguments.index("--output-last-message") + 1])
sys.stdin.read()
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
final_path.write_text(
    json.dumps({"status": "complete", "summary": "parent exited"}),
    encoding="utf-8",
)
print(json.dumps({"type": "turn.completed"}), flush=True)
""",
    )

    started_at = time.monotonic()
    result = run_codex(request, workspace, prompt="task", executable=str(executable))

    assert result.final_message.summary == "parent exited"
    # The descendant holds the pipes for 60 s; the runner must not wait for it.
    assert time.monotonic() - started_at < 15.0


def test_buffered_file_change_does_not_invent_before_snapshot(tmp_path: Path) -> None:
    _request, workspace = _workspace(tmp_path)
    recorder = CodexTraceRecorder(workspace, capture_mode="buffered_fallback")
    recorder.start_process(
        command=("codex", "exec"),
        cwd=workspace.root,
        timeout_seconds=1.0,
        task_text="task",
    )
    change = {"path": str(workspace.model_path), "kind": "update"}
    recorder.observe_stdout_line(
        json.dumps(
            {
                "type": "item.started",
                "item": {
                    "id": "buffered-change",
                    "type": "file_change",
                    "changes": [change],
                    "status": "in_progress",
                },
            }
        )
        + "\n"
    )
    recorder.observe_stdout_line(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "buffered-change",
                    "type": "file_change",
                    "changes": [change],
                    "status": "completed",
                },
            }
        )
        + "\n"
    )
    recorder.finish_process(outcome="exited", returncode=0, elapsed_seconds=0.1)

    timeline = [
        json.loads(line)
        for line in workspace.codex_timeline_path.read_text(encoding="utf-8").splitlines()
    ]
    completed = next(
        entry
        for entry in timeline
        if entry.get("event_type") == "item.completed"
        and entry.get("item", {}).get("id") == "buffered-change"
    )
    evidence = completed["file_changes"][0]
    assert evidence["capture_status"] == "before_unavailable_buffered_replay"
    assert evidence["before"] is None
    assert completed["item_timing"]["duration_seconds"] is None


def test_malformed_events_do_not_break_item_pairing_or_offsets(tmp_path: Path) -> None:
    _request, workspace = _workspace(tmp_path)
    recorder = CodexTraceRecorder(workspace, capture_mode="streaming")
    recorder.start_process(
        command=("codex", "exec"),
        cwd=workspace.root,
        timeout_seconds=1.0,
        task_text="task",
    )
    started = (
        json.dumps(
            {
                "type": "item.started",
                "item": {
                    "id": "paired-command",
                    "type": "command_execution",
                    "command": "echo test",
                    "status": "in_progress",
                },
            }
        )
        + "\n"
    )
    malformed = "not-json\n"
    non_object = "[]\n"
    completed = (
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "paired-command",
                    "type": "command_execution",
                    "command": "echo test",
                    "status": "completed",
                    "exit_code": 0,
                    "aggregated_output": "test\n",
                },
            }
        )
        + "\n"
    )
    for line in (started, malformed, non_object, completed):
        recorder.observe_stdout_line(line)
    recorder.finish_process(outcome="exited", returncode=0, elapsed_seconds=0.1)

    timeline = [
        json.loads(line)
        for line in workspace.codex_timeline_path.read_text(encoding="utf-8").splitlines()
    ]
    stdout_events = [entry for entry in timeline if entry["record_type"] == "codex.stdout"]
    assert [entry["stream"]["offset_bytes"] for entry in stdout_events] == [
        0,
        len(started.encode("utf-8")),
        len((started + malformed).encode("utf-8")),
        len((started + malformed + non_object).encode("utf-8")),
    ]
    assert stdout_events[1]["raw_line"] == malformed
    assert stdout_events[2]["parse_error"] == "Codex JSONL event is not an object"
    assert stdout_events[-1]["item_timing"]["source"] == "host_stream_observation"
    assert stdout_events[-1]["item_timing"]["duration_seconds"] >= 0.0


def test_file_change_pairs_reordered_paths_by_identity(tmp_path: Path) -> None:
    _request, workspace = _workspace(tmp_path)
    first = workspace.root / "first.py"
    second = workspace.root / "second.py"
    first.write_text("first_before = True\n", encoding="utf-8")
    second.write_text("second_before = True\n", encoding="utf-8")
    recorder = CodexTraceRecorder(workspace, capture_mode="streaming")
    recorder.start_process(
        command=("codex", "exec"),
        cwd=workspace.root,
        timeout_seconds=1.0,
        task_text="task",
    )
    recorder.observe_stdout_line(
        json.dumps(
            {
                "type": "item.started",
                "item": {
                    "id": "multi-change",
                    "type": "file_change",
                    "changes": [
                        {"path": str(first), "kind": "update"},
                        {"path": str(second), "kind": "update"},
                    ],
                    "status": "in_progress",
                },
            }
        )
        + "\n"
    )
    first.write_text("first_after = True\n", encoding="utf-8")
    second.write_text("second_after = True\n", encoding="utf-8")
    recorder.observe_stdout_line(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "multi-change",
                    "type": "file_change",
                    "changes": [
                        {"path": str(second), "kind": "update"},
                        {"path": str(first), "kind": "update"},
                    ],
                    "status": "completed",
                },
            }
        )
        + "\n"
    )
    recorder.finish_process(outcome="exited", returncode=0, elapsed_seconds=0.1)

    timeline = [
        json.loads(line)
        for line in workspace.codex_timeline_path.read_text(encoding="utf-8").splitlines()
    ]
    completed = next(
        entry
        for entry in timeline
        if entry.get("event_type") == "item.completed"
        and entry.get("item", {}).get("id") == "multi-change"
    )
    evidence_by_path = {
        evidence["workspace_relative_path"]: evidence for evidence in completed["file_changes"]
    }
    assert evidence_by_path["first.py"]["before"]["text"] == "first_before = True\n"
    assert evidence_by_path["first.py"]["after"]["text"] == "first_after = True\n"
    assert evidence_by_path["second.py"]["before"]["text"] == "second_before = True\n"
    assert evidence_by_path["second.py"]["after"]["text"] == "second_after = True\n"
