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
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import SimpleQueue
from typing import Callable, Mapping, Protocol, Sequence, TextIO, cast

from usdcraft.pipeline.contracts import CodexFinalMessage, CodexRunResult
from usdcraft.pipeline.trace import CodexTraceRecorder

CODEX_FINAL_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["complete", "blocked"]},
        "summary": {"type": "string", "minLength": 1},
    },
    "required": ["status", "summary"],
}
CODEX_PROJECT_DOC_CONFIG = "project_doc_max_bytes=0"
# Codex features every product-launched clean-context Codex process disables: one Agent works from
# its generated task and staged workspace only, so it gets no sub-agents, plugins, apps, browser or
# computer use, hooks, image generation, skill discovery/installation or MCP elicitation that could
# add context, tools or delegation from outside the task. Used with ``--ignore-user-config`` so the
# user's personal Codex configuration cannot re-enable them.
CLEAN_CONTEXT_DISABLED_CODEX_FEATURES: tuple[str, ...] = (
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "multi_agent_v2",
    "plugins",
    "remote_plugin",
    "skill_mcp_dependency_install",
    "skill_search",
    "tool_call_mcp_elicitation",
    "workspace_dependencies",
)


@dataclass(slots=True, frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


class ProcessRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        stdin_text: str,
        cwd: Path,
        timeout_seconds: float | None,
        env: Mapping[str, str],
    ) -> ProcessResult: ...


class CodexTaskRequest(Protocol):
    """Runtime fields shared by direct Codex authoring tasks.

    The Codex arm has no time limit: an authoring process runs until Codex exits or the user (or
    a Batch supervisor) stops it.
    """

    @property
    def repo_root(self) -> Path: ...

    @property
    def model(self) -> str | None: ...

    @property
    def reasoning_effort(self) -> str: ...


class CodexTaskWorkspace(Protocol):
    """Filesystem fields required by the shared direct Codex runner."""

    @property
    def repo_root(self) -> Path: ...

    @property
    def root(self) -> Path: ...

    @property
    def task_path(self) -> Path: ...

    @property
    def codex_schema_path(self) -> Path: ...

    @property
    def codex_last_message_path(self) -> Path: ...

    @property
    def codex_stdout_path(self) -> Path: ...

    @property
    def codex_stderr_path(self) -> Path: ...

    @property
    def codex_timeline_path(self) -> Path: ...

    @property
    def codex_trace_errors_path(self) -> Path: ...

    @property
    def temp_dir(self) -> Path: ...

    @property
    def cache_dir(self) -> Path: ...

    def write_task(self, text: str) -> None: ...

    def write_codex_schema(self, schema: Mapping[str, object]) -> None: ...

    def write_codex_logs(self, *, stdout: str, stderr: str) -> None: ...


class CodexExecutionError(RuntimeError):
    pass


class CodexUnavailableError(CodexExecutionError):
    pass


_LineObserver = Callable[[str, float, datetime], None]
_ProcessSpawnObserver = Callable[[int], None]
_ObserverTimeoutCallback = Callable[[], None]
_LineObservation = tuple[str, str, float, datetime]


def build_codex_task_command(
    request: CodexTaskRequest,
    workspace: CodexTaskWorkspace,
    *,
    attached_images: Sequence[Path] = (),
    executable: str = "codex",
    ignore_user_config: bool = False,
    disabled_features: Sequence[str] = (),
) -> tuple[str, ...]:
    """Build the shared ephemeral ``codex exec`` command for one authoring task."""

    if workspace.repo_root != request.repo_root:
        raise ValueError("Codex workspace and request use different repository roots.")
    if request.model is None:
        raise ValueError("Codex execution requires a resolved model.")
    resolved_images = tuple(Path(path).expanduser().resolve() for path in attached_images)
    for path in resolved_images:
        if not path.is_file():
            raise FileNotFoundError(f"Codex attached image is missing: {path}")

    command = [
        executable,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
    ]
    if ignore_user_config:
        command.append("--ignore-user-config")
    command.extend(
        [
            "--ignore-rules",
            "--sandbox",
            "workspace-write",
            "--color",
            "never",
            "--json",
        ]
    )
    for feature in disabled_features:
        command.extend(("--disable", str(feature)))
    command.extend(
        [
            "--output-schema",
            str(workspace.codex_schema_path),
            "--output-last-message",
            str(workspace.codex_last_message_path),
            "-C",
            str(workspace.root),
            "--model",
            request.model,
            "-c",
            f"model_reasoning_effort={json.dumps(request.reasoning_effort, ensure_ascii=False)}",
            "-c",
            CODEX_PROJECT_DOC_CONFIG,
        ]
    )
    for path in resolved_images:
        command.extend(("--image", str(path)))
    command.append("-")
    return tuple(command)


def _default_process_runner(
    command: Sequence[str],
    *,
    stdin_text: str,
    cwd: Path,
    timeout_seconds: float | None,
    env: Mapping[str, str],
) -> ProcessResult:
    return _run_streaming_process(
        command,
        stdin_text=stdin_text,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        env=env,
    )


def _run_streaming_process(
    command: Sequence[str],
    *,
    stdin_text: str,
    cwd: Path,
    timeout_seconds: float | None,
    env: Mapping[str, str],
    stdout_observer: _LineObserver | None = None,
    stderr_observer: _LineObserver | None = None,
    spawn_observer: _ProcessSpawnObserver | None = None,
    observer_timeout_callback: _ObserverTimeoutCallback | None = None,
    observer_drain_timeout_seconds: float = 15.0,
) -> ProcessResult:
    """Run Codex while draining both output streams and observing complete lines."""

    started_at = time.monotonic()
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=dict(env),
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    if spawn_observer is not None:
        try:
            spawn_observer(process.pid)
        except Exception:
            # The debug observer is explicitly fail-open and cannot affect Codex.
            pass
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    observation_queue: SimpleQueue[_LineObservation | None] = SimpleQueue()
    observation_enqueue_lock = threading.Lock()

    def drain(
        stream: TextIO | None,
        destination: list[str],
        stream_name: str,
        has_observer: bool,
    ) -> None:
        if stream is None:
            return
        try:
            while True:
                line = stream.readline()
                if line == "":
                    return
                destination.append(line)
                if has_observer:
                    with observation_enqueue_lock:
                        observation_queue.put(
                            (
                                stream_name,
                                line,
                                time.monotonic(),
                                datetime.now(timezone.utc),
                            )
                        )
        except (OSError, ValueError):
            return

    def deliver_observations() -> None:
        while True:
            observation = observation_queue.get()
            if observation is None:
                return
            stream_name, line, observed_monotonic, observed_at_utc = observation
            observer = stdout_observer if stream_name == "stdout" else stderr_observer
            if observer is None:
                continue
            try:
                observer(line, observed_monotonic, observed_at_utc)
            except Exception:
                # The debug observer is explicitly fail-open and cannot affect Codex.
                continue

    def close_output_streams() -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            with suppress(OSError, ValueError):
                if stream is not None:
                    stream.close()

    def kill_process_group() -> None:
        with suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            with suppress(OSError):
                process.kill()

    def write_stdin() -> None:
        if process.stdin is None:
            return
        try:
            process.stdin.write(stdin_text)
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return
        finally:
            with suppress(OSError, ValueError):
                process.stdin.close()

    def settle_io_threads(*, kill_descendants_if_open: bool) -> None:
        threads = tuple(started_io_threads)
        for thread in threads:
            thread.join(timeout=0.25)
        if kill_descendants_if_open and any(thread.is_alive() for thread in threads):
            # A background descendant can outlive Codex while retaining an inherited pipe. The
            # dedicated session makes it safe to terminate only this Author process tree.
            kill_process_group()
        deadline = time.monotonic() + 5.0
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if not any(thread.is_alive() for thread in threads):
            close_output_streams()

    def finish_observation_worker() -> None:
        if not observer_thread_started:
            return
        observation_queue.put(None)
        observer_thread.join(timeout=max(0.0, observer_drain_timeout_seconds))
        if observer_thread.is_alive() and observer_timeout_callback is not None:
            try:
                observer_timeout_callback()
            except Exception:
                pass

    stdout_thread = threading.Thread(
        target=drain,
        args=(process.stdout, stdout_parts, "stdout", stdout_observer is not None),
        name="usdcraft-codex-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=(process.stderr, stderr_parts, "stderr", stderr_observer is not None),
        name="usdcraft-codex-stderr",
        daemon=True,
    )
    stdin_thread = threading.Thread(
        target=write_stdin,
        name="usdcraft-codex-stdin",
        daemon=True,
    )
    observer_thread = threading.Thread(
        target=deliver_observations,
        name="usdcraft-codex-trace-observer",
        daemon=True,
    )
    started_io_threads: list[threading.Thread] = []
    observer_thread_started = False

    try:
        observer_thread.start()
        observer_thread_started = True
        for thread in (stdout_thread, stderr_thread, stdin_thread):
            thread.start()
            started_io_threads.append(thread)
        remaining = (
            None
            if timeout_seconds is None
            else max(0.0, timeout_seconds - (time.monotonic() - started_at))
        )
        returncode = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        kill_process_group()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5.0)
        settle_io_threads(kill_descendants_if_open=False)
        finish_observation_worker()
        raise subprocess.TimeoutExpired(
            cmd=exc.cmd,
            timeout=cast(float, timeout_seconds),
            output="".join(stdout_parts),
            stderr="".join(stderr_parts),
        ) from exc
    except BaseException:
        kill_process_group()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5.0)
        settle_io_threads(kill_descendants_if_open=False)
        finish_observation_worker()
        raise

    settle_io_threads(kill_descendants_if_open=True)
    finish_observation_worker()
    return ProcessResult(
        returncode=int(returncode),
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
    )


def _build_environment(
    request: CodexTaskRequest,
    workspace: CodexTaskWorkspace,
    overrides: Mapping[str, str] | None,
    *,
    workspace_environment_variable: str,
) -> dict[str, str]:
    env = dict(os.environ)
    if overrides:
        env.update({str(key): str(value) for key, value in overrides.items()})
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(request.repo_root), existing_pythonpath) if item
    )
    env["TMPDIR"] = str(workspace.temp_dir)
    env["UV_CACHE_DIR"] = str(workspace.cache_dir / "uv")
    env[workspace_environment_variable] = str(workspace.root)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _parse_final_message(path: Path) -> CodexFinalMessage:
    if not path.is_file():
        raise CodexExecutionError(f"Codex did not write its final response: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CodexExecutionError(f"Codex final response is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CodexExecutionError("Codex final response must be a JSON object.")
    try:
        return CodexFinalMessage.from_payload(payload)
    except ValueError as exc:
        raise CodexExecutionError(str(exc)) from exc


def _stream_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


def _replay_buffered_lines(value: object, observer: _LineObserver) -> None:
    text = _stream_text(value)
    for line in text.splitlines(keepends=True):
        observer(line, time.monotonic(), datetime.now(timezone.utc))


def run_codex_task(
    request: CodexTaskRequest,
    workspace: CodexTaskWorkspace,
    *,
    prompt: str,
    attached_images: Sequence[Path] = (),
    executable: str | None = None,
    env_overrides: Mapping[str, str] | None = None,
    process_runner: ProcessRunner = _default_process_runner,
    ignore_user_config: bool = False,
    disabled_features: Sequence[str] = (),
    workspace_environment_variable: str = "USDCRAFT_AUTHOR_WORKSPACE",
    unavailable_message: str = "Codex authoring requires the `codex` executable.",
) -> CodexRunResult:
    """Run one direct, ephemeral Codex authoring process with ordered image attachments.

    The process has no deadline (the Codex arm has no time or turn limit).
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("run_codex_task requires a non-empty task.")
    prompt_text = prompt
    workspace.write_task(prompt_text)
    workspace.write_codex_schema(CODEX_FINAL_SCHEMA)
    command = build_codex_task_command(
        request,
        workspace,
        attached_images=attached_images,
        executable=(executable or os.environ.get("USDCRAFT_CODEX_BIN", "").strip() or "codex"),
        ignore_user_config=ignore_user_config,
        disabled_features=disabled_features,
    )
    env = _build_environment(
        request,
        workspace,
        env_overrides,
        workspace_environment_variable=workspace_environment_variable,
    )

    uses_streaming_runner = process_runner is _default_process_runner
    trace = CodexTraceRecorder(
        workspace,
        capture_mode="streaming" if uses_streaming_runner else "buffered_fallback",
    )
    trace.start_process(
        command=command,
        cwd=workspace.root,
        timeout_seconds=None,
        task_text=prompt_text,
    )
    started_at = time.monotonic()
    try:
        if uses_streaming_runner:
            process_result = _run_streaming_process(
                command,
                stdin_text=prompt_text,
                cwd=workspace.root,
                timeout_seconds=None,
                env=env,
                stdout_observer=trace.observe_stdout_line,
                stderr_observer=trace.observe_stderr_line,
                spawn_observer=trace.observe_process_spawned,
                observer_timeout_callback=trace.report_observer_timeout,
            )
        else:
            # Keep the public ProcessRunner call shape unchanged. Injected runners cannot provide
            # event timestamps, so replay their completed buffers without inventing them.
            process_result = process_runner(
                command,
                stdin_text=prompt_text,
                cwd=workspace.root,
                timeout_seconds=None,
                env=env,
            )
            _replay_buffered_lines(process_result.stdout, trace.observe_stdout_line)
            _replay_buffered_lines(process_result.stderr, trace.observe_stderr_line)
    except FileNotFoundError as exc:
        elapsed_seconds = time.monotonic() - started_at
        trace.finish_process(
            outcome="executable_unavailable",
            returncode=None,
            elapsed_seconds=elapsed_seconds,
            error=str(exc),
        )
        raise CodexUnavailableError(unavailable_message) from exc
    except BaseException as exc:
        elapsed_seconds = time.monotonic() - started_at
        trace.finish_process(
            outcome="runner_error",
            returncode=None,
            elapsed_seconds=elapsed_seconds,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    elapsed_seconds = time.monotonic() - started_at

    process_result = ProcessResult(
        returncode=int(process_result.returncode),
        stdout=_stream_text(process_result.stdout),
        stderr=_stream_text(process_result.stderr),
    )
    trace.finish_process(
        outcome="exited" if process_result.returncode == 0 else "nonzero_exit",
        returncode=process_result.returncode,
        elapsed_seconds=elapsed_seconds,
    )

    workspace.write_codex_logs(
        stdout=process_result.stdout,
        stderr=process_result.stderr,
    )
    if process_result.returncode != 0:
        details = (process_result.stderr or process_result.stdout or "(no output)").strip()
        raise CodexExecutionError(
            f"Codex exited with status {process_result.returncode}: {details[-4_000:]}"
        )

    final_message = _parse_final_message(workspace.codex_last_message_path)
    return CodexRunResult(
        command=command,
        returncode=process_result.returncode,
        stdout=process_result.stdout,
        stderr=process_result.stderr,
        elapsed_seconds=elapsed_seconds,
        final_message=final_message,
    )


__all__ = [
    "CLEAN_CONTEXT_DISABLED_CODEX_FEATURES",
    "CODEX_FINAL_SCHEMA",
    "CodexExecutionError",
    "CodexUnavailableError",
    "ProcessResult",
    "ProcessRunner",
    "build_codex_task_command",
    "run_codex_task",
]
