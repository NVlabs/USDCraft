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

import difflib
import hashlib
import json
import os
import stat
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping, Protocol, Sequence


class TraceWorkspace(Protocol):
    @property
    def root(self) -> Path: ...

    @property
    def task_path(self) -> Path: ...

    @property
    def codex_timeline_path(self) -> Path: ...

    @property
    def codex_trace_errors_path(self) -> Path: ...


TRACE_SCHEMA_VERSION = 1
MAX_INLINE_TEXT_BYTES = 4 * 1024 * 1024
MAX_INLINE_EVENT_BYTES = 1024 * 1024
MAX_WORKSPACE_SCAN_FILES = 1_000
MAX_WORKSPACE_SCAN_SECONDS = 0.05
_EXCLUDED_WORKSPACE_DIRS = frozenset({".cache", ".codex-run", "tmp"})

TraceCaptureMode = Literal["streaming", "buffered_fallback"]


def _format_utc(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(slots=True, frozen=True)
class _FileState:
    path: str
    captured_at_utc: str
    exists: bool
    file_type: str
    size_bytes: int | None = None
    mtime_ns: int | None = None
    sha256: str | None = None
    text: str | None = None
    content_omitted_reason: str | None = None
    symlink_target: str | None = None
    capture_stable: bool | None = None

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.exists,
            self.file_type,
            self.size_bytes,
            self.mtime_ns,
            self.sha256,
            self.symlink_target,
        )

    def to_trace(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": self.path,
            "captured_at_utc": self.captured_at_utc,
            "exists": self.exists,
            "file_type": self.file_type,
        }
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        if self.mtime_ns is not None:
            payload["mtime_ns"] = self.mtime_ns
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        if self.text is not None:
            payload["text"] = self.text
        if self.content_omitted_reason is not None:
            payload["content_omitted_reason"] = self.content_omitted_reason
        if self.symlink_target is not None:
            payload["symlink_target"] = self.symlink_target
        if self.capture_stable is not None:
            payload["capture_stable"] = self.capture_stable
        return payload


@dataclass(slots=True)
class _ReportedFileTarget:
    reported_path: str
    reported_kind: str | None
    relative_path: str | None
    before: _FileState | None
    capture_status: str


@dataclass(slots=True)
class _PendingItem:
    item_id: str
    item_type: str
    command: str | None
    started_at_utc: str | None
    started_monotonic: float | None
    file_targets: list[_ReportedFileTarget]


class CodexTraceRecorder:
    """Best-effort Host observations for debugging one Generation Agent process.

    This recorder deliberately does not participate in Generation success, promotion, or any
    storage validity contract. Every public method is fail-open and reports recorder failures only
    to ``trace_errors.log`` when that file remains writable.
    """

    def __init__(
        self,
        workspace: TraceWorkspace,
        *,
        capture_mode: TraceCaptureMode,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self.workspace = workspace
        self.capture_mode = capture_mode
        self._root = workspace.root.resolve()
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._lock = threading.RLock()
        self._error_lock = threading.Lock()
        self._sequence = 0
        self._stdout_sequence = 0
        self._stderr_sequence = 0
        self._stdout_offset_bytes = 0
        self._stderr_offset_bytes = 0
        self._process_started_monotonic: float | None = None
        self._process_spawned_monotonic: float | None = None
        self._last_observed_monotonic: float | None = None
        self._pending: dict[str, _PendingItem] = {}
        self._inventory: dict[str, _FileState] = {}
        self._last_scan_diagnostics: dict[str, object] = {}

    def start_process(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        timeout_seconds: float | None,
        task_text: str,
    ) -> None:
        try:
            with self._lock:
                now_monotonic = self._monotonic_clock()
                now_utc = _format_utc(self._wall_clock())
                self._process_started_monotonic = now_monotonic
                self._last_observed_monotonic = now_monotonic
                stdin_bytes = task_text.encode("utf-8")
                try:
                    task_file_bytes = self.workspace.task_path.read_bytes()
                except OSError as exc:
                    task_file_bytes = None
                    task_file_error: str | None = f"{type(exc).__name__}: {exc}"
                else:
                    task_file_error = None
                workspace_file: dict[str, object] = {
                    "path": "TASK.md",
                    "persisted_trace_path": "codex_task.md",
                    "exists": task_file_bytes is not None,
                }
                if task_file_bytes is not None:
                    workspace_file.update(
                        {
                            "size_bytes": len(task_file_bytes),
                            "sha256": _sha256_bytes(task_file_bytes),
                            "text": task_file_bytes.decode("utf-8"),
                        }
                    )
                else:
                    workspace_file["capture_error"] = task_file_error
                record = self._new_record_unlocked(
                    "process.launch_requested",
                    now_utc=now_utc,
                    now_monotonic=now_monotonic,
                    include_wait=False,
                )
                record["process"] = {
                    "command": [str(part) for part in command],
                    "cwd": str(cwd),
                    "timeout_seconds": (
                        float(timeout_seconds) if timeout_seconds is not None else None
                    ),
                    "task": {
                        "stdin": {
                            "size_bytes": len(stdin_bytes),
                            "sha256": _sha256_bytes(stdin_bytes),
                            "text": task_text,
                        },
                        "workspace_file": workspace_file,
                    },
                    "workspace_observation": {
                        "excluded_top_level_directories": sorted(_EXCLUDED_WORKSPACE_DIRS),
                        "max_inline_text_bytes": MAX_INLINE_TEXT_BYTES,
                        "scope": "host_observed_activity_boundaries_not_os_syscall_audit",
                    },
                }
                self._append_record_unlocked(record)
                baseline_started = self._monotonic_clock()
                baseline_utc = _format_utc(self._wall_clock())
                self._inventory = self._scan_workspace(baseline_utc)
                baseline_finished = self._monotonic_clock()
                baseline_record = self._new_record_unlocked(
                    "workspace.baseline_completed",
                    now_utc=_format_utc(self._wall_clock()),
                    now_monotonic=baseline_finished,
                )
                baseline_record["workspace_baseline"] = {
                    "file_count": len(self._inventory),
                    "scan_seconds": round(max(0.0, baseline_finished - baseline_started), 6),
                    **self._last_scan_diagnostics,
                }
                baseline_record["active_items_after_event"] = []
                self._append_record_unlocked(baseline_record)
        except Exception as exc:  # pragma: no cover - exercised through fail-open integration tests
            self._report_error("process.launch_requested", exc)

    def observe_process_spawned(self, pid: int) -> None:
        try:
            with self._lock:
                now_monotonic = self._monotonic_clock()
                now_utc = _format_utc(self._wall_clock())
                self._process_spawned_monotonic = now_monotonic
                record = self._new_record_unlocked(
                    "process.spawned",
                    now_utc=now_utc,
                    now_monotonic=now_monotonic,
                )
                record["process"] = {"pid": int(pid)}
                record["active_items_after_event"] = []
                self._append_record_unlocked(record)
        except Exception as exc:  # pragma: no cover - individual failures are environment-specific
            self._report_error("process.spawned", exc)

    def report_observer_timeout(self) -> None:
        self._report_error(
            "observer.drain_timeout",
            TimeoutError("timeline enrichment did not drain within the bounded observer wait"),
        )

    def observe_stdout_line(
        self,
        line: str,
        observed_monotonic: float | None = None,
        observed_at_utc: datetime | None = None,
    ) -> None:
        try:
            with self._lock:
                self._observe_stdout_line_unlocked(
                    line,
                    observed_monotonic=observed_monotonic,
                    observed_at_utc=observed_at_utc,
                )
        except Exception as exc:  # pragma: no cover - individual failures are environment-specific
            self._report_error("stdout.observe", exc)

    def observe_stderr_line(
        self,
        line: str,
        observed_monotonic: float | None = None,
        observed_at_utc: datetime | None = None,
    ) -> None:
        try:
            with self._lock:
                now_monotonic = (
                    self._monotonic_clock() if observed_monotonic is None else observed_monotonic
                )
                now_utc = _format_utc(observed_at_utc or self._wall_clock())
                raw_bytes = line.encode("utf-8")
                offset = self._stderr_offset_bytes
                self._stderr_offset_bytes += len(raw_bytes)
                self._stderr_sequence += 1
                record = self._new_record_unlocked(
                    "codex.stderr",
                    now_utc=now_utc,
                    now_monotonic=now_monotonic,
                )
                stream: dict[str, object] = {
                    "file": "codex_stderr.log",
                    "line_sequence": self._stderr_sequence,
                    "offset_bytes": offset,
                    "length_bytes": len(raw_bytes),
                    "sha256": _sha256_bytes(raw_bytes),
                }
                record["stream"] = stream
                if len(raw_bytes) <= MAX_INLINE_EVENT_BYTES:
                    stream["text"] = line
                else:
                    stream["text_omitted_reason"] = (
                        f"larger_than_{MAX_INLINE_EVENT_BYTES}_byte_inline_event_limit"
                    )
                record["active_items_after_event"] = self._pending_summaries_unlocked()
                self._append_record_unlocked(record)
        except Exception as exc:  # pragma: no cover - individual failures are environment-specific
            self._report_error("stderr.observe", exc)

    def finish_process(
        self,
        *,
        outcome: str,
        returncode: int | None,
        elapsed_seconds: float,
        error: str | None = None,
    ) -> None:
        if not self._lock.acquire(timeout=1.0):
            self._report_error(
                "process.completed",
                TimeoutError("trace observer did not release its lock within 1 second"),
            )
            return
        try:
            now_monotonic = self._monotonic_clock()
            now_utc = _format_utc(self._wall_clock())
            workspace_changes = self._workspace_delta_unlocked(now_utc)
            record = self._new_record_unlocked(
                "process.completed",
                now_utc=now_utc,
                now_monotonic=now_monotonic,
            )
            interrupted_items: list[dict[str, object]] = []
            for pending in self._pending.values():
                item: dict[str, object] = {
                    "item_id": pending.item_id,
                    "item_type": pending.item_type,
                    "command": pending.command,
                    "outcome": outcome,
                }
                if pending.started_at_utc is not None:
                    item["started_at_utc"] = pending.started_at_utc
                if pending.started_monotonic is not None:
                    item["duration_seconds"] = round(
                        max(0.0, now_monotonic - pending.started_monotonic),
                        6,
                    )
                else:
                    item["duration_seconds"] = None
                interrupted_items.append(item)
            self._pending.clear()
            record["process"] = {
                "outcome": outcome,
                "returncode": returncode,
                "elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 6),
                "error": error,
                "interrupted_items": interrupted_items,
            }
            if workspace_changes:
                record["workspace_changes_since_previous_activity"] = workspace_changes
            record["workspace_scan"] = dict(self._last_scan_diagnostics)
            record["active_items_after_event"] = []
            self._append_record_unlocked(record)
        except Exception as exc:  # pragma: no cover - exercised through fail-open integration tests
            self._report_error("process.completed", exc)
        finally:
            self._lock.release()

    def _observe_stdout_line_unlocked(
        self,
        line: str,
        *,
        observed_monotonic: float | None,
        observed_at_utc: datetime | None,
    ) -> None:
        now_monotonic = (
            self._monotonic_clock() if observed_monotonic is None else observed_monotonic
        )
        now_utc = _format_utc(observed_at_utc or self._wall_clock())
        raw_bytes = line.encode("utf-8")
        offset = self._stdout_offset_bytes
        self._stdout_offset_bytes += len(raw_bytes)
        self._stdout_sequence += 1

        record = self._new_record_unlocked(
            "codex.stdout",
            now_utc=now_utc,
            now_monotonic=now_monotonic,
        )
        record["stream"] = {
            "file": "codex_events.jsonl",
            "line_sequence": self._stdout_sequence,
            "offset_bytes": offset,
            "length_bytes": len(raw_bytes),
            "sha256": _sha256_bytes(raw_bytes),
        }
        try:
            payload = json.loads(line)
        except (TypeError, ValueError) as exc:
            record["parse_error"] = str(exc)
            if len(raw_bytes) <= MAX_INLINE_EVENT_BYTES:
                record["raw_line"] = line
            else:
                record["raw_line_omitted_reason"] = (
                    f"larger_than_{MAX_INLINE_EVENT_BYTES}_byte_inline_event_limit"
                )
            record["active_items_after_event"] = self._pending_summaries_unlocked()
            self._append_record_unlocked(record)
            return

        if not isinstance(payload, dict):
            record["parse_error"] = "Codex JSONL event is not an object"
            if len(raw_bytes) <= MAX_INLINE_EVENT_BYTES:
                record["raw_event"] = payload
            else:
                record["raw_event_omitted_reason"] = (
                    f"larger_than_{MAX_INLINE_EVENT_BYTES}_byte_inline_event_limit"
                )
            record["active_items_after_event"] = self._pending_summaries_unlocked()
            self._append_record_unlocked(record)
            return

        if len(raw_bytes) <= MAX_INLINE_EVENT_BYTES:
            record["raw_event"] = payload
        else:
            record["raw_event_omitted_reason"] = (
                f"larger_than_{MAX_INLINE_EVENT_BYTES}_byte_inline_event_limit"
            )
        event_type = str(payload.get("type") or "unknown")
        record["event_type"] = event_type
        item = payload.get("item")
        if not isinstance(item, dict):
            record["active_items_after_event"] = self._pending_summaries_unlocked()
            self._append_record_unlocked(record)
            return

        item_id = str(item.get("id") or "")
        item_type = str(item.get("type") or "unknown")
        item_summary: dict[str, object] = {
            "id": item_id or None,
            "type": item_type,
            "status": item.get("status"),
        }
        if item_type == "command_execution":
            command = item.get("command")
            output = item.get("aggregated_output")
            item_summary["command"] = command
            item_summary["exit_code"] = item.get("exit_code")
            if isinstance(output, str):
                output_bytes = output.encode("utf-8")
                item_summary["aggregated_output"] = {
                    "raw_event_field": "item.aggregated_output",
                    "size_bytes": len(output_bytes),
                    "sha256": _sha256_bytes(output_bytes),
                }
        elif item_type == "agent_message":
            message = item.get("text")
            if isinstance(message, str) and len(message.encode("utf-8")) <= MAX_INLINE_EVENT_BYTES:
                item_summary["text"] = message
            elif isinstance(message, str):
                item_summary["text_omitted_reason"] = (
                    f"larger_than_{MAX_INLINE_EVENT_BYTES}_byte_inline_event_limit"
                )
        record["item"] = item_summary

        pending = self._pending.get(item_id) if item_id else None
        started_file_targets = (
            self._capture_reported_targets_unlocked(item, now_utc)
            if self.capture_mode == "streaming"
            and event_type == "item.started"
            and item_type == "file_change"
            else []
        )
        completed_file_traces = (
            self._completed_target_traces_unlocked(item, pending, now_utc)
            if event_type == "item.completed" and item_type == "file_change"
            else []
        )
        if self.capture_mode == "streaming" and event_type == "item.completed":
            workspace_changes = self._workspace_delta_unlocked(now_utc)
            if workspace_changes:
                record["workspace_changes_since_previous_activity"] = workspace_changes
            record["workspace_scan"] = dict(self._last_scan_diagnostics)

        if event_type == "item.started" and item_id:
            started_at_utc: str | None = now_utc if self.capture_mode == "streaming" else None
            started_monotonic = now_monotonic if self.capture_mode == "streaming" else None
            pending = _PendingItem(
                item_id=item_id,
                item_type=item_type,
                command=str(item.get("command")) if item.get("command") is not None else None,
                started_at_utc=started_at_utc,
                started_monotonic=started_monotonic,
                file_targets=started_file_targets,
            )
            self._pending[item_id] = pending
            record["item_timing"] = {
                "source": (
                    "host_stream_observation"
                    if self.capture_mode == "streaming"
                    else "unavailable_buffered_replay"
                ),
                "started_at_utc": started_at_utc,
                "completed_at_utc": None,
                "duration_seconds": None,
            }
            if item_type == "file_change":
                record["file_changes"] = [
                    self._started_target_trace(target) for target in started_file_targets
                ]
        elif event_type == "item.completed":
            timing_source = "unavailable_missing_started_event"
            started_at_utc = None
            duration_seconds: float | None = None
            if self.capture_mode == "buffered_fallback":
                timing_source = "unavailable_buffered_replay"
            elif pending is not None and pending.started_monotonic is not None:
                timing_source = "host_stream_observation"
                started_at_utc = pending.started_at_utc
                duration_seconds = round(
                    max(0.0, now_monotonic - pending.started_monotonic),
                    6,
                )
            record["item_timing"] = {
                "source": timing_source,
                "started_at_utc": started_at_utc,
                "completed_at_utc": now_utc if self.capture_mode == "streaming" else None,
                "duration_seconds": duration_seconds,
            }
            if item_type == "file_change":
                record["file_changes"] = completed_file_traces
            if item_id:
                self._pending.pop(item_id, None)

        record["active_items_after_event"] = self._pending_summaries_unlocked()
        self._append_record_unlocked(record)

    def _new_record_unlocked(
        self,
        record_type: str,
        *,
        now_utc: str,
        now_monotonic: float,
        include_wait: bool = True,
    ) -> dict[str, object]:
        self._sequence += 1
        started = self._process_started_monotonic
        spawned = self._process_spawned_monotonic
        previous = self._last_observed_monotonic
        trace_elapsed = 0.0 if started is None else max(0.0, now_monotonic - started)
        process_elapsed = None if spawned is None else max(0.0, now_monotonic - spawned)
        delta = 0.0 if previous is None else max(0.0, now_monotonic - previous)
        record: dict[str, object] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "sequence": self._sequence,
            "record_type": record_type,
            "capture_mode": self.capture_mode,
            "observed_at_utc": now_utc,
            "trace_elapsed_seconds": round(trace_elapsed, 6),
            "process_elapsed_seconds": (
                round(process_elapsed, 6) if process_elapsed is not None else None
            ),
            "since_previous_observation_seconds": round(delta, 6),
        }
        if include_wait:
            active_items = self._pending_summaries_unlocked()
            record["wait_since_previous_observation"] = {
                "duration_seconds": round(delta, 6),
                "active_items": active_items,
                "activity_when_no_item_active": (
                    None if active_items else "codex_internal_or_unreported_processing"
                ),
                "meaning": "host_observation_gap_not_hidden_reasoning_time",
            }
        self._last_observed_monotonic = now_monotonic
        return record

    def _pending_summaries_unlocked(self) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for pending in self._pending.values():
            summaries.append(
                {
                    "item_id": pending.item_id,
                    "item_type": pending.item_type,
                    "command": pending.command,
                    "waiting_for": "item.completed",
                }
            )
        return summaries

    def _capture_reported_targets_unlocked(
        self,
        item: Mapping[str, object],
        captured_at_utc: str,
    ) -> list[_ReportedFileTarget]:
        raw_changes = item.get("changes")
        if not isinstance(raw_changes, list):
            return []
        targets: list[_ReportedFileTarget] = []
        for raw_change in raw_changes:
            if not isinstance(raw_change, dict):
                continue
            reported = raw_change.get("path")
            reported_path = str(reported) if reported is not None else ""
            reported_kind = (
                str(raw_change.get("kind")) if raw_change.get("kind") is not None else None
            )
            relative_path, capture_status = self._resolve_reported_path(reported_path)
            before = (
                self._capture_relative_file(relative_path, captured_at_utc)
                if relative_path is not None
                else None
            )
            targets.append(
                _ReportedFileTarget(
                    reported_path=reported_path,
                    reported_kind=reported_kind,
                    relative_path=relative_path,
                    before=before,
                    capture_status=capture_status,
                )
            )
        return targets

    def _started_target_trace(self, target: _ReportedFileTarget) -> dict[str, object]:
        return {
            "reported_path": target.reported_path,
            "reported_kind": target.reported_kind,
            "workspace_relative_path": target.relative_path,
            "capture_status": target.capture_status,
            "before": target.before.to_trace() if target.before is not None else None,
            "after": None,
        }

    def _completed_target_traces_unlocked(
        self,
        item: Mapping[str, object],
        pending: _PendingItem | None,
        captured_at_utc: str,
    ) -> list[dict[str, object]]:
        pending_targets = list(pending.file_targets) if pending is not None else []
        raw_changes = item.get("changes")
        completed_changes = raw_changes if isinstance(raw_changes, list) else []
        matched_targets: list[tuple[_ReportedFileTarget, str | None, str | None]] = []
        for raw_change in completed_changes:
            if not isinstance(raw_change, dict):
                continue
            completed_path = (
                str(raw_change.get("path")) if raw_change.get("path") is not None else ""
            )
            completed_kind = (
                str(raw_change.get("kind")) if raw_change.get("kind") is not None else None
            )
            completed_relative, completed_status = self._resolve_reported_path(completed_path)
            match_index: int | None = None
            for index, candidate in enumerate(pending_targets):
                if completed_relative is not None and candidate.relative_path == completed_relative:
                    match_index = index
                    break
                if completed_relative is None and candidate.reported_path == completed_path:
                    match_index = index
                    break
            if match_index is not None:
                target = pending_targets.pop(match_index)
            else:
                if completed_relative is not None:
                    unavailable = (
                        "before_unavailable_buffered_replay"
                        if self.capture_mode == "buffered_fallback"
                        else "before_unavailable_missing_started_event"
                    )
                else:
                    unavailable = completed_status
                target = _ReportedFileTarget(
                    reported_path=completed_path,
                    reported_kind=completed_kind,
                    relative_path=completed_relative,
                    before=None,
                    capture_status=unavailable,
                )
            matched_targets.append((target, completed_path, completed_kind))
        matched_targets.extend((target, None, None) for target in pending_targets)

        traces: list[dict[str, object]] = []
        for target, matched_path, matched_kind in matched_targets:
            after = (
                self._capture_relative_file(target.relative_path, captured_at_utc)
                if target.relative_path is not None
                else None
            )
            trace: dict[str, object] = {
                "reported_path": matched_path or target.reported_path,
                "reported_kind": matched_kind or target.reported_kind,
                "started_reported_path": target.reported_path,
                "completed_reported_path": matched_path,
                "workspace_relative_path": target.relative_path,
                "capture_status": target.capture_status,
                "before": target.before.to_trace() if target.before is not None else None,
                "after": after.to_trace() if after is not None else None,
            }
            diff = self._unified_diff(target.before, after)
            if diff is not None:
                trace["unified_diff"] = diff
            traces.append(trace)
        return traces

    def _resolve_reported_path(self, reported_path: str) -> tuple[str | None, str]:
        if not reported_path:
            return None, "invalid_empty_path"
        candidate = Path(reported_path).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(self._root)
        except ValueError:
            return None, "outside_generation_workspace_not_read"
        return relative.as_posix(), "captured"

    def _workspace_delta_unlocked(self, captured_at_utc: str) -> list[dict[str, object]]:
        current = self._scan_workspace(captured_at_utc)
        if not bool(self._last_scan_diagnostics.get("complete")):
            for relative_path, previous in self._inventory.items():
                current.setdefault(
                    relative_path,
                    replace(previous, captured_at_utc=captured_at_utc),
                )
        changes: list[dict[str, object]] = []
        for relative_path in sorted(set(self._inventory) | set(current)):
            before = self._inventory.get(relative_path)
            after = current.get(relative_path)
            if before is not None and after is not None and before.fingerprint == after.fingerprint:
                continue
            if before is None:
                # Keys come from the union of both inventories, so an added path is present now.
                assert after is not None
                change_kind = "added"
                before = self._missing_state(relative_path, captured_at_utc)
            elif after is None:
                change_kind = "deleted"
                after = self._missing_state(relative_path, captured_at_utc)
            else:
                change_kind = "modified"
            change: dict[str, object] = {
                "path": relative_path,
                "change": change_kind,
                "before": before.to_trace(),
                "after": after.to_trace(),
            }
            diff = self._unified_diff(before, after)
            if diff is not None:
                change["unified_diff"] = diff
            changes.append(change)
        self._inventory = current
        return changes

    def _scan_workspace(self, captured_at_utc: str) -> dict[str, _FileState]:
        inventory: dict[str, _FileState] = {}
        scan_started = time.monotonic()
        examined_files = 0
        truncated_reason: str | None = None
        for directory, dirnames, filenames in os.walk(self._root, topdown=True, followlinks=False):
            if time.monotonic() - scan_started >= MAX_WORKSPACE_SCAN_SECONDS:
                truncated_reason = "time_budget_exceeded"
                break
            current = Path(directory)
            if current == self._root:
                dirnames[:] = [name for name in dirnames if name not in _EXCLUDED_WORKSPACE_DIRS]
            dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
            for filename in sorted(filenames):
                if examined_files >= MAX_WORKSPACE_SCAN_FILES:
                    truncated_reason = "file_count_budget_exceeded"
                    break
                if time.monotonic() - scan_started >= MAX_WORKSPACE_SCAN_SECONDS:
                    truncated_reason = "time_budget_exceeded"
                    break
                examined_files += 1
                path = current / filename
                relative_path = path.relative_to(self._root).as_posix()
                previous = self._inventory.get(relative_path)
                try:
                    metadata = path.lstat()
                except OSError:
                    continue
                if (
                    previous is not None
                    and previous.exists
                    and previous.size_bytes == metadata.st_size
                    and previous.mtime_ns == metadata.st_mtime_ns
                    and (
                        (stat.S_ISREG(metadata.st_mode) and previous.file_type == "regular")
                        or (stat.S_ISLNK(metadata.st_mode) and previous.file_type == "symlink")
                    )
                ):
                    inventory[relative_path] = replace(
                        previous,
                        captured_at_utc=captured_at_utc,
                    )
                    continue
                inventory[relative_path] = self._capture_relative_file(
                    relative_path,
                    captured_at_utc,
                    metadata=metadata,
                )
            if truncated_reason is not None:
                break
        self._last_scan_diagnostics = {
            "complete": truncated_reason is None,
            "examined_files": examined_files,
            "max_files": MAX_WORKSPACE_SCAN_FILES,
            "max_seconds": MAX_WORKSPACE_SCAN_SECONDS,
            "truncated_reason": truncated_reason,
            "elapsed_seconds": round(max(0.0, time.monotonic() - scan_started), 6),
        }
        return inventory

    def _capture_relative_file(
        self,
        relative_path: str,
        captured_at_utc: str,
        *,
        metadata: os.stat_result | None = None,
    ) -> _FileState:
        path = self._root / relative_path
        try:
            initial = metadata or path.lstat()
        except FileNotFoundError:
            return self._missing_state(relative_path, captured_at_utc)
        except OSError as exc:
            return _FileState(
                path=relative_path,
                captured_at_utc=captured_at_utc,
                exists=True,
                file_type="unreadable",
                content_omitted_reason=f"{type(exc).__name__}: {exc}",
            )

        if stat.S_ISLNK(initial.st_mode):
            try:
                target = os.readlink(path)
            except OSError as exc:
                target = f"unreadable: {type(exc).__name__}: {exc}"
            return _FileState(
                path=relative_path,
                captured_at_utc=captured_at_utc,
                exists=True,
                file_type="symlink",
                size_bytes=initial.st_size,
                mtime_ns=initial.st_mtime_ns,
                symlink_target=target,
            )
        if not stat.S_ISREG(initial.st_mode):
            return _FileState(
                path=relative_path,
                captured_at_utc=captured_at_utc,
                exists=True,
                file_type="other",
                size_bytes=initial.st_size,
                mtime_ns=initial.st_mtime_ns,
                content_omitted_reason="not_a_regular_file",
            )

        file_descriptor: int | None = None
        try:
            file_descriptor = self._open_relative_file_no_follow(relative_path)
            opened = os.fstat(file_descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise OSError("opened workspace path is not a regular file")
            if opened.st_size > MAX_INLINE_TEXT_BYTES:
                return _FileState(
                    path=relative_path,
                    captured_at_utc=captured_at_utc,
                    exists=True,
                    file_type="regular",
                    size_bytes=opened.st_size,
                    mtime_ns=opened.st_mtime_ns,
                    content_omitted_reason=f"larger_than_{MAX_INLINE_TEXT_BYTES}_byte_capture_limit",
                )

            hasher = hashlib.sha256()
            content = bytearray()
            exceeded_limit = False
            with os.fdopen(file_descriptor, "rb", closefd=True) as stream:
                file_descriptor = None
                while chunk := stream.read(
                    min(1024 * 1024, MAX_INLINE_TEXT_BYTES + 1 - len(content))
                ):
                    hasher.update(chunk)
                    content.extend(chunk)
                    if len(content) > MAX_INLINE_TEXT_BYTES:
                        exceeded_limit = True
                        break
                final = os.fstat(stream.fileno())
        except OSError as exc:
            return _FileState(
                path=relative_path,
                captured_at_utc=captured_at_utc,
                exists=True,
                file_type="unreadable",
                size_bytes=initial.st_size,
                mtime_ns=initial.st_mtime_ns,
                content_omitted_reason=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if file_descriptor is not None:
                with suppress(OSError):
                    os.close(file_descriptor)

        stable = (
            initial.st_dev == opened.st_dev
            and initial.st_ino == opened.st_ino
            and opened.st_size == final.st_size
            and opened.st_mtime_ns == final.st_mtime_ns
        )
        text: str | None = None
        omitted_reason: str | None = None
        sha256: str | None = hasher.hexdigest()
        if exceeded_limit:
            content = bytearray()
            sha256 = None
            omitted_reason = f"grew_beyond_{MAX_INLINE_TEXT_BYTES}_byte_capture_limit"
        elif b"\x00" in content:
            omitted_reason = "binary_content"
        else:
            try:
                text = bytes(content).decode("utf-8")
            except UnicodeDecodeError:
                omitted_reason = "non_utf8_content"
        return _FileState(
            path=relative_path,
            captured_at_utc=captured_at_utc,
            exists=True,
            file_type="regular",
            size_bytes=final.st_size,
            mtime_ns=final.st_mtime_ns,
            sha256=sha256,
            text=text,
            content_omitted_reason=omitted_reason,
            capture_stable=stable,
        )

    def _open_relative_file_no_follow(self, relative_path: str) -> int:
        parts = Path(relative_path).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise OSError("invalid workspace-relative path")
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        close_on_exec = getattr(os, "O_CLOEXEC", 0)
        directory_fd = os.open(self._root, directory_flags | close_on_exec)
        try:
            for part in parts[:-1]:
                next_fd = os.open(
                    part,
                    directory_flags | no_follow | close_on_exec,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd
            return os.open(
                parts[-1],
                os.O_RDONLY | no_follow | close_on_exec,
                dir_fd=directory_fd,
            )
        finally:
            with suppress(OSError):
                os.close(directory_fd)

    @staticmethod
    def _missing_state(relative_path: str, captured_at_utc: str) -> _FileState:
        return _FileState(
            path=relative_path,
            captured_at_utc=captured_at_utc,
            exists=False,
            file_type="missing",
        )

    @staticmethod
    def _unified_diff(before: _FileState | None, after: _FileState | None) -> str | None:
        if before is None or after is None:
            return None
        before_text = "" if not before.exists else before.text
        after_text = "" if not after.exists else after.text
        if before_text is None or after_text is None:
            return None
        return "".join(
            difflib.unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=f"{before.path}@before",
                tofile=f"{after.path}@after",
            )
        )

    def _append_record_unlocked(self, record: Mapping[str, object]) -> None:
        serialized = json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.workspace.codex_timeline_path.open("a", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()

    def _report_error(self, stage: str, exc: Exception) -> None:
        try:
            payload = {
                "observed_at_utc": _format_utc(self._wall_clock()),
                "stage": stage,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._error_lock:
                with self.workspace.codex_trace_errors_path.open("a", encoding="utf-8") as stream:
                    stream.write(serialized)
                    stream.flush()
        except Exception:
            return


__all__ = [
    "MAX_INLINE_EVENT_BYTES",
    "MAX_INLINE_TEXT_BYTES",
    "TRACE_SCHEMA_VERSION",
    "CodexTraceRecorder",
    "TraceCaptureMode",
]
