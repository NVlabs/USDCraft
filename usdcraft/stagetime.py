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

"""Shared per-run pipeline stage timing for Generation and Reconstruction.

One `StageTimer` instance times the coarse Host-visible stages of a run — freeze, subject
isolation, mesh preprocess, image observation, author, visual review, host gate, persist — and
reports each stage three ways: a log line at stage start and finish, an append-only JSONL trace
event stream, and an ordered in-memory report suitable for embedding in a persisted runtime report.
The log lines use the Provider progress grammar (``<label> | activity="stage started" |
stage=<name>``), so the Codex arms print them to stderr directly and the Provider arms route them
into the same progress display as the harness. User preflight and confirmation are deliberately
never timed. Timing is descriptive run metadata; it is never a gate and a trace write failure never
fails the run.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

STAGE_TRACE_FILENAME = "stage_trace.jsonl"


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _stderr_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class StageTimer:
    """Time named pipeline stages with console logs, a JSONL trace, and an ordered report."""

    def __init__(
        self,
        *,
        pipeline: str,
        log: Callable[[str], None] | None = _stderr_log,
        log_label: str | None = None,
        trace_path: Path | None = None,
    ) -> None:
        self._pipeline = str(pipeline)
        # `pipeline` names the run in the trace and report; `log_label` is only the prefix of the
        # log lines, so a Provider arm can use its progress label without renaming its records.
        self._log_label = str(log_label) if log_label else self._pipeline
        self._log = log if log is not None else (lambda _message: None)
        self._trace_path: Path | None = None
        self._buffered_events: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []
        if trace_path is not None:
            self.attach_trace_path(trace_path)

    def attach_trace_path(self, trace_path: Path) -> None:
        """Bind the JSONL trace file and flush any events recorded before the path existed."""

        self._trace_path = Path(trace_path)
        buffered, self._buffered_events = self._buffered_events, []
        for event in buffered:
            self._write_trace_event(event)

    def _write_trace_event(self, event: dict[str, Any]) -> None:
        if self._trace_path is None:
            self._buffered_events.append(event)
            return
        # The stage trace is a diagnostic sidecar: losing one event must not fail the run.
        with suppress(OSError):
            with self._trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")

    @contextmanager
    def stage(self, name: str, *, detail: str | None = None) -> Iterator[None]:
        """Time one stage; a raised exception marks the stage failed and propagates unchanged."""

        stage_name = str(name)
        suffix = f" | detail={json.dumps(detail, ensure_ascii=False)}" if detail else ""
        started_at = _utc_now_text()
        started = time.monotonic()
        self._log(f'{self._log_label} | activity="stage started" | stage={stage_name}{suffix}')
        event: dict[str, Any] = {
            "event": "stage.started",
            "pipeline": self._pipeline,
            "stage": stage_name,
            "at_utc": started_at,
        }
        if detail:
            event["detail"] = detail
        self._write_trace_event(event)
        status = "ok"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            duration = round(time.monotonic() - started, 3)
            record: dict[str, Any] = {
                "stage": stage_name,
                "started_at_utc": started_at,
                "duration_seconds": duration,
                "status": status,
            }
            if detail:
                record["detail"] = detail
            self._records.append(record)
            self._log(
                f'{self._log_label} | activity="stage finished" | stage={stage_name} | '
                f"status={status} | duration={duration:.2f}s{suffix}"
            )
            finished: dict[str, Any] = {
                "event": "stage.finished",
                "pipeline": self._pipeline,
                "stage": stage_name,
                "at_utc": _utc_now_text(),
                "duration_seconds": duration,
                "status": status,
            }
            if detail:
                finished["detail"] = detail
            self._write_trace_event(finished)

    def report(self) -> list[dict[str, Any]]:
        """Return the ordered stage records for embedding in a persisted runtime report."""

        return [dict(record) for record in self._records]


__all__ = ["STAGE_TRACE_FILENAME", "StageTimer"]
