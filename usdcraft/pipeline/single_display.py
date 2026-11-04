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

import os
import queue
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TextIO

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from usdcraft.pipeline.display_common import (
    PROVIDER_SUCCESS_STATUSES,
    decoded_provider_field,
    environment_flag,
    provider_message_fields,
    quoted,
    read_progress_message,
    safe_provider_message,
    timestamp,
)

_SENTINEL = object()
# Live mode keeps this many recent progress lines under the status bar; plain mode prints every
# line, so the history exists only for the TTY block.
LIVE_HISTORY_LINES = 8


@dataclass(slots=True, frozen=True)
class _ProgressObservation:
    timestamp: datetime
    message: str


@dataclass(slots=True)
class _SingleState:
    started_at: datetime | None = None
    model: str = ""
    reasoning_effort: str = ""
    max_turns: int | None = None
    turn: int | None = None
    status: str = "running"
    stage: str | None = None
    latest_message: str = 'Provider Generation | activity="waiting for progress"'
    latest_at: datetime | None = None
    history: deque[tuple[datetime, str]] = field(
        default_factory=lambda: deque(maxlen=LIVE_HISTORY_LINES)
    )

    def apply(self, observation: _ProgressObservation) -> None:
        safe_message = safe_provider_message(observation.message)
        reading = read_progress_message(safe_message)
        activity = reading.activity
        if activity == "stage started" and reading.stage:
            self.stage = reading.stage
        elif activity == "stage finished" and reading.stage and self.stage == reading.stage:
            self.stage = None

        if self.started_at is None:
            self.started_at = observation.timestamp
        if activity == "starting":
            if reading.model:
                self.model = reading.model
            if reading.effort:
                self.reasoning_effort = reading.effort
        if reading.turn is not None:
            self.turn = reading.turn
        if reading.max_turns is not None:
            self.max_turns = reading.max_turns

        # The harness reports `agent completed` before the final Host compile and promotion. Only
        # the outer primitive's terminal lifecycle events may complete this presentation block.
        if activity == "completed" and reading.status in PROVIDER_SUCCESS_STATUSES:
            self.status = "success"
        elif activity == "failed":
            self.status = "failed"
        else:
            self.status = "running"
        self.latest_message = safe_message
        self.latest_at = observation.timestamp
        self.history.append((observation.timestamp, safe_message))


class ProviderSingleDisplay:
    """Fail-open single-item Provider display backed by one terminal-writer thread."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        force_plain: bool = False,
        terminal_size: tuple[int, int] | os.terminal_size | None = None,
        refresh_hz: float = 10.0,
        label: str = "Provider Generation",
    ) -> None:
        self._label = str(label).strip() or "Provider Generation"
        self._stream = sys.stderr if stream is None else stream
        self._force_plain = force_plain
        self._terminal_size = terminal_size
        self._refresh_hz = max(float(refresh_hz), 1.0)
        self._observations: queue.Queue[_ProgressObservation | object] = queue.Queue()
        self._state = _SingleState(
            latest_message=f'{self._label} | activity="waiting for progress"'
        )
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._disabled = False

    def __enter__(self) -> ProviderSingleDisplay:
        self._ensure_started()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def __call__(self, message: str) -> None:
        """Queue one existing Provider progress string without writing in the caller's thread."""

        try:
            self._ensure_started()
            observation = _ProgressObservation(
                timestamp=datetime.now().astimezone(),
                message=safe_provider_message(message),
            )
            with self._lifecycle_lock:
                if self._closed or self._disabled or self._thread is None:
                    return
                self._observations.put_nowait(observation)
        except Exception:
            # Presentation cannot become Generation, validation, or promotion authority.
            pass

    def close(self) -> None:
        thread: threading.Thread | None = None
        try:
            with self._lifecycle_lock:
                if self._closed:
                    return
                self._closed = True
                thread = self._thread
                if thread is not None:
                    try:
                        self._observations.put_nowait(_SENTINEL)
                    except Exception:
                        pass
            if thread is not None:
                try:
                    thread.join(timeout=5.0)
                except Exception:
                    pass
        except Exception:
            # Closing a display is also presentation-only and must remain idempotent/fail-open.
            pass

    def _ensure_started(self) -> None:
        try:
            with self._lifecycle_lock:
                if self._closed or self._disabled or self._thread is not None:
                    return
                thread = threading.Thread(
                    target=self._render_loop,
                    name="usdcraft-provider-single-display",
                    daemon=True,
                )
                try:
                    thread.start()
                except Exception:
                    self._disabled = True
                    return
                # Store only a thread whose `start()` succeeded. This keeps close/join fail-open
                # when the runtime cannot allocate a renderer thread.
                self._thread = thread
        except Exception:
            try:
                with self._lifecycle_lock:
                    self._disabled = True
            except Exception:
                pass

    def _render_loop(self) -> None:
        live: Live | None = None
        mode: str | None = None
        last_plain_message: str | None = None
        header_written = False
        refresh_seconds = 1.0 / self._refresh_hz
        try:
            while True:
                try:
                    queued = self._observations.get(timeout=refresh_seconds)
                except queue.Empty:
                    if live is not None:
                        live.update(self._render_live(), refresh=True)
                    continue
                if queued is _SENTINEL:
                    break
                if not isinstance(queued, _ProgressObservation):
                    continue
                observation = queued
                self._state.apply(observation)
                if mode is None:
                    mode = "live" if self._can_render_live() else "plain"
                    if mode == "live":
                        live = self._make_live()
                        live.start(refresh=False)
                if mode == "plain":
                    line, header_written = self._plain_line(
                        observation,
                        header_written=header_written,
                        last_message=last_plain_message,
                    )
                    if line is not None:
                        self._stream.write(line + "\n")
                        self._stream.flush()
                        last_plain_message = safe_provider_message(observation.message)
                elif live is not None:
                    live.update(self._render_live(), refresh=True)
        except Exception:
            # Output failures stay outside authoring, validation, and promotion.
            pass
        finally:
            if live is not None:
                try:
                    live.stop()
                except Exception:
                    pass
            try:
                with self._lifecycle_lock:
                    self._disabled = True
            except Exception:
                pass

    def _dimensions(self) -> tuple[int, int]:
        if self._terminal_size is not None:
            return (int(self._terminal_size[0]), int(self._terminal_size[1]))
        try:
            size = os.get_terminal_size(self._stream.fileno())
        except (AttributeError, OSError, ValueError):
            size = os.terminal_size((120, 40))
        return (size.columns, size.lines)

    def _can_render_live(self) -> bool:
        if (
            self._force_plain
            or environment_flag("CI")
            or os.environ.get("TERM", "").strip().lower() == "dumb"
        ):
            return False
        try:
            if not self._stream.isatty():
                return False
        except (AttributeError, OSError):
            return False
        columns, lines = self._dimensions()
        width = max(columns, 1)
        header_lines = max((len(self._header_text()) + width - 1) // width, 1)
        # Header, status row and at least one history line.
        return lines >= header_lines + 2

    def _history_capacity(self) -> int:
        columns, lines = self._dimensions()
        width = max(columns, 1)
        header_lines = max((len(self._header_text()) + width - 1) // width, 1)
        return max(min(LIVE_HISTORY_LINES, lines - header_lines - 1), 1)

    def _make_live(self) -> Live:
        columns, lines = self._dimensions()
        console = Console(
            file=self._stream,
            force_terminal=True,
            width=max(columns, 20),
            height=max(lines, 1),
        )
        return Live(
            self._render_live(),
            console=console,
            auto_refresh=False,
            refresh_per_second=self._refresh_hz,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
            vertical_overflow="ellipsis",
        )

    def _render_live(self) -> RenderableType:
        state = self._state
        header = Text(self._header_text(), overflow="fold")
        row = Table.grid(expand=True, padding=(0, 1))
        row.add_column(no_wrap=True)
        row.add_column(ratio=1)
        row.add_row(self._status_label(), self._status_bar())
        history = list(state.history)[-self._history_capacity() :]
        if not history:
            message_at = state.latest_at or state.started_at or datetime.now().astimezone()
            history = [(message_at, state.latest_message)]
        for message_at, message in history:
            row.add_row(
                Text(
                    f"  [{timestamp(message_at)}] {message}",
                    overflow="ellipsis",
                    no_wrap=True,
                ),
                Text(""),
            )
        return Group(header, row)

    def _header_text(self) -> str:
        state = self._state
        started_at = state.started_at or datetime.now().astimezone()
        fields = [
            f'[{timestamp(started_at)}] {self._label} | activity="starting"',
            f"model={quoted(state.model)}",
            f"effort={quoted(state.reasoning_effort)}",
        ]
        if state.max_turns is not None:
            fields.append(f"max_turns={state.max_turns}")
        return " | ".join(fields)

    def _status_label(self) -> Text:
        state = self._state
        started_at = state.started_at or datetime.now().astimezone()
        try:
            seconds = max(int((datetime.now().astimezone() - started_at).total_seconds()), 0)
        except (TypeError, ValueError):
            seconds = 0
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        fields = [self._label, state.status.upper()]
        if state.stage is not None:
            fields.append(f"stage={state.stage}")
        if state.turn is not None:
            turn = str(state.turn)
            if state.max_turns is not None:
                turn += f"/{state.max_turns}"
            fields.append(f"turn={turn}")
        fields.append(f"elapsed={hours:02d}:{minutes:02d}:{seconds:02d}")
        style = "green" if state.status == "success" else "red" if state.status == "failed" else ""
        return Text(" | ".join(fields), style=style, overflow="ellipsis", no_wrap=True)

    def _status_bar(self) -> ProgressBar:
        if self._state.status == "running":
            return ProgressBar(total=None, pulse=True, width=None)
        style = "green" if self._state.status == "success" else "red"
        return ProgressBar(
            total=1,
            completed=1,
            width=None,
            complete_style=style,
            finished_style=style,
        )

    def _plain_line(
        self,
        observation: _ProgressObservation,
        *,
        header_written: bool,
        last_message: str | None,
    ) -> tuple[str | None, bool]:
        safe_message = safe_provider_message(observation.message)
        activity = decoded_provider_field(provider_message_fields(safe_message).get("activity"))
        if not header_written and activity.casefold() == "starting":
            return self._header_text(), True
        if safe_message == last_message:
            return None, header_written
        return f"[{timestamp(observation.timestamp)}] {safe_message}", header_written


__all__ = ["LIVE_HISTORY_LINES", "ProviderSingleDisplay"]
