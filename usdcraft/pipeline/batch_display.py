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
from dataclasses import dataclass, field
from datetime import datetime
from typing import TextIO

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from usdcraft.pipeline.batch_progress import ProviderBatchProgressEvent
from usdcraft.pipeline.display_common import (
    PROVIDER_SUCCESS_STATUSES as _SUCCESS_STATUSES,
)
from usdcraft.pipeline.display_common import (
    environment_flag as _environment_flag,
)
from usdcraft.pipeline.display_common import (
    quoted as _quoted,
)
from usdcraft.pipeline.display_common import (
    read_progress_message as _read_progress_message,
)
from usdcraft.pipeline.display_common import (
    safe_provider_message as _safe_provider_message,
)
from usdcraft.pipeline.display_common import (
    single_line as _single_line,
)
from usdcraft.pipeline.display_common import (
    timestamp as _timestamp,
)

_SENTINEL = object()


@dataclass(slots=True)
class _SlotState:
    slot: int
    row_index: int | None = None
    row_id: str | None = None
    status: str = "idle"
    message: str = 'Provider Generation | activity="waiting for row"'
    message_timestamp: datetime | None = None
    started_at: datetime | None = None
    # The row's current Host stage and Author turn, read from its own progress lines so the slot
    # title shows where the row is even when its latest line is an older one.
    stage: str | None = None
    turn: int | None = None
    max_turns: int | None = None

    def apply_message(self, message: str) -> None:
        reading = _read_progress_message(message)
        if reading.activity == "stage started" and reading.stage:
            self.stage = reading.stage
        elif reading.activity == "stage finished" and reading.stage and self.stage == reading.stage:
            self.stage = None
        if reading.turn is not None:
            self.turn = reading.turn
        if reading.max_turns is not None:
            self.max_turns = reading.max_turns

    def reset_for_row(self, event: ProviderBatchProgressEvent, *, label: str) -> None:
        self.row_index = event.row_index
        self.row_id = event.row_id
        self.status = event.status or "running"
        self.message = (
            _safe_provider_message(event.message, label=label)
            if event.message
            else f'{label} | activity="row started"'
        )
        self.message_timestamp = event.timestamp
        self.started_at = event.timestamp
        self.stage = None
        self.turn = None
        self.max_turns = None


@dataclass(slots=True)
class _BatchState:
    label: str = "Provider Generation"
    total: int = 0
    effective_concurrency: int = 0
    model: str = ""
    reasoning_effort: str = ""
    started_at: datetime | None = None
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    slots: dict[int, _SlotState] = field(default_factory=dict)
    finished_rows: set[tuple[int | None, str | None]] = field(default_factory=set)

    def apply(self, event: ProviderBatchProgressEvent) -> None:
        if event.kind == "batch_started":
            self.total = max(event.total or 0, 0)
            self.effective_concurrency = max(event.effective_concurrency or 0, 0)
            self.model = _single_line(event.model)
            self.reasoning_effort = _single_line(event.reasoning_effort)
            self.started_at = event.timestamp
            self.slots = {
                slot: _SlotState(slot=slot, message=self._idle_message())
                for slot in range(1, self.effective_concurrency + 1)
            }
            return

        if event.kind in {"row_started", "row_progress", "row_finished"}:
            if event.slot is None:
                return
            slot = self.slots.setdefault(
                event.slot, _SlotState(slot=event.slot, message=self._idle_message())
            )
            if event.kind == "row_started":
                slot.reset_for_row(event, label=self.label)
                return
            if event.kind == "row_progress":
                slot.row_index = event.row_index if event.row_index is not None else slot.row_index
                slot.row_id = event.row_id if event.row_id is not None else slot.row_id
                slot.status = "running"
                slot.message = _safe_provider_message(event.message, label=self.label)
                slot.message_timestamp = event.timestamp
                slot.apply_message(slot.message)
                return

            slot.row_index = event.row_index if event.row_index is not None else slot.row_index
            slot.row_id = event.row_id if event.row_id is not None else slot.row_id
            slot.status = event.status or "failed"
            slot.message = _finished_message(event, label=self.label)
            slot.message_timestamp = event.timestamp
            row_key = (event.row_index, event.row_id)
            if row_key not in self.finished_rows:
                self.finished_rows.add(row_key)
                self.completed += 1
                if slot.status in _SUCCESS_STATUSES:
                    self.succeeded += 1
                else:
                    self.failed += 1
            return

        if event.kind == "batch_finished":
            if event.succeeded is not None:
                self.succeeded = max(event.succeeded, 0)
            if event.failed is not None:
                self.failed = max(event.failed, 0)
            self.completed = min(self.succeeded + self.failed, self.total)

    def _idle_message(self) -> str:
        return f'{self.label} | activity="waiting for row"'


def _finished_message(event: ProviderBatchProgressEvent, *, label: str) -> str:
    if event.message:
        return _safe_provider_message(event.message, label=label)
    fields = [label, 'activity="completed"', f"status={event.status or 'failed'}"]
    if event.record_id:
        fields.append(f"record_id={_quoted(event.record_id)}")
    if event.error_type:
        fields.append(f"error_type={_quoted(event.error_type)}")
    return " | ".join(fields)


class ProviderBatchDisplay:
    """Fail-open Batch progress sink with one renderer thread owning all terminal writes.

    Calls from worker threads only enqueue immutable events. Use the display as a context manager
    so its renderer is stopped and a Rich terminal is restored before the final CLI result prints.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        force_plain: bool = False,
        terminal_size: tuple[int, int] | os.terminal_size | None = None,
        refresh_hz: float = 10.0,
        label: str = "Provider Generation",
    ) -> None:
        self._stream = sys.stderr if stream is None else stream
        self._force_plain = force_plain
        self._terminal_size = terminal_size
        self._refresh_hz = max(float(refresh_hz), 1.0)
        self._events: queue.Queue[ProviderBatchProgressEvent | object] = queue.Queue()
        self._state = _BatchState(label=str(label).strip() or "Provider Generation")
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._disabled = False

    def __enter__(self) -> ProviderBatchDisplay:
        self._ensure_started()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def __call__(self, event: ProviderBatchProgressEvent) -> None:
        """Enqueue a presentation event without writing from the caller's thread."""

        try:
            self._ensure_started()
            with self._lifecycle_lock:
                if self._closed or self._disabled or self._thread is None:
                    return
                self._events.put_nowait(event)
        except Exception:
            # The display is presentation-only and must not become a Batch execution gate.
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
                        self._events.put_nowait(_SENTINEL)
                    except Exception:
                        pass
            if thread is not None:
                try:
                    thread.join(timeout=5.0)
                except Exception:
                    pass
        except Exception:
            # Display shutdown is presentation-only and must remain idempotent/fail-open.
            pass

    def _ensure_started(self) -> None:
        try:
            with self._lifecycle_lock:
                if self._closed or self._disabled or self._thread is not None:
                    return
                thread = threading.Thread(
                    target=self._render_loop,
                    name="usdcraft-provider-batch-display",
                    daemon=True,
                )
                try:
                    thread.start()
                except Exception:
                    self._disabled = True
                    return
                # Store only a successfully started thread so close never joins an unstarted one.
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
        last_plain_progress: dict[tuple[int | None, int | None, str | None], str] = {}
        refresh_seconds = 1.0 / self._refresh_hz
        try:
            while True:
                try:
                    queued = self._events.get(timeout=refresh_seconds)
                except queue.Empty:
                    if live is not None:
                        live.update(self._render_live(), refresh=True)
                    continue
                if queued is _SENTINEL:
                    break
                if not isinstance(queued, ProviderBatchProgressEvent):
                    continue
                event = queued
                self._state.apply(event)
                if mode is None:
                    mode = "live" if self._can_render_live() else "plain"
                    if mode == "live":
                        live = self._make_live()
                        live.start(refresh=False)
                if mode == "plain":
                    line = self._plain_line(event, last_plain_progress)
                    if line is not None:
                        self._stream.write(line + "\n")
                        self._stream.flush()
                elif live is not None:
                    live.update(self._render_live(), refresh=True)
        except Exception:
            # Output failures stay outside scheduling, authoring, validation, and promotion.
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
            or _environment_flag("CI")
            or os.environ.get("TERM", "").strip().lower() == "dumb"
        ):
            return False
        try:
            if not self._stream.isatty():
                return False
        except (AttributeError, OSError):
            return False
        columns, lines = self._dimensions()
        header_lines = max((len(self._header_text()) + max(columns, 1) - 1) // max(columns, 1), 1)
        required_lines = 2 + header_lines + (2 * self._state.effective_concurrency)
        return self._state.effective_concurrency > 0 and lines >= required_lines

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
        header = Text(
            self._header_text(),
            overflow="fold",
        )
        started_at = state.started_at or datetime.now().astimezone()

        global_row = Table.grid(expand=True, padding=(0, 1))
        global_row.add_column(no_wrap=True)
        global_row.add_column(ratio=1)
        active = sum(slot.status == "running" for slot in state.slots.values())
        queued = max(state.total - state.completed - active, 0)
        global_row.add_row(
            Text(
                f"Batch {state.completed}/{state.total} | active={active} | queued={queued} | "
                f"success={state.succeeded} | failed={state.failed}",
                no_wrap=True,
            ),
            ProgressBar(
                total=max(state.total, 1),
                completed=min(state.completed, max(state.total, 1)),
                width=None,
                complete_style="green",
                finished_style="green",
            ),
        )

        slot_rows = Table.grid(expand=True, padding=(0, 1))
        slot_rows.add_column(no_wrap=True)
        slot_rows.add_column(ratio=1)
        now = datetime.now().astimezone()
        for slot_number in range(1, state.effective_concurrency + 1):
            slot = state.slots[slot_number]
            slot_rows.add_row(self._slot_label(slot, now), self._slot_bar(slot))
            message_at = slot.message_timestamp or started_at
            slot_rows.add_row(
                Text(
                    f"  [{_timestamp(message_at)}] {slot.message}",
                    overflow="ellipsis",
                    no_wrap=True,
                ),
                Text(""),
            )
        return Group(header, global_row, slot_rows)

    def _header_text(self) -> str:
        state = self._state
        started_at = state.started_at or datetime.now().astimezone()
        return (
            f'[{_timestamp(started_at)}] {state.label} Batch | activity="starting" | '
            f"total={state.total} | concurrency={state.effective_concurrency} | "
            f"model={_quoted(state.model)} | effort={_quoted(state.reasoning_effort)}"
        )

    def _slot_label(self, slot: _SlotState, now: datetime) -> Text:
        if slot.row_index is None:
            return Text(f"Slot {slot.slot} | idle", style="dim", no_wrap=True)
        elapsed = "00:00:00"
        if slot.started_at is not None:
            try:
                seconds = max(int((now - slot.started_at).total_seconds()), 0)
            except (TypeError, ValueError):
                seconds = 0
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        style = (
            "green"
            if slot.status in _SUCCESS_STATUSES
            else "red"
            if slot.status != "running"
            else ""
        )
        fields = [
            f"Slot {slot.slot}",
            f"row {slot.row_index}/{self._state.total}",
            _single_line(slot.row_id),
            slot.status.upper(),
        ]
        if slot.status == "running":
            if slot.stage is not None:
                fields.append(f"stage={slot.stage}")
            if slot.turn is not None:
                turn = str(slot.turn)
                if slot.max_turns is not None:
                    turn += f"/{slot.max_turns}"
                fields.append(f"turn={turn}")
        fields.append(f"elapsed={elapsed}")
        return Text(" | ".join(fields), style=style, overflow="ellipsis", no_wrap=True)

    @staticmethod
    def _slot_bar(slot: _SlotState) -> ProgressBar:
        if slot.status == "running":
            return ProgressBar(total=None, pulse=True, width=None)
        if slot.status in _SUCCESS_STATUSES:
            return ProgressBar(
                total=1,
                completed=1,
                width=None,
                complete_style="green",
                finished_style="green",
            )
        if slot.status == "idle":
            return ProgressBar(total=1, completed=0, width=None, style="grey23")
        return ProgressBar(
            total=1,
            completed=1,
            width=None,
            complete_style="red",
            finished_style="red",
        )

    def _plain_line(
        self,
        event: ProviderBatchProgressEvent,
        last_progress: dict[tuple[int | None, int | None, str | None], str],
    ) -> str | None:
        prefix = f"[{_timestamp(event.timestamp)}]"
        if event.kind == "batch_started":
            return (
                f'{prefix} {self._state.label} Batch | activity="starting" | total={event.total or 0} | '
                f"concurrency={event.effective_concurrency or 0} | model={_quoted(event.model)} | "
                f"effort={_quoted(event.reasoning_effort)}"
            )
        if event.kind == "row_started":
            return (
                f'{prefix} {self._state.label} Batch | activity="row started" | slot={event.slot} | '
                f"row={event.row_index}/{self._state.total} | row_id={_quoted(event.row_id)}"
            )
        if event.kind == "row_progress":
            message = _safe_provider_message(event.message, label=self._state.label)
            key = (event.slot, event.row_index, event.row_id)
            if last_progress.get(key) == message:
                return None
            last_progress[key] = message
            return (
                f"{prefix} {message} | slot={event.slot} | row={event.row_index}/"
                f"{self._state.total} | row_id={_quoted(event.row_id)}"
            )
        if event.kind == "row_finished":
            fields = [
                f'{prefix} {self._state.label} Batch | activity="row finished"',
                f"slot={event.slot}",
                f"row={event.row_index}/{self._state.total}",
                f"row_id={_quoted(event.row_id)}",
                f"status={_single_line(event.status or 'failed')}",
            ]
            if event.record_id:
                fields.append(f"record_id={_quoted(event.record_id)}")
            if event.error_type:
                fields.append(f"error_type={_quoted(event.error_type)}")
            return " | ".join(fields)
        if event.kind == "batch_finished":
            succeeded = self._state.succeeded
            failed = self._state.failed
            return (
                f'{prefix} {self._state.label} Batch | activity="completed" | total={self._state.total} | '
                f"succeeded={succeeded} | failed={failed}"
            )
        return None


__all__ = ["ProviderBatchDisplay"]
