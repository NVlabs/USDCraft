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

import io
import re
import threading
from datetime import datetime, timedelta, timezone

import pytest

from usdcraft.pipeline.batch_display import ProviderBatchDisplay
from usdcraft.pipeline.batch_progress import (
    ProviderBatchProgressEvent,
    emit_provider_batch_progress,
)

_NOW = datetime(2026, 8, 4, 0, 12, tzinfo=timezone(timedelta(hours=8)))
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        raise OSError("test stream has no file descriptor")


def _event(kind: str, **kwargs: object) -> ProviderBatchProgressEvent:
    return ProviderBatchProgressEvent(kind=kind, timestamp=_NOW, **kwargs)  # type: ignore[arg-type]


def test_progress_event_is_aware_and_callback_failure_is_fail_open() -> None:
    event = _event(
        "batch_started",
        total=2,
        effective_concurrency=2,
        model="nvidia-model",
        reasoning_effort="high",
    )

    assert event.timestamp.utcoffset() == timedelta(hours=8)

    def fail(_event: ProviderBatchProgressEvent) -> None:
        raise RuntimeError("presentation broke")

    emit_provider_batch_progress(fail, event)
    emit_provider_batch_progress(None, event)


def test_plain_display_logs_state_deduplicates_progress_and_omits_sensitive_fields() -> None:
    stream = io.StringIO()
    sensitive_message = (
        'Provider Generation | activity="running final Host compile" | '
        'prompt="do not print me" | image="/secret/reference.png" | '
        'arguments="tool secret" | result="tool result" | error="provider body"'
    )

    with ProviderBatchDisplay(stream=stream, force_plain=True) as display:
        display(
            _event(
                "batch_started",
                total=2,
                effective_concurrency=2,
                model="azure/openai/test-model-b",
                reasoning_effort="high",
            )
        )
        display(_event("row_started", slot=1, row_index=1, row_id="swing"))
        display(
            _event(
                "row_progress",
                slot=1,
                row_index=1,
                row_id="swing",
                message=sensitive_message,
            )
        )
        display(
            _event(
                "row_progress",
                slot=1,
                row_index=1,
                row_id="swing",
                message=sensitive_message,
            )
        )
        display(
            _event(
                "row_finished",
                slot=1,
                row_index=1,
                row_id="swing",
                status="generation_ready",
                record_id="rec_swing",
            )
        )
        display(_event("row_started", slot=2, row_index=2, row_id="lamp"))
        display(
            _event(
                "row_finished",
                slot=2,
                row_index=2,
                row_id="lamp",
                status="failed",
                error_type="RuntimeError",
            )
        )
        display(_event("batch_finished", total=2, succeeded=1, failed=1))

    output = stream.getvalue()
    assert (
        '[2026-08-04T00:12:00+08:00] Provider Generation Batch | activity="starting" | total=2 | '
        'concurrency=2 | model="azure/openai/test-model-b" | effort="high"'
    ) in output
    assert output.count('activity="running final Host compile"') == 1
    assert 'row=1/2 | row_id="swing" | status=generation_ready' in output
    assert 'record_id="rec_swing"' in output
    assert 'row=2/2 | row_id="lamp" | status=failed | error_type="RuntimeError"' in output
    assert 'activity="completed" | total=2 | succeeded=1 | failed=1' in output
    assert "do not print me" not in output
    assert "/secret/reference.png" not in output
    assert "tool secret" not in output
    assert "tool result" not in output
    assert "provider body" not in output
    assert "\x1b" not in output


def test_live_display_renders_header_global_progress_and_exact_fixed_slots(
    monkeypatch,
) -> None:
    # A CI environment (GitHub Actions exports CI=true) forces the plain stream by design; the
    # live rendering under test needs a terminal-like environment.
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = _TTYBuffer()

    with ProviderBatchDisplay(
        stream=stream,
        terminal_size=(220, 20),
        refresh_hz=30,
    ) as display:
        display(
            _event(
                "batch_started",
                total=3,
                effective_concurrency=2,
                model="azure/openai/test-model-b",
                reasoning_effort="high",
            )
        )
        display(_event("row_started", slot=1, row_index=1, row_id="swing"))
        display(
            _event(
                "row_progress",
                slot=1,
                row_index=1,
                row_id="swing",
                message=(
                    'Provider Generation | activity="running final Host compile" | '
                    'prompt="hidden prompt"'
                ),
            )
        )
        display(
            _event(
                "row_finished",
                slot=1,
                row_index=1,
                row_id="swing",
                status="generation_ready",
                record_id="rec_swing",
            )
        )
        display(_event("row_started", slot=2, row_index=2, row_id="lamp"))

    rendered = _ANSI_ESCAPE.sub("", stream.getvalue()).replace("\r", "")
    assert "Provider Generation Batch" in rendered
    assert "total=3 | concurrency=2" in rendered
    assert 'model="azure/openai/test-model-b" | effort="high"' in rendered
    assert "Batch 1/3" in rendered
    assert "Slot 1" in rendered
    assert "Slot 2" in rendered
    assert "Slot 3" not in rendered
    assert 'activity="running final Host compile"' in rendered
    assert "hidden prompt" not in rendered


def test_short_tty_falls_back_to_plain_without_changing_reported_concurrency(
    monkeypatch,
) -> None:
    # A CI environment (GitHub Actions exports CI=true) forces the plain stream by design; the
    # live rendering under test needs a terminal-like environment.
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = _TTYBuffer()

    with ProviderBatchDisplay(stream=stream, terminal_size=(100, 6)) as display:
        display(
            _event(
                "batch_started",
                total=5,
                effective_concurrency=3,
                model="nvidia-model",
                reasoning_effort="high",
            )
        )
        display(_event("batch_finished", total=5, succeeded=0, failed=0))

    output = stream.getvalue()
    assert "concurrency=3" in output
    assert "\x1b" not in output


def test_ci_tty_uses_plain_logs(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("CI", "true")
    stream = _TTYBuffer()

    with ProviderBatchDisplay(stream=stream, terminal_size=(120, 40)) as display:
        display(
            _event(
                "batch_started",
                total=1,
                effective_concurrency=1,
                model="nvidia-model",
                reasoning_effort="high",
            )
        )
        display(_event("batch_finished", total=1, succeeded=1, failed=0))

    output = stream.getvalue()
    assert 'activity="starting"' in output
    assert "\x1b" not in output


def test_renderer_thread_start_failure_is_fail_open(monkeypatch) -> None:
    stream = io.StringIO()

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with ProviderBatchDisplay(stream=stream, force_plain=True) as display:
        display(
            _event(
                "batch_started",
                total=1,
                effective_concurrency=1,
                model="nvidia-model",
                reasoning_effort="high",
            )
        )

    assert stream.getvalue() == ""


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize(
    "label,source_label",
    [
        ("Codex Reconstruction", "Reconstruction Codex"),
        ("Codex Reconstruction", "Codex Reconstruction"),
        ("Provider Reconstruction", "Provider Reconstruction"),
    ],
)
def test_batch_arm_labels_retain_stages_and_filter_private_fields(
    monkeypatch, live, label, source_label
):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = _TTYBuffer() if live else io.StringIO()
    with ProviderBatchDisplay(
        stream=stream, force_plain=not live, terminal_size=(240, 30), label=label
    ) as display:
        display(
            _event(
                "batch_started",
                total=1,
                effective_concurrency=1,
                model="test-model-a",
                reasoning_effort="low",
            )
        )
        display(_event("row_started", slot=1, row_index=1, row_id="case"))
        for message in (
            "unrecognized private error body",
            f'{source_label} | activity="stage started" | stage=author | prompt="private task"',
            f'{source_label} | activity="stage finished" | stage=author | status=ok | duration=12.34s | error="private error"',
        ):
            display(_event("row_progress", slot=1, row_index=1, row_id="case", message=message))
        display(
            _event(
                "row_finished",
                slot=1,
                row_index=1,
                row_id="case",
                status="reconstruction_ready",
                record_id="rec_case",
            )
        )
        display(_event("batch_finished", total=1, succeeded=1, failed=0))
    output = _ANSI_ESCAPE.sub("", stream.getvalue()).replace("\r", "")
    assert f'{label} Batch | activity="starting"' in output
    assert f'{label} | activity="running"' in output
    assert f'{label} | activity="stage started" | stage=author' in output
    assert "stage=author | status=ok | duration=12.34s" in output
    assert "private" not in output
    assert "Provider Generation" not in output
    assert "Provider Batch" not in output
    if label == "Codex Reconstruction":
        assert "Provider" not in output
    if not live:
        assert f'{label} Batch | activity="row started"' in output
        assert f'{label} Batch | activity="row finished"' in output
        assert f'{label} Batch | activity="completed"' in output
    assert display._state.succeeded == 1
    assert display._state.slots[1].stage is None
