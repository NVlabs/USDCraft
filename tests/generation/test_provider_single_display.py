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
from datetime import datetime

import pytest

from usdcraft.pipeline.single_display import ProviderSingleDisplay

_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        raise OSError("test stream has no file descriptor")


class _RecordingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.writer_threads: list[int] = []

    def write(self, value: str) -> int:
        self.writer_threads.append(threading.get_ident())
        return super().write(value)


class _FailingStream(io.StringIO):
    def write(self, _value: str) -> int:
        raise OSError("presentation stream failed")

    def flush(self) -> None:
        raise OSError("presentation stream failed")


def _without_ansi(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value).replace("\r", "")


def _assert_plain_timestamps_are_timezone_aware(output: str) -> None:
    lines = [line for line in output.splitlines() if line]
    assert lines
    for line in lines:
        assert line.startswith("[")
        timestamp_text, separator, _message = line[1:].partition("] ")
        assert separator
        timestamp = datetime.fromisoformat(timestamp_text)
        assert timestamp.utcoffset() is not None


def _starting_message() -> str:
    return (
        "Provider Generation | activity=starting | "
        'model="azure/openai/test-model-b" | effort=high | max_turns=64'
    )


def test_plain_display_logs_start_activity_and_completion_with_aware_timestamps() -> None:
    stream = io.StringIO()

    with ProviderSingleDisplay(stream=stream, force_plain=True) as display:
        display(_starting_message())
        display('Provider Generation | turn=8/64 | activity="running final Host compile"')
        display(
            "Provider Generation | activity=completed | status=generation_ready | "
            'record_id="rec_one" | elapsed=12.30s'
        )

    output = stream.getvalue()
    assert "Provider Generation" in output
    assert 'activity="starting"' in output
    assert 'model="azure/openai/test-model-b"' in output
    assert 'effort="high"' in output
    assert "max_turns=64" in output
    assert 'activity="running final Host compile"' in output
    assert "turn=8/64" in output
    assert "activity=completed" in output
    assert "status=generation_ready" in output
    assert 'record_id="rec_one"' in output
    assert "elapsed=12.30s" in output
    assert "\x1b" not in output
    _assert_plain_timestamps_are_timezone_aware(output)


def test_plain_display_projects_allowlisted_fields_and_removes_control_characters() -> None:
    stream = io.StringIO()
    sensitive_message = (
        "Provider Generation | turn=2/64 | "
        'activity="running final Host compile" | tool="compile_model" | status=running | '
        'prompt="do not\nprint\x1b[31m" | image="/secret/reference.png" | '
        'arguments="tool secret" | result="tool result" | error="provider body\x7f"'
    )

    with ProviderSingleDisplay(stream=stream, force_plain=True) as display:
        display(sensitive_message)

    output = stream.getvalue()
    assert "turn=2/64" in output
    assert 'activity="running final Host compile"' in output
    assert 'tool="compile_model"' in output
    assert "status=running" in output
    assert "do not" not in output
    assert "/secret/reference.png" not in output
    assert "tool secret" not in output
    assert "tool result" not in output
    assert "provider body" not in output
    assert "\x1b" not in output
    assert "\x7f" not in output
    assert len(output.splitlines()) == 1


def test_live_display_is_one_generation_block_and_reaches_success(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = _TTYBuffer()

    with ProviderSingleDisplay(
        stream=stream,
        terminal_size=(180, 12),
        refresh_hz=30,
    ) as display:
        display(_starting_message())
        display('Provider Generation | turn=8/64 | activity="running final Host compile"')
        display(
            "Provider Generation | activity=completed | status=generation_ready | "
            'record_id="rec_one" | elapsed=12.30s'
        )

    raw_output = stream.getvalue()
    output = _without_ansi(raw_output)
    assert "\x1b" in raw_output
    assert "Provider Generation" in output
    assert 'model="azure/openai/test-model-b"' in output
    assert "effort=high" in output
    assert "max_turns=64" in output
    assert 'activity="running final Host compile"' in output
    assert "status=generation_ready" in output
    assert 'record_id="rec_one"' in output
    assert "Provider Batch" not in output
    assert "Slot " not in output
    assert "row=" not in output
    assert "Batch " not in output
    assert "active=" not in output
    assert "queued=" not in output
    assert "succeeded=" not in output


def test_live_display_reaches_failure_without_exposing_error_body(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = _TTYBuffer()

    with ProviderSingleDisplay(
        stream=stream,
        terminal_size=(180, 12),
        refresh_hz=30,
    ) as display:
        display(_starting_message())
        display('Provider Generation | activity="requesting NVIDIA response" | turn=4/64')
        display(
            "Provider Generation | activity=failed | error_type=RuntimeError | elapsed=3.20s | "
            'error="provider body must stay hidden"'
        )

    output = _without_ansi(stream.getvalue())
    assert "activity=failed" in output
    assert "error_type=RuntimeError" in output
    assert "elapsed=3.20s" in output
    assert "provider body must stay hidden" not in output
    assert "Provider Batch" not in output


@pytest.mark.parametrize("fallback", ["force_plain", "ci", "term_dumb", "short_tty"])
def test_tty_plain_fallbacks_have_no_ansi(monkeypatch, fallback: str) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    force_plain = False
    terminal_size = (180, 12)
    if fallback == "force_plain":
        force_plain = True
    elif fallback == "ci":
        monkeypatch.setenv("CI", "true")
    elif fallback == "term_dumb":
        monkeypatch.setenv("TERM", "dumb")
    elif fallback == "short_tty":
        terminal_size = (180, 2)

    stream = _TTYBuffer()
    with ProviderSingleDisplay(
        stream=stream,
        force_plain=force_plain,
        terminal_size=terminal_size,
    ) as display:
        display(_starting_message())
        display("Provider Generation | activity=completed | status=generation_ready")

    output = stream.getvalue()
    assert 'activity="starting"' in output
    assert "status=generation_ready" in output
    assert "\x1b" not in output
    _assert_plain_timestamps_are_timezone_aware(output)


def test_close_drains_is_idempotent_and_ignores_later_progress() -> None:
    stream = io.StringIO()
    display = ProviderSingleDisplay(stream=stream, force_plain=True)
    display(_starting_message())
    display('Provider Generation | activity="running final Host compile"')
    display("Provider Generation | activity=completed | status=generation_ready")

    display.close()
    output_after_close = stream.getvalue()
    display.close()
    display('Provider Generation | activity="must not appear after close"')

    assert 'activity="starting"' in output_after_close
    assert 'activity="running final Host compile"' in output_after_close
    assert "status=generation_ready" in output_after_close
    assert stream.getvalue() == output_after_close


def test_stream_write_failure_is_fail_open() -> None:
    display = ProviderSingleDisplay(stream=_FailingStream(), force_plain=True)
    with display:
        display(_starting_message())
        display('Provider Generation | activity="running final Host compile"')
        display("Provider Generation | activity=completed | status=generation_ready")
    display.close()


def test_renderer_thread_start_failure_is_fail_open(monkeypatch) -> None:
    stream = io.StringIO()

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with ProviderSingleDisplay(stream=stream, force_plain=True) as display:
        display(_starting_message())
        display("Provider Generation | activity=completed | status=generation_ready")

    assert stream.getvalue() == ""


def test_close_before_first_progress_has_no_ghost_output(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = _TTYBuffer()

    with ProviderSingleDisplay(
        stream=stream,
        terminal_size=(180, 12),
        refresh_hz=30,
    ):
        pass

    assert stream.getvalue() == ""
