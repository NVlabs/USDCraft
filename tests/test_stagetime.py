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
from pathlib import Path

import pytest

from usdcraft.stagetime import STAGE_TRACE_FILENAME, StageTimer


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stage_records_log_lines_trace_events_and_ordered_report(tmp_path: Path) -> None:
    lines: list[str] = []
    trace = tmp_path / STAGE_TRACE_FILENAME
    timer = StageTimer(pipeline="Reconstruction", log=lines.append, trace_path=trace)

    with timer.stage("freeze"):
        pass
    with timer.stage("author", detail="sample=2"):
        pass

    report = timer.report()
    assert [record["stage"] for record in report] == ["freeze", "author"]
    assert all(record["status"] == "ok" for record in report)
    assert all(record["duration_seconds"] >= 0.0 for record in report)
    assert all(record["started_at_utc"].endswith("Z") for record in report)
    assert report[1]["detail"] == "sample=2"

    assert lines[0] == 'Reconstruction | activity="stage started" | stage=freeze'
    assert lines[1].startswith(
        'Reconstruction | activity="stage finished" | stage=freeze | status=ok | duration='
    )
    assert lines[2] == (
        'Reconstruction | activity="stage started" | stage=author | detail="sample=2"'
    )
    assert lines[3].startswith(
        'Reconstruction | activity="stage finished" | stage=author | status=ok | duration='
    )
    assert lines[3].endswith(' | detail="sample=2"')

    events = _events(trace)
    assert [event["event"] for event in events] == [
        "stage.started",
        "stage.finished",
        "stage.started",
        "stage.finished",
    ]
    assert events[3]["stage"] == "author"
    assert events[3]["status"] == "ok"
    assert events[3]["detail"] == "sample=2"


def test_failed_stage_is_recorded_and_reraises(tmp_path: Path) -> None:
    trace = tmp_path / STAGE_TRACE_FILENAME
    lines: list[str] = []
    timer = StageTimer(
        pipeline="Generation Provider",
        log=lines.append,
        log_label="Provider Generation",
        trace_path=trace,
    )

    with pytest.raises(ValueError, match="boom"):
        with timer.stage("host_gate"):
            raise ValueError("boom")

    (record,) = timer.report()
    assert record["stage"] == "host_gate"
    assert record["status"] == "failed"
    assert _events(trace)[-1]["status"] == "failed"
    # The trace and report keep the pipeline name; only the log lines carry the display label.
    assert _events(trace)[-1]["pipeline"] == "Generation Provider"
    assert lines[0] == 'Provider Generation | activity="stage started" | stage=host_gate'
    assert lines[1].startswith(
        'Provider Generation | activity="stage finished" | stage=host_gate | status=failed'
    )


def test_attach_trace_path_flushes_events_buffered_before_the_workspace_exists(
    tmp_path: Path,
) -> None:
    timer = StageTimer(pipeline="Reconstruction", log=None)
    with timer.stage("freeze"):
        pass
    trace = tmp_path / STAGE_TRACE_FILENAME
    timer.attach_trace_path(trace)
    with timer.stage("mesh_preprocess"):
        pass

    events = _events(trace)
    assert [event["stage"] for event in events] == [
        "freeze",
        "freeze",
        "mesh_preprocess",
        "mesh_preprocess",
    ]


def test_report_returns_copies_not_live_records(tmp_path: Path) -> None:
    timer = StageTimer(pipeline="Generation Provider", log=None)
    with timer.stage("freeze"):
        pass
    first = timer.report()
    first[0]["stage"] = "tampered"
    assert timer.report()[0]["stage"] == "freeze"
