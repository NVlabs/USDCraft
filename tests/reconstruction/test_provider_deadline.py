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
import subprocess
import sys
import time
from threading import Event

import pytest

from usdcraft.reconstruction.provider_deadline import execute_with_deadline, validate_run_timeout

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process-group deadline")


@pytest.mark.parametrize("live_group", [False, True])
def test_cleanup_eperm_ignores_only_exited_descendant_group(monkeypatch, live_group):
    from types import SimpleNamespace

    from usdcraft.pipeline import deadline

    stopped = []
    process = SimpleNamespace(pid=100, poll=lambda: 0, wait=lambda timeout: stopped.append(100))
    monkeypatch.setattr(deadline, "_descendant_groups", lambda pid: {100, 200})
    monkeypatch.setattr(
        deadline.subprocess, "check_output", lambda *a, **kw: "200\n" if live_group else "300\n"
    )

    def kill(group, sig):
        if group == 200:
            raise PermissionError(1, "Operation not permitted")
        raise ProcessLookupError()

    monkeypatch.setattr(deadline.os, "killpg", kill)
    if live_group:
        with pytest.raises(PermissionError):
            deadline._stop_group(process)
    else:
        deadline._stop_group(process)
    assert stopped == [100]


def test_cleanup_signals_every_group_and_ignores_zombie_only_groups(monkeypatch):
    from types import SimpleNamespace

    from usdcraft.pipeline import deadline

    stopped = []
    signalled = []
    process = SimpleNamespace(pid=100, poll=lambda: 0, wait=lambda timeout: stopped.append(100))
    monkeypatch.setattr(deadline, "_descendant_groups", lambda pid: {100, 200, 300, 400})
    # Group 200 holds only zombies; group 300 still has a live (sleeping) member.
    monkeypatch.setattr(
        deadline.subprocess,
        "check_output",
        lambda *a, **kw: "  200 Z\n  200 Z+\n  300 S\n  300 Z\n  400 R\n",
    )

    def kill(group, sig):
        signalled.append(group)
        if group in {200, 300}:
            raise PermissionError(1, "Operation not permitted")
        if group == 100:
            raise ProcessLookupError()

    monkeypatch.setattr(deadline.os, "killpg", kill)
    with pytest.raises(PermissionError):
        deadline._stop_group(process)
    # The live inaccessible group fails only after every other group and the worker were signalled.
    assert signalled == [200, 300, 400, 100]
    assert stopped == [100]

    # Without the live group, a zombie-only group alone is treated as already gone.
    signalled.clear()
    stopped.clear()
    monkeypatch.setattr(deadline, "_descendant_groups", lambda pid: {100, 200, 400})
    deadline._stop_group(process)
    assert signalled == [200, 400, 100]
    assert stopped == [100]


def test_cleanup_eperm_on_the_worker_group_kills_the_live_worker(monkeypatch):
    from usdcraft.pipeline import deadline

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(deadline, "_descendant_groups", lambda pid: {process.pid})

    def kill(group, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(deadline.os, "killpg", kill)
    deadline._stop_group(process)
    assert process.poll() is not None


def worker_stub(tmp_path, monkeypatch, *, finish=False):
    script = tmp_path / "worker.py"
    script.write_text("""import json, subprocess, sys, time
from pathlib import Path
control=Path(sys.argv[1]); request=json.loads((control/'request.json').read_text())
root=Path(request['repo_root']); author=root/'data/local/reconstruction_author_runs'/request['run_id']
author.mkdir(parents=True)
(author/'model.py').write_text('x = 1\\n')
(author/'author_feedback.json').write_text('{"revision": 1, "axial_checks": 1}')
(control/'progress.jsonl').write_text(json.dumps({'message':'stage=author turn=7'})+'\\n')
if sys.argv[2]=='finish':
    outcome={'status':'reconstruction_ready','finished_monotonic':time.monotonic(),'result':{'record_id':'rec_test','record_dir':str(root/'data/records/rec_test'),'warnings':[]}}
    (control/'result.json').write_text(json.dumps(outcome))
else:
    subprocess.Popen([sys.executable,'-c',"import time; from pathlib import Path; p=Path("+repr(str(control/'heartbeat'))+"); [(p.write_text(str(i)),time.sleep(.05)) for i in range(200)]"], start_new_session=True)
    time.sleep(20)
""")
    original = subprocess.Popen

    def spawn(argv, **kwargs):
        if argv[0] == "ps":
            return original(argv, **kwargs)
        assert kwargs["start_new_session"] is True
        return original(
            [sys.executable, str(script), argv[-1], "finish" if finish else "hang"], **kwargs
        )

    monkeypatch.setattr("usdcraft.pipeline.deadline.subprocess.Popen", spawn)


def arguments(tmp_path):
    return {
        "repo_root": tmp_path,
        "source": tmp_path / "input.glb",
        "semantic_image": tmp_path / "image.jpg",
        "keep_workspace": False,
    }


@pytest.mark.slow
def test_total_deadline_stops_descendants_and_retains_last_state(tmp_path, monkeypatch):
    worker_stub(tmp_path, monkeypatch)
    messages = []
    result = execute_with_deadline(
        arguments(tmp_path),
        row_id="case_s00",
        timeout_seconds=1,
        progress=messages.append,
        cancellation_event=Event(),
    )
    assert result.status == "timeout" and result.record_id is None
    # The stub worker hangs for 20 s, so a bounded elapsed time proves the deadline stopped it.
    assert 1 <= result.elapsed_seconds < 10
    assert (result.workspace_dir / "model.py").read_text() == "x = 1\n"
    assert json.loads((result.workspace_dir / "author_feedback.json").read_text())["revision"] == 1
    snapshot = json.loads(result.snapshot_path.read_text())
    assert snapshot["last_progress"] == "stage=author turn=7"
    assert snapshot["status"] == "timeout" and snapshot["timeout_seconds"] == 1
    # The detached grandchild must be stopped too: its heartbeat stops changing (or, if it was
    # stopped before its first write, never appears).
    heartbeat = result.snapshot_path.parent / "heartbeat"
    if heartbeat.exists():
        before = heartbeat.read_bytes()
        time.sleep(0.2)
        assert heartbeat.read_bytes() == before
    else:
        time.sleep(0.2)
        assert not heartbeat.exists()
    assert result.to_dict()["snapshot_path"] == str(result.snapshot_path)
    assert not (tmp_path / "data/records").exists()


@pytest.mark.parametrize("limit", [None, 30])
def test_result_within_limit_succeeds_and_cleans_workspace(tmp_path, monkeypatch, limit):
    worker_stub(tmp_path, monkeypatch, finish=True)
    result = execute_with_deadline(
        arguments(tmp_path),
        row_id="case_s00",
        timeout_seconds=limit,
        progress=lambda _: None,
        cancellation_event=Event(),
    )
    assert result.status == "reconstruction_ready" and result.record_id == "rec_test"
    assert result.elapsed_seconds < 10
    assert not (tmp_path / "data/local/reconstruction_author_runs" / result.run_id).exists()
    assert result.snapshot_path.is_file()


def test_user_cancellation_is_distinct_from_timeout(tmp_path, monkeypatch):
    worker_stub(tmp_path, monkeypatch)
    cancel = Event()
    cancel.set()
    result = execute_with_deadline(
        arguments(tmp_path),
        row_id="case_s00",
        timeout_seconds=3,
        progress=lambda _: None,
        cancellation_event=cancel,
    )
    assert result.status == "cancelled"


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True])
def test_bad_deadlines_fail_before_execution(seconds):
    with pytest.raises(ValueError, match="finite and positive"):
        validate_run_timeout(seconds)


@pytest.mark.slow
def test_actual_worker_failure_is_reported_without_credentials_or_api_calls(tmp_path):
    result = execute_with_deadline(
        arguments(tmp_path),
        row_id="missing_inputs",
        timeout_seconds=120,
        progress=lambda _: None,
        cancellation_event=Event(),
    )
    assert result.status == "failed"
    snapshot = json.loads(result.snapshot_path.read_text())
    assert snapshot["worker_result"]["error_type"]
    assert "result" not in snapshot["worker_result"]
