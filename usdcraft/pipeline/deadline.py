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

"""Product-neutral POSIX per-row supervision for direct-execution Batch rows.

The supervisor runs one Batch row's direct primitive in a separate worker process group, owns the
row's wall-clock deadline (optional) and retained state, and turns the outcome into one
``ProviderBatchRowResult``. A timed-out or cancelled process group is stopped with its descendants
(Codex starts its own sessions), so it cannot continue authoring in the background after its Batch
slot has been released.

Each product contributes a ``DeadlineProduct``: its run-id prefix, control and workspace locations,
success status, progress labels and the dotted path of the worker primitive executed in the child.
The worker primitive receives the JSON request arguments, the arm name and a progress callback and
returns the primitive's result (an object with ``status`` and ``to_dict()``).
"""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

from usdcraft.pipeline.batch import ProviderBatchRowResult

WorkerPrimitive = Callable[[dict[str, Any], str, Callable[[str], None]], Any]


@dataclass(frozen=True, slots=True)
class DeadlineProduct:
    """What one product contributes to the shared per-row supervisor."""

    run_id_prefix: str
    # Repository-relative directory receiving one control directory per supervised row.
    control_root: str
    success_status: str
    # ``module:function`` of the worker primitive, imported inside the child process so import and
    # runtime initialization count against the row's deadline.
    worker: str
    arm_labels: Mapping[str, str]
    # Named workspace paths for a run id, in snapshot order. ``primary_workspace`` names the one
    # reported as the row's ``workspace_dir`` when it exists (otherwise the control directory).
    workspaces: Callable[[Path, str], dict[str, Path]]
    primary_workspace: str


# The Provider batch commands' default per-row deadline (`--run-timeout`; 0 disables it).
DEFAULT_RUN_TIMEOUT_SECONDS = 1920.0


def validate_run_timeout(seconds: float | None) -> None:
    if seconds is None:
        return
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("run_timeout_seconds must be finite and positive")
    if os.name != "posix":
        raise ValueError("Batch run timeout currently requires POSIX process groups")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _descendant_groups(pid: int) -> set[int]:
    """Codex starts its own session; include those descendant groups when stopping a row."""
    listing = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid="], text=True, timeout=2)
    processes = [tuple(map(int, line.split())) for line in listing.splitlines() if line.strip()]
    descendants = {pid}
    while True:
        children = {child for child, parent, _ in processes if parent in descendants}
        if children <= descendants:
            break
        descendants.update(children)
    return {group for child, _, group in processes if child in descendants} - {os.getpgrp()}


def _group_is_live(group: int) -> bool:
    """Whether ``group`` still has a member that is not a zombie (``ps`` state ``Z``)."""
    listing = subprocess.check_output(["ps", "-axo", "pgid=,stat="], text=True, timeout=2)
    for line in listing.splitlines():
        fields = line.split()
        if not fields or int(fields[0]) != group:
            continue
        if len(fields) < 2 or not fields[1].startswith("Z"):
            return True
    return False


def _stop_group(process: subprocess.Popen) -> None:
    first_error: Exception | None = None
    try:
        # Codex and auxiliary subprocesses start separate sessions under this worker. Signal every
        # group before reporting a failure so one inaccessible group cannot leave the others alive.
        for group in sorted(_descendant_groups(process.pid) - {process.pid}):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                # macOS may return EPERM after the last live member of a group exits, including
                # while only unreaped zombies remain. Ignore only such a group; a live
                # inaccessible group is a real failure.
                try:
                    live = _group_is_live(group)
                except Exception:
                    live = True
                if live and first_error is None:
                    first_error = exc
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
    finally:
        # Always stop the worker, including when descendant enumeration fails.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS can report EPERM for a group whose members are exiting; stop the worker
            # itself so the row still ends as a timeout. wait() raises if it survives.
            if process.poll() is None:
                process.kill()
        process.wait(timeout=5)


def execute_with_deadline(
    product: DeadlineProduct,
    arguments: dict[str, Any],
    *,
    row_id: str,
    timeout_seconds: float | None,
    arm: str = "provider",
    progress: Callable[[str], None],
    cancellation_event: Event,
) -> ProviderBatchRowResult:
    """Run one row's primitive in a supervised worker process group.

    ``timeout_seconds=None`` supervises without a deadline, so cancellation still stops the row.
    """

    validate_run_timeout(timeout_seconds)
    if arm not in product.arm_labels:
        raise ValueError(f"Unknown supervised arm: {arm!r}")
    if os.name != "posix":
        raise ValueError("Supervised batches require POSIX")
    label = product.arm_labels[arm]
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    deadline = started + timeout_seconds if timeout_seconds is not None else math.inf
    root = Path(arguments["repo_root"])
    run_id = f"{product.run_id_prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    control = root / product.control_root / run_id
    control.mkdir(parents=True)
    workspaces = product.workspaces(root, run_id)
    request: dict[str, Any] = {
        k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()
    }
    keep_workspace = request.get("keep_workspace", False)
    request.update(run_id=run_id, keep_workspace=True)
    request["batch_context"] = {
        **(request.get("batch_context") or {}),
        "run_timeout_seconds": timeout_seconds,
    }
    _write_json(control / "request.json", request)
    _write_json(control / "transport.json", {"arm": arm, "worker": product.worker})
    progress(
        f'{label} | activity="deadline started" | run_id={run_id!r} | run_timeout_seconds={timeout_seconds}'
    )
    events = control / "progress.jsonl"
    events.touch()
    result_path = control / "result.json"
    outcome: dict[str, Any] | None = None
    last_progress: str | None = None
    state = "failed"
    process = subprocess.Popen(
        [sys.executable, "-m", "usdcraft.pipeline.deadline", str(control)],
        cwd=root,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with events.open() as stream:
            while True:
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        stream.seek(position)
                        break
                    message = json.loads(line)["message"]
                    last_progress = message
                    progress(message)
                if result_path.is_file():
                    outcome = json.loads(result_path.read_text())
                    state = (
                        outcome["status"]
                        if outcome["finished_monotonic"] <= deadline
                        else "timeout"
                    )
                    break
                if time.monotonic() >= deadline:
                    state = "timeout"
                    break
                if cancellation_event.is_set():
                    state = "cancelled"
                    break
                if process.poll() is not None:
                    # The atomic result may have appeared between the first check and poll.
                    if result_path.is_file():
                        continue
                    break
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    finally:
        _stop_group(process)
    stopped_elapsed = time.monotonic() - started
    elapsed = (
        outcome["finished_monotonic"] - started
        if outcome and state != "timeout"
        else stopped_elapsed
    )
    snapshot = control / "snapshot.json"
    payload = outcome.get("result", {}) if outcome else {}
    succeeded = state == product.success_status
    _write_json(
        snapshot,
        {
            "status": state,
            "row_id": row_id,
            "run_id": run_id,
            "started_at": started_at,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": round(elapsed, 3),
            "stopped_elapsed_seconds": round(stopped_elapsed, 3),
            "last_progress": last_progress,
            "worker_returncode": process.returncode,
            **{name: str(path) for name, path in workspaces.items()},
            "progress_path": str(events),
            "batch": request["batch_context"],
            "worker_result": outcome,
            "state_scope": "Completed on-disk traces, model, feedback and build at stop; an unfinished remote response is unavailable.",
        },
    )
    if succeeded and not keep_workspace:
        for path in workspaces.values():
            if path.is_dir():
                try:
                    shutil.rmtree(path)
                except OSError:
                    payload.setdefault("warnings", []).append(
                        f"Completed workspace retained: {path}"
                    )
    if state in {"timeout", "cancelled"}:
        progress(f"{label} | activity={state} | run_id={run_id!r} | snapshot={str(snapshot)!r}")
    primary = workspaces[product.primary_workspace]
    return ProviderBatchRowResult(
        row_id=row_id,
        status=state,
        record_id=payload.get("record_id") if succeeded else None,
        record_dir=Path(payload["record_dir"]) if succeeded else None,
        run_id=run_id,
        workspace_dir=primary if primary.is_dir() else control,
        warnings=tuple(payload.get("warnings", ())),
        error=(
            f"Run exceeded {timeout_seconds:g} seconds"
            if state == "timeout"
            else None
            if succeeded
            else (outcome or {}).get("error_type", state)
        ),
        elapsed_seconds=round(elapsed, 3),
        timeout_seconds=timeout_seconds,
        snapshot_path=snapshot,
    )


def _resolve_worker(dotted: str) -> WorkerPrimitive:
    module_name, _, function_name = dotted.partition(":")
    return getattr(importlib.import_module(module_name), function_name)


def run_worker(control: Path, worker: WorkerPrimitive | None = None) -> None:
    """Child-process body: run the product primitive once and write ``result.json`` atomically."""

    # Import and runtime initialization are inside the supervisor's total deadline.
    arguments = json.loads((control / "request.json").read_text())

    def report(message: str) -> None:
        with (control / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"message": message}) + "\n")

    try:
        transport_path = control / "transport.json"
        transport = json.loads(transport_path.read_text()) if transport_path.is_file() else {}
        arm = str(transport.get("arm") or "provider")
        primitive = worker if worker is not None else _resolve_worker(str(transport["worker"]))
        result = primitive(arguments, arm, report)
        outcome = {"status": result.status, "result": result.to_dict()}
    except BaseException as exc:
        # Never persist arbitrary transport exception bodies or credentials.
        outcome = {"status": "failed", "error_type": type(exc).__name__}
    outcome["finished_monotonic"] = time.monotonic()
    _write_json(control / "result.json", outcome)


__all__ = [
    "DEFAULT_RUN_TIMEOUT_SECONDS",
    "DeadlineProduct",
    "execute_with_deadline",
    "run_worker",
    "validate_run_timeout",
]


if __name__ == "__main__":
    run_worker(Path(sys.argv[1]))
