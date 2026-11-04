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

"""Reconstruction's contribution to the shared per-row Batch supervisor.

`usdcraft/pipeline/deadline.py` owns the clock, process-group supervision and retained state; this
module supplies the Reconstruction run locations and the worker primitive that runs one row
through the Provider or Codex direct primitive inside the supervised child process.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any, Callable

from usdcraft.pipeline.batch import ProviderBatchRowResult
from usdcraft.pipeline.deadline import DeadlineProduct, run_worker, validate_run_timeout
from usdcraft.pipeline.deadline import execute_with_deadline as _execute_with_deadline


def _workspaces(root: Path, run_id: str) -> dict[str, Path]:
    return {
        "run_workspace": root / "data/local/reconstruction_runs" / run_id,
        "author_workspace": root / "data/local/reconstruction_author_runs" / run_id,
    }


RECONSTRUCTION_DEADLINE = DeadlineProduct(
    run_id_prefix="recon",
    control_root="data/local/reconstruction_deadlines",
    success_status="reconstruction_ready",
    worker="usdcraft.reconstruction.provider_deadline:run_reconstruction_row",
    arm_labels={"provider": "Provider Reconstruction", "codex": "Codex Reconstruction"},
    workspaces=_workspaces,
    primary_workspace="author_workspace",
)


def execute_with_deadline(
    arguments: dict[str, Any],
    *,
    row_id: str,
    timeout_seconds: float | None,
    arm: str = "provider",
    progress: Callable[[str], None],
    cancellation_event: Event,
) -> ProviderBatchRowResult:
    """Supervise one Reconstruction row (Provider or Codex arm)."""

    return _execute_with_deadline(
        RECONSTRUCTION_DEADLINE,
        arguments,
        row_id=row_id,
        timeout_seconds=timeout_seconds,
        arm=arm,
        progress=progress,
        cancellation_event=cancellation_event,
    )


def run_reconstruction_row(
    arguments: dict[str, Any], arm: str, report: Callable[[str], None]
) -> Any:
    """Worker primitive: run one Reconstruction row in the supervised child process."""

    for name in ("repo_root", "source", "semantic_image"):
        arguments[name] = Path(arguments[name])
    if arm == "codex":
        from usdcraft.reconstruction.command import (
            execute_reconstruction,
            preflight_reconstruction,
        )

        run_id = arguments.pop("run_id")
        batch_context = arguments.pop("batch_context", None)
        preflight = preflight_reconstruction(**arguments)
        return execute_reconstruction(
            preflight=preflight,
            direct=True,
            run_id=run_id,
            batch_context=batch_context,
            progress=report,
        )
    from usdcraft.reconstruction.provider_command import execute_provider_reconstruction

    return execute_provider_reconstruction(**arguments, progress=report)


def _worker(control: Path) -> None:
    run_worker(control, run_reconstruction_row)


__all__ = [
    "RECONSTRUCTION_DEADLINE",
    "execute_with_deadline",
    "run_reconstruction_row",
    "validate_run_timeout",
]
