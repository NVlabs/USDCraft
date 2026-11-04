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

"""Direct Codex Reconstruction CSV execution using the shared row contract and scheduler.

Rows use the same CSV contract as `reconstruction-batch`, including the optional `mesh_cleanup`
column; only the Provider-only `max_turns` and `api_style` cells are rejected. The Codex arm has
no time or turn limit. Rows still run in supervised worker process groups so a Ctrl-C cancellation
stops each row's Codex processes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from threading import Event
from typing import Any, Mapping

from usdcraft.pipeline.batch import ProviderBatchResult, ProviderBatchRowResult, run_provider_batch
from usdcraft.pipeline.batch_progress import ProviderBatchProgressCallback
from usdcraft.pipeline.codex import CodexUnavailableError
from usdcraft.pipeline.provider.progress import ProviderProgressCallback
from usdcraft.pipeline.runtime import _codex_executable
from usdcraft.reconstruction.command import _resolve_runtime, reconstruction_failure_context
from usdcraft.reconstruction.provider_batch import ReconstructionBatchRow, load_reconstruction_batch
from usdcraft.reconstruction.provider_deadline import execute_with_deadline

# CSV columns the Provider arm honours and the Codex arm has no equivalent for.
PROVIDER_ONLY_CSV_COLUMNS = ("max_turns", "api_style")


def validate_codex_batch_rows(rows: tuple[ReconstructionBatchRow, ...]) -> None:
    """Reject Provider-only row options: the Codex arm has no turn budget or wire protocol."""

    for row in rows:
        if row.max_turns is not None:
            raise ValueError(
                f"Row {row.row_id!r}: max_turns is a Provider-only CSV option; "
                "the Codex arm has no turn limit."
            )
        if row.api_style is not None:
            raise ValueError(f"Row {row.row_id!r}: api_style is a Provider-only CSV option.")


def execute_codex_reconstruction_batch(
    *,
    csv_path: Path,
    repo_root: Path,
    model: str | None = None,
    reasoning: str | None = None,
    concurrency: int = 1,
    keep_workspace: bool = False,
    progress: ProviderBatchProgressCallback | None = None,
) -> ProviderBatchResult:
    """Each row authorizes direct execution; each child has its own Codex conversation."""
    resolved_csv = Path(csv_path).expanduser().resolve()
    resolved_repo = Path(repo_root).expanduser().resolve()
    rows = load_reconstruction_batch(resolved_csv)
    validate_codex_batch_rows(rows)
    # Resolve inherited runtime in the calling task before spawning standalone workers.
    if (model is None) != (reasoning is None):
        raise ValueError("--model and --reasoning must be provided together.")
    runtime_by_row = {}
    for row in rows:
        runtime = _resolve_runtime(
            model=row.model if row.model is not None else model,
            reasoning=row.reasoning if row.model is not None else reasoning,
            environ=None,
            codex_home=None,
        )
        runtime_by_row[row.row_id] = (runtime.model_id, runtime.reasoning_effort)

    # Missing local CLI configuration affects every row; reject before staging any evidence.
    if shutil.which(_codex_executable(None)) is None:
        raise CodexUnavailableError(
            "Codex Reconstruction batch requires the `codex` executable; "
            "export USDCRAFT_CODEX_BIN to its executable path in this shell "
            "(reload ~/.zshrc if you just added it) and verify it with --version. "
            "No batch rows have started."
        )

    def execute_row(
        row: ReconstructionBatchRow,
        row_progress: ProviderProgressCallback,
        cancellation_event: Event,
        batch_context: Mapping[str, Any],
    ) -> ProviderBatchRowResult:
        row_model, row_reasoning = runtime_by_row[row.row_id]
        return execute_with_deadline(
            dict(
                repo_root=resolved_repo,
                source=row.source_path,
                semantic_image=row.image_path,
                up_axis=row.up_axis,
                front_axis=row.front_axis,
                meters_per_unit=row.meters_per_unit,
                guidance=row.guidance,
                prompt=row.prompt,
                label=row.label,
                model=row_model,
                reasoning=row_reasoning,
                keep_workspace=keep_workspace,
                image_observation=row.image_observation,
                mesh_cleanup=row.mesh_cleanup,
                batch_context=dict(batch_context),
            ),
            arm="codex",
            row_id=row.row_id,
            # Supervised without a deadline: the Codex arm has no time limit.
            timeout_seconds=None,
            progress=row_progress,
            cancellation_event=cancellation_event,
        )

    return run_provider_batch(
        csv_path=resolved_csv,
        rows=rows,
        repo_root=resolved_repo,
        execute_row=execute_row,
        failure_context=reconstruction_failure_context,
        provider="codex-cli",
        runtime_by_row=runtime_by_row,
        concurrency=concurrency,
        progress=progress,
    )
