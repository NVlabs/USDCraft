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

"""Direct Codex Generation CSV execution using the shared row contract and scheduler.

Rows use the same CSV contract as `generation-batch`; only the Provider-only `api_style` cell is
rejected. The Codex arm has no time limit. Rows still run in supervised worker process groups so a
Ctrl-C cancellation stops each row's Codex processes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from threading import Event
from typing import Any, Mapping

from usdcraft.generation.command import generation_failure_context
from usdcraft.generation.provider_batch import (
    GENERATION_DEADLINE,
    ProviderBatchRow,
    load_provider_batch,
)
from usdcraft.pipeline.batch import ProviderBatchResult, ProviderBatchRowResult, run_provider_batch
from usdcraft.pipeline.batch_progress import ProviderBatchProgressCallback
from usdcraft.pipeline.codex import CodexUnavailableError
from usdcraft.pipeline.deadline import execute_with_deadline
from usdcraft.pipeline.provider.progress import ProviderProgressCallback
from usdcraft.pipeline.runtime import _codex_executable, resolve_codex_runtime

# CSV columns the Provider arm honours and the Codex arm has no equivalent for.
PROVIDER_ONLY_CSV_COLUMNS = ("api_style",)


def validate_codex_generation_batch_rows(rows: tuple[ProviderBatchRow, ...]) -> None:
    """Reject Provider-only row options: the Codex arm has no wire-protocol choice."""

    for row in rows:
        if row.api_style is not None:
            raise ValueError(f"Row {row.row_id!r}: api_style is a Provider-only CSV option.")


def execute_codex_generation_batch(
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
    rows = load_provider_batch(resolved_csv)
    validate_codex_generation_batch_rows(rows)
    # Resolve inherited runtime in the calling task before spawning standalone workers.
    if (model is None) != (reasoning is None):
        raise ValueError("--model and --reasoning must be provided together.")
    runtime_by_row = {}
    for row in rows:
        runtime = resolve_codex_runtime(
            model=row.model if row.model is not None else model,
            reasoning_effort=row.reasoning if row.model is not None else reasoning,
        )
        runtime_by_row[row.row_id] = (runtime.model_id, runtime.reasoning_effort)

    # Missing local CLI configuration affects every row; reject before staging any input.
    if shutil.which(_codex_executable(None)) is None:
        raise CodexUnavailableError(
            "Codex Generation batch requires the `codex` executable; "
            "export USDCRAFT_CODEX_BIN to its executable path in this shell "
            "(reload ~/.zshrc if you just added it) and verify it with --version. "
            "No batch rows have started."
        )

    def execute_row(
        row: ProviderBatchRow,
        row_progress: ProviderProgressCallback,
        cancellation_event: Event,
        batch_context: Mapping[str, Any],
    ) -> ProviderBatchRowResult:
        row_model, row_reasoning = runtime_by_row[row.row_id]
        return execute_with_deadline(
            GENERATION_DEADLINE,
            dict(
                repo_root=resolved_repo,
                prompt=row.prompt,
                image=row.image_path,
                model=row_model,
                reasoning=row_reasoning,
                label=row.label,
                keep_workspace=keep_workspace,
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
        failure_context=generation_failure_context,
        provider="codex-cli",
        runtime_by_row=runtime_by_row,
        concurrency=concurrency,
        progress=progress,
    )


__all__ = [
    "PROVIDER_ONLY_CSV_COLUMNS",
    "execute_codex_generation_batch",
    "validate_codex_generation_batch_rows",
]
