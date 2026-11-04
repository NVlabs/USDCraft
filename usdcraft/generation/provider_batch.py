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

"""Generation Batch: the CSV row contract and the row executor for `usdcraft generation-batch`.

Scheduling, failure isolation, progress events and the Ctrl-C snapshot protocol live in the shared
`usdcraft/pipeline/batch.py` coordinator, and the optional per-row deadline in the shared
`usdcraft/pipeline/deadline.py` supervisor; this module contributes only the Generation CSV columns
and the call into the same direct primitive used by `usdcraft generation`. The same CSV contract
and supervised worker also serve `generation-codex-batch` (`usdcraft/generation/codex_batch.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

from usdcraft.generation.command import GenerationResult, generation_failure_context
from usdcraft.pipeline.batch import (
    MAX_PROVIDER_BATCH_CONCURRENCY,
    ProviderBatchInterrupted,
    ProviderBatchResult,
    ProviderBatchRowResult,
    iter_batch_csv,
    optional_text,
    reject_unknown_columns,
    resolve_csv_relative_path,
    resolve_provider_row_runtimes,
    row_runtime_override,
    run_provider_batch,
)
from usdcraft.pipeline.batch_progress import ProviderBatchProgressCallback
from usdcraft.pipeline.deadline import DeadlineProduct, validate_run_timeout
from usdcraft.pipeline.provider.client import resolve_provider_endpoint
from usdcraft.pipeline.provider.progress import ProviderProgressCallback

_REQUIRED_COLUMNS = frozenset({"row_id", "prompt", "image_path"})
# The complete Generation CSV contract, in the order the unknown-column error lists it.
GENERATION_CSV_COLUMNS = (
    "row_id",
    "prompt",
    "image_path",
    "label",
    "model",
    "reasoning",
    "api_style",
)


@dataclass(slots=True, frozen=True)
class ProviderBatchRow:
    row_id: str
    prompt: str | None
    image_path: Path | None
    label: str | None
    model: str | None = None
    reasoning: str | None = None
    api_style: str | None = None


def load_provider_batch(csv_path: Path) -> tuple[ProviderBatchRow, ...]:
    """Validate a `generation-batch` CSV without creating workflow state.

    ``image_path`` resolves relative to the CSV directory; image decodability is checked by the
    primitive when the row runs, exactly as `usdcraft generation` checks it.
    """

    resolved_csv = Path(csv_path).expanduser().resolve()
    rows: list[ProviderBatchRow] = []
    for location, raw_row, row_id in iter_batch_csv(
        resolved_csv, required_columns=_REQUIRED_COLUMNS
    ):
        reject_unknown_columns(location, raw_row, allowed_columns=GENERATION_CSV_COLUMNS)
        runtime_override = row_runtime_override(location, raw_row)
        image_value = optional_text(raw_row.get("image_path"))
        rows.append(
            ProviderBatchRow(
                row_id=row_id,
                prompt=optional_text(raw_row.get("prompt")),
                image_path=(
                    resolve_csv_relative_path(image_value, csv_dir=resolved_csv.parent)
                    if image_value is not None
                    else None
                ),
                label=optional_text(raw_row.get("label")),
                model=runtime_override.model,
                reasoning=runtime_override.reasoning,
                api_style=runtime_override.api_style,
            )
        )
    return tuple(rows)


def _generation_workspaces(root: Path, run_id: str) -> dict[str, Path]:
    return {"workspace": root / "data/local/generation_runs" / run_id}


GENERATION_DEADLINE = DeadlineProduct(
    run_id_prefix="gen",
    control_root="data/local/generation_deadlines",
    success_status="generation_ready",
    worker="usdcraft.generation.provider_batch:run_generation_row",
    arm_labels={"provider": "Provider Generation", "codex": "Codex Generation"},
    workspaces=_generation_workspaces,
    primary_workspace="workspace",
)


def run_generation_row(arguments: dict[str, Any], arm: str, report: Callable[[str], None]) -> Any:
    """Worker primitive: run one Generation row (Provider or Codex arm) in the supervised child."""

    arguments["repo_root"] = Path(arguments["repo_root"])
    if arguments.get("image") is not None:
        arguments["image"] = Path(arguments["image"])
    if arm == "codex":
        from usdcraft.generation.command import execute_generation, preflight_generation

        run_id = arguments.pop("run_id")
        batch_context = arguments.pop("batch_context", None)
        preflight = preflight_generation(**arguments)
        return execute_generation(
            preflight=preflight,
            direct=True,
            run_id=run_id,
            batch_context=batch_context,
            progress=report,
        )
    from usdcraft.generation.provider_command import execute_provider_generation

    return execute_provider_generation(**arguments, progress=report)


def execute_provider_batch(
    *,
    csv_path: Path,
    repo_root: Path,
    model: str | None = None,
    reasoning: str | None = None,
    timeout_seconds: float | None = None,
    run_timeout_seconds: float | None = None,
    concurrency: int = 1,
    keep_workspace: bool = False,
    progress: ProviderBatchProgressCallback | None = None,
    api_style: str = "responses",
) -> ProviderBatchResult:
    """Run each CSV row through the same direct Provider Generation primitive."""

    resolved_csv = Path(csv_path).expanduser().resolve()
    resolved_repo = Path(repo_root).expanduser().resolve()
    rows = load_provider_batch(resolved_csv)
    validate_run_timeout(run_timeout_seconds)
    runtime_by_row = resolve_provider_row_runtimes(
        resolved_repo, rows, model=model, reasoning=reasoning
    )

    def execute_row(
        row: ProviderBatchRow,
        row_progress: ProviderProgressCallback,
        cancellation_event: Event,
        batch_context: Mapping[str, Any],
    ) -> ProviderBatchRowResult:
        arguments: dict[str, Any] = dict(
            repo_root=resolved_repo,
            prompt=row.prompt,
            image=row.image_path,
            model=runtime_by_row[row.row_id][0] if runtime_by_row else model,
            reasoning=runtime_by_row[row.row_id][1] if runtime_by_row else reasoning,
            timeout_seconds=timeout_seconds,
            label=row.label,
            keep_workspace=keep_workspace,
            batch_context=batch_context,
        )
        if row.api_style is not None or api_style != "responses":
            arguments["api_style"] = row.api_style or api_style
        if run_timeout_seconds is not None:
            from usdcraft.pipeline.deadline import execute_with_deadline

            return execute_with_deadline(
                GENERATION_DEADLINE,
                {**arguments, "batch_context": dict(batch_context)},
                row_id=row.row_id,
                timeout_seconds=run_timeout_seconds,
                progress=row_progress,
                cancellation_event=cancellation_event,
            )
        # Import lazily so the CSV contract stays inspectable without loading the API client, and
        # keep the import inside the row boundary so a broken optional runtime fails one item.
        from usdcraft.generation.provider_command import execute_provider_generation

        result: GenerationResult = execute_provider_generation(
            **arguments,
            progress=row_progress,
            cancellation_event=cancellation_event,
        )
        return ProviderBatchRowResult(
            row_id=row.row_id,
            status=result.status,
            record_id=result.record_id,
            record_dir=result.record_dir,
            run_id=(
                result.run_id
                if result.run_id is not None
                else result.workspace_dir.name
                if result.workspace_dir is not None
                else None
            ),
            workspace_dir=result.workspace_dir,
            warnings=result.warnings,
        )

    return run_provider_batch(
        csv_path=resolved_csv,
        rows=rows,
        repo_root=resolved_repo,
        execute_row=execute_row,
        provider=resolve_provider_endpoint(resolved_repo),
        failure_context=generation_failure_context,
        model=model,
        reasoning=reasoning,
        runtime_by_row=runtime_by_row,
        concurrency=concurrency,
        progress=progress,
    )


__all__ = [
    "GENERATION_CSV_COLUMNS",
    "GENERATION_DEADLINE",
    "MAX_PROVIDER_BATCH_CONCURRENCY",
    "ProviderBatchInterrupted",
    "ProviderBatchResult",
    "ProviderBatchRow",
    "ProviderBatchRowResult",
    "execute_provider_batch",
    "load_provider_batch",
    "run_generation_row",
]
