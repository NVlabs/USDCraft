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

"""Reconstruction Batch: the CSV row contract and row executor for `usdcraft reconstruction-batch`.

Scheduling, failure isolation, progress events and the Ctrl-C snapshot protocol live in the shared
`usdcraft/pipeline/batch.py` coordinator; this module contributes only the Reconstruction CSV
columns and the call into the same direct primitive used by `usdcraft reconstruction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Mapping

from usdcraft.pipeline.batch import (
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
from usdcraft.pipeline.provider.client import resolve_provider_endpoint
from usdcraft.pipeline.provider.progress import ProviderProgressCallback
from usdcraft.reconstruction.command import reconstruction_failure_context
from usdcraft.reconstruction.harness_policy import resolve_harness_policy

_REQUIRED_COLUMNS = frozenset({"row_id", "source_path", "image_path"})
# The complete Reconstruction CSV contract, in the order the unknown-column error lists it.
RECONSTRUCTION_CSV_COLUMNS = (
    "row_id",
    "source_path",
    "image_path",
    "prompt",
    "guidance",
    "label",
    "mesh_cleanup",
    "image_observation",
    "max_turns",
    "up_axis",
    "front_axis",
    "meters_per_unit",
    "model",
    "reasoning",
    "api_style",
)


@dataclass(slots=True, frozen=True)
class ReconstructionBatchRow:
    row_id: str
    source_path: Path
    image_path: Path
    prompt: str | None
    guidance: str | None
    label: str | None
    image_observation: bool
    max_turns: int | None
    up_axis: str | None
    front_axis: str | None
    meters_per_unit: float | None
    model: str | None = None
    reasoning: str | None = None
    api_style: str | None = None
    # Optional Subject-Isolation Agent (both arms); empty or omitted keeps the retain-all default.
    mesh_cleanup: bool = False


def load_reconstruction_batch(csv_path: Path) -> tuple[ReconstructionBatchRow, ...]:
    """Validate a `reconstruction-batch` CSV without touching a source, image or workspace.

    ``source_path`` and ``image_path`` are required and resolve relative to the CSV directory; the
    source closure and image decodability are checked by the primitive when the row runs, exactly
    as `usdcraft reconstruction` checks them.
    """

    resolved_csv = Path(csv_path).expanduser().resolve()
    rows: list[ReconstructionBatchRow] = []
    for location, raw_row, row_id in iter_batch_csv(
        resolved_csv, required_columns=_REQUIRED_COLUMNS
    ):
        source_value = optional_text(raw_row.get("source_path"))
        if source_value is None:
            raise ValueError(f"Provider batch {location} has an empty source_path.")
        image_value = optional_text(raw_row.get("image_path"))
        if image_value is None:
            raise ValueError(f"Provider batch {location} has an empty image_path.")
        reject_unknown_columns(location, raw_row, allowed_columns=RECONSTRUCTION_CSV_COLUMNS)

        def boolean(name: str, default: bool) -> bool:
            value = optional_text(raw_row.get(name))
            if value is None:
                return default
            if value.lower() not in {"true", "false", "1", "0"}:
                raise ValueError(f"Provider batch {location}: {name} must be true/false or 1/0.")
            return value.lower() in {"true", "1"}

        runtime_override = row_runtime_override(location, raw_row)
        try:
            max_turns = (
                int(raw_row["max_turns"]) if optional_text(raw_row.get("max_turns")) else None
            )
        except ValueError as exc:
            raise ValueError(f"Provider batch {location}: max_turns must be an integer.") from exc
        options: dict[str, Any] = {
            "image_observation": boolean("image_observation", True),
            "max_turns": max_turns,
        }
        try:
            resolve_harness_policy(**options)
        except ValueError as exc:
            raise ValueError(f"Provider batch {location}: {exc}") from exc
        meters_value = optional_text(raw_row.get("meters_per_unit"))
        meters_per_unit: float | None = None
        if meters_value is not None:
            try:
                meters_per_unit = float(meters_value)
            except ValueError as exc:
                raise ValueError(
                    f"Provider batch {location}: meters_per_unit must be a number, "
                    f"got {meters_value!r}."
                ) from exc
        rows.append(
            ReconstructionBatchRow(
                row_id=row_id,
                source_path=resolve_csv_relative_path(source_value, csv_dir=resolved_csv.parent),
                image_path=resolve_csv_relative_path(image_value, csv_dir=resolved_csv.parent),
                prompt=optional_text(raw_row.get("prompt")),
                guidance=optional_text(raw_row.get("guidance")),
                label=optional_text(raw_row.get("label")),
                **options,
                up_axis=optional_text(raw_row.get("up_axis")),
                front_axis=optional_text(raw_row.get("front_axis")),
                meters_per_unit=meters_per_unit,
                model=runtime_override.model,
                reasoning=runtime_override.reasoning,
                api_style=runtime_override.api_style,
                mesh_cleanup=boolean("mesh_cleanup", False),
            )
        )
    return tuple(rows)


def execute_reconstruction_batch(
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
    """Run each CSV row through the same direct Provider Reconstruction primitive."""

    resolved_csv = Path(csv_path).expanduser().resolve()
    resolved_repo = Path(repo_root).expanduser().resolve()
    rows = load_reconstruction_batch(resolved_csv)
    from usdcraft.reconstruction.provider_deadline import validate_run_timeout

    validate_run_timeout(run_timeout_seconds)
    runtime_by_row = resolve_provider_row_runtimes(
        resolved_repo, rows, model=model, reasoning=reasoning
    )

    def execute_row(
        row: ReconstructionBatchRow,
        row_progress: ProviderProgressCallback,
        cancellation_event: Event,
        batch_context: Mapping[str, Any],
    ) -> ProviderBatchRowResult:
        # Imported inside the row boundary so a broken optional runtime fails one item.
        from usdcraft.reconstruction.provider_command import execute_provider_reconstruction

        arguments: dict[str, Any] = dict(
            repo_root=resolved_repo,
            source=row.source_path,
            semantic_image=row.image_path,
            up_axis=row.up_axis,
            front_axis=row.front_axis,
            meters_per_unit=row.meters_per_unit,
            guidance=row.guidance,
            prompt=row.prompt,
            label=row.label,
            model=runtime_by_row[row.row_id][0] if runtime_by_row else model,
            reasoning=runtime_by_row[row.row_id][1] if runtime_by_row else reasoning,
            timeout_seconds=timeout_seconds,
            keep_workspace=keep_workspace,
            image_observation=row.image_observation,
            max_turns=row.max_turns,
            mesh_cleanup=row.mesh_cleanup,
            batch_context=batch_context,
        )
        if row.api_style is not None or api_style != "responses":
            arguments["api_style"] = row.api_style or api_style
        if run_timeout_seconds is not None:
            from usdcraft.reconstruction.provider_deadline import execute_with_deadline

            return execute_with_deadline(
                arguments,
                row_id=row.row_id,
                timeout_seconds=run_timeout_seconds,
                progress=row_progress,
                cancellation_event=cancellation_event,
            )
        result = execute_provider_reconstruction(
            **arguments, progress=row_progress, cancellation_event=cancellation_event
        )
        return ProviderBatchRowResult(
            row_id=row.row_id,
            status=result.status,
            record_id=result.record_id,
            record_dir=result.record_dir,
            run_id=result.run_id,
            workspace_dir=result.workspace_dir,
            warnings=result.warnings,
        )

    return run_provider_batch(
        csv_path=resolved_csv,
        rows=rows,
        repo_root=resolved_repo,
        execute_row=execute_row,
        provider=resolve_provider_endpoint(resolved_repo),
        failure_context=reconstruction_failure_context,
        model=model,
        reasoning=reasoning,
        runtime_by_row=runtime_by_row,
        concurrency=concurrency,
        progress=progress,
    )


__all__ = [
    "RECONSTRUCTION_CSV_COLUMNS",
    "ReconstructionBatchRow",
    "execute_reconstruction_batch",
    "load_reconstruction_batch",
]
