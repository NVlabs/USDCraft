# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from fastapi import HTTPException

from storage.artifact_contract import (
    COMPILE_REPORT_SCHEMA_VERSION,
    PHYSX_USDC_ARTIFACT,
    USDZ_ARTIFACT,
    MaterializationDisplayResult,
    classify_materialization_displayability,
)
from storage.identifiers import validate_record_id
from storage.repo import StorageRepo
from storage.revisions import (
    active_cost_path,
    active_inputs_dir,
    active_model_path,
    active_prompt_path,
    active_provenance_path,
    active_revision_id,
    active_traces_dir,
    validate_revision_id,
)
from viewer.api.media import trace_media_type


class ViewerRecordStore(Protocol):
    repo: StorageRepo


_DISPLAYABLE_INVALID_EXACT_PATHS = {
    ("model.urdf",),
    Path(USDZ_ARTIFACT).parts,
    Path(PHYSX_USDC_ARTIFACT).parts,
}
_DISPLAYABLE_INVALID_PREFIXES = {
    ("assets",),
    ("renders",),
    ("isaac", "textures"),
}


def is_materialized_viewer_path_allowed(
    path: Path,
    *,
    display: MaterializationDisplayResult,
    report: Mapping[str, Any] | None,
) -> bool:
    """Apply the shared narrow read-only allowlist for one Viewer materialization path."""

    if not display.displayable:
        return False
    if display.status == "contract_invalid":
        if path.parts in _DISPLAYABLE_INVALID_EXACT_PATHS:
            return True
        return any(
            len(path.parts) > len(prefix) and path.parts[: len(prefix)] == prefix
            for prefix in _DISPLAYABLE_INVALID_PREFIXES
        )
    if report is None:
        return False
    if report.get("schema_version") != COMPILE_REPORT_SCHEMA_VERSION:
        return False
    hashes = report.get("artifacts")
    return isinstance(hashes, Mapping) and path.as_posix() in hashes


class ViewerFileResolver:
    """Resolve only canonical, already-existing record artifacts.

    Resolution never hydrates, compiles, copies, or otherwise changes repository state.
    """

    def __init__(self, store: ViewerRecordStore) -> None:
        self.repo = store.repo

    def resolve_record_target(self, record_id: str, file_path: str) -> tuple[Path, Path]:
        record = self._record(record_id)
        requested = self._validated_relative_path(file_path)
        record_dir = self.repo.layout.record_dir(record_id).resolve()
        materialization_dir = self.repo.layout.record_materialization_dir(record_id).resolve()

        active_files = {
            ("prompt.txt",): active_prompt_path(self.repo, record_id, record=record),
            ("model.py",): active_model_path(self.repo, record_id, record=record),
            ("provenance.json",): active_provenance_path(self.repo, record_id, record=record),
            ("cost.json",): active_cost_path(self.repo, record_id, record=record),
        }
        if requested.parts == ("record.json",):
            root = record_dir
            target = self.repo.layout.record_metadata_path(record_id).resolve()
        elif requested.parts == ("generation.json",):
            revision_id = active_revision_id(self.repo, record_id, record=record)
            root = self.repo.layout.record_revision_dir(record_id, revision_id).resolve()
            target = (root / "generation.json").resolve()
        elif requested.parts in active_files:
            target = active_files[requested.parts].resolve()
            root = target.parent
        elif len(requested.parts) >= 2 and requested.parts[0] == "inputs":
            root = active_inputs_dir(self.repo, record_id, record=record).resolve()
            target = (root / Path(*requested.parts[1:])).resolve()
        elif self._is_materialized_path(
            record_id,
            record,
            requested,
            materialization_dir,
        ):
            root = materialization_dir
            target = (materialization_dir / requested).resolve()
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        self._ensure_within_root(target, root)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        return root, target

    def resolve_record_trace_target(self, record_id: str, file_path: str) -> tuple[Path, str]:
        record = self._record(record_id)
        creator = record.get("creator")
        if isinstance(creator, dict) and creator.get("trace_available") is False:
            raise HTTPException(status_code=404, detail=f"Trace file not found: {file_path}")
        root = active_traces_dir(self.repo, record_id, record=record).resolve()
        target = (root / self._validated_relative_path(file_path)).resolve()
        self._ensure_within_root(target, root)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"Trace file not found: {file_path}")
        return target, trace_media_type(target)

    def resolve_record_revision_target(
        self,
        record_id: str,
        revision_id: str,
        file_path: str,
    ) -> tuple[Path, Path]:
        self._record(record_id)
        try:
            revision_id = validate_revision_id(revision_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid revision ID format") from exc
        root = self.repo.layout.record_revision_dir(record_id, revision_id).resolve()
        if not root.is_dir():
            raise HTTPException(status_code=404, detail=f"Revision not found: {revision_id}")
        target = (root / self._validated_relative_path(file_path)).resolve()
        self._ensure_within_root(target, root)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        return root, target

    def resolve_record_revision_trace_target(
        self,
        record_id: str,
        revision_id: str,
        file_path: str,
    ) -> tuple[Path, str]:
        record = self._record(record_id)
        creator = record.get("creator")
        if isinstance(creator, dict) and creator.get("trace_available") is False:
            raise HTTPException(status_code=404, detail=f"Trace file not found: {file_path}")
        try:
            revision_id = validate_revision_id(revision_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid revision ID format") from exc
        root = self.repo.layout.record_revision_traces_dir(record_id, revision_id).resolve()
        target = (root / self._validated_relative_path(file_path)).resolve()
        self._ensure_within_root(target, root)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"Trace file not found: {file_path}")
        return target, trace_media_type(target)

    def _record(self, record_id: str) -> dict[str, Any]:
        try:
            validate_record_id(record_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid record ID format") from exc
        try:
            record = self.repo.read_json(self.repo.layout.record_metadata_path(record_id))
        except (OSError, ValueError):
            # A corrupt record.json must resolve to not-found, never a 500.
            record = None
        if (
            not isinstance(record, dict)
            or str(record.get("pipeline") or "generation") != "generation"
        ):
            raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
        return record

    def _is_materialized_path(
        self,
        record_id: str,
        record: dict[str, Any],
        path: Path,
        materialization: Path,
    ) -> bool:
        if path.parts == ("compile_report.json",):
            return True
        revision_id = active_revision_id(self.repo, record_id, record=record)
        revision_dir = self.repo.layout.record_revision_dir(record_id, revision_id)
        generation_path = self.repo.layout.record_revision_generation_path(
            record_id,
            revision_id,
        )
        display = classify_materialization_displayability(
            materialization,
            generation_path,
            revision_dir=revision_dir,
        )
        if not display.displayable:
            return False
        try:
            report = self.repo.read_json(materialization / "compile_report.json")
        except (OSError, ValueError):
            report = None
        return is_materialized_viewer_path_allowed(
            path,
            display=display,
            report=report if isinstance(report, Mapping) else None,
        )

    @staticmethod
    def _validated_relative_path(file_path: str) -> Path:
        requested = Path(file_path)
        if not file_path or requested.is_absolute() or ".." in requested.parts:
            raise HTTPException(status_code=400, detail="Invalid file path")
        return requested

    @staticmethod
    def _ensure_within_root(target: Path, root: Path) -> None:
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Access denied") from exc
