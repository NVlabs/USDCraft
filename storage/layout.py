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

from dataclasses import dataclass
from pathlib import Path

from storage.artifact_contract import (
    NEWTON_USDC_ARTIFACT,
)


@dataclass(slots=True, frozen=True)
class StorageLayout:
    """Canonical and local paths shared by Generation, Reconstruction, and Viewer."""

    root: Path

    @property
    def data_root(self) -> Path:
        return self.root / "data"

    @property
    def records_root(self) -> Path:
        return self.data_root / "records"

    @property
    def local_root(self) -> Path:
        return self.data_root / "local"

    @property
    def generation_runs_root(self) -> Path:
        return self.local_root / "generation_runs"

    @property
    def promotion_staging_root(self) -> Path:
        return self.local_root / "promotion_staging"

    @property
    def materialization_replacement_staging_root(self) -> Path:
        return self.local_root / "materialization_replacement_staging"

    def materialization_replacement_staging_dir(self, record_id: str) -> Path:
        return self.materialization_replacement_staging_root / record_id

    @property
    def record_locks_root(self) -> Path:
        return self.local_root / "locks" / "records"

    @property
    def record_deletion_staging_root(self) -> Path:
        return self.local_root / "record_deletion_staging"

    def record_deletion_staging_dir(self, record_id: str) -> Path:
        return self.record_deletion_staging_root / record_id

    def record_lock_path(self, record_id: str) -> Path:
        return self.record_locks_root / f"{record_id}.lock"

    @property
    def cache_root(self) -> Path:
        return self.data_root / "cache"

    @property
    def record_materializations_root(self) -> Path:
        return self.cache_root / "record_materialization"

    def record_dir(self, record_id: str) -> Path:
        return self.records_root / record_id

    def record_metadata_path(self, record_id: str) -> Path:
        return self.record_dir(record_id) / "record.json"

    def record_collections_dir(self, record_id: str) -> Path:
        return self.record_dir(record_id) / "collections"

    def record_workbench_entry_path(self, record_id: str) -> Path:
        return self.record_collections_dir(record_id) / "workbench.json"

    def record_revisions_dir(self, record_id: str) -> Path:
        return self.record_dir(record_id) / "revisions"

    def record_revision_dir(self, record_id: str, revision_id: str) -> Path:
        return self.record_revisions_dir(record_id) / revision_id

    def record_revision_metadata_path(self, record_id: str, revision_id: str) -> Path:
        return self.record_revision_dir(record_id, revision_id) / "revision.json"

    def record_revision_prompt_path(self, record_id: str, revision_id: str) -> Path:
        return self.record_revision_dir(record_id, revision_id) / "prompt.txt"

    def record_revision_model_path(self, record_id: str, revision_id: str) -> Path:
        return self.record_revision_dir(record_id, revision_id) / "model.py"

    def record_revision_provenance_path(self, record_id: str, revision_id: str) -> Path:
        return self.record_revision_dir(record_id, revision_id) / "provenance.json"

    def record_revision_inputs_dir(self, record_id: str, revision_id: str) -> Path:
        return self.record_revision_dir(record_id, revision_id) / "inputs"

    def record_revision_traces_dir(self, record_id: str, revision_id: str) -> Path:
        return self.record_revision_dir(record_id, revision_id) / "traces"

    def record_revision_generation_path(self, record_id: str, revision_id: str) -> Path:
        return self.record_revision_dir(record_id, revision_id) / "generation.json"

    def record_revision_reconstruction_path(self, record_id: str, revision_id: str) -> Path:
        return self.record_revision_dir(record_id, revision_id) / "reconstruction.json"

    def generation_run_dir(self, run_id: str) -> Path:
        return self.generation_runs_root / run_id

    def record_materialization_dir(self, record_id: str) -> Path:
        return self.record_materializations_root / record_id

    def record_materialization_urdf_path(self, record_id: str) -> Path:
        return self.record_materialization_dir(record_id) / "model.urdf"

    def record_materialization_newton_usdc_path(self, record_id: str) -> Path:
        return self.record_materialization_dir(record_id) / NEWTON_USDC_ARTIFACT

    def record_materialization_compile_report_path(self, record_id: str) -> Path:
        return self.record_materialization_dir(record_id) / "compile_report.json"

    def record_materialization_renders_dir(self, record_id: str) -> Path:
        return self.record_materialization_dir(record_id) / "renders"

    def ensure_base_dirs(self) -> None:
        for path in (
            self.records_root,
            self.generation_runs_root,
            self.promotion_staging_root,
            self.materialization_replacement_staging_root,
            self.record_materializations_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
