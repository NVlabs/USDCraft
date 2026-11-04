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

from typing import Any, Literal

from pydantic import BaseModel, Field

from storage.artifact_contract import MaterializationDisplayStatus
from viewer.api.run_settings import (
    GuidanceStatus,
    MeshCleanupStatus,
    ProviderApiStyle,
    RequestPromptKind,
)

# The Viewer list projection may defer the full canonical contract check; ``not_validated``
# marks a row whose displayability is a cheap, hash-free hint rather than a validated status.
# It is never produced by ``storage.artifact_contract`` and never claims ``available``.
ListMaterializationStatus = MaterializationDisplayStatus | Literal["not_validated"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    repo_root: str | None = None
    # True when the Viewer was started with `--read-only`: rating and deletion routes answer 403
    # and the frontend hides their controls.
    read_only: bool = False


class MaterializationIssueResponse(BaseModel):
    path: str
    message: str


class RecordSummaryResponse(BaseModel):
    record_id: str
    pipeline: Literal["generation"] = "generation"
    status: Literal["generation_ready"] | None = None
    input_kind: Literal["text", "text_image"] | None = None
    title: str
    prompt_preview: str
    rating: int | None = Field(default=None, ge=1, le=5)
    created_at: str | None = None
    updated_at: str | None = None
    viewer_asset_updated_at: str | None = None
    provider: str | None = None
    model_id: str | None = None
    agent_harness: Literal["codex", "provider"] | None = None
    # Run settings: the selection source (`runtime_source`), the Provider wire protocol (None on
    # the Codex arm) and how the inspiration image reached the Author.
    runtime_source: str | None = None
    api_style: ProviderApiStyle | None = None
    image_observation: str | None = None
    has_traces: bool = False
    thinking_level: str | None = None
    turn_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_cost_usd: float | None = None
    active_revision_id: str | None = None
    revision_count: int = 0
    materialization_status: ListMaterializationStatus = "missing"
    materialization_displayable: bool = False
    materialization_validated: bool = True
    materialization_issues: list[MaterializationIssueResponse] = Field(default_factory=list)
    has_compile_report: bool = False
    has_provenance: bool = False
    has_cost: bool = False
    # Batch membership when a batch command produced the record (None for single-item runs).
    batch_id: str | None = None
    batch_row_id: str | None = None


class WorkbenchEntryResponse(BaseModel):
    record_id: str
    added_at: str
    label: str | None = None
    tags: list[str] = Field(default_factory=list)
    archived: bool = False
    record: RecordSummaryResponse | None = None


class ViewerBootstrapResponse(BaseModel):
    repo_root: str
    generated_at: str
    workbench_entries: list[WorkbenchEntryResponse]


class RecordDetailResponse(BaseModel):
    summary: RecordSummaryResponse
    record: dict[str, Any]
    compile_report: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    cost: dict[str, Any] | None = None


class RecordHistoryRevisionResponse(BaseModel):
    record_id: str
    revision_id: str
    active: bool = False
    created_at: str | None = None
    prompt_preview: str = ""
    provider: str | None = None
    model_id: str | None = None
    thinking_level: str | None = None
    status: str | None = None
    has_cost: bool = False
    has_traces: bool = False
    has_model: bool = False
    has_provenance: bool = False


class RecordHistoryResponse(BaseModel):
    record_id: str
    active_revision_id: str | None = None
    revisions: list[RecordHistoryRevisionResponse] = Field(default_factory=list)


class RecordInputImageResponse(BaseModel):
    """Path relative to the active record file endpoint, if inspiration was supplied."""

    path: str | None = None


class RecordRatingRequest(BaseModel):
    rating: int | None = Field(strict=True, ge=1, le=5)


class RecordDeleteRequest(BaseModel):
    confirm_record_id: str = Field(min_length=1)


class RecordDeleteResponse(BaseModel):
    status: Literal["deleted"] = "deleted"
    record_id: str
    materialization_deleted: bool
    cleanup_pending: bool = Field(
        description=(
            "Canonical deletion is committed, but physical tombstone cleanup must be retried."
        )
    )


class RecordTextFileResponse(BaseModel):
    record_id: str
    file_path: str
    content: str
    truncated: bool = False
    byte_count: int
    preview_byte_limit: int | None = None


class AxialDiagnosticResponse(BaseModel):
    """Record-level readings of the Author's last `check_axial` call (mechanical, never a gate)."""

    binding_status: Literal["current", "stale", "unverified"] = "unverified"
    calls: int = Field(ge=1)
    revision: int | None = None
    first_recall: float | None = None
    recall: float | None = None
    input_cells: int | None = None
    matched_cells: int | None = None
    missing_cells: int | None = None
    candidate_only_cells: int | None = None
    exterior_excess_cells: int | None = None
    exterior_excess_fraction: float | None = None
    surface_fit_p50_mm: float | None = None
    surface_fit_p90_mm: float | None = None
    surface_faces_off_plane: int | None = None
    candidate_openings: int | None = None
    unmodelled_source_openings: int | None = None


class AuthorTimelineEventResponse(BaseModel):
    """One Author turn of a Provider run, reduced from the run's tool-completion events."""

    turn: int
    tool: str
    ok: bool
    revision: int | None = None
    tool_elapsed_seconds: float | None = None
    response_elapsed_seconds: float | None = None
    recall: float | None = None


class AuthorTimelineResponse(BaseModel):
    record_id: str
    source: Literal["provider_events"]
    events: list[AuthorTimelineEventResponse] = Field(default_factory=list)


class AxialExcessSliceResponse(BaseModel):
    stack_slice: int
    side: str
    excess_cells: int
    bounds_min_xyz_m: list[float]
    bounds_max_xyz_m: list[float]


class AxialGeometryResponse(BaseModel):
    """Geometry of the last `check_axial` call for the viewport overlay (recon_world metres, Z up)."""

    record_id: str
    source: Literal["provider_events"]
    grid_bounds_min_xyz_m: list[float]
    grid_bounds_max_xyz_m: list[float]
    stack_pitch_m: float | None = None
    excess_slices: list[AxialExcessSliceResponse] = Field(default_factory=list)


class ReconstructionRecordSummaryResponse(BaseModel):
    record_id: str
    pipeline: Literal["reconstruction"] = "reconstruction"
    status: Literal["reconstruction_ready"] | None = None
    input_kind: Literal["mesh_image", "mesh_image_text"] | None = None
    title: str
    prompt_preview: str = ""
    rating: int | None = Field(default=None, ge=1, le=5)
    created_at: str | None = None
    updated_at: str | None = None
    viewer_asset_updated_at: str | None = None
    provider: str | None = None
    model_id: str | None = None
    agent_harness: Literal["codex", "provider"] | None = None
    reasoning_effort: str | None = None
    thinking_level: str | None = None
    turn_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_cost_usd: float | None = None
    active_revision_id: str | None = None
    revision_count: int = Field(default=0, ge=0)
    input_count: int = Field(default=0, ge=0)
    source_available: bool = False
    candidate_available: bool = False
    # Run settings: selection source, Provider wire protocol (None on the Codex arm), the image
    # observation mode, the enforced Provider turn budget (None on the Codex arm, which has no turn
    # limit) and whether the optional Subject-Isolation mesh cleanup ran.
    runtime_source: str | None = None
    api_style: ProviderApiStyle | None = None
    image_observation: str | None = None
    max_turns: int | None = None
    mesh_cleanup: MeshCleanupStatus | None = None
    duration_seconds: float | None = None
    # How the run's metres were decided
    # (`declared`, `format_contract`, `embedded_metadata` or `estimated_from_semantic_image`).
    scale_source: str | None = None
    scale_meters_per_unit: float | None = None
    scale_estimate_category: str | None = None
    scale_estimate_longest_m: float | None = None
    # Batch membership when a batch command produced the record (None for single-item runs).
    batch_id: str | None = None
    batch_row_id: str | None = None
    axial_diagnostic: AxialDiagnosticResponse | None = None
    materialization_status: ListMaterializationStatus = "missing"
    materialization_displayable: bool = False
    materialization_validated: bool = True
    materialization_issues: list[MaterializationIssueResponse] = Field(default_factory=list)
    has_compile_report: bool = False
    has_provenance: bool = False
    has_cost: bool = False
    has_traces: bool = False


class ReconstructionRecordEvidenceResponse(BaseModel):
    role: str
    name: str
    label: str
    relative_path: str
    file_url: str
    media_kind: Literal["image", "mesh", "text", "json", "binary"]
    byte_count: int = Field(ge=0)
    sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReconstructionRecordAssetResponse(BaseModel):
    available: bool
    label: str
    file_url: str | None = None
    base_file_url: str | None = None
    format: str | None = None
    revision_key: str | None = None


class ReconstructionRecordDetailResponse(BaseModel):
    summary: ReconstructionRecordSummaryResponse
    source: ReconstructionRecordAssetResponse
    candidate: ReconstructionRecordAssetResponse
    inputs: list[ReconstructionRecordEvidenceResponse] = Field(default_factory=list)
    record: dict[str, Any]
    reconstruction_manifest: dict[str, Any] | None = None
    evidence_manifest: dict[str, Any] | None = None
    request_text: str | None = None
    # `default` when `request_text` is the Host's default request, else `custom`.
    request_prompt: RequestPromptKind | None = None
    # User guidance recorded in `reconstruction.json` `input.guidance` (`unrecorded` without it).
    guidance_status: GuidanceStatus = "unrecorded"
    guidance: str | None = None
    provenance: dict[str, Any] | None = None
    compile_report: dict[str, Any] | None = None
