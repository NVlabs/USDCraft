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

"""Shared mechanical completion and promotion for both Reconstruction transports."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

from usdcraft.compile.command import CompileArtifacts
from usdcraft.pipeline.sdk_context import PRELOADED_SDK_DOC_PATHS
from usdcraft.reconstruction.axial_summary import summarize_axial_payloads
from usdcraft.reconstruction.contract import ReconstructionEvidenceBundle, ReconstructionPreflight
from usdcraft.reconstruction.diagnostic_binding import diagnostic_binding
from usdcraft.reconstruction.observation import ImageObservationArtifacts
from usdcraft.reconstruction.persist import (
    ReconstructionAuthoringIdentity,
    ReconstructionEvidence,
    RuntimeSelection,
)
from usdcraft.reconstruction.workspace import (
    ReconstructionAuthorWorkspace,
    ReconstructionRunWorkspace,
)
from usdcraft.stagetime import STAGE_TRACE_FILENAME, StageTimer


def author_runtime_metadata(
    *,
    preflight: ReconstructionPreflight,
    workspace: ReconstructionAuthorWorkspace,
    evidence: ReconstructionEvidenceBundle,
    image_observation: ImageObservationArtifacts | None,
    timer: StageTimer,
    axial_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    policy = preflight.request.harness_policy
    return {
        "stage_timings": timer.report(),
        # The Provider harness records its enforced ``max_turns``; the Codex arm has no turn
        # budget and records none.
        "harness": policy.to_dict(),
        "preloaded_docs": list(PRELOADED_SDK_DOC_PATHS),
        "image_observation": "external_inventory"
        if image_observation is not None
        else "author_only"
        if policy.image_observation
        else "disabled",
        "axial_diagnostic": summarize_axial_payloads(
            axial_payloads,
            final_binding=diagnostic_binding(
                evidence.normalized_source_path, workspace.build_dir / "isaac/model.usdc"
            ),
        ),
    }


def promote_author_output(
    *,
    preflight: ReconstructionPreflight,
    run_workspace: ReconstructionRunWorkspace,
    workspace: ReconstructionAuthorWorkspace,
    evidence: ReconstructionEvidenceBundle,
    image_observation: ImageObservationArtifacts | None,
    runtime: RuntimeSelection,
    runtime_report: Mapping[str, Any],
    artifacts: CompileArtifacts,
    authoring: ReconstructionAuthoringIdentity,
    timer: StageTimer,
    persist_runner: Callable[..., Path],
) -> Path:
    """Archive the same evidence and stage traces before the same canonical promotion."""
    run_trace = run_workspace.root / STAGE_TRACE_FILENAME
    if run_trace.is_file():
        shutil.copy2(run_trace, workspace.root / STAGE_TRACE_FILENAME)
    if image_observation is not None:
        shutil.copy2(image_observation.inventory_path, workspace.root / "image_observation.json")
        shutil.copy2(
            image_observation.manifest_path, workspace.root / "image_observation_manifest.json"
        )
    canonical_evidence = ReconstructionEvidence.from_bundle(
        evidence,
        author_evidence_manifest_path=workspace.context_manifest_path,
        original_source_manifest_path=run_workspace.frozen_inputs.manifest_path,
    )
    with timer.stage("persist"):
        return persist_runner(
            repo_root=preflight.request.repo_root,
            run_id=run_workspace.run_id,
            title=preflight.request.object_name,
            request_text=preflight.request.prompt,
            runtime=runtime,
            workspace_dir=workspace.root,
            runtime_report=runtime_report,
            artifacts=artifacts,
            evidence=canonical_evidence,
            label=preflight.request.label,
            guidance=preflight.request.guidance,
            authoring=authoring,
        )


def cleanup_completed_workspaces(
    run_workspace: ReconstructionRunWorkspace,
    workspace: ReconstructionAuthorWorkspace,
    *,
    keep_workspace: bool,
    warnings: list[str],
) -> Path | None:
    if keep_workspace:
        return workspace.root
    errors = []
    for path in (run_workspace.root, workspace.root):
        try:
            shutil.rmtree(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        warnings.append(
            "Promotion succeeded but Reconstruction workspaces could not be removed: "
            + "; ".join(errors)
        )
    return workspace.root if workspace.root.exists() else None
