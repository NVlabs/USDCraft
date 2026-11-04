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

"""Deterministic mesh-preprocess stage: normalized source, AXIAL evidence, source carrier.

For Reconstruction, `run_mesh_preprocess` is the one Host stage that turns the frozen,
isolation-resolved source mesh into every deterministic mesh-derived artifact the pipeline
consumes: normalized retained geometry, the AXIAL surface-evidence package, the voxel
sidecar and a same-frame source carrier for Viewer and on-demand rendering. The stage reads no image
semantically and starts no Agent; it contains no SDK rebuild, attempt selection or canonical
record state, which belong to later pipeline stages.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from usdcraft.pipeline.hashing import sha256_file as _sha256
from usdcraft.reconstruction.axial import (
    AGENT_CORE_BYTE_BUDGET,
    build_axial_observation,
    format_axial,
    format_voxel_artifact,
)
from usdcraft.reconstruction.source import (
    NormalizedSourceArtifacts,
    NormalizedSourceScene,
    SourceInput,
    load_normalized_source,
    write_normalized_source,
)
from usdcraft.reconstruction.source_overlay import SourceOverlayArtifacts, prepare_source_overlay

MESH_PREPROCESS_MANIFEST_SCHEMA = "reconstruction_mesh_preprocess_case"


class ObservationCaseError(RuntimeError):
    """The requested Phase 1 case cannot be created without overwriting or partial evidence."""


@dataclass(frozen=True, slots=True)
class ObservationCaseArtifacts:
    root: Path
    normalized_source: Path
    source_manifest: Path
    subject_isolation: Path
    axial: Path
    voxel_evidence: Path
    case_manifest: Path


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_observation_artifacts(
    *,
    root: Path,
    scene: NormalizedSourceScene,
    source: NormalizedSourceArtifacts,
    subject_isolation: Path,
    isolation_decision: object,
) -> tuple[Path, Path, Path]:
    observation = build_axial_observation(scene)
    observation_dir = root / "axial"
    observation_dir.mkdir(parents=True, exist_ok=False)
    axial_path = observation_dir / "axial.txt"
    voxel_path = observation_dir / "axial_voxels.json"
    manifest_path = root / "case_manifest.json"

    axial_path.write_text(format_axial(observation), encoding="utf-8")
    voxel_path.write_text(
        json.dumps(format_voxel_artifact(observation), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    components = tuple(getattr(scene, "components"))
    classifications = observation.classifications
    evidence_counts = {
        "mesh_sampled_surface": sum(
            item.source_kind == "mesh" and item.evidence_kind == "surface_hit_band"
            for item in classifications
        ),
        "point_sampled_surface": sum(item.source_kind == "points" for item in classifications),
    }
    manifest = {
        "schema": MESH_PREPROCESS_MANIFEST_SCHEMA,
        "status": "ready",
        "scope": "reconstruction_mesh_preprocessing",
        "canonical_record": False,
        "agent_run": False,
        "agent_run_scope": "this_mesh_preprocess_transaction_only",
        "preprocessing": {
            "subject_isolation": {
                "required": True,
                "status": "ready",
                "order": "before_normalized_source_axial_and_source_overlay",
                "upstream_agent_run": bool(
                    getattr(isolation_decision, "upstream_agent_run", False)
                ),
                "upstream_agent_context": getattr(
                    isolation_decision, "upstream_agent_context", None
                ),
                "source_identity_sha256": getattr(isolation_decision, "source_identity_sha256"),
                "retained_component_count": len(
                    getattr(isolation_decision, "retained_component_ids")
                ),
                "excluded_component_count": len(
                    getattr(isolation_decision, "excluded_component_ids")
                ),
                "unresolved_component_count": len(
                    getattr(isolation_decision, "unresolved_component_ids")
                ),
                "same_retained_scene_for_all_downstream_evidence": True,
            }
        },
        "frame": {
            "coordinate_system": "recon_world",
            "up_axis": "+Z",
            "meters_per_unit": 1.0,
            "semantic_front_resolved": getattr(scene.frame, "front_axis") is not None,
        },
        "geometry": {
            "component_count": len(components),
            "mesh_component_count": sum(item.kind == "mesh" for item in components),
            "point_component_count": sum(item.kind == "points" for item in components),
            "bounds_min_xyz_m": [float(value) for value in scene.bounds_min_xyz],
            "bounds_max_xyz_m": [float(value) for value in scene.bounds_max_xyz],
        },
        "axial": {
            "format": "AXIAL",
            "agent_core_byte_count": observation.agent_core_byte_count,
            "agent_core_max_byte_budget": AGENT_CORE_BYTE_BUDGET,
            "agent_core_selection_byte_budget": observation.selection_core_byte_budget,
            "agent_core_selection_reason": observation.selection_reason,
            "stack_longest_axis_cells": observation.stack.longest_axis_cells,
            "stack_axis": "XYZ"[observation.stack.stack_axis],
            "stack_slice_plane": ("YZ", "XZ", "XY")[observation.stack.stack_axis],
            "evidence_counts": evidence_counts,
            "geometry_authority": "surface_only",
            "interior_semantics": "unknown",
            "solid_authority": "none",
            "empty_cell_semantics": "unknown_not_free_space",
            "projected_negative_evidence": "none",
            "micro_slice_bounds": {
                "format": "MICRO_SLICE_BOUNDS_Z192",
                "selection": observation.micro_slice_bounds.selection,
                "contoured_island_count": (observation.micro_slice_bounds.contoured_island_count),
                "section_bytes": observation.micro_slice_bounds.section_bytes,
                "candidate_bytes": observation.micro_slice_bounds.candidate_bytes,
                "stack_slice_count": observation.micro_slice_bounds.stack_slice_count,
                "encoded_slice_count": observation.micro_slice_bounds.encoded_slice_count,
                "island_count": observation.micro_slice_bounds.island_count,
                "connectivity": 8,
            },
            "face_features": {
                "format": "FACE_FEATURES",
                "shown_feature_count": len(observation.face_features.features),
                "pitch_m": observation.face_features.pitch_m,
                "section_bytes": observation.face_features.section_bytes,
            },
            "member_tracks": {
                "format": "MEMBER_TRACKS192",
                "shown_track_count": len(observation.member_tracks.tracks),
                "detected_track_count": observation.member_tracks.detected_track_count,
                "section_bytes": observation.member_tracks.section_bytes,
            },
        },
        "artifacts": {
            "normalized_source": {
                "path": _relative(root, source.normalized_usdc),
                "sha256": source.normalized_usdc_sha256,
                "agent_input": False,
            },
            "source_manifest": {
                "path": _relative(root, source.source_manifest),
                "sha256": source.source_manifest_sha256,
                "agent_input": False,
            },
            "subject_isolation": {
                "path": _relative(root, subject_isolation),
                "sha256": _sha256(subject_isolation),
                "agent_input": False,
                "role": "frozen_preprocessing_decision_binding",
            },
            "axial": {
                "path": _relative(root, axial_path),
                "sha256": _sha256(axial_path),
                "agent_input": False,
                "future_policy": "required_geometry_evidence",
            },
            "voxel_evidence": {
                "path": _relative(root, voxel_path),
                "sha256": _sha256(voxel_path),
                "agent_input": False,
                "role": "viewer_and_posthoc_diagnostics",
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return axial_path, voxel_path, manifest_path


def create_observation_case(
    source_input: SourceInput,
    output_dir: Path,
    *,
    subject_isolation: Path,
) -> ObservationCaseArtifacts:
    """Create one atomic evidence case and publish it only after every check succeeds.

    This is the case half of the mesh-preprocess stage: the normalized retained source, the frozen
    isolation binding, the AXIAL package with its voxel sidecar and the case manifest. It renders
    nothing; source camera images are produced only by later Author requests.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ObservationCaseError(f"output already exists; refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent))
    )
    try:
        from usdcraft.reconstruction.isolation import (
            apply_subject_isolation,
            load_subject_isolation_decision,
        )

        raw_scene = load_normalized_source(source_input)
        isolation_decision = load_subject_isolation_decision(subject_isolation, raw_scene)
        scene = apply_subject_isolation(raw_scene, isolation_decision)
        source = write_normalized_source(scene, staging / "source")
        isolation_binding = staging / "source" / "subject_isolation.json"
        isolation_binding.write_text(
            json.dumps(
                {
                    "schema": "reconstruction_subject_isolation_binding",
                    "status": "ready",
                    "source_identity_sha256": isolation_decision.source_identity_sha256,
                    "upstream_manifest_sha256": _sha256(isolation_decision.manifest_path),
                    "upstream_agent_run": isolation_decision.upstream_agent_run,
                    "upstream_agent_context": isolation_decision.upstream_agent_context,
                    "retained_component_ids": list(isolation_decision.retained_component_ids),
                    "excluded_component_ids": list(isolation_decision.excluded_component_ids),
                    "unresolved_component_ids": list(isolation_decision.unresolved_component_ids),
                    "retained_samples_unchanged": True,
                    "reconstruction_agent_input": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        axial, voxel, case_manifest = _write_observation_artifacts(
            root=staging,
            scene=scene,
            source=source,
            subject_isolation=isolation_binding,
            isolation_decision=isolation_decision,
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return ObservationCaseArtifacts(
        root=output_dir,
        normalized_source=output_dir / _relative(staging, source.normalized_usdc),
        source_manifest=output_dir / _relative(staging, source.source_manifest),
        subject_isolation=output_dir / _relative(staging, isolation_binding),
        axial=output_dir / _relative(staging, axial),
        voxel_evidence=output_dir / _relative(staging, voxel),
        case_manifest=output_dir / _relative(staging, case_manifest),
    )


@dataclass(frozen=True, slots=True)
class MeshPreprocessResult:
    """Every deterministic mesh-derived artifact of one preprocessing run."""

    case: ObservationCaseArtifacts
    source_overlay: SourceOverlayArtifacts


def run_mesh_preprocess(
    source_input: SourceInput,
    upstream_dir: Path,
    *,
    subject_isolation: Path,
) -> MeshPreprocessResult:
    """Run the complete deterministic mesh-preprocess stage under ``upstream_dir``.

    Creates the atomic evidence case at ``upstream_dir / "mesh_preprocess"`` (normalized source,
    isolation binding, AXIAL package, voxel sidecar, case manifest) and a geometry-only Viewer
    carrier at ``upstream_dir / "source_overlay"``, both from the
    same frozen isolation decision applied to the same reloaded source.
    """

    from usdcraft.reconstruction.isolation import (
        apply_subject_isolation,
        load_subject_isolation_decision,
    )

    case = create_observation_case(
        source_input,
        upstream_dir / "mesh_preprocess",
        subject_isolation=subject_isolation,
    )
    # Reload the approved raw source and apply the exact frozen decision so native materials,
    # textures and vertex colors remain available to on-demand source rendering. The same
    # deterministic retained components produced the case and its AXIAL package above.
    raw_scene = load_normalized_source(source_input)
    decision = load_subject_isolation_decision(subject_isolation, raw_scene)
    retained_scene = apply_subject_isolation(raw_scene, decision)
    normalized_artifacts = NormalizedSourceArtifacts(
        root=case.normalized_source.parent,
        normalized_usdc=case.normalized_source,
        source_manifest=case.source_manifest,
        normalized_usdc_sha256=_sha256(case.normalized_source),
        source_manifest_sha256=_sha256(case.source_manifest),
    )
    source_overlay = prepare_source_overlay(
        retained_scene,
        normalized_artifacts,
        upstream_dir / "source_overlay",
    )
    return MeshPreprocessResult(case=case, source_overlay=source_overlay)


__all__ = [
    "MESH_PREPROCESS_MANIFEST_SCHEMA",
    "MeshPreprocessResult",
    "ObservationCaseArtifacts",
    "ObservationCaseError",
    "create_observation_case",
    "run_mesh_preprocess",
]
