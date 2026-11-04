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

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from storage.artifact_contract import RECONSTRUCTION_SEMANTIC_PROVENANCE_ROLE
from usdcraft.pipeline.contracts import CodexRuntimeSelection
from usdcraft.pipeline.hashing import sha256_file
from usdcraft.reconstruction.axial import AGENT_CORE_TEXT_MAX_BYTES
from usdcraft.reconstruction.default_prompt import resolve_reconstruction_prompt
from usdcraft.reconstruction.harness_policy import ReconstructionHarnessPolicy
from usdcraft.values import normalize_reasoning_level

RECONSTRUCTION_SCHEMA_VERSION = 1
RECONSTRUCTION_PIPELINE = "reconstruction"
RECONSTRUCTION_VALIDATION_PROFILE = "reconstruction"
# The semantic image is always the reconstruction target's semantic record; its relationship to
# the source mesh is neither declared nor distinguished.
SEMANTIC_PROVENANCE_ROLE = RECONSTRUCTION_SEMANTIC_PROVENANCE_ROLE
# Bound to the encoder's own fuse (core ceiling plus the face-derived sections' allowance) so the
# Host gate can never drift from the evidence policy.
AXIAL_HARD_MAX_BYTES = AGENT_CORE_TEXT_MAX_BYTES

_SIGNED_AXES = frozenset({"+X", "-X", "+Y", "-Y", "+Z", "-Z"})


def _nonempty(value: object, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty.")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{label} must not contain control characters.")
    return normalized


def _input_file(value: Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


@dataclass(slots=True, frozen=True)
class ReconstructionInputIdentity:
    path: Path
    sha256: str
    byte_count: int

    @classmethod
    def from_path(cls, path: Path) -> ReconstructionInputIdentity:
        resolved = _input_file(path, label="Reconstruction input")
        return cls(
            path=resolved,
            sha256=sha256_file(resolved),
            byte_count=resolved.stat().st_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }


@dataclass(slots=True, frozen=True)
class ReconstructionRequest:
    """One formal Reconstruction request before deterministic evidence preprocessing.

    Both arms run exactly one Author per request. Time and turn limits are Provider-only: the
    Provider arm records its per-request HTTP deadline (``timeout_seconds``) and its Author turn
    budget (``max_turns``); the Codex arm (``turn_budget=False``) has neither.
    """

    repo_root: Path
    source_path: Path
    semantic_image_path: Path
    up_axis: str | None = None
    front_axis: str | None = None
    meters_per_unit: float | None = None
    # Provenance of ``meters_per_unit``: ``None`` for a user declaration or the format contract,
    # ``estimated_from_semantic_image`` when the Provider arm set it from a category estimate of a
    # unit-normalized mesh; ``scale_estimate`` then carries that estimate for the record.
    meters_per_unit_source: str | None = None
    scale_estimate: Mapping[str, Any] | None = None
    model: str | None = None
    reasoning_effort: str = "high"
    runtime_source: str = "unreported"
    selection_source: str = "unreported"
    # Provider arm only: the deadline in seconds for each Provider HTTP request. Always None on the
    # Codex arm, which has no time limit; a None value is omitted from the request record.
    timeout_seconds: float | None = None
    label: str | None = None
    guidance: str | None = None
    mesh_cleanup: bool = False
    # Request text: what to reconstruct. Explicit user text, or the deterministic default that
    # names the primary subject. Highest authority for the subject's
    # category and stated must-haves; never for measured geometry.
    prompt: str | None = None
    image_observation: bool = True
    max_turns: int | None = None
    # Whether the arm enforces an Author turn budget: True for the Provider arm (``max_turns``,
    # default 80), False for the Codex arm, which records no turn budget and rejects ``max_turns``.
    turn_budget: bool = True

    def __post_init__(self) -> None:
        _ = self.harness_policy  # Validate capability inputs before staging.
        repo_root = Path(self.repo_root).expanduser().resolve()
        if not repo_root.is_dir():
            raise ValueError(f"USDCraft repository root is not a directory: {repo_root}")
        source_path = _input_file(self.source_path, label="Reconstruction source")
        semantic_image_path = _input_file(
            self.semantic_image_path,
            label="Reconstruction semantic image",
        )
        up_axis = str(self.up_axis).strip().upper() if self.up_axis is not None else None
        front_axis = str(self.front_axis).strip().upper() if self.front_axis is not None else None
        if up_axis is not None and up_axis not in _SIGNED_AXES:
            raise ValueError("up_axis must be one of +X/-X/+Y/-Y/+Z/-Z.")
        if front_axis is not None and front_axis not in _SIGNED_AXES:
            raise ValueError("front_axis must be one of +X/-X/+Y/-Y/+Z/-Z.")
        if up_axis is not None and front_axis is not None and up_axis[-1] == front_axis[-1]:
            raise ValueError("up_axis and front_axis must name different cardinal axes.")

        meters_per_unit: float | None = None
        if self.meters_per_unit is not None:
            try:
                meters_per_unit = float(self.meters_per_unit)
            except (TypeError, ValueError) as exc:
                raise ValueError("meters_per_unit must be a positive number.") from exc
            if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
                raise ValueError("meters_per_unit must be a positive finite number.")

        model = str(self.model).strip() if self.model is not None else None
        if model == "":
            model = None
        reasoning_effort = normalize_reasoning_level(
            _nonempty(self.reasoning_effort, label="Codex reasoning effort")
        )
        timeout_seconds: float | None = None
        if self.timeout_seconds is not None:
            try:
                timeout_seconds = float(self.timeout_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError("Provider request timeout must be a positive number.") from exc
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError("Provider request timeout must be a positive finite number.")
        label = str(self.label).strip() if self.label is not None else None
        if label == "":
            label = None
        guidance = str(self.guidance).strip() if self.guidance is not None else None
        if guidance == "":
            guidance = None
        if guidance is not None:
            _nonempty(guidance, label="User Reconstruction guidance")
        if not isinstance(self.mesh_cleanup, bool):
            raise ValueError("mesh_cleanup must be a boolean.")
        prompt = resolve_reconstruction_prompt(prompt=self.prompt)

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "semantic_image_path", semantic_image_path)
        object.__setattr__(self, "up_axis", up_axis)
        object.__setattr__(self, "front_axis", front_axis)
        object.__setattr__(self, "meters_per_unit", meters_per_unit)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "reasoning_effort", reasoning_effort)
        object.__setattr__(
            self,
            "runtime_source",
            _nonempty(self.runtime_source, label="Codex runtime source"),
        )
        object.__setattr__(
            self,
            "selection_source",
            _nonempty(self.selection_source, label="Codex selection source"),
        )
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "guidance", guidance)

    @property
    def harness_policy(self) -> ReconstructionHarnessPolicy:
        from usdcraft.reconstruction.harness_policy import resolve_harness_policy

        return resolve_harness_policy(
            image_observation=self.image_observation,
            max_turns=self.max_turns,
            turn_budget=self.turn_budget,
        )

    @property
    def runtime(self) -> CodexRuntimeSelection:
        if self.model is None:
            raise ValueError("Reconstruction execution requires a resolved Codex model.")
        return CodexRuntimeSelection(
            model_id=self.model,
            reasoning_effort=self.reasoning_effort,
            runtime_source=self.runtime_source,
            selection_source=self.selection_source,
        )

    @property
    def object_name(self) -> str:
        """Display name: the user's label, else the request text (as Generation does)."""

        value = " ".join((self.label or self.prompt or "").split())
        if len(value) > 120:
            value = value[:117].rstrip() + "..."
        return value or "Reconstruction"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
            "pipeline": RECONSTRUCTION_PIPELINE,
            "source": {
                "path": str(self.source_path),
                "role": "surface_geometry_input",
                "up_axis": self.up_axis,
                "front_axis": self.front_axis,
                "meters_per_unit": self.meters_per_unit,
                "meters_per_unit_source": self.meters_per_unit_source,
                "scale_estimate": dict(self.scale_estimate) if self.scale_estimate else None,
            },
            "semantic_image": {
                "path": str(self.semantic_image_path),
                "role": "semantic_subject_evidence",
                "provenance_role": SEMANTIC_PROVENANCE_ROLE,
            },
            "request_text": self.prompt,
            "harness": self.harness_policy.to_dict(),
            "codex": {
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "runtime_source": self.runtime_source,
                "selection_source": self.selection_source,
                **(
                    {"timeout_seconds": self.timeout_seconds}
                    if self.timeout_seconds is not None
                    else {}
                ),
            },
            "mesh_cleanup": {
                "requested": self.mesh_cleanup,
                "role": "optional_subject_isolation_agent_for_source_mesh_artifact_cleanup",
            },
            "label": self.label,
            "user_guidance": self.guidance,
        }


@dataclass(slots=True, frozen=True)
class ReconstructionPreflight:
    request: ReconstructionRequest
    source_dependencies: tuple[ReconstructionInputIdentity, ...]
    semantic_image: ReconstructionInputIdentity
    scaffold: ReconstructionInputIdentity
    codex_executable: str
    python_executable: str
    keep_workspace: bool
    fingerprint: str

    @property
    def workspace_base(self) -> Path:
        return self.request.repo_root / "data" / "local" / "reconstruction_runs"

    @property
    def author_workspace_base(self) -> Path:
        return self.request.repo_root / "data" / "local" / "reconstruction_author_runs"

    @property
    def records_root(self) -> Path:
        return self.request.repo_root / "data" / "records"

    @property
    def materialization_root(self) -> Path:
        return self.request.repo_root / "data" / "cache" / "record_materialization"

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
            "pipeline": RECONSTRUCTION_PIPELINE,
            "request": self.request.to_dict(),
            "source_dependencies": [item.to_dict() for item in self.source_dependencies],
            "semantic_image": self.semantic_image.to_dict(),
            "execution": {
                "scaffold": self.scaffold.to_dict(),
                "codex_executable": self.codex_executable,
                "python_executable": self.python_executable,
                "keep_workspace": self.keep_workspace,
                "workspace_base": str(self.workspace_base),
                "author_workspace_base": str(self.author_workspace_base),
                "records_root": str(self.records_root),
                "materialization_root": str(self.materialization_root),
            },
            "author_evidence_contract": {
                "attached_images": ["semantic_image"],
                "attached_image_count": 1,
                "axial": "AXIAL surface-only",
                "source_views": {
                    "initial": 0,
                    "on_demand": True,
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "awaiting_approval",
            "reconstruction_preflight": True,
            "record_created": False,
            "workspace_created": False,
            "isolation_agent_started": False,
            "author_agent_started": False,
            "confirmation_required": True,
            "fingerprint": self.fingerprint,
            "case": self.fingerprint_payload(),
        }


@dataclass(slots=True, frozen=True)
class ReconstructionEvidenceBundle:
    """Frozen upstream evidence; only ``author_*`` artifacts enter the Author workspace."""

    semantic_image_path: Path
    scaffold_path: Path
    axial_path: Path
    original_source_path: Path
    original_source_dependencies: tuple[Path, ...]
    normalized_source_path: Path
    source_manifest_path: Path
    subject_isolation_manifest_path: Path
    subject_isolation_binding_path: Path
    mesh_preprocess_manifest_path: Path
    source_overlay_manifest_path: Path
    source_overlay_urdf_path: Path
    source_overlay_glb_path: Path
    original_source_tree_root: Path | None = None
    approved_inputs_manifest_path: Path | None = None
    approved_inputs_manifest_sha256: str | None = None
    approved_inputs_manifest_byte_count: int | None = None
    source_overlay_dependency_paths: tuple[Path, ...] = ()
    voxel_evidence_path: Path | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "semantic_image_path",
            "scaffold_path",
            "axial_path",
            "original_source_path",
            "normalized_source_path",
            "source_manifest_path",
            "subject_isolation_manifest_path",
            "subject_isolation_binding_path",
            "mesh_preprocess_manifest_path",
            "source_overlay_manifest_path",
            "source_overlay_urdf_path",
            "source_overlay_glb_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _input_file(getattr(self, field_name), label=field_name),
            )
        source_dependencies = tuple(
            _input_file(path, label="original_source_dependency")
            for path in self.original_source_dependencies
        )
        if self.original_source_path not in source_dependencies:
            raise ValueError(
                "original_source_dependencies must include the primary original_source_path."
            )
        if len(set(source_dependencies)) != len(source_dependencies):
            raise ValueError("original_source_dependencies must not contain duplicates.")
        object.__setattr__(self, "original_source_dependencies", source_dependencies)
        frozen_declarations = (
            self.original_source_tree_root,
            self.approved_inputs_manifest_path,
            self.approved_inputs_manifest_sha256,
            self.approved_inputs_manifest_byte_count,
        )
        if any(value is not None for value in frozen_declarations) and not all(
            value is not None for value in frozen_declarations
        ):
            raise ValueError(
                "Frozen Reconstruction evidence must declare source tree root, approved-input "
                "manifest, and its hash and size together."
            )
        if self.original_source_tree_root is not None:
            tree_root = Path(self.original_source_tree_root).expanduser().resolve()
            if not tree_root.is_dir():
                raise ValueError(f"original_source_tree_root is not a directory: {tree_root}")
            for path in source_dependencies:
                try:
                    path.relative_to(tree_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Frozen original source dependency escapes its tree root: {path}"
                    ) from exc
            object.__setattr__(self, "original_source_tree_root", tree_root)
        if self.approved_inputs_manifest_path is not None:
            manifest_path = _input_file(
                self.approved_inputs_manifest_path,
                label="approved_inputs_manifest_path",
            )
            object.__setattr__(self, "approved_inputs_manifest_path", manifest_path)
            manifest_sha256 = str(self.approved_inputs_manifest_sha256)
            if not (
                len(manifest_sha256) == 64
                and all(character in "0123456789abcdef" for character in manifest_sha256)
            ):
                raise ValueError("approved_inputs_manifest_sha256 must be a lowercase sha256.")
            object.__setattr__(self, "approved_inputs_manifest_sha256", manifest_sha256)
            manifest_byte_count = self.approved_inputs_manifest_byte_count
            if (
                isinstance(manifest_byte_count, bool)
                or not isinstance(manifest_byte_count, int)
                or manifest_byte_count < 0
            ):
                raise ValueError("approved_inputs_manifest_byte_count must be non-negative.")
            if (
                sha256_file(manifest_path) != manifest_sha256
                or manifest_path.stat().st_size != manifest_byte_count
            ):
                raise ValueError("Approved Reconstruction input manifest identity changed.")
        overlay_dependencies = tuple(
            _input_file(path, label="source_overlay_dependency")
            for path in self.source_overlay_dependency_paths
        )
        if len(set(overlay_dependencies)) != len(overlay_dependencies):
            raise ValueError("source_overlay_dependency_paths must not contain duplicates.")
        object.__setattr__(self, "source_overlay_dependency_paths", overlay_dependencies)
        if self.voxel_evidence_path is not None:
            object.__setattr__(
                self,
                "voxel_evidence_path",
                _input_file(self.voxel_evidence_path, label="voxel_evidence_path"),
            )
        axial_bytes = self.axial_path.read_bytes()
        if not axial_bytes.startswith(b"AXIAL\n"):
            raise ValueError("Reconstruction requires AXIAL evidence.")
        if len(axial_bytes) > AXIAL_HARD_MAX_BYTES:
            raise ValueError(f"AXIAL exceeds the {AXIAL_HARD_MAX_BYTES // 1024} KiB hard maximum.")


__all__ = [
    "AXIAL_HARD_MAX_BYTES",
    "RECONSTRUCTION_PIPELINE",
    "RECONSTRUCTION_SCHEMA_VERSION",
    "RECONSTRUCTION_VALIDATION_PROFILE",
    "ReconstructionEvidenceBundle",
    "ReconstructionInputIdentity",
    "ReconstructionPreflight",
    "ReconstructionRequest",
    "SEMANTIC_PROVENANCE_ROLE",
]
