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

import json
import os
import platform
import re
import shutil
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, Mapping, Protocol

from storage.artifact_contract import (
    GENERATION_VALIDATION_PROFILE,
    RECONSTRUCTION_APPROVED_INPUTS_SCHEMA,
    RECONSTRUCTION_INPUT_MANIFEST_SCHEMA,
    RECONSTRUCTION_INPUT_MANIFEST_SCHEMA_VERSION,
    RECONSTRUCTION_MANIFEST_SCHEMA_VERSION,
    RECONSTRUCTION_VALIDATION_PROFILE,
    physics_output_contract,
    validate_reconstruction_materialization_contract,
)
from storage.canonical_promotion import (
    CanonicalPromotion,
    begin_canonical_promotion,
    commit_canonical_promotion,
    finish_canonical_promotion,
)
from storage.record_locks import record_lock
from storage.record_mutations import write_promotion_manifest
from storage.repo import StorageRepo
from usdcraft.compile.command import CompileArtifacts
from usdcraft.pipeline.files import _format_utc, _git_commit, _utc_now, _write_json
from usdcraft.pipeline.hashing import sha256_file, sha256_text
from usdcraft.reconstruction.contract import (
    AXIAL_HARD_MAX_BYTES,
    SEMANTIC_PROVENANCE_ROLE,
    ReconstructionEvidenceBundle,
)

REVISION_ID = "rev_000001"
APPROVED_INPUTS_SCHEMA = RECONSTRUCTION_APPROVED_INPUTS_SCHEMA


class RuntimeSelection(Protocol):
    def to_dict(self) -> dict[str, str]: ...


@dataclass(slots=True, frozen=True)
class ReconstructionAuthoringIdentity:
    """Identify the Reconstruction authoring arm without changing the shared record contract.

    ``trace_files`` maps workspace-relative trace sources to their archived ``traces/`` names;
    ``required_trace_files`` names the subset that must exist for a completed run.
    """

    agent_harness: Literal["codex", "provider"]
    # The Codex arm is `codex-cli`; the Provider arm records its endpoint (`nvidia` or `openai`).
    provider: Literal["codex-cli", "nvidia", "openai"]
    trace_prefix: str
    trace_files: Mapping[str, str]
    required_trace_files: tuple[str, ...]

    @property
    def task_trace_name(self) -> str:
        return f"{self.trace_prefix}_task.md"


CODEX_RECONSTRUCTION_AUTHORING = ReconstructionAuthoringIdentity(
    agent_harness="codex",
    provider="codex-cli",
    trace_prefix="codex",
    trace_files={
        ".codex-run/stdout.jsonl": "codex_events.jsonl",
        ".codex-run/final.json": "codex_final.json",
        ".codex-run/stderr.log": "codex_stderr.log",
        ".codex-run/timeline.jsonl": "codex_timeline.jsonl",
        ".codex-run/trace_errors.log": "codex_trace_errors.log",
    },
    required_trace_files=(
        ".codex-run/stdout.jsonl",
        ".codex-run/final.json",
        ".codex-run/stderr.log",
    ),
)
NVIDIA_PROVIDER_RECONSTRUCTION_AUTHORING = ReconstructionAuthoringIdentity(
    agent_harness="provider",
    provider="nvidia",
    trace_prefix="provider",
    trace_files={
        ".provider-run/events.jsonl": "provider_events.jsonl",
        ".provider-run/final.json": "provider_final.json",
        ".provider-run/stderr.log": "provider_stderr.log",
        ".provider-run/timeline.jsonl": "provider_timeline.jsonl",
        ".provider-run/trace_errors.log": "provider_trace_errors.log",
    },
    required_trace_files=(
        ".provider-run/events.jsonl",
        ".provider-run/final.json",
        ".provider-run/stderr.log",
    ),
)


def provider_reconstruction_authoring(
    endpoint: Literal["nvidia", "openai"],
) -> ReconstructionAuthoringIdentity:
    """The Provider-arm authoring identity for the endpoint that served the run."""

    if endpoint not in {"nvidia", "openai"}:
        raise ValueError("Provider endpoint must be nvidia or openai.")
    if endpoint == NVIDIA_PROVIDER_RECONSTRUCTION_AUTHORING.provider:
        return NVIDIA_PROVIDER_RECONSTRUCTION_AUTHORING
    return replace(NVIDIA_PROVIDER_RECONSTRUCTION_AUTHORING, provider=endpoint)


def _input_file(value: Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _safe_relative_path(value: str, *, label: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    path = Path(normalized)
    windows_path = PureWindowsPath(normalized)
    if (
        not normalized
        or normalized == "."
        or path.is_absolute()
        or windows_path.is_absolute()
        or ".." in path.parts
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", normalized)
    ):
        raise ValueError(f"{label} must be a safe relative path.")
    return path.as_posix()


@dataclass(slots=True, frozen=True)
class ReconstructionSourceDependency:
    path: Path
    relative_path: str
    source_logical_path: str | None = None
    approved_original_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _input_file(self.path, label="Reconstruction source dependency"),
        )
        object.__setattr__(
            self,
            "relative_path",
            _safe_relative_path(
                self.relative_path,
                label="Reconstruction dependency relative_path",
            ),
        )
        if self.source_logical_path is not None:
            logical_path = str(self.source_logical_path).strip().replace("\\", "/")
            if not logical_path:
                raise ValueError("source_logical_path must not be empty when provided.")
            object.__setattr__(self, "source_logical_path", logical_path)
        if self.approved_original_path is not None:
            approved_original_path = str(self.approved_original_path).strip()
            if not approved_original_path:
                raise ValueError("approved_original_path must not be empty when provided.")
            object.__setattr__(self, "approved_original_path", approved_original_path)


def _approved_manifest_payload(
    manifest_path: Path,
    *,
    required: bool,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Approved Reconstruction input manifest is invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Approved Reconstruction input manifest must be a JSON object.")
    if payload.get("schema") != APPROVED_INPUTS_SCHEMA:
        if required:
            raise ValueError(
                f"Approved Reconstruction input manifest must use {APPROVED_INPUTS_SCHEMA}."
            )
        return None
    if payload.get("schema_version") != 1 or payload.get("pipeline") != "reconstruction":
        raise ValueError("Approved Reconstruction input manifest has an invalid contract.")
    fingerprint = payload.get("approved_fingerprint")
    if not (
        isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("Approved Reconstruction input manifest fingerprint is invalid.")
    return payload


def _approved_file_identity(payload: Mapping[str, Any], *, label: str) -> tuple[str, int]:
    sha256 = payload.get("sha256")
    byte_count = payload.get("byte_count")
    if not (
        isinstance(sha256, str)
        and len(sha256) == 64
        and all(character in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(f"Approved Reconstruction {label} sha256 is invalid.")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError(f"Approved Reconstruction {label} byte_count is invalid.")
    return sha256, byte_count


def _validate_approved_snapshot(
    manifest_path: Path,
    *,
    source_files: Mapping[str, Path],
    entry_relative_path: str,
    semantic_image_path: Path,
    scaffold_path: Path,
    required: bool,
) -> dict[str, str]:
    """Bind one approved-input manifest to the exact frozen files used for publication."""

    payload = _approved_manifest_payload(manifest_path, required=required)
    if payload is None:
        return {}
    source_payload: Any = payload.get("source")
    raw_files = source_payload.get("files") if isinstance(source_payload, Mapping) else None
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("Approved Reconstruction source manifest files must be a non-empty array.")

    manifest_entries: dict[str, Mapping[str, Any]] = {}
    entry_paths: list[str] = []
    logical_paths: dict[str, str] = {}
    seen_frozen_paths: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, Mapping):
            raise ValueError(f"Approved Reconstruction source file {index} must be an object.")
        relative_path = _safe_relative_path(
            str(raw_file.get("tree_relative_path") or ""),
            label=f"Approved source file {index} tree_relative_path",
        )
        frozen_path = _safe_relative_path(
            str(raw_file.get("frozen_path") or ""),
            label=f"Approved source file {index} frozen_path",
        )
        if frozen_path != f"source/tree/{relative_path}":
            raise ValueError(
                "Approved Reconstruction source frozen_path must preserve the source tree."
            )
        if relative_path in manifest_entries or frozen_path in seen_frozen_paths:
            raise ValueError("Approved Reconstruction source manifest contains duplicate files.")
        if not isinstance(raw_file.get("entry"), bool):
            raise ValueError("Approved Reconstruction source entry flags must be booleans.")
        original_path = raw_file.get("original_path")
        if not isinstance(original_path, str) or not original_path.strip():
            raise ValueError("Approved Reconstruction source original_path must be non-empty.")
        _approved_file_identity(raw_file, label=f"source file {relative_path}")
        manifest_entries[relative_path] = raw_file
        seen_frozen_paths.add(frozen_path)
        logical_paths[relative_path] = original_path
        if raw_file["entry"]:
            entry_paths.append(relative_path)

    declared_entry = source_payload.get("entry_tree_relative_path")
    if entry_paths != [entry_relative_path] or declared_entry != entry_relative_path:
        raise ValueError("Approved Reconstruction source manifest does not bind its source entry.")
    if set(manifest_entries) != set(source_files):
        raise ValueError(
            "Approved Reconstruction source manifest does not bind the complete source closure."
        )

    manifest_root = manifest_path.parent.resolve()
    for relative_path, source_path in source_files.items():
        raw_file = manifest_entries[relative_path]
        expected_path = (manifest_root / str(raw_file["frozen_path"])).resolve()
        if source_path.resolve() != expected_path:
            raise ValueError(
                f"Approved Reconstruction source path differs from its snapshot: {relative_path}"
            )
        sha256, byte_count = _approved_file_identity(
            raw_file,
            label=f"source file {relative_path}",
        )
        if sha256_file(source_path) != sha256 or source_path.stat().st_size != byte_count:
            raise ValueError(
                f"Approved Reconstruction source bytes differ from its snapshot: {relative_path}"
            )

    for key, expected_path in (
        ("semantic_image", semantic_image_path),
        ("scaffold", scaffold_path),
    ):
        raw_file = payload.get(key)
        if not isinstance(raw_file, Mapping):
            raise ValueError(f"Approved Reconstruction {key} binding must be an object.")
        frozen_path = _safe_relative_path(
            str(raw_file.get("frozen_path") or ""),
            label=f"Approved {key} frozen_path",
        )
        if (manifest_root / frozen_path).resolve() != expected_path.resolve():
            raise ValueError(f"Approved Reconstruction {key} path differs from its snapshot.")
        sha256, byte_count = _approved_file_identity(raw_file, label=key)
        if sha256_file(expected_path) != sha256 or expected_path.stat().st_size != byte_count:
            raise ValueError(f"Approved Reconstruction {key} bytes differ from its snapshot.")
    return logical_paths


@dataclass(slots=True, frozen=True)
class ReconstructionEvidence:
    """Complete canonical evidence, including inputs withheld from the Author."""

    original_source_path: Path
    original_source_dependencies: tuple[ReconstructionSourceDependency, ...]
    normalized_source_path: Path
    source_manifest_path: Path
    subject_isolation_binding_path: Path
    subject_isolation_manifest_path: Path
    mesh_preprocess_manifest_path: Path
    semantic_image_path: Path
    scaffold_path: Path
    axial_path: Path
    source_overlay_manifest_path: Path
    author_evidence_manifest_path: Path
    source_overlay_urdf_path: Path
    source_overlay_glb_path: Path
    original_source_manifest_path: Path | None = None
    original_source_relative_path: str | None = None
    approved_original_source_path: str | None = None
    approved_inputs_manifest_sha256: str | None = None
    approved_inputs_manifest_byte_count: int | None = None
    source_overlay_dependencies: tuple[ReconstructionSourceDependency, ...] = ()
    voxel_sidecar_path: Path | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "original_source_path",
            "normalized_source_path",
            "source_manifest_path",
            "subject_isolation_binding_path",
            "subject_isolation_manifest_path",
            "mesh_preprocess_manifest_path",
            "semantic_image_path",
            "scaffold_path",
            "axial_path",
            "source_overlay_manifest_path",
            "author_evidence_manifest_path",
            "source_overlay_urdf_path",
            "source_overlay_glb_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _input_file(getattr(self, field_name), label=field_name),
            )
        for field_name in ("original_source_manifest_path", "voxel_sidecar_path"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _input_file(value, label=field_name))
        if (self.approved_inputs_manifest_sha256 is None) != (
            self.approved_inputs_manifest_byte_count is None
        ):
            raise ValueError("Approved-input manifest hash and size must be declared together.")
        if self.approved_inputs_manifest_sha256 is not None:
            manifest_sha256 = str(self.approved_inputs_manifest_sha256)
            manifest_byte_count = self.approved_inputs_manifest_byte_count
            if not (
                len(manifest_sha256) == 64
                and all(character in "0123456789abcdef" for character in manifest_sha256)
            ):
                raise ValueError("approved_inputs_manifest_sha256 must be a lowercase sha256.")
            if (
                isinstance(manifest_byte_count, bool)
                or not isinstance(manifest_byte_count, int)
                or manifest_byte_count < 0
            ):
                raise ValueError("approved_inputs_manifest_byte_count must be non-negative.")
            object.__setattr__(self, "approved_inputs_manifest_sha256", manifest_sha256)

        original_source_relative_path = _safe_relative_path(
            self.original_source_relative_path or self.original_source_path.name,
            label="Reconstruction original source relative_path",
        )
        object.__setattr__(
            self,
            "original_source_relative_path",
            original_source_relative_path,
        )
        if self.approved_original_source_path is not None:
            approved_original_source_path = str(self.approved_original_source_path).strip()
            if not approved_original_source_path:
                raise ValueError("approved_original_source_path must not be empty when provided.")
            object.__setattr__(
                self,
                "approved_original_source_path",
                approved_original_source_path,
            )

        source_dependencies = tuple(self.original_source_dependencies)
        overlay_dependencies = tuple(self.source_overlay_dependencies)
        for dependencies, label in (
            (source_dependencies, "original source"),
            (overlay_dependencies, "source overlay"),
        ):
            if not all(isinstance(item, ReconstructionSourceDependency) for item in dependencies):
                raise TypeError(f"Every {label} dependency must be ReconstructionSourceDependency.")
            relative_paths = [item.relative_path for item in dependencies]
            if len(set(relative_paths)) != len(relative_paths):
                raise ValueError(f"{label} dependency relative paths must be unique.")
        if original_source_relative_path in {item.relative_path for item in source_dependencies}:
            raise ValueError("Original source dependency collides with the source entry path.")
        if {"model.urdf", "assets/source.glb"} & {
            item.relative_path for item in overlay_dependencies
        }:
            raise ValueError("Source overlay dependency collides with the carrier entry files.")
        object.__setattr__(self, "original_source_dependencies", source_dependencies)
        object.__setattr__(self, "source_overlay_dependencies", overlay_dependencies)

        axial_bytes = self.axial_path.read_bytes()
        if not axial_bytes.startswith(b"AXIAL\n"):
            raise ValueError("Reconstruction requires AXIAL evidence.")
        if len(axial_bytes) > AXIAL_HARD_MAX_BYTES:
            raise ValueError(f"AXIAL exceeds the {AXIAL_HARD_MAX_BYTES // 1024} KiB hard maximum.")

    @classmethod
    def from_bundle(
        cls,
        bundle: ReconstructionEvidenceBundle,
        *,
        author_evidence_manifest_path: Path,
        original_source_manifest_path: Path | None = None,
    ) -> ReconstructionEvidence:
        """Adapt the execution bundle while preserving dependency-relative layouts."""

        original_root = bundle.original_source_tree_root or bundle.original_source_path.parent
        original_source_relative_path = bundle.original_source_path.relative_to(
            original_root
        ).as_posix()
        if (
            bundle.approved_inputs_manifest_path is not None
            and original_source_manifest_path is not None
            and original_source_manifest_path.resolve()
            != bundle.approved_inputs_manifest_path.resolve()
        ):
            raise ValueError("Canonical evidence must use the frozen approved-input manifest.")
        approved_manifest = original_source_manifest_path or bundle.approved_inputs_manifest_path
        approved_original_paths: dict[str, str] = {}
        if approved_manifest is not None:
            source_files: dict[str, Path] = {}
            for path in bundle.original_source_dependencies:
                try:
                    relative_path = path.relative_to(original_root).as_posix()
                except ValueError:
                    logical_path = os.path.relpath(path, original_root).replace("\\", "/")
                    relative_path = "_external/" + sha256_text(logical_path)[:16] + "/" + path.name
                source_files[relative_path] = path
            approved_original_paths = _validate_approved_snapshot(
                approved_manifest,
                source_files=source_files,
                entry_relative_path=original_source_relative_path,
                semantic_image_path=bundle.semantic_image_path,
                scaffold_path=bundle.scaffold_path,
                required=bundle.approved_inputs_manifest_path is not None,
            )
        approved_entry_path = approved_original_paths.get(original_source_relative_path)
        approved_entry_parent = (
            Path(approved_entry_path).parent if approved_entry_path is not None else None
        )
        source_dependencies: list[ReconstructionSourceDependency] = []
        for dependency in bundle.original_source_dependencies:
            if dependency == bundle.original_source_path:
                continue
            logical_path = os.path.relpath(dependency, original_root).replace("\\", "/")
            try:
                relative_path = dependency.relative_to(original_root).as_posix()
            except ValueError:
                relative_path = (
                    "_external/" + sha256_text(logical_path)[:16] + "/" + dependency.name
                )
            source_dependencies.append(
                ReconstructionSourceDependency(
                    path=dependency,
                    relative_path=relative_path,
                    source_logical_path=(
                        os.path.relpath(
                            approved_original_paths[relative_path],
                            approved_entry_parent,
                        ).replace("\\", "/")
                        if approved_entry_parent is not None
                        and relative_path in approved_original_paths
                        else logical_path
                    ),
                    approved_original_path=approved_original_paths.get(relative_path),
                )
            )

        overlay_root = bundle.source_overlay_urdf_path.parent
        overlay_dependencies: list[ReconstructionSourceDependency] = []
        for dependency in bundle.source_overlay_dependency_paths:
            if dependency in {
                bundle.source_overlay_urdf_path,
                bundle.source_overlay_glb_path,
            }:
                continue
            try:
                relative_path = dependency.relative_to(overlay_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    "Source overlay dependency escapes the carrier directory."
                ) from exc
            overlay_dependencies.append(
                ReconstructionSourceDependency(path=dependency, relative_path=relative_path)
            )

        return cls(
            original_source_path=bundle.original_source_path,
            original_source_dependencies=tuple(source_dependencies),
            original_source_manifest_path=approved_manifest,
            original_source_relative_path=original_source_relative_path,
            approved_original_source_path=approved_entry_path,
            approved_inputs_manifest_sha256=bundle.approved_inputs_manifest_sha256,
            approved_inputs_manifest_byte_count=(bundle.approved_inputs_manifest_byte_count),
            normalized_source_path=bundle.normalized_source_path,
            source_manifest_path=bundle.source_manifest_path,
            subject_isolation_binding_path=bundle.subject_isolation_binding_path,
            subject_isolation_manifest_path=bundle.subject_isolation_manifest_path,
            mesh_preprocess_manifest_path=bundle.mesh_preprocess_manifest_path,
            semantic_image_path=bundle.semantic_image_path,
            scaffold_path=bundle.scaffold_path,
            axial_path=bundle.axial_path,
            source_overlay_manifest_path=bundle.source_overlay_manifest_path,
            author_evidence_manifest_path=author_evidence_manifest_path,
            source_overlay_urdf_path=bundle.source_overlay_urdf_path,
            source_overlay_glb_path=bundle.source_overlay_glb_path,
            source_overlay_dependencies=tuple(overlay_dependencies),
            voxel_sidecar_path=bundle.voxel_evidence_path,
        )


def _verify_evidence_approved_snapshot(evidence: ReconstructionEvidence) -> None:
    manifest_path = evidence.original_source_manifest_path
    if (
        manifest_path is None
        or evidence.approved_inputs_manifest_sha256 is None
        or evidence.approved_inputs_manifest_byte_count is None
    ):
        raise ValueError(
            "Canonical Reconstruction persistence requires the frozen approved-input snapshot."
        )
    if (
        sha256_file(manifest_path) != evidence.approved_inputs_manifest_sha256
        or manifest_path.stat().st_size != evidence.approved_inputs_manifest_byte_count
    ):
        raise ValueError("Frozen approved-input manifest changed before publication.")
    # Full frozen-byte validation runs when the evidence is built from the bundle and again by the
    # storage contract against the copied canonical bytes; this entry gate only rejects evidence
    # that lacks the approved snapshot or whose manifest identity drifted.


def _slug(value: str, *, limit: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized or "reconstruction")[:limit].rstrip("-")


def _record_id(title: str, run_id: str, created_at: datetime) -> str:
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    digest = sha256_text(title)[:8]
    run_suffix = re.sub(r"[^a-zA-Z0-9]+", "", run_id)[-8:] or "run"
    # The default request already starts with "Reconstruct"; keep the prefix from repeating it.
    slug = _slug(title).removeprefix("reconstruct-")
    return f"rec_reconstruct-{slug}_{timestamp}_{digest}_{run_suffix}"


def _suffix(path: Path, fallback: str) -> str:
    return path.suffix.lower() or fallback


def _required_compile_paths(artifacts: CompileArtifacts) -> tuple[Path, ...]:
    return (
        artifacts.urdf_path,
        artifacts.usdc_path,
        artifacts.newton_usdc_path,
        artifacts.asset_usda_path,
        artifacts.physics_usdc_path,
        artifacts.physx_usda_path,
        artifacts.mujoco_usda_path,
        artifacts.usdz_path,
        artifacts.report_path,
    )


def persist_reconstruction(
    *,
    repo_root: Path,
    run_id: str,
    title: str,
    request_text: str,
    runtime: RuntimeSelection,
    workspace_dir: Path,
    runtime_report: dict[str, Any],
    artifacts: CompileArtifacts,
    evidence: ReconstructionEvidence,
    label: str | None = None,
    guidance: str | None = None,
    authoring: ReconstructionAuthoringIdentity = CODEX_RECONSTRUCTION_AUTHORING,
) -> Path:
    """Promote one validated Reconstruction author workspace into a canonical record.

    ``authoring`` selects the arm's identity and trace layout. ``guidance`` is the request's
    optional user guidance, recorded verbatim as ``reconstruction.json`` ``input.guidance``
    (``null`` when the run had none).
    """

    repo_root = Path(repo_root).resolve()
    _verify_evidence_approved_snapshot(evidence)
    workspace_dir = Path(workspace_dir).resolve()
    title = " ".join(str(title or "").split())
    if not title:
        raise ValueError("Reconstruction title must not be empty.")
    request_text = str(request_text or "").rstrip()
    if guidance is not None:
        if not isinstance(guidance, str):
            raise ValueError("Reconstruction guidance must be a string or None.")
        guidance = guidance.strip() or None
    for path in _required_compile_paths(artifacts):
        if not path.is_file():
            raise FileNotFoundError(f"Validated compile artifact disappeared: {path}")
    for name in ("model.py", "TASK.md"):
        path = workspace_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Validated Reconstruction {name} disappeared: {path}")
    for relative in authoring.required_trace_files:
        path = workspace_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Reconstruction Author trace disappeared: {path}")

    source_contract = validate_reconstruction_materialization_contract(
        artifacts.output_dir,
        {
            "schema_version": RECONSTRUCTION_MANIFEST_SCHEMA_VERSION,
            "pipeline": "reconstruction",
            "status": "reconstruction_ready",
            "run_id": run_id,
            "validation_profile": RECONSTRUCTION_VALIDATION_PROFILE,
            "compile_validation_profile": GENERATION_VALIDATION_PROFILE,
            "outputs": physics_output_contract(),
        },
        phase="compile_output",
    )
    if not source_contract.ok:
        raise ValueError(
            "Validated compile output violates the Reconstruction artifact contract: "
            + source_contract.summary()
        )

    created_at = _utc_now()
    record_id = _record_id(title, run_id, created_at)
    repo = StorageRepo(repo_root)
    repo.ensure_layout()
    promotion = begin_canonical_promotion(
        repo,
        record_id,
        manifest_writer=write_promotion_manifest,
    )
    succeeded = False
    try:
        result = _stage_and_commit(
            promotion=promotion,
            repo_root=repo_root,
            run_id=run_id,
            title=title,
            request_text=request_text,
            runtime=runtime,
            workspace_dir=workspace_dir,
            runtime_report=runtime_report,
            artifacts=artifacts,
            evidence=evidence,
            label=label,
            guidance=guidance,
            created_at=created_at,
            record_id=record_id,
            authoring=authoring,
        )
        succeeded = True
        return result
    finally:
        finish_canonical_promotion(promotion, succeeded=succeeded)


def _stage_and_commit(
    *,
    promotion: CanonicalPromotion,
    repo_root: Path,
    run_id: str,
    title: str,
    request_text: str,
    runtime: RuntimeSelection,
    workspace_dir: Path,
    runtime_report: dict[str, Any],
    artifacts: CompileArtifacts,
    evidence: ReconstructionEvidence,
    label: str | None,
    guidance: str | None,
    created_at: datetime,
    record_id: str,
    authoring: ReconstructionAuthoringIdentity,
) -> Path:
    identity_fields = {
        "agent": authoring.agent_harness,
        "agent_harness": authoring.agent_harness,
        "provider": authoring.provider,
    }
    revision_dir = promotion.record_dir / "revisions" / REVISION_ID
    revision_dir.mkdir(parents=True)
    (revision_dir / "inputs").mkdir()
    traces_dir = revision_dir / "traces"
    traces_dir.mkdir()
    now = _format_utc(created_at)

    shutil.copy2(workspace_dir / "model.py", revision_dir / "model.py")
    (revision_dir / "request.txt").write_text(
        request_text + ("\n" if request_text else ""),
        encoding="utf-8",
    )
    shutil.copy2(workspace_dir / "TASK.md", traces_dir / authoring.task_trace_name)
    for relative, archived_name in authoring.trace_files.items():
        source = workspace_dir / relative
        if relative in authoring.required_trace_files or source.is_file():
            shutil.copy2(source, traces_dir / archived_name)
    stage_trace = workspace_dir / "stage_trace.jsonl"
    if stage_trace.is_file():
        shutil.copy2(stage_trace, traces_dir / "stage_trace.jsonl")
    for observation_name in ("image_observation.json", "image_observation_manifest.json"):
        observation_source = workspace_dir / observation_name
        if observation_source.is_file():
            shutil.copy2(observation_source, traces_dir / observation_name)

    # Same-session tool feedback belongs to the audit trail, never the evidence allowlist.
    # Do not follow links out of the disposable feedback tree when retaining local artifacts.
    feedback_root = workspace_dir / "author_feedback"
    if feedback_root.is_dir() and not feedback_root.is_symlink():
        feedback_index = []
        for source in sorted(feedback_root.rglob("*")):
            if source.is_symlink() or not source.is_file():
                continue
            if not source.resolve().is_relative_to(feedback_root.resolve()):
                continue
            feedback_relative = source.relative_to(feedback_root)
            destination = traces_dir / "author_feedback" / feedback_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            feedback_index.append(
                {"path": feedback_relative.as_posix(), "sha256": sha256_file(destination)}
            )
        _write_json(traces_dir / "author_feedback_index.json", {"files": feedback_index})

    input_manifest, descriptors = _copy_evidence(revision_dir, evidence)
    input_manifest_path = revision_dir / "inputs" / "manifest.json"
    _write_json(input_manifest_path, input_manifest)
    input_manifest_sha256 = sha256_file(input_manifest_path)
    author_descriptors = [item for item in descriptors if item["author_input"]]
    host_descriptors = [item for item in descriptors if not item["author_input"]]

    shutil.copytree(artifacts.output_dir, promotion.materialization_dir)
    compile_report_path = promotion.materialization_dir / "compile_report.json"
    compile_report = json.loads(compile_report_path.read_text(encoding="utf-8"))
    compile_report["record_id"] = record_id
    compile_report["model_path"] = f"data/records/{record_id}/revisions/{REVISION_ID}/model.py"
    compile_report["usdcraft"] = {
        "promotion_transaction_id": promotion.transaction_id,
    }
    _write_json(compile_report_path, compile_report)

    model_sha256 = sha256_file(revision_dir / "model.py")
    request_sha256 = sha256_text(request_text)
    runtime_projection = runtime.to_dict()
    input_kind = "mesh_image_text" if request_text else "mesh_image"

    def descriptor_projection(item: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in item.items() if key != "author_input"}

    author_projection = [descriptor_projection(item) for item in author_descriptors]
    host_projection = [descriptor_projection(item) for item in host_descriptors]
    reconstruction_manifest = {
        "schema_version": RECONSTRUCTION_MANIFEST_SCHEMA_VERSION,
        "pipeline": "reconstruction",
        "status": "reconstruction_ready",
        "record_id": record_id,
        "revision_id": REVISION_ID,
        "run_id": run_id,
        "input": {
            "kind": input_kind,
            "request_path": "request.txt",
            "request_sha256": request_sha256,
            "guidance": guidance,
            "manifest_path": "inputs/manifest.json",
            "manifest_sha256": input_manifest_sha256,
            "author_inputs": author_projection,
            "host_preprocessing_inputs": host_projection,
        },
        "creator": {
            **identity_fields,
            **runtime_projection,
        },
        "model_sha256": model_sha256,
        "validation_profile": RECONSTRUCTION_VALIDATION_PROFILE,
        "compile_validation_profile": GENERATION_VALIDATION_PROFILE,
        "outputs": physics_output_contract(),
        "compile_report": (
            f"../../../../cache/record_materialization/{record_id}/compile_report.json"
        ),
        "created_at": now,
        "usdcraft": {"promotion_transaction_id": promotion.transaction_id},
    }
    _write_json(revision_dir / "reconstruction.json", reconstruction_manifest)
    staged_contract = validate_reconstruction_materialization_contract(
        promotion.materialization_dir,
        reconstruction_manifest,
        phase="canonical",
        revision_dir=revision_dir,
    )
    if not staged_contract.ok:
        raise ValueError(
            "Staged canonical record violates the Reconstruction artifact contract: "
            + staged_contract.summary()
        )

    provenance = {
        "schema_version": 3,
        "record_id": record_id,
        "pipeline": "reconstruction",
        "runtime": {
            **runtime_report,
            **runtime_projection,
            **identity_fields,
        },
        "prompting": {
            "system_prompt_file": f"traces/{authoring.task_trace_name}",
            "system_prompt_sha256": sha256_file(workspace_dir / "TASK.md"),
        },
        "evidence": {
            "input_manifest_path": "inputs/manifest.json",
            "input_manifest_sha256": input_manifest_sha256,
            "author_inputs": author_projection,
            "host_preprocessing_inputs": host_projection,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": sys.platform,
            "git_commit": _git_commit(repo_root),
        },
        "run_summary": {"final_status": "reconstruction_ready"},
        "source": {"run_id": run_id},
    }
    _write_json(revision_dir / "provenance.json", provenance)
    relative_base = f"revisions/{REVISION_ID}"
    revision_artifacts = {
        "request_txt": f"{relative_base}/request.txt",
        "model_py": f"{relative_base}/model.py",
        "provenance_json": f"{relative_base}/provenance.json",
        "input_manifest_json": f"{relative_base}/inputs/manifest.json",
        "inputs_dir": f"{relative_base}/inputs",
        "traces_dir": f"{relative_base}/traces",
        "reconstruction_json": f"{relative_base}/reconstruction.json",
    }
    revision = {
        "schema_version": 1,
        "record_id": record_id,
        "revision_id": REVISION_ID,
        "created_at": now,
        "pipeline": "reconstruction",
        "status": "reconstruction_ready",
        "request_kind": "optional_user_guidance",
        "request_sha256": request_sha256,
        "source": {"run_id": run_id},
        "reconstruction": {
            "agent_harness": authoring.agent_harness,
            "provider": authoring.provider,
            **runtime_projection,
        },
        "artifacts": revision_artifacts,
        "hashes": {
            "request_sha256": request_sha256,
            "model_py_sha256": model_sha256,
            "input_manifest_sha256": input_manifest_sha256,
        },
        "run_summary": {"final_status": "reconstruction_ready"},
    }
    _write_json(revision_dir / "revision.json", revision)
    record = {
        "schema_version": 3,
        "record_id": record_id,
        "created_at": now,
        "updated_at": now,
        "rating": None,
        "kind": "structured_3d_asset",
        "pipeline": "reconstruction",
        "status": "reconstruction_ready",
        "input_kind": input_kind,
        "request_kind": "optional_user_guidance",
        "source": {"run_id": run_id},
        "agent_harness": authoring.agent_harness,
        "provider": authoring.provider,
        **runtime_projection,
        "display": {"title": title, "prompt_preview": request_text[:240]},
        "artifacts": revision_artifacts,
        "hashes": {
            "request_sha256": request_sha256,
            "model_py_sha256": model_sha256,
            "input_manifest_sha256": input_manifest_sha256,
        },
        "evidence": {
            "manifest_path": f"{relative_base}/inputs/manifest.json",
            "manifest_sha256": input_manifest_sha256,
            "author_inputs": author_projection,
            "host_preprocessing_inputs": host_projection,
        },
        "collections": ["workbench"],
        "active_revision_id": REVISION_ID,
        "lineage": {"origin_record_id": record_id, "parent_record_id": None},
        "creator": {
            "mode": "external_agent",
            **identity_fields,
            "trace_available": True,
        },
        "usdcraft": {"promotion_transaction_id": promotion.transaction_id},
    }
    _write_json(promotion.record_dir / "record.json", record)
    return commit_canonical_promotion(
        promotion,
        added_at=now,
        label=label,
        manifest_writer=write_promotion_manifest,
        record_lock_factory=record_lock,
    )


def _copy_evidence(
    revision_dir: Path,
    evidence: ReconstructionEvidence,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    descriptors: list[dict[str, Any]] = []
    keys: set[str] = set()
    paths: set[str] = set()

    def copy(
        *,
        key: str,
        source: Path,
        relative_path: str,
        role: str,
        authority: str,
        author_input: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if key in keys:
            raise ValueError(f"Duplicate Reconstruction evidence key: {key}")
        safe_path = _safe_relative_path(relative_path, label=f"Evidence path for {key}")
        if not safe_path.startswith("inputs/"):
            raise ValueError(f"Reconstruction evidence path must be below inputs/: {safe_path}")
        if safe_path in paths:
            raise ValueError(f"Duplicate Reconstruction evidence path: {safe_path}")
        destination = revision_dir / safe_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        descriptor = {
            "key": key,
            "path": safe_path,
            "sha256": sha256_file(destination),
            "byte_count": destination.stat().st_size,
            "role": role,
            "authority": authority,
            "author_input": author_input,
        }
        if extra:
            descriptor.update(extra)
        descriptors.append(descriptor)
        keys.add(key)
        paths.add(safe_path)

    def copy_audit_tree(
        *,
        channel: str,
        source_root: Path,
        relative_root: str,
    ) -> tuple[list[str], dict[Path, str]]:
        """Copy one Host-created audit tree without invalidating its internal relative paths."""

        resolved_root = Path(source_root).resolve()
        if not resolved_root.is_dir():
            raise ValueError(
                f"Reconstruction {channel} audit root is not a directory: {resolved_root}"
            )
        safe_root = _safe_relative_path(
            relative_root,
            label=f"Reconstruction {channel} audit destination",
        )
        if not safe_root.startswith("inputs/"):
            raise ValueError(f"Reconstruction audit tree must be below inputs/: {safe_root}")
        source_files: list[tuple[str, Path]] = []
        for candidate in sorted(resolved_root.rglob("*"), key=lambda item: item.as_posix()):
            if candidate.is_symlink():
                raise ValueError(
                    f"Reconstruction {channel} audit tree must not contain symlinks: {candidate}"
                )
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(resolved_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"Reconstruction {channel} audit file escapes its root: {candidate}"
                ) from exc
            source_files.append((relative, resolved))
        if not source_files:
            raise ValueError(f"Reconstruction {channel} audit tree contains no files.")

        file_keys: list[str] = []
        source_to_key: dict[Path, str] = {}
        for index, (relative, source) in enumerate(source_files, start=1):
            key = f"{channel}_audit_{index:04d}"
            copy(
                key=key,
                source=source,
                relative_path=f"{safe_root}/{relative}",
                role=f"{channel}_audit_artifact",
                authority="host_preprocessing_audit",
                author_input=False,
                extra={
                    "audit_channel": channel,
                    "audit_relative_path": relative,
                },
            )
            file_keys.append(key)
            source_to_key[source] = key
        return file_keys, source_to_key

    original_base = "inputs/source/original"
    copy(
        key="original_source",
        source=evidence.original_source_path,
        relative_path=f"{original_base}/{evidence.original_source_relative_path}",
        role="user_supplied_source_surface_entry",
        authority="surface_evidence_provenance_only",
        author_input=False,
        extra={
            "source_relative_path": evidence.original_source_relative_path,
            "source_resolved_path": (
                evidence.approved_original_source_path or str(evidence.original_source_path)
            ),
        },
    )
    source_dependency_keys: list[str] = []
    for index, dependency in enumerate(
        sorted(evidence.original_source_dependencies, key=lambda item: item.relative_path),
        start=1,
    ):
        key = f"source_dependency_{index:04d}"
        source_dependency_keys.append(key)
        copy(
            key=key,
            source=dependency.path,
            relative_path=f"{original_base}/{dependency.relative_path}",
            role="user_supplied_source_dependency",
            authority="surface_evidence_provenance_only",
            author_input=False,
            extra={
                "source_relative_path": dependency.relative_path,
                "source_logical_path": (dependency.source_logical_path or dependency.relative_path),
                "source_resolved_path": (dependency.approved_original_path or str(dependency.path)),
                "canonicalized_external": dependency.relative_path.startswith("_external/"),
            },
        )
    original_source_manifest_key: str | None = None
    if evidence.original_source_manifest_path is not None:
        original_source_manifest_key = "original_source_manifest"
        copy(
            key=original_source_manifest_key,
            source=evidence.original_source_manifest_path,
            relative_path=(
                "inputs/provenance/original_source_manifest"
                + _suffix(evidence.original_source_manifest_path, ".json")
            ),
            role="original_source_dependency_manifest",
            authority="host_provenance",
            author_input=False,
        )

    copy(
        key="normalized_source",
        source=evidence.normalized_source_path,
        relative_path=(
            "inputs/source/normalized/normalized"
            + _suffix(evidence.normalized_source_path, ".usdc")
        ),
        role="post_isolation_normalized_source",
        authority="same_frame_metric_surface_evidence",
        author_input=True,
    )
    for key, source, destination, role in (
        (
            "source_manifest",
            evidence.source_manifest_path,
            "inputs/provenance/source_manifest.json",
            "source_and_frame_provenance",
        ),
        (
            "subject_isolation_binding",
            evidence.subject_isolation_binding_path,
            "inputs/preprocessing/subject_isolation_binding.json",
            "exact_retained_scene_binding",
        ),
        (
            "subject_isolation_manifest",
            evidence.subject_isolation_manifest_path,
            "inputs/preprocessing/subject_isolation_manifest.json",
            "subject_isolation_audit",
        ),
        (
            "mesh_preprocess_manifest",
            evidence.mesh_preprocess_manifest_path,
            "inputs/preprocessing/mesh_preprocess_manifest.json",
            "mesh_preprocess_audit",
        ),
        (
            "source_overlay_manifest",
            evidence.source_overlay_manifest_path,
            "inputs/preprocessing/source_overlay_manifest.json",
            "source_overlay_same_source_binding",
        ),
    ):
        copy(
            key=key,
            source=source,
            relative_path=destination,
            role=role,
            authority="host_preprocessing_audit",
            author_input=False,
        )

    semantic_suffix = _suffix(evidence.semantic_image_path, ".png")
    copy(
        key="semantic_image",
        source=evidence.semantic_image_path,
        relative_path=f"inputs/semantic/semantic{semantic_suffix}",
        role="semantic_subject_evidence",
        authority="positive_visibility_bounded_semantic",
        author_input=True,
    )
    copy(
        key="author_scaffold",
        source=evidence.scaffold_path,
        relative_path="inputs/author/scaffold.py",
        role="initial_author_scaffold",
        authority="authoring_scaffold_only",
        author_input=True,
    )
    copy(
        key="axial",
        source=evidence.axial_path,
        relative_path="inputs/geometry/axial.txt",
        role="metric_q0_sampled_surface_evidence",
        authority="metric_surface_evidence",
        author_input=True,
    )
    copy(
        key="author_evidence_manifest",
        source=evidence.author_evidence_manifest_path,
        relative_path="inputs/geometry/author_evidence_manifest.json",
        role="author_evidence_allowlist_binding",
        authority="author_context_audit",
        author_input=True,
    )
    overlay_base = "inputs/source/overlay/carrier"
    copy(
        key="source_overlay_urdf",
        source=evidence.source_overlay_urdf_path,
        relative_path=f"{overlay_base}/model.urdf",
        role="viewer_source_overlay_carrier",
        authority="viewer_same_frame_overlay_only",
        author_input=False,
    )
    copy(
        key="source_overlay_glb",
        source=evidence.source_overlay_glb_path,
        relative_path=f"{overlay_base}/assets/source.glb",
        role="viewer_source_overlay_geometry",
        authority="viewer_same_frame_overlay_only",
        author_input=False,
    )
    overlay_dependency_keys: list[str] = []
    for index, dependency in enumerate(
        sorted(evidence.source_overlay_dependencies, key=lambda item: item.relative_path),
        start=1,
    ):
        key = f"source_overlay_dependency_{index:04d}"
        overlay_dependency_keys.append(key)
        copy(
            key=key,
            source=dependency.path,
            relative_path=f"{overlay_base}/{dependency.relative_path}",
            role="viewer_source_overlay_dependency",
            authority="viewer_same_frame_overlay_only",
            author_input=False,
            extra={"carrier_relative_path": dependency.relative_path},
        )

    if evidence.voxel_sidecar_path is not None:
        copy(
            key="voxel_sidecar",
            source=evidence.voxel_sidecar_path,
            relative_path=(
                "inputs/diagnostics/voxel_sidecar" + _suffix(evidence.voxel_sidecar_path, ".json")
            ),
            role="posthoc_diagnostic_only",
            authority="viewer_diagnostic_only",
            author_input=False,
        )

    isolation_audit_keys, isolation_audit_sources = copy_audit_tree(
        channel="subject_isolation",
        source_root=evidence.subject_isolation_manifest_path.parent,
        relative_root="inputs/preprocessing/audit/subject_isolation",
    )
    mesh_preprocess_audit_keys, mesh_preprocess_audit_sources = copy_audit_tree(
        channel="mesh_preprocess",
        source_root=evidence.mesh_preprocess_manifest_path.parent,
        relative_root="inputs/preprocessing/audit/mesh_preprocess",
    )
    overlay_audit_keys, overlay_audit_sources = copy_audit_tree(
        channel="source_overlay",
        source_root=evidence.source_overlay_manifest_path.parent,
        relative_root="inputs/preprocessing/audit/source_overlay",
    )

    author_context_keys: list[str] = []
    author_context_sources: dict[Path, str] = {}

    def copy_author_context(*, key: str, source: Path, relative_path: str, role: str) -> None:
        copy(
            key=key,
            source=source,
            relative_path=f"inputs/author_context/{relative_path}",
            role=role,
            authority="exact_author_context_audit_copy",
            author_input=False,
            extra={
                "audit_channel": "author_context",
                "audit_relative_path": relative_path,
            },
        )
        author_context_keys.append(key)
        author_context_sources[source.resolve()] = key

    copy_author_context(
        key="author_context_manifest_snapshot",
        source=evidence.author_evidence_manifest_path,
        relative_path="evidence_manifest.json",
        role="author_evidence_allowlist_snapshot",
    )
    copy_author_context(
        key="author_context_scaffold_snapshot",
        source=evidence.scaffold_path,
        relative_path="model.py",
        role="initial_author_scaffold_snapshot",
    )
    copy_author_context(
        key="author_context_semantic_snapshot",
        source=evidence.semantic_image_path,
        relative_path=("inputs/semantic_image" + _suffix(evidence.semantic_image_path, ".png")),
        role="semantic_image_author_context_snapshot",
    )
    copy_author_context(
        key="author_context_axial_snapshot",
        source=evidence.axial_path,
        relative_path="inputs/axial.txt",
        role="axial_author_context_snapshot",
    )
    copy_author_context(
        key="author_context_source_mesh_snapshot",
        source=evidence.normalized_source_path,
        relative_path="inputs/source_mesh.usdc",
        role="normalized_source_author_context_snapshot",
    )
    context = json.loads(evidence.author_evidence_manifest_path.read_text())
    source_render = context.get("allowed_evidence", {}).get("source_render", {})
    if source_render.get("available"):
        carrier_root = evidence.source_overlay_urdf_path.parent
        for index, relative in enumerate(sorted(source_render.get("files", {}))):
            relative_carrier = Path(relative).relative_to("inputs/source_render")
            if ".." in relative_carrier.parts:
                raise ValueError("Invalid source render context path")
            source = carrier_root / relative_carrier
            if sha256_file(source) != source_render["files"][relative]:
                raise ValueError("Staged source render carrier identity changed")
            copy_author_context(
                key=f"author_context_source_render_{index:04d}",
                source=source,
                relative_path=relative,
                role="source_render_author_context_snapshot",
            )

    def audit_key(source: Path, mapping: dict[Path, str], *, label: str) -> str:
        key = mapping.get(source.resolve())
        if key is None:
            raise ValueError(f"{label} is not retained in its canonical audit closure: {source}")
        return key

    primary_audit_bindings = {
        "subject_isolation_manifest": audit_key(
            evidence.subject_isolation_manifest_path,
            isolation_audit_sources,
            label="Subject-isolation manifest",
        ),
        "normalized_source": audit_key(
            evidence.normalized_source_path,
            mesh_preprocess_audit_sources,
            label="Normalized source",
        ),
        "source_manifest": audit_key(
            evidence.source_manifest_path,
            mesh_preprocess_audit_sources,
            label="Source manifest",
        ),
        "subject_isolation_binding": audit_key(
            evidence.subject_isolation_binding_path,
            mesh_preprocess_audit_sources,
            label="Subject-isolation binding",
        ),
        "mesh_preprocess_manifest": audit_key(
            evidence.mesh_preprocess_manifest_path,
            mesh_preprocess_audit_sources,
            label="Mesh-preprocess manifest",
        ),
        "source_overlay_manifest": audit_key(
            evidence.source_overlay_manifest_path,
            overlay_audit_sources,
            label="Source-overlay manifest",
        ),
        "source_overlay_urdf": audit_key(
            evidence.source_overlay_urdf_path,
            overlay_audit_sources,
            label="Source-overlay URDF",
        ),
        "source_overlay_glb": audit_key(
            evidence.source_overlay_glb_path,
            overlay_audit_sources,
            label="Source-overlay GLB",
        ),
        "author_evidence_manifest": audit_key(
            evidence.author_evidence_manifest_path,
            author_context_sources,
            label="Author evidence manifest",
        ),
        "author_scaffold": audit_key(
            evidence.scaffold_path,
            author_context_sources,
            label="Author scaffold",
        ),
        "semantic_image": audit_key(
            evidence.semantic_image_path,
            author_context_sources,
            label="Semantic image",
        ),
        "axial": audit_key(
            evidence.axial_path,
            author_context_sources,
            label="AXIAL",
        ),
    }
    if evidence.voxel_sidecar_path is not None:
        primary_audit_bindings["voxel_sidecar"] = audit_key(
            evidence.voxel_sidecar_path,
            mesh_preprocess_audit_sources,
            label="Voxel sidecar",
        )

    author_keys = [item["key"] for item in descriptors if item["author_input"]]
    host_keys = [item["key"] for item in descriptors if not item["author_input"]]
    manifest = {
        "source_view_policy": "author_requested",
        "schema": RECONSTRUCTION_INPUT_MANIFEST_SCHEMA,
        "schema_version": RECONSTRUCTION_INPUT_MANIFEST_SCHEMA_VERSION,
        "pipeline": "reconstruction",
        "path_base": "revision",
        "files": descriptors,
        "author_inputs": author_keys,
        "host_preprocessing_inputs": host_keys,
        "channels": {
            "original_source": {
                "entry_key": "original_source",
                "dependency_keys": source_dependency_keys,
                "manifest_key": original_source_manifest_key,
            },
            "normalized_source": {
                "file_key": "normalized_source",
                "same_frame_as_candidate": True,
            },
            "subject_isolation": {
                "binding_key": "subject_isolation_binding",
                "manifest_key": "subject_isolation_manifest",
            },
            "mesh_preprocess": {"manifest_key": "mesh_preprocess_manifest"},
            "semantic_image": {
                "file_key": "semantic_image",
                "provenance_role": SEMANTIC_PROVENANCE_ROLE,
            },
            "geometry": {
                "axial_key": "axial",
                "evidence_manifest_key": "author_evidence_manifest",
            },
            "source_overlay": {
                "manifest_key": "source_overlay_manifest",
                "base_path": overlay_base,
                "urdf_key": "source_overlay_urdf",
                "glb_key": "source_overlay_glb",
                "dependency_keys": overlay_dependency_keys,
                "same_frame_as_candidate": True,
                "viewer_only": True,
            },
            "audit_closures": {
                "subject_isolation": {
                    "base_path": "inputs/preprocessing/audit/subject_isolation",
                    "file_keys": isolation_audit_keys,
                    "manifest_primary_key": "subject_isolation_manifest",
                    "manifest_audit_key": primary_audit_bindings["subject_isolation_manifest"],
                },
                "mesh_preprocess": {
                    "base_path": "inputs/preprocessing/audit/mesh_preprocess",
                    "file_keys": mesh_preprocess_audit_keys,
                    "manifest_primary_key": "mesh_preprocess_manifest",
                    "manifest_audit_key": primary_audit_bindings["mesh_preprocess_manifest"],
                },
                "source_overlay": {
                    "base_path": "inputs/preprocessing/audit/source_overlay",
                    "file_keys": overlay_audit_keys,
                    "manifest_primary_key": "source_overlay_manifest",
                    "manifest_audit_key": primary_audit_bindings["source_overlay_manifest"],
                },
                "author_context": {
                    "base_path": "inputs/author_context",
                    "file_keys": author_context_keys,
                    "manifest_primary_key": "author_evidence_manifest",
                    "manifest_audit_key": primary_audit_bindings["author_evidence_manifest"],
                },
                "primary_bindings": primary_audit_bindings,
                "all_copies_byte_identical": True,
                "internal_relative_paths_resolve_against_each_base_path": True,
            },
            "voxel_sidecar": (
                {
                    "file_key": "voxel_sidecar",
                    "viewer_only": True,
                    "author_input": False,
                }
                if evidence.voxel_sidecar_path is not None
                else None
            ),
        },
    }
    return manifest, descriptors


__all__ = [
    "CODEX_RECONSTRUCTION_AUTHORING",
    "NVIDIA_PROVIDER_RECONSTRUCTION_AUTHORING",
    "ReconstructionAuthoringIdentity",
    "REVISION_ID",
    "ReconstructionEvidence",
    "ReconstructionSourceDependency",
    "persist_reconstruction",
    "provider_reconstruction_authoring",
]
