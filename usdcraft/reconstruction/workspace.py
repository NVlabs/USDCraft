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
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

from usdcraft.pipeline.files import _write_json
from usdcraft.pipeline.hashing import sha256_file
from usdcraft.reconstruction.contract import (
    ReconstructionEvidenceBundle,
    ReconstructionInputIdentity,
    ReconstructionPreflight,
)
from usdcraft.reconstruction.harness_policy import policy_for_workspace
from usdcraft.reconstruction.source import (
    FrameValueSource,
    NormalizedSourceScene,
    SourceDependency,
    SourceInput,
    load_normalized_source,
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class ReconstructionInputFreezeError(RuntimeError):
    """Approved Recon inputs could not be frozen as one self-contained snapshot."""


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"recon_{timestamp}_{uuid.uuid4().hex[:8]}"


def _run_id(value: str | None) -> str:
    candidate = _default_run_id() if value is None else str(value).strip()
    if not _RUN_ID_RE.fullmatch(candidate):
        raise ValueError(
            "Reconstruction run id must be 1-96 characters using letters, digits, '.', '_' or '-'."
        )
    return candidate


@dataclass(slots=True, frozen=True)
class FrozenSourceFile:
    original: ReconstructionInputIdentity
    staged: ReconstructionInputIdentity
    tree_relative_path: str
    entry: bool


@dataclass(slots=True, frozen=True)
class ReconstructionFrozenInputs:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest_byte_count: int
    source_tree_root: Path
    source_path: Path
    source_files: tuple[FrozenSourceFile, ...]
    source_scene: NormalizedSourceScene
    semantic_image_path: Path
    scaffold_path: Path

    @property
    def source_dependencies(self) -> tuple[ReconstructionInputIdentity, ...]:
        return tuple(item.staged for item in self.source_files)


@dataclass(slots=True, frozen=True)
class ReconstructionRunWorkspace:
    run_id: str
    repo_root: Path
    root: Path
    request_path: Path
    upstream_dir: Path
    author_workspace_base: Path
    frozen_inputs: ReconstructionFrozenInputs


@dataclass(slots=True, frozen=True)
class ReconstructionAuthorWorkspace:
    run_id: str
    repo_root: Path
    root: Path
    model_path: Path
    task_path: Path
    context_manifest_path: Path
    semantic_image_path: Path
    axial_path: Path
    codex_dir: Path
    codex_schema_path: Path
    codex_last_message_path: Path
    codex_stdout_path: Path
    codex_stderr_path: Path
    codex_timeline_path: Path
    codex_trace_errors_path: Path
    build_dir: Path
    temp_dir: Path
    cache_dir: Path
    # Provider arm only: the Author-facing AXIAL digest derived from the full text at staging.
    axial_brief_path: Path | None = None

    @property
    def source_mesh_path(self) -> Path:
        return self.root / "inputs" / "source_mesh.usdc"

    @property
    def attached_images(self) -> tuple[Path, ...]:
        return (self.semantic_image_path,)

    def write_task(self, text: str) -> None:
        self.task_path.write_text(text.rstrip() + "\n", encoding="utf-8")

    def write_codex_schema(self, schema: Mapping[str, object]) -> None:
        self.codex_schema_path.write_text(
            json.dumps(dict(schema), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_codex_logs(self, *, stdout: str, stderr: str) -> None:
        self.codex_stdout_path.write_text(stdout, encoding="utf-8")
        self.codex_stderr_path.write_text(stderr, encoding="utf-8")


def create_reconstruction_run_workspace(
    preflight: ReconstructionPreflight,
    *,
    run_id: str | None = None,
) -> ReconstructionRunWorkspace:
    """Create the run root and atomically freeze every approved input before any Agent."""

    resolved_run_id = _run_id(run_id)
    # Resolve the bases before creating descendants so a deliberately symlinked local-data root
    # uses one physical path space throughout copy, dependency reopen, containment and cleanup.
    root = (preflight.workspace_base / resolved_run_id).resolve()
    author_workspace_base = (preflight.author_workspace_base / resolved_run_id).resolve()
    if root.exists() or author_workspace_base.exists():
        raise FileExistsError(f"Reconstruction workspace already exists: {resolved_run_id}")
    try:
        root.mkdir(parents=True)
        upstream_dir = root / "upstream"
        upstream_dir.mkdir()
        author_workspace_base.mkdir(parents=True)
        frozen_inputs = _freeze_reconstruction_inputs(preflight, root)
        request_path = root / "request.json"
        _write_json(
            request_path,
            {
                **preflight.request.to_dict(),
                "run_id": resolved_run_id,
                "approved_fingerprint": preflight.fingerprint,
                "workspace": {
                    "upstream": "upstream",
                    "approved_inputs": frozen_inputs.root.relative_to(root).as_posix(),
                    "approved_inputs_manifest": (
                        frozen_inputs.manifest_path.relative_to(root).as_posix()
                    ),
                    "author_workspace_external": str(author_workspace_base),
                },
            },
        )
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(author_workspace_base, ignore_errors=True)
        raise
    return ReconstructionRunWorkspace(
        run_id=resolved_run_id,
        repo_root=preflight.request.repo_root,
        root=root,
        request_path=request_path,
        upstream_dir=upstream_dir,
        author_workspace_base=author_workspace_base,
        frozen_inputs=frozen_inputs,
    )


def _source_tree_anchor(preflight: ReconstructionPreflight) -> Path:
    parents = [str(identity.path.parent) for identity in preflight.source_dependencies]
    try:
        return Path(os.path.commonpath(parents)).resolve()
    except ValueError as exc:
        raise ReconstructionInputFreezeError(
            "Approved source dependencies do not share one filesystem tree."
        ) from exc


def _copy_approved_file(
    identity: ReconstructionInputIdentity,
    destination: Path,
) -> ReconstructionInputIdentity:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(identity.path, destination)
    staged = ReconstructionInputIdentity.from_path(destination)
    if staged.sha256 != identity.sha256 or staged.byte_count != identity.byte_count:
        raise ReconstructionInputFreezeError(
            f"Approved Reconstruction input changed while it was being frozen: {identity.path}"
        )
    return staged


def _freeze_reconstruction_inputs(
    preflight: ReconstructionPreflight,
    run_root: Path,
) -> ReconstructionFrozenInputs:
    approved_root = run_root / "approved_inputs"
    staging = Path(tempfile.mkdtemp(prefix=".approved-inputs-", dir=str(run_root)))
    try:
        source_tree = staging / "source" / "tree"
        anchor = _source_tree_anchor(preflight)
        approved_source = preflight.request.source_path
        source_files: list[FrozenSourceFile] = []
        for identity in preflight.source_dependencies:
            try:
                tree_relative = identity.path.relative_to(anchor)
            except ValueError as exc:
                raise ReconstructionInputFreezeError(
                    f"Approved source dependency escapes its common tree: {identity.path}"
                ) from exc
            staged = _copy_approved_file(identity, source_tree / tree_relative)
            source_files.append(
                FrozenSourceFile(
                    original=identity,
                    staged=staged,
                    tree_relative_path=tree_relative.as_posix(),
                    entry=identity.path == approved_source,
                )
            )
        entries = [item for item in source_files if item.entry]
        if len(entries) != 1:
            raise ReconstructionInputFreezeError(
                "Approved source closure must contain exactly one source entry."
            )
        staged_source = entries[0].staged.path
        source_scene = load_normalized_source(
            SourceInput(
                staged_source,
                up_axis=preflight.request.up_axis,
                front_axis=preflight.request.front_axis,
                meters_per_unit=preflight.request.meters_per_unit,
                meters_per_unit_source=cast(
                    "FrameValueSource | None", preflight.request.meters_per_unit_source
                ),
            )
        )
        expected = {item.tree_relative_path: item for item in source_files}
        observed: dict[str, SourceDependency] = {}
        for dependency in source_scene.source_dependencies:
            try:
                relative = dependency.path.relative_to(source_tree).as_posix()
            except ValueError as exc:
                raise ReconstructionInputFreezeError(
                    "Frozen source still resolves a dependency outside approved_inputs; "
                    "make the source closure self-contained before Reconstruction: "
                    f"{dependency.path}"
                ) from exc
            observed[relative] = dependency
        if set(observed) != set(expected):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise ReconstructionInputFreezeError(
                "Frozen source closure differs from the approved closure "
                f"(missing={missing}, extra={extra})."
            )
        for relative, dependency in observed.items():
            approved = expected[relative].original
            if dependency.sha256 != approved.sha256 or dependency.byte_count != approved.byte_count:
                raise ReconstructionInputFreezeError(
                    f"Frozen source dependency bytes differ from approval: {relative}"
                )

        semantic_suffix = preflight.semantic_image.path.suffix.lower() or ".png"
        semantic = _copy_approved_file(
            preflight.semantic_image,
            staging / "semantic" / f"semantic{semantic_suffix}",
        )
        scaffold = _copy_approved_file(
            preflight.scaffold,
            staging / "author" / "scaffold.py",
        )
        manifest_path = staging / "manifest.json"
        _write_json(
            manifest_path,
            {
                "schema": "reconstruction_approved_inputs",
                "schema_version": 1,
                "pipeline": "reconstruction",
                "approved_fingerprint": preflight.fingerprint,
                "source": {
                    "tree_anchor": str(anchor),
                    "entry_tree_relative_path": entries[0].tree_relative_path,
                    "files": [
                        {
                            "original_path": str(item.original.path),
                            "frozen_path": item.staged.path.relative_to(staging).as_posix(),
                            "tree_relative_path": item.tree_relative_path,
                            "sha256": item.staged.sha256,
                            "byte_count": item.staged.byte_count,
                            "entry": item.entry,
                        }
                        for item in source_files
                    ],
                },
                "semantic_image": {
                    "original_path": str(preflight.semantic_image.path),
                    "frozen_path": semantic.path.relative_to(staging).as_posix(),
                    "sha256": semantic.sha256,
                    "byte_count": semantic.byte_count,
                },
                "scaffold": {
                    "original_path": str(preflight.scaffold.path),
                    "frozen_path": scaffold.path.relative_to(staging).as_posix(),
                    "sha256": scaffold.sha256,
                    "byte_count": scaffold.byte_count,
                },
            },
        )
        manifest_sha256 = sha256_file(manifest_path)
        manifest_byte_count = manifest_path.stat().st_size
        os.replace(staging, approved_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    def published(path: Path) -> Path:
        return approved_root / path.relative_to(staging)

    published_files = tuple(
        replace(
            item,
            staged=ReconstructionInputIdentity(
                path=published(item.staged.path),
                sha256=item.staged.sha256,
                byte_count=item.staged.byte_count,
            ),
        )
        for item in source_files
    )
    published_dependencies = tuple(
        SourceDependency(
            path=published(dependency.path),
            sha256=dependency.sha256,
            byte_count=dependency.byte_count,
        )
        for dependency in source_scene.source_dependencies
    )
    published_scene = replace(
        source_scene,
        source_path=published(source_scene.source_path),
        source_dependencies=published_dependencies,
    )
    return ReconstructionFrozenInputs(
        root=approved_root,
        manifest_path=approved_root / "manifest.json",
        manifest_sha256=manifest_sha256,
        manifest_byte_count=manifest_byte_count,
        source_tree_root=approved_root / "source" / "tree",
        source_path=published(staged_source),
        source_files=published_files,
        source_scene=published_scene,
        semantic_image_path=published(semantic.path),
        scaffold_path=published(scaffold.path),
    )


def _staged_context(workspace: ReconstructionAuthorWorkspace) -> dict[str, Any]:
    return {
        "schema": "reconstruction_author_context",
        "pipeline": "reconstruction",
        "harness": policy_for_workspace(workspace.root).to_dict(),
        "clean_context": True,
        "user_config_loaded": False,
        "project_documents_loaded": False,
        "allowed_evidence": {
            "source_mesh": {
                "path": "inputs/source_mesh.usdc",
                "sha256": sha256_file(workspace.source_mesh_path)
                if workspace.source_mesh_path.is_file()
                else None,
                "available": workspace.source_mesh_path.is_file(),
                "role": "normalized_surface_for_author_measurement",
                "units": "metres",
                "up_axis": "Z",
            },
            "semantic_image": {
                "ordinal": 1,
                "path": workspace.semantic_image_path.relative_to(workspace.root).as_posix(),
                "sha256": sha256_file(workspace.semantic_image_path),
                "role": "positive_visibility_bounded_semantic_subject_evidence",
            },
            "axial": {
                "path": workspace.axial_path.relative_to(workspace.root).as_posix(),
                "sha256": sha256_file(workspace.axial_path),
                "role": "metric_q0_sampled_surface_evidence",
                "format": "AXIAL",
                **(
                    {
                        "author_brief": {
                            "path": workspace.axial_brief_path.relative_to(
                                workspace.root
                            ).as_posix(),
                            "sha256": sha256_file(workspace.axial_brief_path),
                            "derived_from": "axial.txt",
                        }
                    }
                    if workspace.axial_brief_path is not None
                    else {}
                ),
            },
            "source_render": {
                "available": True,
                "path": "inputs/source_render",
                "role": "same_source_qualitative_geometry_on_demand",
                "files": {
                    str(path.relative_to(workspace.root)): sha256_file(path)
                    for path in sorted((workspace.root / "inputs" / "source_render").rglob("*"))
                    if path.is_file()
                },
            },
            "scaffold": {
                "path": workspace.model_path.relative_to(workspace.root).as_posix(),
                "sha256": sha256_file(workspace.model_path),
            },
        },
        "attached_images": [
            workspace.semantic_image_path.relative_to(workspace.root).as_posix(),
        ],
        "withheld_from_author_workspace": [
            "raw_source_mesh",
            "source_manifest",
            "subject_isolation_inventory_contact_sheets_decision_and_trace",
            "voxel_json",
            "prior_attempts",
            "viewer_state",
            "source_or_hidden_evaluator_feedback",
            "ground_truth_code",
        ],
        "filesystem_isolation": "evidence_protocol_not_os_security_boundary",
    }


def create_reconstruction_author_workspace(
    preflight: ReconstructionPreflight,
    run_workspace: ReconstructionRunWorkspace,
    evidence: ReconstructionEvidenceBundle,
    *,
    root: Path,
    author_brief: bool = False,
) -> ReconstructionAuthorWorkspace:
    """Stage the exact Reconstruction Author allowlist into one fresh external workspace.

    Both arms run one Author and stage it at ``root`` — the run's author workspace base itself,
    matching Generation's one-directory-per-run layout.
    """

    if run_workspace.repo_root != preflight.request.repo_root:
        raise ValueError("Reconstruction run workspace and preflight use different repositories.")
    frozen = run_workspace.frozen_inputs
    if evidence.semantic_image_path != frozen.semantic_image_path:
        raise ValueError("Reconstruction Author evidence must use the frozen semantic image.")
    if evidence.scaffold_path != frozen.scaffold_path:
        raise ValueError("Reconstruction Author evidence must use the frozen scaffold.")
    if sha256_file(evidence.semantic_image_path) != preflight.semantic_image.sha256:
        raise ValueError("Semantic image changed after Reconstruction approval.")
    if sha256_file(evidence.scaffold_path) != preflight.scaffold.sha256:
        raise ValueError("Scaffold changed after Reconstruction approval.")

    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Reconstruction Author workspace is not empty: {root}")
    try:
        inputs_dir = root / "inputs"
        codex_dir = root / ".codex-run"
        build_dir = root / "build"
        temp_dir = root / "tmp"
        cache_dir = root / ".cache"
        for directory in (inputs_dir, codex_dir, build_dir, temp_dir, cache_dir):
            directory.mkdir(parents=True, exist_ok=False)

        model_path = root / "model.py"
        shutil.copy2(evidence.scaffold_path, model_path)
        semantic_image_path = inputs_dir / (
            "semantic_image" + evidence.semantic_image_path.suffix.lower()
        )
        shutil.copy2(evidence.semantic_image_path, semantic_image_path)
        if sha256_file(model_path) != preflight.scaffold.sha256:
            raise ValueError("Frozen scaffold changed while staging the Author workspace.")
        if sha256_file(semantic_image_path) != preflight.semantic_image.sha256:
            raise ValueError("Frozen semantic image changed while staging the Author workspace.")
        policy = preflight.request.harness_policy
        (root / "harness_policy.json").write_text(json.dumps(policy.to_dict(), indent=2) + "\n")
        shutil.copy2(evidence.normalized_source_path, inputs_dir / "source_mesh.usdc")
        axial_path = inputs_dir / "axial.txt"
        shutil.copy2(evidence.axial_path, axial_path)
        axial_brief_path: Path | None = None
        if author_brief:
            from usdcraft.reconstruction.axial_brief import render_axial_brief

            axial_brief_path = inputs_dir / "axial_brief.txt"
            axial_brief_path.write_text(
                render_axial_brief(axial_path.read_text(encoding="utf-8")), encoding="utf-8"
            )
        carrier = inputs_dir / "source_render"
        (carrier / "assets").mkdir(parents=True)
        shutil.copy2(evidence.source_overlay_urdf_path, carrier / "model.urdf")
        shutil.copy2(evidence.source_overlay_glb_path, carrier / "assets" / "source.glb")
        for dependency in evidence.source_overlay_dependency_paths:
            relative = dependency.relative_to(evidence.source_overlay_urdf_path.parent)
            destination = carrier / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dependency, destination)

        workspace = ReconstructionAuthorWorkspace(
            run_id=run_workspace.run_id,
            repo_root=preflight.request.repo_root,
            root=root,
            model_path=model_path,
            task_path=root / "TASK.md",
            context_manifest_path=root / "evidence_manifest.json",
            semantic_image_path=semantic_image_path,
            axial_path=axial_path,
            codex_dir=codex_dir,
            codex_schema_path=codex_dir / "final.schema.json",
            codex_last_message_path=codex_dir / "final.json",
            codex_stdout_path=codex_dir / "stdout.jsonl",
            codex_stderr_path=codex_dir / "stderr.log",
            codex_timeline_path=codex_dir / "timeline.jsonl",
            codex_trace_errors_path=codex_dir / "trace_errors.log",
            build_dir=build_dir,
            temp_dir=temp_dir,
            cache_dir=cache_dir,
            axial_brief_path=axial_brief_path,
        )
        _write_json(workspace.context_manifest_path, _staged_context(workspace))
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return workspace


__all__ = [
    "ReconstructionAuthorWorkspace",
    "ReconstructionFrozenInputs",
    "ReconstructionInputFreezeError",
    "ReconstructionRunWorkspace",
    "create_reconstruction_author_workspace",
    "create_reconstruction_run_workspace",
]
