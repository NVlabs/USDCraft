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

import hashlib
import json
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom

import usdcraft.reconstruction.persist as persist_module
from storage.artifact_contract import (
    REQUIRED_BACKEND_CHECKS,
    REQUIRED_COMPILE_CHECKS,
    classify_reconstruction_materialization_displayability,
    exporter_asset_structure_contract,
    physics_output_contract,
)
from storage.data_validation import validate_data_format
from storage.repo import StorageRepo
from usdcraft.compile.command import CompileArtifacts
from usdcraft.pipeline.contracts import CodexRuntimeSelection
from usdcraft.reconstruction.contract import (
    ReconstructionEvidenceBundle,
)
from usdcraft.reconstruction.persist import (
    ReconstructionEvidence,
    ReconstructionSourceDependency,
    persist_reconstruction,
)


def _artifacts(root: Path) -> CompileArtifacts:
    output = root / "build"
    usdc = output / "isaac" / "model.usdc"
    usdc.parent.mkdir(parents=True)
    newton_usdc = output / "isaac" / "model_newton.usdc"
    asset_usda = output / "isaac" / "asset.usda"
    physics_usdc = output / "isaac" / "physics.usdc"
    physx_usda = output / "isaac" / "physx.usda"
    mujoco_usda = output / "isaac" / "mujoco.usda"
    urdf = output / "model.urdf"
    usdz = output / "model.usdz"
    report = output / "compile_report.json"
    urdf.write_text("<robot name='reconstruction'/>", encoding="utf-8")
    for path in (
        usdc,
        newton_usdc,
        asset_usda,
        physics_usdc,
        physx_usda,
        mujoco_usda,
    ):
        stage = Usd.Stage.CreateNew(path.as_posix())
        root_prim = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
        stage.SetDefaultPrim(root_prim)
        stage.GetRootLayer().Save()
    usdz.write_bytes(b"usdz")
    artifact_paths = {
        "model.urdf": urdf,
        "model.usdz": usdz,
        "isaac/model.usdc": usdc,
        "isaac/model_newton.usdc": newton_usdc,
        "isaac/asset.usda": asset_usda,
        "isaac/physics.usdc": physics_usdc,
        "isaac/physx.usda": physx_usda,
        "isaac/mujoco.usda": mujoco_usda,
    }
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "success",
                "validation_profile": "generation",
                "asset_structure": physics_output_contract(),
                "checks_run": sorted(REQUIRED_COMPILE_CHECKS),
                "warnings": [],
                "artifacts": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in artifact_paths.items()
                },
                "usd_dependency_closure": {
                    name: [name] for name in artifact_paths if name.startswith("isaac/")
                },
                "metrics": {
                    "asset_structure": exporter_asset_structure_contract(),
                    "backend_validation": {
                        backend: {
                            "status": "success",
                            "checks": sorted(checks),
                            "runtime_status": "not_run",
                        }
                        for backend, checks in REQUIRED_BACKEND_CHECKS.items()
                    },
                },
                "signal_bundle": {"status": "success", "summary": "ok", "signals": []},
            }
        ),
        encoding="utf-8",
    )
    return CompileArtifacts(
        output_dir=output,
        urdf_path=urdf,
        usdc_path=usdc,
        newton_usdc_path=newton_usdc,
        asset_usda_path=asset_usda,
        physics_usdc_path=physics_usdc,
        physx_usda_path=physx_usda,
        mujoco_usda_path=mujoco_usda,
        usdz_path=usdz,
        report_path=report,
        render_paths=(),
        warnings=(),
    )


def _runtime() -> CodexRuntimeSelection:
    return CodexRuntimeSelection(
        model_id="test-model",
        reasoning_effort="high",
        runtime_source="codex_root_thread_state",
        selection_source="inherited_main_codex_runtime",
    )


def _workspace(root: Path) -> Path:
    workspace = root / "workspace"
    codex_dir = workspace / ".codex-run"
    codex_dir.mkdir(parents=True)
    (workspace / "model.py").write_text("object_model = object()\n", encoding="utf-8")
    (workspace / "TASK.md").write_text("Reconstruct one subject.\n", encoding="utf-8")
    (codex_dir / "stdout.jsonl").write_text("{}\n", encoding="utf-8")
    (codex_dir / "final.json").write_text(
        '{"status":"complete","summary":"done"}\n', encoding="utf-8"
    )
    (codex_dir / "stderr.log").write_text("", encoding="utf-8")
    (codex_dir / "timeline.jsonl").write_text("{}\n", encoding="utf-8")
    (workspace / "stage_trace.jsonl").write_text(
        '{"event":"stage.finished","stage":"author","status":"ok"}\n', encoding="utf-8"
    )
    (workspace / "image_observation.json").write_text('{"parts": []}\n', encoding="utf-8")
    (workspace / "image_observation_manifest.json").write_text(
        '{"attempts": 1}\n', encoding="utf-8"
    )
    return workspace


def _write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def _evidence(root: Path) -> ReconstructionEvidence:
    evidence = root / "evidence"
    original = _write(
        evidence / "original" / "scene.gltf",
        '{"buffers":[{"uri":"buffers/scene.bin"}],"images":[{"uri":"textures/a.png"}]}',
    )
    original_buffer = _write(evidence / "original" / "buffers" / "scene.bin", b"mesh")
    original_texture = _write(evidence / "original" / "textures" / "a.png", b"png")
    original_manifest = _write(
        evidence / "original" / "source_dependencies.json", '{"schema_version":1}\n'
    )
    isolation_root = evidence / "subject_isolation"
    isolation_inventory = _write(isolation_root / "component_inventory.json", "{}\n")
    isolation_native = _write(isolation_root / "evidence" / "native_contact_sheet.png", b"native")
    isolation_ids = _write(
        isolation_root / "evidence" / "component_id_contact_sheet.png", b"component-ids"
    )
    isolation_task = _write(isolation_root / "agent" / "TASK.md", "isolate\n")
    isolation_context = _write(isolation_root / "agent" / "context_manifest.json", "{}\n")
    isolation_final = _write(
        isolation_root / "agent" / ".codex-run" / "final.json",
        '{"status":"complete"}\n',
    )
    isolation_stdout = _write(isolation_root / "agent" / ".codex-run" / "stdout.jsonl", "{}\n")
    isolation_stderr = _write(isolation_root / "agent" / ".codex-run" / "stderr.log", "")
    isolation_retained = _write(isolation_root / "retained_source" / "normalized.usdc", b"retained")
    isolation_retained_manifest = _write(
        isolation_root / "retained_source" / "source_manifest.json", "{}\n"
    )
    isolation_manifest = _write(
        isolation_root / "subject_isolation.json",
        json.dumps(
            {
                "artifacts": {
                    "inventory": {
                        "path": isolation_inventory.relative_to(isolation_root).as_posix()
                    },
                    "native_contact_sheet": {
                        "path": isolation_native.relative_to(isolation_root).as_posix()
                    },
                    "component_id_contact_sheet": {
                        "path": isolation_ids.relative_to(isolation_root).as_posix()
                    },
                    "agent_task": {"path": isolation_task.relative_to(isolation_root).as_posix()},
                    "agent_context": {
                        "path": isolation_context.relative_to(isolation_root).as_posix()
                    },
                    "agent_final": {"path": isolation_final.relative_to(isolation_root).as_posix()},
                    "agent_stdout": {
                        "path": isolation_stdout.relative_to(isolation_root).as_posix()
                    },
                    "agent_stderr": {
                        "path": isolation_stderr.relative_to(isolation_root).as_posix()
                    },
                    "retained_source": {
                        "path": isolation_retained.relative_to(isolation_root).as_posix()
                    },
                    "retained_source_manifest": {
                        "path": isolation_retained_manifest.relative_to(isolation_root).as_posix()
                    },
                }
            }
        )
        + "\n",
    )

    mesh_preprocess_root = evidence / "mesh_preprocess"
    normalized = _write(mesh_preprocess_root / "source" / "normalized.usdc", b"normalized")
    source_manifest = _write(mesh_preprocess_root / "source" / "source_manifest.json", "{}\n")
    isolation_binding = _write(mesh_preprocess_root / "source" / "subject_isolation.json", "{}\n")
    preprocess_preview = _write(
        mesh_preprocess_root / "source" / "preview" / "orientation_preview.png", b"preview"
    )
    preprocess_derived = _write(
        mesh_preprocess_root / "source" / "preview" / "derived_45.png", b"derived"
    )
    preprocess_render_manifest = _write(
        mesh_preprocess_root / "source" / "preview" / "render_manifest.json", "{}\n"
    )
    _write(mesh_preprocess_root / "source" / "preview" / "carrier" / "model.urdf", "<robot/>\n")
    _write(
        mesh_preprocess_root / "source" / "preview" / "carrier" / "assets" / "source.glb",
        b"preview-glb",
    )
    semantic = _write(evidence / "semantic.png", b"semantic")
    scaffold = _write(evidence / "scaffold.py", "object_model = object()\n")
    axial = _write(
        mesh_preprocess_root / "axial" / "axial.txt",
        "AXIAL\nBOUNDS 0 1 0 1 0 1\n",
    )
    voxel = _write(mesh_preprocess_root / "axial" / "axial_voxels.json", "{}\n")
    mesh_preprocess_manifest = _write(
        mesh_preprocess_root / "case_manifest.json",
        json.dumps(
            {
                "artifacts": {
                    "normalized_source": {
                        "path": normalized.relative_to(mesh_preprocess_root).as_posix()
                    },
                    "source_manifest": {
                        "path": source_manifest.relative_to(mesh_preprocess_root).as_posix()
                    },
                    "subject_isolation": {
                        "path": isolation_binding.relative_to(mesh_preprocess_root).as_posix()
                    },
                    "orientation_preview": {
                        "path": preprocess_preview.relative_to(mesh_preprocess_root).as_posix()
                    },
                    "derived_45": {
                        "path": preprocess_derived.relative_to(mesh_preprocess_root).as_posix()
                    },
                    "render_manifest": {
                        "path": preprocess_render_manifest.relative_to(
                            mesh_preprocess_root
                        ).as_posix()
                    },
                    "axial": {"path": axial.relative_to(mesh_preprocess_root).as_posix()},
                    "voxel_evidence": {"path": voxel.relative_to(mesh_preprocess_root).as_posix()},
                }
            }
        )
        + "\n",
    )

    overlay_root = evidence / "source_overlay"
    overlay_manifest = _write(overlay_root / "manifest.json", "{}\n")
    author_manifest = _write(evidence / "author" / "evidence_manifest.json", "{}\n")
    overlay_urdf = _write(
        overlay_root / "carrier" / "model.urdf",
        '<robot name="source"><link name="source"><visual><geometry>'
        '<mesh filename="assets/source.glb"/></geometry></visual></link></robot>\n',
    )
    overlay_glb = _write(overlay_root / "carrier" / "assets" / "source.glb", b"glb")
    overlay_metadata = _write(overlay_root / "carrier" / "metadata.json", "{}\n")
    author_manifest.write_text(
        json.dumps(
            {
                "schema": "reconstruction_author_context",
                "allowed_evidence": {
                    "semantic_image": {
                        "path": "inputs/semantic_image.png",
                        "sha256": hashlib.sha256(semantic.read_bytes()).hexdigest(),
                    },
                    "axial": {
                        "path": "inputs/axial.txt",
                        "sha256": hashlib.sha256(axial.read_bytes()).hexdigest(),
                    },
                    "scaffold": {
                        "path": "model.py",
                        "sha256": hashlib.sha256(scaffold.read_bytes()).hexdigest(),
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return ReconstructionEvidence(
        original_source_path=original,
        original_source_dependencies=(
            ReconstructionSourceDependency(path=original_buffer, relative_path="buffers/scene.bin"),
            ReconstructionSourceDependency(path=original_texture, relative_path="textures/a.png"),
        ),
        original_source_manifest_path=original_manifest,
        normalized_source_path=normalized,
        source_manifest_path=source_manifest,
        subject_isolation_binding_path=isolation_binding,
        subject_isolation_manifest_path=isolation_manifest,
        mesh_preprocess_manifest_path=mesh_preprocess_manifest,
        semantic_image_path=semantic,
        scaffold_path=scaffold,
        axial_path=axial,
        source_overlay_manifest_path=overlay_manifest,
        author_evidence_manifest_path=author_manifest,
        source_overlay_urdf_path=overlay_urdf,
        source_overlay_glb_path=overlay_glb,
        source_overlay_dependencies=(
            ReconstructionSourceDependency(path=overlay_metadata, relative_path="metadata.json"),
        ),
        voxel_sidecar_path=voxel,
    )


def _current_approved_evidence(root: Path) -> ReconstructionEvidence:
    evidence = _evidence(root)
    approved_root = root / "approved_inputs"
    source_root = approved_root / "source" / "tree"
    source_files = (
        (
            evidence.original_source_relative_path,
            evidence.original_source_path,
            True,
        ),
        *(
            (dependency.relative_path, dependency.path, False)
            for dependency in evidence.original_source_dependencies
        ),
    )
    frozen_source_paths: dict[str, Path] = {}
    manifest_source_files = []
    for relative_path, original_path, entry in source_files:
        assert relative_path is not None
        frozen = _write(source_root / relative_path, original_path.read_bytes())
        frozen_source_paths[relative_path] = frozen
        manifest_source_files.append(
            {
                "original_path": str(original_path),
                "frozen_path": f"source/tree/{relative_path}",
                "tree_relative_path": relative_path,
                "sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                "byte_count": frozen.stat().st_size,
                "entry": entry,
            }
        )
    frozen_semantic = _write(
        approved_root / "semantic" / "semantic.png",
        evidence.semantic_image_path.read_bytes(),
    )
    frozen_scaffold = _write(
        approved_root / "author" / "scaffold.py",
        evidence.scaffold_path.read_bytes(),
    )
    manifest = {
        "schema": "reconstruction_approved_inputs",
        "schema_version": 1,
        "pipeline": "reconstruction",
        "approved_fingerprint": "a" * 64,
        "source": {
            "tree_anchor": str(evidence.original_source_path.parent),
            "entry_tree_relative_path": evidence.original_source_relative_path,
            "files": manifest_source_files,
        },
        "semantic_image": {
            "original_path": str(evidence.semantic_image_path),
            "frozen_path": "semantic/semantic.png",
            "sha256": hashlib.sha256(frozen_semantic.read_bytes()).hexdigest(),
            "byte_count": frozen_semantic.stat().st_size,
        },
        "scaffold": {
            "original_path": str(evidence.scaffold_path),
            "frozen_path": "author/scaffold.py",
            "sha256": hashlib.sha256(frozen_scaffold.read_bytes()).hexdigest(),
            "byte_count": frozen_scaffold.stat().st_size,
        },
    }
    manifest_path = _write(
        approved_root / "manifest.json",
        json.dumps(manifest, sort_keys=True) + "\n",
    )
    return ReconstructionEvidence(
        original_source_path=frozen_source_paths[str(evidence.original_source_relative_path)],
        original_source_relative_path=evidence.original_source_relative_path,
        approved_original_source_path=str(evidence.original_source_path),
        original_source_dependencies=tuple(
            ReconstructionSourceDependency(
                path=frozen_source_paths[dependency.relative_path],
                relative_path=dependency.relative_path,
                source_logical_path=dependency.source_logical_path,
                approved_original_path=str(dependency.path),
            )
            for dependency in evidence.original_source_dependencies
        ),
        original_source_manifest_path=manifest_path,
        approved_inputs_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        approved_inputs_manifest_byte_count=manifest_path.stat().st_size,
        normalized_source_path=evidence.normalized_source_path,
        source_manifest_path=evidence.source_manifest_path,
        subject_isolation_binding_path=evidence.subject_isolation_binding_path,
        subject_isolation_manifest_path=evidence.subject_isolation_manifest_path,
        mesh_preprocess_manifest_path=evidence.mesh_preprocess_manifest_path,
        semantic_image_path=frozen_semantic,
        scaffold_path=frozen_scaffold,
        axial_path=evidence.axial_path,
        source_overlay_manifest_path=evidence.source_overlay_manifest_path,
        author_evidence_manifest_path=evidence.author_evidence_manifest_path,
        source_overlay_urdf_path=evidence.source_overlay_urdf_path,
        source_overlay_glb_path=evidence.source_overlay_glb_path,
        source_overlay_dependencies=evidence.source_overlay_dependencies,
        voxel_sidecar_path=evidence.voxel_sidecar_path,
    )


def _persist(
    tmp_path: Path,
    *,
    request_text: str = "",
    feedback: bool = False,
    guidance: str | None = None,
) -> Path:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    if feedback:
        out = workspace / "author_feedback" / "view_001"
        out.mkdir(parents=True)
        (out / "feedback.json").write_text('{"ok": true}')
        (out / "escaped.json").symlink_to(workspace / "TASK.md")
    return persist_reconstruction(
        repo_root=tmp_path,
        run_id="recon_test_run",
        title="Test Drawer",
        request_text=request_text,
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"final_status": "complete"},
        artifacts=_artifacts(workspace),
        evidence=_current_approved_evidence(tmp_path),
        label=None,
        guidance=guidance,
    )


def test_persist_reconstruction_preserves_complete_evidence_and_validates(
    tmp_path: Path,
) -> None:
    record_dir = _persist(tmp_path)
    record_id = record_dir.name
    repo = StorageRepo(tmp_path)
    revision_dir = record_dir / "revisions" / "rev_000001"
    record = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    provenance_path = revision_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest = json.loads((revision_dir / "inputs" / "manifest.json").read_text(encoding="utf-8"))

    assert record["pipeline"] == "reconstruction"
    assert record["status"] == "reconstruction_ready"
    assert record["display"]["title"] == "Test Drawer"
    assert record["input_kind"] == "mesh_image"
    assert provenance["schema_version"] == 3
    assert (revision_dir / "request.txt").read_text(encoding="utf-8") == ""
    assert (revision_dir / "traces" / "stage_trace.jsonl").is_file()
    assert (revision_dir / "traces" / "image_observation.json").is_file()
    assert (revision_dir / "traces" / "image_observation_manifest.json").is_file()
    assert manifest["channels"]["source_overlay"] == {
        "manifest_key": "source_overlay_manifest",
        "base_path": "inputs/source/overlay/carrier",
        "urdf_key": "source_overlay_urdf",
        "glb_key": "source_overlay_glb",
        "dependency_keys": ["source_overlay_dependency_0001"],
        "same_frame_as_candidate": True,
        "viewer_only": True,
    }
    files = {item["key"]: item for item in manifest["files"]}
    assert files["original_source"]["path"] == "inputs/source/original/scene.gltf"
    assert files["source_dependency_0001"]["path"] == ("inputs/source/original/buffers/scene.bin")
    assert files["source_dependency_0002"]["path"] == ("inputs/source/original/textures/a.png")
    assert files["source_overlay_urdf"]["path"] == ("inputs/source/overlay/carrier/model.urdf")
    assert files["source_overlay_glb"]["path"] == (
        "inputs/source/overlay/carrier/assets/source.glb"
    )
    assert files["source_overlay_manifest"]["author_input"] is False
    assert files["voxel_sidecar"]["authority"] == "viewer_diagnostic_only"
    closures = manifest["channels"]["audit_closures"]
    assert closures["internal_relative_paths_resolve_against_each_base_path"] is True
    assert set(closures) >= {
        "subject_isolation",
        "mesh_preprocess",
        "source_overlay",
        "author_context",
        "primary_bindings",
    }
    for closure_name in (
        "subject_isolation",
        "mesh_preprocess",
        "source_overlay",
        "author_context",
    ):
        closure = closures[closure_name]
        assert closure["file_keys"]
        for key in closure["file_keys"]:
            assert (revision_dir / files[key]["path"]).is_file()
    for primary_key, audit_key in closures["primary_bindings"].items():
        assert files[primary_key]["sha256"] == files[audit_key]["sha256"]
    scaffold_bytes = (tmp_path / "evidence" / "scaffold.py").read_bytes()
    assert (
        revision_dir / files["author_context_scaffold_snapshot"]["path"]
    ).read_bytes() == scaffold_bytes
    assert scaffold_bytes == (revision_dir / files["author_scaffold"]["path"]).read_bytes()
    author_context_root = revision_dir / closures["author_context"]["base_path"]
    author_context = json.loads((author_context_root / "evidence_manifest.json").read_text())
    allowed = author_context["allowed_evidence"]
    author_references = [
        allowed["semantic_image"],
        allowed["axial"],
        allowed["scaffold"],
    ]
    for reference in author_references:
        target = author_context_root / reference["path"]
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == reference["sha256"]
    for closure_name, manifest_name in (
        ("subject_isolation", "subject_isolation.json"),
        ("mesh_preprocess", "case_manifest.json"),
    ):
        audit_root = revision_dir / closures[closure_name]["base_path"]
        audit_manifest = json.loads((audit_root / manifest_name).read_text())
        for artifact in audit_manifest["artifacts"].values():
            assert (audit_root / artifact["path"]).is_file()
    assert set(manifest["author_inputs"]) == {
        "normalized_source",
        "semantic_image",
        "author_scaffold",
        "axial",
        "author_evidence_manifest",
    }
    assert "original_source" in manifest["host_preprocessing_inputs"]
    assert "source_dependency_0001" in manifest["host_preprocessing_inputs"]
    assert "source_overlay_urdf" in manifest["host_preprocessing_inputs"]
    assert (
        record["evidence"]["manifest_sha256"]
        == hashlib.sha256((revision_dir / "inputs" / "manifest.json").read_bytes()).hexdigest()
    )
    assert {item["key"] for item in record["evidence"]["author_inputs"]} == set(
        manifest["author_inputs"]
    )
    assert {item["key"] for item in record["evidence"]["host_preprocessing_inputs"]} == set(
        manifest["host_preprocessing_inputs"]
    )
    assert validate_data_format(repo, record_ids=[record_id]).errors == []

    display = classify_reconstruction_materialization_displayability(
        repo.layout.record_materialization_dir(record_id),
        revision_dir / "reconstruction.json",
        revision_dir=revision_dir,
    )
    assert display.status == "available"
    assert display.displayable is True
    assert display.issues == ()


def test_current_approved_snapshot_survives_canonical_copy_and_storage_validation(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    evidence = _current_approved_evidence(tmp_path)

    record_dir = persist_reconstruction(
        repo_root=tmp_path,
        run_id="recon_approved_snapshot",
        title="Approved Snapshot",
        request_text="",
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"final_status": "complete"},
        artifacts=_artifacts(workspace),
        evidence=evidence,
        label=None,
    )

    revision_dir = record_dir / "revisions" / "rev_000001"
    input_manifest = json.loads(
        (revision_dir / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    files = {item["key"]: item for item in input_manifest["files"]}
    assert files["original_source"]["source_resolved_path"].endswith(
        "/evidence/original/scene.gltf"
    )
    assert files["source_dependency_0001"]["source_resolved_path"].endswith(
        "/evidence/original/buffers/scene.bin"
    )
    assert validate_data_format(repo, record_ids=[record_dir.name]).errors == []


def test_reconstruction_persistence_rejects_evidence_without_approved_snapshot(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="requires the frozen approved-input snapshot"):
        persist_reconstruction(
            repo_root=tmp_path,
            run_id="recon_missing_approved_snapshot",
            title="Missing Approved Snapshot",
            request_text="",
            runtime=_runtime(),
            workspace_dir=workspace,
            runtime_report={"final_status": "complete"},
            artifacts=_artifacts(workspace),
            evidence=_evidence(tmp_path),
            label=None,
        )

    assert not any(repo.layout.records_root.iterdir())


def test_canonical_copy_rechecks_approved_snapshot_after_prevalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    evidence = _current_approved_evidence(tmp_path)
    verify_snapshot = persist_module._verify_evidence_approved_snapshot

    def verify_then_mutate(current: ReconstructionEvidence) -> None:
        verify_snapshot(current)
        current.original_source_path.write_bytes(
            current.original_source_path.read_bytes() + b"tampered after prevalidation"
        )

    monkeypatch.setattr(
        persist_module,
        "_verify_evidence_approved_snapshot",
        verify_then_mutate,
    )

    with pytest.raises(ValueError, match="differs from approval"):
        persist_reconstruction(
            repo_root=tmp_path,
            run_id="recon_copy_toctou",
            title="Copy TOCTOU",
            request_text="",
            runtime=_runtime(),
            workspace_dir=workspace,
            runtime_report={"final_status": "complete"},
            artifacts=_artifacts(workspace),
            evidence=evidence,
            label=None,
        )
    assert not repo.layout.records_root.exists() or not any(repo.layout.records_root.iterdir())


@pytest.mark.parametrize(
    ("guidance", "recorded"),
    [(None, None), ("  Treat the handles as fixed.\n", "Treat the handles as fixed.")],
)
def test_reconstruction_guidance_is_recorded_in_the_manifest_input(
    tmp_path: Path, guidance: str | None, recorded: str | None
) -> None:
    record_dir = _persist(tmp_path, guidance=guidance)
    revision_dir = record_dir / "revisions" / "rev_000001"
    manifest = json.loads((revision_dir / "reconstruction.json").read_text(encoding="utf-8"))
    # Recorded explicitly for both outcomes: null means the run had no guidance.
    assert "guidance" in manifest["input"] and manifest["input"]["guidance"] == recorded
    assert validate_data_format(StorageRepo(tmp_path), record_ids=[record_dir.name]).errors == []


@pytest.mark.parametrize("guidance", ["", "  ", " padded ", 7, ["list"]])
def test_reconstruction_contract_rejects_malformed_guidance(
    tmp_path: Path, guidance: object
) -> None:
    record_dir = _persist(tmp_path, guidance="Keep the lid closed.")
    record_id = record_dir.name
    repo = StorageRepo(tmp_path)
    revision_dir = record_dir / "revisions" / "rev_000001"
    manifest_path = revision_dir / "reconstruction.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input"]["guidance"] = guidance
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    display = classify_reconstruction_materialization_displayability(
        repo.layout.record_materialization_dir(record_id),
        manifest_path,
        revision_dir=revision_dir,
    )
    assert display.status == "contract_invalid"
    assert any("input.guidance" in issue.message for issue in display.issues)
    errors = validate_data_format(repo, record_ids=[record_id]).errors
    assert any("input.guidance" in str(error) for error in errors)


def test_reconstruction_request_text_is_optional_but_changes_input_kind(tmp_path: Path) -> None:
    record_dir = _persist(tmp_path, request_text="Preserve the drawer pulls.")
    record = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    revision_dir = record_dir / "revisions" / "rev_000001"

    assert record["input_kind"] == "mesh_image_text"
    assert (revision_dir / "request.txt").read_text(encoding="utf-8") == (
        "Preserve the drawer pulls.\n"
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "inputs/semantic/semantic.png",
        "inputs/source/original/buffers/scene.bin",
        "inputs/source/normalized/normalized.usdc",
        "inputs/preprocessing/source_overlay_manifest.json",
        "inputs/source/overlay/carrier/assets/source.glb",
    ],
)
def test_reconstruction_validation_detects_tampered_evidence(
    tmp_path: Path,
    relative_path: str,
) -> None:
    record_dir = _persist(tmp_path)
    record_id = record_dir.name
    evidence_path = record_dir / "revisions" / "rev_000001" / relative_path
    evidence_path.write_bytes(evidence_path.read_bytes() + b"tampered")

    errors = validate_data_format(StorageRepo(tmp_path), record_ids=[record_id]).errors

    assert any("sha256 does not match the input file" in error for error in errors)


def test_reconstruction_validation_detects_broken_audit_relocation(
    tmp_path: Path,
) -> None:
    record_dir = _persist(tmp_path)
    revision_dir = record_dir / "revisions" / "rev_000001"
    input_manifest_path = revision_dir / "inputs" / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    input_manifest["channels"]["audit_closures"]["primary_bindings"]["semantic_image"] = (
        input_manifest["channels"]["audit_closures"]["primary_bindings"]["axial"]
    )
    input_manifest_path.write_text(json.dumps(input_manifest), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(input_manifest_path.read_bytes()).hexdigest()
    projections = (
        (revision_dir / "reconstruction.json", ("input", "manifest_sha256")),
        (revision_dir / "revision.json", ("hashes", "input_manifest_sha256")),
        (revision_dir / "provenance.json", ("evidence", "input_manifest_sha256")),
        (record_dir / "record.json", ("hashes", "input_manifest_sha256")),
        (record_dir / "record.json", ("evidence", "manifest_sha256")),
    )
    for path, (section, field) in projections:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[section][field] = manifest_sha256
        path.write_text(json.dumps(payload), encoding="utf-8")

    errors = validate_data_format(StorageRepo(tmp_path), record_ids=[record_dir.name]).errors

    assert any("semantic_image" in error and "byte-identical" in error for error in errors)


def test_reconstruction_rejects_unsafe_source_dependency_path(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    dependency = evidence.original_source_dependencies[0]

    with pytest.raises(ValueError, match="safe relative path"):
        ReconstructionEvidence(
            original_source_path=evidence.original_source_path,
            original_source_dependencies=(
                ReconstructionSourceDependency(path=dependency.path, relative_path="../mesh.bin"),
            ),
            original_source_manifest_path=evidence.original_source_manifest_path,
            normalized_source_path=evidence.normalized_source_path,
            source_manifest_path=evidence.source_manifest_path,
            subject_isolation_binding_path=evidence.subject_isolation_binding_path,
            subject_isolation_manifest_path=evidence.subject_isolation_manifest_path,
            mesh_preprocess_manifest_path=evidence.mesh_preprocess_manifest_path,
            semantic_image_path=evidence.semantic_image_path,
            scaffold_path=evidence.scaffold_path,
            axial_path=evidence.axial_path,
            source_overlay_manifest_path=evidence.source_overlay_manifest_path,
            author_evidence_manifest_path=evidence.author_evidence_manifest_path,
            source_overlay_urdf_path=evidence.source_overlay_urdf_path,
            source_overlay_glb_path=evidence.source_overlay_glb_path,
            source_overlay_dependencies=evidence.source_overlay_dependencies,
            voxel_sidecar_path=evidence.voxel_sidecar_path,
        )


def test_nonapproved_external_dependency_bundle_cannot_bypass_frozen_snapshot(
    tmp_path: Path,
) -> None:
    canonical = _evidence(tmp_path)
    external_dependency = _write(tmp_path / "outside" / "shared.usda", "#usda 1.0\n")
    bundle = ReconstructionEvidenceBundle(
        semantic_image_path=canonical.semantic_image_path,
        scaffold_path=canonical.scaffold_path,
        axial_path=canonical.axial_path,
        original_source_path=canonical.original_source_path,
        original_source_dependencies=(
            canonical.original_source_path,
            external_dependency,
        ),
        normalized_source_path=canonical.normalized_source_path,
        source_manifest_path=canonical.source_manifest_path,
        subject_isolation_manifest_path=canonical.subject_isolation_manifest_path,
        subject_isolation_binding_path=canonical.subject_isolation_binding_path,
        mesh_preprocess_manifest_path=canonical.mesh_preprocess_manifest_path,
        source_overlay_manifest_path=canonical.source_overlay_manifest_path,
        source_overlay_urdf_path=canonical.source_overlay_urdf_path,
        source_overlay_glb_path=canonical.source_overlay_glb_path,
        source_overlay_dependency_paths=tuple(
            dependency.path for dependency in canonical.source_overlay_dependencies
        ),
        voxel_evidence_path=canonical.voxel_sidecar_path,
    )
    adapted = ReconstructionEvidence.from_bundle(
        bundle,
        author_evidence_manifest_path=canonical.author_evidence_manifest_path,
        original_source_manifest_path=canonical.original_source_manifest_path,
    )

    assert len(adapted.original_source_dependencies) == 1
    dependency = adapted.original_source_dependencies[0]
    assert dependency.relative_path.startswith("_external/")
    assert dependency.source_logical_path is not None
    assert "outside/shared.usda" in dependency.source_logical_path

    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    with pytest.raises(ValueError, match="requires the frozen approved-input snapshot"):
        persist_reconstruction(
            repo_root=tmp_path,
            run_id="external_dependency_run",
            title="External USD Dependency",
            request_text="",
            runtime=_runtime(),
            workspace_dir=workspace,
            runtime_report={"final_status": "complete"},
            artifacts=_artifacts(workspace),
            evidence=adapted,
            label=None,
        )

    assert not any(repo.layout.records_root.iterdir())


def test_author_feedback_survives_workspace_cleanup_as_audit_only(tmp_path: Path) -> None:
    record = _persist(tmp_path, feedback=True)
    revision = record / "revisions" / "rev_000001"
    root = revision / "traces"
    feedback = root / "author_feedback" / "view_001" / "feedback.json"
    assert feedback.read_text() == '{"ok": true}'
    assert not (feedback.parent / "escaped.json").exists()
    index = json.loads((root / "author_feedback_index.json").read_text())
    assert index["files"] == [
        {
            "path": "view_001/feedback.json",
            "sha256": hashlib.sha256(feedback.read_bytes()).hexdigest(),
        }
    ]
    manifest = json.loads((revision / "inputs" / "manifest.json").read_text())
    assert "normalized_source" in manifest["author_inputs"]
    assert not any("feedback" in key for key in manifest["author_inputs"])
    assert (revision / "inputs/author_context/inputs/source_mesh.usdc").is_file()


def test_on_demand_source_render_carrier_is_snapshotted(tmp_path: Path, monkeypatch) -> None:
    import sys

    original = _current_approved_evidence

    def evidence(root):
        e = original(root)
        context = json.loads(e.author_evidence_manifest_path.read_text())
        context["allowed_evidence"]["source_render"] = {
            "available": True,
            "files": {
                "inputs/source_render/model.urdf": hashlib.sha256(
                    e.source_overlay_urdf_path.read_bytes()
                ).hexdigest(),
                "inputs/source_render/assets/source.glb": hashlib.sha256(
                    e.source_overlay_glb_path.read_bytes()
                ).hexdigest(),
            },
        }
        e.author_evidence_manifest_path.write_text(json.dumps(context))
        return e

    monkeypatch.setattr(sys.modules[__name__], "_current_approved_evidence", evidence)
    record = _persist(tmp_path, feedback=True)
    revision = record / "revisions/rev_000001"
    manifest = json.loads((revision / "inputs/manifest.json").read_text())
    assert manifest["source_view_policy"] == "author_requested"
    assert (revision / "inputs/author_context/inputs/source_render/assets/source.glb").is_file()
    assert not validate_data_format(StorageRepo(tmp_path), record_ids=[record.name]).errors


def test_record_id_does_not_repeat_the_reconstruct_prefix() -> None:
    from datetime import datetime, timezone

    created_at = datetime(2026, 9, 28, 9, 50, 34, tzinfo=timezone.utc)
    default = persist_module._record_id(
        "Reconstruct the primary subject object shown in the semantic image", "recon_1", created_at
    )
    labelled = persist_module._record_id("Office cabinet", "recon_1", created_at)

    assert default.startswith("rec_reconstruct-the-primary-subject-object-shown-in_")
    assert labelled.startswith("rec_reconstruct-office-cabinet_")
