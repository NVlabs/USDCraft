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

"""Reconstruction Codex arm, kept in step with the Provider arm.

One Author per request, the reconstruction-target semantic image, the AXIAL brief plus the four
mesh reference views, the shell equivalents of the Provider tools, the scale estimate after the
user's approval, and the same record shape as the Provider arm.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh
from PIL import Image

from tests.prompt_snapshot import assert_matches_golden
from usdcraft import cli as usdcraft_cli
from usdcraft.cli import main as cli_main
from usdcraft.pipeline.codex import ProcessResult, build_codex_task_command
from usdcraft.pipeline.sdk_context import PRELOADED_SDK_DOC_PATHS
from usdcraft.reconstruction.command import (
    AXIAL_PAYLOAD_DIRNAME,
    ReconstructionApprovalError,
    ReconstructionBlockedError,
    ReconstructionInputChangedError,
    ReconstructionPreprocessingResult,
    ReconstructionResult,
    begin_reconstruction_run,
    execute_reconstruction,
    execute_reconstruction_author,
    fingerprint_reconstruction_preflight,
    preflight_reconstruction,
    reconstruction_failure_context,
    resolve_reconstruction_scale,
)
from usdcraft.reconstruction.contract import ReconstructionEvidenceBundle
from usdcraft.reconstruction.isolation_agent import build_subject_isolation_task
from usdcraft.reconstruction.persist import ReconstructionEvidence
from usdcraft.reconstruction.prompt import build_reconstruction_author_prompt
from usdcraft.reconstruction.scale_estimate import ESTIMATED_SCALE_SOURCE
from usdcraft.reconstruction.workspace import (
    ReconstructionRunWorkspace,
    create_reconstruction_author_workspace,
    create_reconstruction_run_workspace,
)


@pytest.fixture(autouse=True)
def temporary_checkout_with_render_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    # These CLI tests use temporary repository roots and fake every render; the checkout and
    # render-setup checks have their own tests in tests/test_setup_cli.py.
    monkeypatch.setattr(usdcraft_cli, "_checkout_root", usdcraft_cli._repo_root)
    monkeypatch.setattr(usdcraft_cli, "_require_render_setup", lambda: None)


def _inputs(
    tmp_path: Path,
    *,
    keep_workspace: bool = False,
    meters_per_unit: float | None = 1.0,
    source_suffix: str = ".obj",
):  # type: ignore[no-untyped-def]
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "scaffold.py").write_text(
        "from sdk import ArticulatedObject, TestContext, TestReport\n"
        "def build_object_model():\n"
        "    return ArticulatedObject('draft')\n"
        "def run_tests():\n"
        "    return TestContext(object_model).report()\n"
        "object_model = build_object_model()\n",
        encoding="utf-8",
    )
    # A GLB is metres by format contract (no unit metadata), so a 1.0-unit box reads as
    # suspected unit-normalized when the user declares no scale; an OBJ needs the declaration.
    source = repo / f"source{source_suffix}"
    trimesh.creation.box(extents=(1.0, 0.8, 0.6)).export(source)
    semantic = repo / "semantic.png"
    Image.new("RGB", (32, 32), (90, 120, 150)).save(semantic)
    preflight = preflight_reconstruction(
        repo_root=repo,
        source=source,
        semantic_image=semantic,
        up_axis="+Z",
        front_axis="-Y",
        meters_per_unit=meters_per_unit,
        guidance="Keep the user-visible latch.",
        model="gpt-test",
        reasoning="high",
        keep_workspace=keep_workspace,
        environ={"PATH": ""},
    )
    return repo, source, semantic, preflight


def _evidence(
    run: ReconstructionRunWorkspace,
) -> ReconstructionEvidenceBundle:
    upstream = run.upstream_dir
    axial = upstream / "axial.txt"
    axial.write_text(
        "AXIAL\nprofile: agent_core_surface\n"
        "micro_grid192: dims_xyz=1,1,1 pitch=1 grid_min_xyz=0,0,0\n"
        "[SLICE_STACK1 axis=Z dims_xyz=1,1,1 pitch=1 grid_min_xyz=0,0,0 "
        "slice_order=-Z->+Z rows=+Y->-Y cols=-X->+X]\n"
        "slice_runs: half-open indices in slice_order\nrun a: slices=0:1\n\n[PATTERN a]\n~\n",
        encoding="utf-8",
    )
    files = {
        "normalized_source_path": upstream / "normalized.usdc",
        "source_manifest_path": upstream / "source_manifest.json",
        "subject_isolation_manifest_path": upstream / "subject_isolation.json",
        "subject_isolation_binding_path": upstream / "subject_isolation_binding.json",
        "mesh_preprocess_manifest_path": upstream / "mesh_preprocess.json",
        "source_overlay_manifest_path": upstream / "source_overlay.json",
        "source_overlay_urdf_path": upstream / "source_overlay.urdf",
        "source_overlay_glb_path": upstream / "source_overlay.glb",
        "voxel_evidence_path": upstream / "voxels.json",
    }
    for path in files.values():
        path.write_text("{}\n", encoding="utf-8")
    return ReconstructionEvidenceBundle(
        scaffold_path=run.frozen_inputs.scaffold_path,
        semantic_image_path=run.frozen_inputs.semantic_image_path,
        axial_path=axial,
        original_source_path=run.frozen_inputs.source_path,
        original_source_dependencies=tuple(
            identity.path for identity in run.frozen_inputs.source_dependencies
        ),
        original_source_tree_root=run.frozen_inputs.source_tree_root,
        approved_inputs_manifest_path=run.frozen_inputs.manifest_path,
        approved_inputs_manifest_sha256=run.frozen_inputs.manifest_sha256,
        approved_inputs_manifest_byte_count=run.frozen_inputs.manifest_byte_count,
        **files,
    )


def _codex_workspace(preflight, run, evidence):  # type: ignore[no-untyped-def]
    """The Codex Author workspace exactly as `execute_reconstruction_author` stages it."""

    return create_reconstruction_author_workspace(
        preflight,
        run,
        evidence,
        root=run.author_workspace_base,
        author_brief=True,
    )


def _axial_payload(recall: float) -> dict[str, object]:
    return {
        "ok": True,
        "revision": None,
        "axial_call": 1,
        "output": {
            "text": "reference mesh — ...\nDEFECTS: none\n",
            "result": {
                "evidence_comparison": {
                    "recall": recall,
                    "input_cell_count": 10,
                    "matched_cell_count": int(round(recall * 10)),
                    "missing_cell_count": 10 - int(round(recall * 10)),
                    "candidate_only_evidence": {"cell_count": 0, "severity": "soft_warning"},
                }
            },
        },
    }


def test_preflight_is_zero_write_and_fixes_the_target_role(tmp_path: Path) -> None:
    repo, _source, _semantic, preflight = _inputs(tmp_path)

    assert preflight.request.to_dict()["semantic_image"]["provenance_role"] == (
        "reconstruction_target"
    )
    assert preflight.request.guidance == "Keep the user-visible latch."
    assert not (repo / "data").exists()
    payload = preflight.fingerprint_payload()
    assert payload["author_evidence_contract"] == {
        "attached_images": ["semantic_image"],
        "attached_image_count": 1,
        "axial": "AXIAL surface-only",
        "source_views": {"initial": 0, "on_demand": True},
    }
    assert payload["request"]["semantic_image"]["provenance_role"] == "reconstruction_target"
    serialized = json.dumps(payload)
    assert "Keep the user-visible latch." in serialized

    isolation_task = build_subject_isolation_task(inventory_sha256="0" * 64)
    assert "Subject-Isolation Agent for a USDCraft Reconstruction run" in isolation_task
    assert "Lab" not in isolation_task


def test_mesh_cleanup_is_optional_defaults_off_and_binds_the_fingerprint(
    tmp_path: Path,
) -> None:
    repo, source, semantic, preflight = _inputs(tmp_path)

    assert preflight.request.mesh_cleanup is False
    payload = preflight.fingerprint_payload()
    assert payload["request"]["mesh_cleanup"] == {
        "requested": False,
        "role": "optional_subject_isolation_agent_for_source_mesh_artifact_cleanup",
    }

    cleanup_preflight = preflight_reconstruction(
        repo_root=repo,
        source=source,
        semantic_image=semantic,
        up_axis="+Z",
        front_axis="-Y",
        meters_per_unit=1.0,
        guidance="Keep the user-visible latch.",
        mesh_cleanup=True,
        model="gpt-test",
        reasoning="high",
        environ={"PATH": ""},
    )
    assert cleanup_preflight.request.mesh_cleanup is True
    assert cleanup_preflight.fingerprint != preflight.fingerprint
    assert not (repo / "data").exists()


def test_preflight_rejects_an_undecodable_semantic_image_without_writing(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    source = repo / "source.obj"
    trimesh.creation.box(extents=(1.0, 0.8, 0.6)).export(source)
    semantic = repo / "semantic.png"
    semantic.write_bytes(b"not a raster image")

    with pytest.raises(ValueError, match="Reconstruction semantic image is invalid"):
        preflight_reconstruction(
            repo_root=repo,
            source=source,
            semantic_image=semantic,
            up_axis="+Z",
            front_axis="-Y",
            meters_per_unit=1.0,
            model="gpt-test",
            reasoning="high",
            environ={"PATH": ""},
        )

    assert not (repo / "data").exists()


def test_confirmation_and_input_binding_are_checked_before_any_workspace_or_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, _semantic, preflight = _inputs(tmp_path)

    monkeypatch.setattr(
        "usdcraft.reconstruction.command.resolve_reconstruction_scale",
        lambda *_args, **_kwargs: pytest.fail("scale estimate must not run before approval"),
    )
    monkeypatch.setattr(
        "usdcraft.reconstruction.command.recover_pending_promotions",
        lambda _repo: pytest.fail("recovery must not run before approval"),
    )
    monkeypatch.setattr(
        "usdcraft.reconstruction.command._build_reconstruction_preprocessing",
        lambda *_args, **_kwargs: pytest.fail("preprocessing must not run before approval"),
    )
    with pytest.raises(ReconstructionApprovalError, match="does not match"):
        execute_reconstruction(
            preflight=preflight,
            approved_fingerprint="0" * 64,
        )
    assert not (repo / "data").exists()

    source.write_text(source.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ReconstructionInputChangedError, match="changed"):
        begin_reconstruction_run(
            preflight,
            approved_fingerprint=preflight.fingerprint,
            run_id="must_not_exist",
        )
    assert not (repo / "data").exists()


def test_author_workspace_is_the_run_root_with_brief_and_mesh_reference_views(
    tmp_path: Path,
) -> None:
    _repo, source, _semantic, preflight = _inputs(tmp_path)
    run = begin_reconstruction_run(
        preflight,
        approved_fingerprint=preflight.fingerprint,
        run_id="recon_test",
    )
    evidence = _evidence(run)

    workspace = _codex_workspace(preflight, run, evidence)

    assert workspace.root == run.author_workspace_base
    assert [path.name for path in workspace.attached_images] == [
        "semantic_image.png",
    ]
    assert workspace.axial_brief_path is not None and workspace.axial_brief_path.is_file()
    assert not (workspace.root / source.name).exists()
    assert not (workspace.root / "normalized.usdc").exists()
    assert workspace.source_mesh_path.read_bytes() == evidence.normalized_source_path.read_bytes()
    assert not (workspace.root / "voxels.json").exists()
    context = workspace.context_manifest_path.read_text(encoding="utf-8")
    assert str(source) not in context
    assert '"author_brief"' in context
    assert "subject_isolation_inventory_contact_sheets_decision_and_trace" in context
    assert '"project_documents_loaded": false' in context
    assert '"user_config_loaded": false' in context
    with pytest.raises(FileExistsError, match="not empty"):
        _codex_workspace(preflight, run, evidence)

    command = build_codex_task_command(
        preflight.request,
        workspace,
        attached_images=workspace.attached_images,
        executable="codex-test",
        ignore_user_config=True,
    )
    assert [command[index + 1] for index, value in enumerate(command) if value == "-c"].count(
        "project_doc_max_bytes=0"
    ) == 1
    attached = [command[index + 1] for index, value in enumerate(command) if value == "--image"]
    assert attached == [str(path) for path in workspace.attached_images]


def test_approved_run_freezes_complete_source_closure_and_never_rereads_user_inputs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    materials = repo / "materials"
    textures = repo / "textures"
    materials.mkdir(parents=True)
    textures.mkdir()
    scaffold = repo / "scaffold.py"
    scaffold.write_text("object_model = object()\n", encoding="utf-8")
    texture = textures / "base color.png"
    Image.new("RGB", (2, 2), (90, 120, 150)).save(texture)
    material = materials / "surface.mtl"
    material.write_text(
        'newmtl surface\nKd 1 1 1\nmap_Kd "../textures/base color.png"\n',
        encoding="utf-8",
    )
    source = repo / "source.obj"
    source.write_text(
        "mtllib materials/surface.mtl\n"
        "usemtl surface\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "f 1/1 2/2 3/3\n",
        encoding="utf-8",
    )
    semantic = repo / "semantic.png"
    Image.new("RGB", (32, 32), (90, 120, 150)).save(semantic)
    approved_bytes = {
        path: path.read_bytes() for path in (source, material, texture, semantic, scaffold)
    }
    preflight = preflight_reconstruction(
        repo_root=repo,
        source=source,
        semantic_image=semantic,
        up_axis="+Z",
        front_axis="-Y",
        meters_per_unit=1.0,
        model="gpt-test",
        reasoning="high",
        environ={"PATH": ""},
    )

    run = begin_reconstruction_run(
        preflight,
        approved_fingerprint=preflight.fingerprint,
        run_id="recon_freeze_closure",
    )

    relative_files = {item.tree_relative_path: item for item in run.frozen_inputs.source_files}
    assert set(relative_files) == {
        "source.obj",
        "materials/surface.mtl",
        "textures/base color.png",
    }
    for relative_path, original in (
        ("source.obj", source),
        ("materials/surface.mtl", material),
        ("textures/base color.png", texture),
    ):
        frozen = relative_files[relative_path]
        assert frozen.staged.path.read_bytes() == approved_bytes[original]
        assert frozen.staged.sha256 == frozen.original.sha256
        assert frozen.staged.byte_count == frozen.original.byte_count
    manifest_bytes = run.frozen_inputs.manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == run.frozen_inputs.manifest_sha256
    assert len(manifest_bytes) == run.frozen_inputs.manifest_byte_count
    manifest = json.loads(manifest_bytes)
    assert manifest["approved_fingerprint"] == preflight.fingerprint
    assert manifest["source"]["entry_tree_relative_path"] == "source.obj"

    for path in approved_bytes:
        path.unlink()
    evidence = _evidence(run)
    workspace = _codex_workspace(preflight, run, evidence)
    assert workspace.model_path.read_bytes() == approved_bytes[scaffold]
    assert workspace.semantic_image_path.read_bytes() == approved_bytes[semantic]
    assert evidence.original_source_path.read_bytes() == approved_bytes[source]
    canonical = ReconstructionEvidence.from_bundle(
        evidence,
        author_evidence_manifest_path=workspace.context_manifest_path,
    )
    assert canonical.approved_original_source_path == str(source.resolve())
    assert {item.relative_path for item in canonical.original_source_dependencies} == {
        "materials/surface.mtl",
        "textures/base color.png",
    }


@pytest.mark.parametrize("target", ["source", "semantic", "scaffold", "manifest"])
def test_frozen_input_tampering_fails_before_author_process(
    tmp_path: Path,
    target: str,
) -> None:
    _repo, _source, _semantic, preflight = _inputs(tmp_path)
    run = begin_reconstruction_run(
        preflight,
        approved_fingerprint=preflight.fingerprint,
        run_id=f"recon_tamper_{target}",
    )
    evidence = _evidence(run)
    target_path = {
        "source": run.frozen_inputs.source_path,
        "semantic": run.frozen_inputs.semantic_image_path,
        "scaffold": run.frozen_inputs.scaffold_path,
        "manifest": run.frozen_inputs.manifest_path,
    }[target]
    target_path.write_bytes(target_path.read_bytes() + b"tampered")

    def forbidden_process(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        pytest.fail("Author process must not start after frozen-input tampering.")

    with pytest.raises(ReconstructionInputChangedError, match="Frozen Reconstruction"):
        execute_reconstruction_author(
            preflight=preflight,
            run_workspace=run,
            evidence=evidence,
            process_runner=forbidden_process,
        )


def test_frozen_input_tampering_during_recovery_fails_before_preprocessing_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, _source, _semantic, preflight = _inputs(tmp_path)

    def tamper_after_freeze(_repo):  # type: ignore[no-untyped-def]
        manifests = tuple(preflight.workspace_base.glob("*/approved_inputs/manifest.json"))
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        semantic = manifests[0].parent / manifest["semantic_image"]["frozen_path"]
        semantic.write_bytes(semantic.read_bytes() + b"tampered")
        return SimpleNamespace(errors=())

    monkeypatch.setattr(
        "usdcraft.reconstruction.command.recover_pending_promotions",
        tamper_after_freeze,
    )
    monkeypatch.setattr(
        "usdcraft.reconstruction.command.run_subject_isolation",
        lambda *_args, **_kwargs: pytest.fail(
            "Preprocessing Agent must not start after frozen-input tampering."
        ),
    )

    with pytest.raises(ReconstructionInputChangedError, match="Frozen Reconstruction semantic"):
        execute_reconstruction(
            preflight=preflight,
            approved_fingerprint=preflight.fingerprint,
        )


def test_freeze_uses_one_physical_path_space_under_symlinked_local_data(
    tmp_path: Path,
) -> None:
    repo, _source, _semantic, preflight = _inputs(tmp_path)
    external_local = tmp_path / "external-local"
    external_local.mkdir()
    (repo / "data").mkdir()
    (repo / "data" / "local").symlink_to(external_local, target_is_directory=True)

    run = begin_reconstruction_run(
        preflight,
        approved_fingerprint=preflight.fingerprint,
        run_id="recon_symlinked_local",
    )

    assert run.root.is_relative_to(external_local.resolve())
    assert run.author_workspace_base.is_relative_to(external_local.resolve())
    assert run.frozen_inputs.source_path.is_file()
    assert run.frozen_inputs.source_path.is_relative_to(run.frozen_inputs.source_tree_root)


def _codex_prompt(workspace, preflight, **overrides):  # type: ignore[no-untyped-def]
    arguments = dict(
        workspace=workspace,
        user_guidance=preflight.request.guidance,
        check_command=("usdcraft", "check", "model.py"),
        compile_command=("usdcraft", "compile", "model.py", "--no-render"),
        axial_reencode_command=(
            "usdcraft",
            "recon-check-axial",
            "inputs/axial.txt",
            "build/isaac/model.usdc",
            "--source",
            "/host/normalized.usdc",
            "--payload-dir",
            "axial_checks",
        ),
        measure_command=("usdcraft", "recon-measure-face", "inputs/axial.txt"),
        request_text="Reconstruct the toaster shown in the semantic image and the source mesh.",
    )
    arguments.update(overrides)
    return build_reconstruction_author_prompt(**arguments)


def test_codex_task_is_the_provider_doctrine_in_shell_wording(tmp_path: Path) -> None:
    _repo, _source, _semantic, preflight = _inputs(tmp_path)
    run = begin_reconstruction_run(
        preflight,
        approved_fingerprint=preflight.fingerprint,
        run_id="recon_prompt",
    )
    workspace = _codex_workspace(preflight, run, _evidence(run))
    shell = _codex_prompt(workspace, preflight, image_observation_text="- body: count=1")
    provider = build_reconstruction_author_prompt(
        workspace=workspace,
        user_guidance=preflight.request.guidance,
        check_command="check_model",
        compile_command="compile_model",
        axial_reencode_command="check_axial",
        reference_access="provider_tools",
        request_text="Reconstruct the toaster shown in the semantic image and the source mesh.",
        image_observation_text="- body: count=1",
    )

    # Semantic decisions and observation advice are identical; tool transport differs.
    def doctrine(text: str) -> str:
        return text.split("EVIDENCE AND DECISIONS", 1)[1].split("AUTHOR TOOLS", 1)[0]

    assert doctrine(shell) == doctrine(provider)
    assert_matches_golden(
        "reconstruction_codex",
        shell,
        paths={str(workspace.root): "<workspace>", str(preflight.request.repo_root): "<repo>"},
    )
    assert "Python/OpenUSD" in shell and "inputs/source_mesh.usdc" in shell
    assert "recon-measure-mesh" in shell and "recon-render" in shell
    assert "recon-check-axial" in shell and "recon-measure-face" in shell
    assert "schema-conforming complete or blocked" in shell
    assert "prior cases, Viewer state" in shell
    assert "not required" in shell and "routing map" in shell
    assert "PRELOADED SDK REFERENCES" in shell
    for provider_only in ("apply_patch", "read_file", "finish_reconstruction"):
        assert provider_only not in shell
    with pytest.raises(ValueError, match="per-face measurement command"):
        _codex_prompt(workspace, preflight, measure_command=None)
    # The scale paragraph reaches the Author on both arms.
    scaled = _codex_prompt(workspace, preflight, scale_note="metres were set from an estimate")
    assert "SCALE\nmetres were set from an estimate" in scaled


@pytest.mark.parametrize("observation_enabled", [True, False])
def test_completed_author_runs_codex_then_final_clean_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observation_enabled: bool,
) -> None:
    _repo, _source, _semantic, preflight = _inputs(tmp_path)
    preflight = replace(
        preflight,
        request=replace(preflight.request, image_observation=observation_enabled),
    )
    preflight = replace(preflight, fingerprint=fingerprint_reconstruction_preflight(preflight))
    run = begin_reconstruction_run(
        preflight,
        approved_fingerprint=preflight.fingerprint,
        run_id="recon_execute",
    )
    evidence = _evidence(run)
    observed: dict[str, object] = {}

    def process_runner(command, *, stdin_text, cwd, timeout_seconds, env):  # type: ignore[no-untyped-def]
        observed["command"] = tuple(command)
        observed["stdin"] = stdin_text
        observed["environment"] = dict(env)
        (cwd / ".codex-run" / "final.json").write_text(
            json.dumps({"status": "complete", "summary": "authored"}),
            encoding="utf-8",
        )
        # The Author's `recon-check-axial --payload-dir` calls leave one payload per call.
        payload_dir = cwd / AXIAL_PAYLOAD_DIRNAME
        payload_dir.mkdir()
        (payload_dir / "check_axial_001.json").write_text(
            json.dumps({**_axial_payload(0.7), "call_id": "call-1"}), encoding="utf-8"
        )
        (payload_dir / "check_axial_002.json").write_text(
            json.dumps({**_axial_payload(0.9), "axial_call": 2, "call_id": "call-2"}),
            encoding="utf-8",
        )
        return ProcessResult(returncode=0, stdout="", stderr="")

    def fake_compile(model_path, output_dir, **kwargs):  # type: ignore[no-untyped-def]
        observed["compile"] = (model_path, output_dir, kwargs)
        return SimpleNamespace(warnings=("author warning",))

    monkeypatch.setattr("usdcraft.reconstruction.command.compile_model", fake_compile)
    result = execute_reconstruction_author(
        preflight=preflight,
        run_workspace=run,
        evidence=evidence,
        process_runner=process_runner,
        scale_note="metres were set from an estimate",
    )

    assert result.status == "compiled_pending_promotion"
    assert result.warnings == ("author warning",)
    report = result.runtime_report
    assert report["image_observation"] == ("author_only" if observation_enabled else "disabled")
    assert report["harness"]["image_observation"] is observation_enabled
    # The Codex arm has no turn budget: the recorded harness claims none.
    assert "max_turns" not in report["harness"]
    assert report["batch"] is None
    assert report["preloaded_docs"] == list(PRELOADED_SDK_DOC_PATHS)
    axial = report["axial_diagnostic"]
    assert isinstance(axial, dict)
    assert axial["calls"] == 2 and axial["first_recall"] == 0.7 and axial["recall"] == 0.9
    stage_timings = report["stage_timings"]
    assert [record["stage"] for record in stage_timings] == ["author", "host_gate"]
    assert all(record["status"] == "ok" for record in stage_timings)
    assert (run.root / "stage_trace.jsonl").is_file()
    assert result.author_workspace.root == run.author_workspace_base

    task = (result.author_workspace.root / "TASK.md").read_text(encoding="utf-8")
    assert task == observed["stdin"]
    normalized = evidence.normalized_source_path
    assert (
        f"recon-check-axial {result.author_workspace.axial_path} "
        f"{result.author_workspace.build_dir / 'isaac' / 'model.usdc'} --source {normalized} "
        f"--payload-dir {result.author_workspace.root / AXIAL_PAYLOAD_DIRNAME}"
    ) in task
    assert f"recon-measure-face {result.author_workspace.axial_path} --face=<face>" in task
    assert "SCALE\nmetres were set from an estimate" in task
    command = observed["command"]
    assert isinstance(command, tuple)
    attached = [command[index + 1] for index, value in enumerate(command) if value == "--image"]
    assert attached == [str(path) for path in result.author_workspace.attached_images]
    assert len(attached) == 1
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["USDCRAFT_RECONSTRUCTION_WORKSPACE"] == str(result.author_workspace.root)
    compile_call = observed["compile"]
    assert isinstance(compile_call, tuple)
    assert compile_call[2] == {
        "clean": True,
        "render": False,
        "require_generation_root_floating": True,
    }


def test_blocked_author_retains_the_workspace_for_diagnosis(
    tmp_path: Path,
) -> None:
    _repo, _source, _semantic, preflight = _inputs(tmp_path)
    run = begin_reconstruction_run(
        preflight,
        approved_fingerprint=preflight.fingerprint,
        run_id="recon_blocked",
    )
    evidence = _evidence(run)

    def process_runner(command, *, stdin_text, cwd, timeout_seconds, env):  # type: ignore[no-untyped-def]
        (cwd / ".codex-run" / "final.json").write_text(
            json.dumps({"status": "blocked", "summary": "evidence conflict"}),
            encoding="utf-8",
        )
        return ProcessResult(returncode=0, stdout="", stderr="")

    with pytest.raises(ReconstructionBlockedError, match="evidence conflict") as caught:
        execute_reconstruction_author(
            preflight=preflight,
            run_workspace=run,
            evidence=evidence,
            process_runner=process_runner,
        )
    assert caught.value.workspace_dir == run.author_workspace_base
    assert (run.author_workspace_base / "TASK.md").is_file()


def test_formal_cli_preflight_is_compact_and_zero_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, source, semantic, _preflight = _inputs(tmp_path)

    assert (
        cli_main(
            [
                "reconstruction-codex",
                str(source),
                "--image",
                str(semantic),
                "--up-axis=+Z",
                "--front-axis=-Y",
                "--meters-per-unit",
                "1",
                "--guidance",
                "Keep the user-visible latch.",
                "--label",
                "User Mailbox",
                "--model",
                "gpt-test",
                "--reasoning",
                "high",
                "--repo-root",
                str(repo),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output.count("model=gpt-test") == 1
    assert output.count("reasoning_effort=high") == 1
    assert f'source_path="{source}"' in output
    assert f'semantic_image_path="{semantic}"' in output
    assert output.count('label="User Mailbox"') == 1
    assert 'guidance="Keep the user-visible latch."' in output
    assert "mesh_cleanup=skipped_retain_all" in output
    assert "approval_fingerprint=" in output
    assert not (repo / "data").exists()


@pytest.mark.parametrize("direct", [False, True])
def test_formal_cli_prints_the_single_record_result_block(
    direct: bool,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, semantic, preflight = _inputs(tmp_path)
    result = ReconstructionResult(
        status="reconstruction_ready",
        run_id="recon_cli",
        record_id="rec_one",
        record_dir=repo / "data" / "records" / "rec_one",
        model_path=repo / "data" / "records" / "rec_one" / "model.py",
        workspace_dir=None,
        warnings=("Promotion recovery: recovered stale transaction",),
    )
    captured: dict[str, object] = {}

    def fake_execute(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return result

    monkeypatch.setattr(
        "usdcraft.reconstruction.command.preflight_reconstruction",
        lambda **_kwargs: preflight,
    )
    monkeypatch.setattr("usdcraft.reconstruction.command.execute_reconstruction", fake_execute)

    assert (
        cli_main(
            [
                "reconstruction-codex",
                str(source),
                "--image",
                str(semantic),
                *(["--direct"] if direct else ["--confirm-reconstruction", preflight.fingerprint]),
                "--repo-root",
                str(repo),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert captured["approved_fingerprint"] == (None if direct else preflight.fingerprint)
    assert captured["direct"] is direct
    assert output.splitlines()[:6] == [
        "status=reconstruction_ready",
        "pipeline=reconstruction",
        "provider=codex-cli",
        "run_id=recon_cli",
        "record_id=rec_one",
        f"record_dir={repo / 'data' / 'records' / 'rec_one'}",
    ]
    assert 'warnings=["Promotion recovery: recovered stale transaction"]' in output
    assert "viewer=uv run usdcraft viewer --repo-root" in output
    payload = result.to_dict()
    assert payload["provider"] == "codex-cli"
    assert payload["pipeline"] == "reconstruction"


@pytest.mark.parametrize("keep_workspace", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_execute_reconstruction_runs_the_provider_chain_once_and_promotes_one_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keep_workspace: bool,
    direct: bool,
) -> None:
    repo, source, semantic, preflight = _inputs(tmp_path, keep_workspace=keep_workspace)
    calls: dict[str, object] = {"preprocess": 0, "persisted": [], "recovery_after_freeze": False}
    observed: dict[str, object] = {}

    def fake_recovery(_repo):  # type: ignore[no-untyped-def]
        manifests = tuple(preflight.workspace_base.glob("*/approved_inputs/manifest.json"))
        assert len(manifests) == 1
        source.unlink()
        semantic.unlink()
        calls["recovery_after_freeze"] = True
        return SimpleNamespace(errors=())

    def fake_preprocessing(current_preflight, run, **kwargs):  # type: ignore[no-untyped-def]
        calls["preprocess"] = int(calls["preprocess"]) + 1
        observed["run"] = run.root
        observed["authors"] = run.author_workspace_base
        placeholder = SimpleNamespace()
        return ReconstructionPreprocessingResult(
            isolation=placeholder,
            mesh_preprocess=placeholder,
            evidence=_evidence(run),
        )

    def fake_author(*, preflight, run_workspace, evidence, **kwargs):  # type: ignore[no-untyped-def]
        observed["author_kwargs"] = set(kwargs)
        workspace = _codex_workspace(preflight, run_workspace, evidence)
        return SimpleNamespace(
            author_workspace=workspace,
            artifacts=SimpleNamespace(),
            runtime_report={"stage_timings": []},
            warnings=("author warning",),
        )

    def fake_persist(**kwargs):  # type: ignore[no-untyped-def]
        persisted = calls["persisted"]
        assert isinstance(persisted, list)
        persisted.append(kwargs)
        record = Path(kwargs["repo_root"]) / "data" / "records" / "rec_codex"
        model = record / "revisions" / "rev_000001" / "model.py"
        model.parent.mkdir(parents=True)
        model.write_text("# canonical\n", encoding="utf-8")
        return record

    monkeypatch.setattr("usdcraft.reconstruction.command.recover_pending_promotions", fake_recovery)
    monkeypatch.setattr(
        "usdcraft.reconstruction.command._build_reconstruction_preprocessing", fake_preprocessing
    )
    monkeypatch.setattr(
        "usdcraft.reconstruction.command.execute_reconstruction_author", fake_author
    )
    monkeypatch.setattr("usdcraft.reconstruction.command.persist_reconstruction", fake_persist)

    result = execute_reconstruction(
        preflight=preflight,
        approved_fingerprint=None if direct else preflight.fingerprint,
        direct=direct,
        batch_context={"row_id": "toast_repeat01"},
    )

    assert calls["persisted"][0]["runtime_report"]["authorization"] == (
        "direct" if direct else "fingerprint"
    )
    assert calls["persisted"][0]["runtime_report"]["batch"] == {"row_id": "toast_repeat01"}
    assert calls["preprocess"] == 1 and calls["recovery_after_freeze"] is True
    assert {"image_observation", "process_runner", "scale_note", "stage_timer"} <= observed[
        "author_kwargs"
    ]
    persisted = calls["persisted"]
    assert isinstance(persisted, list) and len(persisted) == 1
    persist_call = persisted[0]
    assert persist_call["authoring"].agent_harness == "codex"
    assert persist_call["guidance"] == "Keep the user-visible latch."
    assert persist_call["runtime_report"]["scale"]["source"] == "explicit"
    assert persist_call["runtime_report"]["scale"]["meters_per_unit"] == 1.0
    assert persist_call["workspace_dir"] == observed["authors"]
    assert result.status == "reconstruction_ready"
    assert result.record_id == "rec_codex"
    assert (
        result.model_path
        == repo / "data" / "records" / "rec_codex" / "revisions" / ("rev_000001") / "model.py"
    )
    assert result.warnings == ("author warning",)
    assert result.workspace_dir == (observed["authors"] if keep_workspace else None)
    assert observed["run"].exists() is keep_workspace
    assert observed["authors"].exists() is keep_workspace
    assert (repo / "data" / "records" / "rec_codex").is_dir()


def _scale_runner(reply: dict[str, object] | None, *, returncode: int = 0):  # type: ignore[no-untyped-def]
    calls: list[dict[str, object]] = []

    def runner(command, *, stdin_text, cwd, timeout_seconds, env):  # type: ignore[no-untyped-def]
        calls.append({"command": tuple(command), "stdin": stdin_text, "timeout": timeout_seconds})
        if reply is not None:
            (cwd / ".codex-run" / "final.json").write_text(json.dumps(reply), encoding="utf-8")
        return SimpleNamespace(returncode=returncode, stdout="", stderr="estimator failed")

    return runner, calls


def test_scale_estimate_runs_after_approval_and_is_recorded_without_refingerprinting(
    tmp_path: Path,
) -> None:
    _repo, _source, semantic, preflight = _inputs(
        tmp_path, meters_per_unit=None, source_suffix=".glb"
    )
    runner, calls = _scale_runner(
        {
            "category": "toaster",
            "longest_dimension_m": 0.3,
            "confidence": "medium",
            "rationale": "a two-slice toaster is about 30 cm long",
        }
    )

    resolution = resolve_reconstruction_scale(preflight, process_runner=runner)

    # The 1.0-unit OBJ box carries no units: suspected unit-normalized, estimated from the image.
    assert resolution.inspection.suspected_unit_normalized is True
    assert resolution.inspection.matched_target == 1.0
    assert resolution.source == ESTIMATED_SCALE_SOURCE
    assert resolution.estimate is not None and resolution.estimate.category == "toaster"
    resolved = resolution.preflight
    assert resolved.request.meters_per_unit == pytest.approx(0.3)
    assert resolved.request.meters_per_unit_source == ESTIMATED_SCALE_SOURCE
    assert resolved.request.scale_estimate["longest_dimension_m"] == 0.3
    assert resolved.fingerprint == preflight.fingerprint
    assert "0.30 m" in (resolution.author_note or "")
    report = resolution.report()
    assert report["source"] == ESTIMATED_SCALE_SOURCE and report["meters_per_unit"] == 0.3
    assert report["estimate"]["model"] == "gpt-test"
    assert len(calls) == 1
    command = calls[0]["command"]
    assert isinstance(command, tuple) and command[1] == "exec"
    assert "--output-schema" in command
    image_args = [command[i + 1] for i, value in enumerate(command) if value == "--image"]
    assert len(image_args) == 1 and Path(image_args[0]).name == f"input_1_{semantic.name}"
    assert "Answer with exactly one JSON object" in str(calls[0]["stdin"])
    # The Codex arm has no time limit: neither the request nor the estimate call has a timeout.
    assert preflight.request.timeout_seconds is None
    assert "timeout_seconds" not in preflight.fingerprint_payload()["request"]["codex"]
    assert calls[0]["timeout"] is None

    # The frozen request records the Host derivation under the user's approved fingerprint
    # (the resolved preflight is frozen directly; approval was verified on the user's request).
    run = create_reconstruction_run_workspace(resolved)
    request_json = json.loads((run.root / "request.json").read_text(encoding="utf-8"))
    assert request_json["approved_fingerprint"] == preflight.fingerprint
    assert request_json["source"]["meters_per_unit"] == pytest.approx(0.3)
    assert request_json["source"]["meters_per_unit_source"] == ESTIMATED_SCALE_SOURCE


def test_scale_estimate_is_skipped_for_a_declaration_and_degrades_to_metres_on_failure(
    tmp_path: Path,
) -> None:
    _repo, _source, _semantic, declared = _inputs(tmp_path, meters_per_unit=1.0)
    runner, calls = _scale_runner(None)
    resolution = resolve_reconstruction_scale(declared, process_runner=runner)
    assert resolution.source == "explicit" and resolution.estimate is None
    assert resolution.preflight is declared and calls == []

    _repo2, _source2, _semantic2, unknown = _inputs(
        tmp_path / "second", meters_per_unit=None, source_suffix=".glb"
    )
    failing, failing_calls = _scale_runner(None, returncode=1)
    degraded = resolve_reconstruction_scale(unknown, process_runner=failing)
    assert len(failing_calls) == 2  # one retry
    assert degraded.preflight is unknown and degraded.estimate is None
    assert degraded.source == "format_contract"
    assert len(degraded.warnings) == 1 and "treated as metres" in degraded.warnings[0]
    assert degraded.report()["meters_per_unit"] == 1.0


def test_shared_preprocessing_failure_reports_retained_run_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, _source, _semantic, preflight = _inputs(tmp_path)
    monkeypatch.setattr(
        "usdcraft.reconstruction.command.recover_pending_promotions",
        lambda _repo: SimpleNamespace(errors=()),
    )
    monkeypatch.setattr(
        "usdcraft.reconstruction.command._build_reconstruction_preprocessing",
        lambda _preflight, _run, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("render unavailable")
        ),
    )

    with pytest.raises(RuntimeError, match="render unavailable") as caught:
        execute_reconstruction(
            preflight=preflight,
            approved_fingerprint=preflight.fingerprint,
        )

    context = reconstruction_failure_context(caught.value)
    assert context is not None
    assert context.workspace_dir.is_dir()
    assert context.workspace_dir.name == context.run_id


def test_author_embeds_external_image_observation_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usdcraft.reconstruction.observation import ImageObservationArtifacts

    _repo, _source, _semantic, preflight = _inputs(tmp_path)
    run = begin_reconstruction_run(
        preflight,
        approved_fingerprint=preflight.fingerprint,
        run_id="recon_observed",
    )
    evidence = _evidence(run)
    observation_root = tmp_path / "image_observation"
    observation_root.mkdir()
    inventory_payload = {
        "status": "complete",
        "parts": [
            {
                "name": "drawer_front",
                "visible_count": 1,
                "mobility": "movable",
                "level": "major_part",
                "visibility": "visible",
                "geometry": "flat rectangular panel, most of the front",
                "arrangement": "front face",
                "affordances": ["pull"],
                "attaches_to": None,
                "duplicate_of": None,
                "mirror_of": None,
                "carried_as": "none",
                "notes": None,
            }
        ],
        "removals": [],
        "summary": "one drawer",
    }
    inventory_path = observation_root / "observation_inventory.json"
    inventory_path.write_text(json.dumps(inventory_payload), encoding="utf-8")
    observation = ImageObservationArtifacts(
        root=observation_root,
        inventory_path=inventory_path,
        manifest_path=inventory_path,
        inventory=inventory_payload,
        attempts=1,
    )

    def process_runner(command, *, stdin_text, cwd, timeout_seconds, env):  # type: ignore[no-untyped-def]
        (cwd / ".codex-run" / "final.json").write_text(
            json.dumps({"status": "complete", "summary": "authored"}),
            encoding="utf-8",
        )
        return ProcessResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "usdcraft.reconstruction.command.compile_model",
        lambda *args, **kwargs: SimpleNamespace(warnings=()),
    )
    result = execute_reconstruction_author(
        preflight=preflight,
        run_workspace=run,
        evidence=evidence,
        image_observation=observation,
        process_runner=process_runner,
    )

    assert result.runtime_report["image_observation"] == "external_inventory"
    assert result.runtime_report["axial_diagnostic"] is None
    task = (result.author_workspace.root / "TASK.md").read_text(encoding="utf-8")
    assert "INDEPENDENT IMAGE OBSERVATION" in task
    assert "- drawer_front: count=1, movable, major_part, visible" in task
    assert "rename, merge, split, add or discard" in task
    assert "`inputs/axial_brief.txt`" in task


def test_direct_still_checks_inputs_and_rejects_ambiguous_authorization(tmp_path):
    repo, source, _, preflight = _inputs(tmp_path)
    with pytest.raises(ReconstructionApprovalError):
        execute_reconstruction(
            preflight=preflight, direct=True, approved_fingerprint=preflight.fingerprint
        )
    with pytest.raises(ReconstructionApprovalError):
        execute_reconstruction(preflight=preflight)
    source.write_text(source.read_text() + "# changed\n")
    with pytest.raises(ReconstructionInputChangedError):
        execute_reconstruction(preflight=preflight, direct=True)
    assert not (repo / "data").exists()
