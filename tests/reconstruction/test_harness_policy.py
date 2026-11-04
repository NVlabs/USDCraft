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
from dataclasses import replace
from pathlib import Path

import pytest

from tests.reconstruction.test_reconstruction_codex_author import _evidence, _inputs
from tests.reconstruction.test_reconstruction_provider import _prompt_common, _provider_workspace
from usdcraft.pipeline.tools import ProviderToolExecution, ProviderToolResult
from usdcraft.reconstruction.command import fingerprint_reconstruction_preflight
from usdcraft.reconstruction.harness_policy import resolve_harness_policy
from usdcraft.reconstruction.prompt import build_reconstruction_author_prompt
from usdcraft.reconstruction.provider_tools import (
    ReconstructionToolExecutor,
    build_reconstruction_tool_schemas,
)
from usdcraft.reconstruction.workspace import (
    create_reconstruction_author_workspace,
    create_reconstruction_run_workspace,
)


def test_defaults_and_independent_switches() -> None:
    policy = resolve_harness_policy()
    assert policy.to_dict() == {"image_observation": True, "max_turns": 80}
    assert resolve_harness_policy(image_observation=False) == replace(
        policy, image_observation=False
    )
    for options in ({"max_turns": 0}, {"max_turns": True}, {"image_observation": "false"}):
        with pytest.raises(ValueError):
            resolve_harness_policy(**options)


def test_codex_policy_records_no_turn_budget() -> None:
    policy = resolve_harness_policy(image_observation=False, turn_budget=False)
    assert policy.max_turns is None
    assert policy.to_dict() == {"image_observation": False}
    with pytest.raises(ValueError, match="max_turns is a Provider-only option"):
        resolve_harness_policy(max_turns=40, turn_budget=False)


def test_image_observation_changes_approval_fingerprint(tmp_path: Path) -> None:
    _, _, _, preflight = _inputs(tmp_path)
    changed = replace(preflight, request=replace(preflight.request, image_observation=False))
    assert fingerprint_reconstruction_preflight(changed) != fingerprint_reconstruction_preflight(
        preflight
    )


def test_codex_preflight_binds_no_turn_budget(tmp_path: Path) -> None:
    _, _, _, preflight = _inputs(tmp_path)
    assert "max_turns" not in preflight.fingerprint_payload()["request"]["harness"]
    with pytest.raises(ValueError, match="Provider-only"):
        replace(preflight.request, max_turns=39)


def test_tool_surface_always_includes_source_rendering() -> None:
    names = {s["name"] for s in build_reconstruction_tool_schemas()}
    assert {
        "validate_model",
        "measure_face",
        "finish_reconstruction",
        "write_model",
        "check_model",
        "compile_model",
        "check_axial",
        "measure_mesh",
        "render_candidate",
        "compare_part",
        "render_source",
    } <= names


@pytest.mark.parametrize("fail_stage", [None, "check", "compile"])
def test_combined_validation_order_and_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_stage: str | None
) -> None:
    _, _, _, _, workspace = _provider_workspace(tmp_path)
    calls = []

    def runner(name):
        def run(*args, **kwargs):
            calls.append(name)
            return (
                ProviderToolResult.error("repair me")
                if fail_stage == name
                else ProviderToolResult.ok({})
            )

        return run

    executor = ReconstructionToolExecutor(
        workspace,
        check_runner=runner("check"),
        compile_runner=runner("compile"),
    )

    def axial(args):
        calls.append("axial")
        return {"ok": True, "residual": 0.3}

    def image(name, args):
        calls.append("render")
        return ProviderToolExecution({"ok": True}, image_paths=(workspace.root / "view.png",))

    monkeypatch.setattr(executor, "_check_axial", axial)
    monkeypatch.setattr(executor, "_author_feedback", image)
    first_line = workspace.model_path.read_text().splitlines()[0]
    patch = (
        "*** Begin Patch\n*** Update File: model.py\n@@\n "
        + first_line
        + "\n+# authored\n*** End Patch\n"
    )
    assert executor.execute("apply_patch", {"patch": patch}).payload["ok"]
    result = executor.execute("validate_model", {})
    expected = (
        ["check"]
        if fail_stage == "check"
        else ["check", "compile"]
        if fail_stage == "compile"
        else ["check", "compile", "axial", "render"]
    )
    assert calls == expected
    assert result.payload["ok"] == (fail_stage is None)
    assert bool(result.image_paths) == (fail_stage is None)
    if fail_stage is None:
        assert (
            executor.execute(
                "finish_reconstruction", {"status": "complete", "summary": "done"}
            ).completion_status
            == "complete"
        )


def test_staged_inputs_and_prompt_match_policy(tmp_path: Path) -> None:
    repo, _, _, preflight = _inputs(tmp_path)
    run = create_reconstruction_run_workspace(preflight)
    workspace = create_reconstruction_author_workspace(
        preflight,
        run,
        _evidence(run),
        root=run.author_workspace_base,
        author_brief=True,
    )
    assert workspace.source_mesh_path.is_file()
    assert (workspace.root / "inputs/source_render/model.urdf").is_file()
    assert len(workspace.attached_images) == 1
    manifest = json.loads(workspace.context_manifest_path.read_text())
    assert manifest["harness"] == preflight.request.harness_policy.to_dict()
    prompt = build_reconstruction_author_prompt(
        **_prompt_common(preflight, workspace),
        check_command="check_model",
        compile_command="compile_model",
        axial_reencode_command="check_axial",
        reference_access="provider_tools",
    )
    assert all(
        name in prompt
        for name in ("measure_mesh", "render_source", "render_candidate", "compare_part")
    )
    assert "STRICT DESIGN ORDER" not in prompt


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("provider", [False, True])
def test_observation_switch_skips_the_stage_on_both_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, provider: bool
) -> None:
    from types import SimpleNamespace

    from usdcraft.reconstruction import command

    _, _, _, preflight = _inputs(tmp_path)
    preflight = replace(preflight, request=replace(preflight.request, image_observation=enabled))
    run = create_reconstruction_run_workspace(preflight)
    evidence = _evidence(run)
    case = SimpleNamespace(
        axial=evidence.axial_path,
        normalized_source=evidence.normalized_source_path,
        source_manifest=evidence.source_manifest_path,
        subject_isolation=evidence.subject_isolation_binding_path,
        case_manifest=evidence.mesh_preprocess_manifest_path,
        voxel_evidence=evidence.voxel_evidence_path,
    )
    source_overlay = SimpleNamespace(
        manifest=evidence.source_overlay_manifest_path,
        carrier_urdf=evidence.source_overlay_urdf_path,
        carrier_glb=evidence.source_overlay_glb_path,
    )
    monkeypatch.setattr(
        command,
        "run_mesh_preprocess",
        lambda *a, **k: SimpleNamespace(case=case, source_overlay=source_overlay),
    )
    monkeypatch.setattr(
        command,
        "write_retain_all_subject_isolation",
        lambda *a, **k: evidence.subject_isolation_manifest_path,
    )
    calls = []
    observed = object()

    def observe(**kwargs):
        calls.append(kwargs)
        return observed

    monkeypatch.setattr(command, "run_image_observation", observe)
    result = command._build_reconstruction_preprocessing(
        preflight, run, image_observation_runner=observe if provider else None
    )
    assert len(calls) == int(enabled)
    assert result.image_observation is (observed if enabled else None)
    assert result.warnings == ()


def test_codex_author_starts_the_render_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from usdcraft.reconstruction import author, render_service

    _, _, _, preflight = _inputs(tmp_path)
    run = create_reconstruction_run_workspace(preflight)
    workspace = create_reconstruction_author_workspace(
        preflight, run, _evidence(run), root=run.author_workspace_base
    )
    sentinel = object()

    def execute(request, staged, **kwargs):
        assert kwargs["attached_images"] == (workspace.semantic_image_path,)
        return sentinel

    started = []

    class RecordingService:
        def __init__(self, root: Path) -> None:
            started.append(root)

        def __enter__(self) -> RecordingService:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(author, "run_codex_task", execute)
    monkeypatch.setattr(render_service, "CandidateRenderService", RecordingService)
    assert author.run_reconstruction_author(preflight.request, workspace, prompt="test") is sentinel
    assert started == [workspace.root]


@pytest.mark.parametrize("command", ["reconstruction", "reconstruction-codex"])
def test_public_switch_options(command: str) -> None:
    from usdcraft.cli import _build_parser, _reconstruction_harness_options

    parser = _build_parser()
    options = _reconstruction_harness_options(
        parser.parse_args(
            [
                command,
                "source.usdz",
                "--image",
                "image.png",
                "--no-image-observation",
            ]
        )
    )
    expected: dict[str, object] = dict(image_observation=False)
    if command == "reconstruction":
        # Provider only: the Codex arm has no turn limit and no --max-turns.
        expected["max_turns"] = None
    assert options == expected
