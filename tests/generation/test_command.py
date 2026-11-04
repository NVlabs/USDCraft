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
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from storage.artifact_contract import physics_output_contract
from usdcraft.compile.command import CompileArtifacts
from usdcraft.generation.command import (
    GenerationApprovalError,
    GenerationBlockedError,
    GenerationInputChangedError,
    execute_generation,
    generation_failure_context,
    preflight_generation,
)
from usdcraft.pipeline.codex import CodexExecutionError
from usdcraft.pipeline.contracts import (
    CodexCompletionStatus,
    CodexFinalMessage,
    CodexRunResult,
    CodexRuntimeSelection,
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    return root


def _write_image(path: Path, color: tuple[int, int, int] = (80, 120, 160)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _codex_result(status: CodexCompletionStatus) -> CodexRunResult:
    return CodexRunResult(
        command=("codex",),
        returncode=0,
        stdout="{}\n",
        stderr="",
        elapsed_seconds=1.25,
        final_message=CodexFinalMessage(status=status, summary="done"),
    )


def test_generate_promotes_then_removes_successful_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_codex(request, workspace, *, prompt, executable):  # noqa: ANN001
        captured["workspace"] = workspace.root
        captured["prompt"] = prompt
        captured["executable"] = executable
        workspace.codex_dir.mkdir(exist_ok=True)
        workspace.codex_stdout_path.write_text("{}\n", encoding="utf-8")
        workspace.codex_stderr_path.write_text("", encoding="utf-8")
        workspace.codex_last_message_path.write_text(
            json.dumps({"status": "complete", "summary": "done"}), encoding="utf-8"
        )
        workspace.model_path.write_text("object_model = object()\n", encoding="utf-8")
        return _codex_result(CodexCompletionStatus.COMPLETE)

    def fake_compile(model_path: Path, output_dir: Path, **kwargs: object) -> CompileArtifacts:
        captured["compile_kwargs"] = kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "urdf": output_dir / "model.urdf",
            "usdc": output_dir / "isaac" / "model.usdc",
            "newton_usdc": output_dir / "isaac" / "model_newton.usdc",
            "asset_usda": output_dir / "isaac" / "asset.usda",
            "physics_usdc": output_dir / "isaac" / "physics.usdc",
            "physx_usda": output_dir / "isaac" / "physx.usda",
            "mujoco_usda": output_dir / "isaac" / "mujoco.usda",
            "usdz": output_dir / "model.usdz",
            "report": output_dir / "compile_report.json",
        }
        paths["usdc"].parent.mkdir()
        for key, path in paths.items():
            path.write_text("{}" if key == "report" else key, encoding="utf-8")
        return CompileArtifacts(
            output_dir=output_dir,
            urdf_path=paths["urdf"],
            usdc_path=paths["usdc"],
            newton_usdc_path=paths["newton_usdc"],
            asset_usda_path=paths["asset_usda"],
            physics_usdc_path=paths["physics_usdc"],
            physx_usda_path=paths["physx_usda"],
            mujoco_usda_path=paths["mujoco_usda"],
            usdz_path=paths["usdz"],
            report_path=paths["report"],
            render_paths=(),
            warnings=(),
        )

    def fake_persist(**kwargs) -> Path:  # noqa: ANN003
        captured["persist"] = kwargs
        record_dir = root / "data" / "records" / "rec_test"
        model_path = record_dir / "revisions" / "rev_000001" / "model.py"
        model_path.parent.mkdir(parents=True)
        model_path.write_text("object_model = object()\n", encoding="utf-8")
        return record_dir

    monkeypatch.setattr("usdcraft.generation.command.run_codex", fake_run_codex)
    monkeypatch.setattr("usdcraft.generation.command.compile_model", fake_compile)
    monkeypatch.setattr("usdcraft.generation.command.persist_generation", fake_persist)

    preflight = preflight_generation(
        repo_root=root,
        prompt="Make a lamp",
        model="gpt-test",
        reasoning="ultra",
    )
    result = execute_generation(
        preflight=preflight,
        approved_fingerprint=preflight.fingerprint,
    )

    workspace = captured["workspace"]
    assert isinstance(workspace, Path)
    assert not workspace.exists()
    assert result.status == "generation_ready"
    assert result.record_id == "rec_test"
    assert result.run_id
    assert result.workspace_dir is None
    assert result.model_path.is_file()
    prompt = str(captured["prompt"])
    assert "usdcraft.cli check" in prompt
    assert "usdcraft.cli compile" in prompt
    assert "--no-render" in prompt
    assert "Inspect the PNG views" not in prompt
    assert "compare the inspected render" not in prompt
    assert "generation-render" in prompt
    assert captured["executable"] == preflight.codex_executable
    assert captured["compile_kwargs"] == {
        "clean": True,
        "render": False,
        "require_generation_root_floating": True,
    }
    persist = captured["persist"]
    assert isinstance(persist, dict)
    assert persist["image_path"] is None
    assert persist["runtime_report"]["final_status"] == "complete"
    assert persist["runtime"].to_dict() == {
        "model_id": "gpt-test",
        "reasoning_effort": "ultra",
        "runtime_source": "explicit_generation_request",
        "selection_source": "explicit_generation_request",
    }


def test_generate_keeps_blocked_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)

    def fake_run_codex(_request, workspace, *, prompt, executable):  # noqa: ANN001
        assert executable
        assert prompt
        return _codex_result(CodexCompletionStatus.BLOCKED)

    monkeypatch.setattr("usdcraft.generation.command.run_codex", fake_run_codex)

    preflight = preflight_generation(
        repo_root=root,
        prompt="Make a lamp",
        model="gpt-test",
        reasoning="high",
    )
    with pytest.raises(GenerationBlockedError) as exc_info:
        execute_generation(
            preflight=preflight,
            approved_fingerprint=preflight.fingerprint,
        )

    assert exc_info.value.workspace_dir.is_dir()
    assert (exc_info.value.workspace_dir / "request.json").is_file()
    context = generation_failure_context(exc_info.value)
    assert context is not None
    assert context.run_id == context.workspace_dir.name
    assert context.workspace_dir == exc_info.value.workspace_dir
    assert context.workspace_dir.is_absolute()


@pytest.mark.parametrize(
    ("failure_stage", "failure"),
    [
        pytest.param("codex", CodexExecutionError("Codex exited with status 1"), id="codex-exit"),
        pytest.param("codex", CodexExecutionError("invalid final JSON"), id="codex-json"),
        pytest.param("compile", RuntimeError("final Host compile failed"), id="host-compile"),
        pytest.param("promotion", OSError("promotion failed"), id="promotion"),
    ],
)
def test_every_post_workspace_failure_keeps_type_and_exposes_workspace_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    failure: Exception,
) -> None:
    root = _repo(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_codex(_request, workspace, *, prompt, executable):  # noqa: ANN001
        assert prompt
        assert executable
        captured["workspace"] = workspace
        if failure_stage == "codex":
            raise failure
        return _codex_result(CodexCompletionStatus.COMPLETE)

    def fake_compile(*_args: object, **_kwargs: object) -> object:
        if failure_stage == "compile":
            raise failure
        return object()

    def fake_persist(**_kwargs: object) -> Path:
        assert failure_stage == "promotion"
        raise failure

    monkeypatch.setattr("usdcraft.generation.command.run_codex", fake_run_codex)
    monkeypatch.setattr("usdcraft.generation.command.compile_model", fake_compile)
    monkeypatch.setattr("usdcraft.generation.command.persist_generation", fake_persist)

    preflight = preflight_generation(
        repo_root=root,
        prompt="Make a lamp",
        model="gpt-test",
        reasoning="high",
    )
    with pytest.raises(type(failure)) as exc_info:
        execute_generation(
            preflight=preflight,
            approved_fingerprint=preflight.fingerprint,
        )

    assert exc_info.value is failure
    context = generation_failure_context(exc_info.value)
    assert context is not None
    workspace = captured["workspace"]
    assert context.run_id == workspace.run_id
    assert context.workspace_dir == workspace.root
    assert context.workspace_dir.is_absolute()
    assert context.workspace_dir.is_dir()


def test_post_workspace_failure_preserves_original_exception_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    cause = ValueError("bad JSON token")

    def fake_run_codex(*_args: object, **_kwargs: object) -> CodexRunResult:
        raise CodexExecutionError("Codex final response is not valid JSON") from cause

    monkeypatch.setattr("usdcraft.generation.command.run_codex", fake_run_codex)
    preflight = preflight_generation(
        repo_root=root,
        prompt="Make a lamp",
        model="gpt-test",
        reasoning="high",
    )

    with pytest.raises(CodexExecutionError) as exc_info:
        execute_generation(
            preflight=preflight,
            approved_fingerprint=preflight.fingerprint,
        )

    assert exc_info.value.__cause__ is cause
    assert generation_failure_context(exc_info.value) is not None


def test_preflight_fingerprint_binds_all_execution_inputs(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    image = tmp_path / "reference.png"
    _write_image(image, (10, 20, 30))
    first = preflight_generation(
        repo_root=root,
        prompt="  Make a lamp  ",
        image=image,
        model="gpt-test",
        reasoning="future-tier",
        label="Lamp",
        keep_workspace=True,
    )

    assert first.request.prompt == "Make a lamp"
    assert first.request.image_path == image.resolve()
    assert first.fingerprint_payload()["input"]["kind"] == "text_image"
    assert first.fingerprint_payload()["input"]["image_role"] == "inspiration"
    assert first.fingerprint_payload()["execution"]["keep_workspace"] is True
    # The Codex arm has no time limit, so the approved plan binds none.
    assert "timeout_seconds" not in first.fingerprint_payload()["codex"]
    assert first.fingerprint_payload()["artifact_contract"] == {
        "generation_manifest_schema_version": 3,
        "compile_report_schema_version": 2,
        "validation_profile": "generation",
    }
    assert first.fingerprint_payload()["outputs"] == physics_output_contract()

    _write_image(image, (30, 20, 10))
    changed = preflight_generation(
        repo_root=root,
        prompt="Make a lamp",
        image=image,
        model="gpt-test",
        reasoning="future-tier",
        label="Lamp",
        keep_workspace=True,
    )
    assert changed.fingerprint != first.fingerprint


def test_explicit_prompt_overrides_image_only_default(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    image = tmp_path / "reference.jpg"
    _write_image(image)

    preflight = preflight_generation(
        repo_root=root,
        prompt="  Build exactly one cart with a child seat.  ",
        image=image,
        model="gpt-test",
        reasoning="high",
    )

    assert preflight.request.prompt == "Build exactly one cart with a child seat."


def test_preflight_rejects_missing_text_and_image(tmp_path: Path) -> None:
    root = _repo(tmp_path)

    with pytest.raises(ValueError, match="text prompt or one inspiration image"):
        preflight_generation(
            repo_root=root,
            model="gpt-test",
            reasoning="high",
        )


def test_preflight_rejects_an_undecodable_image_without_writing(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    image = tmp_path / "reference.png"
    image.write_bytes(b"not a raster image")

    with pytest.raises(ValueError, match="Generation inspiration image is invalid"):
        preflight_generation(
            repo_root=root,
            prompt="Make a lamp",
            image=image,
            model="gpt-test",
            reasoning="high",
        )

    assert not (root / "data").exists()


def test_preflight_fingerprint_changes_for_each_execution_choice(tmp_path: Path) -> None:
    root = _repo(tmp_path)

    def build(**overrides):  # noqa: ANN003, ANN202
        values = {
            "repo_root": root,
            "prompt": "Make a lamp",
            "model": "gpt-test",
            "reasoning": "high",
            "label": "Lamp",
            "keep_workspace": False,
        }
        values.update(overrides)
        return preflight_generation(**values)

    baseline = build()
    variants = (
        build(prompt="Make another lamp"),
        build(label="Other label"),
        build(keep_workspace=True),
        build(model="gpt-other"),
        build(reasoning="ultra"),
        build(environ={"PATH": "", "USDCRAFT_CODEX_BIN": "custom-codex"}),
    )
    assert all(item.fingerprint != baseline.fingerprint for item in variants)

    alternate_scaffold = tmp_path / "alternate_scaffold.py"
    alternate_scaffold.write_text("object_model = object()\n", encoding="utf-8")
    assert build(scaffold_path=alternate_scaffold).fingerprint != baseline.fingerprint
    (root / "scaffold.py").write_text("object_model = changed()\n", encoding="utf-8")
    assert build().fingerprint != baseline.fingerprint


def test_preflight_fingerprint_binds_runtime_and_selection_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    inherited = CodexRuntimeSelection(
        model_id="gpt-test",
        reasoning_effort="high",
        runtime_source="codex_root_thread_state",
        selection_source="inherited_main_codex_runtime",
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            "usdcraft.generation.command.resolve_codex_runtime",
            lambda **_kwargs: inherited,
        )
        inherited_preflight = preflight_generation(
            repo_root=root,
            prompt="Make a lamp",
        )
    explicit_preflight = preflight_generation(
        repo_root=root,
        prompt="Make a lamp",
        model="gpt-test",
        reasoning="high",
    )

    assert inherited_preflight.fingerprint != explicit_preflight.fingerprint


def test_mismatched_approval_never_creates_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    preflight = preflight_generation(
        repo_root=root,
        prompt="Make a lamp",
        model="gpt-test",
        reasoning="high",
    )
    monkeypatch.setattr(
        "usdcraft.generation.command.create_generation_workspace",
        lambda *_args, **_kwargs: pytest.fail("approval mismatch must remain zero-write"),
    )

    with pytest.raises(GenerationApprovalError, match="does not match"):
        execute_generation(preflight=preflight, approved_fingerprint="0" * 64)

    assert not (root / "data").exists()


def test_changed_staged_image_is_cleaned_before_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    image = tmp_path / "reference.png"
    _write_image(image)
    preflight = preflight_generation(
        repo_root=root,
        prompt="Make a lamp",
        image=image,
        model="gpt-test",
        reasoning="high",
    )
    from usdcraft.generation.workspace import create_generation_workspace as real_create

    captured: dict[str, Path] = {}

    def tampered_workspace(request, **kwargs):  # noqa: ANN001
        workspace = real_create(request, base_dir=tmp_path / "runs", **kwargs)
        assert workspace.input_image_path is not None
        workspace.input_image_path.write_bytes(b"changed after approval")
        captured["root"] = workspace.root
        return workspace

    monkeypatch.setattr(
        "usdcraft.generation.command.create_generation_workspace",
        tampered_workspace,
    )
    monkeypatch.setattr(
        "usdcraft.generation.command.run_codex",
        lambda *_args, **_kwargs: pytest.fail("changed staged input must not reach Codex"),
    )

    with pytest.raises(GenerationInputChangedError, match="image changed") as exc_info:
        execute_generation(
            preflight=preflight,
            approved_fingerprint=preflight.fingerprint,
        )

    assert not captured["root"].exists()
    context = generation_failure_context(exc_info.value)
    assert context is not None
    assert context.run_id == context.workspace_dir.name
    assert context.workspace_dir == captured["root"]
    assert context.workspace_dir.is_absolute()


def _fake_success_stack(monkeypatch: pytest.MonkeyPatch, root: Path, captured: dict) -> None:
    def fake_run_codex(request, workspace, *, prompt, executable):  # noqa: ANN001
        captured["prompt"] = prompt
        workspace.codex_dir.mkdir(exist_ok=True)
        workspace.codex_stdout_path.write_text("{}\n", encoding="utf-8")
        workspace.codex_stderr_path.write_text("", encoding="utf-8")
        workspace.codex_last_message_path.write_text(
            json.dumps({"status": "complete", "summary": "done"}), encoding="utf-8"
        )
        workspace.model_path.write_text("object_model = object()\n", encoding="utf-8")
        return _codex_result(CodexCompletionStatus.COMPLETE)

    def fake_compile(model_path: Path, output_dir: Path, **kwargs: object):  # noqa: ANN003
        output_dir.mkdir(parents=True, exist_ok=True)
        report = output_dir / "compile_report.json"
        report.write_text("{}", encoding="utf-8")
        return SimpleNamespace(warnings=(), output_dir=output_dir, report_path=report)

    def fake_persist(**kwargs):  # noqa: ANN003
        captured["persist"] = kwargs
        record_dir = root / "data" / "records" / "rec_observation"
        model_path = record_dir / "revisions" / "rev_000001" / "model.py"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("object_model = object()\n", encoding="utf-8")
        return record_dir

    monkeypatch.setattr("usdcraft.generation.command.run_codex", fake_run_codex)
    monkeypatch.setattr("usdcraft.generation.command.compile_model", fake_compile)
    monkeypatch.setattr("usdcraft.generation.command.persist_generation", fake_persist)


def test_image_run_embeds_external_observation_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usdcraft.pipeline.observation import ImageObservationArtifacts

    root = _repo(tmp_path)
    image = tmp_path / "inspiration.png"
    _write_image(image)
    captured: dict[str, object] = {}
    _fake_success_stack(monkeypatch, root, captured)

    inventory_payload = {
        "status": "complete",
        "parts": [
            {
                "name": "seat_disc",
                "visible_count": 1,
                "mobility": "fixed",
                "level": "primary_body",
                "visibility": "visible",
                "geometry": "flat round disc, as wide as the stool",
                "arrangement": "round top",
                "affordances": ["sit on"],
                "attaches_to": None,
                "duplicate_of": None,
                "mirror_of": None,
                "carried_as": "none",
                "notes": None,
            }
        ],
        "removals": [],
        "summary": "a stool",
    }

    def fake_observation(**kwargs):  # noqa: ANN003
        captured["observation_image"] = kwargs["image_path"]
        observation_root = Path(kwargs["output_dir"])
        observation_root.mkdir(parents=True)
        inventory_path = observation_root / "observation_inventory.json"
        inventory_path.write_text(json.dumps(inventory_payload), encoding="utf-8")
        return ImageObservationArtifacts(
            root=observation_root,
            inventory_path=inventory_path,
            manifest_path=inventory_path,
            inventory=inventory_payload,
            attempts=1,
        )

    monkeypatch.setattr(
        "usdcraft.generation.command.run_generation_image_observation",
        fake_observation,
    )
    preflight = preflight_generation(
        repo_root=root,
        prompt="Make a stool",
        image=image,
        model="gpt-test",
        reasoning="ultra",
    )
    result = execute_generation(preflight=preflight, approved_fingerprint=preflight.fingerprint)

    assert result.status == "generation_ready"
    prompt = captured["prompt"]
    assert isinstance(prompt, str)
    assert "EXTERNAL IMAGE-OBSERVATION INVENTORY (advisory first reading)" in prompt
    assert "- seat_disc: count=1, fixed, primary_body, visible" in prompt
    assert "advisory first reading" in prompt
    assert "do not re-verify" not in prompt
    persist_kwargs = captured["persist"]
    assert isinstance(persist_kwargs, dict)
    assert persist_kwargs["runtime_report"]["image_observation"] == "external_inventory"
    assert persist_kwargs["runtime_report"]["candidate_render"]["enabled"] is True


def test_image_run_degrades_to_author_only_reading_on_observation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usdcraft.generation.observation import ImageObservationError

    root = _repo(tmp_path)
    image = tmp_path / "inspiration.png"
    _write_image(image)
    captured: dict[str, object] = {}
    _fake_success_stack(monkeypatch, root, captured)
    monkeypatch.setattr(
        "usdcraft.generation.command.run_generation_image_observation",
        lambda **_kwargs: (_ for _ in ()).throw(ImageObservationError("codex missing")),
    )
    preflight = preflight_generation(
        repo_root=root,
        prompt="Make a stool",
        image=image,
        model="gpt-test",
        reasoning="ultra",
    )
    result = execute_generation(preflight=preflight, approved_fingerprint=preflight.fingerprint)

    assert result.status == "generation_ready"
    assert any("Image observation degraded" in warning for warning in result.warnings)
    prompt = captured["prompt"]
    assert isinstance(prompt, str)
    assert "EXTERNAL IMAGE-OBSERVATION INVENTORY" not in prompt
    assert "consistent object" in prompt
    assert "Identify characteristic parts" in prompt
    persist_kwargs = captured["persist"]
    assert isinstance(persist_kwargs, dict)
    assert persist_kwargs["runtime_report"]["image_observation"] == "author_only"


def test_text_only_run_skips_image_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    captured: dict[str, object] = {}
    _fake_success_stack(monkeypatch, root, captured)

    def forbidden_observation(**_kwargs):  # noqa: ANN003
        raise AssertionError("text-only Generation must not start an observation Agent")

    monkeypatch.setattr(
        "usdcraft.generation.command.run_generation_image_observation",
        forbidden_observation,
    )
    preflight = preflight_generation(
        repo_root=root,
        prompt="Make a stool",
        model="gpt-test",
        reasoning="ultra",
    )
    result = execute_generation(preflight=preflight, approved_fingerprint=preflight.fingerprint)

    assert result.status == "generation_ready"
    persist_kwargs = captured["persist"]
    assert isinstance(persist_kwargs, dict)
    assert persist_kwargs["runtime_report"]["image_observation"] == "no_image"


def test_direct_execution_needs_no_fingerprint_and_records_its_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    captured: dict[str, object] = {}
    _fake_success_stack(monkeypatch, root, captured)
    preflight = preflight_generation(
        repo_root=root, prompt="Make a stool", model="gpt-test", reasoning="ultra"
    )
    messages: list[str] = []
    result = execute_generation(
        preflight=preflight,
        direct=True,
        run_id="gen_direct_row",
        batch_context={"row_id": "a"},
        progress=messages.append,
    )

    assert result.status == "generation_ready"
    assert result.run_id == "gen_direct_row"
    report = captured["persist"]["runtime_report"]
    assert report["authorization"] == "direct"
    assert report["batch"] == {"row_id": "a"}
    assert any("stage" in message for message in messages)


def test_confirmed_execution_records_fingerprint_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    captured: dict[str, object] = {}
    _fake_success_stack(monkeypatch, root, captured)
    preflight = preflight_generation(
        repo_root=root, prompt="Make a stool", model="gpt-test", reasoning="ultra"
    )
    execute_generation(preflight=preflight, approved_fingerprint=preflight.fingerprint)
    report = captured["persist"]["runtime_report"]
    assert report["authorization"] == "fingerprint" and report["batch"] is None


def test_direct_and_fingerprint_together_are_rejected_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    preflight = preflight_generation(
        repo_root=root, prompt="Make a lamp", model="gpt-test", reasoning="high"
    )
    monkeypatch.setattr(
        "usdcraft.generation.command.create_generation_workspace",
        lambda *_args, **_kwargs: pytest.fail("a rejected authorization must remain zero-write"),
    )
    with pytest.raises(GenerationApprovalError, match="not both"):
        execute_generation(
            preflight=preflight, approved_fingerprint=preflight.fingerprint, direct=True
        )
    assert not (root / "data").exists()
