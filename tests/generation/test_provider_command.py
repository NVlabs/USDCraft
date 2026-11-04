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
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from usdcraft.compile.command import CompileArtifacts
from usdcraft.generation.command import generation_failure_context
from usdcraft.generation.persist import (
    NVIDIA_PROVIDER_AUTHORING_IDENTITY,
    OPENAI_PROVIDER_AUTHORING_IDENTITY,
)
from usdcraft.generation.provider_command import (
    ProviderGenerationBlockedError,
    ProviderGenerationCancelledError,
    execute_provider_generation,
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    return root


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (80, 120, 160)).save(path)


def _artifacts(output_dir: Path) -> CompileArtifacts:
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
    paths["usdc"].parent.mkdir(exist_ok=True)
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


def _fake_harness_class(run):  # type: ignore[no-untyped-def]
    """A stand-in for ProviderHarness whose run() delegates to the test's callable."""

    class _FakeHarness:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

        def run(self, *, prompt, image_path=None, cache_prefix=None):  # type: ignore[no-untyped-def]
            from usdcraft.generation.prompt import generation_cache_prefix

            assert cache_prefix == generation_cache_prefix(prompt)
            assert prompt.startswith(cache_prefix)
            return run(prompt=prompt, image_path=image_path, **self.kwargs)

        def resume(self, **_kwargs):  # type: ignore[no-untyped-def]
            pytest.fail("visual review must not resume the Author in this test")

    return _FakeHarness


def _patch_provider_runtime(
    monkeypatch: pytest.MonkeyPatch,
    config_calls: list[dict[str, object]] | None = None,
    *,
    endpoint: str = "nvidia",
) -> None:
    def from_repo(*_args, **kwargs):  # type: ignore[no-untyped-def]
        if config_calls is not None:
            config_calls.append(kwargs)
        return SimpleNamespace(
            api_key="test-key",
            model="nvidia-test-model",
            reasoning_effort="high",
            timeout_seconds=45.0,
            api_style=kwargs.get("api_style", "responses"),
            endpoint=endpoint,
            runtime_source=f"{endpoint}-{kwargs.get('api_style', 'responses')}-api",
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.NvidiaProviderConfig",
        SimpleNamespace(from_repo=from_repo),
    )
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.NvidiaResponsesClient",
        lambda _config, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.recover_pending_promotions",
        lambda _repo: SimpleNamespace(errors=()),
    )


def test_provider_cancellation_before_workspace_creation_has_no_side_effects(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    cancellation_event = threading.Event()
    cancellation_event.set()

    with pytest.raises(ProviderGenerationCancelledError):
        execute_provider_generation(
            repo_root=root,
            prompt="a lamp",
            cancellation_event=cancellation_event,
        )

    assert not (root / "data").exists()


def test_provider_rejects_an_undecodable_image_before_workspace_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    image = tmp_path / "reference.png"
    image.write_bytes(b"not a raster image")
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.recover_pending_promotions",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not begin recovery"),
    )
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.create_generation_workspace",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not create a workspace"),
    )

    with pytest.raises(ValueError, match="Generation inspiration image is invalid"):
        execute_provider_generation(repo_root=root, prompt="a lamp", image=image)

    assert not (root / "data").exists()


def test_provider_single_directly_resolves_image_prompt_and_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    image = tmp_path / "reference.jpg"
    _write_image(image)
    captured: dict[str, object] = {}
    progress: list[str] = []
    progress_callback = progress.append
    _patch_provider_runtime(monkeypatch)

    def fake_observation(**kwargs):  # type: ignore[no-untyped-def]
        captured["observation_image"] = kwargs["image_path"]
        from usdcraft.generation.observation import ImageObservationError

        raise ImageObservationError("observation transport unavailable in this test")

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.run_generation_image_observation_via_provider",
        fake_observation,
    )

    def fake_harness(**kwargs):
        workspace = kwargs["workspace"]
        captured["harness_progress"] = kwargs["progress"]
        captured["workspace"] = workspace.root
        captured["task"] = kwargs["prompt"]
        captured["image"] = kwargs["image_path"]
        captured["request"] = json.loads(workspace.request_path.read_text(encoding="utf-8"))
        workspace.model_path.write_text(
            "object_model = object()\n# provider edit\n", encoding="utf-8"
        )
        return SimpleNamespace(
            status="complete",
            summary="done",
            runtime_report=lambda: {
                "returncode": 0,
                "elapsed_seconds": 1.0,
                "final_status": "complete",
                "final_summary": "done",
                "turn_count": 2,
                "tool_call_count": 4,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        )

    def fake_compile(model_path: Path, output_dir: Path, **kwargs: object) -> CompileArtifacts:
        captured["compile_model"] = model_path
        captured["compile_kwargs"] = kwargs
        return _artifacts(output_dir)

    def fake_persist(**kwargs) -> Path:
        captured["persist"] = kwargs
        record_dir = root / "data" / "records" / "rec_provider"
        model_path = record_dir / "revisions" / "rev_000001" / "model.py"
        model_path.parent.mkdir(parents=True)
        model_path.write_text("object_model = object()\n", encoding="utf-8")
        return record_dir

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.ProviderHarness",
        _fake_harness_class(fake_harness),
    )
    monkeypatch.setattr("usdcraft.generation.provider_command.compile_model", fake_compile)
    monkeypatch.setattr("usdcraft.generation.provider_command.persist_generation", fake_persist)

    result = execute_provider_generation(
        repo_root=root,
        image=image,
        timeout_seconds=45,
        progress=progress_callback,
    )

    assert result.status == "generation_ready"
    assert result.record_id == "rec_provider"
    assert result.run_id
    assert result.workspace_dir is None
    assert not Path(captured["workspace"]).exists()
    assert captured["image"].name == "reference.jpg"
    task = str(captured["task"])
    assert "check_model" in task
    assert "compile_model" in task
    request_payload = captured["request"]
    assert isinstance(request_payload, dict)
    assert request_payload["provider"]["name"] == "nvidia"
    assert "codex" not in request_payload
    assert "fingerprint" not in request_payload
    assert captured["compile_kwargs"] == {
        "clean": True,
        "render": False,
        "require_generation_root_floating": True,
    }
    persisted = captured["persist"]
    assert isinstance(persisted, dict)
    assert persisted["authoring_identity"] is NVIDIA_PROVIDER_AUTHORING_IDENTITY
    assert persisted["runtime"].to_dict()["model_id"] == "nvidia-test-model"
    assert persisted["runtime_report"]["usage"]["total_tokens"] == 15
    assert persisted["runtime_report"]["candidate_render"] == {"enabled": True, "calls": 0}
    assert persisted["runtime_report"]["author_validation"]["operations"] == {
        stage: {"revision": None, "status": "not_run"} for stage in ("check", "compile", "render")
    }
    assert "render_candidate" in task
    assert captured["harness_progress"] is progress_callback
    assert "nvidia-test-model" in progress[0]
    # Host stages narrate themselves through the same progress stream as the harness.
    stage_lines = [message for message in progress if 'activity="stage ' in message]
    stages = [message.split("stage=", 1)[1].split(" |")[0] for message in stage_lines]
    assert stages == [
        "freeze",
        "freeze",
        "image_observation",
        "image_observation",
        "author",
        "author",
        "host_gate",
        "host_gate",
        "persist",
        "persist",
    ]
    assert "generation_ready" in progress[-1] and "rec_provider" in progress[-1]


def test_provider_blocked_retains_workspace_and_failure_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    _patch_provider_runtime(monkeypatch)
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.ProviderHarness",
        _fake_harness_class(
            lambda **_kwargs: SimpleNamespace(status="blocked", summary="unsupported topology")
        ),
    )
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.compile_model",
        lambda *_args, **_kwargs: pytest.fail("blocked Provider item must not enter Host compile"),
    )

    with pytest.raises(ProviderGenerationBlockedError) as exc_info:
        execute_provider_generation(repo_root=root, prompt="an unsupported closed-loop mechanism")

    context = generation_failure_context(exc_info.value)
    assert context is not None
    assert context.run_id
    assert context.workspace_dir.is_dir()


def test_provider_direct_accepts_multiline_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    _patch_provider_runtime(monkeypatch)
    captured: dict[str, object] = {}

    def fake_harness(**kwargs):
        captured["task"] = kwargs["prompt"]
        workspace = kwargs["workspace"]
        workspace.model_path.write_text("object_model = object()\n", encoding="utf-8")
        return SimpleNamespace(
            status="complete",
            summary="done",
            runtime_report=lambda: {"returncode": 0, "final_status": "complete"},
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.ProviderHarness", _fake_harness_class(fake_harness)
    )
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.compile_model",
        lambda _model, output, **_kwargs: _artifacts(output),
    )

    def fake_persist(**_kwargs) -> Path:
        record_dir = root / "data" / "records" / "rec_multiline"
        model_path = record_dir / "revisions" / "rev_000001" / "model.py"
        model_path.parent.mkdir(parents=True)
        model_path.write_text("object_model = object()\n", encoding="utf-8")
        return record_dir

    monkeypatch.setattr("usdcraft.generation.provider_command.persist_generation", fake_persist)

    execute_provider_generation(
        repo_root=root,
        prompt="Build a lamp\nwith two articulated arms.",
    )

    assert '"text": "Build a lamp\\nwith two articulated arms."' in str(captured["task"])


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "USDCRAFT_COMPILE_TIMEOUT_SECONDS",
            "0",
            "requires a positive finite USDCRAFT_COMPILE_TIMEOUT_SECONDS",
        ),
        (
            "USDCRAFT_COMPILE_TIMEOUT_SECONDS",
            "nan",
            "requires a positive finite USDCRAFT_COMPILE_TIMEOUT_SECONDS",
        ),
    ],
)
def test_provider_rejects_compile_modes_that_run_generated_code_with_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    root = _repo(tmp_path)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        execute_provider_generation(repo_root=root, prompt="a lamp")

    assert not (root / "data").exists()


@pytest.mark.parametrize("api_style", ["responses", "chat-completions"])
def test_provider_api_style_reaches_config_and_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_style: str,
) -> None:
    root = _repo(tmp_path)
    config_calls: list[dict[str, object]] = []
    _patch_provider_runtime(monkeypatch, config_calls)
    captured: dict[str, object] = {}

    def fake_harness(**kwargs):
        kwargs["workspace"].model_path.write_text("object_model = object()\n", encoding="utf-8")
        return SimpleNamespace(
            status="complete",
            summary="done",
            runtime_report=lambda: {"returncode": 0, "final_status": "complete"},
        )

    def fake_persist(**kwargs) -> Path:
        captured["persist"] = kwargs
        record_dir = root / "data" / "records" / "rec_api_style"
        model_path = record_dir / "revisions" / "rev_000001" / "model.py"
        model_path.parent.mkdir(parents=True)
        model_path.write_text("object_model = object()\n", encoding="utf-8")
        return record_dir

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.ProviderHarness", _fake_harness_class(fake_harness)
    )
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.compile_model",
        lambda _model, output, **_kwargs: _artifacts(output),
    )
    monkeypatch.setattr("usdcraft.generation.provider_command.persist_generation", fake_persist)

    result = execute_provider_generation(
        repo_root=root,
        prompt="Build a lamp.",
        run_id="gen_supervised",
        **({"api_style": api_style} if api_style != "responses" else {}),
    )

    # The same NvidiaProviderConfig drives the Author and the image observation.
    assert config_calls[0].get("api_style", "responses") == api_style
    assert result.run_id == "gen_supervised" and result.provider == "nvidia"
    persisted = captured["persist"]
    assert isinstance(persisted, dict)
    report = persisted["runtime_report"]
    runtime = persisted["runtime"].to_dict()
    if api_style == "chat-completions":
        assert report["api_style"] == "chat-completions"
        assert runtime["runtime_source"] == "nvidia-chat-completions-api"
    else:
        assert "api_style" not in report
        assert runtime["runtime_source"] == "nvidia-responses-api"


def test_provider_openai_endpoint_names_the_request_record_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repo(tmp_path)
    _patch_provider_runtime(monkeypatch, endpoint="openai")
    captured: dict[str, object] = {}

    def fake_harness(**kwargs):
        workspace = kwargs["workspace"]
        captured["request"] = json.loads(workspace.request_path.read_text(encoding="utf-8"))
        workspace.model_path.write_text("object_model = object()\n", encoding="utf-8")
        return SimpleNamespace(
            status="complete",
            summary="done",
            runtime_report=lambda: {"returncode": 0, "final_status": "complete"},
        )

    def fake_persist(**kwargs) -> Path:
        captured["persist"] = kwargs
        record_dir = root / "data" / "records" / "rec_openai"
        model_path = record_dir / "revisions" / "rev_000001" / "model.py"
        model_path.parent.mkdir(parents=True)
        model_path.write_text("object_model = object()\n", encoding="utf-8")
        return record_dir

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.ProviderHarness", _fake_harness_class(fake_harness)
    )
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.compile_model",
        lambda _model, output, **_kwargs: _artifacts(output),
    )
    monkeypatch.setattr("usdcraft.generation.provider_command.persist_generation", fake_persist)

    result = execute_provider_generation(repo_root=root, prompt="Build a lamp.")

    assert result.provider == "openai" and result.to_dict()["provider"] == "openai"
    request_payload = captured["request"]
    assert isinstance(request_payload, dict)
    assert request_payload["provider"]["name"] == "openai"
    persisted = captured["persist"]
    assert isinstance(persisted, dict)
    assert persisted["authoring_identity"] is OPENAI_PROVIDER_AUTHORING_IDENTITY
    assert persisted["authoring_identity"].to_dict() == {
        "agent": "provider",
        "agent_harness": "provider",
        "provider": "openai",
    }
    assert persisted["runtime"].to_dict()["runtime_source"] == "openai-responses-api"
