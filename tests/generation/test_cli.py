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

import argparse
import json
import sqlite3
import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from usdcraft import cli as usdcraft_cli
from usdcraft.compile.models import CompileSignalBundle
from usdcraft.pipeline.codex import CodexExecutionError


@pytest.fixture(autouse=True)
def temporary_checkout_with_render_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    # These CLI tests use temporary repository roots and fake every render; the checkout and
    # render-setup checks have their own tests in tests/test_setup_cli.py.
    monkeypatch.setattr(usdcraft_cli, "_checkout_root", usdcraft_cli._repo_root)
    monkeypatch.setattr(usdcraft_cli, "_require_render_setup", lambda: None)


def test_cli_reports_the_pyproject_version(capsys: pytest.CaptureFixture[str]) -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    version = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]

    with pytest.raises(SystemExit) as exc_info:
        usdcraft_cli.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"usdcraft {version}\n"


def test_cli_check_reports_authoring_only_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model.py"
    model.write_text("object_model = object()\n", encoding="utf-8")
    monkeypatch.setattr(
        "usdcraft.compile.check_model",
        lambda path: SimpleNamespace(
            model_path=Path(path).resolve(),
            warnings=(),
            signal_bundle=CompileSignalBundle(
                status="success",
                summary="authoring checks passed",
            ),
        ),
    )

    assert usdcraft_cli.main(["check", str(model)]) == 0

    output = capsys.readouterr().out
    assert "status=success" in output
    assert "scope=authoring_only" in output
    assert f"model={model.resolve()}" in output
    assert "authoring checks passed" in output


def test_cli_check_returns_failure_without_creating_canonical_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model.py"
    model.write_text("object_model = object()\n", encoding="utf-8")

    def fail_check(_path: Path) -> None:
        raise ValueError("authored assertion failed")

    monkeypatch.setattr("usdcraft.compile.check_model", fail_check)

    assert usdcraft_cli.main(["check", str(model)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Authoring checks failed: ValueError: authored assertion failed" in captured.err
    assert not (tmp_path / "data").exists()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    return root


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (80, 120, 160)).save(path)


def _base_args(root: Path, prompt: str = "Build a functional lamp") -> list[str]:
    return [
        "generation-codex",
        prompt,
        "--repo-root",
        str(root),
        "--model",
        "gpt-test",
        "--reasoning",
        "ultra",
    ]


def _codex_state(codex_home: Path) -> None:
    codex_home.mkdir(parents=True)
    connection = sqlite3.connect(codex_home / "state_5.sqlite")
    connection.executescript(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            model TEXT,
            reasoning_effort TEXT
        );
        CREATE TABLE thread_spawn_edges (
            parent_thread_id TEXT NOT NULL,
            child_thread_id TEXT NOT NULL PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO threads VALUES ('root', 'gpt-main', 'max');
        INSERT INTO threads VALUES ('child', 'gpt-child', 'low');
        INSERT INTO thread_spawn_edges VALUES ('root', 'child', 'open');
        """
    )
    connection.commit()
    connection.close()


def test_default_cli_prints_only_compact_zero_write_text_approval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    before = sorted(path.relative_to(root) for path in root.rglob("*"))

    assert usdcraft_cli.main(_base_args(root, "Build a lamp\nwith a movable arm")) == 0

    after = sorted(path.relative_to(root) for path in root.rglob("*"))
    assert after == before
    output = capsys.readouterr().out
    lines = output.splitlines()
    keys = [line.partition("=")[0] for line in lines]
    assert keys == [
        "prompt",
        "model",
        "reasoning_effort",
        "input_mode",
        "approval_fingerprint",
    ]
    assert lines[:4] == [
        'prompt="Build a lamp\\nwith a movable arm"',
        "model=gpt-test",
        "reasoning_effort=ultra",
        "input_mode=text",
    ]
    assert len(lines) == 5
    assert lines[4].startswith("approval_fingerprint=")
    assert len(lines[4].removeprefix("approval_fingerprint=")) == 64
    assert "object_name=" not in output
    assert keys.count("model") == 1
    assert keys.count("reasoning_effort") == 1
    assert keys.count("approval_fingerprint") == 1
    for hidden_field in (
        "repo_root=",
        "workspace_base=",
        "materialization_root=",
        "physics_backends=",
        "asset_structure_",
        "schema_version=",
        "scaffold=",
        "sha256=",
        "executable=",
        "timeout_seconds=",
        "keep_workspace=",
        "runtime_source=",
        "selection_source=",
    ):
        assert hidden_field not in output
    assert not (root / "data").exists()


def test_default_image_preflight_adds_only_absolute_image_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    image = tmp_path / "reference image.png"
    _write_image(image)

    assert (
        usdcraft_cli.main(
            [
                *_base_args(root, "Build a reference lamp"),
                "--label",
                "Reference lamp",
                "--image",
                str(image),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert lines[:5] == [
        'prompt="Build a reference lamp"',
        "model=gpt-test",
        "reasoning_effort=ultra",
        "input_mode=text_image",
        f"image_path={json.dumps(str(image.resolve()))}",
    ]
    assert len(lines) == 6
    assert lines[5].startswith("approval_fingerprint=")
    assert "object_name=" not in output
    assert all("image_sha256" not in line for line in lines)
    assert not (root / "data").exists()


def test_json_image_preflight_is_absolute_and_zero_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    image = tmp_path / "reference.png"
    _write_image(image)

    assert usdcraft_cli.main([*_base_args(root), "--image", str(image), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "awaiting_approval"
    assert payload["record_created"] is False
    assert payload["case"]["input"]["kind"] == "text_image"
    assert payload["case"]["input"]["image_role"] == "inspiration"
    assert payload["case"]["input"]["image_path"] == str(image.resolve())
    assert len(payload["case"]["input"]["image_sha256"]) == 64
    assert payload["case"]["outputs"]["physics_backends"] == ["physx", "newton"]
    assert payload["case"]["outputs"]["entries"] == {
        "physx": "isaac/model.usdc",
        "newton": "isaac/model_newton.usdc",
    }
    assert len(payload["fingerprint"]) == 64
    assert not (root / "data").exists()


def test_json_image_only_cli_uses_primary_subject_default_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    image = tmp_path / "reference.jpg"
    _write_image(image)

    assert (
        usdcraft_cli.main(
            [
                "generation-codex",
                "--image",
                str(image),
                "--repo-root",
                str(root),
                "--model",
                "gpt-test",
                "--reasoning",
                "high",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["case"]["prompt"].strip()
    assert payload["case"]["input"]["kind"] == "text_image"
    assert not (root / "data").exists()


def test_cli_rejects_missing_text_and_image(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)

    assert (
        usdcraft_cli.main(
            [
                "generation-codex",
                "--repo-root",
                str(root),
                "--model",
                "gpt-test",
                "--reasoning",
                "high",
                "--json",
            ]
        )
        == 1
    )

    payload = json.loads(capsys.readouterr().out)
    assert "text prompt or one inspiration image" in payload["error"]
    assert "run_id" not in payload
    assert "workspace_dir" not in payload
    assert not (root / "data").exists()


def test_default_cli_inherits_root_runtime_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    codex_home = tmp_path / "codex"
    _codex_state(codex_home)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CODEX_THREAD_ID", "child")
    before = sorted(path.relative_to(root) for path in root.rglob("*"))

    assert (
        usdcraft_cli.main(["generation-codex", "Build a lamp", "--repo-root", str(root), "--json"])
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    codex = payload["case"]["codex"]
    assert codex["model_id"] == "gpt-main"
    assert codex["reasoning_effort"] == "max"
    assert codex["runtime_source"] == "codex_root_thread_state"
    assert codex["selection_source"] == "inherited_main_codex_runtime"
    # The Codex arm has no time limit, so the plan binds no timeout at all.
    assert "timeout_seconds" not in codex
    assert codex["executable"]
    assert sorted(path.relative_to(root) for path in root.rglob("*")) == before
    assert not (root / "data").exists()


def test_cli_old_fingerprint_is_rejected_before_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    assert usdcraft_cli.main([*_base_args(root, "First prompt"), "--json"]) == 0
    fingerprint = json.loads(capsys.readouterr().out)["fingerprint"]
    monkeypatch.setattr(
        "usdcraft.generation.command.create_generation_workspace",
        lambda *_args, **_kwargs: pytest.fail("mismatched approval must remain zero-write"),
    )

    assert (
        usdcraft_cli.main(
            [
                *_base_args(root, "Changed prompt"),
                "--confirm-generation",
                fingerprint,
            ]
        )
        == 1
    )

    error = capsys.readouterr().err
    assert "does not match the current preflight" in error
    assert "run_id=" not in error
    assert "workspace_dir=" not in error
    assert not (root / "data").exists()


def test_default_inherit_mode_fails_closed_without_codex_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)

    assert (
        usdcraft_cli.main(["generation-codex", "Build a lamp", "--repo-root", str(root), "--json"])
        == 1
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert "pass both --model and --reasoning explicitly" in payload["error"]
    assert "run_id" not in payload
    assert "workspace_dir" not in payload
    assert not (root / "data").exists()


def test_json_generation_failure_reports_run_id_and_absolute_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    args = [*_base_args(root), "--json"]
    assert usdcraft_cli.main(args) == 0
    fingerprint = json.loads(capsys.readouterr().out)["fingerprint"]

    def fail_after_workspace(*_args: object, **_kwargs: object) -> None:
        raise CodexExecutionError("Codex exited with status 1: interrupted")

    monkeypatch.setattr("usdcraft.generation.command.run_codex", fail_after_workspace)
    assert usdcraft_cli.main([*args, "--confirm-generation", fingerprint]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert (payload["pipeline"], payload["provider"]) == ("generation", "codex-cli")
    assert payload["error"].startswith("CodexExecutionError:")
    assert payload["run_id"]
    workspace_dir = Path(payload["workspace_dir"])
    assert workspace_dir.is_absolute()
    assert workspace_dir.name == payload["run_id"]
    assert workspace_dir.is_dir()


def test_text_generation_failure_reports_run_id_and_absolute_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    args = _base_args(root)
    assert usdcraft_cli.main([*args, "--json"]) == 0
    fingerprint = json.loads(capsys.readouterr().out)["fingerprint"]

    def fail_after_workspace(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("final Host compile failed")

    monkeypatch.setattr("usdcraft.generation.command.run_codex", fail_after_workspace)
    assert usdcraft_cli.main([*args, "--confirm-generation", fingerprint]) == 1

    lines = capsys.readouterr().err.splitlines()
    # Live stage logs precede the failure report on stderr.
    assert lines[0] == 'Generation Codex | activity="stage started" | stage=freeze'
    failure_index = lines.index("Generation failed: RuntimeError: final Host compile failed")
    run_line = lines[failure_index + 1]
    workspace_line = lines[failure_index + 2]
    assert run_line.startswith("run_id=")
    assert workspace_line.startswith("workspace_dir=/")
    workspace_dir = Path(workspace_line.removeprefix("workspace_dir="))
    assert workspace_dir.name == run_line.removeprefix("run_id=")
    assert workspace_dir.is_dir()


def test_provider_cli_executes_directly_without_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.generation.command import GenerationResult

    root = _repo(tmp_path)
    record_dir = root / "data" / "records" / "rec_provider"
    captured: dict[str, object] = {}

    def fake_execute_provider_generation(**kwargs):
        captured.update(kwargs)
        progress = kwargs["progress"]
        progress(
            "Provider Generation | activity=starting | model='nvidia-test-model' | effort='high'"
        )
        progress('Provider Generation | turn=1/64 | activity="running final Host compile"')
        progress(
            "Provider Generation | activity=completed | status=generation_ready | "
            "record_id='rec_provider'"
        )
        return GenerationResult(
            status="generation_ready",
            record_id="rec_provider",
            record_dir=record_dir,
            model_path=record_dir / "revisions" / "rev_000001" / "model.py",
            workspace_dir=None,
            warnings=(),
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    assert (
        usdcraft_cli.main(
            [
                "generation",
                "Build a Provider lamp",
                "--repo-root",
                str(root),
                "--model",
                "nvidia-test-model",
                "--reasoning",
                "high",
                "--json",
            ]
        )
        == 0
    )

    captured_output = capsys.readouterr()
    payload = json.loads(captured_output.out)
    assert payload["status"] == "generation_ready"
    assert payload["provider"] == "openai"
    assert captured["prompt"] == "Build a Provider lamp"
    assert captured["model"] == "nvidia-test-model"
    assert "approved_fingerprint" not in payload
    assert captured_output.err.startswith("[")
    assert '] Provider Generation | activity="starting"' in captured_output.err
    assert 'model="nvidia-test-model" | effort="high"' in captured_output.err
    assert 'activity="running final Host compile"' in captured_output.err
    assert "activity=completed | status=generation_ready" in captured_output.err
    assert "\x1b[" not in captured_output.err
    assert "Provider Batch" not in captured_output.err
    assert "Slot " not in captured_output.err
    timestamp = captured_output.err.split("]", maxsplit=1)[0].removeprefix("[")
    assert datetime.fromisoformat(timestamp).utcoffset() is not None


def test_provider_cli_json_failure_before_progress_has_no_ghost_display(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)

    def fail_before_progress(**_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fail_before_progress,
    )

    assert (
        usdcraft_cli.main(
            [
                "generation",
                "Build a Provider lamp",
                "--repo-root",
                str(root),
                "--json",
            ]
        )
        == 1
    )

    captured_output = capsys.readouterr()
    payload = json.loads(captured_output.out)
    assert payload == {
        "status": "failed",
        "pipeline": "generation",
        "provider": "openai",
        "error": "RuntimeError: provider unavailable",
    }
    assert captured_output.err == ""


def test_provider_cli_text_output_reports_run_record_and_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.generation.command import GenerationResult

    root = _repo(tmp_path)
    record_dir = root / "data" / "records" / "rec_provider"
    workspace = root / "data" / "local" / "generation_runs" / "gen_kept"
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        lambda **_kwargs: GenerationResult(
            status="generation_ready",
            record_id="rec_provider",
            record_dir=record_dir,
            model_path=record_dir / "revisions" / "rev_000001" / "model.py",
            workspace_dir=workspace,
            warnings=("USDZ packaging degraded",),
            run_id="gen_kept",
        ),
    )

    assert (
        usdcraft_cli.main(
            [
                "generation",
                "lamp",
                "--keep-workspace",
                "--repo-root",
                str(root),
            ]
        )
        == 0
    )

    lines = capsys.readouterr().out.splitlines()
    for line in (
        "status=generation_ready",
        "pipeline=generation",
        "provider=openai",
        "run_id=gen_kept",
        "record_id=rec_provider",
        f"workspace_dir={workspace}",
        'warnings=["USDZ packaging degraded"]',
    ):
        assert line in lines


def test_provider_cli_reports_the_openai_endpoint_as_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.generation.command import GenerationResult

    root = _repo(tmp_path)
    # Only OPENAI_API_KEY in `.env` selects the OpenAI endpoint; it is the Provider-arm identity.
    (root / ".env").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    record_dir = root / "data" / "records" / "rec_provider"
    outcomes: list[object] = [
        GenerationResult(
            status="generation_ready",
            record_id="rec_provider",
            record_dir=record_dir,
            model_path=record_dir / "revisions" / "rev_000001" / "model.py",
            workspace_dir=None,
            warnings=(),
            run_id="gen_openai",
            provider="openai",
        ),
        RuntimeError("provider unavailable"),
    ]

    def fake_execute_provider_generation(**_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    assert usdcraft_cli.main(["generation", "lamp", "--repo-root", str(root)]) == 0
    assert "provider=openai" in capsys.readouterr().out.splitlines()
    assert usdcraft_cli.main(["generation", "lamp", "--repo-root", str(root), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["provider"] == "openai"


def test_batch_cli_is_nvidia_only_and_reports_each_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.generation.provider_batch import (
        ProviderBatchResult,
        ProviderBatchRowResult,
    )
    from usdcraft.pipeline.batch_progress import ProviderBatchProgressEvent

    root = _repo(tmp_path)
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("row_id,prompt,image_path\none,a lamp,\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_execute_provider_batch(**kwargs):
        captured.update(kwargs)
        progress = kwargs["progress"]
        timestamp = datetime.now().astimezone()
        progress(
            ProviderBatchProgressEvent(
                kind="batch_started",
                timestamp=timestamp,
                total=1,
                effective_concurrency=1,
                model="nvidia-test-model",
                reasoning_effort="high",
            )
        )
        progress(
            ProviderBatchProgressEvent(
                kind="row_started",
                timestamp=timestamp,
                slot=1,
                row_index=1,
                row_id="one",
            )
        )
        progress(
            ProviderBatchProgressEvent(
                kind="row_progress",
                timestamp=timestamp,
                slot=1,
                row_index=1,
                row_id="one",
                message='Provider Generation | activity="running final Host compile"',
            )
        )
        progress(
            ProviderBatchProgressEvent(
                kind="row_finished",
                timestamp=timestamp,
                slot=1,
                row_index=1,
                row_id="one",
                status="generation_ready",
                record_id="rec_one",
            )
        )
        progress(
            ProviderBatchProgressEvent(
                kind="batch_finished",
                timestamp=timestamp,
                total=1,
                succeeded=1,
                failed=0,
                status="complete",
            )
        )
        return ProviderBatchResult(
            csv_path=csv_path.resolve(),
            rows=(
                ProviderBatchRowResult(
                    row_id="one",
                    status="generation_ready",
                    record_id="rec_one",
                    record_dir=root / "data" / "records" / "rec_one",
                ),
            ),
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_batch.execute_provider_batch",
        fake_execute_provider_batch,
    )

    assert (
        usdcraft_cli.main(
            [
                "generation-batch",
                str(csv_path),
                "--repo-root",
                str(root),
                "--concurrency",
                "3",
                "--json",
            ]
        )
        == 0
    )

    captured_output = capsys.readouterr()
    payload = json.loads(captured_output.out)
    assert (payload["status"], payload["pipeline"], payload["provider"]) == (
        "complete",
        "generation",
        "openai",
    )
    assert payload["rows"][0]["record_id"] == "rec_one"
    assert captured["concurrency"] == 3
    # Provider-only limits pass through (defaults), and the default wire protocol is omitted.
    assert captured["timeout_seconds"] is None and captured["run_timeout_seconds"] == 1920
    assert "api_style" not in captured
    assert callable(captured["progress"])
    assert "Provider Generation Batch" in captured_output.err
    assert 'model="nvidia-test-model"' in captured_output.err
    assert 'activity="running final Host compile"' in captured_output.err
    assert "\x1b[" not in captured_output.err


def test_batch_text_output_reports_failure_run_and_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.generation.provider_batch import (
        ProviderBatchResult,
        ProviderBatchRowResult,
    )

    root = _repo(tmp_path)
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("row_id,prompt,image_path\nbad,a lamp,\n", encoding="utf-8")
    workspace = root / "data" / "local" / "generation_runs" / "gen_failed"
    monkeypatch.setattr(
        "usdcraft.generation.provider_batch.execute_provider_batch",
        lambda **_kwargs: ProviderBatchResult(
            csv_path=csv_path.resolve(),
            rows=(
                ProviderBatchRowResult(
                    row_id="bad",
                    status="failed",
                    run_id="gen_failed",
                    workspace_dir=workspace,
                    error="RuntimeError: failed",
                ),
            ),
        ),
    )

    assert usdcraft_cli.main(["generation-batch", str(csv_path), "--repo-root", str(root)]) == 1

    output = capsys.readouterr().out
    assert "row=bad status=failed" in output
    assert "run_id=gen_failed" in output
    assert f"workspace_dir={workspace}" in output


def test_batch_cli_json_interrupt_reports_partial_results_and_returns_130(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.generation.provider_batch import (
        ProviderBatchInterrupted,
        ProviderBatchResult,
        ProviderBatchRowResult,
    )

    root = _repo(tmp_path)
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "row_id,prompt,image_path\ndone,a lamp,\nrunning,a chair,\nqueued,a table,\n",
        encoding="utf-8",
    )
    completed = ProviderBatchRowResult(
        row_id="done",
        status="generation_ready",
        record_id="rec_done",
        run_id="gen_done",
    )
    interruption = ProviderBatchInterrupted(
        partial_result=ProviderBatchResult(
            csv_path=csv_path.resolve(),
            rows=(completed,),
        ),
        total=3,
        cancelled_row_ids=("queued",),
        in_flight_row_ids=("running",),
    )

    def interrupt_batch(**_kwargs):
        raise interruption

    monkeypatch.setattr(
        "usdcraft.generation.provider_batch.execute_provider_batch",
        interrupt_batch,
    )

    assert (
        usdcraft_cli.main(
            [
                "generation-batch",
                str(csv_path),
                "--repo-root",
                str(root),
                "--json",
            ]
        )
        == 130
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"pipeline": "generation", **interruption.to_dict()}
    assert captured.err == ""


def test_batch_cli_text_interrupt_reports_row_and_outstanding_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.generation.provider_batch import (
        ProviderBatchInterrupted,
        ProviderBatchResult,
        ProviderBatchRowResult,
    )

    root = _repo(tmp_path)
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("row_id,prompt,image_path\ndone,a lamp,\n", encoding="utf-8")
    interruption = ProviderBatchInterrupted(
        partial_result=ProviderBatchResult(
            csv_path=csv_path.resolve(),
            rows=(
                ProviderBatchRowResult(
                    row_id="done",
                    status="generation_ready",
                    record_id="rec_done",
                    run_id="gen_done",
                ),
            ),
        ),
        total=3,
        cancelled_row_ids=("queued",),
        in_flight_row_ids=("running",),
    )
    monkeypatch.setattr(
        "usdcraft.generation.provider_batch.execute_provider_batch",
        lambda **_kwargs: (_ for _ in ()).throw(interruption),
    )

    assert usdcraft_cli.main(["generation-batch", str(csv_path), "--repo-root", str(root)]) == 130

    captured = capsys.readouterr()
    assert "status=interrupted" in captured.out
    assert "completed=1" in captured.out
    assert "row=done status=generation_ready record_id=rec_done run_id=gen_done" in captured.out
    assert 'cancelled_row_ids=["queued"]' in captured.out
    assert 'in_flight_row_ids=["running"]' in captured.out
    assert captured.err == ""


def test_generation_batch_passes_api_style_and_run_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.generation.provider_batch import ProviderBatchResult

    root = _repo(tmp_path)
    captured: dict[str, object] = {}

    def fake_execute_provider_batch(**kwargs):
        captured.update(kwargs)
        return ProviderBatchResult(csv_path=tmp_path / "jobs.csv", rows=())

    monkeypatch.setattr(
        "usdcraft.generation.provider_batch.execute_provider_batch", fake_execute_provider_batch
    )
    args = ["generation-batch", str(tmp_path / "jobs.csv"), "--repo-root", str(root)]
    options = ["--api-style", "chat-completions", "--run-timeout", "1920", "--timeout", "600"]
    assert usdcraft_cli.main([*args, *options, "--json"]) == 0
    assert captured["api_style"] == "chat-completions"
    assert captured["run_timeout_seconds"] == 1920
    assert captured["timeout_seconds"] == 600
    # An explicit value overrides the default, and 0 removes the per-row limit.
    assert usdcraft_cli.main([*args, "--run-timeout", "600", "--json"]) == 0
    assert captured["run_timeout_seconds"] == 600
    assert usdcraft_cli.main([*args, "--run-timeout", "0", "--json"]) == 0
    assert captured["run_timeout_seconds"] is None


def test_generation_batch_validate_only_checks_the_csv_contract_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "row_id,prompt,image_path,model,reasoning\na,a lamp,missing.png,test-model-a,low\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "usdcraft.generation.provider_batch.execute_provider_batch",
        lambda **_kwargs: pytest.fail("validation must not start the Provider"),
    )
    monkeypatch.setattr(
        "usdcraft.pipeline.provider.resolve_nvidia_runtime_identity",
        lambda *a, **kw: pytest.fail("validation must not resolve a runtime"),
    )
    before = set(tmp_path.rglob("*"))
    assert usdcraft_cli.main(["generation-batch", str(csv_path), "--validate-only", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "validated",
        "total": 1,
        "scope": "csv_contract",
    }
    assert set(tmp_path.rglob("*")) == before

    csv_path.write_text("row_id,prompt,image_path,lable\na,a lamp,,Lamp\n", encoding="utf-8")
    assert usdcraft_cli.main(["generation-batch", str(csv_path), "--validate-only"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Provider Generation Batch failed: ValueError:")
    assert "unknown columns: lable" in err
    assert (
        usdcraft_cli.main(
            ["generation-batch", str(csv_path), "--validate-only", "--run-timeout", "-5"]
        )
        == 1
    )
    assert "finite and positive" in capsys.readouterr().err
    csv_path.write_text("row_id,prompt,image_path\na,a lamp,\n", encoding="utf-8")
    assert (
        usdcraft_cli.main(
            ["generation-batch", str(csv_path), "--validate-only", "--run-timeout", "0"]
        )
        == 0
    )


def test_generation_single_passes_api_style_only_when_not_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from usdcraft.generation.command import GenerationResult

    root = _repo(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_execute_provider_generation(**kwargs):
        calls.append(kwargs)
        record_dir = root / "data" / "records" / "rec_one"
        return GenerationResult(
            status="generation_ready",
            record_id="rec_one",
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=(),
            run_id="gen_one",
            provider="nvidia",
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )
    base = ["generation", "lamp", "--repo-root", str(root), "--json"]
    assert usdcraft_cli.main(base) == 0
    assert usdcraft_cli.main([*base, "--api-style", "chat-completions"]) == 0
    assert "api_style" not in calls[0]
    assert calls[1]["api_style"] == "chat-completions"


@pytest.mark.parametrize(
    "command",
    [
        ["generation-codex", "lamp"],
        ["reconstruction-codex", "source.glb", "--image", "image.png"],
        ["reconstruction-codex-batch", "jobs.csv"],
    ],
)
@pytest.mark.parametrize("flag", ["--timeout", "--run-timeout", "--max-turns", "--api-style"])
def test_codex_commands_have_no_time_turn_or_protocol_options(
    command: list[str], flag: str
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        usdcraft_cli.main([*command, flag, "60"])
    assert exc_info.value.code == 2


def test_codex_single_run_result_identifies_pipeline_and_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.generation.command import GenerationResult

    root = _repo(tmp_path)
    record_dir = root / "data" / "records" / "rec_one"
    result = GenerationResult(
        status="generation_ready",
        record_id="rec_one",
        record_dir=record_dir,
        model_path=record_dir / "model.py",
        workspace_dir=None,
        warnings=(),
        run_id="gen_one",
    )
    monkeypatch.setattr("usdcraft.generation.command.execute_generation", lambda **_kwargs: result)
    assert usdcraft_cli.main([*_base_args(root), "--confirm-generation", "token"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert {"status=generation_ready", "pipeline=generation", "provider=codex-cli"} <= set(lines)
    assert "record_id=rec_one" in lines

    assert usdcraft_cli.main([*_base_args(root), "--confirm-generation", "token", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (payload["pipeline"], payload["provider"]) == ("generation", "codex-cli")


def test_compile_prints_backend_entries_and_loads_the_repository_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import os

    output = tmp_path / "build"
    loaded: dict[str, object] = {}

    def fake_compile(model, output_dir, **_kwargs):  # type: ignore[no-untyped-def]
        loaded["timeout"] = os.environ.get("USDCRAFT_COMPILE_TIMEOUT_SECONDS")
        return SimpleNamespace(
            urdf_path=output_dir / "model.urdf",
            usdc_path=output_dir / "isaac" / "model.usdc",
            newton_usdc_path=output_dir / "isaac" / "model_newton.usdc",
            asset_usda_path=output_dir / "isaac" / "asset.usda",
            usdz_path=output_dir / "model.usdz",
            report_path=output_dir / "compile_report.json",
        )

    monkeypatch.setattr("usdcraft.compile.compile_model", fake_compile)
    # Register the variable with monkeypatch so the value loaded from .env is undone afterwards.
    monkeypatch.setenv("USDCRAFT_COMPILE_TIMEOUT_SECONDS", "unset")
    monkeypatch.delenv("USDCRAFT_COMPILE_TIMEOUT_SECONDS")
    (tmp_path / ".env").write_text("USDCRAFT_COMPILE_TIMEOUT_SECONDS=123\n", encoding="utf-8")
    model = tmp_path / "model.py"
    args = ["compile", str(model), "--output", str(output), "--repo-root", str(tmp_path)]
    assert usdcraft_cli.main(args) == 0
    keys = [line.partition("=")[0] for line in capsys.readouterr().out.splitlines()]
    assert keys == [
        "status",
        "urdf",
        "physx_usdc",
        "newton_usdc",
        "asset_usda",
        "usdz",
        "report",
    ]
    # compile loads the repository .env exactly like every other command.
    assert loaded["timeout"] == "123"


def test_check_loads_the_repository_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    loaded: dict[str, object] = {}

    def fake_check(path):  # type: ignore[no-untyped-def]
        loaded["timeout"] = os.environ.get("USDCRAFT_COMPILE_TIMEOUT_SECONDS")
        return SimpleNamespace(
            model_path=Path(path).resolve(),
            warnings=(),
            signal_bundle=CompileSignalBundle(status="success", summary="ok"),
        )

    monkeypatch.setattr("usdcraft.compile.check_model", fake_check)
    # Register the variable with monkeypatch so the value loaded from .env is undone afterwards.
    monkeypatch.setenv("USDCRAFT_COMPILE_TIMEOUT_SECONDS", "unset")
    monkeypatch.delenv("USDCRAFT_COMPILE_TIMEOUT_SECONDS")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("USDCRAFT_COMPILE_TIMEOUT_SECONDS=77\n", encoding="utf-8")
    assert usdcraft_cli.main(["check", str(tmp_path / "model.py")]) == 0
    assert loaded["timeout"] == "77"


RUN_COMMANDS = (
    "generation",
    "generation-codex",
    "generation-batch",
    "generation-codex-batch",
    "reconstruction",
    "reconstruction-codex",
    "reconstruction-batch",
    "reconstruction-codex-batch",
)


def test_every_run_command_has_one_keep_workspace_flag_and_wording() -> None:
    parser = usdcraft_cli._build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    helps = set()
    for name in RUN_COMMANDS:
        options = {
            option: action
            for action in subparsers.choices[name]._actions
            for option in action.option_strings
        }
        helps.add(options["--keep-workspace"].help)
    assert helps == {usdcraft_cli._KEEP_WORKSPACE_HELP}


def test_every_public_option_has_help() -> None:
    parser = usdcraft_cli._build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    missing = [
        (name, action.dest)
        for name, command in subparsers.choices.items()
        for action in command._actions
        if not action.help or action.help == argparse.SUPPRESS
    ]
    assert missing == []
