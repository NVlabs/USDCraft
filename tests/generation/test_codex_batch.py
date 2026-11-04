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

"""Generation Codex arm: `generation-codex --direct` and `generation-codex-batch` (no real Codex)."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
from PIL import Image

from usdcraft import cli as usdcraft_cli
from usdcraft.cli import main
from usdcraft.generation.codex_batch import execute_codex_generation_batch
from usdcraft.generation.command import preflight_generation
from usdcraft.generation.provider_batch import GENERATION_DEADLINE
from usdcraft.pipeline.batch import (
    ProviderBatchInterrupted,
    ProviderBatchResult,
    ProviderBatchRowResult,
)
from usdcraft.pipeline.deadline import run_worker


@pytest.fixture(autouse=True)
def temporary_checkout_with_render_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    # These CLI tests use temporary repository roots and fake every render; the checkout and
    # render-setup checks have their own tests in tests/test_setup_cli.py.
    monkeypatch.setattr(usdcraft_cli, "_checkout_root", usdcraft_cli._repo_root)
    monkeypatch.setattr(usdcraft_cli, "_require_render_setup", lambda: None)


@pytest.fixture(autouse=True)
def available_executable(monkeypatch):
    # These routing tests replace execution; do not depend on a developer's Codex install.
    monkeypatch.setenv("USDCRAFT_CODEX_BIN", sys.executable)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("configured", [False, True])
def test_missing_codex_cli_fails_before_rows_in_real_subprocess(tmp_path, configured):
    csv = tmp_path / "jobs.csv"
    csv.write_text("row_id,prompt,image_path\na,a lamp,\n")
    env = dict(os.environ, USDCRAFT_CODEX_BIN=str(tmp_path / "absent-codex"))
    if not configured:
        env.pop("USDCRAFT_CODEX_BIN")
        env["PATH"] = str(tmp_path)
    (tmp_path / "scaffold.py").write_text("object_model = object()\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "usdcraft"\n')
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            # The real CLI in a real subprocess; only the render-setup check is faked so the
            # test does not depend on a local `usdcraft setup`.
            "import sys; from usdcraft import cli; cli._require_render_setup = lambda: None; "
            "sys.exit(cli.main(sys.argv[1:]))",
            "generation-codex-batch",
            str(csv),
            "--repo-root",
            str(tmp_path),
            "--model",
            "test-model-a",
            "--reasoning",
            "low",
            "--concurrency",
            "4",
            "--json",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 1
    assert "CodexUnavailableError" in completed.stdout
    assert "No batch rows have started" in completed.stdout
    assert "row started" not in completed.stderr
    assert not (tmp_path / "data").exists()


def test_parallel_codex_rows_run_supervised_direct_rows_with_failure_isolation(
    tmp_path, monkeypatch
):
    csv = tmp_path / "jobs.csv"
    csv.write_text("row_id,prompt,image_path,label\na,a lamp,,Lamp\nb,a chair,,\n")
    barrier = Barrier(2)
    captured = {}

    def execute(product, arguments, **kwargs):
        captured[kwargs["row_id"]] = (product, arguments, kwargs)
        barrier.wait(timeout=5)
        if kwargs["row_id"] == "b":
            raise ValueError("row failure")
        return ProviderBatchRowResult(row_id="a", status="generation_ready", record_id="rec_a")

    monkeypatch.setattr("usdcraft.generation.codex_batch.execute_with_deadline", execute)
    monkeypatch.setattr(
        "usdcraft.pipeline.provider.resolve_nvidia_runtime_identity",
        lambda *a, **kw: pytest.fail("Codex must not resolve NVIDIA runtime"),
    )
    result = execute_codex_generation_batch(
        csv_path=csv,
        repo_root=tmp_path,
        model="test-model-a",
        reasoning="low",
        concurrency=2,
        keep_workspace=True,
    )
    assert result.to_dict()["provider"] == "codex-cli"
    assert result.status == "partial_failure"
    assert [row.row_id for row in result.rows] == ["a", "b"]
    assert result.rows[1].status == "failed"
    product, arguments, supervisor = captured["a"]
    assert product is GENERATION_DEADLINE
    assert arguments["prompt"] == "a lamp" and arguments["image"] is None
    assert arguments["label"] == "Lamp" and arguments["keep_workspace"] is True
    assert arguments["model"] == "test-model-a" and arguments["reasoning"] == "low"
    assert arguments["batch_context"]["row_id"] == "a"
    # The Codex arm has no time limit and no wire-protocol choice: rows carry neither.
    assert "timeout_seconds" not in arguments and "api_style" not in arguments
    # The supervisor worker passes these arguments (minus batch_context) to the real preflight.
    inspect.signature(preflight_generation).bind(
        **{name: value for name, value in arguments.items() if name != "batch_context"}
    )
    # Rows stay supervised (Ctrl-C stops them) but without a deadline.
    assert supervisor["arm"] == "codex" and supervisor["timeout_seconds"] is None


def test_codex_worker_uses_direct_primitive_and_bound_run_id(tmp_path, monkeypatch):
    control = tmp_path / "control"
    control.mkdir()
    (control / "transport.json").write_text(
        json.dumps({"arm": "codex", "worker": GENERATION_DEADLINE.worker})
    )
    (control / "request.json").write_text(
        json.dumps(
            dict(
                repo_root=str(tmp_path),
                prompt="a lamp",
                image="/image.jpg",
                model="test-model-a",
                reasoning="low",
                label=None,
                run_id="gen_test",
                batch_context={"row_id": "a"},
                keep_workspace=True,
            )
        )
    )
    preflight = object()
    captured = {}

    def prepare(**kwargs):
        captured["preflight"] = kwargs
        return preflight

    def execute(**kwargs):
        captured["execute"] = kwargs
        kwargs["progress"]("stage=author")
        return SimpleNamespace(status="generation_ready", to_dict=lambda: {"record_id": "rec_a"})

    monkeypatch.setattr("usdcraft.generation.command.preflight_generation", prepare)
    monkeypatch.setattr("usdcraft.generation.command.execute_generation", execute)
    run_worker(control)
    assert captured["preflight"]["image"] == Path("/image.jpg")
    assert captured["preflight"]["repo_root"] == tmp_path
    assert captured["preflight"]["keep_workspace"] is True
    assert "run_id" not in captured["preflight"] and "batch_context" not in captured["preflight"]
    assert captured["execute"]["direct"] is True
    assert captured["execute"]["preflight"] is preflight
    assert captured["execute"]["run_id"] == "gen_test"
    assert captured["execute"]["batch_context"] == {"row_id": "a"}
    assert json.loads((control / "result.json").read_text())["status"] == "generation_ready"
    assert "stage=author" in (control / "progress.jsonl").read_text()


def test_codex_batch_rejects_provider_only_api_style(tmp_path, monkeypatch, capsys):
    csv = tmp_path / "jobs.csv"
    csv.write_text("row_id,prompt,image_path,api_style\na,a lamp,,chat-completions\n")
    monkeypatch.setattr(
        "usdcraft.generation.codex_batch.execute_with_deadline",
        lambda *a, **kw: pytest.fail("a Provider-only cell must fail before any row starts"),
    )
    with pytest.raises(ValueError, match="api_style is a Provider-only CSV option"):
        execute_codex_generation_batch(
            csv_path=csv, repo_root=tmp_path, model="test-model-a", reasoning="low"
        )
    assert main(["generation-codex-batch", str(csv), "--validate-only", "--json"]) == 1
    assert "api_style is a Provider-only CSV option" in capsys.readouterr().err
    # An empty Provider-only cell is simply unused.
    csv.write_text("row_id,prompt,image_path,api_style\na,a lamp,,\n")
    assert main(["generation-codex-batch", str(csv), "--validate-only", "--json"]) == 0


def test_codex_batch_has_no_time_limit_or_provider_options(tmp_path):
    for flag, value in (("--timeout", "60"), ("--run-timeout", "60"), ("--api-style", "responses")):
        with pytest.raises(SystemExit) as exc:
            main(["generation-codex-batch", str(tmp_path / "jobs.csv"), flag, value])
        assert exc.value.code == 2


def test_codex_batch_validate_only_needs_no_runtime_or_inputs(tmp_path, capsys):
    csv = tmp_path / "jobs.csv"
    csv.write_text("row_id,prompt,image_path\na,,missing.png\n")
    assert main(["generation-codex-batch", str(csv), "--validate-only", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 1
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("with_defaults", [False, True])
def test_codex_mixed_models_resolve_before_workers_without_inherited_runtime(
    tmp_path, monkeypatch, with_defaults
):
    csv = tmp_path / "mixed.csv"
    csv.write_text(
        "row_id,prompt,image_path,model,reasoning\n"
        "a,a lamp,,test-model-a,low\n"
        "b,a lamp,,test-model-b,high\n" + ("c,a lamp,,,\n" if with_defaults else "")
    )
    monkeypatch.setattr(
        "usdcraft.pipeline.runtime.current_main_codex_runtime",
        lambda **kw: pytest.fail("All effective runtime pairs are explicit"),
    )
    monkeypatch.setattr(
        "usdcraft.pipeline.provider.resolve_nvidia_runtime_identity",
        lambda *a, **kw: pytest.fail("Codex must not use Provider runtime"),
    )
    calls = {}
    events = []

    def supervised(product, arguments, **kwargs):
        calls[kwargs["row_id"]] = arguments
        assert kwargs["arm"] == "codex" and kwargs["timeout_seconds"] is None
        return ProviderBatchRowResult(row_id=kwargs["row_id"], status="generation_ready")

    monkeypatch.setattr("usdcraft.generation.codex_batch.execute_with_deadline", supervised)
    result = execute_codex_generation_batch(
        csv_path=csv,
        repo_root=tmp_path,
        concurrency=2,
        model="test-model-a" if with_defaults else None,
        reasoning="high" if with_defaults else None,
        progress=events.append,
    )
    assert result.succeeded == (3 if with_defaults else 2)
    assert (calls["a"]["model"], calls["a"]["reasoning"]) == ("test-model-a", "low")
    assert (calls["b"]["model"], calls["b"]["reasoning"]) == ("test-model-b", "high")
    if with_defaults:
        assert (calls["c"]["model"], calls["c"]["reasoning"]) == ("test-model-a", "high")
    assert events[0].model == events[0].reasoning_effort == "mixed"


def test_codex_batch_requires_a_complete_default_runtime_pair(tmp_path):
    csv = tmp_path / "jobs.csv"
    csv.write_text("row_id,prompt,image_path\na,a lamp,\n")
    with pytest.raises(ValueError, match="together"):
        execute_codex_generation_batch(csv_path=csv, repo_root=tmp_path, model="test-model-a")


def test_codex_batch_cli_uses_codex_identity_and_shared_options(tmp_path, monkeypatch, capsys):
    captured = {}

    def execute(**kwargs):
        captured.update(kwargs)
        return ProviderBatchResult(
            csv_path=tmp_path / "jobs.csv",
            rows=(ProviderBatchRowResult(row_id="a", status="generation_ready"),),
            provider="codex-cli",
        )

    monkeypatch.setattr("usdcraft.generation.codex_batch.execute_codex_generation_batch", execute)
    assert (
        main(
            [
                "generation-codex-batch",
                str(tmp_path / "jobs.csv"),
                "--repo-root",
                str(tmp_path),
                "--model",
                "test-model-a",
                "--reasoning",
                "low",
                "--concurrency",
                "3",
                "--keep-workspace",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert (payload["provider"], payload["pipeline"]) == ("codex-cli", "generation")
    assert captured["concurrency"] == 3 and captured["keep_workspace"] is True
    assert (captured["model"], captured["reasoning"]) == ("test-model-a", "low")
    # No Provider-only options reach the Codex arm.
    assert not {"timeout_seconds", "run_timeout_seconds", "api_style"} & set(captured)
    interrupted = ProviderBatchInterrupted(
        partial_result=ProviderBatchResult(
            csv_path=tmp_path / "jobs.csv", rows=(), provider="codex-cli"
        ),
        total=2,
        cancelled_row_ids=("b",),
        in_flight_row_ids=("a",),
    )
    assert interrupted.to_dict()["provider"] == "codex-cli"


def test_direct_and_confirmation_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        main(["generation-codex", "a lamp", "--direct", "--confirm-generation", "token"])
    assert exc.value.code == 2


def test_direct_cli_executes_without_a_fingerprint_exchange(tmp_path, monkeypatch, capsys):
    from usdcraft.generation.command import GenerationResult

    root = _repo(tmp_path)
    image = tmp_path / "reference.png"
    Image.new("RGB", (8, 8)).save(image)
    captured = {}

    def execute(**kwargs):
        captured.update(kwargs)
        record = root / "data/records/rec_direct"
        return GenerationResult(
            status="generation_ready",
            record_id="rec_direct",
            record_dir=record,
            model_path=record / "revisions/rev_000001/model.py",
            workspace_dir=None,
            warnings=(),
            run_id="gen_direct",
        )

    monkeypatch.setattr("usdcraft.generation.command.execute_generation", execute)
    args = ["generation-codex", "a lamp", "--image", str(image), "--repo-root", str(root)]
    runtime = ["--model", "test-model-a", "--reasoning", "low"]
    assert main([*args, *runtime, "--direct", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (payload["status"], payload["provider"]) == ("generation_ready", "codex-cli")
    assert captured["direct"] is True and captured["approved_fingerprint"] is None
    assert captured["preflight"].request.image_path == image.resolve()
    # Without --direct the same inputs stay a zero-write preflight.
    captured.clear()
    assert main([*args, *runtime, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "awaiting_approval"
    assert not captured
