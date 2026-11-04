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

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from usdcraft import cli as usdcraft_cli
from usdcraft.cli import main
from usdcraft.pipeline.batch import (
    ProviderBatchInterrupted,
    ProviderBatchResult,
    ProviderBatchRowResult,
)
from usdcraft.reconstruction.codex_batch import execute_codex_reconstruction_batch
from usdcraft.reconstruction.command import preflight_reconstruction
from usdcraft.reconstruction.provider_deadline import _worker


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


@pytest.mark.slow
@pytest.mark.parametrize("configured", [False, True])
def test_missing_codex_cli_fails_before_rows_in_real_subprocess(tmp_path, configured):
    csv = tmp_path / "jobs.csv"
    csv.write_text("row_id,source_path,image_path\na,missing.glb,missing.jpg\n")
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
            "reconstruction-codex-batch",
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


def test_parallel_codex_rows_preserve_policy_order_and_failure_isolation(tmp_path, monkeypatch):
    csv = tmp_path / "jobs.csv"
    csv.write_text(
        "row_id,source_path,image_path,image_observation,guidance\n"
        "a,source.glb,image.jpg,false,Keep the lid closed.\n"
        "b,source.glb,image.jpg,true,\n"
    )
    barrier = Barrier(2)
    captured = {}

    def execute(arguments, **kwargs):
        captured[kwargs["row_id"]] = (arguments, kwargs)
        barrier.wait(timeout=5)
        if kwargs["row_id"] == "b":
            raise ValueError("row failure")
        return ProviderBatchRowResult(row_id="a", status="reconstruction_ready", record_id="rec_a")

    monkeypatch.setattr("usdcraft.reconstruction.codex_batch.execute_with_deadline", execute)
    monkeypatch.setattr(
        "usdcraft.pipeline.provider.resolve_nvidia_runtime_identity",
        lambda *a, **kw: pytest.fail("Codex must not resolve NVIDIA runtime"),
    )
    result = execute_codex_reconstruction_batch(
        csv_path=csv,
        repo_root=tmp_path,
        model="test-model-a",
        reasoning="low",
        concurrency=2,
    )
    assert result.to_dict()["provider"] == "codex-cli"
    assert result.status == "partial_failure"
    assert [row.row_id for row in result.rows] == ["a", "b"]
    assert not result.rows[0].warnings and result.rows[1].status == "failed"
    arguments, supervisor = captured["a"]
    assert arguments["source"] == tmp_path / "source.glb"
    assert arguments["semantic_image"] == tmp_path / "image.jpg"
    assert arguments["image_observation"] is False
    # Each row's guidance reaches the single-run primitive, which records it in the record.
    assert arguments["guidance"] == "Keep the lid closed."
    assert captured["b"][0]["guidance"] is None
    # The Codex arm has no time or turn limit: rows carry neither.
    assert "max_turns" not in arguments and "timeout_seconds" not in arguments
    # The supervisor worker passes these arguments (minus batch_context) to the real preflight.
    inspect.signature(preflight_reconstruction).bind(
        **{name: value for name, value in arguments.items() if name != "batch_context"}
    )
    assert arguments["model"] == "test-model-a" and arguments["reasoning"] == "low"
    assert arguments["batch_context"]["row_id"] == "a"
    # Rows stay supervised (Ctrl-C stops them) but without a deadline.
    assert supervisor["arm"] == "codex" and supervisor["timeout_seconds"] is None


def test_codex_worker_uses_direct_primitive_and_bound_run_id(tmp_path, monkeypatch):
    control = tmp_path / "control"
    control.mkdir()
    (control / "transport.json").write_text('{"arm":"codex"}')
    (control / "request.json").write_text(
        json.dumps(
            dict(
                repo_root=str(tmp_path),
                source="/source.glb",
                semantic_image="/image.jpg",
                run_id="recon_test",
                image_observation=False,
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
        return SimpleNamespace(
            status="reconstruction_ready", to_dict=lambda: {"record_id": "rec_a"}
        )

    monkeypatch.setattr("usdcraft.reconstruction.command.preflight_reconstruction", prepare)
    monkeypatch.setattr("usdcraft.reconstruction.command.execute_reconstruction", execute)
    _worker(control)
    assert captured["preflight"]["image_observation"] is False
    assert captured["preflight"]["source"] == Path("/source.glb")
    assert captured["preflight"]["keep_workspace"] is True
    assert captured["execute"]["direct"] is True
    assert captured["execute"]["preflight"] is preflight
    assert captured["execute"]["run_id"] == "recon_test"
    assert captured["execute"]["batch_context"] == {"row_id": "a"}
    assert json.loads((control / "result.json").read_text())["status"] == "reconstruction_ready"
    assert (control / "progress.jsonl").is_file()


@pytest.mark.parametrize("column,value", [("max_turns", "40"), ("api_style", "chat-completions")])
def test_codex_batch_rejects_provider_only_cells(tmp_path, monkeypatch, capsys, column, value):
    csv = tmp_path / "jobs.csv"
    csv.write_text(f"row_id,source_path,image_path,{column}\na,missing.glb,missing.jpg,{value}\n")
    monkeypatch.setattr(
        "usdcraft.reconstruction.codex_batch.execute_with_deadline",
        lambda *a, **kw: pytest.fail("a Provider-only cell must fail before any row starts"),
    )
    with pytest.raises(ValueError, match=f"{column} is a Provider-only CSV option"):
        execute_codex_reconstruction_batch(
            csv_path=csv, repo_root=tmp_path, model="test-model-a", reasoning="low"
        )
    assert main(["reconstruction-codex-batch", str(csv), "--validate-only", "--json"]) == 1
    assert f"{column} is a Provider-only CSV option" in capsys.readouterr().err
    # An empty Provider-only cell is simply unused.
    csv.write_text(f"row_id,source_path,image_path,{column}\na,missing.glb,missing.jpg,\n")
    assert main(["reconstruction-codex-batch", str(csv), "--validate-only", "--json"]) == 0


def test_codex_batch_has_no_time_limit_options(tmp_path):
    for flag in ("--timeout", "--run-timeout"):
        with pytest.raises(SystemExit) as exc:
            main(["reconstruction-codex-batch", str(tmp_path / "jobs.csv"), flag, "60"])
        assert exc.value.code == 2


def test_codex_batch_validate_only_needs_no_runtime_or_inputs(tmp_path, capsys):
    csv = tmp_path / "jobs.csv"
    csv.write_text(
        "row_id,source_path,image_path,image_observation\na,missing.glb,missing.jpg,false\n"
    )
    assert main(["reconstruction-codex-batch", str(csv), "--validate-only", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 1
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("with_defaults", [False, True])
def test_codex_mixed_models_resolve_before_workers_without_inherited_runtime(
    tmp_path, monkeypatch, with_defaults
):
    csv = tmp_path / "mixed.csv"
    csv.write_text(
        "row_id,source_path,image_path,model,reasoning\n"
        "a,m.glb,i.jpg,test-model-a,low\n"
        "b,m.glb,i.jpg,test-model-b,high\n" + ("c,m.glb,i.jpg,,\n" if with_defaults else "")
    )
    monkeypatch.setattr(
        "usdcraft.reconstruction.command.current_main_codex_runtime",
        lambda **kw: pytest.fail("All effective runtime pairs are explicit"),
    )
    monkeypatch.setattr(
        "usdcraft.pipeline.provider.resolve_nvidia_runtime_identity",
        lambda *a, **kw: pytest.fail("Codex must not use Provider runtime"),
    )
    calls = {}
    events = []

    def supervised(arguments, **kwargs):
        calls[kwargs["row_id"]] = arguments
        assert kwargs["arm"] == "codex" and kwargs["timeout_seconds"] is None
        return ProviderBatchRowResult(row_id=kwargs["row_id"], status="reconstruction_ready")

    monkeypatch.setattr("usdcraft.reconstruction.codex_batch.execute_with_deadline", supervised)
    result = execute_codex_reconstruction_batch(
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


def test_codex_batch_cli_uses_codex_identity(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "usdcraft.reconstruction.codex_batch.execute_codex_reconstruction_batch",
        lambda **kw: ProviderBatchResult(
            csv_path=tmp_path / "jobs.csv",
            rows=(ProviderBatchRowResult(row_id="a", status="reconstruction_ready"),),
            provider="codex-cli",
        ),
    )
    assert (
        main(
            [
                "reconstruction-codex-batch",
                str(tmp_path / "jobs.csv"),
                "--repo-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["provider"] == "codex-cli"
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
        main(
            [
                "reconstruction-codex",
                "source.glb",
                "--image",
                "image.jpg",
                "--direct",
                "--confirm-reconstruction",
                "token",
            ]
        )
    assert exc.value.code == 2
