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

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from usdcraft import cli as usdcraft_cli
from usdcraft.pipeline.batch import ProviderBatchResult, ProviderBatchRowResult
from usdcraft.pipeline.batch_progress import ProviderBatchProgressEvent
from usdcraft.reconstruction.provider_batch import (
    execute_reconstruction_batch,
    load_reconstruction_batch,
)
from usdcraft.reconstruction.provider_command import ProviderReconstructionResult

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_COLUMNS = (
    "row_id",
    "source_path",
    "image_path",
    "prompt",
    "guidance",
    "label",
    "up_axis",
    "front_axis",
    "meters_per_unit",
)


@pytest.fixture(autouse=True)
def temporary_checkout_with_render_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    # These CLI tests use temporary repository roots and fake every render; the checkout and
    # render-setup checks have their own tests in tests/test_setup_cli.py.
    monkeypatch.setattr(usdcraft_cli, "_checkout_root", usdcraft_cli._repo_root)
    monkeypatch.setattr(usdcraft_cli, "_require_render_setup", lambda: None)


def _write_csv(path: Path, rows: list[dict[str, str]], *, columns=_COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    return root


def test_loader_resolves_paths_relative_to_the_csv_and_applies_defaults(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs" / "recon.csv"
    csv_path.parent.mkdir()
    _write_csv(
        csv_path,
        [
            {
                "row_id": "microwave",
                "source_path": "cases/microwave/input_mesh.glb",
                "image_path": "/abs/microwave.jpg",
            },
            {
                "row_id": "toaster",
                "source_path": "/abs/toaster.glb",
                "image_path": "cases/toaster/input_image.jpg",
                "prompt": "Reconstruct the toaster.",
                "guidance": "Two slots.",
                "label": "Toaster",
                "up_axis": "+Y",
                "front_axis": "-Y",
                "meters_per_unit": "0.01",
            },
        ],
    )

    first, second = load_reconstruction_batch(csv_path)

    assert first.source_path == (csv_path.parent / "cases/microwave/input_mesh.glb").resolve()
    assert first.image_path == Path("/abs/microwave.jpg").resolve()
    assert first.prompt is None and first.guidance is None and first.label is None
    assert first.up_axis is None and first.front_axis is None and first.meters_per_unit is None
    assert second.source_path == Path("/abs/toaster.glb").resolve()
    assert second.image_path == (csv_path.parent / "cases/toaster/input_image.jpg").resolve()
    assert second.prompt == "Reconstruct the toaster."
    assert second.guidance == "Two slots."
    assert second.label == "Toaster"
    assert second.up_axis == "+Y" and second.front_axis == "-Y"
    assert second.meters_per_unit == 0.01


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"row_id": "a", "source_path": "", "image_path": "x.jpg"}, "empty source_path"),
        ({"row_id": "a", "source_path": "m.glb", "image_path": ""}, "empty image_path"),
        (
            {"row_id": "a", "source_path": "m.glb", "image_path": "x.jpg", "meters_per_unit": "cm"},
            "meters_per_unit must be a number",
        ),
    ],
)
def test_loader_reports_invalid_cells_with_the_record_location(
    tmp_path: Path, row: dict[str, str], message: str
) -> None:
    csv_path = tmp_path / "recon.csv"
    _write_csv(csv_path, [row])

    with pytest.raises(ValueError, match=message) as excinfo:
        load_reconstruction_batch(csv_path)
    assert "data record 1" in str(excinfo.value)


def test_loader_requires_source_and_image_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "recon.csv"
    _write_csv(csv_path, [{"row_id": "a", "prompt": "x"}], columns=("row_id", "prompt"))

    with pytest.raises(ValueError, match="missing required columns: image_path, source_path"):
        load_reconstruction_batch(csv_path)


def test_batch_runs_each_row_through_the_direct_reconstruction_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "recon.csv"
    _write_csv(
        csv_path,
        [
            {"row_id": "first", "source_path": "a.glb", "image_path": "a.jpg"},
            {
                "row_id": "second",
                "source_path": "b.glb",
                "image_path": "b.jpg",
                "label": "B",
                "guidance": "Keep the lid closed.",
            },
        ],
    )
    calls: list[dict[str, object]] = []

    def fake_execute_provider_reconstruction(**kwargs):
        calls.append(dict(kwargs))
        kwargs["progress"]('Provider Reconstruction | activity="stage started" | stage=author')
        row = Path(kwargs["source"]).stem
        record_dir = tmp_path / f"rec_{row}"
        return ProviderReconstructionResult(
            status="reconstruction_ready",
            run_id=f"recon_{row}",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=("exporter note",),
        )

    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_command.execute_provider_reconstruction",
        fake_execute_provider_reconstruction,
    )
    events = []

    result = execute_reconstruction_batch(
        csv_path=csv_path,
        repo_root=tmp_path,
        model="nvidia-test-model",
        reasoning="high",
        concurrency=2,
        progress=events.append,
    )

    assert result.status == "complete"
    assert result.succeeded == 2 and result.failed == 0
    assert result.provider == "openai"
    assert [row.row_id for row in result.rows] == ["first", "second"]
    assert [row.run_id for row in result.rows] == ["recon_a", "recon_b"]
    assert result.rows[0].warnings == ("exporter note",)
    by_source = {Path(call["source"]).name: call for call in calls}
    assert by_source["b.glb"]["label"] == "B"
    # Each row's guidance reaches the single-run primitive, which records it in the record.
    assert by_source["a.glb"]["guidance"] is None
    assert by_source["b.glb"]["guidance"] == "Keep the lid closed."
    assert all(call["model"] == "nvidia-test-model" for call in calls)
    assert all(call["semantic_image"].suffix == ".jpg" for call in calls)
    assert events[0].kind == "batch_started" and events[0].effective_concurrency == 2
    progress_lines = [event.message for event in events if event.kind == "row_progress"]
    assert (
        progress_lines
        == [
            'Provider Reconstruction | activity="stage started" | stage=author',
        ]
        * 2
    )
    assert events[-1].kind == "batch_finished" and events[-1].succeeded == 2


def test_batch_summary_names_the_openai_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only OPENAI_API_KEY in `.env` selects the OpenAI endpoint; the batch summary names it.
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    csv_path = tmp_path / "recon.csv"
    _write_csv(csv_path, [{"row_id": "first", "source_path": "a.glb", "image_path": "a.jpg"}])

    def fake_execute_provider_reconstruction(**kwargs):
        record_dir = tmp_path / "rec_a"
        return ProviderReconstructionResult(
            status="reconstruction_ready",
            run_id="recon_a",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            provider="openai",
        )

    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_command.execute_provider_reconstruction",
        fake_execute_provider_reconstruction,
    )

    result = execute_reconstruction_batch(csv_path=csv_path, repo_root=tmp_path, concurrency=1)

    assert result.status == "complete"
    assert result.provider == "openai" and result.to_dict()["provider"] == "openai"


def test_batch_isolates_a_failed_row_and_keeps_csv_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "recon.csv"
    _write_csv(
        csv_path,
        [
            {"row_id": "bad", "source_path": "bad.glb", "image_path": "bad.jpg"},
            {"row_id": "good", "source_path": "good.glb", "image_path": "good.jpg"},
        ],
    )

    def fake_execute_provider_reconstruction(**kwargs):
        if Path(kwargs["source"]).stem == "bad":
            raise ValueError("source closure escapes the snapshot")
        record_dir = tmp_path / "rec_good"
        return ProviderReconstructionResult(
            status="reconstruction_ready",
            run_id="recon_good",
            record_id="rec_good",
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
        )

    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_command.execute_provider_reconstruction",
        fake_execute_provider_reconstruction,
    )

    result = execute_reconstruction_batch(csv_path=csv_path, repo_root=tmp_path)

    assert result.status == "partial_failure"
    assert [row.status for row in result.rows] == ["failed", "reconstruction_ready"]
    assert result.rows[0].error == "ValueError: source closure escapes the snapshot"
    assert result.rows[1].record_id == "rec_good"


def test_reconstruction_batch_cli_reports_rows_and_uses_the_reconstruction_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    csv_path = tmp_path / "recon.csv"
    csv_path.write_text("row_id,source_path,image_path\none,a.glb,a.jpg\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_execute_reconstruction_batch(**kwargs):
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
                kind="row_started", timestamp=timestamp, slot=1, row_index=1, row_id="one"
            )
        )
        progress(
            ProviderBatchProgressEvent(
                kind="row_finished",
                timestamp=timestamp,
                slot=1,
                row_index=1,
                row_id="one",
                status="reconstruction_ready",
                record_id="rec_one",
            )
        )
        return ProviderBatchResult(
            csv_path=csv_path.resolve(),
            rows=(
                ProviderBatchRowResult(
                    row_id="one",
                    status="reconstruction_ready",
                    record_id="rec_one",
                    record_dir=root / "data" / "records" / "rec_one",
                    run_id="recon_one",
                    warnings=("note",),
                ),
            ),
        )

    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_batch.execute_reconstruction_batch",
        fake_execute_reconstruction_batch,
    )

    assert (
        usdcraft_cli.main(
            [
                "reconstruction-batch",
                str(csv_path),
                "--repo-root",
                str(root),
                "--concurrency",
                "2",
                "--keep-workspace",
            ]
        )
        == 0
    )

    output = capsys.readouterr()
    assert captured["concurrency"] == 2
    assert captured["keep_workspace"] is True
    # The default per-row limit applies without an explicit --run-timeout.
    assert captured["run_timeout_seconds"] == 1920
    assert captured["csv_path"] == csv_path
    assert "status=complete" in output.out
    assert "succeeded=1" in output.out and "failed=0" in output.out
    assert (
        'row=one status=reconstruction_ready record_id=rec_one run_id=recon_one warnings=["note"]'
        in output.out
    )
    assert "Provider Reconstruction Batch" in output.err
    assert 'status=reconstruction_ready | record_id="rec_one"' in output.err
    assert "Provider Generation" not in output.err

    # The same coordinator result in JSON mode; a reconstruction row counts as a success.
    assert (
        usdcraft_cli.main(
            ["reconstruction-batch", str(csv_path), "--repo-root", str(root), "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete" and payload["succeeded"] == 1
    assert payload["rows"][0]["status"] == "reconstruction_ready"


def test_reconstruction_batch_cli_reports_loader_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repo(tmp_path)
    csv_path = tmp_path / "recon.csv"
    csv_path.write_text("row_id,prompt\none,x\n", encoding="utf-8")

    assert usdcraft_cli.main(["reconstruction-batch", str(csv_path), "--repo-root", str(root)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Provider Reconstruction Batch failed: ValueError:")
    assert "image_path, source_path" in err


def test_batch_binds_boolean_switches(tmp_path: Path) -> None:
    csv_path = tmp_path / "switches.csv"
    csv_path.write_text(
        "row_id,source_path,image_path,image_observation,max_turns\na,m.usdz,i.png,0,33\n"
    )
    row = load_reconstruction_batch(csv_path)[0]
    assert row.image_observation is False and row.max_turns == 33


@pytest.mark.parametrize("column", ["reasoning_effort", "lable"])
def test_batch_rejects_unknown_columns(tmp_path: Path, column: str) -> None:
    path = tmp_path / "typo.csv"
    path.write_text(f"row_id,source_path,image_path,{column}\na,m.glb,i.jpg,true\n")
    with pytest.raises(ValueError, match=f"unknown columns: {column}") as exc_info:
        load_reconstruction_batch(path)
    message = str(exc_info.value)
    assert "Allowed columns: row_id, source_path, image_path, prompt," in message
    # The model/reasoning pairing is mentioned only by the actual pairing error.
    assert "together" not in message


def test_batch_rejects_extra_csv_cells(tmp_path: Path) -> None:
    path = tmp_path / "extra.csv"
    path.write_text("row_id,source_path,image_path\na,m.glb,i.jpg,false\n")
    with pytest.raises(ValueError, match="more cells"):
        load_reconstruction_batch(path)


@pytest.mark.parametrize("pair", ["test-model-a,", ",low"])
def test_batch_runtime_override_must_be_a_pair(tmp_path: Path, pair: str) -> None:
    path = tmp_path / "runtime.csv"
    path.write_text(f"row_id,source_path,image_path,model,reasoning\na,m.glb,i.jpg,{pair}\n")
    with pytest.raises(ValueError, match="model and reasoning must be provided together"):
        load_reconstruction_batch(path)


def test_provider_row_runtime_overrides_cli_and_reaches_deadline(tmp_path, monkeypatch):
    path = tmp_path / "runtime.csv"
    path.write_text(
        "row_id,source_path,image_path,model,reasoning\n"
        "a,m.glb,i.jpg,azure/openai/test-model-b,high\n"
        "b,m.glb,i.jpg,,\n"
    )
    calls = {}
    events = []

    def supervised(arguments, **kwargs):
        calls[kwargs["row_id"]] = arguments
        assert kwargs["timeout_seconds"] == 1920
        return ProviderBatchRowResult(row_id=kwargs["row_id"], status="reconstruction_ready")

    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_deadline.execute_with_deadline", supervised
    )
    result = execute_reconstruction_batch(
        csv_path=path,
        repo_root=tmp_path,
        model="azure/openai/test-model-a",
        reasoning="low",
        run_timeout_seconds=1920,
        progress=events.append,
    )
    assert result.succeeded == 2
    assert (calls["a"]["model"], calls["a"]["reasoning"]) == ("azure/openai/test-model-b", "high")
    assert (calls["b"]["model"], calls["b"]["reasoning"]) == ("azure/openai/test-model-a", "low")
    assert events[0].model == events[0].reasoning_effort == "mixed"


def test_validation_only_never_starts_provider_or_writes_workspace(tmp_path, monkeypatch, capsys):
    path = tmp_path / "experiment.csv"
    path.write_text("row_id,source_path,image_path\na,missing.glb,missing.jpg\n")

    def forbidden(**kwargs):
        pytest.fail("validation started the Provider")

    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_batch.execute_reconstruction_batch", forbidden
    )
    before = set(tmp_path.rglob("*"))
    assert usdcraft_cli.main(["reconstruction-batch", str(path), "--validate-only", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "validated",
        "total": 1,
        "scope": "csv_contract",
    }
    assert set(tmp_path.rglob("*")) == before


def test_experiment_groups_reach_the_same_provider_primitive(tmp_path, monkeypatch):
    groups = [("full", True), ("no_observation", False)]
    path = tmp_path / "experiment.csv"
    columns = (*_COLUMNS, "image_observation", "max_turns")
    _write_csv(
        path,
        [
            {
                "row_id": name,
                "source_path": "same.glb",
                "image_path": "same.jpg",
                "image_observation": str(observation).lower(),
                "max_turns": "40",
            }
            for name, observation in groups
        ],
        columns=columns,
    )
    calls = {}

    def execute(**kwargs):
        name = kwargs["batch_context"]["row_id"]
        calls[name] = kwargs
        return ProviderReconstructionResult(
            status="reconstruction_ready",
            run_id=f"recon_{name}",
            record_id=f"rec_{name}",
            record_dir=tmp_path / f"rec_{name}",
            model_path=tmp_path / f"{name}.py",
            workspace_dir=None,
        )

    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_command.execute_provider_reconstruction", execute
    )
    result = execute_reconstruction_batch(
        csv_path=path,
        repo_root=tmp_path,
        model="azure/openai/test-model-b",
        reasoning="high",
        concurrency=3,
    )
    assert result.succeeded == len(groups)
    for name, observation in groups:
        call = calls[name]
        assert call["image_observation"] is observation
        assert call["max_turns"] == 40
        assert call["model"] == "azure/openai/test-model-b" and call["reasoning"] == "high"
        assert call["source"] == (tmp_path / "same.glb").resolve()
    assert len({id(call["batch_context"]) for call in calls.values()}) == len(groups)
