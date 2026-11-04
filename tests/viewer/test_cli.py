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

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.responses import FileResponse

from storage.artifact_contract import physics_output_contract
from storage.repo import StorageRepo
from storage.revisions import INITIAL_REVISION_ID
from tests.viewer.test_api import _write_generation_record
from tests.viewer.test_reconstruction_records import _write_reconstruction_record
from usdcraft import cli as usdcraft_cli
from viewer.api.app import create_app


def _write_generation_record_with_invalid_materialization(
    root: Path, *, record_id: str = "rec_generation_001"
) -> None:
    """A Generation record whose materialization is displayable but contract-invalid.

    `model.urdf` exists and compile status is "success", so the cheap hint alone cannot
    resolve the status; only the full canonical contract can, via a `generation.json`/
    `compile_report.json` schema mismatch (missing `checks_run`).
    """

    repo = StorageRepo(root)
    repo.ensure_layout()
    _write_generation_record(repo, record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "<robot/>")
    revision_dir = repo.layout.record_revision_dir(record_id, INITIAL_REVISION_ID)
    repo.write_json(
        revision_dir / "generation.json",
        {
            "schema_version": 3,
            "pipeline": "generation",
            "status": "generation_ready",
            "record_id": record_id,
            "revision_id": INITIAL_REVISION_ID,
            "validation_profile": "generation",
            "outputs": physics_output_contract(),
        },
    )
    repo.write_json(
        materialization / "compile_report.json",
        {"schema_version": 2, "status": "success", "validation_profile": "generation"},
    )


def test_viewer_default_stays_on_port_8770(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    served: list[tuple[Any, str, int, bool]] = []
    monkeypatch.setattr(usdcraft_cli, "_ensure_viewer_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, *, host, port, reload: served.append((app, host, port, reload)),
    )

    assert usdcraft_cli.main(["viewer", "--repo-root", str(tmp_path)]) == 0

    assert served == [("viewer.api.app:app", "127.0.0.1", 8770, False)]
    assert capsys.readouterr().out == ""


def test_custom_frontend_dist_is_served_by_the_api_app(tmp_path: Path) -> None:
    frontend_dist = tmp_path / "temporary-frontend"
    frontend_dist.mkdir()
    (frontend_dist / "index.html").write_text("<h1>temporary viewer</h1>", encoding="utf-8")

    app = create_app(repo_root=tmp_path, frontend_dist=frontend_dist)
    index_route = next(route for route in app.routes if getattr(route, "path", None) == "/")
    response = asyncio.run(index_route.endpoint())

    assert isinstance(response, FileResponse)
    assert Path(response.path) == frontend_dist / "index.html"
    assert response.headers["cache-control"] == "no-cache"


def test_reconstruction_validate_cli_reports_one_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_reconstruction_record(tmp_path)

    exit_code = usdcraft_cli.main(
        ["reconstruction-validate", "rec_reconstruction_001", "--repo-root", str(tmp_path)]
    )

    out = capsys.readouterr().out
    # Previewability remains separate from canonical validity.
    assert exit_code == 1
    assert "record_id=rec_reconstruction_001" in out
    assert "materialization_status=contract_invalid" in out
    assert "issue:" in out


def test_reconstruction_validate_cli_all_reports_every_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_reconstruction_record(tmp_path, record_id="rec_reconstruction_a")
    _write_reconstruction_record(tmp_path, record_id="rec_reconstruction_b")

    exit_code = usdcraft_cli.main(
        ["reconstruction-validate", "--all", "--repo-root", str(tmp_path)]
    )

    out = capsys.readouterr().out
    # --all also runs the repository data-format validation over the listed records; these
    # synthetic fixtures are deliberately not full canonical records, so it reports them.
    assert exit_code == 1
    assert "record_id=rec_reconstruction_a" in out
    assert "record_id=rec_reconstruction_b" in out
    assert "data_format=invalid records=2" in out and "issue:" in out


def test_reconstruction_validate_cli_exits_nonzero_for_a_blocked_materialization(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from storage.repo import StorageRepo

    _write_reconstruction_record(tmp_path)
    repo = StorageRepo(tmp_path)
    materialization = repo.layout.record_materialization_dir("rec_reconstruction_001")
    (materialization / "model.urdf").unlink()

    exit_code = usdcraft_cli.main(
        ["reconstruction-validate", "rec_reconstruction_001", "--repo-root", str(tmp_path)]
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "materialization_status=missing" in out


def test_reconstruction_validate_cli_requires_a_record_id_or_all(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = usdcraft_cli.main(["reconstruction-validate", "--repo-root", str(tmp_path)])

    assert exit_code == 2
    assert "Pass a record_id or --all." in capsys.readouterr().err


def test_reconstruction_validate_cli_reports_missing_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = usdcraft_cli.main(
        ["reconstruction-validate", "rec_missing", "--repo-root", str(tmp_path)]
    )

    assert exit_code == 1
    assert "record_id=rec_missing status=invalid" in capsys.readouterr().out


def test_generation_validate_cli_reports_one_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_generation_record_with_invalid_materialization(tmp_path)

    exit_code = usdcraft_cli.main(
        ["generation-validate", "rec_generation_001", "--repo-root", str(tmp_path)]
    )

    out = capsys.readouterr().out
    # Previewability remains separate from canonical validity.
    assert exit_code == 1
    assert "record_id=rec_generation_001" in out
    assert "materialization_status=contract_invalid" in out
    assert "issue:" in out


def test_generation_validate_cli_all_reports_every_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_generation_record_with_invalid_materialization(tmp_path, record_id="rec_generation_a")
    _write_generation_record_with_invalid_materialization(tmp_path, record_id="rec_generation_b")

    exit_code = usdcraft_cli.main(["generation-validate", "--all", "--repo-root", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "record_id=rec_generation_a" in out
    assert "record_id=rec_generation_b" in out
    assert "data_format=invalid records=2" in out and "issue:" in out


def test_validate_all_reports_data_format_ok_for_a_fully_persisted_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.reconstruction.test_reconstruction_persist import _persist

    record_dir = _persist(tmp_path)

    exit_code = usdcraft_cli.main(
        ["reconstruction-validate", "--all", "--repo-root", str(tmp_path)]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert f"record_id={record_dir.name}" in out
    assert "data_format=ok records=1" in out


def test_generation_validate_cli_exits_nonzero_for_a_blocked_materialization(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo, "rec_generation_001")

    exit_code = usdcraft_cli.main(
        ["generation-validate", "rec_generation_001", "--repo-root", str(tmp_path)]
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "materialization_status=missing" in out


def test_generation_validate_cli_requires_a_record_id_or_all(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = usdcraft_cli.main(["generation-validate", "--repo-root", str(tmp_path)])

    assert exit_code == 2
    assert "Pass a record_id or --all." in capsys.readouterr().err


def test_generation_validate_cli_reports_missing_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = usdcraft_cli.main(
        ["generation-validate", "rec_missing", "--repo-root", str(tmp_path)]
    )

    assert exit_code == 1
    assert "record_id=rec_missing status=invalid" in capsys.readouterr().out


def test_viewer_read_only_flag_reaches_the_served_app_through_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    monkeypatch.setattr(usdcraft_cli, "_ensure_viewer_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("USDCRAFT_VIEWER_READ_ONLY", raising=False)
    monkeypatch.delenv("USDCRAFT_VIEWER_ALLOWED_HOSTS", raising=False)

    assert usdcraft_cli.main(["viewer", "--repo-root", str(tmp_path), "--read-only"]) == 0
    assert os.environ["USDCRAFT_VIEWER_READ_ONLY"] == "1"
    assert usdcraft_cli.main(["viewer", "--repo-root", str(tmp_path)]) == 0
    assert os.environ["USDCRAFT_VIEWER_READ_ONLY"] == "0"


def test_viewer_host_reaches_the_host_check_through_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    monkeypatch.setattr(usdcraft_cli, "_ensure_viewer_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("USDCRAFT_VIEWER_ALLOWED_HOSTS", "proxy.example")

    assert usdcraft_cli.main(["viewer", "--repo-root", str(tmp_path), "--host", "10.0.0.5"]) == 0
    assert os.environ["USDCRAFT_VIEWER_ALLOWED_HOSTS"] == "10.0.0.5,proxy.example"
    assert usdcraft_cli.main(["viewer", "--repo-root", str(tmp_path), "--host", "10.0.0.5"]) == 0
    assert os.environ["USDCRAFT_VIEWER_ALLOWED_HOSTS"] == "10.0.0.5,proxy.example"


@pytest.mark.parametrize("command", ["generation-validate", "reconstruction-validate"])
@pytest.mark.parametrize("contents", [b"{broken", b"\xff", b"[]", b"{}", None])
def test_validate_all_reports_unreadable_or_unclassified_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, contents: bytes | None
) -> None:
    record = tmp_path / "data/records/rec_broken"
    record.mkdir(parents=True)
    if contents is not None:
        (record / "record.json").write_bytes(contents)
    assert usdcraft_cli.main([command, "--all", "--repo-root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "record_id=rec_broken status=invalid" in out
    assert "data_format=invalid" in out


def test_validate_all_does_not_hide_bad_record_among_good_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.reconstruction.test_reconstruction_persist import _persist

    good = _persist(tmp_path)
    bad = good.parent / "rec_broken"
    bad.mkdir()
    (bad / "record.json").write_text("{broken")
    assert (
        usdcraft_cli.main(["reconstruction-validate", "--all", "--repo-root", str(tmp_path)]) == 1
    )
    out = capsys.readouterr().out
    assert f"record_id={good.name}" in out
    assert "record_id=rec_broken status=invalid" in out


@pytest.mark.parametrize("pipeline", ["generation", "reconstruction"])
def test_validate_all_keeps_records_without_workbench_registration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pipeline: str
) -> None:
    writer = (
        _write_generation_record_with_invalid_materialization
        if pipeline == "generation"
        else _write_reconstruction_record
    )
    writer(tmp_path)
    repo = StorageRepo(tmp_path)
    repo.layout.record_workbench_entry_path(f"rec_{pipeline}_001").unlink()
    assert usdcraft_cli.main([f"{pipeline}-validate", "--all", "--repo-root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"record_id=rec_{pipeline}_001" in out
    assert "workbench.json" in out


@pytest.mark.parametrize("pipeline", ["generation", "reconstruction"])
def test_empty_product_selection_does_not_validate_other_product(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pipeline: str,
) -> None:
    other = "generation" if pipeline == "reconstruction" else "reconstruction"
    record = tmp_path / "data/records/rec_other"
    record.mkdir(parents=True)
    (record / "record.json").write_text('{"pipeline": "' + other + '"}')

    assert usdcraft_cli.main([f"{pipeline}-validate", "--all", "--repo-root", str(tmp_path)]) == 0
    assert "data_format=ok records=0" in capsys.readouterr().out
    assert (
        usdcraft_cli.main([f"{pipeline}-validate", "rec_other", "--repo-root", str(tmp_path)]) == 1
    )
    assert "expected pipeline" in capsys.readouterr().out
