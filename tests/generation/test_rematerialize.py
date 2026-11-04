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

import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Usd, UsdGeom

import storage.materialization_mutations as materialization_mutations
import usdcraft.generation.rematerialize as rematerialize_module
from storage.artifact_contract import (
    REQUIRED_BACKEND_CHECKS,
    REQUIRED_COMPILE_CHECKS,
    classify_materialization_displayability,
    exporter_asset_structure_contract,
    physics_output_contract,
    validate_materialization_contract,
)
from storage.materialization_mutations import begin_materialization_replacement
from storage.record_locks import record_lock
from storage.record_mutations import RecordMutationStore
from storage.repo import StorageRepo
from usdcraft import cli as usdcraft_cli
from usdcraft.generation.rematerialize import (
    RematerializationScopeError,
    rematerialize_record,
)

RECORD_ID = "rec_current_generation_for_rematerialization"
REVISION_ID = "rev_000001"
PROMOTION_ID = "1" * 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_current_record(
    repo: StorageRepo,
    *,
    promotion_id: str = PROMOTION_ID,
) -> tuple[Path, Path, dict[str, object]]:
    repo.ensure_layout()
    record_dir = repo.layout.record_dir(RECORD_ID)
    revision_dir = repo.layout.record_revision_dir(RECORD_ID, REVISION_ID)
    revision_dir.mkdir(parents=True)
    (revision_dir / "inputs").mkdir()
    (revision_dir / "traces").mkdir()
    model_path = revision_dir / "model.py"
    prompt_path = revision_dir / "prompt.txt"
    model_path.write_text("object_model = object()\n", encoding="utf-8")
    prompt = "Create a current simulator-ready asset"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    model_sha256 = _sha256(model_path)

    owner_namespace = {"usdcraft": {"promotion_transaction_id": promotion_id}}
    record = {
        "schema_version": 3,
        "record_id": RECORD_ID,
        "pipeline": "generation",
        "status": "generation_ready",
        "input_kind": "text",
        "active_revision_id": REVISION_ID,
        "artifacts": {"model_py": f"revisions/{REVISION_ID}/model.py"},
        "hashes": {
            "model_py_sha256": model_sha256,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        },
        **owner_namespace,
    }
    repo.write_json(record_dir / "record.json", record)
    revision = {
        "schema_version": 1,
        "record_id": RECORD_ID,
        "revision_id": REVISION_ID,
        "pipeline": "generation",
        "status": "generation_ready",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "hashes": {
            "model_py_sha256": model_sha256,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        },
    }
    repo.write_json(revision_dir / "revision.json", revision)
    repo.write_json(
        revision_dir / "provenance.json",
        {"schema_version": 3, "record_id": RECORD_ID, "pipeline": "generation"},
    )
    generation: dict[str, object] = {
        "schema_version": 3,
        "pipeline": "generation",
        "status": "generation_ready",
        "record_id": RECORD_ID,
        "revision_id": REVISION_ID,
        "validation_profile": "generation",
        "model_sha256": model_sha256,
        "compile_report": (
            f"../../../../cache/record_materialization/{RECORD_ID}/compile_report.json"
        ),
        "input": {
            "kind": "text",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "image_role": None,
            "image_path": None,
            "image_sha256": None,
        },
    }
    generation["outputs"] = physics_output_contract()
    generation.update(owner_namespace)
    repo.write_json(revision_dir / "generation.json", generation)

    materialization = repo.layout.record_materialization_dir(RECORD_ID)
    materialization.mkdir(parents=True)
    (materialization / "old-cache.txt").write_text("preserve until commit", encoding="utf-8")
    old_report = {
        "schema_version": 2,
        "status": "success",
        "validation_profile": "generation",
        **owner_namespace,
    }
    repo.write_json(materialization / "compile_report.json", old_report)
    return record_dir, materialization, generation


def _write_compiled_replacement(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_paths = tuple(
        output_dir / relative
        for relative in (
            "isaac/model.usdc",
            "isaac/model_newton.usdc",
            "isaac/asset.usda",
            "isaac/physics.usdc",
            "isaac/physx.usda",
            "isaac/mujoco.usda",
        )
    )
    for path in stage_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(path.as_posix())
        root_prim = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
        stage.SetDefaultPrim(root_prim)
        stage.GetRootLayer().Save()
    (output_dir / "model.urdf").write_text("<robot name='new'/>", encoding="utf-8")
    (output_dir / "model.usdz").write_bytes(b"new usdz")
    artifact_paths = {
        "model.urdf": output_dir / "model.urdf",
        "model.usdz": output_dir / "model.usdz",
        **{path.relative_to(output_dir).as_posix(): path for path in stage_paths},
    }
    repo_report = {
        "schema_version": 2,
        "status": "success",
        "validation_profile": "generation",
        "asset_structure": physics_output_contract(),
        "model_path": "staged/model.py",
        "checks_run": sorted(REQUIRED_COMPILE_CHECKS),
        "warnings": ["authoring diagnostic retained"],
        "artifacts": {name: _sha256(path) for name, path in artifact_paths.items()},
        "usd_dependency_closure": {
            path.relative_to(output_dir).as_posix(): [path.relative_to(output_dir).as_posix()]
            for path in stage_paths
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
    (output_dir / "compile_report.json").write_text(
        json.dumps(repo_report),
        encoding="utf-8",
    )


def _fake_compile(
    canonical_model: Path,
    canonical_record_dir: Path,
):
    def compile_model(
        model_path: Path,
        output_dir: Path,
        *,
        render: bool,
        require_generation_root_floating: bool,
    ) -> SimpleNamespace:
        assert require_generation_root_floating is True
        assert render is False
        assert model_path != canonical_model
        assert "materialization_replacement_staging" in model_path.parts
        assert _sha256(model_path) == _sha256(canonical_model)
        (model_path.parent / "worker-cwd-write.txt").write_text(
            "must stay outside canonical revision",
            encoding="utf-8",
        )
        assert not (canonical_record_dir / "worker-cwd-write.txt").exists()
        _write_compiled_replacement(Path(output_dir))
        return SimpleNamespace(output_dir=Path(output_dir))

    return compile_model


def test_rematerialize_rebuilds_from_copy_and_preserves_canonical_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_dir, materialization, generation = _write_current_record(repo)
    source_before = _tree_bytes(record_dir)
    canonical_model = repo.layout.record_revision_model_path(RECORD_ID, REVISION_ID)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        _fake_compile(canonical_model, record_dir),
    )

    result = rematerialize_record(
        repo_root=tmp_path,
        record_id=RECORD_ID,
        render=False,
    )

    assert result.record_id == RECORD_ID
    assert result.revision_id == REVISION_ID
    assert result.recovered is False
    assert result.cleanup_pending is False
    assert result.warnings == ("authoring diagnostic retained",)
    assert _tree_bytes(record_dir) == source_before
    assert not (materialization / "old-cache.txt").exists()
    assert (materialization / "model.urdf").read_text(encoding="utf-8") == ("<robot name='new'/>")
    report = repo.read_json(materialization / "compile_report.json")
    assert report["record_id"] == RECORD_ID
    assert report["model_path"] == (f"data/records/{RECORD_ID}/revisions/{REVISION_ID}/model.py")
    assert report["usdcraft"] == {
        "promotion_transaction_id": PROMOTION_ID,
        "rematerialization_transaction_id": result.transaction_id,
        "source_model_sha256": _sha256(canonical_model),
        "source_revision_id": REVISION_ID,
    }
    contract = validate_materialization_contract(
        materialization,
        generation,
        phase="canonical",
        revision_dir=repo.layout.record_revision_dir(RECORD_ID, REVISION_ID),
    )
    assert contract.ok, contract.summary()
    assert not repo.layout.materialization_replacement_staging_dir(RECORD_ID).exists()


def test_report_without_root_body_binding_evidence_is_contract_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_dir, materialization, generation = _write_current_record(repo)
    canonical_model = repo.layout.record_revision_model_path(RECORD_ID, REVISION_ID)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        _fake_compile(canonical_model, record_dir),
    )
    rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)

    report_path = materialization / "compile_report.json"
    report = repo.read_json(report_path)
    report["metrics"]["backend_validation"]["physx"]["checks"].remove(
        "articulation_root_body_binding"
    )
    repo.write_json(report_path, report)

    display = classify_materialization_displayability(
        materialization,
        generation,
        revision_dir=repo.layout.record_revision_dir(RECORD_ID, REVISION_ID),
    )
    assert display.status == "contract_invalid"
    assert any(
        issue.path == "compile_report.json" and "articulation_root_body_binding" in issue.message
        for issue in display.issues
    )


def test_user_can_explicitly_rematerialize_missing_root_binding_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_dir, materialization, generation = _write_current_record(repo)
    canonical_model = repo.layout.record_revision_model_path(RECORD_ID, REVISION_ID)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        _fake_compile(canonical_model, record_dir),
    )
    rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)

    report_path = materialization / "compile_report.json"
    old_report = repo.read_json(report_path)
    for backend in ("physx", "newton"):
        old_report["metrics"]["backend_validation"][backend]["checks"].remove(
            "articulation_root_body_binding"
        )
    repo.write_json(report_path, old_report)
    assert (
        classify_materialization_displayability(
            materialization,
            generation,
            revision_dir=repo.layout.record_revision_dir(RECORD_ID, REVISION_ID),
        ).status
        == "contract_invalid"
    )

    result = rematerialize_record(
        repo_root=tmp_path,
        record_id=RECORD_ID,
        render=False,
    )

    rebuilt_report = repo.read_json(result.report_path)
    for backend in ("physx", "newton"):
        assert (
            "articulation_root_body_binding"
            in rebuilt_report["metrics"]["backend_validation"][backend]["checks"]
        )
    assert validate_materialization_contract(
        result.materialization_dir,
        generation,
        phase="canonical",
        revision_dir=repo.layout.record_revision_dir(RECORD_ID, REVISION_ID),
    ).ok


def test_fixed_generation_root_compile_failure_preserves_old_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_dir, materialization, _ = _write_current_record(repo)
    record_before = _tree_bytes(record_dir)
    materialization_before = _tree_bytes(materialization)

    def fail_compile(*_args: object, **kwargs: object) -> None:
        assert kwargs["require_generation_root_floating"] is True
        raise RuntimeError("Generation root BaseMode must be floating")

    monkeypatch.setattr(rematerialize_module, "compile_model", fail_compile)

    with pytest.raises(RuntimeError, match="BaseMode must be floating"):
        rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)

    assert _tree_bytes(record_dir) == record_before
    assert _tree_bytes(materialization) == materialization_before
    assert not repo.layout.materialization_replacement_staging_dir(RECORD_ID).exists()


def test_unsupported_atomic_exchange_preserves_old_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_dir, materialization, _ = _write_current_record(repo)
    materialization_before = _tree_bytes(materialization)
    canonical_model = repo.layout.record_revision_model_path(RECORD_ID, REVISION_ID)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        _fake_compile(canonical_model, record_dir),
    )
    monkeypatch.setattr(
        materialization_mutations,
        "_atomic_exchange_paths",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.ENOTSUP, "atomic exchange unsupported")),
    )

    with pytest.raises(OSError, match="atomic exchange unsupported"):
        rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)

    assert _tree_bytes(materialization) == materialization_before
    assert not repo.layout.materialization_replacement_staging_dir(RECORD_ID).exists()


def test_next_invocation_recovers_committed_replacement_without_recompile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_dir, materialization, _ = _write_current_record(repo)
    canonical_model = repo.layout.record_revision_model_path(RECORD_ID, REVISION_ID)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        _fake_compile(canonical_model, record_dir),
    )
    real_rmtree = materialization_mutations.shutil.rmtree
    cleanup_attempted = False

    def leave_previous_for_recovery(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal cleanup_attempted
        if Path(path) == (
            repo.layout.materialization_replacement_staging_dir(RECORD_ID) / "replacement"
        ):
            cleanup_attempted = True
            raise OSError("simulated cleanup interruption")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        materialization_mutations.shutil,
        "rmtree",
        leave_previous_for_recovery,
    )
    first = rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)
    assert cleanup_attempted is True
    assert first.cleanup_pending is True
    assert (materialization / "model.urdf").is_file()
    assert repo.layout.materialization_replacement_staging_dir(RECORD_ID).is_dir()

    monkeypatch.setattr(materialization_mutations.shutil, "rmtree", real_rmtree)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        lambda *_args, **_kwargs: pytest.fail("recovery must not recompile"),
    )
    recovered = rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)

    assert recovered.recovered is True
    assert recovered.transaction_id == first.transaction_id
    assert recovered.cleanup_pending is False
    assert not repo.layout.materialization_replacement_staging_dir(RECORD_ID).exists()


def test_empty_manifestless_cleanup_tail_is_safely_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_dir, materialization, _ = _write_current_record(repo)
    canonical_model = repo.layout.record_revision_model_path(RECORD_ID, REVISION_ID)
    old_report = repo.read_json(materialization / "compile_report.json")
    old_report["usdcraft"] = {
        "promotion_transaction_id": PROMOTION_ID,
        "rematerialization_transaction_id": "4" * 32,
    }
    repo.write_json(materialization / "compile_report.json", old_report)
    repo.layout.materialization_replacement_staging_dir(RECORD_ID).mkdir(parents=True)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        _fake_compile(canonical_model, record_dir),
    )

    result = rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)

    assert result.recovered is False
    assert not repo.layout.materialization_replacement_staging_dir(RECORD_ID).exists()


def test_rematerialize_rejects_mixed_promotion_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    _write_current_record(repo)
    report_path = repo.layout.record_materialization_compile_report_path(RECORD_ID)
    report = repo.read_json(report_path)
    report["usdcraft"]["promotion_transaction_id"] = "b" * 32
    repo.write_json(report_path, report)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        lambda *_args, **_kwargs: pytest.fail("mixed ownership must not compile"),
    )

    with pytest.raises(RematerializationScopeError, match="ownership"):
        rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)


def test_rematerialize_validates_prompt_binding_before_model_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    _write_current_record(repo)
    generation_path = repo.layout.record_revision_generation_path(RECORD_ID, REVISION_ID)
    generation = repo.read_json(generation_path)
    generation["input"]["prompt_sha256"] = "0" * 64
    repo.write_json(generation_path, generation)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        lambda *_args, **_kwargs: pytest.fail("invalid source binding must not execute model.py"),
    )

    with pytest.raises(RematerializationScopeError, match="prompt_sha256"):
        rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)


@pytest.mark.parametrize(
    ("image_path", "image_sha256", "expected"),
    [
        ("../outside.png", "0" * 64, "revision-relative"),
        ("inputs/inspiration.png", "0" * 64, "missing or has the wrong hash"),
    ],
)
def test_rematerialize_validates_image_containment_and_hash_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_path: str,
    image_sha256: str,
    expected: str,
) -> None:
    repo = StorageRepo(tmp_path)
    _write_current_record(repo)
    record_path = repo.layout.record_metadata_path(RECORD_ID)
    record = repo.read_json(record_path)
    record["input_kind"] = "text_image"
    repo.write_json(record_path, record)
    generation_path = repo.layout.record_revision_generation_path(RECORD_ID, REVISION_ID)
    generation = repo.read_json(generation_path)
    generation["input"].update(
        {
            "kind": "text_image",
            "image_role": "inspiration",
            "image_path": image_path,
            "image_sha256": image_sha256,
        }
    )
    repo.write_json(generation_path, generation)
    if image_path.startswith("inputs/"):
        image = repo.layout.record_revision_dir(RECORD_ID, REVISION_ID) / image_path
        image.write_bytes(b"canonical image")
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        lambda *_args, **_kwargs: pytest.fail("invalid image binding must not execute model.py"),
    )

    with pytest.raises(RematerializationScopeError, match=expected):
        rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)


def test_rematerialize_rejects_pending_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    _write_current_record(repo)
    repo.layout.record_deletion_staging_dir(RECORD_ID).mkdir(parents=True)
    monkeypatch.setattr(
        rematerialize_module,
        "compile_model",
        lambda *_args, **_kwargs: pytest.fail("pending deletion must block compile"),
    )

    with pytest.raises(RematerializationScopeError, match="deletion recovery is pending"):
        rematerialize_record(repo_root=tmp_path, record_id=RECORD_ID, render=False)


def test_delete_recovers_uncommitted_replacement_before_mutating_record(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    _write_current_record(repo)
    with record_lock(repo, RECORD_ID):
        replacement = begin_materialization_replacement(repo, RECORD_ID, "2" * 32)
        replacement.replacement_dir.mkdir()
        (replacement.replacement_dir / "partial.txt").write_text("partial", encoding="utf-8")

    result = RecordMutationStore(repo).delete_record(RECORD_ID)

    assert result.materialization_deleted is True
    assert not repo.layout.record_dir(RECORD_ID).exists()
    assert not repo.layout.record_materialization_dir(RECORD_ID).exists()
    assert not replacement.staging_dir.exists()


def test_rematerialize_cli_is_explicit_and_reports_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "data/cache/record_materialization/record/compile_report.json"
    result = SimpleNamespace(
        record_id=RECORD_ID,
        revision_id=REVISION_ID,
        model_path=tmp_path / "data/records/record/model.py",
        materialization_dir=report.parent,
        report_path=report,
        transaction_id="3" * 32,
        warnings=("one warning",),
        recovered=False,
        cleanup_pending=False,
    )
    seen_render: list[bool] = []

    def fake_rematerialize(**kwargs: object) -> SimpleNamespace:
        seen_render.append(bool(kwargs["render"]))
        return result

    monkeypatch.setattr(rematerialize_module, "rematerialize_record", fake_rematerialize)

    assert (
        usdcraft_cli.main(
            [
                "rematerialize",
                RECORD_ID,
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "status=success" in output
    assert f"record_id={RECORD_ID}" in output
    assert f"materialization={report.parent}" in output
    assert "recovered=false" in output
    assert "cleanup_pending=false" in output
    assert "warning=one warning" in output
    assert seen_render == [False]

    assert (
        usdcraft_cli.main(
            [
                "rematerialize",
                RECORD_ID,
                "--repo-root",
                str(tmp_path),
                "--render",
            ]
        )
        == 0
    )
    assert seen_render == [False, True]
