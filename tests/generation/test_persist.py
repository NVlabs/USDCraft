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

import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest
from pxr import Sdf, Usd, UsdGeom

import usdcraft.generation.persist as persist_module
from storage.artifact_contract import (
    REQUIRED_BACKEND_CHECKS,
    REQUIRED_COMPILE_CHECKS,
    exporter_asset_structure_contract,
    physics_output_contract,
)
from storage.collections import CollectionStore
from storage.data_validation import validate_data_format
from storage.record_locks import record_lock as storage_record_lock
from storage.record_mutations import PROMOTION_MANIFEST, recover_pending_promotions
from storage.repo import StorageRepo
from usdcraft.compile.command import CompileArtifacts
from usdcraft.generation.persist import (
    NVIDIA_PROVIDER_AUTHORING_IDENTITY,
    OPENAI_PROVIDER_AUTHORING_IDENTITY,
    GenerationAuthoringIdentity,
    persist_generation,
    provider_authoring_identity,
)
from usdcraft.pipeline.contracts import CodexRuntimeSelection, ProviderRuntimeSelection


def _artifacts(root: Path) -> CompileArtifacts:
    output = root / "build"
    usdc = output / "isaac" / "model.usdc"
    usdc.parent.mkdir(parents=True)
    newton_usdc = output / "isaac" / "model_newton.usdc"
    asset_usda = output / "isaac" / "asset.usda"
    physics_usdc = output / "isaac" / "physics.usdc"
    physx_usda = output / "isaac" / "physx.usda"
    mujoco_usda = output / "isaac" / "mujoco.usda"
    urdf = output / "model.urdf"
    usdz = output / "model.usdz"
    report = output / "compile_report.json"
    urdf.write_text("<robot name='test'/>", encoding="utf-8")
    for path in (
        usdc,
        newton_usdc,
        asset_usda,
        physics_usdc,
        physx_usda,
        mujoco_usda,
    ):
        stage = Usd.Stage.CreateNew(path.as_posix())
        root_prim = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
        stage.SetDefaultPrim(root_prim)
        stage.GetRootLayer().Save()
    usdz.write_bytes(b"usdz")
    artifact_paths = {
        "model.urdf": urdf,
        "model.usdz": usdz,
        "isaac/model.usdc": usdc,
        "isaac/model_newton.usdc": newton_usdc,
        "isaac/asset.usda": asset_usda,
        "isaac/physics.usdc": physics_usdc,
        "isaac/physx.usda": physx_usda,
        "isaac/mujoco.usda": mujoco_usda,
    }
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "success",
                "validation_profile": "generation",
                "asset_structure": physics_output_contract(),
                "checks_run": sorted(REQUIRED_COMPILE_CHECKS),
                "warnings": [],
                "artifacts": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in artifact_paths.items()
                },
                "usd_dependency_closure": {
                    name: [name] for name in artifact_paths if name.startswith("isaac/")
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
        ),
        encoding="utf-8",
    )
    return CompileArtifacts(
        output_dir=output,
        urdf_path=urdf,
        usdc_path=usdc,
        newton_usdc_path=newton_usdc,
        asset_usda_path=asset_usda,
        physics_usdc_path=physics_usdc,
        physx_usda_path=physx_usda,
        mujoco_usda_path=mujoco_usda,
        usdz_path=usdz,
        report_path=report,
        render_paths=(),
        warnings=(),
    )


def _workspace(root: Path, *, with_timeline: bool = True) -> Path:
    workspace = root / "workspace"
    codex_dir = workspace / ".codex-run"
    codex_dir.mkdir(parents=True)
    (workspace / "model.py").write_text("object_model = object()\n", encoding="utf-8")
    (workspace / "TASK.md").write_text("Create one asset.\n", encoding="utf-8")
    (codex_dir / "stdout.jsonl").write_text("{}\n", encoding="utf-8")
    (codex_dir / "final.json").write_text(
        '{"status":"complete","summary":"done"}\n', encoding="utf-8"
    )
    (codex_dir / "stderr.log").write_text("", encoding="utf-8")
    if with_timeline:
        (codex_dir / "timeline.jsonl").write_text(
            '{"schema_version":1,"record_type":"process.started"}\n',
            encoding="utf-8",
        )
    return workspace


def _runtime() -> CodexRuntimeSelection:
    return CodexRuntimeSelection(
        model_id="test-model",
        reasoning_effort="ultra",
        runtime_source="codex_root_thread_state",
        selection_source="inherited_main_codex_runtime",
    )


def _provider_runtime() -> ProviderRuntimeSelection:
    return ProviderRuntimeSelection(
        model_id="nvidia-test-model",
        reasoning_effort="high",
    )


def test_semantic_signal_bundle_does_not_block_promotion(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    artifacts = _artifacts(workspace)
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    report["signal_bundle"] = {
        "status": "failed",
        "summary": "authoring diagnostic only",
        "signals": [{"name": "semantic_qc", "status": "failed"}],
    }
    artifacts.report_path.write_text(json.dumps(report), encoding="utf-8")

    record_dir = persist_generation(
        repo_root=tmp_path,
        run_id="gen_semantic_diagnostic",
        prompt="Create an asset with a non-blocking semantic diagnostic",
        image_path=None,
        label=None,
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"final_status": "complete"},
        artifacts=artifacts,
    )

    assert record_dir.is_dir()
    assert repo.layout.record_materialization_compile_report_path(record_dir.name).is_file()


def test_persist_generation_is_browsable_and_schema_valid(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)

    record_dir = persist_generation(
        repo_root=tmp_path,
        run_id="gen_test_0001",
        prompt="Create a functional desk lamp",
        image_path=None,
        label="Desk lamp",
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"elapsed_seconds": 1.0, "final_status": "complete"},
        artifacts=_artifacts(workspace),
    )

    record_id = record_dir.name
    workbench = CollectionStore(repo).load_workbench()
    assert workbench is not None
    assert [entry["record_id"] for entry in workbench["entries"]] == [record_id]
    assert (record_dir / "revisions" / "rev_000001" / "traces" / "codex_final.json").is_file()
    record = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    revision_dir = record_dir / "revisions" / "rev_000001"
    timeline_path = revision_dir / "traces" / "codex_timeline.jsonl"
    assert timeline_path.read_text(encoding="utf-8") == (
        '{"schema_version":1,"record_type":"process.started"}\n'
    )
    generation = json.loads((revision_dir / "generation.json").read_text(encoding="utf-8"))
    provenance = json.loads((revision_dir / "provenance.json").read_text(encoding="utf-8"))
    revision = json.loads((revision_dir / "revision.json").read_text(encoding="utf-8"))
    expected = {
        "model_id": "test-model",
        "reasoning_effort": "ultra",
        "runtime_source": "codex_root_thread_state",
        "selection_source": "inherited_main_codex_runtime",
    }
    assert {key: record[key] for key in expected} == expected
    assert {key: generation["creator"][key] for key in expected} == expected
    assert {key: provenance["runtime"][key] for key in expected} == expected
    assert {key: revision["generation"][key] for key in expected} == expected
    assert provenance["schema_version"] == 3
    task_path = revision_dir / "traces" / "codex_task.md"
    assert (
        provenance["prompting"]["system_prompt_sha256"]
        == hashlib.sha256(task_path.read_bytes()).hexdigest()
    )
    result = validate_data_format(repo, require_records=True, record_ids=[record_id])
    assert result.errors == []
    assert "codex_timeline.jsonl" not in json.dumps(generation["outputs"])
    assert "codex_timeline.jsonl" not in json.dumps(record["hashes"])
    assert "codex_timeline.jsonl" not in json.dumps(revision["hashes"])
    compile_report = json.loads(
        repo.layout.record_materialization_compile_report_path(record_id).read_text(
            encoding="utf-8"
        )
    )
    assert "codex_timeline.jsonl" not in json.dumps(compile_report["artifacts"])
    assert Usd.Stage.Open(repo.layout.record_materialization_newton_usdc_path(record_id).as_posix())
    assert not repo.layout.record_materialization_renders_dir(record_id).exists()

    timeline_path.write_text("not valid jsonl\n", encoding="utf-8")
    assert validate_data_format(repo, require_records=True, record_ids=[record_id]).errors == []
    timeline_path.unlink()
    assert validate_data_format(repo, require_records=True, record_ids=[record_id]).errors == []

    record["reasoning_effort"] = "low"
    repo.write_json(record_dir / "record.json", record)
    mismatch = validate_data_format(repo, require_records=True, record_ids=[record_id])
    assert any("reasoning_effort must be 'low', got 'ultra'" in error for error in mismatch.errors)


def test_provider_promotion_reuses_contract_with_distinct_identity_and_traces(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    provider_dir = workspace / ".provider-run"
    provider_dir.mkdir()
    (provider_dir / "events.jsonl").write_text('{"type":"tool_result"}\n', encoding="utf-8")
    (provider_dir / "final.json").write_text(
        '{"status":"complete","summary":"done"}\n',
        encoding="utf-8",
    )
    (provider_dir / "stderr.log").write_text("", encoding="utf-8")

    record_dir = persist_generation(
        repo_root=tmp_path,
        run_id="provider_test_0001",
        prompt="Create a Provider-authored desk lamp",
        image_path=None,
        label="Provider desk lamp",
        runtime=_provider_runtime(),
        workspace_dir=workspace,
        runtime_report={"elapsed_seconds": 2.0, "final_status": "complete", "turn_count": 3},
        artifacts=_artifacts(workspace),
        authoring_identity=NVIDIA_PROVIDER_AUTHORING_IDENTITY,
    )

    revision_dir = record_dir / "revisions" / "rev_000001"
    record = repo.read_json(record_dir / "record.json")
    generation = repo.read_json(revision_dir / "generation.json")
    provenance = repo.read_json(revision_dir / "provenance.json")
    revision = repo.read_json(revision_dir / "revision.json")
    assert record["agent_harness"] == "provider"
    assert record["provider"] == "nvidia"
    assert record["creator"]["agent"] == "provider"
    assert generation["creator"]["agent_harness"] == "provider"
    assert generation["creator"]["provider"] == "nvidia"
    assert provenance["runtime"]["agent"] == "provider"
    assert provenance["runtime"]["provider"] == "nvidia"
    assert revision["generation"]["agent_harness"] == "provider"
    traces = revision_dir / "traces"
    assert (traces / "provider_task.md").is_file()
    assert (traces / "provider_events.jsonl").is_file()
    assert (traces / "provider_final.json").is_file()
    assert not any(path.name.startswith("codex_") for path in traces.iterdir())
    assert (
        validate_data_format(
            repo,
            require_records=True,
            record_ids=[record_dir.name],
        ).errors
        == []
    )

    provenance["runtime"]["agent"] = "codex"
    repo.write_json(revision_dir / "provenance.json", provenance)
    mismatch = validate_data_format(repo, require_records=True, record_ids=[record_dir.name])
    assert any(
        "runtime.agent must be 'provider', got 'codex'" in error for error in mismatch.errors
    )


def test_provider_authoring_identity_follows_the_endpoint() -> None:
    assert provider_authoring_identity("nvidia") is NVIDIA_PROVIDER_AUTHORING_IDENTITY
    assert provider_authoring_identity("openai") is OPENAI_PROVIDER_AUTHORING_IDENTITY
    with pytest.raises(ValueError, match="requires provider 'codex-cli'"):
        GenerationAuthoringIdentity(agent_harness="codex", provider="openai")
    with pytest.raises(ValueError, match="requires provider 'nvidia' or 'openai'"):
        GenerationAuthoringIdentity(agent_harness="provider", provider="codex-cli")


def test_openai_provider_record_persists_its_endpoint_and_validates(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    provider_dir = workspace / ".provider-run"
    provider_dir.mkdir()
    (provider_dir / "events.jsonl").write_text('{"type":"tool_result"}\n', encoding="utf-8")
    (provider_dir / "final.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    (provider_dir / "stderr.log").write_text("", encoding="utf-8")

    record_dir = persist_generation(
        repo_root=tmp_path,
        run_id="provider_openai_0001",
        prompt="Create an OpenAI-served desk lamp",
        image_path=None,
        label=None,
        runtime=ProviderRuntimeSelection(
            model_id="gpt-test-model",
            reasoning_effort="high",
            runtime_source="openai-responses-api",
        ),
        workspace_dir=workspace,
        runtime_report={"elapsed_seconds": 2.0, "final_status": "complete", "turn_count": 3},
        artifacts=_artifacts(workspace),
        authoring_identity=provider_authoring_identity("openai"),
    )

    revision_dir = record_dir / "revisions" / "rev_000001"
    record = repo.read_json(record_dir / "record.json")
    generation = repo.read_json(revision_dir / "generation.json")
    provenance = repo.read_json(revision_dir / "provenance.json")
    assert record["agent_harness"] == "provider" and record["provider"] == "openai"
    assert generation["creator"]["provider"] == "openai"
    assert provenance["runtime"]["provider"] == "openai"
    assert provenance["runtime"]["runtime_source"] == "openai-responses-api"
    assert (
        validate_data_format(repo, require_records=True, record_ids=[record_dir.name]).errors == []
    )

    # The Codex arm publishes only as codex-cli; an OpenAI endpoint on a Codex record is invalid.
    record["agent_harness"] = "codex"
    repo.write_json(record_dir / "record.json", record)
    mismatch = validate_data_format(repo, require_records=True, record_ids=[record_dir.name])
    assert any(
        "provider must be 'codex-cli' for agent_harness 'codex', got 'openai'" in error
        for error in mismatch.errors
    )

    record["agent_harness"] = "provider"
    record["provider"] = "anthropic"
    repo.write_json(record_dir / "record.json", record)
    unknown = validate_data_format(repo, require_records=True, record_ids=[record_dir.name])
    assert any(
        "provider must be 'nvidia' or 'openai' for agent_harness 'provider', got 'anthropic'"
        in error
        for error in unknown.errors
    )


def test_persist_generation_accepts_missing_debug_timeline(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path, with_timeline=False)

    record_dir = persist_generation(
        repo_root=tmp_path,
        run_id="gen_without_debug_timeline",
        prompt="Create an asset without an optional debug sidecar",
        image_path=None,
        label=None,
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"elapsed_seconds": 1.0, "final_status": "complete"},
        artifacts=_artifacts(workspace),
    )

    timeline_path = record_dir / "revisions" / "rev_000001" / "traces" / "codex_timeline.jsonl"
    assert not timeline_path.exists()
    assert (
        validate_data_format(
            repo,
            require_records=True,
            record_ids=[record_dir.name],
        ).errors
        == []
    )


def test_debug_timeline_copy_failure_does_not_block_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    artifacts = _artifacts(workspace)
    real_copy2 = persist_module.shutil.copy2

    def fail_timeline_copy(source: object, destination: object, *args: object, **kwargs: object):
        if Path(source).name == "timeline.jsonl":
            raise OSError("diagnostic copy failed")
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(persist_module.shutil, "copy2", fail_timeline_copy)

    record_dir = persist_generation(
        repo_root=tmp_path,
        run_id="gen_debug_copy_failure",
        prompt="Create an asset while optional trace copying fails",
        image_path=None,
        label=None,
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"elapsed_seconds": 1.0, "final_status": "complete"},
        artifacts=artifacts,
    )

    traces = record_dir / "revisions" / "rev_000001" / "traces"
    assert not (traces / "codex_timeline.jsonl").exists()
    assert "diagnostic copy failed" in (traces / "codex_trace_errors.log").read_text(
        encoding="utf-8"
    )
    assert (
        validate_data_format(
            repo,
            require_records=True,
            record_ids=[record_dir.name],
        ).errors
        == []
    )


def test_data_validation_rejects_tampered_newton_artifact(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    record_dir = persist_generation(
        repo_root=tmp_path,
        run_id="gen_newton_hash_0001",
        prompt="Create a dual-backend asset",
        image_path=None,
        label=None,
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"final_status": "complete"},
        artifacts=_artifacts(workspace),
    )

    record_id = record_dir.name
    repo.layout.record_materialization_newton_usdc_path(record_id).write_bytes(b"tampered")
    result = validate_data_format(repo, require_records=True, record_ids=[record_id])

    assert any("artifacts.isaac/model_newton.usdc" in error for error in result.errors)


def test_data_validation_recomputes_usd_dependency_closure(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    record_dir = persist_generation(
        repo_root=tmp_path,
        run_id="gen_dependency_tamper_0001",
        prompt="Create a dual-backend asset",
        image_path=None,
        label=None,
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"final_status": "complete"},
        artifacts=_artifacts(workspace),
    )

    record_id = record_dir.name
    materialization = repo.layout.record_materialization_dir(record_id)
    dependency = materialization / "assets" / "unreported.png"
    dependency.parent.mkdir(parents=True)
    dependency.write_bytes(b"unreported")
    stage_path = repo.layout.record_materialization_newton_usdc_path(record_id)
    stage = Usd.Stage.Open(stage_path.as_posix())
    stage.GetDefaultPrim().CreateAttribute(
        "usdcraft:testDependency",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("../assets/unreported.png"))
    stage.GetRootLayer().Save()

    report_path = materialization / "compile_report.json"
    report = repo.read_json(report_path)
    report["artifacts"]["isaac/model_newton.usdc"] = hashlib.sha256(
        stage_path.read_bytes()
    ).hexdigest()
    repo.write_json(report_path, report)

    result = validate_data_format(repo, require_records=True, record_ids=[record_id])
    assert any(
        "usd_dependency_closure.isaac/model_newton.usdc must equal" in error
        for error in result.errors
    )


def test_persist_rejects_disappeared_asset_structure_sidecar(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    artifacts = _artifacts(workspace)
    artifacts.mujoco_usda_path.unlink()

    with pytest.raises(FileNotFoundError, match="mujoco.usda"):
        persist_generation(
            repo_root=tmp_path,
            run_id="gen_missing_sidecar_0001",
            prompt="Create a dual-backend asset",
            image_path=None,
            label=None,
            runtime=_runtime(),
            workspace_dir=workspace,
            runtime_report={"final_status": "complete"},
            artifacts=artifacts,
        )
    assert list(repo.layout.records_root.iterdir()) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("failed_status", "status must be 'success'"),
        ("unknown_schema", "schema_version must be 2"),
        ("missing_check", "newton_usd_strict"),
        ("missing_root_check", "generation_root_floating"),
        ("missing_finite_backend_check", "finite_authored_values"),
        ("failed_backend", "backend_validation.newton.status"),
    ],
)
def test_persist_rejects_invalid_current_compile_evidence(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    artifacts = _artifacts(workspace)
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    if mutation == "failed_status":
        report["status"] = "failed"
    elif mutation == "unknown_schema":
        report["schema_version"] = 99
    elif mutation == "missing_check":
        report["checks_run"].remove("newton_usd_strict")
    elif mutation == "missing_root_check":
        report["checks_run"].remove("generation_root_floating")
    elif mutation == "missing_finite_backend_check":
        report["metrics"]["backend_validation"]["physx"]["checks"].remove("finite_authored_values")
    elif mutation == "failed_backend":
        report["metrics"]["backend_validation"]["newton"]["status"] = "failed"
    artifacts.report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        persist_generation(
            repo_root=tmp_path,
            run_id=f"gen_invalid_{mutation}",
            prompt="Create a dual-backend asset",
            image_path=None,
            label=None,
            runtime=_runtime(),
            workspace_dir=workspace,
            runtime_report={"final_status": "complete"},
            artifacts=artifacts,
        )
    assert list(repo.layout.records_root.iterdir()) == []


def test_persist_rejects_malformed_compile_report(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    artifacts = _artifacts(workspace)
    artifacts.report_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        persist_generation(
            repo_root=tmp_path,
            run_id="gen_malformed_report",
            prompt="Create a dual-backend asset",
            image_path=None,
            label=None,
            runtime=_runtime(),
            workspace_dir=workspace,
            runtime_report={"final_status": "complete"},
            artifacts=artifacts,
        )
    assert list(repo.layout.records_root.iterdir()) == []


def test_failed_staging_retains_recovery_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)

    def fail_staging(**_kwargs: object) -> Path:
        raise RuntimeError("injected staging failure")

    monkeypatch.setattr(persist_module, "_stage_and_commit_promotion", fail_staging)

    with pytest.raises(RuntimeError, match="injected staging failure"):
        persist_generation(
            repo_root=tmp_path,
            run_id="gen_staging_failure",
            prompt="Create a recoverable staging failure",
            image_path=None,
            label=None,
            runtime=_runtime(),
            workspace_dir=workspace,
            runtime_report={"final_status": "complete"},
            artifacts=_artifacts(workspace),
        )

    staging_dirs = list(repo.layout.promotion_staging_root.iterdir())
    assert len(staging_dirs) == 1
    assert repo.read_json(staging_dirs[0] / PROMOTION_MANIFEST)["state"] == "staging"

    recovered = recover_pending_promotions(repo)
    assert recovered.errors == ()
    assert recovered.cleaned_staging_dirs == (staging_dirs[0].name,)


def test_committed_manifest_failure_rolls_back_and_remains_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    real_write_manifest = persist_module.write_promotion_manifest

    def fail_committed_manifest(
        manifest_repo: StorageRepo,
        staging_dir: Path,
        *,
        record_id: str,
        transaction_id: str,
        state: Literal["staging", "committing", "committed"],
    ) -> None:
        if state == "committed":
            raise OSError("injected committed marker failure")
        real_write_manifest(
            manifest_repo,
            staging_dir,
            record_id=record_id,
            transaction_id=transaction_id,
            state=state,
        )

    monkeypatch.setattr(persist_module, "write_promotion_manifest", fail_committed_manifest)

    with pytest.raises(OSError, match="injected committed marker failure"):
        persist_generation(
            repo_root=tmp_path,
            run_id="gen_commit_marker_failure",
            prompt="Create a recoverable commit marker failure",
            image_path=None,
            label=None,
            runtime=_runtime(),
            workspace_dir=workspace,
            runtime_report={"final_status": "complete"},
            artifacts=_artifacts(workspace),
        )

    staging_dirs = list(repo.layout.promotion_staging_root.iterdir())
    assert len(staging_dirs) == 1
    manifest = repo.read_json(staging_dirs[0] / PROMOTION_MANIFEST)
    record_id = manifest["record_id"]
    assert manifest["state"] == "committing"
    assert not repo.layout.record_dir(record_id).exists()
    assert not repo.layout.record_materialization_dir(record_id).exists()

    recovered = recover_pending_promotions(repo)
    assert recovered.errors == ()
    assert recovered.rolled_back_record_ids == (record_id,)


def test_persist_generation_uses_one_timestamp_for_id_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    expected_time = datetime(2026, 7, 13, 23, 59, 59, tzinfo=timezone.utc)
    calls = 0

    def fixed_utc_now() -> datetime:
        nonlocal calls
        calls += 1
        return expected_time

    monkeypatch.setattr(persist_module, "_utc_now", fixed_utc_now)

    record_dir = persist_generation(
        repo_root=tmp_path,
        run_id="gen_timestamp_test",
        prompt="Create a timestamped asset",
        image_path=None,
        label=None,
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"final_status": "complete"},
        artifacts=_artifacts(workspace),
    )

    assert calls == 1
    assert "_20260713_235959_" in record_dir.name
    record = repo.read_json(record_dir / "record.json")
    revision_dir = record_dir / "revisions" / "rev_000001"
    generation = repo.read_json(revision_dir / "generation.json")
    revision = repo.read_json(revision_dir / "revision.json")
    workbench = CollectionStore(repo).load_workbench()
    assert workbench is not None
    assert record["created_at"] == "2026-07-13T23:59:59Z"
    assert record["updated_at"] == record["created_at"]
    assert generation["created_at"] == record["created_at"]
    assert revision["created_at"] == record["created_at"]
    assert workbench["entries"][0]["added_at"] == record["created_at"]


def test_persist_generation_rechecks_conflict_inside_commit_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    artifacts = _artifacts(workspace)
    record_id = "rec_commit_conflict"
    real_record_lock = storage_record_lock
    monkeypatch.setattr(
        persist_module,
        "_record_id",
        lambda prompt, run_id, created_at: record_id,
    )

    @contextmanager
    def inject_conflict(locked_repo: StorageRepo, locked_record_id: str):
        with real_record_lock(locked_repo, locked_record_id):
            locked_repo.write_text(
                locked_repo.layout.record_dir(locked_record_id) / "sentinel.txt",
                "existing",
            )
            yield

    monkeypatch.setattr(persist_module, "record_lock", inject_conflict)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        persist_generation(
            repo_root=tmp_path,
            run_id="gen_conflict",
            prompt="Create a conflict",
            image_path=None,
            label=None,
            runtime=_runtime(),
            workspace_dir=workspace,
            runtime_report={"final_status": "complete"},
            artifacts=artifacts,
        )

    assert (repo.layout.record_dir(record_id) / "sentinel.txt").read_text() == "existing"
    assert not repo.layout.record_materialization_dir(record_id).exists()


def test_persist_generation_uses_shared_record_lock_for_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    workspace = _workspace(tmp_path)
    artifacts = _artifacts(workspace)
    record_id = "rec_shared_commit_lock"
    commit_attempted = threading.Event()
    results: list[Path] = []
    errors: list[BaseException] = []
    monkeypatch.setattr(
        persist_module,
        "_record_id",
        lambda prompt, run_id, created_at: record_id,
    )

    def observed_record_lock(locked_repo: StorageRepo, locked_record_id: str):
        commit_attempted.set()
        return storage_record_lock(locked_repo, locked_record_id)

    monkeypatch.setattr(persist_module, "record_lock", observed_record_lock)

    def persist() -> None:
        try:
            results.append(
                persist_generation(
                    repo_root=tmp_path,
                    run_id="gen_shared_lock",
                    prompt="Create a serialized asset",
                    image_path=None,
                    label=None,
                    runtime=_runtime(),
                    workspace_dir=workspace,
                    runtime_report={"final_status": "complete"},
                    artifacts=artifacts,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    with storage_record_lock(repo, record_id):
        worker = threading.Thread(target=persist, daemon=True)
        worker.start()
        assert commit_attempted.wait(timeout=5)
        worker.join(timeout=0.05)
        assert worker.is_alive()
        assert not repo.layout.record_dir(record_id).exists()

    worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []
    assert results == [repo.layout.record_dir(record_id)]


def test_candidate_images_are_archived_with_hashes_and_without_external_links(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    feedback = workspace / "author_feedback" / "view"
    feedback.mkdir(parents=True)
    image = feedback / "candidate.png"
    image.write_bytes(b"candidate bytes")
    outside = tmp_path / "outside.json"
    outside.write_text("private")
    (feedback / "outside.json").symlink_to(outside)
    record = persist_generation(
        repo_root=tmp_path,
        run_id="gen_feedback",
        prompt="a lamp",
        image_path=None,
        label=None,
        runtime=_runtime(),
        workspace_dir=workspace,
        runtime_report={"final_status": "complete"},
        artifacts=_artifacts(workspace),
    )
    traces = record / "revisions/rev_000001/traces"
    assert (traces / "author_feedback/view/candidate.png").read_bytes() == image.read_bytes()
    assert not (traces / "author_feedback/view/outside.json").exists()
    assert json.loads((traces / "author_feedback_index.json").read_text())["files"] == [
        {"path": "view/candidate.png", "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    ]
