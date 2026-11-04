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
import hashlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from storage.artifact_contract import MaterializationDisplayResult
from storage.collections import CollectionStore
from storage.record_mutations import RecordMutationStore
from storage.repo import StorageRepo
from storage.revisions import INITIAL_REVISION_ID
from viewer.api.app import create_app
from viewer.api.reconstruction_records import ReconstructionRecordStore
from viewer.api.routes.reconstruction_records import reconstruction_record_text_file

_MODEL_SOURCE = "from __future__ import annotations\n\n" + ("# authored model\n" * 400)


def _write_reconstruction_record(
    root: Path,
    *,
    record_id: str = "rec_reconstruction_001",
    add_to_workbench: bool = True,
    agent_harness: str = "codex",
    runtime_extra: dict | None = None,
    trace_lines: list[str] | None = None,
) -> None:
    repo = StorageRepo(root)
    revision_dir = repo.layout.record_revision_dir(record_id, INITIAL_REVISION_ID)
    inputs = revision_dir / "inputs"
    inputs.mkdir(parents=True)
    files = {
        "original_source": (
            inputs / "raw_source.glb",
            b"raw-source",
            "surface_geometry_input",
            False,
        ),
        "normalized_source": (
            inputs / "normalized_source.usdc",
            b"normalized-source",
            "canonical_upstream_source_snapshot",
            False,
        ),
        "semantic_image": (
            inputs / "semantic.png",
            b"\x89PNG\r\n\x1a\nsemantic",
            "semantic_subject_evidence",
            True,
        ),
        "axial": (
            inputs / "axial.txt",
            b"AXIAL\nGRID 1 1 1\n",
            "metric_surface_evidence",
            True,
        ),
        "mesh_preprocess_audit_0008": (
            inputs
            / "preprocessing"
            / "audit"
            / "mesh_preprocess"
            / "source"
            / "preview"
            / "orientation_preview.png",
            b"\x89PNG\r\n\x1a\norientation-preview",
            "mesh_preprocess_audit_artifact",
            False,
        ),
        "subject_isolation_audit_0012": (
            inputs
            / "preprocessing"
            / "audit"
            / "subject_isolation"
            / "evidence"
            / "component_id"
            / "azimuth_135.png",
            b"\x89PNG\r\n\x1a\ncomponent-id-135",
            "subject_isolation_audit_artifact",
            False,
        ),
        "source_overlay_urdf": (
            inputs / "source" / "overlay" / "carrier" / "model.urdf",
            b'<robot name="source"/>',
            "viewer_source_overlay_carrier",
            False,
        ),
        "source_overlay_glb": (
            inputs / "source" / "overlay" / "carrier" / "assets" / "source.glb",
            b"source-overlay",
            "viewer_source_overlay_geometry",
            False,
        ),
    }
    manifest_files = []
    for key, (path, content, role, author_input) in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest_files.append(
            {
                "key": key,
                "path": path.relative_to(revision_dir).as_posix(),
                "role": role,
                "author_input": author_input,
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        )
    input_manifest = {
        "schema": "reconstruction_input_manifest",
        "schema_version": 2,
        "pipeline": "reconstruction",
        "path_base": "revision",
        "files": manifest_files,
        "channels": {
            "normalized_source": {"same_frame_as_candidate": True},
            "source_overlay": {
                "base_path": "inputs/source/overlay/carrier",
                "urdf_key": "source_overlay_urdf",
                "glb_key": "source_overlay_glb",
                "same_frame_as_candidate": True,
                "viewer_only": True,
            },
        },
    }
    repo.write_json(inputs / "manifest.json", input_manifest)
    repo.write_json(
        revision_dir / "reconstruction.json",
        {
            "schema_version": 1,
            "pipeline": "reconstruction",
            "status": "reconstruction_ready",
            "input": {
                "manifest_path": "inputs/manifest.json",
                "request_path": "request.txt",
            },
        },
    )
    repo.write_text(revision_dir / "model.py", _MODEL_SOURCE)
    repo.write_text(revision_dir / "request.txt", "Preserve the declared proportions.\n")
    repo.write_json(
        revision_dir / "provenance.json",
        {
            "pipeline": "reconstruction",
            "runtime": {
                "provider": "nvidia" if agent_harness == "provider" else "codex-cli",
                "model_id": "test-model",
                "reasoning_effort": "high",
                **(runtime_extra or {}),
            },
            "run_summary": {"final_status": "reconstruction_ready"},
        },
    )
    if trace_lines is not None:
        traces = revision_dir / "traces"
        traces.mkdir(parents=True, exist_ok=True)
        (traces / "provider_events.jsonl").write_text("".join(trace_lines), encoding="utf-8")
    repo.write_json(
        repo.layout.record_metadata_path(record_id),
        {
            "schema_version": 3,
            "record_id": record_id,
            "pipeline": "reconstruction",
            "status": "reconstruction_ready",
            "input_kind": "mesh_image_text",
            "rating": 4,
            "created_at": "2026-08-21T08:00:00Z",
            "updated_at": "2026-08-21T08:00:00Z",
            "provider": "nvidia" if agent_harness == "provider" else "codex-cli",
            "model_id": "test-model",
            "reasoning_effort": "high",
            "agent_harness": agent_harness,
            "active_revision_id": INITIAL_REVISION_ID,
            "display": {
                "title": "Backyard playset",
                "prompt_preview": "Reconstruct the declared primary subject.",
            },
            "artifacts": {
                "request_txt": f"revisions/{INITIAL_REVISION_ID}/request.txt",
                "model_py": f"revisions/{INITIAL_REVISION_ID}/model.py",
                "inputs_dir": f"revisions/{INITIAL_REVISION_ID}/inputs",
                "provenance_json": f"revisions/{INITIAL_REVISION_ID}/provenance.json",
                "reconstruction_json": f"revisions/{INITIAL_REVISION_ID}/reconstruction.json",
            },
            "creator": {
                "agent": agent_harness,
                "agent_harness": agent_harness,
                "provider": "nvidia" if agent_harness == "provider" else "codex-cli",
            },
        },
    )
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", '<robot name="candidate"/>')
    repo.write_json(
        materialization / "compile_report.json",
        {
            "status": "success",
            "artifacts": {"model.urdf": "unused", "private.txt": "untrusted"},
        },
    )
    (materialization / "private.txt").write_text("not declared", encoding="utf-8")
    if add_to_workbench:
        CollectionStore(repo).append_workbench_entry(
            record_id=record_id,
            added_at="2026-08-21T08:00:01Z",
        )


def test_reconstruction_store_projects_manifest_declared_inputs(tmp_path: Path) -> None:
    _write_reconstruction_record(tmp_path)
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    summaries = store.list_records()
    assert [item.record_id for item in summaries] == ["rec_reconstruction_001"]
    assert summaries[0].input_count == 3
    assert summaries[0].source_available is True
    assert summaries[0].candidate_available is True
    # The list projection never runs the full canonical contract: with no prior validation it
    # reports the neutral "not_validated" placeholder rather than guessing a contract status.
    assert summaries[0].materialization_status == "not_validated"
    assert summaries[0].materialization_displayable is True
    assert summaries[0].materialization_validated is False
    assert summaries[0].materialization_issues == []
    assert summaries[0].has_compile_report is True
    assert summaries[0].input_kind == "mesh_image_text"
    assert summaries[0].rating == 4
    assert summaries[0].provider == "codex-cli"
    assert summaries[0].model_id == "test-model"
    assert summaries[0].agent_harness == "codex"
    assert summaries[0].reasoning_effort == "high"
    assert summaries[0].thinking_level == "high"
    assert summaries[0].revision_count == 1
    assert summaries[0].viewer_asset_updated_at is not None
    assert summaries[0].has_provenance is True
    assert summaries[0].has_cost is False
    assert summaries[0].turn_count is None
    assert summaries[0].input_tokens is None
    assert summaries[0].output_tokens is None
    assert summaries[0].total_cost_usd is None

    detail = store.detail("rec_reconstruction_001")
    assert detail is not None
    assert detail.evidence_manifest is not None
    assert detail.evidence_manifest["schema"] == "reconstruction_input_manifest"
    assert detail.source.file_url == (
        "/api/reconstruction-records/rec_reconstruction_001/files/inputs/normalized_source.usdc"
    )
    assert detail.source.base_file_url == (
        "/api/reconstruction-records/rec_reconstruction_001/files/inputs/source/overlay/carrier"
    )
    assert [item.metadata["key"] for item in detail.inputs] == [
        "original_source",
        "semantic_image",
        "axial",
    ]
    assert "mesh_preprocess_audit_0008" not in {item.metadata["key"] for item in detail.inputs}
    assert "subject_isolation_audit_0012" not in {item.metadata["key"] for item in detail.inputs}
    assert detail.candidate.base_file_url == (
        "/api/reconstruction-records/rec_reconstruction_001/materialized"
    )
    assert detail.summary.materialization_status == "contract_invalid"
    assert detail.summary.materialization_validated is True
    assert detail.request_text == "Preserve the declared proportions.\n"

    # A full check (via `detail`) warms the validation cache, so the next list call reuses the
    # real classification instead of the cheap hint, without re-running the contract itself.
    cached_summaries = store.list_records()
    assert cached_summaries[0].materialization_status == "contract_invalid"
    assert cached_summaries[0].materialization_validated is True
    assert cached_summaries[0].materialization_issues


def test_reconstruction_store_requires_workbench_and_pipeline_binding(tmp_path: Path) -> None:
    _write_reconstruction_record(tmp_path, add_to_workbench=False)
    repo = StorageRepo(tmp_path)
    repo.write_json(
        repo.layout.record_metadata_path("rec_generation_001"),
        {"record_id": "rec_generation_001", "pipeline": "generation"},
    )
    CollectionStore(repo).append_workbench_entry(
        record_id="rec_generation_001",
        added_at="2026-08-21T08:00:01Z",
    )

    store = create_app(repo_root=tmp_path).state.reconstruction_record_store
    assert store.list_records() == []
    assert store.detail("rec_generation_001") is None


def test_reconstruction_file_access_is_manifest_and_compile_report_allowlisted(
    tmp_path: Path,
) -> None:
    _write_reconstruction_record(tmp_path)
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    declared = store.declared_file(
        "rec_reconstruction_001",
        "inputs/semantic.png",
    )
    assert declared.name == "semantic.png"
    assert store.materialized_file("rec_reconstruction_001", "model.urdf").is_file()
    assert store.text_file("rec_reconstruction_001", "model.py").name == "model.py"
    assert store.text_file("rec_reconstruction_001", "request.txt").name == "request.txt"
    assert store.text_file("rec_reconstruction_001", "provenance.json").name == ("provenance.json")
    assert store.text_file("rec_reconstruction_001", "reconstruction.json").name == (
        "reconstruction.json"
    )
    assert store.text_file("rec_reconstruction_001", "compile_report.json").name == (
        "compile_report.json"
    )
    assert store.text_file("rec_reconstruction_001", "model.urdf").name == "model.urdf"
    assert store.text_file("rec_reconstruction_001", "inputs/axial.txt") == (
        declared.parent / "axial.txt"
    )
    orientation_preview = store.declared_file(
        "rec_reconstruction_001",
        "inputs/preprocessing/audit/mesh_preprocess/source/preview/orientation_preview.png",
    )
    assert orientation_preview.name == "orientation_preview.png"
    component_id = store.declared_file(
        "rec_reconstruction_001",
        ("inputs/preprocessing/audit/subject_isolation/evidence/component_id/azimuth_135.png"),
    )
    assert component_id.name == "azimuth_135.png"
    source_overlay = store.declared_file(
        "rec_reconstruction_001",
        "inputs/source/overlay/carrier/assets/source.glb",
    )
    assert source_overlay.name == "source.glb"

    with pytest.raises(HTTPException, match="not declared"):
        store.declared_file("rec_reconstruction_001", "inputs/hidden.txt")
    with pytest.raises(HTTPException, match="not found"):
        store.materialized_file("rec_reconstruction_001", "private.txt")
    with pytest.raises(HTTPException, match="Invalid file path"):
        store.declared_file("rec_reconstruction_001", "../record.json")


def test_reconstruction_text_endpoint_is_bounded_and_supports_full_reads(
    tmp_path: Path,
) -> None:
    _write_reconstruction_record(tmp_path)
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store
    record_id = "rec_reconstruction_001"

    preview = asyncio.run(
        reconstruction_record_text_file(
            record_id,
            "model.py",
            store,
            preview_bytes=4096,
            full=False,
        )
    )
    assert preview.record_id == record_id
    assert preview.file_path == "model.py"
    assert preview.truncated is True
    assert preview.byte_count == len(_MODEL_SOURCE.encode("utf-8"))
    assert preview.preview_byte_limit == 4096
    assert len(preview.content.encode("utf-8")) == 4096

    full = asyncio.run(
        reconstruction_record_text_file(
            record_id,
            "model.py",
            store,
            preview_bytes=4096,
            full=True,
        )
    )
    assert full.content == _MODEL_SOURCE
    assert full.truncated is False
    assert full.preview_byte_limit is None

    with pytest.raises(HTTPException, match="only supported for text files") as non_text:
        asyncio.run(
            reconstruction_record_text_file(
                record_id,
                "inputs/semantic.png",
                store,
                preview_bytes=4096,
                full=False,
            )
        )
    assert non_text.value.status_code == 400


def test_reconstruction_text_access_rejects_undeclared_traversal_and_generation_records(
    tmp_path: Path,
) -> None:
    _write_reconstruction_record(tmp_path)
    repo = StorageRepo(tmp_path)
    record_id = "rec_reconstruction_001"
    revision_dir = repo.layout.record_revision_dir(record_id, INITIAL_REVISION_ID)
    repo.write_text(revision_dir / "hidden.txt", "private")
    repo.write_text(revision_dir / "inputs" / "hidden.txt", "undeclared input")
    repo.write_json(
        repo.layout.record_metadata_path("rec_generation_001"),
        {
            "record_id": "rec_generation_001",
            "pipeline": "generation",
            "active_revision_id": INITIAL_REVISION_ID,
        },
    )
    repo.write_text(
        repo.layout.record_revision_dir("rec_generation_001", INITIAL_REVISION_ID) / "model.py",
        "generation source",
    )
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    with pytest.raises(HTTPException, match="Invalid file path") as traversal:
        store.text_file(record_id, "../record.json")
    assert traversal.value.status_code == 400
    for path in ("record.json", "hidden.txt", "inputs/hidden.txt"):
        with pytest.raises(HTTPException) as undeclared:
            store.text_file(record_id, path)
        assert undeclared.value.status_code == 404
    with pytest.raises(HTTPException, match="Reconstruction record not found") as generation:
        store.text_file("rec_generation_001", "model.py")
    assert generation.value.status_code == 404


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [("failed_report", "compile_failed"), ("missing_urdf", "missing")],
)
def test_reconstruction_candidate_blocks_non_displayable_materializations(
    tmp_path: Path,
    mutation: str,
    expected_status: str,
) -> None:
    _write_reconstruction_record(tmp_path)
    repo = StorageRepo(tmp_path)
    record_id = "rec_reconstruction_001"
    materialization = repo.layout.record_materialization_dir(record_id)
    if mutation == "failed_report":
        repo.write_json(materialization / "compile_report.json", {"status": "failed"})
    else:
        (materialization / "model.urdf").unlink()

    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    # `compile_failed`/`missing` are cheap, hash-free objective signals, so the fast list
    # projection reports them correctly even with no prior full validation.
    fast_summaries = store.list_records()
    assert fast_summaries[0].materialization_status == expected_status
    assert fast_summaries[0].materialization_displayable is False
    assert fast_summaries[0].materialization_validated is False

    detail = store.detail(record_id)

    assert detail is not None
    assert detail.summary.materialization_status == expected_status
    assert detail.summary.materialization_displayable is False
    assert detail.summary.materialization_issues
    assert detail.summary.candidate_available is False
    assert detail.candidate.available is False
    assert detail.candidate.base_file_url is None
    assert store.text_file(record_id, "model.py").is_file()
    assert store.text_file(record_id, "request.txt").is_file()
    assert store.text_file(record_id, "provenance.json").is_file()
    assert store.text_file(record_id, "compile_report.json").is_file()
    diagnostic = asyncio.run(
        reconstruction_record_text_file(
            record_id,
            "compile_report.json",
            store,
            preview_bytes=4096,
            full=False,
        )
    )
    assert '"status"' in diagnostic.content
    if mutation == "failed_report":
        assert '"failed"' in diagnostic.content
    with pytest.raises(HTTPException, match="not found"):
        store.materialized_file(record_id, "model.urdf")
    with pytest.raises(HTTPException, match="not found"):
        store.text_file(record_id, "model.urdf")


def test_reconstruction_available_materialization_uses_report_artifact_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise Viewer's available branch; persistence tests own full canonical validation."""

    _write_reconstruction_record(tmp_path)
    repo = StorageRepo(tmp_path)
    record_id = "rec_reconstruction_001"
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_json(
        materialization / "compile_report.json",
        {
            "schema_version": 2,
            "status": "success",
            "artifacts": {"model.urdf": "unused"},
        },
    )
    monkeypatch.setattr(
        ReconstructionRecordStore,
        "_materialization_display",
        lambda self, resolved_record_id, record: MaterializationDisplayResult(
            status="available",
            issues=(),
        ),
    )
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    detail = store.detail(record_id)
    assert detail is not None
    assert detail.summary.materialization_status == "available"
    assert detail.summary.materialization_displayable is True
    assert detail.candidate.available is True
    assert store.text_file(record_id, "model.urdf").is_file()
    with pytest.raises(HTTPException, match="not found"):
        store.materialized_file(record_id, "private.txt")


def test_reconstruction_store_reports_the_provider_arm(tmp_path: Path) -> None:
    _write_reconstruction_record(tmp_path, agent_harness="provider")
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    summaries = store.list_records()
    assert summaries[0].agent_harness == "provider"
    assert summaries[0].provider == "nvidia"


def _axial_payload(recall: float, *, revision: int, call: int) -> dict:
    return {
        "ok": True,
        "revision": revision,
        "axial_call": call,
        "output": {
            "text": "...",
            "result": {
                "evidence_comparison": {
                    "recall": recall,
                    "input_cell_count": 1000,
                    "matched_cell_count": int(recall * 1000),
                    "missing_cell_count": 1000 - int(recall * 1000),
                    "candidate_only_evidence": {"cell_count": 120},
                },
                "provider_audit": {
                    "exterior_excess": {"excess_cells": 7, "fraction_of_candidate": 0.012},
                    "surface_fit": {
                        "candidate_to_source_p50_m": 0.0042,
                        "candidate_to_source_p90_m": 0.0151,
                        "faces": [
                            {"face": "-Y front", "reading": "face follows the source surface"},
                            {"face": "+Y back", "reading": "face plane or scale off"},
                        ],
                    },
                    "opening_placement": {
                        "candidate_openings": [{}, {}],
                        "unmodelled_source_openings": [{}],
                    },
                },
            },
        },
    }


def test_reconstruction_summary_reads_the_persisted_axial_diagnostic(tmp_path: Path) -> None:
    from usdcraft.reconstruction.axial_summary import summarize_axial_payloads

    persisted = summarize_axial_payloads(
        [_axial_payload(0.61, revision=3, call=1), _axial_payload(0.74, revision=5, call=2)]
    )
    _write_reconstruction_record(
        tmp_path, agent_harness="provider", runtime_extra={"axial_diagnostic": persisted}
    )
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    diagnostic = store.list_records()[0].axial_diagnostic
    assert diagnostic is not None
    assert diagnostic.calls == 2 and diagnostic.revision == 5
    assert diagnostic.first_recall == 0.61 and diagnostic.recall == 0.74
    assert diagnostic.exterior_excess_fraction == 0.012
    assert diagnostic.surface_fit_p50_mm == 4.2 and diagnostic.surface_faces_off_plane == 1
    assert diagnostic.candidate_openings == 2 and diagnostic.unmodelled_source_openings == 1


def test_reconstruction_rating_and_delete_routes_are_bound_to_reconstruction_records(
    tmp_path: Path,
) -> None:
    from storage.record_mutations import RecordMutationStore
    from viewer.api.routes.reconstruction_records import (
        delete_reconstruction_record,
        update_reconstruction_record_rating,
    )
    from viewer.api.routes.records import update_record_rating
    from viewer.api.schemas import RecordDeleteRequest, RecordRatingRequest

    _write_reconstruction_record(tmp_path, agent_harness="provider")
    app = create_app(repo_root=tmp_path)
    repo = StorageRepo(tmp_path)
    store = app.state.reconstruction_record_store
    mutations = RecordMutationStore(repo)
    record_id = "rec_reconstruction_001"

    summary = asyncio.run(
        update_reconstruction_record_rating(
            record_id, RecordRatingRequest(rating=2), store, mutations
        )
    )
    assert summary.record_id == record_id and summary.rating == 2
    assert summary.agent_harness == "provider"
    assert repo.read_json(repo.layout.record_metadata_path(record_id))["rating"] == 2

    # The Generation route refuses the Reconstruction record and leaves it untouched.
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            update_record_rating(
                record_id, RecordRatingRequest(rating=5), app.state.viewer_store, mutations
            )
        )
    assert excinfo.value.status_code == 404
    assert repo.read_json(repo.layout.record_metadata_path(record_id))["rating"] == 2

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            delete_reconstruction_record(
                record_id, RecordDeleteRequest(confirm_record_id="rec_other"), mutations
            )
        )
    assert excinfo.value.status_code == 409
    assert repo.layout.record_dir(record_id).is_dir()

    deleted = asyncio.run(
        delete_reconstruction_record(
            record_id, RecordDeleteRequest(confirm_record_id=record_id), mutations
        )
    )
    assert deleted.record_id == record_id and deleted.materialization_deleted is True
    assert not repo.layout.record_dir(record_id).exists()
    assert store.list_records() == []


def test_reconstruction_store_validate_runs_and_caches_the_full_contract(
    tmp_path: Path,
) -> None:
    """`ReconstructionRecordStore.validate` is the "I suspect a mesh is wrong" action.

    It is deliberately not exposed as a Viewer HTTP route: rating update/clear and explicitly
    confirmed deletion remain the only two Workbench mutations
    (`test_api_exposes_only_narrow_workbench_mutations`). The equivalent user entry point is
    the `usdcraft reconstruction-validate` CLI command, exercised in `tests/viewer/test_cli.py`.
    """

    _write_reconstruction_record(tmp_path)
    record_id = "rec_reconstruction_001"
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    # Before any full check, the list projection is the neutral placeholder.
    assert store.list_records()[0].materialization_status == "not_validated"

    validated = store.validate(record_id)
    assert validated is not None
    assert validated.materialization_status == "contract_invalid"
    assert validated.materialization_validated is True
    assert validated.materialization_issues

    # The explicit validate call warms the same cache the list projection reads.
    assert store.list_records()[0].materialization_status == "contract_invalid"
    assert store.list_records()[0].materialization_validated is True

    assert store.validate("rec_missing") is None


def test_reconstruction_validation_cache_does_not_survive_a_materialization_change(
    tmp_path: Path,
) -> None:
    _write_reconstruction_record(tmp_path)
    repo = StorageRepo(tmp_path)
    record_id = "rec_reconstruction_001"
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    first = store.validate(record_id)
    assert first is not None
    assert first.materialization_status == "contract_invalid"
    assert store.list_records()[0].materialization_validated is True

    # An explicit compile failure changes the materialization on disk; the stat-only signature
    # this cache keys on must change too, so the stale cached result cannot survive.
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_json(materialization / "compile_report.json", {"status": "failed"})

    lazy = store.list_records()[0]
    assert lazy.materialization_status == "compile_failed"
    # The cheap hint alone is conclusive for `compile_failed`, so this is trustworthy without
    # re-running the full contract, but it is honestly reported as not re-validated.
    assert lazy.materialization_validated is False


def test_reconstruction_list_survives_a_missing_reconstruction_manifest(tmp_path: Path) -> None:
    _write_reconstruction_record(tmp_path, agent_harness="provider")
    for manifest in tmp_path.rglob("reconstruction.json"):
        manifest.unlink()
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    assert [item.record_id for item in store.list_records()] == ["rec_reconstruction_001"]


def test_reconstruction_summary_reports_scale(tmp_path: Path) -> None:
    _write_reconstruction_record(
        tmp_path,
        agent_harness="provider",
        runtime_extra={
            "scale": {
                "meters_per_unit": 0.2506,
                "source": "estimated_from_semantic_image",
                "estimate": {"category": "microwave oven", "longest_dimension_m": 0.5},
            },
        },
    )
    _write_reconstruction_record(
        tmp_path,
        record_id="rec_reconstruction_declared",
        agent_harness="provider",
        runtime_extra={
            "scale": {"meters_per_unit": 1.0, "source": "explicit"},
        },
    )
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    summaries = {item.record_id: item for item in store.list_records()}
    estimated = summaries["rec_reconstruction_001"]
    assert estimated.scale_source == "estimated_from_semantic_image"
    assert estimated.scale_meters_per_unit == 0.2506
    assert estimated.scale_estimate_category == "microwave oven"
    assert estimated.scale_estimate_longest_m == 0.5
    declared = summaries["rec_reconstruction_declared"]
    assert declared.scale_source == "declared" and declared.scale_estimate_category is None


def test_reconstruction_summary_reports_batch_membership(tmp_path: Path) -> None:
    _write_reconstruction_record(
        tmp_path,
        agent_harness="provider",
        runtime_extra={
            "batch": {
                "batch_id": "round2_20260905T140000Z",
                "csv": "round2.csv",
                "row_id": "microwave_oven",
                "row_index": 2,
                "total": 6,
            }
        },
    )
    _write_reconstruction_record(
        tmp_path, record_id="rec_reconstruction_single", agent_harness="provider"
    )
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    summaries = {item.record_id: item for item in store.list_records()}
    assert summaries["rec_reconstruction_001"].batch_id == "round2_20260905T140000Z"
    assert summaries["rec_reconstruction_001"].batch_row_id == "microwave_oven"
    assert summaries["rec_reconstruction_single"].batch_id is None


def test_author_timeline_and_axial_geometry_are_reduced_from_the_provider_trace(
    tmp_path: Path,
) -> None:
    payload = _axial_payload(0.66, revision=3, call=1)
    payload["output"]["result"]["grid"] = {
        "bounds_min_xyz_m": [-1.0, -0.5, 0.0],
        "bounds_max_xyz_m": [1.0, 0.5, 0.8],
    }
    payload["output"]["result"]["provider_audit"]["stack_pitch_m"] = 0.05
    payload["output"]["result"]["provider_audit"]["exterior_excess"]["worst_slices"] = [
        {
            "stack_slice": 4,
            "side": "+X",
            "excess_cells": 9,
            "bounds_min_xyz_m": [0.9, -0.2, 0.2],
            "bounds_max_xyz_m": [1.0, 0.2, 0.25],
        }
    ]
    events = [
        json.dumps(
            {"record_type": "provider.response.completed", "turn": 1, "elapsed_seconds": 12.5}
        )
        + "\n",
        json.dumps(
            {
                "record_type": "provider.tool.completed",
                "turn": 1,
                "tool_name": "read_file",
                "ok": True,
                "revision": 0,
                "elapsed_seconds": 0.01,
            }
        )
        + "\n",
        json.dumps(
            {"record_type": "provider.response.completed", "turn": 2, "elapsed_seconds": 40.0}
        )
        + "\n",
        json.dumps(
            {
                "record_type": "provider.tool.completed",
                "turn": 2,
                "tool_name": "check_axial",
                "ok": True,
                "revision": 3,
                "elapsed_seconds": 2.0,
                "output": payload,
            }
        )
        + "\n",
    ]
    _write_reconstruction_record(tmp_path, agent_harness="provider", trace_lines=events)
    _write_reconstruction_record(
        tmp_path, record_id="rec_reconstruction_codex", agent_harness="codex"
    )
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    timeline = store.author_timeline("rec_reconstruction_001")
    assert timeline is not None and timeline.source == "provider_events"
    assert [(e.turn, e.tool, e.ok) for e in timeline.events] == [
        (1, "read_file", True),
        (2, "check_axial", True),
    ]
    assert timeline.events[0].response_elapsed_seconds == 12.5
    assert timeline.events[1].recall == 0.66 and timeline.events[1].revision == 3
    assert store.author_timeline("rec_reconstruction_codex") is None

    geometry = store.axial_geometry("rec_reconstruction_001")
    assert geometry is not None
    assert geometry.grid_bounds_min_xyz_m == [-1.0, -0.5, 0.0]
    assert geometry.stack_pitch_m == 0.05
    assert [s.side for s in geometry.excess_slices] == ["+X"]
    assert geometry.excess_slices[0].bounds_max_xyz_m == [1.0, 0.2, 0.25]
    assert not hasattr(geometry, "controls")
    assert store.axial_geometry("rec_reconstruction_codex") is None


def test_reconstruction_summary_exposes_run_settings(tmp_path: Path) -> None:
    batch = {"batch_id": "batch_001", "row_id": "case__backyard_repeat03"}
    _write_reconstruction_record(
        tmp_path,
        agent_harness="provider",
        runtime_extra={
            "harness": {"image_observation": True, "max_turns": 40},
            "image_observation": "external_inventory",
            "runtime_source": "nvidia-chat-completions-api",
            "api_style": "chat-completions",
            "stage_timings": [
                {"stage": "freeze", "duration_seconds": 0.1, "status": "ok"},
                {
                    "stage": "subject_isolation",
                    "detail": "skipped_retain_all",
                    "duration_seconds": 0.1,
                    "status": "ok",
                },
            ],
            "batch": batch,
        },
    )
    summary = create_app(repo_root=tmp_path).state.reconstruction_record_store.list_records()[0]
    assert summary.runtime_source == "nvidia-chat-completions-api"
    assert summary.api_style == "chat-completions"
    assert summary.image_observation == "external_inventory"
    assert summary.max_turns == 40
    assert summary.mesh_cleanup == "skipped"
    assert (summary.batch_id, summary.batch_row_id) == ("batch_001", "case__backyard_repeat03")
    assert "harness" not in summary.model_dump()
    # Titles are shown as stored: run settings and batch repeats never decorate them.
    assert summary.title == "Backyard playset"


def test_reconstruction_codex_run_settings_carry_no_turn_budget_or_api_style(
    tmp_path: Path,
) -> None:
    _write_reconstruction_record(
        tmp_path,
        agent_harness="codex",
        runtime_extra={
            "harness": {"image_observation": False},
            "image_observation": "disabled",
            "runtime_source": "codex_root_thread_state",
            "stage_timings": [
                {"stage": "subject_isolation", "detail": "agent", "duration_seconds": 9.0}
            ],
        },
    )
    summary = create_app(repo_root=tmp_path).state.reconstruction_record_store.list_records()[0]
    assert summary.agent_harness == "codex"
    assert summary.api_style is None
    assert summary.max_turns is None
    assert summary.image_observation == "disabled"
    assert summary.mesh_cleanup == "run"


def test_provider_api_style_defaults_to_responses_when_not_recorded(tmp_path: Path) -> None:
    _write_reconstruction_record(
        tmp_path,
        agent_harness="provider",
        runtime_extra={"runtime_source": "nvidia-responses-api"},
    )
    summary = create_app(repo_root=tmp_path).state.reconstruction_record_store.list_records()[0]
    assert summary.api_style == "responses"
    assert summary.mesh_cleanup is None
    assert summary.image_observation is None


def _record_guidance(root: Path, guidance: str | None) -> None:
    """Set ``input.guidance`` in the fixture's ``reconstruction.json`` as the Host records it."""

    path = (
        StorageRepo(root).layout.record_revision_dir("rec_reconstruction_001", INITIAL_REVISION_ID)
        / "reconstruction.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["input"]["guidance"] = guidance
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("agent_harness", ["provider", "codex"])
def test_reconstruction_detail_reads_the_recorded_guidance(
    tmp_path: Path, agent_harness: str
) -> None:
    _write_reconstruction_record(tmp_path, agent_harness=agent_harness)
    _record_guidance(tmp_path, "Treat the handles as fixed.\n\nKeep doors shut.")
    detail = create_app(repo_root=tmp_path).state.reconstruction_record_store.detail(
        "rec_reconstruction_001"
    )
    assert detail is not None
    assert detail.guidance_status == "present"
    assert detail.guidance == "Treat the handles as fixed.\n\nKeep doors shut."
    # The fixture's request text is not the Host default.
    assert detail.request_prompt == "custom"


def test_reconstruction_detail_distinguishes_no_guidance_from_an_unrecorded_field(
    tmp_path: Path,
) -> None:
    _write_reconstruction_record(tmp_path, agent_harness="provider")
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store
    detail = store.detail("rec_reconstruction_001")
    assert detail is not None
    assert (detail.guidance_status, detail.guidance) == ("unrecorded", None)

    _record_guidance(tmp_path, None)
    detail = store.detail("rec_reconstruction_001")
    assert detail is not None
    assert (detail.guidance_status, detail.guidance) == ("absent", None)


def test_reconstruction_detail_recognizes_the_default_request(tmp_path: Path) -> None:
    from viewer.api.run_settings import DEFAULT_RECONSTRUCTION_PROMPT

    _write_reconstruction_record(tmp_path)
    request_path = (
        StorageRepo(tmp_path).layout.record_revision_dir(
            "rec_reconstruction_001", INITIAL_REVISION_ID
        )
        / "request.txt"
    )
    request_path.write_text(DEFAULT_RECONSTRUCTION_PROMPT + "\n", encoding="utf-8")
    detail = create_app(repo_root=tmp_path).state.reconstruction_record_store.detail(
        "rec_reconstruction_001"
    )
    assert detail is not None
    assert detail.request_prompt == "default"


def test_viewer_default_request_matches_the_host_default() -> None:
    from usdcraft.reconstruction.default_prompt import DEFAULT_RECONSTRUCTION_PROMPT as HOST
    from viewer.api.run_settings import DEFAULT_RECONSTRUCTION_PROMPT as VIEWER

    assert VIEWER == HOST


def test_reconstruction_listing_reflects_rating_and_materialization_changes(
    tmp_path: Path,
) -> None:
    """A repeated listing is stable, while a rating change or a materialization change is
    visible on the very next listing."""

    _write_reconstruction_record(tmp_path)
    repo = StorageRepo(tmp_path)
    record_id = "rec_reconstruction_001"
    store = create_app(repo_root=tmp_path).state.reconstruction_record_store

    first = store.list_records()[0]
    assert first.rating == 4
    assert store.list_records()[0] == first

    RecordMutationStore(repo).update_rating(record_id, 2, pipeline="reconstruction")
    assert store.list_records()[0].rating == 2

    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_json(materialization / "compile_report.json", {"status": "failed"})
    failed = store.list_records()[0]
    assert failed.materialization_status == "compile_failed"
    assert failed.candidate_available is False


def test_reconstruction_evidence_resolution_matches_per_file_realpath_and_drops_escapes(
    tmp_path: Path,
) -> None:
    """The memoized per-directory resolution used by `_evidence` must agree with plain per-file
    `realpath` containment: real files resolve identically, a symlinked directory or file that
    stays inside the revision resolves to its target, and any symlink escaping the revision is
    dropped from both projections."""

    _write_reconstruction_record(tmp_path)
    repo = StorageRepo(tmp_path)
    record_id = "rec_reconstruction_001"
    revision_dir = repo.layout.record_revision_dir(record_id, INITIAL_REVISION_ID)
    inputs = revision_dir / "inputs"
    carrier = inputs / "source" / "overlay" / "carrier"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside.png").write_bytes(b"\x89PNG\r\n\x1a\noutside")
    (inputs / "escape_file.png").symlink_to(outside / "outside.png")
    (inputs / "escape_dir").symlink_to(outside, target_is_directory=True)
    (inputs / "alias_dir").symlink_to(carrier, target_is_directory=True)
    (carrier / "alias.urdf").symlink_to(carrier / "model.urdf")

    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, path in (
        ("escape_file", "inputs/escape_file.png"),
        ("escape_dir_file", "inputs/escape_dir/outside.png"),
        ("alias_dir_file", "inputs/alias_dir/model.urdf"),
        ("alias_file", "inputs/source/overlay/carrier/alias.urdf"),
        ("missing", "inputs/source/overlay/carrier/missing.png"),
    ):
        manifest["files"].append(
            {"key": key, "path": path, "role": "viewer_source_overlay", "author_input": False}
        )
    repo.write_json(manifest_path, manifest)

    store = create_app(repo_root=tmp_path).state.reconstruction_record_store
    resolved_revision_dir = revision_dir.resolve()
    directory_cache: dict[Path, tuple[Path, Path] | None] = {}
    for item in manifest["files"]:
        plain = store._resolve_evidence_path(item["path"], revision_dir=resolved_revision_dir)
        memoized = store._resolve_evidence_path(
            item["path"], revision_dir=resolved_revision_dir, directory_cache=directory_cache
        )
        assert memoized == plain, item["path"]
    assert (
        store._resolve_evidence_path("inputs/escape_file.png", revision_dir=resolved_revision_dir)
        is None
    )
    assert (
        store._resolve_evidence_path(
            "inputs/escape_dir/outside.png", revision_dir=resolved_revision_dir
        )
        is None
    )

    detail = store.detail(record_id)
    assert detail is not None
    keys = {item.metadata["key"] for item in detail.inputs}
    assert {"escape_file", "escape_dir_file", "missing"}.isdisjoint(keys)
    # In-revision aliases resolve to files already declared, so they are deduplicated, not added.
    assert {"alias_dir_file", "alias_file"}.isdisjoint(keys)
    assert store.list_records()[0].input_count == 3
