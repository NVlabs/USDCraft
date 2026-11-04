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
from typing import Any

import pytest
from fastapi import HTTPException
from pxr import Usd, UsdGeom
from pydantic import ValidationError

from storage.artifact_contract import (
    REQUIRED_BACKEND_CHECKS,
    REQUIRED_COMPILE_CHECKS,
    STANDARD_RENDER_ARTIFACTS,
    exporter_asset_structure_contract,
    physics_output_contract,
)
from storage.collections import CollectionStore
from storage.record_mutations import RecordMutationStore
from storage.repo import StorageRepo
from storage.revisions import INITIAL_REVISION_ID, revision_artifacts_payload
from viewer.api.app import create_app
from viewer.api.file_resolver import ViewerFileResolver
from viewer.api.routes.records import (
    delete_record,
    record_detail,
    record_history,
    record_input_image,
    record_summary,
    update_record_rating,
)
from viewer.api.schemas import RecordDeleteRequest, RecordRatingRequest


def _asgi_json_request(
    app: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    host: str = "127.0.0.1:8770",
) -> tuple[int, Any]:
    """Exercise FastAPI request parsing without Starlette's optional httpx2 client."""

    async def _request() -> tuple[int, dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [
                (b"host", host.encode("ascii")),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8770),
        }
        messages: list[dict[str, Any]] = []
        request_delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        await app(scope, receive, send)
        start = next(message for message in messages if message["type"] == "http.response.start")
        response_body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        try:
            return int(start["status"]), json.loads(response_body)
        except ValueError:
            return int(start["status"]), response_body.decode("utf-8", "replace")

    return asyncio.run(_request())


def _write_generation_record(repo: StorageRepo, record_id: str = "rec_generation_001") -> None:
    artifacts = revision_artifacts_payload(
        revision_id=INITIAL_REVISION_ID,
        has_cost_file=False,
    )
    artifacts["generation_json"] = f"revisions/{INITIAL_REVISION_ID}/generation.json"
    repo.write_json(
        repo.layout.record_metadata_path(record_id),
        {
            "schema_version": 3,
            "record_id": record_id,
            "created_at": "2026-07-12T08:00:00Z",
            "updated_at": "2026-07-12T08:00:00Z",
            "rating": None,
            "kind": "structured_3d_asset",
            "pipeline": "generation",
            "status": "generation_ready",
            "input_kind": "text_image",
            "prompt_kind": "single_prompt",
            "source": {"run_id": "run_generation_001"},
            "agent_harness": "codex",
            "provider": "codex-cli",
            "model_id": "gpt-5.5",
            "reasoning_effort": "ultra",
            "runtime_source": "codex_root_thread_state",
            "selection_source": "inherited_main_codex_runtime",
            "display": {
                "title": "Articulated desk lamp",
                "prompt_preview": "Create a functional articulated desk lamp.",
            },
            "artifacts": artifacts,
            "collections": ["workbench"],
            "active_revision_id": INITIAL_REVISION_ID,
            "creator": {
                "mode": "external_agent",
                "agent": "codex",
                "agent_harness": "codex",
                "provider": "codex-cli",
                "trace_available": True,
            },
        },
    )
    revision_dir = repo.layout.record_revision_dir(record_id, INITIAL_REVISION_ID)
    repo.write_text(revision_dir / "prompt.txt", "Create a functional articulated desk lamp.\n")
    repo.write_text(revision_dir / "model.py", "from __future__ import annotations\n")
    repo.write_json(
        revision_dir / "provenance.json",
        {
            "schema_version": 3,
            "record_id": record_id,
            "pipeline": "generation",
            "runtime": {
                "agent": "codex",
                "agent_harness": "codex",
                "provider": "codex-cli",
                "model_id": "gpt-5.5",
                "reasoning_effort": "ultra",
                "runtime_source": "codex_root_thread_state",
                "selection_source": "inherited_main_codex_runtime",
            },
            "run_summary": {"final_status": "generation_ready"},
        },
    )
    input_dir = repo.layout.record_revision_inputs_dir(record_id, INITIAL_REVISION_ID)
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "reference.png").write_bytes(b"generation-inspiration")
    repo.write_json(
        revision_dir / "generation.json",
        {
            "schema_version": 3,
            "pipeline": "generation",
            "status": "generation_ready",
            "record_id": record_id,
            "revision_id": INITIAL_REVISION_ID,
            "input": {
                "kind": "text_image",
                "prompt_sha256": hashlib.sha256(
                    b"Create a functional articulated desk lamp."
                ).hexdigest(),
                "image_role": "inspiration",
                "image_path": "inputs/reference.png",
                "image_sha256": hashlib.sha256(b"generation-inspiration").hexdigest(),
            },
            "creator": {
                "agent": "codex",
                "agent_harness": "codex",
                "provider": "codex-cli",
                "model_id": "gpt-5.5",
                "reasoning_effort": "ultra",
                "runtime_source": "codex_root_thread_state",
                "selection_source": "inherited_main_codex_runtime",
            },
            "model_sha256": hashlib.sha256(b"from __future__ import annotations\n").hexdigest(),
            "validation_profile": "generation",
            "outputs": physics_output_contract(),
            "compile_report": "../../../../cache/record_materialization/"
            + record_id
            + "/compile_report.json",
        },
    )
    traces_dir = repo.layout.record_revision_traces_dir(record_id, INITIAL_REVISION_ID)
    traces_dir.mkdir(parents=True, exist_ok=True)
    repo.write_text(traces_dir / "codex_events.jsonl", '{"type":"complete"}\n')
    CollectionStore(repo).append_workbench_entry(
        record_id=record_id,
        added_at="2026-07-12T08:01:00Z",
    )


def _write_current_materialization(repo: StorageRepo, record_id: str) -> None:
    revision_dir = repo.layout.record_revision_dir(record_id, INITIAL_REVISION_ID)
    materialization = repo.layout.record_materialization_dir(record_id)
    stage_paths = tuple(
        materialization / "isaac" / name
        for name in (
            "asset.usda",
            "model.usdc",
            "model_newton.usdc",
            "mujoco.usda",
            "physics.usdc",
            "physx.usda",
        )
    )
    for path in stage_paths:
        if path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(path.as_posix())
        root_prim = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
        stage.SetDefaultPrim(root_prim)
        stage.GetRootLayer().Save()
    render_paths = tuple(materialization / path for path in STANDARD_RENDER_ARTIFACTS)
    for path in render_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"render:{path.stem}".encode())
    asset_root = materialization / "assets"
    asset_paths = (
        tuple(path for path in asset_root.rglob("*") if path.is_file())
        if asset_root.is_dir()
        else ()
    )

    prompt = (revision_dir / "prompt.txt").read_text(encoding="utf-8").rstrip("\n")
    model = revision_dir / "model.py"
    image = revision_dir / "inputs" / "reference.png"
    repo.write_json(
        revision_dir / "generation.json",
        {
            "schema_version": 3,
            "pipeline": "generation",
            "status": "generation_ready",
            "record_id": record_id,
            "revision_id": INITIAL_REVISION_ID,
            "input": {
                "kind": "text_image",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "image_role": "inspiration",
                "image_path": "inputs/reference.png",
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            },
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "validation_profile": "generation",
            "outputs": physics_output_contract(),
            "compile_report": "../../../../cache/record_materialization/"
            + record_id
            + "/compile_report.json",
        },
    )
    artifact_paths = {
        "model.urdf": materialization / "model.urdf",
        "model.usdz": materialization / "model.usdz",
        **{path.relative_to(materialization).as_posix(): path for path in stage_paths},
        **{path.relative_to(materialization).as_posix(): path for path in render_paths},
        **{path.relative_to(materialization).as_posix(): path for path in asset_paths},
    }
    repo.write_json(
        materialization / "compile_report.json",
        {
            "schema_version": 2,
            "status": "success",
            "validation_profile": "generation",
            "asset_structure": physics_output_contract(),
            "record_id": record_id,
            "model_path": f"data/records/{record_id}/revisions/{INITIAL_REVISION_ID}/model.py",
            "checks_run": sorted(REQUIRED_COMPILE_CHECKS | {"generation_render_set"}),
            "artifacts": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in artifact_paths.items()
            },
            "usd_dependency_closure": {
                path.relative_to(materialization).as_posix(): [
                    path.relative_to(materialization).as_posix()
                ]
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
        },
    )


def test_generation_record_list_and_inspiration_image(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)

    app = create_app(repo_root=tmp_path)
    assert app.title == "USDCraft Viewer API"
    bootstrap = app.state.viewer_store.bootstrap()
    record = bootstrap.workbench_entries[0].record
    assert record is not None
    assert record.record_id == "rec_generation_001"
    assert record.pipeline == "generation"
    assert record.status == "generation_ready"
    assert record.input_kind == "text_image"
    assert record.title == "Articulated desk lamp"
    assert record.agent_harness == "codex"
    assert record.model_id == "gpt-5.5"
    assert record.thinking_level == "ultra"
    assert record.has_traces is True
    # The bootstrap/list projection is the fast, cache-or-cheap-hint path: a missing
    # `model.urdf` is one of the two objective hints `cheap_display_hint` resolves without the
    # full contract, so the status is trustworthy, but the row is honestly reported as not
    # re-validated and carries no full issue list.
    assert record.materialization_status == "missing"
    assert record.materialization_displayable is False
    assert record.materialization_validated is False
    assert record.materialization_issues == []

    summary = asyncio.run(record_summary("rec_generation_001", app.state.viewer_store))
    detail = asyncio.run(record_detail("rec_generation_001", app.state.viewer_store))
    history = asyncio.run(record_history("rec_generation_001", app.state.viewer_store))
    # `record_summary` always resolves a real check (never the list projection's fast path).
    assert summary.materialization_status == "missing"
    assert summary.materialization_displayable is False
    assert summary.materialization_validated is True
    assert [issue.model_dump() for issue in summary.materialization_issues] == [
        {"path": "compile_report.json", "message": "missing JSON object"},
        {"path": "model.urdf", "message": "missing required Viewer model file"},
    ]
    assert detail.record["pipeline"] == "generation"
    assert detail.provenance is not None
    assert history.active_revision_id == INITIAL_REVISION_ID
    assert [revision.revision_id for revision in history.revisions] == [INITIAL_REVISION_ID]

    image = asyncio.run(record_input_image("rec_generation_001", app.state.viewer_store))
    assert image.path == "inputs/reference.png"
    _, image_path = app.state.file_resolver.resolve_record_target(
        "rec_generation_001",
        "inputs/reference.png",
    )
    assert image_path.read_bytes() == b"generation-inspiration"


@pytest.mark.parametrize("endpoint", ["nvidia", "openai"])
def test_viewer_projects_provider_identity_without_fabricating_cost(
    tmp_path: Path, endpoint: str
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    record_id = "rec_provider_001"
    _write_generation_record(repo, record_id)
    record_path = repo.layout.record_metadata_path(record_id)
    record = repo.read_json(record_path)
    record["agent_harness"] = "provider"
    record["provider"] = endpoint
    record["creator"].update(
        {"agent": "provider", "agent_harness": "provider", "provider": endpoint}
    )
    repo.write_json(record_path, record)
    provenance_path = repo.layout.record_revision_provenance_path(
        record_id,
        INITIAL_REVISION_ID,
    )
    provenance = repo.read_json(provenance_path)
    provenance["runtime"].update(
        {
            "agent": "provider",
            "agent_harness": "provider",
            "provider": endpoint,
            "turn_count": 3,
            "usage": {"input_tokens": 120, "output_tokens": 34, "total_tokens": 154},
        }
    )
    repo.write_json(provenance_path, provenance)

    summary = create_app(repo_root=tmp_path).state.viewer_store.record_summary(record_id)

    assert summary is not None
    assert summary.agent_harness == "provider"
    assert summary.provider == endpoint
    assert summary.model_id == "gpt-5.5"
    assert summary.has_traces is True
    assert summary.turn_count == 3
    assert summary.input_tokens == 120
    assert summary.output_tokens == 34
    assert summary.total_cost_usd is None
    assert summary.has_cost is False


def test_record_input_image_returns_404_for_a_missing_record(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    store = create_app(repo_root=tmp_path).state.viewer_store

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(record_input_image("rec_missing", store))

    assert exc_info.value.status_code == 404


def test_api_exposes_only_narrow_workbench_mutations(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)
    app = create_app(repo_root=tmp_path)

    api_paths = app.openapi()["paths"]
    assert "put" in api_paths["/api/records/{record_id}/rating"]
    assert "delete" in api_paths["/api/records/{record_id}"]
    mutating_routes = {
        (path, method)
        for path, methods in api_paths.items()
        for method in methods
        if method in {"post", "put", "patch", "delete"}
    }
    # Rating and explicitly confirmed deletion are the only mutations, for both canonical
    # pipelines; everything else stays a read-only projection.
    assert mutating_routes == {
        ("/api/records/{record_id}", "delete"),
        ("/api/records/{record_id}/rating", "put"),
        ("/api/reconstruction-records/{record_id}", "delete"),
        ("/api/reconstruction-records/{record_id}/rating", "put"),
    }


def test_rating_update_is_strict_and_returns_full_summary(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)
    app = create_app(repo_root=tmp_path)
    store = app.state.viewer_store
    mutations = RecordMutationStore(repo)

    for rating in (1, 5, None):
        summary = asyncio.run(
            update_record_rating(
                "rec_generation_001",
                RecordRatingRequest(rating=rating),
                store,
                mutations,
            )
        )
        assert summary.record_id == "rec_generation_001"
        assert summary.rating == rating
        assert summary.model_id == "gpt-5.5"
        assert summary.input_kind == "text_image"
        record = repo.read_json(repo.layout.record_metadata_path("rec_generation_001"))
        assert record["rating"] == rating
        assert record["updated_at"] == summary.updated_at

    for invalid_payload in (
        {},
        {"rating": 0},
        {"rating": 6},
        {"rating": True},
        {"rating": 3.0},
        {"rating": "3"},
    ):
        with pytest.raises(ValidationError):
            RecordRatingRequest.model_validate(invalid_payload)


def test_http_binding_validates_rating_and_delete_confirmation(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)
    app = create_app(repo_root=tmp_path)
    record_id = "rec_generation_001"

    status, payload = _asgi_json_request(
        app,
        "PUT",
        f"/api/records/{record_id}/rating",
        {"rating": 4},
    )
    assert status == 200
    assert payload["record_id"] == record_id
    assert payload["rating"] == 4
    assert payload["input_kind"] == "text_image"

    status, payload = _asgi_json_request(
        app,
        "PUT",
        f"/api/records/{record_id}/rating",
        {"rating": True},
    )
    assert status == 422
    assert payload["detail"][0]["loc"] == ["body", "rating"]

    status, payload = _asgi_json_request(
        app,
        "DELETE",
        f"/api/records/{record_id}",
        {"confirm_record_id": "rec_wrong"},
    )
    assert status == 409
    assert "does not match" in payload["detail"]
    assert repo.layout.record_dir(record_id).is_dir()

    status, payload = _asgi_json_request(
        app,
        "DELETE",
        f"/api/records/{record_id}",
        {"confirm_record_id": record_id},
    )
    assert status == 200
    assert payload == {
        "status": "deleted",
        "record_id": record_id,
        "materialization_deleted": False,
        "cleanup_pending": False,
    }


def test_delete_requires_exact_confirmation_and_preserves_generation_run(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)
    record_id = "rec_generation_001"
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", '<robot name="desk_lamp"/>\n')
    generation_run = repo.layout.generation_run_dir("run_generation_001")
    repo.write_text(generation_run / "debug.log", "retained\n")
    app = create_app(repo_root=tmp_path)
    store = app.state.viewer_store
    mutations = RecordMutationStore(repo)

    with pytest.raises(HTTPException) as mismatch:
        asyncio.run(
            delete_record(
                record_id,
                RecordDeleteRequest(confirm_record_id="rec_some_other_record"),
                mutations,
            )
        )
    assert mismatch.value.status_code == 409
    assert repo.layout.record_dir(record_id).is_dir()
    assert materialization.is_dir()

    response = asyncio.run(
        delete_record(
            record_id,
            RecordDeleteRequest(confirm_record_id=record_id),
            mutations,
        )
    )
    assert response.model_dump() == {
        "status": "deleted",
        "record_id": record_id,
        "materialization_deleted": True,
        "cleanup_pending": False,
    }
    assert not repo.layout.record_dir(record_id).exists()
    assert not materialization.exists()
    assert (generation_run / "debug.log").read_text(encoding="utf-8") == "retained\n"
    with pytest.raises(HTTPException) as missing:
        asyncio.run(record_detail(record_id, store))
    assert missing.value.status_code == 404
    assert store.bootstrap().workbench_entries == []

    with pytest.raises(HTTPException) as repeated:
        asyncio.run(
            delete_record(
                record_id,
                RecordDeleteRequest(confirm_record_id=record_id),
                mutations,
            )
        )
    assert repeated.value.status_code == 404


def test_viewer_only_serves_prebuilt_materialized_assets(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)
    app = create_app(repo_root=tmp_path)
    resolver = ViewerFileResolver(app.state.viewer_store)

    with pytest.raises(HTTPException) as exc_info:
        resolver.resolve_record_target("rec_generation_001", "model.urdf")
    assert exc_info.value.status_code == 404
    assert not repo.layout.record_materialization_urdf_path("rec_generation_001").exists()

    repo.write_text(
        repo.layout.record_materialization_urdf_path("rec_generation_001"),
        '<robot name="desk_lamp"/>\n',
    )
    materialization = repo.layout.record_materialization_dir("rec_generation_001")
    repo.write_text(materialization / "model.usdz", "usdz")
    (materialization / "assets" / "viewer").mkdir(parents=True)
    (materialization / "assets" / "viewer" / "shade.png").write_bytes(b"texture")
    _write_current_materialization(repo, "rec_generation_001")
    _, available = resolver.resolve_record_target("rec_generation_001", "model.urdf")
    assert available.read_text(encoding="utf-8") == '<robot name="desk_lamp"/>\n'
    _, usdc = resolver.resolve_record_target("rec_generation_001", "isaac/model.usdc")
    _, newton_usdc = resolver.resolve_record_target("rec_generation_001", "isaac/model_newton.usdc")
    sidecars = {
        name: resolver.resolve_record_target("rec_generation_001", f"isaac/{name}")[1]
        for name in ("asset.usda", "physics.usdc", "physx.usda", "mujoco.usda")
    }
    _, texture = resolver.resolve_record_target("rec_generation_001", "assets/viewer/shade.png")
    _, render = resolver.resolve_record_target("rec_generation_001", "renders/level_front.png")
    assert Usd.Stage.Open(usdc.as_posix())
    assert Usd.Stage.Open(newton_usdc.as_posix())
    assert all(Usd.Stage.Open(path.as_posix()) for path in sidecars.values())
    assert texture.read_bytes() == b"texture"
    assert render.read_bytes() == b"render:level_front"


def test_materialization_status_requires_current_dual_stack(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    record_id = "rec_generation_001"
    _write_generation_record(repo, record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "<robot/>")
    repo.write_text(materialization / "model.usdz", "usdz")
    store = create_app(repo_root=tmp_path).state.viewer_store
    resolver = ViewerFileResolver(store)

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
        {
            "schema_version": 2,
            "status": "success",
            "validation_profile": "generation",
        },
    )
    incomplete = store.record_summary(record_id)
    assert incomplete is not None
    assert incomplete.materialization_status == "contract_invalid"
    assert incomplete.materialization_displayable is True
    assert any(
        issue.path == "compile_report.json" and "checks_run is missing" in issue.message
        for issue in incomplete.materialization_issues
    )
    _, incomplete_urdf = resolver.resolve_record_target(record_id, "model.urdf")
    assert incomplete_urdf.read_text(encoding="utf-8") == "<robot/>"

    _write_current_materialization(repo, record_id)
    complete = store.record_summary(record_id)
    assert complete is not None
    assert complete.materialization_status == "available"
    assert complete.materialization_displayable is True
    assert complete.materialization_issues == []
    _, current_newton = resolver.resolve_record_target(record_id, "isaac/model_newton.usdc")
    assert Usd.Stage.Open(current_newton.as_posix())

    report_path = materialization / "compile_report.json"
    report = repo.read_json(report_path)
    report["status"] = "failed"
    repo.write_json(report_path, report)
    failed = store.record_summary(record_id)
    assert failed is not None
    assert failed.materialization_status == "compile_failed"
    assert failed.materialization_displayable is False
    assert any(
        issue.message == "status must be 'success'" for issue in failed.materialization_issues
    )
    with pytest.raises(HTTPException) as failed_artifact:
        resolver.resolve_record_target(record_id, "model.urdf")
    assert failed_artifact.value.status_code == 404
    _, diagnostic_report = resolver.resolve_record_target(record_id, "compile_report.json")
    assert repo.read_json(diagnostic_report)["status"] == "failed"

    report["status"] = "success"
    report["schema_version"] = 99
    repo.write_json(report_path, report)
    unknown = store.record_summary(record_id)
    assert unknown is not None
    assert unknown.materialization_status == "contract_invalid"
    assert unknown.materialization_displayable is True
    assert any(
        "schema_version must be 2" in issue.message for issue in unknown.materialization_issues
    )
    _, unknown_urdf = resolver.resolve_record_target(record_id, "model.urdf")
    assert unknown_urdf.is_file()

    report_path.write_text("{not-json", encoding="utf-8")
    malformed = store.record_summary(record_id)
    assert malformed is not None
    assert malformed.materialization_status == "contract_invalid"
    assert malformed.materialization_displayable is True
    assert any("invalid JSON" in issue.message for issue in malformed.materialization_issues)
    _, malformed_urdf = resolver.resolve_record_target(record_id, "model.urdf")
    assert malformed_urdf.is_file()

    repo.write_text(materialization / "assets" / "viewer" / "still-safe.obj", "mesh")
    _, display_asset = resolver.resolve_record_target(record_id, "assets/viewer/still-safe.obj")
    assert display_asset.is_file()
    repo.write_text(materialization / "internal.bin", "not-viewer-content")
    with pytest.raises(HTTPException) as unexpected_materialization_file:
        resolver.resolve_record_target(record_id, "internal.bin")
    assert unexpected_materialization_file.value.status_code == 404

    (materialization / "model.urdf").unlink()
    missing = store.record_summary(record_id)
    assert missing is not None
    assert missing.materialization_status == "missing"
    assert missing.materialization_displayable is False
    assert any(
        issue.path == "model.urdf" and "missing required Viewer model file" in issue.message
        for issue in missing.materialization_issues
    )
    with pytest.raises(HTTPException) as missing_urdf:
        resolver.resolve_record_target(record_id, "model.urdf")
    assert missing_urdf.value.status_code == 404


def test_generation_store_validate_runs_and_caches_the_full_contract(tmp_path: Path) -> None:
    """`ViewerStore.validate` is the "I suspect this asset is wrong" action.

    It is deliberately not exposed as a Viewer HTTP route: rating update/clear and explicitly
    confirmed deletion remain the only two Workbench mutations
    (`test_api_exposes_only_narrow_workbench_mutations`). The equivalent user entry point is
    the `usdcraft generation-validate` CLI command.
    """

    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    record_id = "rec_generation_001"
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

    store = create_app(repo_root=tmp_path).state.viewer_store

    # Before any full check, `model.urdf` exists and compile status is "success", so the cheap
    # hint alone cannot distinguish `available` from `contract_invalid`; the list projection is
    # the neutral placeholder.
    assert store.list_records()[0].materialization_status == "not_validated"
    assert store.list_records()[0].materialization_validated is False

    validated = store.validate(record_id)
    assert validated is not None
    assert validated.materialization_status == "contract_invalid"
    assert validated.materialization_validated is True
    assert validated.materialization_issues

    # The explicit validate call warms the same cache the list projection reads.
    assert store.list_records()[0].materialization_status == "contract_invalid"
    assert store.list_records()[0].materialization_validated is True

    assert store.validate("rec_generation_missing") is None


def test_generation_validation_cache_does_not_survive_a_materialization_change(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    record_id = "rec_generation_001"
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
    store = create_app(repo_root=tmp_path).state.viewer_store

    first = store.validate(record_id)
    assert first is not None
    assert first.materialization_status == "contract_invalid"
    assert store.list_records()[0].materialization_validated is True

    # An explicit compile failure changes the materialization on disk; the stat-only signature
    # this cache keys on must change too, so the stale cached result cannot survive.
    repo.write_json(materialization / "compile_report.json", {"status": "failed"})

    lazy = store.list_records()[0]
    assert lazy.materialization_status == "compile_failed"
    # The cheap hint alone is conclusive for `compile_failed`, so this is trustworthy without
    # re-running the full contract, but it is honestly reported as not re-validated.
    assert lazy.materialization_validated is False


def test_app_startup_does_not_create_storage(tmp_path: Path) -> None:
    assert not (tmp_path / "data").exists()
    app = create_app(repo_root=tmp_path)
    assert app.state.viewer_store.bootstrap().workbench_entries == []
    assert app.state.deletion_recovery.completed_record_ids == ()
    assert app.state.deletion_recovery.cleanup_pending_record_ids == ()
    assert not (tmp_path / "data").exists()


def test_app_startup_recovers_interrupted_record_deletion(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    record_id = "rec_generation_001"
    _write_generation_record(repo, record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "compiled")
    staging = repo.layout.record_deletion_staging_dir(record_id)
    staging.mkdir(parents=True)
    repo.write_json(
        staging / "transaction.json",
        {
            "schema_version": 1,
            "record_id": record_id,
            "state": "record_staged",
            "materialization_expected": True,
        },
    )
    repo.layout.record_dir(record_id).replace(staging / "record")

    app = create_app(repo_root=tmp_path)

    assert app.state.deletion_recovery.completed_record_ids == (record_id,)
    assert app.state.deletion_recovery.cleanup_pending_record_ids == ()
    assert app.state.viewer_store.bootstrap().workbench_entries == []
    assert not repo.layout.record_dir(record_id).exists()
    assert not materialization.exists()
    assert not staging.exists()


def test_file_resolver_rejects_noncanonical_and_traversal_paths(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)
    resolver = ViewerFileResolver(create_app(repo_root=tmp_path).state.viewer_store)

    for file_path in ("../record.json", "collections/workbench.json"):
        with pytest.raises(HTTPException):
            resolver.resolve_record_target("rec_generation_001", file_path)


def test_bootstrap_isolates_corrupt_record_json(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo, record_id="rec_generation_good")

    corrupt_id = "rec_generation_corrupt"
    _write_generation_record(repo, record_id=corrupt_id)
    repo.layout.record_metadata_path(corrupt_id).write_text("{not valid json", encoding="utf-8")

    app = create_app(repo_root=tmp_path)
    entries = app.state.viewer_store.bootstrap().workbench_entries

    assert [entry.record_id for entry in entries] == ["rec_generation_good"]


def test_file_resolver_returns_404_for_corrupt_record_json(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    record_id = "rec_generation_broken"
    _write_generation_record(repo, record_id=record_id)
    repo.layout.record_metadata_path(record_id).write_text("{not valid json", encoding="utf-8")

    resolver = ViewerFileResolver(create_app(repo_root=tmp_path).state.viewer_store)
    with pytest.raises(HTTPException) as exc_info:
        resolver.resolve_record_target(record_id, "model.urdf")
    assert exc_info.value.status_code == 404


def test_generation_summary_reports_batch_membership(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)
    provenance_path = (
        repo.layout.record_revision_dir("rec_generation_001", INITIAL_REVISION_ID)
        / "provenance.json"
    )
    provenance = repo.read_json(provenance_path)
    assert isinstance(provenance, dict)
    runtime = dict(provenance.get("runtime") or {})
    runtime["batch"] = {
        "batch_id": "provider_batch_example_20260905T120000Z",
        "csv": "jobs.csv",
        "row_id": "toaster",
    }
    provenance["runtime"] = runtime
    repo.write_json(provenance_path, provenance)

    summary = create_app(repo_root=tmp_path).state.viewer_store.record_summary("rec_generation_001")
    assert summary is not None
    assert summary.batch_id == "provider_batch_example_20260905T120000Z"
    assert summary.batch_row_id == "toaster"


def test_read_only_viewer_refuses_mutations_and_announces_itself(tmp_path: Path) -> None:
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from viewer.api.dependencies import get_record_mutations, is_read_only
    from viewer.api.routes.status import status as status_route

    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)

    read_only_app = create_app(repo_root=tmp_path, read_only=True)
    request = SimpleNamespace(app=read_only_app)
    assert is_read_only(request) is True  # type: ignore[arg-type]
    with pytest.raises(HTTPException) as refused:
        get_record_mutations(request)  # type: ignore[arg-type]
    assert refused.value.status_code == 403 and "read-only" in str(refused.value.detail)
    health = asyncio.run(status_route(read_only_app.state.viewer_store, True))
    assert health.read_only is True
    # projections are untouched
    assert read_only_app.state.viewer_store.record_summary("rec_generation_001") is not None

    writable_app = create_app(repo_root=tmp_path)
    assert is_read_only(SimpleNamespace(app=writable_app)) is False  # type: ignore[arg-type]
    assert get_record_mutations(SimpleNamespace(app=writable_app)) is not None  # type: ignore[arg-type]
    assert asyncio.run(status_route(writable_app.state.viewer_store, False)).read_only is False


def test_read_only_flag_comes_from_the_environment_when_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    monkeypatch.setenv("USDCRAFT_VIEWER_READ_ONLY", "1")
    assert create_app(repo_root=tmp_path).state.read_only is True
    monkeypatch.setenv("USDCRAFT_VIEWER_READ_ONLY", "0")
    assert create_app(repo_root=tmp_path).state.read_only is False
    assert create_app(repo_root=tmp_path, read_only=True).state.read_only is True


def test_generation_summary_reports_run_settings(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)
    provenance_path = (
        repo.layout.record_revision_dir("rec_generation_001", INITIAL_REVISION_ID)
        / "provenance.json"
    )
    provenance = repo.read_json(provenance_path)
    assert isinstance(provenance, dict)
    provenance["runtime"] = {**provenance["runtime"], "image_observation": "external_inventory"}
    repo.write_json(provenance_path, provenance)

    summary = create_app(repo_root=tmp_path).state.viewer_store.record_summary("rec_generation_001")
    assert summary is not None
    assert summary.runtime_source == "codex_root_thread_state"
    # The Codex arm has no Provider wire protocol.
    assert summary.api_style is None
    assert summary.image_observation == "external_inventory"


def test_generation_provider_summary_reports_its_wire_protocol(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    repo.ensure_layout()
    _write_generation_record(repo)
    metadata_path = repo.layout.record_metadata_path("rec_generation_001")
    record = repo.read_json(metadata_path)
    assert isinstance(record, dict)
    record.update(
        agent_harness="provider",
        provider="nvidia",
        runtime_source="nvidia-chat-completions-api",
    )
    repo.write_json(metadata_path, record)

    summary = create_app(repo_root=tmp_path).state.viewer_store.record_summary("rec_generation_001")
    assert summary is not None
    assert summary.agent_harness == "provider"
    assert summary.api_style == "chat-completions"


def _status_code(app: Any, host: str) -> int:
    status, _ = _asgi_json_request(app, "GET", "/health", host=host)
    return status


@pytest.mark.parametrize(
    "host",
    ["localhost:8770", "127.0.0.1:8770", "[::1]:8770", "localhost", "LOCALHOST.:5173"],
)
def test_viewer_accepts_loopback_host_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    monkeypatch.delenv("USDCRAFT_VIEWER_ALLOWED_HOSTS", raising=False)
    assert _status_code(create_app(repo_root=tmp_path), host) == 200


@pytest.mark.parametrize(
    "host",
    ["evil.example", "evil.example:8770", "localhost.evil.example", "127.0.0.1.nip.io", ""],
)
def test_viewer_rejects_foreign_host_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    """A DNS-rebound page still sends its own domain as Host; the API refuses it."""

    monkeypatch.delenv("USDCRAFT_VIEWER_ALLOWED_HOSTS", raising=False)
    app = create_app(repo_root=tmp_path)
    status, body = _asgi_json_request(app, "GET", "/api/status", host=host)
    assert status == 400
    assert body == "Invalid host header"


def test_viewer_accepts_an_explicitly_allowed_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("USDCRAFT_VIEWER_ALLOWED_HOSTS", raising=False)
    app = create_app(repo_root=tmp_path, allowed_hosts=["viewer.lan"])
    assert _status_code(app, "viewer.lan:8770") == 200
    assert _status_code(app, "127.0.0.1:8770") == 200
    assert _status_code(app, "evil.example") == 400

    # `usdcraft viewer --host` reaches the uvicorn-imported app through the environment.
    monkeypatch.setenv("USDCRAFT_VIEWER_ALLOWED_HOSTS", "192.168.1.20, ::1")
    env_app = create_app(repo_root=tmp_path)
    assert _status_code(env_app, "192.168.1.20:8770") == 200
    assert _status_code(env_app, "[::1]:8770") == 200
    assert _status_code(env_app, "evil.example:8770") == 400


def test_wildcard_bind_turns_the_host_check_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USDCRAFT_VIEWER_ALLOWED_HOSTS", "0.0.0.0")
    app = create_app(repo_root=tmp_path)
    assert app.state.allowed_hosts is None
    assert _status_code(app, "viewer-host.example:8770") == 200
