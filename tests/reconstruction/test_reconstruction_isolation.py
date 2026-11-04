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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
from PIL import Image

from usdcraft.reconstruction.isolation import (
    SUBJECT_ISOLATION_SCHEMA,
    SUBJECT_ISOLATION_SKIPPED_CONTEXT,
    SubjectIsolationError,
    apply_subject_isolation,
    build_subject_isolation_inventory,
    load_subject_isolation_decision,
    normalized_scene_identity_sha256,
    write_retain_all_subject_isolation,
)
from usdcraft.reconstruction.isolation_agent import (
    IsolationProcessResult,
    run_subject_isolation,
)
from usdcraft.reconstruction.isolation_render import ISOLATION_VIEWS
from usdcraft.reconstruction.source import SourceInput, load_normalized_source
from usdcraft.render.browser import RenderRequest, RenderResult


def _write_two_component_glb(path: Path) -> None:
    scene = trimesh.Scene()
    left = trimesh.creation.box(extents=(0.5, 0.4, 0.3))
    right = trimesh.creation.box(extents=(0.25, 0.2, 0.15))
    left.apply_translation((-1.0, 0.0, 0.0))
    right.apply_translation((1.0, 0.0, 0.0))
    scene.add_geometry(left, geom_name="primary", node_name="primary")
    scene.add_geometry(right, geom_name="background", node_name="background")
    payload = scene.export(file_type="glb")
    assert isinstance(payload, bytes)
    path.write_bytes(payload)


def _bundle_builder(dist_dir: Path) -> None:
    dist_dir.mkdir(parents=True)
    (dist_dir / "render.html").write_text("test bundle", encoding="utf-8")


class _FakeChromium:
    def __init__(self, calls: dict[str, Any]):
        self.calls = calls

    def render(
        self,
        request: RenderRequest,
        *,
        allow_numpy_fallback: bool = True,
    ) -> RenderResult:
        assert allow_numpy_fallback is False
        self.calls.setdefault("render_requests", []).append(request)
        images = []
        for index, (name, azimuth, elevation) in enumerate(request.views):
            path = request.out_dir / f"{name}.png"
            Image.new(
                "RGB",
                (request.size, request.size),
                (80 + index * 20, 120, 160),
            ).save(path)
            images.append(
                {
                    "name": name,
                    "path": str(path),
                    "camera": {"azimuthDeg": azimuth, "elevationDeg": elevation},
                }
            )
        manifest = request.out_dir / "render_manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return RenderResult(
            backend="viewer_chromium",
            capabilities={"geometry": True, "textures": True},
            profile=request.profile,
            images=images,
            manifest_path=manifest,
        )

    def close(self) -> None:
        self.calls["renderer_closed"] = True


def _ready_manifest(path: Path, scene, *, retained: list[str], excluded: list[str]) -> None:  # type: ignore[no-untyped-def]
    path.write_text(
        json.dumps(
            {
                "schema": SUBJECT_ISOLATION_SCHEMA,
                "status": "ready",
                "source_identity_sha256": normalized_scene_identity_sha256(scene),
                "selection": {
                    "retained_component_ids": retained,
                    "excluded_component_ids": excluded,
                    "unresolved_component_ids": [],
                },
                "agent": {"run": True, "context": "clean_subject_isolation_only"},
            }
        ),
        encoding="utf-8",
    )


def test_inventory_exposes_only_complete_components_and_strict_gap_selection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    _write_two_component_glb(source)
    scene = load_normalized_source(SourceInput(source))

    inventory = build_subject_isolation_inventory(scene)

    assert inventory["schema"] == "reconstruction_subject_isolation_inventory"
    assert inventory["evidence_role"] == "subject_isolation_only"
    assert inventory["geometry_authority"] == "surface_only"
    assert inventory["interior_semantics"] == "unknown"
    assert inventory["reconstruction_agent_input"] is False
    assert len(inventory["components"]) == 2
    assert all(item["selection_id"].startswith("component:c") for item in inventory["components"])
    assert inventory["selection_contract"] == {
        "individual_component_selection_id": "component:<component_id>",
        "selection_units_expand_to_complete_components": True,
        "triangle_or_point_clipping": False,
        "unspecified_components": "retain",
        "unresolved_components": "retain",
    }
    gap_units = inventory["selection_units"]
    assert gap_units
    assert all(item["selector_kind"] == "strict_axis_empty_band_side" for item in gap_units)
    assert all(item["component_ids"] for item in gap_units)
    assert all(
        "not_free_space_or_subject_authority" in item["evidence_semantics"] for item in gap_units
    )


def test_ready_decision_is_identity_bound_and_preserves_retained_samples(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    _write_two_component_glb(source)
    scene = load_normalized_source(SourceInput(source))
    retained_id, excluded_id = (component.component_id for component in scene.components)
    manifest = tmp_path / "subject_isolation.json"
    _ready_manifest(manifest, scene, retained=[retained_id], excluded=[excluded_id])

    decision = load_subject_isolation_decision(manifest, scene)
    clean = apply_subject_isolation(scene, decision)

    assert [component.component_id for component in clean.components] == [retained_id]
    assert clean.components[0] is scene.components[0]
    np.testing.assert_array_equal(clean.components[0].vertices, scene.components[0].vertices)
    np.testing.assert_array_equal(clean.components[0].faces, scene.components[0].faces)

    changed_component = replace(
        scene.components[0],
        vertices=scene.components[0].vertices + np.asarray([0.001, 0.0, 0.0]),
    )
    changed_scene = replace(scene, components=(changed_component, scene.components[1]))
    with pytest.raises(SubjectIsolationError, match="identity does not match"):
        load_subject_isolation_decision(manifest, changed_scene)


def test_skipped_mesh_cleanup_freezes_a_loadable_retain_all_decision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    _write_two_component_glb(source)
    scene = load_normalized_source(SourceInput(source))

    manifest_path = write_retain_all_subject_isolation(scene, tmp_path / "subject_isolation")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == SUBJECT_ISOLATION_SCHEMA
    assert manifest["status"] == "ready"
    assert manifest["agent"]["run"] is False
    assert manifest["agent"]["context"] == SUBJECT_ISOLATION_SKIPPED_CONTEXT
    assert manifest["agent"]["attached_image_count"] == 0
    assert manifest["selection"]["excluded_component_ids"] == []
    assert manifest["selection"]["unresolved_component_ids"] == []
    assert (manifest_path.parent / "component_inventory.json").is_file()
    assert not (manifest_path.parent / "evidence").exists()

    decision = load_subject_isolation_decision(manifest_path, scene)
    assert decision.upstream_agent_run is False
    assert decision.upstream_agent_context == SUBJECT_ISOLATION_SKIPPED_CONTEXT
    retained = apply_subject_isolation(scene, decision)
    assert [component.component_id for component in retained.components] == [
        component.component_id for component in scene.components
    ]
    np.testing.assert_array_equal(retained.components[0].vertices, scene.components[0].vertices)

    with pytest.raises(SubjectIsolationError, match="refusing to overwrite"):
        write_retain_all_subject_isolation(scene, tmp_path / "subject_isolation")


def test_retain_all_declaration_cannot_exclude_or_impersonate_an_agent_run(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    _write_two_component_glb(source)
    scene = load_normalized_source(SourceInput(source))
    retained_id, excluded_id = (component.component_id for component in scene.components)

    excluding = tmp_path / "excluding.json"
    excluding.write_text(
        json.dumps(
            {
                "schema": SUBJECT_ISOLATION_SCHEMA,
                "status": "ready",
                "source_identity_sha256": normalized_scene_identity_sha256(scene),
                "selection": {
                    "retained_component_ids": [retained_id],
                    "excluded_component_ids": [excluded_id],
                    "unresolved_component_ids": [],
                },
                "agent": {"run": False, "context": SUBJECT_ISOLATION_SKIPPED_CONTEXT},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SubjectIsolationError, match="retain every complete component"):
        load_subject_isolation_decision(excluding, scene)

    mismatched = tmp_path / "mismatched.json"
    mismatched.write_text(
        json.dumps(
            {
                "schema": SUBJECT_ISOLATION_SCHEMA,
                "status": "ready",
                "source_identity_sha256": normalized_scene_identity_sha256(scene),
                "selection": {
                    "retained_component_ids": [retained_id, excluded_id],
                    "excluded_component_ids": [],
                    "unresolved_component_ids": [],
                },
                "agent": {"run": False, "context": "clean_subject_isolation_only"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SubjectIsolationError, match="explicit clean-context isolation Agent"):
        load_subject_isolation_decision(mismatched, scene)


def test_ready_decision_cannot_exclude_every_component(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    _write_two_component_glb(source)
    scene = load_normalized_source(SourceInput(source))
    manifest = tmp_path / "subject_isolation.json"
    _ready_manifest(
        manifest,
        scene,
        retained=[],
        excluded=[component.component_id for component in scene.components],
    )

    with pytest.raises(SubjectIsolationError, match="retain at least one"):
        load_subject_isolation_decision(manifest, scene)


def test_explicit_isolation_run_uses_clean_context_and_freezes_audited_subset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "isolation"
    _write_two_component_glb(source)
    raw_scene = load_normalized_source(SourceInput(source))
    excluded_id = raw_scene.components[1].component_id
    calls: dict[str, Any] = {}

    def process_runner(
        command,
        *,
        stdin_text,
        cwd,
        env,  # type: ignore[no-untyped-def]
    ) -> IsolationProcessResult:
        # The Codex arm has no time limit: the runner receives no timeout.
        calls["command"] = tuple(command)
        calls["stdin"] = stdin_text
        calls["cwd"] = cwd
        calls["workspace_files_before_run"] = sorted(
            path.relative_to(cwd).as_posix() for path in cwd.rglob("*") if path.is_file()
        )
        final_path = Path(command[command.index("--output-last-message") + 1])
        final_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "decisions": [
                        {
                            "selection_id": f"component:{excluded_id}",
                            "disposition": "exclude_background_or_incidental",
                            "reason": "detached non-subject component",
                            "visual_evidence": "separate in every fixed view",
                        }
                    ],
                    "summary": "retained the coherent primary component",
                }
            ),
            encoding="utf-8",
        )
        return IsolationProcessResult(returncode=0, stdout='{"type":"done"}\n', stderr="")

    result = run_subject_isolation(
        SourceInput(source),
        output,
        model="test-model",
        reasoning_effort="high",
        executable="codex-test",
        environ={},
        bundle_builder=_bundle_builder,
        session_factory=lambda _dist: _FakeChromium(calls),
        process_runner=process_runner,
    )

    assert result.status == "ready"
    command = calls["command"]
    assert command[:3] == ("codex-test", "exec", "--skip-git-repo-check")
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command.count("--disable") >= 10
    assert "plugins" in command and "multi_agent" in command and "browser_use" in command
    assert command.count("--image") == 2
    assert command[command.index("--model") + 1] == "test-model"
    assert [command[index + 1] for index, value in enumerate(command) if value == "-c"].count(
        "project_doc_max_bytes=0"
    ) == 1
    assert calls["workspace_files_before_run"] == [
        ".codex-run/final.schema.json",
        "TASK.md",
        "component_id_contact_sheet.png",
        "component_inventory.json",
        "native_contact_sheet.png",
    ]
    assert str(source) not in calls["stdin"]
    assert "semantic input image is provided" in calls["stdin"]
    assert len(calls["render_requests"]) == 2
    assert all(tuple(request.views) == ISOLATION_VIEWS for request in calls["render_requests"])
    assert calls["renderer_closed"] is True

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == SUBJECT_ISOLATION_SCHEMA
    assert manifest["status"] == "ready"
    assert manifest["canonical_record"] is False
    assert manifest["selection"]["excluded_component_ids"] == [excluded_id]
    assert manifest["selection"]["retained_component_count"] == 1
    assert manifest["selection"]["retained_samples_unchanged"] is True
    assert manifest["selection"]["triangle_or_point_clipping"] is False
    assert manifest["agent"]["context"] == "clean_subject_isolation_only"
    assert manifest["agent"]["attached_image_count"] == 2
    assert manifest["agent"]["semantic_image"] is False
    assert manifest["agent"]["axial"] is False
    assert manifest["agent"]["source_mesh"] is False
    assert manifest["agent"]["user_config_loaded"] is False
    assert manifest["agent"]["project_documents_loaded"] is False
    assert "plugins" in manifest["agent"]["disabled_codex_features"]
    assert (
        manifest["downstream_contract"][
            "same_retained_scene_for_normalized_source_axial_renders_and_evaluation"
        ]
        is True
    )
    assert manifest["artifacts"]["native_contact_sheet"]["isolation_agent_input"] is True
    assert manifest["artifacts"]["native_contact_sheet"]["reconstruction_agent_input"] is False
    assert manifest["artifacts"]["agent_context_manifest"]["isolation_agent_input"] is False
    assert result.retained_source.is_file()
    retained_manifest = json.loads(result.retained_source_manifest.read_text(encoding="utf-8"))
    assert retained_manifest["geometry"]["component_count"] == 1
    with (
        Image.open(result.native_contact_sheet) as native,
        Image.open(result.component_id_contact_sheet) as component_id,
    ):
        assert native.width == component_id.width == 3 * 512
        assert component_id.height > native.height

    decision = load_subject_isolation_decision(result.manifest, raw_scene)
    assert decision.excluded_component_ids == (excluded_id,)


def test_blocked_agent_cannot_publish_an_exclusion(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "isolation"
    _write_two_component_glb(source)
    scene = load_normalized_source(SourceInput(source))

    def process_runner(command, **_kwargs):  # type: ignore[no-untyped-def]
        final_path = Path(command[command.index("--output-last-message") + 1])
        final_path.write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "decisions": [
                        {
                            "selection_id": f"component:{scene.components[0].component_id}",
                            "disposition": "exclude_background_or_incidental",
                            "reason": "uncertain",
                            "visual_evidence": "uncertain",
                        }
                    ],
                    "summary": "could not identify the subject",
                }
            ),
            encoding="utf-8",
        )
        return IsolationProcessResult(0, "", "")

    with pytest.raises(SubjectIsolationError, match="blocked.*must not exclude"):
        run_subject_isolation(
            SourceInput(source),
            output,
            model="test-model",
            reasoning_effort="high",
            executable="codex-test",
            environ={},
            bundle_builder=_bundle_builder,
            session_factory=lambda _dist: _FakeChromium({}),
            process_runner=process_runner,
        )
    assert not output.exists()
