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
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdGeom

from usdcraft.reconstruction import mesh_preprocess as reconstruction_case
from usdcraft.reconstruction.isolation import (
    SUBJECT_ISOLATION_SCHEMA,
    normalized_scene_identity_sha256,
)
from usdcraft.reconstruction.mesh_preprocess import (
    create_observation_case,
)
from usdcraft.reconstruction.source import SourceInput, load_normalized_source


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_box_usd(path: Path) -> None:
    box = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
    vertices = np.asarray(box.vertices, dtype=np.float64)
    faces = np.asarray(box.faces, dtype=np.int64)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Source")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Source/Box")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, vertex)) for vertex in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(value) for value in faces.reshape(-1)])
    stage.GetRootLayer().Save()


def _write_keep_all_isolation(source_input: SourceInput, path: Path) -> Path:
    scene = load_normalized_source(source_input)
    component_ids = [component.component_id for component in scene.components]
    path.write_text(
        json.dumps(
            {
                "schema": SUBJECT_ISOLATION_SCHEMA,
                "status": "ready",
                "source_identity_sha256": normalized_scene_identity_sha256(scene),
                "selection": {
                    "retained_component_ids": component_ids,
                    "excluded_component_ids": [],
                    "unresolved_component_ids": [],
                },
                "agent": {"run": True, "context": "clean_subject_isolation_only"},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.slow
def test_mesh_preprocess_case_is_published_complete_without_any_render(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "formal-observation"
    _write_box_usd(source)
    source_input = SourceInput(source, front_axis="-Y")
    isolation = _write_keep_all_isolation(source_input, tmp_path / "subject_isolation.json")

    result = create_observation_case(source_input, output, subject_isolation=isolation)

    assert result.root == output.resolve()
    for path in (
        result.normalized_source,
        result.source_manifest,
        result.subject_isolation,
        result.axial,
        result.voxel_evidence,
        result.case_manifest,
    ):
        assert path.is_file()
    assert result.axial.name == "axial.txt"
    assert result.voxel_evidence.name == "axial_voxels.json"
    axial_text = result.axial.read_text(encoding="utf-8")
    assert axial_text.startswith("AXIAL\n")
    assert "[MICRO_SLICE_BOUNDS_Z192" in axial_text
    assert "[SLICE_STACK32 axis=Z " in axial_text
    assert result.axial.is_file()
    assert result.voxel_evidence.is_file()
    assert not (result.root / "source" / "preview").exists()
    manifest = json.loads(result.case_manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == "reconstruction_mesh_preprocess_case"
    assert manifest["scope"] == "reconstruction_mesh_preprocessing"
    assert manifest["agent_run_scope"] == "this_mesh_preprocess_transaction_only"
    assert "observation" not in manifest
    assert manifest["axial"]["format"] == "AXIAL"
    assert result.axial.parent.name == "axial"
    assert manifest["canonical_record"] is False and manifest["agent_run"] is False
    assert manifest["preprocessing"]["subject_isolation"]["retained_component_count"] == 1
    assert not (result.root / "source" / "preview").exists()
    for artifact in manifest["artifacts"].values():
        path = result.root / artifact["path"]
        assert _sha256(path) == artifact["sha256"]


def test_failed_axial_budget_leaves_no_case_or_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "failed-axial-observation"
    _write_box_usd(source)
    source_input = SourceInput(source, front_axis="-Y")
    isolation = _write_keep_all_isolation(source_input, tmp_path / "subject_isolation.json")

    def fail_axial(_scene):  # type: ignore[no-untyped-def]
        raise ValueError("no AXIAL slice stack fits the complete Agent Core budget")

    monkeypatch.setattr(reconstruction_case, "build_axial_observation", fail_axial)

    with pytest.raises(ValueError, match="no AXIAL slice stack fits"):
        create_observation_case(source_input, output, subject_isolation=isolation)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".failed-axial-observation.staging-*"))
