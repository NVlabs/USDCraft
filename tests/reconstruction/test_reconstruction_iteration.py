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
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from usdcraft.reconstruction.author_tools import measure_mesh
from usdcraft.reconstruction.axial_summary import summarize_axial_payloads
from usdcraft.reconstruction.diagnostic_binding import diagnostic_binding
from usdcraft.reconstruction.diagnostic_cli import read_axial_payloads, write_axial_payload
from usdcraft.reconstruction.render_service import CandidateRenderService, request_candidate_render


def test_nested_checks_are_registered_ordered_deduplicated_and_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("USDCRAFT_RECONSTRUCTION_WORKSPACE", str(tmp_path))
    source = tmp_path / "source.usda"
    candidate = tmp_path / "candidate.usda"
    for path in [source, candidate]:
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.Cube.Define(stage, "/Box")
        stage.GetRootLayer().Save()
    first = diagnostic_binding(source, candidate)
    result = {"evidence_comparison": {"recall": 0.4}}
    write_axial_payload(tmp_path / "axial_checks", text="first", result=result, binding=first)
    stage = Usd.Stage.Open(str(candidate))
    UsdGeom.Cube(stage.GetPrimAtPath("/Box")).GetSizeAttr().Set(3)
    stage.GetRootLayer().Save()
    final = diagnostic_binding(source, candidate)
    assert first != final
    write_axial_payload(
        tmp_path / "axial_checks" / "final",
        text="second",
        result={"evidence_comparison": {"recall": 0.9}},
        binding=final,
    )
    calls = read_axial_payloads(tmp_path / "axial_checks")
    assert len(calls) == 2
    summary = summarize_axial_payloads(calls, final_binding=final)
    assert summary["calls"] == 2 and summary["first_recall"] == 0.4 and summary["recall"] == 0.9
    assert summary["binding_status"] == "current"
    assert summarize_axial_payloads(calls, final_binding=first)["binding_status"] == "stale"
    assert len(list((tmp_path / "author_feedback/axial_checks").glob("*.json"))) == 2


def test_measurement_previews_points_but_preserves_samples_and_panel_orientation(tmp_path):
    source = tmp_path / "panel.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    points = UsdGeom.Points.Define(stage, "/Panel")
    xy = np.array([(x, y) for x in np.linspace(-0.1, 0.1, 15) for y in np.linspace(-0.1, 0.1, 15)])
    points.GetPointsAttr().Set(np.column_stack([xy, xy[:, 0]]).tolist())
    stage.GetRootLayer().Save()
    result = measure_mesh(source, out_dir=tmp_path / "query", max_points=128)
    assert len(result["points_xyz_m"]) == 16
    assert len(json.loads(Path(result["detail_path"]).read_text())["points_xyz_m"]) == 128
    assert result["plane_fit"]["inclination_to_xy_deg"] == 45
    assert result["plane_fit"]["residual_p90_mm"] < 0.001


def test_host_render_transport_preserves_pose_and_reports_errors(tmp_path, monkeypatch):
    import usdcraft.reconstruction.author_tools as author_tools

    monkeypatch.setenv("USDCRAFT_RECONSTRUCTION_WORKSPACE", str(tmp_path))
    seen = []

    def fake(asset_dir, **kwargs):
        seen.append((asset_dir, kwargs))
        if kwargs["joint_positions"].get("bad"):
            raise ValueError("Unknown joints")
        return {
            "ok": True,
            "backend": "viewer_chromium",
            "images": [],
            "capabilities": {"roughness": True},
        }

    monkeypatch.setattr(author_tools, "render_candidate", fake)
    with CandidateRenderService(tmp_path):
        result = request_candidate_render(
            tmp_path / "build",
            out_dir=tmp_path / "author_feedback/view",
            azimuth=315,
            elevation=40,
            joint_positions={"slide": 0.11},
        )
        assert (
            result["execution"] == "reconstruction_host" and result["backend"] == "viewer_chromium"
        )
        assert seen[0][1]["joint_positions"] == {"slide": 0.11}
        with pytest.raises(RuntimeError, match="Unknown joints"):
            request_candidate_render(
                tmp_path / "build",
                out_dir=tmp_path / "author_feedback/bad",
                azimuth=0,
                elevation=0,
                joint_positions={"bad": 1},
            )
        with pytest.raises(ValueError, match="inside"):
            request_candidate_render(
                tmp_path.parent / "outside",
                out_dir=tmp_path / "author_feedback/bad",
                azimuth=0,
                elevation=0,
                joint_positions={},
            )
    with pytest.raises(RuntimeError, match="unavailable"):
        request_candidate_render(
            tmp_path / "build",
            out_dir=tmp_path / "author_feedback/view",
            azimuth=0,
            elevation=0,
            joint_positions={},
        )


def test_part_comparison_does_not_let_another_part_mask_missing_shape(tmp_path):
    from usdcraft.reconstruction.part_comparison import compare_part

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    import trimesh

    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    cube = UsdGeom.Mesh.Define(stage, "/Cube")
    cube.GetSubdivisionSchemeAttr().Set("none")
    cube.GetPointsAttr().Set(box.vertices.tolist())
    cube.GetFaceVertexCountsAttr().Set([3] * len(box.faces))
    cube.GetFaceVertexIndicesAttr().Set(box.faces.reshape(-1).tolist())
    stage.GetRootLayer().Save()
    (tmp_path / "model.urdf").write_text(
        """<robot name="test"><link name="target"><visual name="button"><origin xyz="0 0.3 0"/><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link><link name="other"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link></robot>"""
    )
    result = compare_part(
        source,
        tmp_path,
        out_dir=tmp_path / "comparison",
        part="target",
        element="button",
        face="-Y",
        bounds=[-0.06, -0.06, -0.06, 0.06, 0.06, 0.06],
    )
    assert result["part"] == "target"
    assert result["local_surface"]["depth_offset_mm"] < -250
    assert Path(result["images"][0]).exists()
    empty = compare_part(
        source,
        tmp_path,
        out_dir=tmp_path / "empty",
        part="target",
        face="-Y",
        bounds=[1, 1, 1, 2, 2, 2],
    )
    assert empty["local_surface"]["status"] == "unassessed"
    with pytest.raises(ValueError, match="No candidate"):
        compare_part(
            source,
            tmp_path,
            out_dir=tmp_path / "bad",
            part="typo",
            face="-Y",
            bounds=[-0.06] * 3 + [0.06] * 3,
        )


def test_provider_compacts_conversation_but_preserves_full_trace_and_readable_detail(tmp_path):
    from tests.reconstruction.test_reconstruction_provider import _provider_workspace
    from usdcraft.pipeline.provider.harness import _FunctionCall, _traced_tool_fields
    from usdcraft.pipeline.tools import ProviderToolResult
    from usdcraft.reconstruction.provider_tools import ReconstructionToolExecutor

    _, _, _, _, workspace = _provider_workspace(tmp_path)
    result = {"evidence_comparison": {"recall": 0.7}, "large_reference": list(range(10000))}
    executor = ReconstructionToolExecutor(
        workspace,
        check_runner=lambda p: ProviderToolResult.ok(),
        compile_runner=lambda p, o, render: ProviderToolResult.ok(),
        axial_runner=lambda *args, **kwargs: ProviderToolResult.ok(
            {"text": "check", "result": result}
        ),
    )
    executor.execute("compile_model", {})
    executor.candidate_usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(executor.candidate_usd_path))
    UsdGeom.Cube.Define(stage, "/Box")
    stage.GetRootLayer().Save()
    execution = executor.execute("check_axial", {})
    assert execution.payload["ok"]
    assert len(json.dumps(execution.payload)) < 2000
    trace = _traced_tool_fields(
        _FunctionCall(call_id="call", name="check_axial", arguments="{}"), execution
    )
    assert trace["output"]["output"]["result"] == result
    detail = execution.payload["output"]["detail_path"]
    assert executor.execute("read_file", {"path": detail}).payload["ok"]
    assert json.loads(Path(detail).read_text())["output"]["result"] == result


@pytest.mark.slow
def test_recompiled_usdc_binding_ignores_serialization_and_tracks_assets(tmp_path):
    from pxr import Sdf

    from tests.compile.test_command import MINIMAL_MODEL
    from usdcraft.compile.command import compile_model
    from usdcraft.reconstruction.diagnostic_binding import usd_fingerprint

    model = tmp_path / "model.py"
    model.write_text(MINIMAL_MODEL)
    for name in ("first", "second"):
        compile_model(model, tmp_path / name, render=False)
    first = tmp_path / "first/isaac/model.usdc"
    second = tmp_path / "second/isaac/model.usdc"
    assert usd_fingerprint(first) is not None
    assert usd_fingerprint(first) == usd_fingerprint(second)
    texture = second.parent / "texture.png"
    texture.write_bytes(b"first texture")
    stage = Usd.Stage.Open(str(second))
    prim = stage.GetDefaultPrim()
    prim.CreateAttribute("testAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("texture.png"))
    stage.GetRootLayer().Save()
    with_texture = usd_fingerprint(second)
    assert with_texture != usd_fingerprint(first)
    texture.write_bytes(b"changed texture")
    assert usd_fingerprint(second) != with_texture


@pytest.mark.parametrize("face", ["-X", "+X", "-Y", "+Y", "-Z", "+Z"])
def test_local_diagnostic_roi_is_a_valid_mesh_query(face):
    import copy
    import shlex

    from usdcraft.reconstruction.author_tools import _bounds
    from usdcraft.reconstruction.local_surface_audit import format_local_surfaces

    row = {
        "anchor": "feature:0",
        "face": face,
        "center_uv_m": [0.0123456, 0.1234567],
        "size_uv_m": [0.02, 0.03],
        "status": "unassessed",
        "reason": "insufficient_source_surface",
    }
    audit = {"regions": [row]}
    before = copy.deepcopy(audit)
    lines = format_local_surfaces(audit, source_bounds=[-1, -2, -3, 1, 2, 3])
    query = shlex.split(next(line for line in lines if "ROI --face=" in line))
    values = [float(x) for x in query[query.index("--bounds") + 1 :]]
    assert _bounds(values) is not None
    depth_axis = "XYZ".index(face[-1])
    assert values[depth_axis] == -(depth_axis + 1)
    assert values[depth_axis + 3] == depth_axis + 1
    assert all(values[i + 3] - values[i] <= 0.031 for i in range(3) if i != depth_axis)
    assert audit == before


def test_host_source_render_uses_staged_carrier_without_candidate(tmp_path, monkeypatch):
    from usdcraft.reconstruction import author_tools
    from usdcraft.reconstruction.render_service import request_source_render

    monkeypatch.setenv("USDCRAFT_RECONSTRUCTION_WORKSPACE", str(tmp_path))
    seen = []

    def fake(asset_dir, **kwargs):
        seen.append((asset_dir, kwargs))
        assert "joint_positions" not in kwargs
        return {"ok": True, "images": [], "backend": "test"}

    monkeypatch.setattr(author_tools, "render_source", fake)
    with CandidateRenderService(tmp_path):
        result = request_source_render(
            tmp_path / "inputs/source_render",
            out_dir=tmp_path / "author_feedback/source_001",
            azimuth=135,
            elevation=25,
        )
        assert result["ok"] and result["execution"] == "reconstruction_host"
        assert seen[0][1]["azimuth"] == 135 and seen[0][1]["elevation"] == 25
        with pytest.raises(ValueError, match="staged source carrier"):
            request_source_render(
                tmp_path / "build", out_dir=tmp_path / "author_feedback/bad", azimuth=0, elevation=0
            )
