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
from types import SimpleNamespace

import pytest
from PIL import Image
from pxr import Usd, UsdGeom

from usdcraft.pipeline.tools import ProviderToolResult
from usdcraft.reconstruction.author_tools import measure_mesh, render_candidate
from usdcraft.reconstruction.provider_tools import ReconstructionToolExecutor


def test_source_render_before_modeling_and_provider_delivery(tmp_path: Path, monkeypatch) -> None:
    from tests.reconstruction.test_reconstruction_provider import _provider_workspace

    _, _, _, _, workspace = _provider_workspace(tmp_path)
    carrier = workspace.root / "inputs" / "source_render"
    original = {p: p.read_bytes() for p in carrier.rglob("*") if p.is_file()}
    seen = []

    def render(request):
        seen.append(request)
        request.out_dir.mkdir(parents=True, exist_ok=True)
        image = request.out_dir / "source.png"
        Image.new("RGB", (8, 8)).save(image)
        return SimpleNamespace(image_paths=(image,), backend="test", capabilities={})

    monkeypatch.setattr("usdcraft.reconstruction.author_tools.render_generated_model_views", render)
    executor = ReconstructionToolExecutor(
        workspace,
        check_runner=lambda p: pytest.fail("source view needs no check"),
        compile_runner=lambda *a, **k: pytest.fail("source view needs no compile"),
    )
    result = executor.execute("render_source", {"azimuth": 225, "elevation": 15})
    assert result.payload["ok"] and result.image_paths == (seen[0].out_dir / "source.png",)
    assert seen[0].asset_dir == carrier and seen[0].views == (("source", 225, 15),)
    assert result.payload["source_carrier_glb_sha256"]
    assert executor.validation_state()["operations"]["render"]["status"] == "not_run"
    assert {p: p.read_bytes() for p in original} == original
    assert not executor.execute("render_source", {"azimuth": 0, "elevation": 100}).payload["ok"]
    assert len(seen) == 1


def _mesh(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Source")
    root.AddTranslateOp().Set((1, 2, 3))
    mesh = UsdGeom.Mesh.Define(stage, "/Source/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    stage.GetRootLayer().Save()
    return path


def test_mesh_region_reads_geometry_without_axial_and_keeps_world_frame(tmp_path: Path) -> None:
    source = _mesh(tmp_path / "source.usdc")
    original = source.read_bytes()
    result = measure_mesh(source, out_dir=tmp_path / "whole", max_points=2)
    assert result["min_xyz_m"] == [1, 2, 3]
    assert result["max_xyz_m"] == [2, 3, 4]
    assert len(result["points_xyz_m"]) == 2
    assert Image.open(result["images"][0]).size == (1536, 560)
    local = measure_mesh(source, out_dir=tmp_path / "slice", bounds=[0.9, 1.9, 2.9, 1.1, 3.1, 3.1])
    assert local["selected_vertex_count"] == 2
    assert local["size_xyz_m"] == [0, 1, 0]
    empty = measure_mesh(source, out_dir=tmp_path / "empty", bounds=[5, 5, 5, 6, 6, 6])
    assert empty["selected_vertex_count"] == 0 and empty["images"] == []
    assert empty["empty_is_absence_evidence"] is False
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="minima"):
        measure_mesh(source, out_dir=tmp_path / "invalid", bounds=[1, 0, 0, 0, 1, 1])
    with pytest.raises(ValueError, match="max_points"):
        measure_mesh(source, out_dir=tmp_path / "invalid", max_points=1000000)


def test_mesh_components_are_anonymous_world_space_and_region_scoped(
    tmp_path: Path, monkeypatch
) -> None:
    source = _mesh(tmp_path / "components.usdc")
    stage = Usd.Stage.Open(str(source))
    for i in range(20):
        mesh = UsdGeom.Mesh.Define(stage, f"/Source/private_part_name_{i}")
        mesh.CreatePointsAttr([(i + 10, 0, 0), (i + 10, 1, 0), (i + 10, 0, 1)])
    stage.GetRootLayer().Save()
    original = source.read_bytes()
    result = measure_mesh(source, out_dir=tmp_path / "whole")
    assert result["source_component_count"] == result["selected_component_count"] == 21
    assert result["returned_component_count"] == len(result["components"]) == 16
    detail = json.loads(Path(result["detail_path"]).read_text())
    assert len(detail["components"]) == detail["stored_component_count"] == 21
    assert not detail["components_truncated"]
    assert "private_part_name" not in json.dumps(detail)
    first = detail["components"][0]
    assert first["min_xyz_m"] == [1, 2, 3]
    assert first["max_xyz_m"] == [2, 3, 4]
    local = measure_mesh(
        source, out_dir=tmp_path / "slice", bounds=[10.9, 1.9, 2.9, 11.1, 2.1, 4.1]
    )
    assert local["source_component_count"] == 21
    assert local["selected_component_count"] == 1
    row = local["components"][0]
    assert row["id"] == detail["components"][1]["id"]
    assert row["source_vertex_count"] == 3 and row["selected_vertex_count"] == 2
    assert row["min_xyz_m"] == [11, 2, 3] and row["max_xyz_m"] == [11, 2, 4]
    monkeypatch.setattr("usdcraft.reconstruction.author_tools.MAX_QUERY_COMPONENTS", 2)
    bounded = measure_mesh(source, out_dir=tmp_path / "bounded")
    assert bounded["selected_component_count"] == 21
    assert bounded["stored_component_count"] == 2 and bounded["components_truncated"]
    assert source.read_bytes() == original


def test_candidate_camera_pose_and_unknown_joint(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "model.urdf").write_text('<robot name="test"><joint name="slide"/></robot>')
    seen = []

    def render(request):
        seen.append(request)
        request.out_dir.mkdir()
        image = request.out_dir / "image.png"
        Image.new("RGB", (8, 8)).save(image)
        return SimpleNamespace(image_paths=(image,), backend="test", capabilities={})

    monkeypatch.setattr("usdcraft.reconstruction.author_tools.render_generated_model_views", render)
    result = render_candidate(
        tmp_path,
        out_dir=tmp_path / "render",
        azimuth=-45,
        elevation=20,
        joint_positions={"slide": 0.1},
    )
    assert seen[0].joint_positions == {"slide": 0.1}
    assert seen[0].views == (("author", -45, 20),)
    assert result["scope"].endswith("not a physics simulation")
    with pytest.raises(ValueError, match="Unknown joints"):
        render_candidate(tmp_path, out_dir=tmp_path / "bad", joint_positions={"wrong": 0.1})


def test_provider_feedback_attaches_images_and_rejects_stale_compile(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.reconstruction.test_reconstruction_provider import _provider_workspace

    _, _, _, _, workspace = _provider_workspace(tmp_path)
    _mesh(workspace.source_mesh_path)
    executor = ReconstructionToolExecutor(
        workspace,
        check_runner=lambda p: ProviderToolResult.ok(),
        compile_runner=lambda p, o, render: ProviderToolResult.ok(),
    )
    result = executor.execute("measure_mesh", {"bounds": None, "max_points": 4})
    assert result.payload["ok"] and len(result.image_paths) == 1
    assert result.payload["components"][0]["min_xyz_m"] == [1, 2, 3]
    detail = executor.execute("read_file", {"path": result.payload["detail_path"]})
    assert detail.payload["ok"] and '"components"' in str(detail.payload)
    assert not executor.execute("render_candidate", {}).payload["ok"]
    executor.execute("compile_model", {})
    image = tmp_path / "render.png"
    Image.new("RGB", (8, 8)).save(image)
    monkeypatch.setattr(
        "usdcraft.reconstruction.author_tools.render_candidate",
        lambda *a, **k: {"ok": True, "images": [str(image)]},
    )
    assert executor.execute("render_candidate", {}).image_paths == (image,)
    executor.revision += 1
    assert not executor.execute("render_candidate", {}).payload["ok"]


def test_provider_requested_images_enter_author_history(tmp_path: Path) -> None:
    from tests.generation.provider.test_harness import _call, _ScriptedClient, _workspace
    from usdcraft.pipeline.provider.harness import ProviderHarness
    from usdcraft.pipeline.tools import ProviderToolExecution, ProviderToolExecutor

    workspace = _workspace(tmp_path)
    image = tmp_path / "candidate.png"
    Image.new("RGB", (8, 8)).save(image)

    class Executor(ProviderToolExecutor):
        def execute(self, name, arguments):
            if name == "render_candidate":
                return ProviderToolExecution({"ok": True}, image_paths=(image,))
            if name == "finish_generation":
                return ProviderToolExecution({"ok": True}, "complete", "inspected")
            return super().execute(name, arguments)

    client = _ScriptedClient(
        [
            [
                _call("view", "render_candidate", {}),
                _call("read", "read_file", {"path": "model.py"}),
            ],
            [_call("done", "finish_generation", {})],
        ]
    )
    executor = Executor(
        workspace,
        finish_tool_name="finish_generation",
        check_runner=lambda p: ProviderToolResult.ok(),
        compile_runner=lambda p, o, render: ProviderToolResult.ok(),
    )
    harness = ProviderHarness(
        progress_label="Provider Generation",
        workspace=workspace,
        client=client,
        tool_executor=executor,
        max_turns=3,
    )
    harness.run(prompt="Inspect and complete")
    history = client.inputs[1]
    assert history[-1]["role"] == "user"
    assert history[-1]["content"][1]["type"] == "input_image"
    assert history[-1]["content"][1]["image_url"].startswith("data:image/png;base64,")
    # Both tool results precede the user image message, preserving multi-call turn ordering.
    assert history[-2]["call_id"] == "read" and history[-3]["call_id"] == "view"
    for trace in workspace.root.glob("**/*.jsonl"):
        assert "data:image/png;base64," not in trace.read_text()


def test_codex_mesh_cli_returns_region_feedback_and_structured_errors(
    tmp_path: Path, capsys
) -> None:
    from usdcraft.cli import main

    source = _mesh(tmp_path / "source.usdc")
    args = ["recon-measure-mesh", str(source), "--output", str(tmp_path / "query")]
    assert main([*args, "--max-points", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_vertex_count"] == 4
    assert len(payload["points_xyz_m"]) == 2 and Path(payload["images"][0]).is_file()
    assert main([*args, "--bounds", "1", "0", "0", "0", "1", "1"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
