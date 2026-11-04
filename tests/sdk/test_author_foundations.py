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

import re
from dataclasses import replace
from pathlib import Path

import pytest

import sdk
from usdcraft.compile.command import compile_model

DOCS = Path(__file__).resolve().parents[2] / "sdk/_docs/common"


def python_blocks(name: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", (DOCS / name).read_text(), re.DOTALL)


def test_preloaded_quickstart_is_an_executable_physical_asset(tmp_path: Path) -> None:
    source = python_blocks("00_quickstart.md")[0]
    namespace: dict = {}
    exec(compile(source, "00_quickstart.md", "exec"), namespace)
    model = namespace["object_model"]
    assert model.base_mode is sdk.BaseMode.FLOATING
    assert len(model.parts) == 1 and not model.articulations
    body = model.parts[0]
    assert body.inertial.mass == pytest.approx(0.20 * 0.12 * 0.04 * 650)
    assert body.rigid_body.physics_material == "wood_contact"
    assert namespace["run_tests"]().passed
    source_path = tmp_path / "model.py"
    source_path.write_text(source)
    result = compile_model(
        source_path, tmp_path / "build", render=False, require_generation_root_floating=True
    )
    assert result.usdc_path.is_file() and result.urdf_path.is_file()


def test_assembly_joint_fragment_uses_public_api() -> None:
    model = sdk.ArticulatedObject("drawer_example")
    namespace = {
        "model": model,
        "cabinet": model.part("cabinet"),
        "drawer": model.part("drawer"),
    }
    exec(
        compile(python_blocks("30_articulated_object.md")[0], "30_articulated_object.md", "exec"),
        namespace,
    )
    joint = namespace["slide"]
    assert joint.motion_limits.upper == pytest.approx(0.28)
    assert joint.axis == (1.0, 0.0, 0.0)
    assert joint.physx_joint_resistance is not None
    assert joint.meta["gravity_hold"] is False and joint.drive is None


def test_testing_nested_fit_fragment_checks_real_pose_geometry() -> None:
    model = sdk.ArticulatedObject("nested_fit")
    outer = model.part("outer")
    inner = model.part("inner")
    outer.visual(sdk.Box((0.04, 0.04, 0.2)), name="outer_sleeve")
    inner.visual(sdk.Box((0.02, 0.02, 0.16)), name="inner_member")
    joint = model.articulation(
        "slide",
        sdk.ArticulationType.PRISMATIC,
        outer,
        inner,
        axis=(0, 0, 1),
        motion_limits=sdk.MotionLimits(lower=0, upper=0.1, effort=5, velocity=0.2),
    )
    ctx = sdk.TestContext(model)
    namespace = {
        "ctx": ctx,
        "outer_stage": outer,
        "inner_stage": inner,
        "slide_joint": joint,
        "slide_upper": 0.1,
    }
    source = python_blocks("80_testing.md")[-1]
    exec(compile(source, "80_testing.md", "exec"), namespace)
    assert ctx.report().passed
    # A member that exits its sleeve must fail the retained-insertion assertion.
    namespace["ctx"] = sdk.TestContext(model)
    namespace["slide_upper"] = 0.19
    joint.motion_limits = replace(joint.motion_limits, upper=0.19)
    exec(compile(source, "80_testing.md", "exec"), namespace)
    assert not namespace["ctx"].report().passed


def test_geometry_inspection_fragment_uses_real_queries() -> None:
    model = sdk.ArticulatedObject("inspection")
    cabinet = model.part("cabinet")
    drawer = model.part("drawer")
    cabinet.visual(sdk.Box((0.4, 0.3, 0.2)))
    drawer.visual(sdk.Box((0.2, 0.1, 0.06)), name="front_panel")
    model.articulation(
        "drawer_slide",
        sdk.ArticulationType.PRISMATIC,
        cabinet,
        drawer,
        axis=(1, 0, 0),
        motion_limits=sdk.MotionLimits(lower=0, upper=0.15, effort=10, velocity=0.1),
    )
    namespace = {"object_model": model}
    exec(python_blocks("70_probe_tooling.md")[0], namespace)
    assert namespace["open_bounds"][0][0] - namespace["rest_bounds"][0][0] == pytest.approx(0.1)
    assert namespace["panel_bounds"] == namespace["open_bounds"]
    assert namespace["ctx"].part_world_aabb("drawer") == namespace["rest_bounds"]


def test_cadquery_attachment_example_preserves_part_frame(tmp_path: Path) -> None:
    path = DOCS.parent / "cadquery/35_cadquery.md"
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL)
    model = sdk.ArticulatedObject("cadquery", assets=sdk.AssetContext(tmp_path))
    door = model.part("door")
    exec(blocks[-1], {"door": door})
    lo, hi = sdk.TestContext(model).part_world_aabb(door)
    assert tuple(b - a for a, b in zip(lo, hi)) == pytest.approx((0.58, 0.02, 0.78))


def test_cadquery_assembly_example_exports_distinct_located_meshes(tmp_path: Path) -> None:
    path = DOCS.parent / "cadquery/39_cadquery_assembly.md"
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL)
    namespace = {"__file__": str(tmp_path / "assembly.py")}
    exec(blocks[0], namespace)
    assert len(namespace["meshes"]) == 2
    assembly = namespace["assembly"]
    assert assembly.objects["boss"].loc.toTuple()[0] == pytest.approx((0.03, 0, 0.005))
    exec(blocks[1], namespace)
    assert assembly.toCompound().isValid()


def test_quickstart_joint_fragment_uses_documented_frames() -> None:
    # Supply the existing model/Parts and geometry inputs documented for the joint fragment.
    source, joint_code = python_blocks("00_quickstart.md")
    namespace = {}
    exec(compile(source, "00_quickstart.md", "exec"), namespace)
    model = namespace["object_model"]
    shape = sdk.Box((0.18, 0.02, 0.30))
    door = model.part(
        "door",
        inertial=sdk.Inertial.from_geometry(shape, sdk.estimate_mass_from_density(shape, 650)),
        rigid_body=sdk.RigidBodyProperties(physics_material="wood_contact"),
    )
    door.visual(shape, origin=sdk.Origin(xyz=(0.09, 0, 0.15)), name="panel")
    namespace.update(model=model, hinge_xyz=(0.10, 0.0, 0.04), door_handle_radius=0.16)
    exec(compile(joint_code, "00_quickstart.md", "exec"), namespace)
    ctx = sdk.TestContext(model)
    with ctx.pose(door_hinge=1.0):
        bounds = ctx.part_world_aabb("door")
        assert bounds is not None and bounds[1][1] > 0.1
    assert model.articulations[0].physx_joint_resistance.static_friction_effort == 0.32
