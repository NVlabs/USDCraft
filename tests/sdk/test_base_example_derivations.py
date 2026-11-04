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

import ast
import math
import re
from pathlib import Path

import pytest
from pxr import Usd, UsdPhysics

import sdk
from usdcraft.compile.command import compile_model

_PYTHON_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_SIMULATOR_READY_PHYSICS_EXAMPLES = (
    (
        "sdk/_examples/base/simulator_ready_fixed_appliance_door_and_knobs.md",
        sdk.BaseMode.FLOATING,
        6,
    ),
    (
        "sdk/_examples/base/simulator_ready_floating_carton_with_flaps.md",
        sdk.BaseMode.FLOATING,
        5,
    ),
    (
        "sdk/_examples/base/simulator_ready_prismatic_drawer.md",
        sdk.BaseMode.FLOATING,
        2,
    ),
    (
        "sdk/downstream_examples/simulator_ready_reduced_order_swing_carrier.md",
        sdk.BaseMode.FIXED,
        1,
    ),
)
_FREE_BODY_RELATIONSHIP_EXAMPLES = (
    (
        "sdk/_examples/base/stocked_supermarket_gondola_shelf.md",
        sdk.BaseMode.FLOATING,
        9,
    ),
    (
        "sdk/_examples/base/simulator_ready_freestanding_table_with_two_chairs.md",
        sdk.BaseMode.FLOATING,
        3,
    ),
)
_MASS_DERIVATION_EXAMPLES = (
    *(path for path, _, _ in _SIMULATOR_READY_PHYSICS_EXAMPLES),
    *(path for path, _, _ in _FREE_BODY_RELATIONSHIP_EXAMPLES),
    "sdk/downstream_examples/simulator_ready_segmented_chain_with_payload.md",
)
_THIN_OR_HOLLOW_MASS_EXAMPLES = (
    "sdk/_examples/base/simulator_ready_fixed_appliance_door_and_knobs.md",
    "sdk/_examples/base/simulator_ready_floating_carton_with_flaps.md",
    "sdk/downstream_examples/simulator_ready_reduced_order_swing_carrier.md",
    "sdk/downstream_examples/simulator_ready_segmented_chain_with_payload.md",
    *(path for path, _, _ in _FREE_BODY_RELATIONSHIP_EXAMPLES),
)
_EXAMPLE_PATHS = (
    "sdk/_examples/base/jet_engine_with_smooth_nacelle_dense_front_fan.md",
    "sdk/_examples/base/atv_quad_bike_with_front_steering_and_suspension.md",
    "sdk/_examples/base/artisan_stand_mixer_with_tilt_head_balloon_whisk_and_articulated_controls.md",
    "sdk/_examples/base/rustic_ceramic_mug_with_glazed_finish.md",
    "sdk/_examples/base/glass_storage_jar_with_metal_lid.md",
    "sdk/_examples/base/labeled_spray_bottle_with_decal.md",
    "sdk/downstream_examples/simulator_ready_segmented_chain_with_payload.md",
    *(path for path, _, _ in _FREE_BODY_RELATIONSHIP_EXAMPLES),
    *(path for path, _, _ in _SIMULATOR_READY_PHYSICS_EXAMPLES),
)
_REFERENCE_MASS_RANGES = (
    (
        "sdk/_examples/base/simulator_ready_fixed_appliance_door_and_knobs.md",
        {"appliance_body": (2.0, 30.0), "drop_door": (0.2, 5.0)},
    ),
    (
        "sdk/_examples/base/simulator_ready_floating_carton_with_flaps.md",
        {"carton_body": (0.1, 3.0), "front_flap": (0.01, 0.5)},
    ),
    (
        "sdk/_examples/base/simulator_ready_prismatic_drawer.md",
        {"cabinet_carcass": (8.0, 60.0), "drawer": (1.0, 15.0)},
    ),
    (
        "sdk/downstream_examples/simulator_ready_reduced_order_swing_carrier.md",
        {"support_beam": (1.0, 15.0), "empty_swing_carrier": (0.3, 5.0)},
    ),
    (
        "sdk/downstream_examples/simulator_ready_segmented_chain_with_payload.md",
        {"support": (1.0, 15.0), "load_chain_link_000": (0.005, 0.5), "payload": (0.05, 5.0)},
    ),
    (
        "sdk/_examples/base/stocked_supermarket_gondola_shelf.md",
        {
            "gondola": (20.0, 60.0),
            "can_01": (0.02, 0.30),
            "carton_01": (0.02, 0.50),
            "jar_01": (0.05, 0.80),
        },
    ),
    (
        "sdk/_examples/base/simulator_ready_freestanding_table_with_two_chairs.md",
        {"table": (10.0, 40.0), "chair_01": (3.0, 15.0)},
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _extract_python_fence(path: Path) -> str:
    match = _PYTHON_FENCE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"No Python fence found in {path}")
    return match.group(1)


def _assert_exported_simulator_physics(
    object_model: sdk.ArticulatedObject,
    usdc_path: Path,
    *,
    expected_movable_count: int,
) -> None:
    stage = Usd.Stage.Open(str(usdc_path))
    assert stage is not None

    physics_material_names = {material.name for material in object_model.physics_materials}
    for material in object_model.physics_materials:
        material_prim = stage.GetPrimAtPath(f"/Asset/PhysicsMaterials/{material.name}")
        assert material_prim.HasAPI(UsdPhysics.MaterialAPI)
        material_api = UsdPhysics.MaterialAPI(material_prim)
        assert material_api.GetStaticFrictionAttr().Get() == pytest.approx(material.static_friction)
        assert material_api.GetDynamicFrictionAttr().Get() == pytest.approx(
            material.dynamic_friction
        )
        assert material_api.GetRestitutionAttr().Get() == pytest.approx(material.restitution)
        if material.density is not None:
            assert material_api.GetDensityAttr().Get() == pytest.approx(material.density)

    child_names = {joint.child for joint in object_model.articulations}
    movable_names = {
        part.name
        for part in object_model.parts
        if part.name in child_names or object_model.base_mode == sdk.BaseMode.FLOATING
    }
    assert len(movable_names) == expected_movable_count

    for part in object_model.parts:
        part_prim = stage.GetPrimAtPath(f"/Asset/Links/{part.name}")
        assert part_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert part_prim.HasAPI(UsdPhysics.MassAPI)
        assert UsdPhysics.MassAPI(part_prim).GetMassAttr().Get() == pytest.approx(
            part.inertial.mass
        )

        collision_prims = [
            prim
            for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(part_prim.GetPath())
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        assert collision_prims
        expected_material = part.rigid_body.physics_material
        assert expected_material in physics_material_names
        for collision_prim in collision_prims:
            targets = collision_prim.GetRelationship("material:binding:physics").GetTargets()
            assert [str(target) for target in targets] == [
                f"/Asset/PhysicsMaterials/{expected_material}"
            ]

    scalar_types = {
        sdk.ArticulationType.REVOLUTE,
        sdk.ArticulationType.CONTINUOUS,
        sdk.ArticulationType.PRISMATIC,
    }
    scalar_joints = [
        joint for joint in object_model.articulations if joint.articulation_type in scalar_types
    ]
    floating_joints = [
        joint
        for joint in object_model.articulations
        if joint.articulation_type == sdk.ArticulationType.FLOATING
    ]
    for joint in floating_joints:
        assert not stage.GetPrimAtPath(f"/Asset/Joints/{joint.name}")
    for joint in scalar_joints:
        joint_prim = stage.GetPrimAtPath(f"/Asset/Joints/{joint.name}")
        angular = joint.articulation_type in {
            sdk.ArticulationType.REVOLUTE,
            sdk.ArticulationType.CONTINUOUS,
        }
        drive_name = "angular" if angular else "linear"
        applied_schemas = joint_prim.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
        assert f"PhysicsDriveAPI:{drive_name}" in applied_schemas

        drive_api = UsdPhysics.DriveAPI.Get(joint_prim, drive_name)
        expected_damping = (
            joint.drive.damping if joint.drive is not None else joint.motion_properties.damping
        )
        expected_stiffness = joint.drive.stiffness if joint.drive is not None else 0.0
        if angular:
            expected_damping *= math.pi / 180.0
            expected_stiffness *= math.pi / 180.0
        assert drive_api.GetDampingAttr().Get() == pytest.approx(expected_damping)
        assert drive_api.GetStiffnessAttr().Get() == pytest.approx(expected_stiffness)
        assert drive_api.GetMaxForceAttr().Get() == pytest.approx(
            joint.drive.max_force
            if joint.drive is not None and joint.drive.max_force is not None
            else joint.motion_limits.effort
        )

        effort_friction = isinstance(
            joint.physx_joint_resistance,
            sdk.PhysxJointFriction,
        )
        if effort_friction:
            assert f"PhysxJointAxisAPI:{drive_name}" in applied_schemas
            namespace = f"physxJointAxis:{drive_name}"
            velocity_attr = joint_prim.GetAttribute(f"{namespace}:maxJointVelocity")
            friction_attr = joint_prim.GetAttribute(f"{namespace}:staticFrictionEffort")
            assert friction_attr.Get() == pytest.approx(
                joint.physx_joint_resistance.static_friction_effort
            )
        else:
            assert "PhysxJointAPI" in applied_schemas
            velocity_attr = joint_prim.GetAttribute("physxJoint:maxJointVelocity")
            friction_attr = joint_prim.GetAttribute("physxJoint:jointFriction")
            expected_friction = (
                joint.physx_joint_resistance.coefficient
                if isinstance(
                    joint.physx_joint_resistance,
                    sdk.PhysxJointFrictionCoefficient,
                )
                else joint.motion_properties.friction
            )
            assert friction_attr.Get() == pytest.approx(expected_friction)
        assert velocity_attr.IsValid() and not velocity_attr.IsCustom()
        assert friction_attr.IsValid() and not friction_attr.IsCustom()
        expected_velocity = joint.motion_limits.velocity
        if angular:
            expected_velocity = math.degrees(expected_velocity)
        assert velocity_attr.Get() == pytest.approx(expected_velocity)


@pytest.mark.slow
@pytest.mark.parametrize("relative_path", _EXAMPLE_PATHS)
def test_base_example_executes_and_passes_sdk_checks(tmp_path: Path, relative_path: str) -> None:
    path = _repo_root() / relative_path
    namespace = {"__file__": str(tmp_path / f"{path.stem}.py")}

    exec(_extract_python_fence(path), namespace)

    object_model = namespace["object_model"]

    ctx = sdk.TestContext(object_model)
    ctx.check_model_valid()
    ctx.check_mesh_assets_ready()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("relative_path", "expected_base_mode", "expected_movable_count"),
    _FREE_BODY_RELATIONSHIP_EXAMPLES,
)
def test_free_body_relationship_examples_compile_to_independent_usd_rigid_bodies(
    tmp_path: Path,
    relative_path: str,
    expected_base_mode: sdk.BaseMode,
    expected_movable_count: int,
) -> None:
    path = _repo_root() / relative_path
    namespace = {"__file__": str(tmp_path / f"{path.stem}.py")}

    exec(_extract_python_fence(path), namespace)

    object_model = namespace["object_model"]
    assert object_model.base_mode == expected_base_mode
    assert object_model.meta["mode"] == "static"
    assert object_model.articulations
    assert all(
        joint.articulation_type == sdk.ArticulationType.FLOATING
        and joint.motion_limits is None
        and joint.motion_properties is None
        and joint.drive is None
        for joint in object_model.articulations
    )
    physics_material_names = {material.name for material in object_model.physics_materials}
    assert all(
        part.body_model == sdk.BodyModel.RIGID
        and part.inertial is not None
        and part.inertial.mass > 0.0
        and part.rigid_body is not None
        and part.rigid_body.physics_material in physics_material_names
        and any(visual.collidable for visual in part.visuals)
        for part in object_model.parts
    )

    example_report = namespace["run_tests"]()
    assert example_report.passed, example_report.failures
    if relative_path.endswith("stocked_supermarket_gondola_shelf.md"):
        stock_names = namespace["STOCK_NAMES"]
        assert len(stock_names) == expected_movable_count - 1 == 8
        assert {joint.child for joint in object_model.articulations} == set(stock_names)
        assert example_report.allowed_isolated_parts == ()
    else:
        assert tuple(part.name for part in object_model.parts) == (
            "table",
            "chair_01",
            "chair_02",
        )
        assert {joint.child for joint in object_model.articulations} == {
            "chair_01",
            "chair_02",
        }
        assert example_report.allowed_isolated_parts == ("chair_01", "chair_02")

    model_path = tmp_path / f"{path.stem}.py"
    model_path.write_text(_extract_python_fence(path), encoding="utf-8")
    artifacts = compile_model(model_path, tmp_path / f"{path.stem}_build", render=False)
    _assert_exported_simulator_physics(
        object_model,
        artifacts.usdc_path,
        expected_movable_count=expected_movable_count,
    )
    if relative_path.endswith("simulator_ready_fixed_appliance_door_and_knobs.md"):
        assert artifacts.newton_usdc_path is not None
        newton = Usd.Stage.Open(artifacts.newton_usdc_path.as_posix())
        assert newton is not None
        door_joint = newton.GetPrimAtPath("/Asset/Joints/body_to_drop_door")
        assert list(door_joint.GetAttribute("mjc:solimpfriction").Get()) == pytest.approx(
            [0.999, 0.9999, 0.001, 0.5, 2.0]
        )
        assert not door_joint.GetAttribute("mjc:solreffriction")

    stage = Usd.Stage.Open(str(artifacts.usdc_path))
    assert stage is not None
    for joint in object_model.articulations:
        assert not stage.GetPrimAtPath(f"/Asset/Joints/{joint.name}")
    fixed_root = UsdPhysics.FixedJoint.Get(stage, "/Asset/Joints/__root_fixed")
    assert bool(fixed_root) is (expected_base_mode == sdk.BaseMode.FIXED)


@pytest.mark.parametrize(
    "relative_path",
    _MASS_DERIVATION_EXAMPLES,
)
def test_simulator_ready_examples_derive_mass_from_density(relative_path: str) -> None:
    source = _extract_python_fence(_repo_root() / relative_path)
    tree = ast.parse(source)

    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "estimate_mass_from_density"
    ]
    assert helper_calls

    bare_mass_literals = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg in {"mass", "segment_mass"} and isinstance(keyword.value, ast.Constant)
    ]
    assert not bare_mass_literals


@pytest.mark.parametrize("relative_path", _THIN_OR_HOLLOW_MASS_EXAMPLES)
def test_thin_or_hollow_examples_state_fill_fraction(relative_path: str) -> None:
    source = _extract_python_fence(_repo_root() / relative_path)
    tree = ast.parse(source)

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "estimate_mass_from_density"
        and any(keyword.arg == "fill_fraction" for keyword in node.keywords)
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(("relative_path", "expected_ranges"), _REFERENCE_MASS_RANGES)
def test_simulator_ready_reference_masses_stay_in_broad_category_ranges(
    tmp_path: Path,
    relative_path: str,
    expected_ranges: dict[str, tuple[float, float]],
) -> None:
    path = _repo_root() / relative_path
    namespace = {"__file__": str(tmp_path / f"{path.stem}.py")}

    exec(_extract_python_fence(path), namespace)

    object_model = namespace["object_model"]
    for part_name, (minimum, maximum) in expected_ranges.items():
        inertial = object_model.get_part(part_name).inertial
        assert inertial is not None
        assert minimum < inertial.mass < maximum


@pytest.mark.slow
@pytest.mark.parametrize(
    ("relative_path", "expected_base_mode", "expected_movable_count"),
    _SIMULATOR_READY_PHYSICS_EXAMPLES,
)
def test_simulator_ready_examples_author_complete_rigid_body_physics(
    tmp_path: Path,
    relative_path: str,
    expected_base_mode: sdk.BaseMode,
    expected_movable_count: int,
) -> None:
    path = _repo_root() / relative_path
    namespace = {"__file__": str(tmp_path / f"{path.stem}.py")}

    exec(_extract_python_fence(path), namespace)

    object_model = namespace["object_model"]
    assert object_model.base_mode == expected_base_mode
    assert object_model.physics_materials
    physics_material_names = {material.name for material in object_model.physics_materials}
    assert all(
        part.body_model == sdk.BodyModel.RIGID
        and part.inertial is not None
        and part.inertial.mass > 0.0
        and part.rigid_body is not None
        and part.rigid_body.enabled
        and part.rigid_body.physics_material in physics_material_names
        for part in object_model.parts
    )
    child_names = {joint.child for joint in object_model.articulations}
    movable_parts = [
        part
        for part in object_model.parts
        if part.name in child_names or object_model.base_mode == sdk.BaseMode.FLOATING
    ]
    assert len(movable_parts) == expected_movable_count

    scalar_types = {
        sdk.ArticulationType.REVOLUTE,
        sdk.ArticulationType.CONTINUOUS,
        sdk.ArticulationType.PRISMATIC,
    }
    scalar_joints = [
        joint for joint in object_model.articulations if joint.articulation_type in scalar_types
    ]
    assert scalar_joints
    for joint in scalar_joints:
        assert joint.motion_limits is not None
        assert joint.motion_limits.effort > 0.0
        assert joint.motion_limits.velocity > 0.0
        if joint.articulation_type != sdk.ArticulationType.CONTINUOUS:
            assert joint.motion_limits.lower is not None
            assert joint.motion_limits.upper is not None
            assert joint.motion_limits.lower <= 0.0 <= joint.motion_limits.upper
        assert joint.motion_properties is not None
        assert joint.motion_properties.damping is not None
        assert joint.motion_properties.friction is not None
        assert joint.physx_joint_resistance is not None
        if joint.drive is not None:
            assert joint.motion_properties.damping == 0.0
            assert math.isfinite(joint.drive.damping)
            assert joint.drive.damping >= 0.0
            if joint.drive.max_force is not None:
                assert math.isfinite(joint.drive.max_force)
                assert 0.0 <= joint.drive.max_force <= joint.motion_limits.effort

    example_report = namespace["run_tests"]()
    assert example_report.passed, example_report.failures

    ctx = sdk.TestContext(object_model)
    ctx.check_model_valid()
    ctx.fail_if_articulation_origin_far_from_geometry()
    ctx.fail_if_part_contains_disconnected_geometry_islands(tol=1e-6)
    report = ctx.report()
    assert report.passed, report.failures

    model_path = tmp_path / "model.py"
    model_path.write_text(_extract_python_fence(path), encoding="utf-8")
    artifacts = compile_model(model_path, tmp_path / "build", render=False)
    assert artifacts.urdf_path.is_file()
    assert artifacts.usdc_path.is_file()
    assert artifacts.usdz_path.is_file()
    _assert_exported_simulator_physics(
        object_model,
        artifacts.usdc_path,
        expected_movable_count=expected_movable_count,
    )
