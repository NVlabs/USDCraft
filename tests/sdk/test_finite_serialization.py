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

from collections.abc import Callable
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

import sdk._core._usd_validate as usd_validate
from sdk import (
    ArticulatedObject,
    ArticulationType,
    Box,
    Cylinder,
    Inertia,
    Inertial,
    JointDrive,
    Material,
    Mesh,
    Mimic,
    MotionLimits,
    MotionProperties,
    NewtonJointSolverTuning,
    Origin,
    PassiveJointSpring,
    PhysxJointFriction,
    Sphere,
    ValidationError,
)
from sdk._core._urdf_export import _format_float

_NONFINITE = float("nan")


@pytest.mark.parametrize(
    "factory",
    (
        lambda: Origin(xyz=(_NONFINITE, 0.0, 0.0)),
        lambda: Origin(rpy=(0.0, _NONFINITE, 0.0)),
        lambda: Box((_NONFINITE, 1.0, 1.0)),
        lambda: Cylinder(radius=_NONFINITE, length=1.0),
        lambda: Cylinder(radius=1.0, length=_NONFINITE),
        lambda: Sphere(radius=_NONFINITE),
        lambda: Mesh(name="mesh", scale=(1.0, _NONFINITE, 1.0)),
        lambda: Mesh(
            name="mesh",
            source_geometry=Box((1.0, 1.0, 1.0)),
            source_transform=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, _NONFINITE),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        lambda: Material("paint", rgba=(_NONFINITE, 0.2, 0.3, 1.0)),
        lambda: Material("paint", emissive=(0.1, _NONFINITE, 0.3)),
        lambda: Material("paint", metallic=_NONFINITE),
        lambda: Material("paint", ior=_NONFINITE),
        lambda: Inertia(_NONFINITE, 0.0, 0.0, 1.0, 0.0, 1.0),
        lambda: Inertial(
            mass=_NONFINITE,
            inertia=Inertia(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        ),
        lambda: Mimic("driver", multiplier=_NONFINITE),
        lambda: Mimic("driver", offset=_NONFINITE),
        lambda: NewtonJointSolverTuning(solref_friction=(_NONFINITE, 1.0)),
        lambda: NewtonJointSolverTuning(solimp_friction=(0.9, 0.95, _NONFINITE, 0.5, 2.0)),
        lambda: PassiveJointSpring(stiffness=_NONFINITE),
        lambda: PassiveJointSpring(stiffness=1.0, reference=_NONFINITE),
    ),
)
def test_serialized_sdk_fields_reject_nonfinite_values_at_construction(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValidationError, match="finite"):
        factory()


def _strict_model() -> ArticulatedObject:
    model = ArticulatedObject("finite_contract")
    material = model.material(
        "paint",
        rgba=(0.1, 0.2, 0.3, 1.0),
        metallic=0.2,
        roughness=0.5,
        emissive=(0.0, 0.0, 0.0),
        ior=1.45,
    )
    model.physics_material("contact", static_friction=0.8, dynamic_friction=0.6)
    base = model.part(
        "base",
        inertial=Inertial(
            mass=1.0,
            inertia=Inertia(0.1, 0.0, 0.0, 0.1, 0.0, 0.1),
            origin=Origin(),
        ),
    )
    base.visual(Box((0.2, 0.2, 0.2)), material=material, origin=Origin())
    base.visual(
        Mesh(
            name="procedural",
            scale=(1.0, 1.0, 1.0),
            source_geometry=Box((0.1, 0.1, 0.1)),
            source_transform=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        collidable=False,
    )
    middle = model.part("middle")
    middle.visual(Box((0.1, 0.1, 0.1)))
    tip = model.part("tip")
    tip.visual(Box((0.1, 0.1, 0.1)))
    model.articulation(
        "driver",
        ArticulationType.REVOLUTE,
        base,
        middle,
        origin=Origin(),
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(effort=2.0, velocity=1.0, lower=-0.5, upper=0.5),
        motion_properties=MotionProperties(damping=0.1, friction=0.05),
        newton_joint_solver_tuning=NewtonJointSolverTuning(
            solref_friction=(0.01, 1.0),
            solimp_friction=(0.99, 0.99, 0.001, 0.5, 2.0),
        ),
        physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(0.1),
        drive=JointDrive(stiffness=1.0, damping=0.1, max_force=2.0),
    )
    model.articulation(
        "follower",
        ArticulationType.REVOLUTE,
        middle,
        tip,
        motion_limits=MotionLimits(effort=2.0, velocity=1.0, lower=-0.5, upper=0.5),
        passive_spring=PassiveJointSpring(stiffness=1.0, reference=0.0),
        mimic=Mimic("driver"),
    )
    return model


def _set_frozen(target: object, field_name: str, value: object) -> None:
    object.__setattr__(target, field_name, value)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda model: _set_frozen(model.parts[0].visuals[0].geometry, "size", (_NONFINITE, 1, 1)),
        lambda model: _set_frozen(
            model.parts[0].visuals[0].origin,
            "rpy",
            (0.0, _NONFINITE, 0.0),
        ),
        lambda model: _set_frozen(
            model.parts[0].visuals[1].geometry,
            "scale",
            (1.0, _NONFINITE, 1.0),
        ),
        lambda model: _set_frozen(
            model.parts[0].visuals[1].geometry,
            "source_transform",
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, _NONFINITE),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        lambda model: setattr(model.materials[0], "rgba", (_NONFINITE, 0.2, 0.3, 1.0)),
        lambda model: setattr(model.materials[0], "emissive", (0.0, _NONFINITE, 0.0)),
        lambda model: setattr(model.materials[0], "ior", _NONFINITE),
        lambda model: _set_frozen(model.physics_materials[0], "density", _NONFINITE),
        lambda model: setattr(model.parts[0].inertial, "mass", _NONFINITE),
        lambda model: _set_frozen(model.parts[0].inertial.inertia, "ixx", _NONFINITE),
        lambda model: _set_frozen(model.parts[0].inertial.origin, "xyz", (_NONFINITE, 0, 0)),
        lambda model: _set_frozen(model.articulations[0].origin, "xyz", (_NONFINITE, 0, 0)),
        lambda model: setattr(model.articulations[0], "axis", (0.0, _NONFINITE, 1.0)),
        lambda model: _set_frozen(model.articulations[0].motion_limits, "effort", _NONFINITE),
        lambda model: _set_frozen(
            model.articulations[0].motion_properties,
            "damping",
            _NONFINITE,
        ),
        lambda model: _set_frozen(model.articulations[0].drive, "stiffness", _NONFINITE),
        lambda model: _set_frozen(
            model.articulations[0].newton_joint_solver_tuning,
            "solref_friction",
            (_NONFINITE, 1.0),
        ),
        lambda model: _set_frozen(
            model.articulations[1].passive_spring,
            "stiffness",
            _NONFINITE,
        ),
        lambda model: _set_frozen(
            model.articulations[0].physx_joint_resistance,
            "static_friction_effort",
            _NONFINITE,
        ),
        lambda model: _set_frozen(model.articulations[1].mimic, "multiplier", _NONFINITE),
    ),
)
def test_strict_object_validation_rejects_mutated_nonfinite_serialized_fields(
    mutate: Callable[[ArticulatedObject], None],
) -> None:
    model = _strict_model()
    mutate(model)

    with pytest.raises(ValidationError, match="finite"):
        model.validate(strict=True)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_urdf_numeric_formatter_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValidationError, match="URDF numeric value must be finite"):
        _format_float(value)


def _write_numeric_usd(path: Path, *, custom_data: bool) -> None:
    stage = Usd.Stage.CreateNew(path.as_posix())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    if custom_data:
        root.GetPrim().SetCustomDataByKey(
            "usdcraft:mimic",
            {"multiplier": float("nan")},
        )
    else:
        root.GetPrim().CreateAttribute(
            "usdcraft:numericProbe",
            Sdf.ValueTypeNames.Point3fArray,
        ).Set(
            Vt.Vec3fArray(
                (
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(1.0, 1.0, 1.0),
                    Gf.Vec3f(2.0, float("inf"), 2.0),
                )
            )
        )
    stage.GetRootLayer().Save()


@pytest.mark.parametrize("custom_data", (False, True), ids=("attribute", "custom-data"))
def test_strict_usd_gate_rejects_nonfinite_authored_values(
    tmp_path: Path,
    custom_data: bool,
) -> None:
    path = tmp_path / "nonfinite.usda"
    _write_numeric_usd(path, custom_data=custom_data)

    with pytest.raises(ValidationError, match="non-finite numeric value"):
        usd_validate._validate_usd_asset(path)


@pytest.mark.parametrize(
    ("type_name", "value"),
    (
        (Sdf.ValueTypeNames.TimeCode, Sdf.TimeCode(float("nan"))),
        (
            Sdf.ValueTypeNames.TimeCodeArray,
            Sdf.TimeCodeArray((Sdf.TimeCode(0.0), Sdf.TimeCode(float("inf")))),
        ),
    ),
    ids=("timecode", "timecode-array"),
)
def test_strict_usd_gate_rejects_nonfinite_authored_timecodes(
    tmp_path: Path,
    type_name: Sdf.ValueTypeName,
    value: object,
) -> None:
    path = tmp_path / "nonfinite-timecode.usda"
    stage = Usd.Stage.CreateNew(path.as_posix())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().CreateAttribute("usdcraft:timecodeProbe", type_name).Set(value)
    stage.GetRootLayer().Save()

    with pytest.raises(ValidationError, match="non-finite numeric value"):
        usd_validate._validate_usd_asset(path)


def test_dense_mesh_attribute_validation_uses_the_vectorized_path(tmp_path: Path) -> None:
    # A dense mesh points attribute must validate in bulk; the per-element recursion took
    # multiple seconds per 100k points and stalled Author no-render compiles past their timeout.
    import time

    import numpy as np

    points = Vt.Vec3fArray.FromNumpy(np.random.rand(300_000, 3).astype(np.float32))
    started = time.perf_counter()
    usd_validate._require_finite_usd_value(
        points,
        location="points",
        stage_path=tmp_path / "dense.usda",
    )
    assert time.perf_counter() - started < 1.0

    corrupted = np.random.rand(200_000, 3).astype(np.float32)
    corrupted[123_456, 2] = float("nan")
    with pytest.raises(ValidationError, match=r"points\[123456\]\[2\]"):
        usd_validate._require_finite_usd_value(
            Vt.Vec3fArray.FromNumpy(corrupted),
            location="points",
            stage_path=tmp_path / "dense.usda",
        )

    usd_validate._require_finite_usd_value(
        Vt.IntArray(list(range(1_000))),
        location="indices",
        stage_path=tmp_path / "dense.usda",
    )
