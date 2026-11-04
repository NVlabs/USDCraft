# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

import pytest

from sdk import (
    ArticulatedObject,
    ArticulationType,
    JointDrive,
    Mimic,
    MotionLimits,
    MotionProperties,
    NewtonJointSolverTuning,
    PassiveJointSpring,
    ValidationError,
)
from sdk._core._urdf_export import _articulation_element, compile_object_to_urdf_xml


@pytest.mark.parametrize("joint_type", ["planar", "d6"])
def test_unknown_articulation_type(joint_type: str) -> None:
    model = ArticulatedObject(name=f"{joint_type}_not_supported")
    base = model.part("base")
    slider = model.part("slider")

    with pytest.raises(ValidationError, match="Unknown articulation type"):
        model.articulation("base_to_slider", joint_type, parent=base, child=slider)


def _spherical_model(*, cone_deg: float | None = 45.0) -> ArticulatedObject:
    model = ArticulatedObject(name="spherical_joint")
    base = model.part("base")
    payload = model.part("payload")
    model.articulation(
        "base_to_payload",
        ArticulationType.SPHERICAL,
        parent=base,
        child=payload,
        axis=(0.0, 0.0, 4.0),
        cone_deg=cone_deg,
    )
    return model


def test_spherical_accepts_axis_and_cone() -> None:
    model = _spherical_model(cone_deg=45)

    model.validate(strict=True)

    joint = model.get_articulation("base_to_payload")
    assert joint.articulation_type == ArticulationType.SPHERICAL
    assert joint.axis == (0.0, 0.0, 4.0)
    assert joint.cone_deg == 45.0


def test_spherical_requires_cone_in_strict_mode() -> None:
    with pytest.raises(ValidationError, match="must set cone_deg"):
        _spherical_model(cone_deg=None).validate(strict=True)


@pytest.mark.parametrize("cone_deg", [0.0, -1.0, 180.0001, float("inf"), float("nan")])
def test_spherical_rejects_invalid_cone(cone_deg: float) -> None:
    with pytest.raises(ValidationError, match=r"cone_deg must be (?:finite|in \(0, 180\])"):
        _spherical_model(cone_deg=cone_deg).validate(strict=True)


def test_spherical_forbids_scalar_motion_limits() -> None:
    model = _spherical_model()
    model.get_articulation("base_to_payload").motion_limits = MotionLimits()

    with pytest.raises(ValidationError, match="does not support motion limits"):
        model.validate(strict=True)


def test_spherical_forbids_drive() -> None:
    model = _spherical_model()
    model.get_articulation("base_to_payload").drive = JointDrive()

    with pytest.raises(ValidationError, match="only supports drives"):
        model.validate(strict=True)


def test_spherical_forbids_mimic() -> None:
    model = _spherical_model()
    model.get_articulation("base_to_payload").mimic = Mimic(joint="other")

    with pytest.raises(ValidationError, match="only supports mimic"):
        model.validate(strict=True)


def test_non_spherical_forbids_cone() -> None:
    model = ArticulatedObject(name="scalar_with_cone")
    base = model.part("base")
    arm = model.part("arm")
    model.articulation(
        "base_to_arm",
        ArticulationType.REVOLUTE,
        parent=base,
        child=arm,
        cone_deg=30.0,
        motion_limits=MotionLimits(lower=-1.0, upper=1.0),
    )

    with pytest.raises(ValidationError, match="only supports cone_deg on spherical"):
        model.validate(strict=True)


def test_urdf_preserves_spherical_semantics_as_portable_floating_joint() -> None:
    model = _spherical_model(cone_deg=45.0)

    xml = compile_object_to_urdf_xml(model)
    elem = _articulation_element(model.get_articulation("base_to_payload"))

    assert 'type="floating"' in xml
    assert elem.attrib == {
        "name": "base_to_payload",
        "type": "floating",
        "usdcraft_type": "spherical",
        "usdcraft_axis": "0 0 1",
        "usdcraft_cone_deg": "45",
    }
    assert elem.find("axis") is None
    assert "mesh2usd_type" not in elem.attrib


def test_continuous_requires_motion_limits() -> None:
    model = ArticulatedObject(name="continuous_requires_limits")
    base = model.part("base")
    mast = model.part("mast")
    model.articulation("mast_slew", "continuous", parent=base, child=mast)

    with pytest.raises(
        ValidationError,
        match="must set motion_limits=MotionLimits\\(effort=..., velocity=...\\)",
    ):
        model.validate(strict=True)


def test_continuous_forbids_lower_upper_limits() -> None:
    model = ArticulatedObject(name="continuous_forbids_bounds")
    base = model.part("base")
    mast = model.part("mast")
    model.articulation(
        "mast_slew",
        "continuous",
        parent=base,
        child=mast,
        motion_limits=MotionLimits(
            effort=12.0,
            velocity=1.5,
            lower=0.0,
            upper=1.0,
        ),
    )

    with pytest.raises(ValidationError, match="cannot include lower/upper limits"):
        model.validate(strict=True)


def test_continuous_accepts_effort_velocity_only() -> None:
    model = ArticulatedObject(name="continuous_valid")
    base = model.part("base")
    mast = model.part("mast")
    model.articulation(
        "mast_slew",
        "continuous",
        parent=base,
        child=mast,
        motion_limits=MotionLimits(effort=12.0, velocity=1.5),
    )
    model.validate(strict=True)


def test_floating_forbids_motion_limits() -> None:
    model = ArticulatedObject(name="floating_forbids_limits")
    base = model.part("base")
    payload = model.part("payload")
    model.articulation(
        "base_to_payload",
        "floating",
        parent=base,
        child=payload,
        motion_limits=MotionLimits(effort=1.0, velocity=1.0),
    )

    with pytest.raises(ValidationError, match="does not support motion limits"):
        model.validate(strict=True)


@pytest.mark.parametrize(
    ("joint_type", "kwargs"),
    [
        (ArticulationType.SPHERICAL, {"cone_deg": 45.0}),
        (ArticulationType.FIXED, {}),
        (ArticulationType.FLOATING, {}),
    ],
)
def test_non_scalar_articulations_forbid_motion_properties(
    joint_type: ArticulationType,
    kwargs: dict[str, float],
) -> None:
    model = ArticulatedObject(name=f"{joint_type.value}_forbids_motion_properties")
    base = model.part("base")
    payload = model.part("payload")
    model.articulation(
        "base_to_payload",
        joint_type,
        parent=base,
        child=payload,
        motion_properties=MotionProperties(friction=0.1),
        **kwargs,
    )

    with pytest.raises(ValidationError, match="does not support motion properties"):
        model.validate(strict=True)


def test_scalar_articulation_accepts_explicit_newton_tuning_and_passive_spring() -> None:
    model = ArticulatedObject(name="scalar_newton_joint_fields")
    base = model.part("base")
    arm = model.part("arm")
    tuning = NewtonJointSolverTuning(
        solref_friction=(0.01, 1.0),
        solimp_friction=(0.99, 0.99, 0.001, 0.5, 2.0),
    )
    spring = PassiveJointSpring(stiffness=4.0, reference=0.2)

    joint = model.joint(
        "base_to_arm",
        ArticulationType.REVOLUTE,
        parent=base,
        child=arm,
        limit=MotionLimits(effort=5.0, velocity=2.0, lower=-0.5, upper=0.5),
        newton_joint_solver_tuning=tuning,
        passive_spring=spring,
    )

    model.validate(strict=True)
    assert joint.newton_joint_solver_tuning is tuning
    assert joint.passive_spring is spring


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        pytest.param(
            "newton_joint_solver_tuning",
            NewtonJointSolverTuning(solimp_friction=(0.99, 0.99, 0.001, 0.5, 2.0)),
            "does not support Newton joint solver tuning",
            id="newton-solver-tuning",
        ),
        pytest.param(
            "passive_spring",
            PassiveJointSpring(stiffness=1.0),
            "does not support a passive spring",
            id="passive-spring",
        ),
    ],
)
def test_non_scalar_articulation_forbids_new_scalar_joint_fields(
    field_name: str,
    value: object,
    message: str,
) -> None:
    model = ArticulatedObject(name=f"fixed_forbids_{field_name}")
    base = model.part("base")
    payload = model.part("payload")
    model.articulation(
        "base_to_payload",
        ArticulationType.FIXED,
        parent=base,
        child=payload,
        **{field_name: value},
    )

    with pytest.raises(ValidationError, match=message):
        model.validate(strict=True)


def test_passive_spring_cannot_coexist_with_active_drive() -> None:
    model = ArticulatedObject(name="spring_drive_conflict")
    base = model.part("base")
    arm = model.part("arm")
    model.articulation(
        "base_to_arm",
        ArticulationType.REVOLUTE,
        parent=base,
        child=arm,
        motion_limits=MotionLimits(effort=5.0, velocity=2.0, lower=-0.5, upper=0.5),
        passive_spring=PassiveJointSpring(stiffness=4.0, reference=0.2),
        drive=JointDrive(target_position=0.0, stiffness=3.0, damping=0.5),
    )

    with pytest.raises(ValidationError, match="cannot set passive_spring with drive"):
        model.validate(strict=True)


def test_mimic_requires_existing_source() -> None:
    model = ArticulatedObject(name="mimic_missing_source")
    base = model.part("base")
    finger = model.part("finger")
    model.articulation(
        "base_to_finger",
        "revolute",
        parent=base,
        child=finger,
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=0.0, upper=1.0),
        mimic=Mimic(joint="missing_joint"),
    )

    with pytest.raises(ValidationError, match="mimic references missing articulation"):
        model.validate(strict=True)


def test_mimic_rejects_incompatible_motion_domain() -> None:
    model = ArticulatedObject(name="mimic_incompatible_domain")
    base = model.part("base")
    slider = model.part("slider")
    arm = model.part("arm")
    model.articulation(
        "base_to_slider",
        "prismatic",
        parent=base,
        child=slider,
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=0.0, upper=0.2),
    )
    model.articulation(
        "slider_to_arm",
        "revolute",
        parent=slider,
        child=arm,
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=-0.5, upper=0.5),
        mimic=Mimic(joint="base_to_slider"),
    )

    with pytest.raises(ValidationError, match="compatible motion domain"):
        model.validate(strict=True)


def test_mimic_rejects_cycles() -> None:
    model = ArticulatedObject(name="mimic_cycle")
    base = model.part("base")
    left = model.part("left")
    right = model.part("right")
    model.articulation(
        "base_to_left",
        "revolute",
        parent=base,
        child=left,
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=-1.0, upper=1.0),
        mimic=Mimic(joint="left_to_right"),
    )
    model.articulation(
        "left_to_right",
        "revolute",
        parent=left,
        child=right,
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=-1.0, upper=1.0),
        mimic=Mimic(joint="base_to_left"),
    )

    with pytest.raises(ValidationError, match="Mimic cycle detected"):
        model.validate(strict=True)


def test_strict_validation_rejects_multiple_root_parts() -> None:
    model = ArticulatedObject(name="multiple_roots")
    model.part("base")
    model.part("rogue")

    with pytest.raises(ValidationError, match="exactly one root part"):
        model.validate(strict=True)


def test_urdf_export_normalizes_motion_axis() -> None:
    model = ArticulatedObject(name="normalized_axis_export")
    base = model.part("base")
    arm = model.part("arm")
    joint = model.articulation(
        "base_to_arm",
        "revolute",
        parent=base,
        child=arm,
        axis=(0.0, 0.0, 10.0),
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=-1.0, upper=1.0),
    )

    elem = _articulation_element(joint)
    axis_elem = elem.find("axis")
    assert axis_elem is not None
    assert axis_elem.attrib["xyz"] == "0 0 1"


def test_urdf_export_emits_mimic_element() -> None:
    model = ArticulatedObject(name="mimic_export")
    base = model.part("base")
    left = model.part("left")
    right = model.part("right")
    model.articulation(
        "base_to_left",
        "revolute",
        parent=base,
        child=left,
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=0.0, upper=0.6),
    )
    joint = model.articulation(
        "base_to_right",
        "revolute",
        parent=base,
        child=right,
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=-0.6, upper=0.0),
        mimic=Mimic(joint="base_to_left", multiplier=-1.0, offset=0.05),
    )

    elem = _articulation_element(joint)
    mimic_elem = elem.find("mimic")
    assert mimic_elem is not None
    assert mimic_elem.attrib == {
        "joint": "base_to_left",
        "multiplier": "-1",
        "offset": "0.05",
    }
