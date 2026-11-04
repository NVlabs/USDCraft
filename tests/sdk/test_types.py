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

import math
from collections.abc import Callable

import pytest

import sdk
from sdk import (
    ArticulatedObject,
    Box,
    Cylinder,
    JointDrive,
    Material,
    Mesh,
    MotionLimits,
    NewtonJointSolverTuning,
    PassiveJointSpring,
    PhysicsMaterial,
    PhysxJointFriction,
    PhysxJointFrictionCoefficient,
    Sphere,
    ValidationError,
    estimate_mass_from_density,
)

_NONFINITE_VALUES = (
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
)


def test_part_get_visual_returns_named_visual() -> None:
    model = ArticulatedObject(name="visual_lookup")
    part = model.part("body")
    visual = part.visual(Box((0.1, 0.1, 0.1)), name="shell")

    assert part.get_visual("shell") is visual


def test_part_get_visual_rejects_missing_name() -> None:
    model = ArticulatedObject(name="visual_lookup")
    part = model.part("body")
    part.visual(Box((0.1, 0.1, 0.1)), name="shell")

    with pytest.raises(Exception, match="Unknown visual"):
        part.get_visual("missing")


def test_cylinder_accepts_height_alias() -> None:
    cylinder = Cylinder(radius=0.1, height=0.4)

    assert cylinder.radius == 0.1
    assert cylinder.length == 0.4
    assert cylinder.height == 0.4


def test_cylinder_rejects_length_and_height_together() -> None:
    with pytest.raises(TypeError, match="only one of 'length' or alias 'height'"):
        Cylinder(radius=0.1, length=0.4, height=0.4)


@pytest.mark.parametrize(
    ("geometry", "expected_volume"),
    [
        pytest.param(Box((2.0, 3.0, 4.0)), 24.0, id="box"),
        pytest.param(Cylinder(radius=2.0, length=3.0), 12.0 * math.pi, id="cylinder"),
        pytest.param(Sphere(radius=3.0), 36.0 * math.pi, id="sphere"),
    ],
)
def test_estimate_mass_from_density_uses_primitive_volume(
    geometry: Box | Cylinder | Sphere,
    expected_volume: float,
) -> None:
    assert estimate_mass_from_density(geometry, density=500.0) == pytest.approx(
        expected_volume * 500.0
    )


def test_estimate_mass_from_density_applies_fill_fraction() -> None:
    mass = estimate_mass_from_density(Box((2.0, 3.0, 4.0)), 500.0, fill_fraction=0.25)

    assert mass == pytest.approx(3000.0)


@pytest.mark.parametrize("density", [0.0, -1.0])
def test_estimate_mass_from_density_rejects_nonpositive_density(density: float) -> None:
    with pytest.raises(ValidationError, match="density must be positive"):
        estimate_mass_from_density(Box((1.0, 1.0, 1.0)), density)


@pytest.mark.parametrize("density", _NONFINITE_VALUES)
def test_estimate_mass_from_density_rejects_nonfinite_density(density: float) -> None:
    with pytest.raises(ValidationError, match="density must be finite"):
        estimate_mass_from_density(Box((1.0, 1.0, 1.0)), density)


@pytest.mark.parametrize("fill_fraction", [0.0, -0.1, 1.01])
def test_estimate_mass_from_density_rejects_out_of_range_fill_fraction(
    fill_fraction: float,
) -> None:
    with pytest.raises(ValidationError, match="greater than 0 and at most 1"):
        estimate_mass_from_density(
            Box((1.0, 1.0, 1.0)),
            500.0,
            fill_fraction=fill_fraction,
        )


@pytest.mark.parametrize("fill_fraction", _NONFINITE_VALUES)
def test_estimate_mass_from_density_rejects_nonfinite_fill_fraction(
    fill_fraction: float,
) -> None:
    with pytest.raises(ValidationError, match="fill_fraction must be finite"):
        estimate_mass_from_density(
            Box((1.0, 1.0, 1.0)),
            500.0,
            fill_fraction=fill_fraction,
        )


@pytest.mark.parametrize(
    ("geometry_factory", "message"),
    [
        pytest.param(
            lambda: Box((0.0, 1.0, 1.0)),
            "box.size.x must be positive",
            id="box-zero",
        ),
        pytest.param(
            lambda: Box((1.0, math.inf, 1.0)),
            r"box.size\[1\] must be finite",
            id="box-inf",
        ),
        pytest.param(
            lambda: Cylinder(radius=-1.0, length=1.0),
            "cylinder.radius must be positive",
            id="cylinder-negative",
        ),
        pytest.param(
            lambda: Cylinder(radius=1.0, length=math.nan),
            "cylinder.length must be finite",
            id="cylinder-nan",
        ),
        pytest.param(
            lambda: Sphere(radius=0.0),
            "sphere.radius must be positive",
            id="sphere-zero",
        ),
        pytest.param(
            lambda: Sphere(radius=math.inf),
            "sphere.radius must be finite",
            id="sphere-inf",
        ),
    ],
)
def test_estimate_mass_from_density_rejects_invalid_dimensions(
    geometry_factory: Callable[[], Box | Cylinder | Sphere],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        estimate_mass_from_density(geometry_factory(), 500.0)


def test_estimate_mass_from_density_rejects_mesh_explicitly() -> None:
    with pytest.raises(ValidationError, match="Cannot estimate mass from mesh geometry"):
        estimate_mass_from_density(Mesh(name="procedural"), 500.0)


def test_estimate_mass_from_density_rejects_nonfinite_result() -> None:
    with pytest.raises(ValidationError, match="estimated mass must be finite and positive"):
        estimate_mass_from_density(Box((1e308, 1e308, 1.0)), 1.0)


def test_part_visual_accepts_color_alias_as_named_material() -> None:
    model = ArticulatedObject(name="visual_color_alias")
    part = model.part("body")

    visual = part.visual(Box((0.1, 0.1, 0.1)), color="steel_blue", name="shell")

    assert visual.material == "steel_blue"


def test_part_visual_accepts_color_alias_as_rgba_tuple() -> None:
    model = ArticulatedObject(name="visual_color_alias")
    part = model.part("body")

    visual = part.visual(Box((0.1, 0.1, 0.1)), color=(0.1, 0.2, 0.3, 0.4), name="shell")

    assert isinstance(visual.material, Material)
    assert visual.material.rgba == (0.1, 0.2, 0.3, 0.4)


def test_part_visual_rejects_material_and_color_together() -> None:
    model = ArticulatedObject(name="visual_color_alias")
    part = model.part("body")

    with pytest.raises(TypeError, match="only one of 'material' or alias 'color'"):
        part.visual(Box((0.1, 0.1, 0.1)), material="steel", color="blue")


def test_material_accepts_color_alias_with_rgb_tuple() -> None:
    material = Material(name="paint", color=(0.1, 0.2, 0.3))

    assert material.rgba == (0.1, 0.2, 0.3, 1.0)


def test_material_accepts_color_alias_with_rgba_tuple() -> None:
    material = Material(name="paint", color=(0.1, 0.2, 0.3, 0.4))

    assert material.rgba == (0.1, 0.2, 0.3, 0.4)


def test_material_rejects_rgba_and_color_together() -> None:
    with pytest.raises(ValidationError, match="Material cannot set both rgba and color"):
        Material(name="paint", rgba=(0.1, 0.2, 0.3, 1.0), color=(0.2, 0.3, 0.4, 1.0))


def test_material_preserves_positional_rgba_usage() -> None:
    material = Material("paint", (0.1, 0.2, 0.3, 1.0))

    assert material.rgba == (0.1, 0.2, 0.3, 1.0)


def test_material_accepts_pbr_fields() -> None:
    material = Material(
        name="glaze",
        rgba=(0.6, 0.5, 0.4, 1.0),
        metallic=0.0,
        roughness=0.35,
        emissive=(0.1, 0.0, 0.0),
        transmission=0.2,
        ior=1.45,
        clearcoat=0.5,
        clearcoat_roughness=0.25,
    )

    assert material.metallic == 0.0
    assert material.roughness == 0.35
    assert material.emissive == (0.1, 0.0, 0.0)
    assert material.transmission == 0.2
    assert material.ior == 1.45
    assert material.clearcoat == 0.5
    assert material.clearcoat_roughness == 0.25
    assert material.has_pbr()


def test_material_normalizes_clearcoat_roughness_to_portable_default() -> None:
    material = Material(name="lacquer", clearcoat=0.4)

    assert material.clearcoat_roughness == pytest.approx(0.2)


def test_material_without_pbr_reports_no_pbr() -> None:
    material = Material(name="paint", rgba=(0.1, 0.2, 0.3, 1.0))

    assert not material.has_pbr()


def test_material_rejects_out_of_range_pbr_scalars() -> None:
    with pytest.raises(ValidationError, match="material.metallic"):
        Material(name="bad", metallic=1.5)
    with pytest.raises(ValidationError, match="material.roughness"):
        Material(name="bad", roughness=-0.1)
    with pytest.raises(ValidationError, match="material.ior"):
        Material(name="bad", ior=0.9)


def test_material_rejects_bad_emissive() -> None:
    with pytest.raises(ValidationError, match="material.emissive"):
        Material(name="bad", emissive=(0.1, 0.2))
    with pytest.raises(ValidationError, match="material.emissive"):
        Material(name="bad", emissive=(-0.1, 0.0, 0.0))


def test_model_material_passes_pbr_fields_through() -> None:
    model = ArticulatedObject(name="pbr_model")
    material = model.material("steel", rgba=(0.7, 0.7, 0.7, 1.0), metallic=0.95, roughness=0.3)

    assert material.metallic == 0.95
    assert material.roughness == 0.3
    assert model.materials[-1] is material


@pytest.mark.parametrize(
    "field_name",
    ["static_friction", "dynamic_friction", "restitution", "density"],
)
@pytest.mark.parametrize("value", _NONFINITE_VALUES)
def test_physics_material_rejects_nonfinite_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError, match=rf"physics_material\.{field_name} must be finite"):
        PhysicsMaterial(name="invalid", **{field_name: value})


def test_physics_material_requires_static_friction_not_below_dynamic() -> None:
    with pytest.raises(ValidationError, match="dynamic_friction must be <= static_friction"):
        PhysicsMaterial(
            name="invalid_order",
            static_friction=0.2,
            dynamic_friction=0.3,
        )


@pytest.mark.parametrize(
    "field_name",
    ["effort", "velocity", "lower", "upper"],
)
@pytest.mark.parametrize("value", _NONFINITE_VALUES)
def test_motion_limits_reject_nonfinite_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError, match=rf"motion_limits\.{field_name} must be finite"):
        MotionLimits(**{field_name: value})


def test_motion_limits_keep_optional_position_bounds() -> None:
    limits = MotionLimits(effort=2.0, velocity=3.0, lower=None, upper=None)

    assert limits.effort == pytest.approx(2.0)
    assert limits.velocity == pytest.approx(3.0)
    assert limits.lower is None
    assert limits.upper is None


def test_physx_effort_friction_helpers_author_valid_models() -> None:
    stay_put = PhysxJointFriction.stay_put(2.0)
    assert stay_put.dynamic_friction_effort == pytest.approx(2.2)
    assert stay_put.static_friction_effort == pytest.approx(2.42)

    breakaway = PhysxJointFriction.from_breakaway_effort(0.08)
    assert breakaway.static_friction_effort == pytest.approx(0.08)
    assert breakaway.dynamic_friction_effort == pytest.approx(0.064)


def test_physx_effort_friction_rejects_invalid_order_and_all_zero() -> None:
    with pytest.raises(ValidationError, match="dynamic_friction_effort must be <="):
        PhysxJointFriction(
            static_friction_effort=1.0,
            dynamic_friction_effort=1.1,
        )
    with pytest.raises(ValidationError, match="at least one non-zero"):
        PhysxJointFriction(
            static_friction_effort=0.0,
            dynamic_friction_effort=0.0,
        )

    assert PhysxJointFrictionCoefficient(0.0).coefficient == pytest.approx(0.0)


def test_newton_joint_solver_tuning_normalizes_explicit_vectors() -> None:
    tuning = NewtonJointSolverTuning(
        solref_friction=[0.01, 1.0],
        solimp_friction=[0.99, 0.99, 0.001, 0.5, 2.0],
    )
    direct = NewtonJointSolverTuning(solref_friction=(-100.0, -10.0))

    assert tuning.solref_friction == (0.01, 1.0)
    assert tuning.solimp_friction == (0.99, 0.99, 0.001, 0.5, 2.0)
    assert direct.solref_friction == (-100.0, -10.0)


def test_newton_joint_solver_tuning_allows_d0_greater_than_dwidth() -> None:
    tuning = NewtonJointSolverTuning(
        solimp_friction=(0.999, 0.95, 0.001, 0.5, 2.0),
    )

    assert tuning.solimp_friction == (0.999, 0.95, 0.001, 0.5, 2.0)


def test_newton_joint_solver_tuning_requires_at_least_one_vector() -> None:
    with pytest.raises(ValidationError, match="must set solref_friction or solimp_friction"):
        NewtonJointSolverTuning()


@pytest.mark.parametrize(
    "solref",
    [
        pytest.param((0.01,), id="wrong-length"),
        pytest.param((0.01, 0.0), id="mixed-standard-zero"),
        pytest.param((-1.0, 1.0), id="mixed-direct-standard"),
        pytest.param((float("nan"), 1.0), id="nonfinite"),
    ],
)
def test_newton_joint_solver_tuning_rejects_invalid_solref(
    solref: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError):
        NewtonJointSolverTuning(solref_friction=solref)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("solimp", "message"),
    [
        pytest.param((0.9, 0.95, 0.001, 0.5), "must have 5 elements", id="length"),
        pytest.param((-0.1, 0.95, 0.001, 0.5, 2.0), "between 0 and 1", id="d0"),
        pytest.param((0.9, 1.1, 0.001, 0.5, 2.0), "between 0 and 1", id="dwidth"),
        pytest.param((0.9, 0.95, 0.0, 0.5, 2.0), "width must be > 0", id="width"),
        pytest.param((0.9, 0.95, 0.001, 1.1, 2.0), "between 0 and 1", id="midpoint"),
        pytest.param((0.9, 0.95, 0.001, 0.5, 0.9), "power must be >= 1", id="power"),
        pytest.param(
            (0.9, 0.95, float("inf"), 0.5, 2.0),
            "must be finite",
            id="nonfinite",
        ),
    ],
)
def test_newton_joint_solver_tuning_rejects_invalid_solimp(
    solimp: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        NewtonJointSolverTuning(solimp_friction=solimp)  # type: ignore[arg-type]


def test_passive_joint_spring_uses_explicit_sdk_units() -> None:
    spring = PassiveJointSpring(stiffness=12.5, reference=-0.3)

    assert spring.stiffness == pytest.approx(12.5)
    assert spring.reference == pytest.approx(-0.3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"stiffness": 0.0}, "stiffness must be > 0", id="zero"),
        pytest.param({"stiffness": -0.1}, "stiffness must be > 0", id="negative"),
        pytest.param({"stiffness": float("nan")}, "stiffness must be finite", id="nan"),
        pytest.param(
            {"stiffness": 1.0, "reference": float("inf")},
            "reference must be finite",
            id="reference-inf",
        ),
    ],
)
def test_passive_joint_spring_rejects_invalid_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        PassiveJointSpring(**kwargs)


def test_new_joint_types_are_public_sdk_exports() -> None:
    assert sdk.NewtonJointSolverTuning is NewtonJointSolverTuning
    assert sdk.PassiveJointSpring is PassiveJointSpring
    assert "NewtonJointSolverTuning" in sdk.__all__
    assert "PassiveJointSpring" in sdk.__all__


@pytest.mark.parametrize(
    "field_name",
    ["target_position", "target_velocity", "stiffness", "damping", "max_force"],
)
@pytest.mark.parametrize("value", _NONFINITE_VALUES)
def test_joint_drive_rejects_nonfinite_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError, match=rf"joint_drive\.{field_name} must be finite"):
        JointDrive(**{field_name: value})
