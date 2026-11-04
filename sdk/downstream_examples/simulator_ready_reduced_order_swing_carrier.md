---
title: 'Simulator-Ready Reduced-Order Swing Carrier'
description: 'Base SDK reference for a fixed support and one empty swing carrier whose mass is a coarse material-and-volume sum rather than a payload guess.'
tags:
  - sdk
  - base sdk
  - simulator ready
  - swing
  - reduced order
  - mass
  - rigid body
  - physics material
---
# Simulator-Ready Reduced-Order Swing Carrier

Use one reduced-order carrier when a single swing degree of freedom is enough
and independently simulated chains are outside the requested scope. The moving
body contains a thin rubberized-fabric seat, two slender steel hanger proxies,
a light cross-axle, and two polymer bushings. Its empty mass is the sum of those
four major primitive material × volume estimates; the two tiny bushings remain
visible but are intentionally omitted from the mass sum. `fill_fraction`
records the plain physical judgment that a thin seat, slender hanger proxies,
and hollow support are not solid blocks.

This mass explicitly excludes a rider or any other payload. The seat box is
also used as a deliberately coarse inertia proxy for the combined carrier; the
example does not recalculate the composite center of mass.

```python
from __future__ import annotations

import math

from sdk import (
    ArticulatedObject,
    ArticulationType,
    BaseMode,
    Box,
    Cylinder,
    Inertial,
    MotionLimits,
    MotionProperties,
    Origin,
    PhysxJointFrictionCoefficient,
    RigidBodyProperties,
    TestContext,
    TestReport,
    estimate_mass_from_density,
)

STEEL_DENSITY = 7850.0
RUBBERIZED_FABRIC_DENSITY = 1050.0

SUPPORT_SHAPE = Box((0.90, 0.06, 0.03))
SUPPORT_ORIGIN = Origin(xyz=(0.0, 0.0, 1.373))
SEAT_SHAPE = Box((0.48, 0.25, 0.035))
SEAT_ORIGIN = Origin(xyz=(0.0, 0.0, -1.0))
HANGER_SHAPE = Cylinder(radius=0.0045, length=0.99)
PIVOT_AXLE_SHAPE = Cylinder(radius=0.006, length=0.40)
BUSHING_SHAPE = Cylinder(radius=0.008, length=0.04)
BUSHING_ROTATION = (0.0, math.pi / 2.0, 0.0)

SUPPORT_MASS = estimate_mass_from_density(
    SUPPORT_SHAPE,
    STEEL_DENSITY,
    fill_fraction=0.25,  # hollow rectangular steel support envelope
)
CARRIER_MASS_COMPONENTS = (
    (SEAT_SHAPE, RUBBERIZED_FABRIC_DENSITY, 0.16),
    (HANGER_SHAPE, STEEL_DENSITY, 0.55),
    (HANGER_SHAPE, STEEL_DENSITY, 0.55),
    (PIVOT_AXLE_SHAPE, STEEL_DENSITY, 0.40),
)
EMPTY_CARRIER_MASS = sum(
    estimate_mass_from_density(shape, density, fill_fraction=fill_fraction)
    for shape, density, fill_fraction in CARRIER_MASS_COMPONENTS
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(
        name="simulator_ready_reduced_order_swing_carrier",
        base_mode=BaseMode.FIXED,
    )
    model.meta["mode"] = "articulated"
    model.meta["mass_scope"] = "empty carrier only; rider payload excluded"

    coated_steel = model.material(
        "coated_steel",
        rgba=(0.18, 0.22, 0.26, 1.0),
        metallic=0.55,
        roughness=0.34,
    )
    seat_fabric = model.material(
        "rubberized_fabric",
        rgba=(0.08, 0.24, 0.62, 1.0),
        metallic=0.0,
        roughness=0.78,
    )
    polymer = model.material(
        "dark_polymer",
        rgba=(0.04, 0.05, 0.06, 1.0),
        metallic=0.0,
        roughness=0.64,
    )
    model.physics_material(
        "steel_contact",
        static_friction=0.50,
        dynamic_friction=0.38,
        restitution=0.03,
        density=STEEL_DENSITY,
    )
    model.physics_material(
        "carrier_contact",
        static_friction=0.72,
        dynamic_friction=0.60,
        restitution=0.04,
        density=RUBBERIZED_FABRIC_DENSITY,
    )

    support = model.part(
        "support_beam",
        inertial=Inertial.from_geometry(
            SUPPORT_SHAPE,
            mass=SUPPORT_MASS,
            origin=SUPPORT_ORIGIN,
        ),
        rigid_body=RigidBodyProperties(physics_material="steel_contact"),
    )
    support.visual(
        SUPPORT_SHAPE,
        origin=SUPPORT_ORIGIN,
        material=coated_steel,
        name="top_beam",
    )

    carrier = model.part(
        "empty_swing_carrier",
        inertial=Inertial.from_geometry(
            SEAT_SHAPE,
            mass=EMPTY_CARRIER_MASS,
            origin=SEAT_ORIGIN,
        ),
        rigid_body=RigidBodyProperties(physics_material="carrier_contact"),
    )
    carrier.visual(
        SEAT_SHAPE,
        origin=SEAT_ORIGIN,
        material=seat_fabric,
        name="thin_seat",
    )
    carrier.visual(
        PIVOT_AXLE_SHAPE,
        origin=Origin(rpy=BUSHING_ROTATION),
        material=coated_steel,
        name="pivot_axle",
    )
    for side, x_position in (("left", -0.20), ("right", 0.20)):
        carrier.visual(
            HANGER_SHAPE,
            origin=Origin(xyz=(x_position, 0.0, -0.4925)),
            material=coated_steel,
            name=f"{side}_hanger",
        )
        carrier.visual(
            BUSHING_SHAPE,
            origin=Origin(xyz=(x_position, 0.0, 0.0), rpy=BUSHING_ROTATION),
            material=polymer,
            name=f"{side}_bushing",
        )

    model.articulation(
        "support_to_swing_carrier",
        ArticulationType.REVOLUTE,
        parent=support,
        child=carrier,
        origin=Origin(xyz=(0.0, 0.0, 1.35)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(
            lower=math.radians(-50.0),
            upper=math.radians(50.0),
            effort=30.0,
            velocity=2.2,
        ),
        motion_properties=MotionProperties(damping=0.45, friction=0.08),
        physx_joint_resistance=PhysxJointFrictionCoefficient(0.03),
    )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    carrier = object_model.get_part("empty_swing_carrier")
    joint = object_model.get_articulation("support_to_swing_carrier")
    ctx.check(
        "empty_carrier_mass_has_a_sensible_small_scale",
        carrier.inertial is not None
        and carrier.inertial.mass == EMPTY_CARRIER_MASS
        and 1.0 < carrier.inertial.mass < 2.0,
        f"empty_carrier_mass={carrier.inertial.mass if carrier.inertial else None!r}",
    )
    ctx.check(
        "carrier_contains_seat_hangers_axle_and_bushings",
        len(carrier.visuals) == 6,
        f"visual_count={len(carrier.visuals)}",
    )
    ctx.check(
        "mass_scope_excludes_rider_payload",
        object_model.meta["mass_scope"] == "empty carrier only; rider payload excluded",
        f"mass_scope={object_model.meta.get('mass_scope')!r}",
    )
    ctx.check(
        "passive_swing_joint_has_complete_physics",
        joint.drive is None
        and joint.motion_limits is not None
        and joint.motion_limits.effort == 30.0
        and joint.motion_limits.velocity == 2.2
        and joint.motion_properties is not None
        and joint.motion_properties.damping == 0.45
        and joint.motion_properties.friction == 0.08,
        "The carrier hinge needs deliberate travel, effort, velocity, damping, and friction.",
    )
    return ctx.report()


object_model = build_object_model()
```

Do not add a child or adult mass to `EMPTY_CARRIER_MASS`. A simulator scenario
that places a separate rider body on the seat supplies that payload through the
rider body itself. The two steel hanger proxies in this reduced-order carrier are
deliberately rigid carrier components, not separately modeled flexible chains or
tethers. If a request instead calls for an actual hose, rope, cable, cord, wire,
chain, strap, belt, or flexible suspension, author each run as one
`BodyModel.FLEXIBLE_1D` static-rest Part attached with `FIXED`; never replace it
with multiple rigid bodies and serial joints. Report detailed chain motion or
tension as unsupported by current USDCraft rather than constructing a segmented
approximation.
