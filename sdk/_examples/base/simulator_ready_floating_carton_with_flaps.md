---
title: 'Simulator-Ready Floating Carton with Flaps'
description: 'Base SDK reference for a floating cardboard carton whose body and four passive flaps all carry explicit mass, collision material binding, and hinge properties.'
tags:
  - sdk
  - base sdk
  - simulator ready
  - floating base
  - cardboard box
  - carton
  - flap
  - hinge
  - physics material
---
# Simulator-Ready Floating Carton with Flaps

A floating articulated asset needs physical properties on its root body as well
as on every moving child. This example builds an open carton from collidable
primitive panels and gives all four flaps explicit inertia, rigid-body contact
material binding, bounded travel, damping, and friction.

Corrugated cardboard is mostly air, so its panel mass uses the registered
paper-fiber density with a coarse `fill_fraction`. The root mass is simply the
sum of its five visible panels. This keeps the estimate explainable without
turning authoring into a center-of-mass or laminate-analysis exercise.

The flaps are passive user-manipulated parts, so none has a target drive. At
`q=0` they stand upright; their symmetric limits permit folding inward or
outward while preserving the assembled rest pose.

```python
from __future__ import annotations

import math

from sdk import (
    ArticulatedObject,
    ArticulationType,
    BaseMode,
    Box,
    Inertial,
    MotionLimits,
    MotionProperties,
    Origin,
    PhysxJointFriction,
    RigidBodyProperties,
    TestContext,
    TestReport,
    estimate_mass_from_density,
)

WIDTH = 0.48
DEPTH = 0.34
HEIGHT = 0.22
THICKNESS = 0.008
FLAP_LENGTH = 0.16
CARDBOARD_FIBER_DENSITY = 700.0
CORRUGATED_FILL_FRACTION = 0.20
CARTON_BOUNDS = Box((WIDTH, DEPTH, HEIGHT))
BOTTOM_PANEL_SHAPE = Box((WIDTH, DEPTH, THICKNESS))
FRONT_BACK_WALL_SHAPE = Box((WIDTH, THICKNESS, HEIGHT))
SIDE_WALL_SHAPE = Box((THICKNESS, DEPTH - 2.0 * THICKNESS, HEIGHT))
CARTON_BODY_COMPONENTS = (
    BOTTOM_PANEL_SHAPE,
    FRONT_BACK_WALL_SHAPE,
    FRONT_BACK_WALL_SHAPE,
    SIDE_WALL_SHAPE,
    SIDE_WALL_SHAPE,
)
CARTON_BODY_MASS = sum(
    estimate_mass_from_density(
        shape,
        CARDBOARD_FIBER_DENSITY,
        fill_fraction=CORRUGATED_FILL_FRACTION,
    )
    for shape in CARTON_BODY_COMPONENTS
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(
        name="simulator_ready_floating_carton",
        base_mode=BaseMode.FLOATING,
    )
    model.meta["mode"] = "articulated"

    cardboard = model.material(
        "kraft_cardboard",
        rgba=(0.56, 0.34, 0.14, 1.0),
        metallic=0.0,
        roughness=0.94,
    )
    model.physics_material(
        "cardboard_contact",
        static_friction=0.66,
        dynamic_friction=0.54,
        restitution=0.02,
        density=CARDBOARD_FIBER_DENSITY,
    )

    body = model.part(
        "carton_body",
        inertial=Inertial.from_geometry(
            CARTON_BOUNDS,
            mass=CARTON_BODY_MASS,
            origin=Origin(xyz=(0.0, 0.0, HEIGHT / 2.0)),
        ),
        rigid_body=RigidBodyProperties(physics_material="cardboard_contact"),
    )
    body.visual(
        BOTTOM_PANEL_SHAPE,
        origin=Origin(xyz=(0.0, 0.0, THICKNESS / 2.0)),
        material=cardboard,
        name="bottom_panel",
    )
    body.visual(
        FRONT_BACK_WALL_SHAPE,
        origin=Origin(xyz=(0.0, -DEPTH / 2.0, HEIGHT / 2.0)),
        material=cardboard,
        name="front_wall",
    )
    body.visual(
        FRONT_BACK_WALL_SHAPE,
        origin=Origin(xyz=(0.0, DEPTH / 2.0, HEIGHT / 2.0)),
        material=cardboard,
        name="back_wall",
    )
    body.visual(
        SIDE_WALL_SHAPE,
        origin=Origin(xyz=(-WIDTH / 2.0, 0.0, HEIGHT / 2.0)),
        material=cardboard,
        name="left_wall",
    )
    body.visual(
        SIDE_WALL_SHAPE,
        origin=Origin(xyz=(WIDTH / 2.0, 0.0, HEIGHT / 2.0)),
        material=cardboard,
        name="right_wall",
    )

    flap_specs = (
        (
            "front_flap",
            Box((WIDTH, THICKNESS, FLAP_LENGTH)),
            Origin(xyz=(0.0, -DEPTH / 2.0, HEIGHT)),
            (1.0, 0.0, 0.0),
        ),
        (
            "back_flap",
            Box((WIDTH, THICKNESS, FLAP_LENGTH)),
            Origin(xyz=(0.0, DEPTH / 2.0, HEIGHT)),
            (-1.0, 0.0, 0.0),
        ),
        (
            "left_flap",
            Box((THICKNESS, DEPTH - 2.0 * THICKNESS, FLAP_LENGTH)),
            Origin(xyz=(-WIDTH / 2.0, 0.0, HEIGHT)),
            (0.0, 1.0, 0.0),
        ),
        (
            "right_flap",
            Box((THICKNESS, DEPTH - 2.0 * THICKNESS, FLAP_LENGTH)),
            Origin(xyz=(WIDTH / 2.0, 0.0, HEIGHT)),
            (0.0, -1.0, 0.0),
        ),
    )
    flap_local_origin = Origin(xyz=(0.0, 0.0, FLAP_LENGTH / 2.0))
    for name, shape, hinge_origin, axis in flap_specs:
        flap = model.part(
            name,
            inertial=Inertial.from_geometry(
                shape,
                mass=estimate_mass_from_density(
                    shape,
                    CARDBOARD_FIBER_DENSITY,
                    fill_fraction=CORRUGATED_FILL_FRACTION,
                ),
                origin=flap_local_origin,
            ),
            rigid_body=RigidBodyProperties(physics_material="cardboard_contact"),
        )
        flap.visual(
            shape,
            origin=flap_local_origin,
            material=cardboard,
            name=f"{name}_panel",
        )
        model.articulation(
            f"carton_to_{name}",
            ArticulationType.REVOLUTE,
            parent=body,
            child=flap,
            origin=hinge_origin,
            axis=axis,
            motion_limits=MotionLimits(
                lower=-math.pi / 2.0,
                upper=math.pi / 2.0,
                effort=2.0,
                velocity=2.0,
            ),
            motion_properties=MotionProperties(damping=0.08, friction=0.04),
            physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(
                0.02,
                dynamic_ratio=0.75,
            ),
        )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    # Adjacent flaps deliberately layer during the closing sequence. Scope the
    # allowance to the exact panels: runtime sibling collisions remain enabled
    # and prevent impossible simultaneous interpenetration.
    for first, second in (
        ("front_flap", "left_flap"),
        ("front_flap", "right_flap"),
        ("back_flap", "left_flap"),
        ("back_flap", "right_flap"),
    ):
        ctx.allow_overlap(
            first,
            second,
            elem_a=f"{first}_panel",
            elem_b=f"{second}_panel",
            reason="cardboard closure layers adjacent flaps; sibling collision constrains order",
        )
    ctx.check(
        "floating_base",
        object_model.base_mode == BaseMode.FLOATING,
        f"base_mode={object_model.base_mode!r}",
    )
    ctx.check(
        "root_and_flaps_have_explicit_physics",
        all(
            part.inertial is not None
            and part.inertial.mass > 0.0
            and part.rigid_body is not None
            and part.rigid_body.physics_material == "cardboard_contact"
            for part in object_model.parts
        ),
        "The floating root and every flap must be independently physical.",
    )
    ctx.check(
        "four_passive_complete_hinges",
        len(object_model.articulations) == 4
        and all(
            joint.drive is None
            and joint.motion_limits is not None
            and joint.motion_limits.effort > 0.0
            and joint.motion_limits.velocity > 0.0
            and joint.motion_properties is not None
            and joint.motion_properties.damping is not None
            and joint.motion_properties.friction is not None
            for joint in object_model.articulations
        ),
        "Expected four passive flap hinges with complete scalar-joint properties.",
    )
    return ctx.report()


object_model = build_object_model()
```

Do not make the carton root fixed merely to stop it falling, and do not omit
the root mass because it is a parent. A floating articulated object must be
physically complete as a whole.
