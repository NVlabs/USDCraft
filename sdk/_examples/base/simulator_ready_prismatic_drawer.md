---
title: 'Simulator-Ready Prismatic Drawer'
description: 'Base SDK reference for a cabinet and passive pull-out drawer with retained insertion, explicit inertia, contact material binding, and complete prismatic joint properties.'
tags:
  - sdk
  - base sdk
  - simulator ready
  - floating root
  - drawer
  - cabinet
  - prismatic joint
  - physics material
  - rigid body
---
# Simulator-Ready Prismatic Drawer

This example uses a floating cabinet root while the drawer is a distinct dynamic
rigid body. The drawer extends into the cabinet behind its joint frame, so it
retains insertion at maximum travel instead of becoming a floating front
panel. It is passive and user-pulled, therefore it has damping and friction but
no target drive.

The carcass and drawer masses are sums of their main wood panels at the same
density used by the contact material. Their bounding boxes remain deliberately
coarse inertia proxies; the example does not attempt a separate center-of-mass
solve.

```python
from __future__ import annotations

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

WOOD_DENSITY = 650.0
CARCASS_BOUNDS = Box((0.36, 0.64, 0.50))
DRAWER_INERTIAL_SHAPE = Box((0.34, 0.56, 0.16))
DRAWER_INERTIAL_ORIGIN = Origin(xyz=(-0.15, 0.0, 0.08))
CARCASS_SIDE_SHAPE = Box((0.36, 0.04, 0.50))
CARCASS_HORIZONTAL_SHAPE = Box((0.36, 0.56, 0.04))
CARCASS_BACK_SHAPE = Box((0.03, 0.56, 0.42))
CARCASS_RAIL_SHAPE = Box((0.04, 0.56, 0.04))
CARCASS_COMPONENTS = (
    CARCASS_SIDE_SHAPE,
    CARCASS_SIDE_SHAPE,
    CARCASS_HORIZONTAL_SHAPE,
    CARCASS_HORIZONTAL_SHAPE,
    CARCASS_BACK_SHAPE,
    CARCASS_RAIL_SHAPE,
)
CARCASS_MASS = sum(
    estimate_mass_from_density(shape, WOOD_DENSITY) for shape in CARCASS_COMPONENTS
)
DRAWER_FRONT_SHAPE = Box((0.03, 0.56, 0.16))
DRAWER_BOTTOM_SHAPE = Box((0.32, 0.52, 0.015))
DRAWER_BACK_SHAPE = Box((0.02, 0.52, 0.12))
DRAWER_SIDE_SHAPE = Box((0.32, 0.015, 0.12))
DRAWER_COMPONENTS = (
    DRAWER_FRONT_SHAPE,
    DRAWER_BOTTOM_SHAPE,
    DRAWER_BACK_SHAPE,
    DRAWER_SIDE_SHAPE,
    DRAWER_SIDE_SHAPE,
)
DRAWER_MASS = sum(
    estimate_mass_from_density(shape, WOOD_DENSITY) for shape in DRAWER_COMPONENTS
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(
        name="simulator_ready_prismatic_drawer",
        base_mode=BaseMode.FLOATING,
    )
    model.meta["mode"] = "articulated"

    wood = model.material(
        "sealed_wood",
        rgba=(0.52, 0.31, 0.16, 1.0),
        metallic=0.0,
        roughness=0.52,
        clearcoat=0.12,
        clearcoat_roughness=0.28,
    )
    model.physics_material(
        "wood_slide_contact",
        static_friction=0.55,
        dynamic_friction=0.42,
        restitution=0.03,
        density=WOOD_DENSITY,
    )

    carcass = model.part(
        "cabinet_carcass",
        inertial=Inertial.from_geometry(
            CARCASS_BOUNDS,
            mass=CARCASS_MASS,
            origin=Origin(xyz=(-0.18, 0.0, 0.25)),
        ),
        rigid_body=RigidBodyProperties(physics_material="wood_slide_contact"),
    )
    carcass.visual(
        CARCASS_SIDE_SHAPE,
        origin=Origin(xyz=(-0.18, -0.30, 0.25)),
        material=wood,
        name="left_side",
    )
    carcass.visual(
        CARCASS_SIDE_SHAPE,
        origin=Origin(xyz=(-0.18, 0.30, 0.25)),
        material=wood,
        name="right_side",
    )
    carcass.visual(
        CARCASS_HORIZONTAL_SHAPE,
        origin=Origin(xyz=(-0.18, 0.0, 0.02)),
        material=wood,
        name="bottom_panel",
    )
    carcass.visual(
        CARCASS_HORIZONTAL_SHAPE,
        origin=Origin(xyz=(-0.18, 0.0, 0.48)),
        material=wood,
        name="top_panel",
    )
    carcass.visual(
        CARCASS_BACK_SHAPE,
        origin=Origin(xyz=(-0.345, 0.0, 0.25)),
        material=wood,
        name="back_panel",
    )
    # This rail supplies a real load path to the reduced-order central joint.
    carcass.visual(
        CARCASS_RAIL_SHAPE,
        origin=Origin(xyz=(-0.02, 0.0, 0.16)),
        material=wood,
        name="front_support_rail",
    )

    drawer = model.part(
        "drawer",
        inertial=Inertial.from_geometry(
            DRAWER_INERTIAL_SHAPE,
            mass=DRAWER_MASS,
            origin=DRAWER_INERTIAL_ORIGIN,
        ),
        rigid_body=RigidBodyProperties(physics_material="wood_slide_contact"),
    )
    drawer.visual(
        DRAWER_FRONT_SHAPE,
        origin=Origin(xyz=(0.015, 0.0, 0.08)),
        material=wood,
        name="drawer_front",
    )
    drawer.visual(
        DRAWER_BOTTOM_SHAPE,
        origin=Origin(xyz=(-0.15, 0.0, 0.0075)),
        material=wood,
        name="drawer_bottom",
    )
    drawer.visual(
        DRAWER_BACK_SHAPE,
        origin=Origin(xyz=(-0.31, 0.0, 0.06)),
        material=wood,
        name="drawer_back",
    )
    drawer.visual(
        DRAWER_SIDE_SHAPE,
        origin=Origin(xyz=(-0.15, -0.2525, 0.06)),
        material=wood,
        name="drawer_left_side",
    )
    drawer.visual(
        DRAWER_SIDE_SHAPE,
        origin=Origin(xyz=(-0.15, 0.2525, 0.06)),
        material=wood,
        name="drawer_right_side",
    )

    model.articulation(
        "cabinet_to_drawer",
        ArticulationType.PRISMATIC,
        parent=carcass,
        child=drawer,
        origin=Origin(xyz=(0.0, 0.0, 0.18)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(
            lower=0.0,
            upper=0.28,
            effort=100.0,
            velocity=0.35,
        ),
        motion_properties=MotionProperties(damping=7.0, friction=2.0),
        physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(
            6.0,
            dynamic_ratio=0.75,
            viscous_friction_coefficient=5.0,
        ),
    )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    drawer = object_model.get_part("drawer")
    joint = object_model.get_articulation("cabinet_to_drawer")
    ctx.check(
        "floating_carcass_and_physical_drawer",
        object_model.base_mode == BaseMode.FLOATING
        and all(
            part.inertial is not None
            and part.inertial.mass > 0.0
            and part.rigid_body is not None
            and part.rigid_body.physics_material == "wood_slide_contact"
            for part in object_model.parts
        ),
        "The parent and moving drawer both need explicit physical properties.",
    )
    ctx.check(
        "passive_complete_prismatic_joint",
        joint.drive is None
        and joint.motion_limits is not None
        and joint.motion_limits.effort == 100.0
        and joint.motion_limits.velocity == 0.35
        and joint.motion_properties is not None
        and joint.motion_properties.damping == 7.0
        and joint.motion_properties.friction == 2.0,
        "The pull-out joint needs deliberate limits, damping, and friction without a target drive.",
    )
    ctx.check(
        "drawer_retains_hidden_insertion",
        DRAWER_INERTIAL_ORIGIN.xyz[0] < -joint.motion_limits.upper / 2.0,
        f"drawer_origin={DRAWER_INERTIAL_ORIGIN.xyz!r}",
    )
    ctx.check(
        "drawer_has_open_tray_geometry",
        len(drawer.visuals) == 5,
        f"visual_count={len(drawer.visuals)}",
    )
    return ctx.report()


object_model = build_object_model()
```

For several drawers, repeat this pattern per drawer rather than grouping them
into one moving part. Each independently pullable drawer needs its own mass,
rigid-body binding, and prismatic joint.
