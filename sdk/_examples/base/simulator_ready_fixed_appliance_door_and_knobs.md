---
title: 'Appliance Door and Knobs with Explicit Physics'
description: 'Base SDK reference for an appliance with a passive drop door, rotary controls, explicit rigid-body physics, and one intentionally driven control.'
tags:
  - sdk
  - base sdk
  - simulator ready
  - floating root
  - appliance
  - door
  - hinge
  - knob
  - physics material
  - rigid body
---
# Appliance Door and Knobs with Explicit Physics

Use this pattern for an appliance with a floating standalone root, passive door
and rotary controls. Scene anchoring belongs to the consuming simulation. Visual PBR and
contact physics are separate: every rigid part has explicit mass/inertia and a
`RigidBodyProperties` binding to a registered `PhysicsMaterial`.

Masses come from the same ordinary material densities registered below. The
enclosure and door use small `fill_fraction` values because their bounding
boxes represent thin, hollow sheet-metal assemblies rather than solid steel.
The knobs are nearly solid polymer. These are intentionally coarse material ×
volume estimates; they avoid arbitrary kilogram guesses without introducing a
center-of-mass reconstruction task.

The door and selector knob are passive. Their PhysX behavior uses absolute
effort friction rather than a load-proportional coefficient. The timer knob is
intentionally spring-returning, so it alone uses `JointDrive`: its
`MotionProperties.damping` is explicitly zero and its drive damping lives only
in `JointDrive.damping`. Do not add a target drive merely to make a passive
hinge look stable.

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
    JointDrive,
    MotionLimits,
    MotionProperties,
    NewtonJointSolverTuning,
    Origin,
    PhysxJointFriction,
    RigidBodyProperties,
    TestContext,
    TestReport,
    estimate_mass_from_density,
)

STEEL_DENSITY = 7800.0
POLYMER_DENSITY = 1100.0
BODY_SHAPE = Box((0.50, 0.38, 0.32))
DOOR_SHAPE = Box((0.46, 0.035, 0.25))
DOOR_LOCAL_ORIGIN = Origin(xyz=(0.0, -0.0175, 0.125))
KNOB_SHAPE = Cylinder(radius=0.028, length=0.035)
KNOB_LOCAL_ORIGIN = Origin(
    xyz=(0.0, -0.0175, 0.0),
    rpy=(math.pi / 2.0, 0.0, 0.0),
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(
        name="simulator_ready_fixed_appliance",
        base_mode=BaseMode.FLOATING,
    )
    model.meta["mode"] = "articulated"

    enamel = model.material(
        "cream_enamel",
        rgba=(0.82, 0.76, 0.62, 1.0),
        metallic=0.0,
        roughness=0.34,
        clearcoat=0.16,
        clearcoat_roughness=0.22,
    )
    dark_polymer = model.material(
        "dark_control_polymer",
        rgba=(0.10, 0.11, 0.12, 1.0),
        metallic=0.0,
        roughness=0.58,
    )
    model.physics_material(
        "painted_metal_contact",
        static_friction=0.52,
        dynamic_friction=0.40,
        restitution=0.03,
        density=STEEL_DENSITY,
    )
    model.physics_material(
        "polymer_control_contact",
        static_friction=0.68,
        dynamic_friction=0.55,
        restitution=0.04,
        density=POLYMER_DENSITY,
    )

    body = model.part(
        "appliance_body",
        inertial=Inertial.from_geometry(
            BODY_SHAPE,
            mass=estimate_mass_from_density(
                BODY_SHAPE,
                STEEL_DENSITY,
                fill_fraction=0.018,
            ),
        ),
        rigid_body=RigidBodyProperties(physics_material="painted_metal_contact"),
    )
    body.visual(BODY_SHAPE, material=enamel, name="enclosure")

    door = model.part(
        "drop_door",
        inertial=Inertial.from_geometry(
            DOOR_SHAPE,
            mass=estimate_mass_from_density(
                DOOR_SHAPE,
                STEEL_DENSITY,
                fill_fraction=0.045,
            ),
            origin=DOOR_LOCAL_ORIGIN,
        ),
        rigid_body=RigidBodyProperties(physics_material="painted_metal_contact"),
    )
    door.visual(
        DOOR_SHAPE,
        origin=DOOR_LOCAL_ORIGIN,
        material=enamel,
        name="door_panel",
    )
    model.articulation(
        "body_to_drop_door",
        ArticulationType.REVOLUTE,
        parent=body,
        child=door,
        origin=Origin(xyz=(0.0, -0.19, -0.14)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(
            lower=0.0,
            upper=math.radians(85.0),
            effort=18.0,
            velocity=1.2,
        ),
        motion_properties=MotionProperties(damping=0.35, friction=0.12),
        # Coarse gravity-load estimate for this horizontal-axis drop door:
        # dynamic friction exceeds the expected peak passive torque, while
        # static friction provides a modest additional break-away margin.
        physx_joint_resistance=PhysxJointFriction(
            static_friction_effort=2.5,
            dynamic_friction_effort=2.3,
        ),
        # Newton dry friction is a soft constraint: frictionloss supplies the
        # threshold, while explicit d0 supplies the pose-hold impedance.
        newton_joint_solver_tuning=NewtonJointSolverTuning(
            solimp_friction=(0.999, 0.9999, 0.001, 0.5, 2.0),
        ),
        meta={"gravity_hold": True},
    )

    knob_specs = (
        ("mode_knob", -0.18, None),
        ("temperature_knob", -0.06, None),
        ("selector_knob", 0.06, None),
        (
            "spring_timer_knob",
            0.18,
            JointDrive(
                mode="force",
                target_position=0.0,
                stiffness=0.8,
                damping=0.08,
                max_force=0.6,
            ),
        ),
    )
    for name, x_position, drive in knob_specs:
        knob = model.part(
            name,
            inertial=Inertial.from_geometry(
                KNOB_SHAPE,
                mass=estimate_mass_from_density(
                    KNOB_SHAPE,
                    POLYMER_DENSITY,
                    fill_fraction=0.85,
                ),
                origin=KNOB_LOCAL_ORIGIN,
            ),
            rigid_body=RigidBodyProperties(physics_material="polymer_control_contact"),
        )
        knob.visual(
            KNOB_SHAPE,
            origin=KNOB_LOCAL_ORIGIN,
            material=dark_polymer,
            name=f"{name}_shell",
        )
        model.articulation(
            f"body_to_{name}",
            ArticulationType.REVOLUTE,
            parent=body,
            child=knob,
            origin=Origin(xyz=(x_position, -0.19, 0.14)),
            axis=(0.0, -1.0, 0.0),
            motion_limits=MotionLimits(
                lower=math.radians(-145.0),
                upper=math.radians(145.0),
                effort=0.8,
                velocity=3.0,
            ),
            # The active timer gets viscous damping from its drive. Its passive
            # damping is therefore an intentional zero rather than a second,
            # competing damping source.
            motion_properties=MotionProperties(
                damping=0.0 if drive is not None else 0.025,
                friction=0.035,
            ),
            physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(
                0.04,
                dynamic_ratio=0.75,
            ),
            drive=drive,
            # The passive controls have gravity-orthogonal axes. The timer is
            # deliberately spring-returning instead of staying where placed.
            meta={"gravity_hold": drive is None},
        )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.check(
        "floating_root",
        object_model.base_mode == BaseMode.FLOATING,
        f"base_mode={object_model.base_mode!r}",
    )
    ctx.check(
        "all_rigid_parts_have_explicit_physics",
        len(object_model.parts) == 6
        and len(object_model.articulations) == 5
        and all(
            part.inertial is not None
            and part.inertial.mass > 0.0
            and part.rigid_body is not None
            and part.rigid_body.physics_material is not None
            for part in object_model.parts
        ),
        "Every enclosure, door, and knob needs mass, inertia, and contact material binding.",
    )
    ctx.check(
        "all_scalar_joints_have_limits_and_motion_properties",
        all(
            joint.motion_limits is not None
            and joint.motion_limits.effort > 0.0
            and joint.motion_limits.velocity > 0.0
            and joint.motion_properties is not None
            and joint.motion_properties.damping is not None
            and joint.motion_properties.friction is not None
            for joint in object_model.articulations
        ),
        "Every scalar joint needs deliberate limits, damping, and friction.",
    )
    driven = [joint.name for joint in object_model.articulations if joint.drive is not None]
    ctx.check(
        "only_timer_is_actively_driven",
        driven == ["body_to_spring_timer_knob"],
        f"driven={driven!r}",
    )
    door_joint = object_model.get_articulation("body_to_drop_door")
    ctx.check(
        "drop_door_has_explicit_dual_backend_pose_hold",
        door_joint.meta.get("gravity_hold") is True
        and door_joint.newton_joint_solver_tuning is not None
        and door_joint.newton_joint_solver_tuning.solimp_friction
        == (0.999, 0.9999, 0.001, 0.5, 2.0),
        "A loaded passive door needs separately sized friction effort and Newton impedance.",
    )
    timer_joint = object_model.get_articulation("body_to_spring_timer_knob")
    ctx.check(
        "active_drive_has_one_damping_source",
        timer_joint.motion_properties is not None
        and timer_joint.motion_properties.damping == 0.0
        and timer_joint.drive is not None
        and timer_joint.drive.damping == 0.08,
        "An active drive must keep MotionProperties.damping at zero and own the damping itself.",
    )
    ctx.check(
        "active_drive_force_is_bounded_by_joint_effort",
        timer_joint.motion_limits is not None
        and timer_joint.drive is not None
        and timer_joint.drive.max_force is not None
        and math.isfinite(timer_joint.drive.max_force)
        and 0.0 <= timer_joint.drive.max_force <= timer_joint.motion_limits.effort,
        "An explicit drive max_force must be finite, non-negative, and at most joint effort.",
    )
    return ctx.report()


object_model = build_object_model()
```

The door and passive controls use explicit effort resistance. The timer has a
position drive with damping and `max_force=0.6`, below its joint effort limit.
Keep drive and passive damping separate. Backend mappings and unsupported cases
are listed in `20_core_types.md`; inspect compile warnings and test dynamics in
the intended simulator before claiming runtime behavior.
