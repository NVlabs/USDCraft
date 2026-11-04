---
title: 'Direct SDK/Downstream Segmented Chain with Payload'
description: 'Direct SDK/downstream-only example of an open chain built from rigid dynamic segments and native joints; prohibited for USDCraft Authors (Generation and Reconstruction).'
tags:
  - sdk
  - base sdk
  - chain
  - rope
  - cable
  - segmented flexible member
  - direct sdk only
  - downstream only
  - not for usdcraft authors
  - rigid body
  - revolute articulation
  - simulator ready
  - payload
---
# Direct SDK/Downstream Segmented Chain with Payload

> **Do not use this example as a USDCraft Author (Generation or Reconstruction).** It is retained
> as a public SDK/downstream capability reference only. An Author must represent a hose, rope,
> cable, cord, wire, chain, strap, belt, flexible suspension, or comparable member as one
> `BodyModel.FLEXIBLE_1D` static-rest Part attached with `FIXED`. It must not copy this helper or
> hand-author an equivalent rigid-body/joint stack. Genuine distributed dynamics are currently
> unsupported by USDCraft.

Direct SDK/downstream callers outside the USDCraft Author tasks may use this pattern when bending, gravity response,
or load transfer is part of their application. Each segment is a rigid body with explicit mass,
inertia, a primitive collision shape, and a native joint. `HangingChainGeometry` is only a visual
mesh and is not dynamically equivalent to this direct-SDK representation.

The support, each repeated link, and the empty polymer load block use the same
short density × primitive-volume estimate as other simulator-ready references.
Coarse fill fractions describe the hollow support, open-link proxy, and hollow
load block without adding center-of-mass or mesh-volume analysis.

This example uses bounded revolute joints because that representation is portable across USDC and
URDF. Use `SpatialSegmentJoint` when native USDC three-dimensional bending is required; its URDF
artifact is only a floating-joint semantic fallback, not equivalent spherical dynamics.

```python
from __future__ import annotations

from sdk import (
    ArticulatedObject,
    ArticulationType,
    BaseMode,
    BodyModel,
    Box,
    Cylinder,
    Inertial,
    Origin,
    PlanarSegmentJoint,
    RigidBodyProperties,
    SegmentedFlexibleMemberSpec,
    TestContext,
    TestReport,
    add_segmented_flexible_member,
    estimate_mass_from_density,
)

STEEL_DENSITY = 7850.0
POLYMER_DENSITY = 1100.0
SUPPORT_SHAPE = Box((0.32, 0.18, 0.08))
SEGMENT_SHAPE = Cylinder(radius=0.012, length=0.10)
PAYLOAD_SHAPE = Box((0.18, 0.14, 0.08))
SUPPORT_MASS = estimate_mass_from_density(
    SUPPORT_SHAPE,
    STEEL_DENSITY,
    fill_fraction=0.11,  # hollow rectangular steel support envelope
)
SEGMENT_MASS = estimate_mass_from_density(
    SEGMENT_SHAPE,
    STEEL_DENSITY,
    fill_fraction=0.14,  # open chain-link proxy inside the cylinder envelope
)
EMPTY_PAYLOAD_MASS = estimate_mass_from_density(
    PAYLOAD_SHAPE,
    POLYMER_DENSITY,
    fill_fraction=0.23,  # hollow polymer load block; no hidden contents
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="segmented_chain_payload", base_mode=BaseMode.FIXED)
    model.meta["mode"] = "articulated"
    model.meta["simulation_fidelity"] = "segmented_rigid_body"

    steel = model.material("steel", rgba=(0.24, 0.27, 0.31, 1.0), metallic=0.75, roughness=0.3)
    payload_material = model.material(
        "payload_paint", rgba=(0.8, 0.18, 0.08, 1.0), metallic=0.1, roughness=0.45
    )
    model.physics_material(
        "chain_contact",
        static_friction=0.55,
        dynamic_friction=0.42,
        restitution=0.03,
        density=STEEL_DENSITY,
    )
    model.physics_material(
        "payload_contact",
        static_friction=0.62,
        dynamic_friction=0.50,
        restitution=0.04,
        density=POLYMER_DENSITY,
    )

    support = model.part(
        "support",
        inertial=Inertial.from_geometry(SUPPORT_SHAPE, mass=SUPPORT_MASS),
        rigid_body=RigidBodyProperties(physics_material="chain_contact"),
    )
    support.visual(SUPPORT_SHAPE, material=steel, name="support_beam")

    chain = add_segmented_flexible_member(
        model,
        "load_chain",
        support,
        attachment_origin=Origin(xyz=(0.0, 0.0, -0.04)),
        material=steel,
        physics_material="chain_contact",
        spec=SegmentedFlexibleMemberSpec(
            link_count=8,
            segment_geometry=SEGMENT_SHAPE,
            segment_mass=SEGMENT_MASS,
            joint=PlanarSegmentJoint(
                axis=(1.0, 0.0, 0.0),
                lower=-0.8,
                upper=0.8,
                effort=25.0,
                velocity=6.0,
                damping=0.08,
                friction=0.03,
            ),
        ),
    )

    payload_origin = Origin(xyz=(0.0, 0.0, -0.04))
    payload = model.part(
        "payload",
        inertial=Inertial.from_geometry(
            PAYLOAD_SHAPE,
            mass=EMPTY_PAYLOAD_MASS,
            origin=payload_origin,
        ),
        rigid_body=RigidBodyProperties(physics_material="payload_contact"),
    )
    payload.visual(
        PAYLOAD_SHAPE,
        origin=payload_origin,
        material=payload_material,
        name="payload_body",
    )
    model.articulation(
        "chain_to_payload",
        ArticulationType.FIXED,
        parent=chain.tail,
        child=payload,
        origin=chain.tail_attachment_origin,
    )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    links = [part for part in object_model.parts if part.name.startswith("load_chain_link_")]
    ctx.check("eight_dynamic_links", len(links) == 8, f"count={len(links)}")
    ctx.check(
        "links_are_physical",
        all(
            part.body_model == BodyModel.RIGID
            and part.inertial is not None
            and part.inertial.mass > 0.0
            and part.rigid_body is not None
            and part.rigid_body.enabled
            and len(part.visuals) == 1
            and part.visuals[0].collidable
            for part in links
        ),
        "Every simulated segment must have rigid-body mass and a primitive collider.",
    )
    return ctx.report()


object_model = build_object_model()
```

This helper represents an open articulation tree. A two-ended cable, belt loop, or two independent
suspension chains ending on one rigid payload is a closed loop and is not representable by the
current SDK. Use a clearly labeled reduced-order carrier when gross motion is sufficient, or report
the limitation; do not add an unconnected visual chain and call it simulated.
