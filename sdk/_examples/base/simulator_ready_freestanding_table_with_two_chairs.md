---
title: 'Simulator-Ready Freestanding Table with Two Chairs'
description: 'Three independently movable furniture rigid bodies arranged on a shared external ground plane, with no physical joints between table and chairs.'
tags:
  - sdk
  - base sdk
  - simulator ready
  - table
  - chair
  - furniture
  - free rigid body
  - floating placement
  - external ground
  - physics material
---
# Simulator-Ready Freestanding Table with Two Chairs

A table and two ordinary chairs form a spatial arrangement, not one
articulation. Moving either chair must not pull the table or the other chair.
This example therefore authors three independent rigid `Part` objects. The
table is the SDK authoring root and the chairs use `ArticulationType.FLOATING`
placement edges solely to satisfy the current single-root authoring contract.
USDC emits no joints for those edges.

All three furniture bodies start with their feet on the same external ground
plane at z=0. The reusable asset intentionally does not add a floor mesh. Since
the chairs are therefore disconnected from the table's geometry,
`allow_isolated_part(...)` records the narrow, explicit external-ground
exception. The accompanying floor-alignment checks are required; do not use
that allowance for a suspended body or for merchandise that should contact its
shelf.

`model.meta["mode"] = "static"` means that this set has no constrained motion
axes. It does not make the three rigid bodies immovable.

```python
from __future__ import annotations

import math

from sdk import (
    ArticulatedObject,
    ArticulationType,
    BaseMode,
    Box,
    Inertial,
    Origin,
    RigidBodyProperties,
    TestContext,
    TestReport,
    estimate_mass_from_density,
)

WOOD_DENSITY = 650.0

TABLE_TOP = Box((1.20, 0.75, 0.05))
TABLE_TOP_ORIGIN = Origin(xyz=(0.0, 0.0, 0.745))
TABLE_LEG = Box((0.06, 0.06, 0.72))
TABLE_LEG_Z = 0.36
TABLE_PROXY = Box((1.20, 0.75, 0.77))
TABLE_PROXY_ORIGIN = Origin(xyz=(0.0, 0.0, 0.385))
TABLE_MASS = sum(
    (
        estimate_mass_from_density(
            TABLE_TOP,
            WOOD_DENSITY,
            fill_fraction=0.65,
        ),
        *(
            estimate_mass_from_density(
                TABLE_LEG,
                WOOD_DENSITY,
                fill_fraction=0.85,
            )
            for _ in range(4)
        ),
    )
)

CHAIR_SEAT = Box((0.46, 0.46, 0.045))
CHAIR_SEAT_ORIGIN = Origin(xyz=(0.0, 0.0, 0.4525))
CHAIR_LEG = Box((0.045, 0.045, 0.43))
CHAIR_LEG_Z = 0.215
CHAIR_BACK_POST = Box((0.045, 0.045, 0.45))
CHAIR_BACK_POST_ORIGIN_Z = 0.675
CHAIR_BACK_PANEL = Box((0.37, 0.035, 0.18))
CHAIR_BACK_PANEL_ORIGIN = Origin(xyz=(0.0, 0.205, 0.78))
CHAIR_PROXY = Box((0.46, 0.50, 0.90))
CHAIR_PROXY_ORIGIN = Origin(xyz=(0.0, 0.0, 0.45))
CHAIR_MASS = sum(
    (
        estimate_mass_from_density(
            CHAIR_SEAT,
            WOOD_DENSITY,
            fill_fraction=0.75,
        ),
        *(
            estimate_mass_from_density(
                CHAIR_LEG,
                WOOD_DENSITY,
                fill_fraction=0.85,
            )
            for _ in range(4)
        ),
        estimate_mass_from_density(
            CHAIR_BACK_PANEL,
            WOOD_DENSITY,
            fill_fraction=0.70,
        ),
    )
)

FURNITURE_PART_NAMES = ("table", "chair_01", "chair_02")
CHAIR_PLACEMENTS = (
    ("chair_01", Origin(xyz=(0.0, -0.78, 0.0))),
    ("chair_02", Origin(xyz=(0.0, 0.78, 0.0), rpy=(0.0, 0.0, math.pi))),
)


def _add_table_geometry(table, *, material) -> None:
    table.visual(
        TABLE_TOP,
        origin=TABLE_TOP_ORIGIN,
        material=material,
        name="table_top",
    )
    for x in (-0.52, 0.52):
        for y in (-0.295, 0.295):
            table.visual(
                TABLE_LEG,
                origin=Origin(xyz=(x, y, TABLE_LEG_Z)),
                material=material,
                name=f"table_leg_{x:+.3f}_{y:+.3f}",
            )


def _add_chair_geometry(chair, *, material) -> None:
    chair.visual(
        CHAIR_SEAT,
        origin=CHAIR_SEAT_ORIGIN,
        material=material,
        name="seat",
    )
    for x in (-0.185, 0.185):
        for y in (-0.185, 0.185):
            chair.visual(
                CHAIR_LEG,
                origin=Origin(xyz=(x, y, CHAIR_LEG_Z)),
                material=material,
                name=f"leg_{x:+.3f}_{y:+.3f}",
            )
    for x in (-0.185, 0.185):
        chair.visual(
            CHAIR_BACK_POST,
            origin=Origin(xyz=(x, 0.205, CHAIR_BACK_POST_ORIGIN_Z)),
            material=material,
            name=f"back_post_{x:+.3f}",
        )
    chair.visual(
        CHAIR_BACK_PANEL,
        origin=CHAIR_BACK_PANEL_ORIGIN,
        material=material,
        name="back_panel",
    )


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(
        name="simulator_ready_freestanding_table_with_two_chairs",
        base_mode=BaseMode.FLOATING,
    )
    model.meta["mode"] = "static"

    finished_wood = model.material(
        "finished_wood",
        rgba=(0.45, 0.22, 0.09, 1.0),
        metallic=0.0,
        roughness=0.48,
        clearcoat=0.12,
        clearcoat_roughness=0.22,
    )
    model.physics_material(
        "wood_floor_contact",
        static_friction=0.72,
        dynamic_friction=0.55,
        restitution=0.02,
        density=WOOD_DENSITY,
    )

    table = model.part(
        "table",
        inertial=Inertial.from_geometry(
            TABLE_PROXY,
            mass=TABLE_MASS,
            origin=TABLE_PROXY_ORIGIN,
        ),
        rigid_body=RigidBodyProperties(physics_material="wood_floor_contact"),
    )
    _add_table_geometry(table, material=finished_wood)

    for chair_name, placement in CHAIR_PLACEMENTS:
        chair = model.part(
            chair_name,
            inertial=Inertial.from_geometry(
                CHAIR_PROXY,
                mass=CHAIR_MASS,
                origin=CHAIR_PROXY_ORIGIN,
            ),
            rigid_body=RigidBodyProperties(physics_material="wood_floor_contact"),
        )
        _add_chair_geometry(chair, material=finished_wood)
        model.articulation(
            f"place_{chair_name}",
            ArticulationType.FLOATING,
            parent=table,
            child=chair,
            origin=placement,
        )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    parts = [object_model.get_part(name) for name in FURNITURE_PART_NAMES]
    ctx.check(
        "table_and_two_chairs_are_three_independent_rigid_bodies",
        object_model.base_mode == BaseMode.FLOATING
        and tuple(part.name for part in parts) == FURNITURE_PART_NAMES
        and all(
            part.inertial is not None
            and part.inertial.mass > 0.0
            and part.rigid_body is not None
            and part.rigid_body.physics_material == "wood_floor_contact"
            and any(visual.collidable for visual in part.visuals)
            for part in parts
        ),
        "The table and both chairs need independent mass, collision, and contact physics.",
    )
    ctx.check(
        "chairs_use_unconstrained_placement_edges_not_fixed_joints",
        len(object_model.articulations) == 2
        and {joint.child for joint in object_model.articulations} == {"chair_01", "chair_02"}
        and all(
            joint.articulation_type == ArticulationType.FLOATING
            and joint.motion_limits is None
            and joint.motion_properties is None
            and joint.drive is None
            for joint in object_model.articulations
        ),
        "Ordinary chairs are freely movable; table-to-chair FIXED joints would weld the set.",
    )
    for part in parts:
        bounds = ctx.part_world_aabb(part)
        ctx.check(
            f"{part.name}_feet_align_with_external_ground",
            bounds is not None and abs(float(bounds[0][2])) <= 1e-6,
            f"bounds={bounds!r}",
        )

    for chair_name, _placement in CHAIR_PLACEMENTS:
        ctx.allow_isolated_part(
            chair_name,
            reason=(
                "intentional freestanding furniture member whose tested feet rest on the "
                "shared external ground plane"
            ),
        )
    ctx.fail_if_isolated_parts(
        name="external_ground_exceptions_are_explicit_for_both_chairs",
    )
    ctx.fail_if_parts_overlap_in_current_pose(
        name="freestanding_furniture_bodies_do_not_interpenetrate",
    )
    return ctx.report()


object_model = build_object_model()
```

The `allow_isolated_part(...)` calls do not make the chairs static and do not
change exported physics. They only explain to the compiler-owned support check
why an intentionally freestanding member has no contact path to the table
inside this reusable, floor-less asset.
