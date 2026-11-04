---
title: 'Simulator-Ready Stocked Supermarket Gondola Shelf'
description: 'Gondola with individually movable contact-supported cans, cartons, and jars; every product is its own unconstrained rigid body.'
tags:
  - sdk
  - base sdk
  - simulator ready
  - supermarket
  - shelf
  - gondola
  - free rigid body
  - floating placement
  - contact support
  - physics material
---
# Simulator-Ready Stocked Supermarket Gondola Shelf

Use this pattern when a stationary shelf carries products that a robot or user
must be able to pick up one at a time. The gondola has a floating standalone root. Every can,
carton, and jar is a separate rigid `Part` with its own mass, contact material,
and collidable shape. A row is only a layout; it is never merged into one rigid
body.

The SDK currently requires one connected authoring tree, so each product uses
an `ArticulationType.FLOATING` placement edge from the gondola. That edge does
not weld the product to the shelf: the USDC exporter emits no USD joint for it
and leaves the two rigid bodies unconstrained. At q=0 every product bottom is
exactly tangent to a collidable shelf surface. Gravity, collision, and friction
therefore provide the support relationship.

`model.meta["mode"] = "static"` means this arrangement has no constrained
motion axes. It does not mean the product rigid bodies are kinematic or welded.

```python
from __future__ import annotations

from sdk import (
    ArticulatedObject,
    ArticulationType,
    BaseMode,
    Box,
    Cylinder,
    Inertial,
    Origin,
    RigidBodyProperties,
    ShelvingUnitGeometry,
    TestContext,
    TestReport,
    estimate_mass_from_density,
    mesh_from_geometry,
)

WIDTH = 0.90
DEPTH = 0.45
HEIGHT = 1.50
SHELF_THICKNESS = 0.028

STEEL_DENSITY = 7850.0
TINPLATE_DENSITY = 7850.0
PAPERBOARD_DENSITY = 700.0
GLASS_DENSITY = 2500.0

GONDOLA_PROXY = Box((WIDTH, DEPTH, HEIGHT))
GONDOLA_ORIGIN = Origin(xyz=(0.0, 0.0, HEIGHT / 2.0))
GONDOLA_MASS = estimate_mass_from_density(
    GONDOLA_PROXY,
    STEEL_DENSITY,
    fill_fraction=0.008,  # thin shelves, posts, back panel, and kick plinth
)

CAN_SHAPE = Cylinder(radius=0.035, length=0.14)
CAN_ORIGIN = Origin(xyz=(0.0, 0.0, 0.07))
CARTON_SHAPE = Box((0.09, 0.065, 0.18))
CARTON_ORIGIN = Origin(xyz=(0.0, 0.0, 0.09))
JAR_SHAPE = Cylinder(radius=0.040, length=0.11)
JAR_ORIGIN = Origin(xyz=(0.0, 0.0, 0.055))

SHELF_UNIT = ShelvingUnitGeometry(
    WIDTH,
    DEPTH,
    HEIGHT,
    shelf_count=3,
    shelf_thickness=SHELF_THICKNESS,
    back_panel=True,
    kick_height=0.10,
    shelf_lip=0.025,
    top_shelf=False,
)
SHELF_SURFACES = tuple(level + SHELF_THICKNESS for level in SHELF_UNIT.shelf_levels)

# name, shape, local origin, visual material, density, fill fraction, x, row
STOCK_LAYOUT = (
    ("can_01", CAN_SHAPE, CAN_ORIGIN, "blue_can", TINPLATE_DENSITY, 0.020, -0.25, 0),
    ("can_02", CAN_SHAPE, CAN_ORIGIN, "blue_can", TINPLATE_DENSITY, 0.020, 0.00, 0),
    ("can_03", CAN_SHAPE, CAN_ORIGIN, "blue_can", TINPLATE_DENSITY, 0.020, 0.25, 0),
    (
        "carton_01",
        CARTON_SHAPE,
        CARTON_ORIGIN,
        "cream_carton",
        PAPERBOARD_DENSITY,
        0.16,
        -0.16,
        1,
    ),
    (
        "carton_02",
        CARTON_SHAPE,
        CARTON_ORIGIN,
        "cream_carton",
        PAPERBOARD_DENSITY,
        0.16,
        0.16,
        1,
    ),
    ("jar_01", JAR_SHAPE, JAR_ORIGIN, "amber_glass", GLASS_DENSITY, 0.18, -0.25, 2),
    ("jar_02", JAR_SHAPE, JAR_ORIGIN, "amber_glass", GLASS_DENSITY, 0.18, 0.00, 2),
    ("jar_03", JAR_SHAPE, JAR_ORIGIN, "amber_glass", GLASS_DENSITY, 0.18, 0.25, 2),
)
STOCK_NAMES = tuple(spec[0] for spec in STOCK_LAYOUT)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(
        name="simulator_ready_stocked_supermarket_gondola",
        base_mode=BaseMode.FLOATING,
    )
    model.meta["mode"] = "static"

    shelving_paint = model.material(
        "powder_coated_shelving",
        rgba=(0.88, 0.88, 0.90, 1.0),
        metallic=0.0,
        roughness=0.50,
    )
    blue_can = model.material(
        "blue_can",
        rgba=(0.08, 0.28, 0.72, 1.0),
        metallic=0.15,
        roughness=0.38,
    )
    cream_carton = model.material(
        "cream_carton",
        rgba=(0.90, 0.78, 0.55, 1.0),
        metallic=0.0,
        roughness=0.72,
    )
    amber_glass = model.material(
        "amber_glass",
        rgba=(0.58, 0.25, 0.07, 1.0),
        metallic=0.0,
        roughness=0.14,
        transmission=0.35,
        ior=1.45,
    )
    lid_metal = model.material(
        "lid_metal",
        rgba=(0.32, 0.34, 0.36, 1.0),
        metallic=0.82,
        roughness=0.34,
    )
    visual_materials = {
        material.name: material
        for material in (blue_can, cream_carton, amber_glass)
    }

    model.physics_material(
        "shelf_contact",
        static_friction=0.62,
        dynamic_friction=0.48,
        restitution=0.02,
        density=STEEL_DENSITY,
    )
    model.physics_material(
        "stock_contact",
        static_friction=0.78,
        dynamic_friction=0.60,
        restitution=0.02,
    )

    gondola = model.part(
        "gondola",
        inertial=Inertial.from_geometry(
            GONDOLA_PROXY,
            mass=GONDOLA_MASS,
            origin=GONDOLA_ORIGIN,
        ),
        rigid_body=RigidBodyProperties(physics_material="shelf_contact"),
    )
    gondola.visual(
        mesh_from_geometry(SHELF_UNIT, "shelving_unit"),
        material=shelving_paint,
        name="shelving_unit",
    )

    for name, shape, local_origin, material_name, density, fill_fraction, x, row in STOCK_LAYOUT:
        item_mass = estimate_mass_from_density(
            shape,
            density,
            fill_fraction=fill_fraction,
        )
        item = model.part(
            name,
            inertial=Inertial.from_geometry(
                shape,
                mass=item_mass,
                origin=local_origin,
            ),
            rigid_body=RigidBodyProperties(physics_material="stock_contact"),
        )
        item.visual(
            shape,
            origin=local_origin,
            material=visual_materials[material_name],
            name=f"{name}_body",
        )
        if name.startswith("jar_"):
            item.visual(
                Cylinder(radius=0.041, length=0.008),
                origin=Origin(xyz=(0.0, 0.0, 0.108)),
                material=lid_metal,
                name=f"{name}_lid",
                collidable=False,
            )
        model.articulation(
            f"place_{name}",
            ArticulationType.FLOATING,
            parent=gondola,
            child=item,
            origin=Origin(xyz=(x, 0.02, SHELF_SURFACES[row])),
        )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    stock_parts = [object_model.get_part(name) for name in STOCK_NAMES]
    stock_edges = [
        joint for joint in object_model.articulations if joint.child in STOCK_NAMES
    ]
    ctx.check(
        "gondola_with_expected_individual_stock_count",
        object_model.base_mode == BaseMode.FLOATING
        and len(stock_parts) == len(STOCK_LAYOUT) == 8
        and len({part.name for part in stock_parts}) == 8,
        f"base_mode={object_model.base_mode!r}, stock_count={len(stock_parts)}",
    )
    ctx.check(
        "every_stock_item_is_an_independent_free_rigid_body",
        len(stock_edges) == len(stock_parts)
        and all(joint.articulation_type == ArticulationType.FLOATING for joint in stock_edges)
        and all(
            part.inertial is not None
            and part.inertial.mass > 0.0
            and part.rigid_body is not None
            and part.rigid_body.physics_material == "stock_contact"
            and any(visual.collidable for visual in part.visuals)
            for part in stock_parts
        ),
        "Each product needs its own FLOATING placement, mass, collider, and contact material.",
    )
    for spec, part in zip(STOCK_LAYOUT, stock_parts, strict=True):
        expected_surface = SHELF_SURFACES[spec[-1]]
        bounds = ctx.part_world_aabb(part)
        ctx.check(
            f"{part.name}_sits_on_its_shelf_surface",
            bounds is not None and abs(float(bounds[0][2]) - expected_surface) <= 1e-6,
            f"bounds={bounds!r}, expected_surface={expected_surface!r}",
        )
    ctx.fail_if_isolated_parts(
        name="all_stock_has_an_exact_contact_path_to_the_gondola",
    )
    ctx.fail_if_parts_overlap_in_current_pose(
        name="independent_stock_is_placed_without_interpenetration",
    )
    return ctx.report()


object_model = build_object_model()
```

Do not replace the FLOATING placement edges with FIXED joints: that would weld
the products to the gondola. Likewise, do not merge repeated products into a
row mesh or row rigid body. If a request explicitly says that a display item is
locked down, only that named item may use a FIXED relationship.
