---
title: 'Rustic Ceramic Mug with Glazed Finish'
description: 'Static kitchen object showing the organic-surface pipeline: boolean shell, smooth_geometry, subtle displace_geometry, and an explicit PBR glaze material.'
tags:
  - sdk
  - base sdk
  - static
  - kitchen
  - mug
  - cup
  - ceramic
  - stoneware
  - organic
  - smoothing
  - smooth_geometry
  - displace_geometry
  - subdivide_geometry
  - pbr
  - material
  - glaze
  - mesh geometry
---
# Rustic Ceramic Mug with Glazed Finish

This base-SDK example is the reference for static, organic household objects.
It shows the recommended pipeline for ceramics and similar hand-made forms:

1. Build the blank with primitives and booleans.
2. Round it with `smooth_geometry` (the rim and base lose their machined edges).
3. Add subtle `displace_geometry` noise so the surface reads as stoneware.
4. Declare an explicit PBR material instead of relying on name heuristics.

Geometry example. Adapt the shape to the current task; author mass, contact properties and required motion using the foundational pages before completing the asset.

```python
from __future__ import annotations

from math import pi

from sdk import (
    ArticulatedObject,
    Box,
    CylinderGeometry,
    Inertial,
    TestContext,
    TestReport,
    TorusGeometry,
    boolean_difference,
    displace_geometry,
    mesh_from_geometry,
    smooth_geometry,
)

OUTER_RADIUS = 0.042
HEIGHT = 0.095
WALL = 0.004


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="rustic_ceramic_mug")
    model.meta["mode"] = "static"

    glaze = model.material(
        "stoneware_glaze",
        rgba=(0.62, 0.45, 0.36, 1.0),
        metallic=0.0,
        roughness=0.35,
        clearcoat=0.45,
        clearcoat_roughness=0.25,
    )

    # Blank: outer cylinder minus an inner cavity that pierces the top rim.
    outer = CylinderGeometry(OUTER_RADIUS, HEIGHT, radial_segments=48).translate(
        0.0, 0.0, HEIGHT / 2.0
    )
    cavity = CylinderGeometry(OUTER_RADIUS - WALL, HEIGHT, radial_segments=48).translate(
        0.0, 0.0, HEIGHT / 2.0 + WALL
    )
    body = boolean_difference(outer, cavity)

    # Round the rim/base edges, then add gentle stoneware irregularity.
    body = smooth_geometry(body, min_sharp_angle=95.0, subdivisions=2)
    body = displace_geometry(body, amplitude=0.0006, scale=0.025, seed=11, octaves=2)

    # Handle: a torus standing in the XZ plane, embedded into the wall.
    handle = (
        TorusGeometry(0.028, 0.0055, radial_segments=14, tubular_segments=36)
        .rotate_x(pi / 2.0)
        .translate(OUTER_RADIUS + 0.016, 0.0, HEIGHT * 0.52)
    )

    mug = model.part("mug")
    mug.visual(mesh_from_geometry(body, "mug_body"), material=glaze, name="mug_body")
    mug.visual(mesh_from_geometry(handle, "mug_handle"), material=glaze, name="mug_handle")
    mug.inertial = Inertial.from_geometry(
        Box((OUTER_RADIUS * 2.0, OUTER_RADIUS * 2.0, HEIGHT)), mass=0.35
    )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    mug = object_model.get_part("mug")
    ctx.check("mug_present", mug is not None, "Expected a mug part.")
    if mug is None:
        return ctx.report()

    aabb = ctx.part_world_aabb(mug)
    ctx.check("mug_aabb_present", aabb is not None, "Expected a world AABB for the mug.")
    if aabb is None:
        return ctx.report()

    mins, maxs = aabb
    size = tuple(float(maxs[i] - mins[i]) for i in range(3))
    ctx.check("mug_height", 0.085 <= size[2] <= 0.105, f"size={size!r}")
    ctx.check("mug_handle_reach", size[0] >= OUTER_RADIUS * 2.0 + 0.01, f"size={size!r}")
    return ctx.report()


object_model = build_object_model()
```
