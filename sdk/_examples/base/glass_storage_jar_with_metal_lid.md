---
title: 'Glass Storage Jar with Metal Lid'
description: 'Static kitchen vessel: JarGeometry body with transmission glass PBR and a brushed-metal CapGeometry lid seated on the mouth ring.'
tags:
  - sdk
  - base sdk
  - static
  - kitchen
  - jar
  - container
  - vessel
  - canister
  - glass
  - lid
  - cap
  - jargeometry
  - capgeometry
  - pbr
  - transmission
---
# Glass Storage Jar with Metal Lid

This base-SDK example is the minimal reference for the container family:
a `JarGeometry` glass body plus a `CapGeometry` lid, with the explicit PBR
fields that make glass and brushed metal read correctly in a physically lit
renderer.

Geometry example. Adapt the shape to the current task; author mass, contact properties and required motion using the foundational pages before completing the asset.

```python
from __future__ import annotations

from sdk import (
    ArticulatedObject,
    CapGeometry,
    Cylinder,
    Inertial,
    JarGeometry,
    TestContext,
    TestReport,
    mesh_from_geometry,
)

JAR_HEIGHT = 0.14
JAR_RADIUS = 0.045
MOUTH_RADIUS = 0.035
LID_HEIGHT = 0.016
LID_OVERLAP = 0.006


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="glass_storage_jar_with_metal_lid")
    model.meta["mode"] = "static"

    glass = model.material(
        "clear_jar_glass",
        rgba=(0.82, 0.88, 0.86, 1.0),
        metallic=0.0,
        roughness=0.05,
        transmission=0.92,
        ior=1.45,
    )
    lid_metal = model.material(
        "brushed_lid_steel",
        rgba=(0.62, 0.62, 0.64, 1.0),
        metallic=0.9,
        roughness=0.38,
    )

    jar = model.part("jar")
    body = JarGeometry(
        height=JAR_HEIGHT,
        radius=JAR_RADIUS,
        mouth_radius=MOUTH_RADIUS,
        shoulder_height=0.02,
        base_fillet=0.004,
    )
    jar.visual(mesh_from_geometry(body, "jar_body"), material=glass, name="jar_body")

    lid = CapGeometry(
        radius=MOUTH_RADIUS + 0.0025,
        height=LID_HEIGHT,
        top_fillet=0.004,
    ).translate(0.0, 0.0, JAR_HEIGHT - LID_OVERLAP)
    jar.visual(mesh_from_geometry(lid, "jar_lid"), material=lid_metal, name="jar_lid")

    jar.inertial = Inertial.from_geometry(
        Cylinder(radius=JAR_RADIUS, length=JAR_HEIGHT + LID_HEIGHT - LID_OVERLAP),
        mass=0.5,
    )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    jar = object_model.get_part("jar")
    ctx.check("jar_present", jar is not None, "Expected a jar part.")
    if jar is None:
        return ctx.report()

    aabb = ctx.part_world_aabb(jar)
    ctx.check("jar_aabb_present", aabb is not None, "Expected a world AABB.")
    if aabb is None:
        return ctx.report()

    mins, maxs = aabb
    size = tuple(float(maxs[i] - mins[i]) for i in range(3))
    total_height = JAR_HEIGHT - LID_OVERLAP + LID_HEIGHT
    ctx.check("jar_diameter", abs(size[0] - JAR_RADIUS * 2.0) <= 0.01, f"size={size!r}")
    ctx.check("jar_height", abs(size[2] - total_height) <= 0.005, f"size={size!r}")
    return ctx.report()


object_model = build_object_model()
```
