---
title: 'Labeled Spray Bottle with Decal'
description: 'Static cleaning-product bottle whose printed front label is authored as a TextureCanvas and wrapped onto the body with Part.decal — not modeled as geometry.'
tags:
  - sdk
  - base sdk
  - static
  - kitchen
  - cleaning
  - bottle
  - spray bottle
  - label
  - decal
  - texture
  - texturecanvas
  - surface graphics
  - printed text
---
# Labeled Spray Bottle with Decal

This base-SDK example is the reference for printed surface graphics. The
bottle body and cap are geometry; the front label (color blocks + text) is a
`TextureCanvas` wrapped onto the cylindrical body with `Part.decal`. The label
is visual-only — it is never collision geometry, and no text is modeled as
extruded shapes.

Geometry example. Adapt the shape to the current task; author mass, contact properties and required motion using the foundational pages before completing the asset.

```python
from __future__ import annotations

from sdk import (
    ArticulatedObject,
    BottleGeometry,
    Box,
    CapGeometry,
    Inertial,
    Origin,
    TestContext,
    TestReport,
    TextureCanvas,
    mesh_from_geometry,
)

BODY_RADIUS = 0.035
HEIGHT = 0.22


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="labeled_spray_bottle")
    model.meta["mode"] = "static"

    bottle_plastic = model.material(
        "bottle_plastic",
        rgba=(0.92, 0.94, 0.96, 1.0),
        metallic=0.0,
        roughness=0.35,
    )
    cap_plastic = model.material(
        "cap_plastic",
        rgba=(0.10, 0.30, 0.55, 1.0),
        metallic=0.0,
        roughness=0.45,
    )

    bottle = model.part("bottle")
    bottle.visual(
        mesh_from_geometry(
            BottleGeometry(HEIGHT, BODY_RADIUS, neck_radius=0.014, neck_height=0.02), "body"
        ),
        material=bottle_plastic,
    )
    cap = CapGeometry(radius=0.016, height=0.03).translate(0.0, 0.0, HEIGHT - 0.004)
    bottle.visual(mesh_from_geometry(cap, "cap"), material=cap_plastic)
    bottle.inertial = Inertial.from_geometry(
        Box((BODY_RADIUS * 2.0, BODY_RADIUS * 2.0, HEIGHT)), mass=0.3
    )

    # Printed front label, authored in code and wrapped onto the body.
    label = TextureCanvas(360, 480, background=(0.95, 0.96, 0.98, 1.0))
    label.rect((0.0, 0.0, 1.0, 0.30), fill=(0.10, 0.30, 0.55, 1.0))  # blue header band
    label.text((0.5, 0.15), "CLEAN", size=0.13, fill=(1, 1, 1, 1), anchor="mm", bold=True)
    label.text((0.5, 0.46), "All-Purpose", size=0.07, fill=(0.1, 0.1, 0.1, 1), anchor="mm")
    label.rect((0.1, 0.6, 0.8, 0.02), fill=(0.10, 0.30, 0.55, 1.0))  # accent rule
    label.text((0.5, 0.74), "500 mL", size=0.06, fill=(0.2, 0.2, 0.2, 1), anchor="mm")

    bottle.decal(
        label,
        width=0.13,
        height=0.10,
        curvature_radius=BODY_RADIUS,
        origin=Origin(xyz=(0.0, 0.0, 0.10)),
        name="front_label",
    )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    bottle = object_model.get_part("bottle")
    ctx.check("bottle_present", bottle is not None, "Expected a bottle part.")
    if bottle is None:
        return ctx.report()

    label = bottle.get_visual("front_label")
    ctx.check("label_present", label is not None, "Expected a front_label decal.")
    ctx.check(
        "label_is_visual_only",
        label is not None and label.collidable is False,
        "Decal labels must be visual-only (collidable=False).",
    )

    aabb = ctx.part_world_aabb(bottle)
    ctx.check("bottle_aabb_present", aabb is not None, "Expected a world AABB.")
    if aabb is None:
        return ctx.report()
    mins, maxs = aabb
    size = tuple(float(maxs[i] - mins[i]) for i in range(3))
    ctx.check("bottle_height", 0.20 <= size[2] <= 0.26, f"size={size!r}")
    return ctx.report()


object_model = build_object_model()
```
