---
title: 'Bevel Gears'
description: 'Create a standalone bevel gear and a meshing bevel pair from the vendored `sdk` gear classes.'
tags:
  - cadquery
  - examples
  - gear
  - bevel
---
# Bevel Gears

Bevel gears and meshing bevel pairs build as regular CadQuery solids and compounds, so they can be dropped into the same authoring flow as the rest of `sdk`. The gear classes are authored in millimetres here and converted to metres by `mesh_from_cadquery(..., unit_scale=0.001)`: their internal construction tolerances (wire combining, spline approximation and shell sewing at 1e-2 model units) assume a millimetre-scale module, and a metre-scale `module=0.001` bevel gear makes the cone sewing degenerate — it runs for minutes and exhausts memory instead of building.

```python
import cadquery as cq

from sdk import BevelGear, BevelGearPair, mesh_from_cadquery

single = BevelGear(module=1.0, teeth_number=18, cone_angle=45.0, face_width=4.0)
pair = BevelGearPair(
    module=1.0,
    gear_teeth=24,
    pinion_teeth=16,
    face_width=4.0,
    axis_angle=90.0,
)

result = cq.Assembly(name="bevel-gears")
result.add(
    single.build(),
    name="single",
    loc=cq.Location(cq.Vector(-18.0, 0.0, 0.0)),
)
result.add(
    pair.build(),
    name="pair",
    loc=cq.Location(cq.Vector(18.0, 0.0, 0.0)),
)

mesh = mesh_from_cadquery(result, "bevel_gears", unit_scale=0.001)
```
