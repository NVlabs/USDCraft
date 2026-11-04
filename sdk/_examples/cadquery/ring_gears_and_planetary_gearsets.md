---
title: 'Ring Gears and Planetary Gearsets'
description: 'Build an internal ring gear and a planetary gearset with the vendored `sdk` gear API.'
tags:
  - cadquery
  - examples
  - gear
  - ring
  - planetary
---
# Ring Gears and Planetary Gearsets

Use ring gear and planetary helpers to build internal teeth and static gear arrangements. Author runtime joints separately.

```python
import cadquery as cq

from sdk import PlanetaryGearset, RingGear

ring = RingGear(module=1.0, teeth_number=42, width=6.0, rim_width=3.0)
planetary = PlanetaryGearset(
    module=1.0,
    sun_teeth_number=12,
    planet_teeth_number=9,
    width=5.0,
    rim_width=3.0,
    n_planets=3,
)

result = cq.Assembly(name="ring-and-planetary")
result.add(
    ring.build(),
    name="ring",
    loc=cq.Location(cq.Vector(-35.0, 0.0, 0.0)),
)
result.add(
    planetary.build(),
    name="planetary",
    loc=cq.Location(cq.Vector(20.0, 0.0, 0.0)),
)
```
