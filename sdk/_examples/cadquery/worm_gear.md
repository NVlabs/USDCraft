---
title: 'Worm Gear'
description: 'Generate a worm gear body with the vendored `sdk` gear implementation.'
tags:
  - cadquery
  - examples
  - gear
  - worm
---
# Worm Gear

Build a worm as a CadQuery solid and attach it with `mesh_from_cadquery(...)`.

```python
from sdk import Worm

worm = Worm(module=1.0, lead_angle=20.0, n_threads=2, length=18.0, bore_d=4.0)
result = worm.build()
```
