# Inspecting model geometry

Use the task's check/compile interface and `TestContext` to inspect the authored
model. Query values at a decisive pose, then encode the lasting requirement in
`run_tests()`. Tool availability is defined by the current task.

## Bounds and poses

The following fragment assumes an existing `object_model` with a `drawer` Part
and a `drawer_slide` articulation. Units are metres for translation and radians
for rotation. Queries return `(minimum_xyz, maximum_xyz)` or `None` when geometry
cannot be resolved.

```python
from sdk import TestContext

ctx = TestContext(object_model)
rest_bounds = ctx.part_world_aabb("drawer")
with ctx.pose(drawer_slide=0.1):
    open_bounds = ctx.part_world_aabb("drawer")
    panel_bounds = ctx.part_element_world_aabb("drawer", elem="front_panel")
```

`part_world_position(...)` returns a Part's origin, not its geometry center.
Bounds are in the current world pose, including Part and Visual transforms.
For a bound pair `(lo, hi)`, size is `hi[i] - lo[i]` and center is
`(hi[i] + lo[i]) / 2` on each axis. Check for `None` before using them.

## Relationships

- `ctx.expect_gap(...)` checks a signed directional gap.
- `ctx.expect_contact(...)` checks minimum geometry distance; inspect penetration separately.
- `ctx.expect_overlap(...)` checks projected overlap on selected axes.
- `ctx.expect_within(...)` checks containment with allowed outward slack.
- `ctx.fail_if_part_contains_disconnected_geometry_islands(...)` diagnoses islands.

Use exact named elements for a local interface. AABB overlap does not prove a
3D collision, and projected containment does not prove physical support. Read
`80_testing.md` for the exact method signatures and allowance scope. Interpret
diagnostics against the intended geometry before moving parts or changing tolerances.
