# SDF Modeling: Smooth Organic Blends

## Purpose

Use SDF (signed-distance field) modeling when parts must **flow into each
other** with smooth transitions that booleans cannot produce: a faucet body
blending into its base, a chair shell, a teapot spout joining the body,
soap-dish curves, plush/molded forms. Boolean union leaves a hard crease at
the seam; `smooth_union` produces a continuous fillet-like blend.

For shapes that are just rounded versions of primitives, prefer
`smooth_geometry` (cheaper); reach for SDFs when the *junction between
shapes* must be organic.

## Import

```python
from sdk import (
    Sdf, mesh_from_sdf,
    sdf_sphere, sdf_box, sdf_cylinder, sdf_capsule, sdf_torus, sdf_ellipsoid,
)
```

## Convention

Fields are **negative inside, positive outside** (the standard SDF
convention). All primitives are centered at the origin; move them with
`.translate(...)`, `.rotate_x/y/z(angle)`, `.scale(factor)` (uniform only).

## Primitives

```python
sdf_sphere(radius)
sdf_box((sx, sy, sz), radius=0.0)   # radius rounds all edges
sdf_cylinder(radius, height)        # Z-aligned
sdf_capsule(height, radius)         # Z-aligned, rounded ends
sdf_torus(radius, tube)             # ring in the XY plane
sdf_ellipsoid((rx, ry, rz))
```

## Combining

```python
a.union(b)                  a.smooth_union(b, k)
a.difference(b)             a.smooth_difference(b, k)
a.intersection(b)           a.smooth_intersection(b, k)
a.round(radius)             # offset outward, rounds everything
a.shell(thickness)          # hollow into a wall of given thickness
```

`k` is the blend radius in meters — the size of the smooth transition zone.
Typical values: 0.005–0.02 for tabletop objects.

## Meshing

```python
geometry = mesh_from_sdf(field, edge_length=0.002)
```

`edge_length` is the surface resolution; aim for ~1/20 of the object size.
Bounds are derived automatically from the field's own bounding box. The
result is a watertight manifold MeshGeometry — booleans, `simplify_geometry`,
and `displace_geometry` all work on it. SDF meshes are dense; finish with
`simplify_geometry(geom, tolerance=0.0003)` to keep parts at a sane triangle
count.

## Example: faucet-like blend

```python
base = sdf_cylinder(0.025, 0.02).translate(0.0, 0.0, 0.01)
riser = sdf_capsule(0.10, 0.012).translate(0.0, 0.0, 0.06)
spout = (
    sdf_capsule(0.07, 0.009)
    .rotate_y(1.35)               # tilt towards -X
    .translate(-0.035, 0.0, 0.115)
)
body = base.smooth_union(riser, 0.012).smooth_union(spout, 0.010)
geometry = simplify_geometry(mesh_from_sdf(body, edge_length=0.0015), tolerance=0.0002)
```

The two `smooth_union` calls are what make this read as one cast-metal piece
instead of three glued cylinders.

## Performance notes

- Meshing cost scales with bounding volume / edge_length³. Keep bounds tight
  (they are by default) and do not use a finer edge_length than the smallest
  feature requires.
- Deeply nested fields evaluate every primitive per sample point; prefer a
  handful of well-placed primitives over dozens.
