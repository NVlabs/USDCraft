# Organic Surfaces: Smoothing, Subdivision, and Displacement

## Purpose

Use this page when an object should read as soft, rounded, hand-made, or
irregular rather than machined: ceramics, cushions, food, molded plastics,
worn household items. These helpers transform any `MeshGeometry` (including
boolean results) into organic-looking surfaces. For hard-edged mechanical
shapes, the regular primitives and CadQuery fillets remain the better tools.

## Import

```python
from sdk import (
    subdivide_geometry,
    smooth_geometry,
    taubin_smooth_geometry,
    displace_geometry,
    simplify_geometry,
)
```

All functions return a new `MeshGeometry` and leave the input unchanged.
All distances are in meters.

## `smooth_geometry` — round off a faceted solid

```python
smooth_geometry(
    geometry: MeshGeometry,
    *,
    min_sharp_angle: float = 60.0,   # degrees; edges sharper than this stay sharp
    min_smoothness: float = 0.0,
    subdivisions: int = 3,           # segments per original edge
) -> MeshGeometry
```

The workhorse for rounded forms. Edges whose dihedral angle exceeds
`min_sharp_angle` are preserved; everything else is interpolated with smooth
tangents. The input must be a valid manifold solid (the same requirement as
`boolean_union`). Higher `min_sharp_angle` allows more edges to smooth. At zero,
edges remain faceted; at 180 degrees, every edge can smooth. `min_smoothness > 0`
also softens edges above the threshold.

Smoothing interpolates through the original vertices, so curved regions can
bulge slightly *outside* the original silhouette (a 90-degree edge bows
outward, it does not fillet inward). Budget a small margin when the part must
clear a neighbor, and re-check overlaps after smoothing. Coarse inputs bulge
more; increase `radial_segments` on cylinders/spheres before smoothing.

```python
# A soft-edged seat cushion from a box
cushion = smooth_geometry(
    BoxGeometry((0.45, 0.45, 0.10)),
    min_sharp_angle=180.0,
    subdivisions=4,
)
```

## `subdivide_geometry` — densify before deforming

```python
subdivide_geometry(geometry: MeshGeometry, *, levels: int = 1) -> MeshGeometry
```

Midpoint-splits every triangle into 4 per level without changing the shape.
Use it to give `taubin_smooth_geometry` or `displace_geometry` enough
vertices to act on. Each level multiplies triangle count by 4; 2–3 levels is
usually enough.

## `taubin_smooth_geometry` — relax without shrinking

```python
taubin_smooth_geometry(
    geometry: MeshGeometry,
    *,
    iterations: int = 10,
    lambda_factor: float = 0.5,
    mu_factor: float = -0.53,
) -> MeshGeometry
```

Taubin lambda/mu smoothing relaxes the mesh towards an organic surface while
avoiding the volume loss of plain Laplacian smoothing. Unlike
`smooth_geometry` it does not require a manifold solid and never adds
triangles — subdivide first when the input is coarse. Good for taking the
"CSG look" off merged shapes.

## `displace_geometry` — procedural surface irregularity

```python
displace_geometry(
    geometry: MeshGeometry,
    *,
    amplitude: float,        # peak offset in meters, e.g. 0.001–0.005
    scale: float = 0.05,     # feature size in meters
    seed: int = 0,
    octaves: int = 1,        # 2-3 adds finer detail on top
) -> MeshGeometry
```

Pushes vertices along their normals with deterministic value noise: same
inputs and seed always give the same mesh. Use small amplitudes — 1–3 mm of
displacement is enough to make bread, ceramic glaze, or hammered metal read
as organic. Subdivide first; noise cannot show detail finer than the mesh.

```python
# Rustic bread loaf
loaf = SphereGeometry(0.09).scale(1.0, 0.7, 0.55)
loaf = subdivide_geometry(loaf, levels=1)
loaf = displace_geometry(loaf, amplitude=0.004, scale=0.03, seed=12, octaves=2)
```

## `simplify_geometry` — tame dense outputs

```python
simplify_geometry(geometry: MeshGeometry, *, tolerance: float) -> MeshGeometry
```

Reduces triangle count while keeping the surface within `tolerance` meters of
the original. Apply as the final step when smoothing/displacement produced a
heavier mesh than the part needs (target a few thousand triangles per part).
Requires a valid manifold solid.

## Recommended pipelines

- Rounded solid: `smooth_geometry(base, min_sharp_angle=..., subdivisions=3)`
- Organic/no-manifold-guarantee: `subdivide_geometry` → `taubin_smooth_geometry`
- Irregular surface: `subdivide_geometry` → `displace_geometry`
- Heavy result: append `simplify_geometry(..., tolerance=0.0005)`

Displacement breaks exact coplanarity, so apply it after any boolean
operations, not before.
