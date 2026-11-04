# Wires, Tubes, and Frames

## Purpose

Use this stack for thin curved parts such as handles, loops, whisk cages, fan
guards, baskets, and tubular frames. Start with the spline helpers for almost
all continuously bent parts.

These helpers create static geometry. Flexible hoses/cables use one
`BodyModel.FLEXIBLE_1D` Part in a rest shape with a `FIXED` attachment. They do not
simulate bending or load transfer; keep rigid fittings in rigid Parts. See
`55_common_assemblies.md` for the attachment pattern.

## Import

```python
from sdk import (
    SweepSection,
    WirePath,
    mesh_health,
    tube_from_spline_points,
    sweep_profile_along_spline,
    beam_between,
    rounded_rect_profile,
)
```

## Recommended APIs

| Shape intent | Helper |
| --- | --- |
| Default for smooth circular rails, handles, loops, and frames | `tube_from_spline_points(...)` |
| Default for smooth non-circular rails and trim | `sweep_profile_along_spline(...)` |
| Straight diagonal/angled rectangular member (ladder rail, A-frame leg, rafter, truss) | `beam_between(...)` |
| Straight diagonal round strut | `tube_from_spline_points([start, end], radius=...)` |
| Readable manual path authoring for smooth arcs and bezier-like runs | `WirePath` plus a spline/sweep helper |

## API Reference

### `tube_from_spline_points(...)`

```python
tube_from_spline_points(
    points,
    *,
    radius: float,
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    radial_segments: int = 16,
    cap_ends: bool = True,
    up_hint=(0.0, 0.0, 1.0),
    min_segment_length: float = 1e-6,
) -> MeshGeometry
```

- Fits a spline through the input points, then builds a circular tube.
- `spline`: `"catmull_rom"` or `"bezier"`.
- `alpha`: Catmull-Rom parameter, used only for `"catmull_rom"`.
- `closed_spline`: closes the fitted path.
- `cap_ends`: adds end caps when the path is open.

#### Routing a hose, cable, or wire

Fit a rest route to its endpoints, outlet directions and visible bends. A hanging
run should sag between supports; a loop or clipped route may rise where supported.
Check spline extrema and clearance because interpolation can overshoot the control
points. The mesh describes this pose only, without flexible-body dynamics.

### `sweep_profile_along_spline(...)`

```python
sweep_profile_along_spline(
    points,
    *,
    profile,
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    cap_profile: bool = True,
    up_hint=(0.0, 0.0, 1.0),
    min_segment_length: float = 1e-6,
    sections=(),
    section_interpolation="linear",
    section_tension=0.0,
    max_segment_length=None,
) -> MeshGeometry
```

- Fits a spline through the input points, then sweeps a closed 2D profile.
- Use this when the section is not circular.

### Variable-section sweeps

Use `SweepSection` when the local profile must taper, widen, flatten, twist, or
move off the path center:

```python
SweepSection(
    position=0.5,          # normalized sampled-centerline arc length, 0..1
    profile=None,          # optional replacement 2D profile
    scale=(1.4, 0.8),     # local profile X/Y scale
    rotation=0.25,        # radians in the profile plane
    offset=(0.002, 0.0),  # local profile-plane offset in meters
    interpolation="catmull_rom",
    tension=0.15,
)
```

`sections` is supported by `SweepGeometry`, `PipeGeometry`,
`ArcPipeGeometry`, and `sweep_profile_along_spline(...)`. On an open path,
missing endpoint controls use the base profile. On a closed path, a missing
position-1 control copies the position-0 control exactly. Explicit controls at
0 and 1 must produce the same **ordered transformed profile**. A cyclic shift
of the raw profile's starting vertex is accepted as relabeling and aligned
before transforms, but a net symmetry twist such as a square ending at
`rotation=pi / 2` is rejected: the mesh generator does not permute vertices at
the path seam. An explicit full turn such as `0 -> 2*pi` is supported because
its ordered endpoint returns to the same orientation, and internal twist is
supported when it returns to the seam orientation. Profiles with different
point counts are aligned and sampled at the union of their normalized perimeter
knots, preserving every authored control-profile corner. Every section position
inserts an exact path ring. Positions must be unique and strictly ordered after
normalization. They use cumulative polyline arc length over the actual sweep
rings; for spline helpers this is the sampled centerline, not an analytic spline
arc-length solution.

Use `section_interpolation="linear"` for controlled straight transitions or
`"catmull_rom"` for smooth profile evolution. `section_tension` and each
section's optional override are in `0..1`; a per-section override controls the
interval following that section. Raw profile coordinates, scale, rotation, and
offset are interpolated separately, then transformed at each generated ring;
this preserves section area during a pure rotation. Rotation values are not
wrapped to a shortest path, so `0 -> 2*pi` authors one full turn. Catmull-Rom
scale overshoot that becomes non-positive is rejected explicitly.
For a truly C1-periodic closed Catmull-Rom seam, use the same raw endpoint
profile, scale, and offset, with rotation differing only by whole `2*pi` turns.
Different parameter decompositions that happen to produce the same endpoint
polygon guarantee positional closure only, not a matching seam derivative.

For a sparse raw path, set `max_segment_length` in meters so interpolation has
enough intermediate rings to appear smooth. It subdivides straight path
segments; it does not turn a polyline into a curved centerline. Sweep generation
refuses any request that would exceed 250,000 output `MeshGeometry` vertices;
increase `max_segment_length`, reduce path samples, or simplify profiles when
that output-object budget is exceeded. Profile alignment is also bounded; reduce section count or simplify
source profiles if that bound is reached.

Example: a handle with thin mounts, a wider grip, and mild twist:

```python
handle = sweep_profile_along_spline(
    [
        (-0.05, 0.00, 0.00),
        (-0.03, 0.03, 0.02),
        (0.03, 0.03, 0.02),
        (0.05, 0.00, 0.00),
    ],
    profile=rounded_rect_profile(0.010, 0.006, radius=0.002),
    sections=(
        SweepSection(0.0, scale=0.65),
        SweepSection(0.5, scale=(1.35, 1.0), rotation=0.15),
        SweepSection(1.0, scale=0.65),
    ),
    section_interpolation="catmull_rom",
    max_segment_length=0.006,
    samples_per_segment=16,
)
```

Variable-section sweep changes real triangle geometry. Aggressive profile
changes or Catmull-Rom overshoot can create local folding or self-intersection;
inspect renders and use a targeted intersection probe as well as
`mesh_health(...)`. Mesh health reports degeneracy, topology, winding, and
closure, but does not detect general triangle-triangle self-intersection. Use
`section_loft(...)` when exact CAD sections, guide curves, or dimension-critical
surfaces matter more than this direct mesh workflow.

### `beam_between(...)`

```python
beam_between(
    start,
    end,
    *,
    width: float,
    height: float | None = None,   # defaults to a square section
    cap_ends: bool = True,
    up_hint=(0.0, 0.0, 1.0),
) -> MeshGeometry
```

- Straight rectangular-section beam whose endpoints are baked in world
  coordinates. Place it directly — **do not compute any rotation yourself.**
- This is the right tool for the angled members that otherwise tempt
  hand-derived `rpy`/`asin`/`atan2` math: ladder rails, A-frame legs, roof
  rafters, trusses, railings. Getting that math wrong silently flips a member to
  the mirror angle, so prefer this helper.
- `up_hint` rolls the section about the beam axis; change it only if a flat
  member comes out edge-on.

### `WirePath`

```python
WirePath(start)
WirePath.from_points(points) -> WirePath
```

Methods:

```python
wp.line_to(point) -> WirePath
wp.line_by(dx, dy, dz) -> WirePath
wp.bezier_to(control1, control2, end, *, samples=12) -> WirePath
wp.arc(*, center, normal, angle, segments=16) -> WirePath
wp.extend(points) -> WirePath
wp.to_points() -> list[tuple[float, float, float]]
```

Use `WirePath` when readability matters more than fitting a spline from an
existing point set. In most cases, finish with `wp.to_points()` and feed the
result into `tube_from_spline_points(...)` or `sweep_profile_along_spline(...)`
so the final visible part remains a smooth swept tube.

## Advice

### Default decision order

- If the part should read as one continuously bent tube or wire, start with
  `tube_from_spline_points(...)`.
- If the part should read as one continuously bent rail but the section is not
  circular, use `sweep_profile_along_spline(...)`.
- Use `WirePath` when the path is easier to author procedurally than to specify
  as a finished point list, then pass `wp.to_points()` into a spline/sweep
  helper.

### Tuning smoothness

- Increase `samples_per_segment` first when a spline-based result looks coarse.
- Increase `radial_segments` when the tube still looks visibly faceted.

## Examples

### Smooth handle

```python
handle = tube_from_spline_points(
    [
        (-0.03, 0.00, 0.00),
        (-0.01, 0.02, 0.01),
        (0.01, 0.02, 0.01),
        (0.03, 0.00, 0.00),
    ],
    radius=0.002,
    samples_per_segment=18,
    radial_segments=20,
    cap_ends=True,
)
```

### Custom-profile rail

```python
trim = sweep_profile_along_spline(
    [
        (-0.04, 0.00, 0.00),
        (-0.01, 0.02, 0.01),
        (0.02, 0.01, 0.01),
        (0.05, 0.00, 0.00),
    ],
    profile=rounded_rect_profile(0.004, 0.002, radius=0.0007),
    samples_per_segment=18,
    cap_profile=True,
)
```

### Angled frame members with `beam_between`

```python
# A-frame: two legs meeting at an apex, baked endpoints — no rotation math.
apex = (0.0, 0.0, 1.2)
left_leg = beam_between((-0.5, 0.0, 0.0), apex, width=0.06)
right_leg = beam_between((0.5, 0.0, 0.0), apex, width=0.06)
crossbar = beam_between((-0.25, 0.0, 0.6), (0.25, 0.0, 0.6), width=0.05)
```

### Readable smooth path with `WirePath`

```python
wp = (
    WirePath((-0.03, 0.00, 0.00))
    .bezier_to(
        (-0.02, 0.02, 0.01),
        (0.02, 0.02, 0.01),
        (0.03, 0.00, 0.00),
        samples=18,
    )
)

handle = tube_from_spline_points(
    wp.to_points(),
    radius=0.002,
    samples_per_segment=6,
    radial_segments=20,
    cap_ends=True,
)
```

Use this pattern when arcs or bezier-like runs are easier to author
procedurally but the final part should still read as one continuous bend.

## See Also

- `40_mesh_geometry.md` for general procedural mesh authoring

## Clarifications for agent usage

- Spline/path angle conventions use radians and the right-hand rule.
- `spline="bezier"` expects chained cubic Bezier control points rather than a generic interpolating spline-through-points contract. For an open chain, `(n - 1) % 3 == 0`; for a closed chain, the authored point list must already close back on itself.
- `up_hint` defines the transported frame used to orient swept profiles. Changing it changes the profile roll even when the path points stay fixed.
