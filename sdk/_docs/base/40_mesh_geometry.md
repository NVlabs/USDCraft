# Procedural Meshes

## Purpose

Use this page for low-level procedural mesh authoring: primitive solids,
lathed forms, extrusions, sweeps, profiles, shell helpers, and mesh export.
If the shape matches a semantic family such as a knob, bezel, wheel, fan, or
panel, prefer the focused geometry-family references listed below.

## Import

```python
from sdk import (
    MeshGeometry,
    MeshHealthReport,
    SnapRefused,
    BoxGeometry,
    CapsuleGeometry,
    CylinderGeometry,
    ConeGeometry,
    DomeGeometry,
    SphereGeometry,
    TorusGeometry,
    LatheGeometry,
    LoftGeometry,
    ExtrudeGeometry,
    ExtrudeWithHolesGeometry,
    SweepGeometry,
    mesh_health,
    snap_to,
    weld,
    smooth_difference,
    rounded_rect_profile,
    superellipse_profile,
    sample_catmull_rom_spline_2d,
    sample_cubic_bezier_spline_2d,
    superellipse_side_loft,
    split_superellipse_side_loft,
    resample_side_sections,
    cut_opening_on_face,
    mesh_from_geometry,
)
```

## Recommended Surface

- `MeshGeometry`
- validation and diagnostics: `MeshGeometry.validate()`, `mesh_health(...)`
- primitive builders: `BoxGeometry`, `CylinderGeometry`, `ConeGeometry`,
  `SphereGeometry`, `DomeGeometry`, `CapsuleGeometry`, `TorusGeometry`
- loft/extrude/sweep builders: `LatheGeometry`, `LoftGeometry`,
  `ExtrudeGeometry`, `ExtrudeWithHolesGeometry`, `SweepGeometry`
- profile helpers: `rounded_rect_profile`, `superellipse_profile`
- spline/profile helpers: `sample_catmull_rom_spline_2d(...)`,
  `sample_cubic_bezier_spline_2d(...)`
- shell helpers: `superellipse_side_loft(...)`,
  `split_superellipse_side_loft(...)`, `resample_side_sections(...)`
- panel opening helper: `cut_opening_on_face(...)`
- placement/blending helpers: `snap_to(...)`, `weld(...)`,
  `smooth_difference(...)`
- export helper: `mesh_from_geometry(...)`

Focused semantic geometry references:

- `41_panels_and_grilles.md` for perforated, slotted, and vent-grille faces
- `42_brackets_and_mounts.md` for clevises, forks, and yokes
- `43_fans_and_rotors.md` for axial rotors and blower wheels
- `44_knobs_and_controls.md` for knobs and control caps
- `47_bezels_and_frames.md` for bezels, trims, and framed openings
- `48_wheels_and_tires.md` for wheels and tire visuals
- `49_hinges.md` for exposed utility hinge geometry

## Core Mesh Type

### `MeshGeometry`

```python
MeshGeometry(
    vertices: list[tuple[float, float, float]] = [],
    faces: list[tuple[int, int, int]] = [],
    uvs: list[tuple[float, float]] | None = None,
    texture: object | None = None,
    normal_crease_deg: float | None = None,
)
```

Methods:

```python
geom.add_vertex(x, y, z) -> int
geom.add_face(a, b, c) -> None
geom.validate() -> MeshGeometry
geom.health(*, tolerance=1e-9) -> MeshHealthReport
geom.copy() -> MeshGeometry
geom.clone() -> MeshGeometry
geom.merge(other) -> MeshGeometry
geom.translate(dx, dy, dz) -> MeshGeometry
geom.scale(sx, sy=None, sz=None) -> MeshGeometry
geom.rotate(axis, angle, origin=(0.0, 0.0, 0.0)) -> MeshGeometry
geom.rotate_x(angle) -> MeshGeometry
geom.rotate_y(angle) -> MeshGeometry
geom.rotate_z(angle) -> MeshGeometry
geom.set_normal_crease_deg(degrees_or_none) -> MeshGeometry
geom.use_hard_normals() -> MeshGeometry
geom.use_smooth_normals() -> MeshGeometry
geom.to_obj() -> str
geom.save_obj(path) -> None
```

- Vertices are 3D points in meters.
- Faces are 0-based triangle indices.
- Construction, `validate()`, managed materialization, and OBJ serialization
  reject non-finite vertices, non-triangle/non-integer faces, repeated corners,
  out-of-range indices, and misaligned UV arrays.
- OBJ serialization starts at six decimal places and deterministically raises
  precision when necessary to preserve a valid small triangle; a face whose
  source coordinates are already degenerate is rejected.
- All transforms and `merge(...)` mutate the receiver in place and return
  `self`; `merge(...)` validates but does not mutate its source geometry.

### Mesh health diagnostics

```python
report = mesh_health(geometry)
for finding in report.findings:
    print(finding.issue, finding.severity, finding.count, finding.bounds)
```

`mesh_health(...)` is a non-mutating authoring diagnostic. It reports empty or
degenerate meshes, sliver and duplicate triangles, duplicate/unused vertices,
boundary and non-manifold edges or vertices, inconsistent winding, disconnected
surface components, closure, and signed orientation. Findings include counts,
up to 16 sample indices, and an affected bounding region when one can be
localized. Tolerance and bounds use meters; `signed_volume` uses cubic meters.

`surface_component_count` is the number of edge-connected triangle shells,
including inward cavity boundaries. `component_count` counts closed,
positive-volume solids, so one outward shell containing a correctly oriented
inward cavity remains one solid. `watertight` reports edge closure and
`winding_consistent` reports compatible shared-edge orientation. For nested
closed shells, valid orientation alternates with containment depth: outer
material positive, cavity negative, material island positive, and so on.

`report.healthy` means no structurally unsafe error finding; `report.clean`
means no finding of any severity. An open presentation surface or intentionally
separate visual islands can therefore remain inspectable warnings. Use it to locate structural defects; it does not establish simulator behavior.

The report is a topology/quality diagnostic, not a general triangle-triangle
self-intersection proof. For aggressive deformation or section interpolation,
also use the inspection tools available in the task to check intersections before
choosing the final geometry.

### Normal crease policy

`normal_crease_deg` controls only rendered normals:

- `None`: existing 60-degree default;
- `0`: hard normal at every triangle corner;
- `180`: smooth across every edge-connected incident face fan at a vertex;
- an intermediate value: smooth only across adjacent faces within that angle.

This controls shading, not silhouette, collision, mass or contact. Highly connected
vertices can exceed the normal-generation budget; simplify that topology if a warning
is reported.

Copying, transforms, placement rebuilds, and single-input smoothing preserve
the policy. Difference operations inherit the base. Merge, union,
intersection, and `weld(...)` retain it only when all effective input policies
agree; otherwise they deliberately return to `None`, the 60-degree default.
Use a CAD fillet, bevel, or subdivision when the physical shape or silhouette
must actually become round.

## Primitive Builders

```python
BoxGeometry(size)
RoundedBoxGeometry(size, radius, *, round_axis="z", corner_segments=6, cap=True)
CylinderGeometry(radius, height, *, radial_segments=24, closed=True)
ConeGeometry(radius, height, *, radial_segments=24, closed=True)
SphereGeometry(radius, *, width_segments=24, height_segments=16)
DomeGeometry(radius, *, radial_segments=24, height_segments=12, closed=True)
CapsuleGeometry(radius, length, *, radial_segments=24, height_segments=8)
TorusGeometry(radius, tube, *, radial_segments=16, tubular_segments=32)
```

Notes:

- Primitive dimensions and solid radii must be finite and positive.
  `RoundedBoxGeometry.radius` may be zero, while capsule mid-section length may
  be zero but not negative. A torus requires `0 < tube < radius`, excluding
  horn/spindle self-intersection.
- `BoxGeometry` is centered at the origin.
- `RoundedBoxGeometry` is a box with the four edges parallel to `round_axis`
  filleted to `radius` — reach for it instead of a raw `BoxGeometry` whenever a
  real moulded/machined part (handle, bar, casing, rail, panel) would read as
  toy-like with hard cuboid edges. For an all-edges-rounded solid, `smooth_geometry`
  a plain box instead.
- `CylinderGeometry` and `ConeGeometry` are centered and extend along local `Z`.
- `CapsuleGeometry.length` is the cylindrical mid-section length between caps.
- `DomeGeometry` builds the upper half with its base on `z=0`.

### Recessed pockets and insets — `carve_pocket(...)`

```python
carve_pocket(host, *, center, size, corner_radius=0.0, round_axis="z") -> MeshGeometry
```

Carves a rectangular pocket (a recess / inset) into `host` by boolean
subtraction. This is the right tool for a **recessed** feature — a recessed door
handle, an inset panel, a grip channel — which must be modeled as material
REMOVED, never as a protruding box stuck on the surface. `center`/`size` are in
the current mesh coordinates; make the pocket overshoot the surface along its
depth axis so the cut is clean, and give `corner_radius` to round the mouth.
`host` must be a manifold solid. The coordinates are in the current mesh frame,
before any containing Visual or Part transform.

```python
# A recessed vertical grip channel in the right edge of a fridge door:
door = BoxGeometry((0.06, 0.5, 0.9))
door = carve_pocket(door, center=(0.031, 0.20, 0.0), size=(0.03, 0.05, 0.5),
                    corner_radius=0.012)
```

## Smooth Mesh Placement and Field Operations

### `snap_to(...)`

```python
piece = snap_to(
    anchor,
    piece,
    overlap=0.004,
    max_move=0.02,
    axis=None,  # or "x", "y", "z"
)
```

Translates `piece` in place to close a small sampled surface gap and add the
requested overlap. It does not rotate, deform, boolean, create a joint, or add
physics. Already touching/intersecting solids are unchanged; excessive or
poorly aligned constrained movement raises `SnapRefused`. Use it for one freely
placeable boss, foot, or spout root—not for a two-ended handle or a mechanism
whose placement is fixed by an articulation axis.

`overlap` and `max_move` are meters. Inputs must pass error-level mesh health;
open visual meshes are allowed. Invalid numeric controls raise `ValueError`,
while a valid but unsafe requested movement raises `SnapRefused`. Only `piece`
is mutated.

### `weld(...)`

```python
molded = weld(
    body,
    protrusion,
    radius=0.006,
    tolerance=0.0015,
    profile="round",  # "tight", "round", or "soft"
    max_gap=0.0,
    trim=None,
)
```

Creates a geometry-only smooth union of healthy closed mesh solids. It samples
signed-distance fields, blends them, and extracts one new manifold mesh.
`radius` controls transition reach; `tolerance` controls field/output
resolution and therefore runtime and memory. Use `trim` to remove a protrusion
that extends through the wall of a hollow body into its cavity.

All distances are meters. `tolerance=None` selects `radius * 0.25`.
`max_gap` explicitly permits a small generated bridge and must not exceed
`radius`; otherwise disconnected inputs are refused. Inputs and an optional
`trim` must be healthy, watertight, single-positive-solid meshes. `weld(...)`
returns a new mesh and leaves every input unchanged.

### `smooth_difference(...)`

```python
recessed = smooth_difference(
    panel,
    cutter,
    radius=0.004,
    tolerance=0.001,
    profile="tight",
    max_gap=0.0,
)
```

Subtracts one or more healthy closed cutter meshes and generates a smooth
transition at the cut boundary. It is appropriate for molded recesses, control
wells, soft vent openings, and sockets. The same meter units, default tolerance,
health requirements, and `max_gap <= radius` rule apply. It returns a new mesh,
does not mutate the base or cutters, and preserves the base normal-crease
policy.

Both field operations deliberately resample the complete input surfaces and
drop UVs, textures, material partitions, and analytic primitive provenance.
Small features can disappear and exact dimensions can drift by the selected
tolerance. Use an exact CAD kernel for mating features,
bores, wall thickness, and dimension-critical radii. A generated high-density
visual mesh also does not require the same collision approximation; author collision
geometry separately when needed. A request above 2,000,000 field
samples is rejected before allocation with a suggested minimum tolerance. A
generated result above 250,000 vertices or 500,000 triangles is refused before
conversion into Python mesh lists; increase tolerance or simplify the inputs.
Health/manifold failures raise SDK `ValidationError`; invalid numeric controls
or disconnected field-operation inputs raise `ValueError`.

## Loft / Extrude / Sweep Builders

### `LatheGeometry`

```python
LatheGeometry(profile, *, segments=32, closed=True)
LatheGeometry.from_shell_profiles(
    outer_profile,
    inner_profile,
    *,
    segments=32,
    start_cap="flat",
    end_cap="flat",
    lip_samples=6,
)
```

- `profile`: iterable of `(radius, z)` points.
- Radii must be non-negative.
- `from_shell_profiles(...)` is the recommended entry point for thin-walled
  revolved shells.

### `LoftGeometry`

```python
LoftGeometry(profiles, *, cap=True, closed=True)
```

- `profiles`: iterable of 3D point loops.
- All profiles must have the same point count.
- This is a low-level mesh loft helper. Prefer `section_loft(...)` for new
  authored shell work.

### `ExtrudeGeometry`

```python
ExtrudeGeometry(profile, height, *, cap=True, center=True, closed=True)
ExtrudeGeometry.centered(profile, height, *, cap=True, closed=True)
ExtrudeGeometry.from_z0(profile, height, *, cap=True, closed=True)
```

- `profile`: 2D closed profile in local XY.
- `height`: positive extrusion length along Z.

### `ExtrudeWithHolesGeometry`

```python
ExtrudeWithHolesGeometry(
    outer_profile,
    hole_profiles,
    height,
    *,
    cap=True,
    center=True,
    closed=True,
)
```

- `outer_profile`: outer 2D loop.
- `hole_profiles`: zero or more through-cut loops inside the outer profile.
- Input loop winding is normalized. For capped closed output, each hole keeps an inward-facing
  sidewall and remains absent from both end caps; solid volume is the outer extrusion minus all
  hole extrusions.

### `SweepGeometry`

```python
SweepGeometry(
    profile,
    path,
    *,
    cap=False,
    closed=True,
    sections=(),
    section_interpolation="linear",
    section_tension=0.0,
    max_segment_length=None,
)
```

- `profile`: 2D profile.
- `path`: 3D path points.
- This keeps the profile plane parallel while translating it along the path.
  `sections` can vary its profile, scale, in-plane rotation, and offset by
  polyline arc length. For tangent-following tube and rail sweeps, and the
  complete `SweepSection` contract, see `45_wires.md`.

## Profile and Shell Helpers

### `rounded_rect_profile(...)`

```python
rounded_rect_profile(
    width: float,
    height: float,
    radius: float,
    *,
    corner_segments: int = 6,
) -> list[tuple[float, float]]
```

### `superellipse_profile(...)`

```python
superellipse_profile(
    width: float,
    height: float,
    exponent: float = 2.6,
    *,
    segments: int = 48,
) -> list[tuple[float, float]]
```

### `sample_catmull_rom_spline_2d(...)`

```python
sample_catmull_rom_spline_2d(
    points,
    *,
    samples_per_segment: int = 12,
    closed: bool = False,
    alpha: float = 0.5,
) -> list[tuple[float, float]]
```

### `sample_cubic_bezier_spline_2d(...)`

```python
sample_cubic_bezier_spline_2d(
    control_points,
    *,
    samples_per_segment: int = 12,
) -> list[tuple[float, float]]
```

### `superellipse_side_loft(...)`

```python
superellipse_side_loft(
    sections,
    *,
    exponents=2.8,
    segments: int = 56,
    cap: bool = True,
    closed: bool = True,
    min_height: float = 0.0001,
    min_width: float = 0.0001,
) -> MeshGeometry
```

### `split_superellipse_side_loft(...)`

```python
split_superellipse_side_loft(
    sections,
    *,
    split_y: float,
    exponents=2.8,
    segments: int = 56,
    cap: bool = True,
    closed: bool = True,
    min_height: float = 0.0001,
    min_width: float = 0.0001,
) -> tuple[MeshGeometry, MeshGeometry, tuple[float, float, float, float]]
```

### `resample_side_sections(...)`

```python
resample_side_sections(
    sections,
    *,
    samples_per_span: int = 2,
    smooth_passes: int = 0,
    min_height: float = 0.0001,
    min_width: float = 0.0001,
) -> list[tuple[float, float, float, float]]
```

## Panel Openings

### `cut_opening_on_face(...)`

```python
cut_opening_on_face(
    shell_geometry: MeshGeometry,
    *,
    face: str,
    opening_profile,
    depth: float,
    offset=(0.0, 0.0),
    taper: float = 0.0,
) -> MeshGeometry
```

- Builds the internal opening throat on the chosen box-like face of an
  existing mesh shell.
- This helper does not perform boolean subtraction.
- It adds the internal side walls for an opening throat and works best when the
  target face is already open, as in a shell-like model.
- `face`: one of `"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"`, `"-z"`.

## Exporting to `sdk.Mesh`

### `mesh_from_geometry(...)`

```python
mesh_from_geometry(
    geometry: MeshGeometry,
    name: str,
) -> Mesh
```

- Materializes the mesh to an internal OBJ managed by the runtime.
- Returns an `sdk.Mesh` descriptor pointing at the managed asset.
- Use this when the final authored visual should be mesh-backed.

## Advice

### Mutating behavior

- `MeshGeometry` transforms mutate in place.
- Call `copy()` or `clone()` before reusing a base mesh in multiple variants.

### Choosing helper level

- Prefer the focused family pages when the visible part is already a known
  semantic shape such as a knob, bezel, wheel, vent grille, or bracket.
- Prefer `section_loft(...)` over raw `LoftGeometry(...)` for new shell/exterior
  loft authoring.
- Prefer the spline-first wire/tube guidance in `45_wires.md` over manual
  sweeps for rails, loops, and frames. In practice, start with
  `tube_from_spline_points(...)` or `sweep_profile_along_spline(...)` unless
  the geometry is intentionally hard-cornered.

### Exporting mesh-backed visuals

- Use procedural meshes to author visible shape.
- Convert the final mesh to `sdk.Mesh` with `mesh_from_geometry(...)`.
- Use stable logical names such as `"shell"` or `"rear_bracket"` rather than
  paths.

## Examples

```python
shell = ExtrudeGeometry(
    rounded_rect_profile(0.12, 0.08, 0.01),
    0.03,
    cap=True,
    center=True,
)
mesh = mesh_from_geometry(shell, "shell")
```

```python
lip = LatheGeometry.from_shell_profiles(
    [(0.42, -0.30), (0.55, -0.12), (0.62, 0.00)],
    [(0.30, -0.24), (0.40, -0.10), (0.48, 0.00)],
    segments=72,
    end_cap="round",
    lip_samples=10,
)
```

## See Also

- `41_panels_and_grilles.md` for panel-face geometry helpers
- `42_brackets_and_mounts.md` for semantic bracket helpers
- `43_fans_and_rotors.md` for fan and blower geometry helpers
- `44_knobs_and_controls.md` for control-surface geometry helpers
- `45_wires.md` for rails, loops, tubes, and frames
- `46_section_lofts.md` for the recommended loft API
- `47_bezels_and_frames.md` for display, trim, and frame helpers
- `48_wheels_and_tires.md` for wheel and tire geometry helpers
- `49_hinges.md` for exposed hinge geometry helpers
- `50_placement.md` for wrapping and mounting mesh-backed geometry

## Clarifications for agent usage

- Angle arguments are radians and use the right-hand rule unless a parameter
  name explicitly ends in `_deg`.
- `LoftGeometry` validates profiles through their XY projection. End caps only
  behave as expected when the first and last profiles are planar at constant `z`.
- `ExtrudeGeometry(..., center=True)` produces a solid centered on the profile
  plane, spanning `z in [-height/2, +height/2]`. Use the `from_z0(...)` form
  when the intended span is `z in [0, height]`.
- `rounded_rect_profile(...)` and `superellipse_profile(...)` return centered
  counter-clockwise XY loops.
- `cut_opening_on_face(...)` does not subtract material from a closed solid by
  itself. It adds the interior throat wall geometry and works best when the
  target face is already open or when you perform the outer cut separately.
- The side-loft helpers use explicit section tuples in the final part frame.
  For `superellipse_side_loft(...)`, `split_superellipse_side_loft(...)`, and
  `resample_side_sections(...)`, each section is `(y, z_min, z_max, width)`
  with the loft axis along `+Y` and each profile section lying in `XZ`.
