# Decals and Labels: Printed Surface Graphics

## Purpose

Use this page for any 2D graphic PRINTED on a surface: product labels, logos,
warning text, dial faces, stickers, patterns. These are NOT geometry — never
model them as extruded blocks or per-glyph shapes. Author the graphic as a
TextureCanvas and attach it as a decal.

**Scope — keep it coarse.** A decal is a minor finishing touch; the 3D shape and
proportions matter far more. Represent a printed graphic with a few flat COLOR
BLOCKS in roughly the right colors and positions. Do NOT reproduce fine text,
logos, pictograms, numbered instructions, or gauge readouts — correct colors are
enough. Reach for the text/line/polygon helpers below only when a label's
identity genuinely depends on one short word, not to recreate the artwork.

## Import

```python
from sdk import TextureCanvas
```

`Part.decal(...)` is a method on the part; no extra import is needed.

## TextureCanvas — draw the graphic in code

A TextureCanvas is the source of truth for a label: it is defined by code (like
geometry) and re-rendered to a texture at compile time. All coordinates and
sizes are NORMALIZED (0..1); colors are display-referred sRGB rgba in 0..1.

```python
label = TextureCanvas(width=512, height=640, background=(0.85, 0.10, 0.10, 1.0))
label.rect((0.08, 0.10, 0.84, 0.22), fill=(0.0, 0.0, 0.55, 1.0))      # xywh, normalized
label.text((0.5, 0.21), "FIRE", size=0.10, fill=(1, 1, 1, 1), anchor="mm", bold=True)
label.ellipse((0.40, 0.50, 0.20, 0.20), outline=(0, 0, 0, 1), line_width=0.01)
label.line([(0.1, 0.9), (0.9, 0.9)], fill=(0, 0, 0, 1), line_width=0.01)
label.polygon([(0.5, 0.6), (0.6, 0.8), (0.4, 0.8)], fill=(1, 1, 0, 1))
```

Drawing methods (all normalized, all chainable). Stroke width is `line_width`
(also accepted as `width`), in normalized units:
- `rect(xywh, *, fill=, outline=, line_width=)`
- `ellipse(xywh, *, fill=, outline=, line_width=)`
- `line(points, *, fill=, line_width=)` — `points` is a LIST of at least two
  `[x, y]` points, e.g. `line([(0.1, 0.9), (0.9, 0.9)], ...)`. It is NOT four
  scalars and NOT a single point; pass `[(x0, y0), (x1, y1)]`, adding more points
  for a polyline.
- `polygon(points, *, fill=, outline=, line_width=)` — `points` is likewise a
  list of at least three `[x, y]` vertices.
- `text(xy, text, *, size=, fill=, anchor="mm", bold=False)` — `anchor` is a
  two-letter PIL anchor (`mm` = centered).
- `paste(other_canvas, xywh)` — composite a sub-canvas; use a loop for repeated
  motifs (pictogram grids, tick marks).

Use a transparent background `(0, 0, 0, 0)` when the label should show only its
graphics over the part's own color; use an opaque background for a full label
panel.

The exporter inspects the rendered pixels. A canvas whose alpha is 1 everywhere
uses the opaque material path; a canvas containing transparent pixels uses the
cutout path. Do not use a nominally transparent canvas when the finished label
is actually a solid panel.

### Text coverage

The bundled font covers Latin/ASCII. Use color blocks for unreadable or unsupported
lettering; do not invent substitute product text. Use a short legible identity word
only when the task needs it.

## Part.decal — attach the graphic to a surface

```python
body.decal(
    label,
    width=0.18,            # physical size on the surface (meters)
    height=0.22,
    curvature_radius=0.05, # >0 wraps the decal onto a cylinder of this radius
    origin=Origin(xyz=(0.0, 0.0, 0.12)),  # place/orient like any visual
)
```

- A flat decal (`curvature_radius=0`) lies in its local XY plane, facing +Z.
- A curved decal wraps its width around a cylinder of `curvature_radius` about
  the local **Z** axis (matching the SDK's Z-axis `Cylinder`); its `height` runs
  along Z. Set `curvature_radius` to the host body's radius and place the decal
  on the body axis so it hugs the surface. An upright (Z-axis) cylinder body
  needs **no tilt** — just rotate about Z (`rpy=(0, 0, yaw)`) to aim the label at
  the side you want (the front face defaults to -Y).
- The decal is pushed slightly proud of the surface automatically (no
  z-fighting) and is VISUAL-ONLY — it never becomes a collision shape.

### Sizing a curved decal

For a decal wrapped on a cylinder of radius `R`, the `width` is an ARC LENGTH on
the surface, so the angle it subtends is `width / R` radians. Keep it under a
sensible fraction of the circumference:

- Set `curvature_radius = R` to the host body's radius.
- Choose `width ≤ 2 * π * R * (coverage_fraction)` — e.g. a band covering at most
  a third of the way around uses `width ≈ 2 * π * R / 3`. A full wrap is
  `width ≈ 2 * π * R`.
- Place `origin` on the body axis at the right height so the decal hugs the
  surface; do not offset it outward yourself (the proud offset is automatic).

```python
R = 0.06                      # host cylinder radius
band = TextureCanvas(width=900, height=180, background=(0, 0, 0, 0))
# ... draw the stripe ...
body.decal(band, width=2 * math.pi * R / 2, height=0.05,  # half-wrap
           curvature_radius=R, origin=Origin(xyz=(0.0, 0.0, 0.07)))
```

### Aiming a curved decal (front vs side) — do NOT over-rotate

A curved decal in its own frame faces -Y at zero yaw. Apply `Origin` relative
to the owning Part, using the object frame declared by the task:

- Zero yaw faces local -Y.
- Yaw `math.pi` faces +Y; `+math.pi/2` faces +X; `-math.pi/2` faces -X.
- Align the decal cylinder axis with the host cylinder axis. An upright host
  needs only yaw; a tilted host needs the corresponding rotation.

## Common failure modes

- **Black rectangle instead of just the graphic.** The canvas background was
  opaque. Use `background=(0, 0, 0, 0)` (transparent) so only your drawn shapes
  show over the part's own color; reserve an opaque background for a full label
  panel that really should cover the surface.
- **Decal wraps past itself / distorts.** `width` exceeded the available arc
  length (`width > 2 * π * curvature_radius`). Reduce `width` or increase
  `curvature_radius`. Make the canvas aspect ratio match the physical
  `width:height` so the artwork is not stretched.
- **Decal floats off or sinks into the surface.** `curvature_radius` did not
  match the host radius, or `origin` was offset off the body axis. Set
  `curvature_radius` to the body radius and keep `origin` on the axis.
- **Graphic looks faint or wrong-colored.** Drawing fully outside the canvas's
  0..1 coordinate range, or a near-zero `line_width`. Keep coords in 0..1 and use
  `line_width` in normalized units (e.g. `0.01`).
- **Tiny text or edges shimmer at a distance.** Dense, high-contrast detail can
  collapse to only a few screen pixels and become temporally unstable under
  minification, anti-aliasing, or video compression. Prefer broad color blocks
  and one short identity word; do not increase the proud offset to treat this as
  z-fighting when the decal is already separated from the host surface.

## Repeated motifs

For grids of icons or tick marks, draw one motif as a sub-canvas and `paste` it
in a loop:

```python
icon = TextureCanvas(64, 64, background=(0, 0, 0, 0))
icon.rect((0.15, 0.15, 0.7, 0.7), outline=(0, 0, 0, 1), line_width=0.1)
for row in range(3):
    for col in range(2):
        label.paste(icon, (0.18 + col * 0.32, 0.42 + row * 0.16, 0.26, 0.13))
```

## When NOT to use a decal

Only printed/painted/applied graphics are decals. Features that are genuinely
raised or recessed in 3D — embossed buttons, molded ribs, physical grilles,
engraved relief — are geometry; model them with meshes as usual.
