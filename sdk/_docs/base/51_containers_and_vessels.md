# Containers and Vessels: Bottles, Jars, Caps

## Purpose

Use this page for kitchen and supermarket vessels: drink bottles, condiment
and storage jars, cosmetic containers, and their lids. All three helpers are
lathe-based solids with smooth, realistic profiles — no manual profile
authoring needed for the common cases.

## Import

```python
from sdk import BottleGeometry, JarGeometry, CapGeometry
```

All are Z-up with the base at z=0, centered on the Z axis, dimensions in
meters. They are valid manifold solids, so booleans and `smooth_geometry`
work on them directly.

## `BottleGeometry`

```python
BottleGeometry(
    height: float = 0.24,
    body_radius: float = 0.035,
    *,
    neck_radius: float = 0.013,     # must be < body_radius
    neck_height: float = 0.025,
    shoulder_height: float = 0.05,  # smooth blend between body and neck
    base_fillet: float = 0.004,
    lip_bead: float = 0.0015,       # rolled bead at the mouth; 0 disables
    waist: float = 0.0,             # >0 pinches the body mid-height inward
    segments: int = 48,
)
```

Recipes: water bottle `waist=0.004`; wine bottle `height=0.30,
body_radius=0.038, neck_radius=0.012, shoulder_height=0.09`; spray/detergent
bottle: increase `neck_radius` and add a trigger as separate geometry.

## `JarGeometry`

```python
JarGeometry(
    height: float = 0.12,
    radius: float = 0.04,
    *,
    mouth_radius: float = 0.032,    # must be < radius
    shoulder_height: float = 0.018,
    base_fillet: float = 0.003,
    mouth_ring_height: float = 0.01,
    segments: int = 48,
)
```

Wide-mouth vessel for jams, sauces, spice jars, canisters. A `CapGeometry`
with `radius` slightly above `mouth_radius` seats on top.

## `CapGeometry`

```python
CapGeometry(
    radius: float = 0.015,
    height: float = 0.014,
    *,
    top_fillet: float = 0.003,   # rounded top edge
    skirt_taper: float = 0.0,    # >0 narrows the open rim slightly
    segments: int = 48,
)
```

Translate onto the bottle/jar mouth: `cap.translate(0, 0, bottle_height - overlap)`.

## Materials

Vessels live or die by their materials — always set explicit PBR:

- Glass bottle/jar: `metallic=0.0, roughness=0.05, transmission=0.9, ior=1.45`
  with a tinted rgba (keep alpha 1.0).
- PET/HDPE plastic: `roughness=0.3` opaque, or `transmission=0.55` for
  translucent milk-jug plastic.
- Metal cap: `metallic=0.9, roughness=0.35`.

## Composition notes

- For a labeled product, add a thin `CylinderGeometry` band slightly larger
  than the body radius as a separate visual with its own material.
- Authored solid (not hollow): correct for closed products on shelves. For an
  open vessel that must read hollow at the mouth, subtract an inner lathe via
  `boolean_difference`, or use `LatheGeometry.from_shell_profiles` directly.
- Filling shelves: combine with `ShelvingUnitGeometry.shelf_levels` to place
  rows of bottles at exact shelf surface heights. In a simulator-ready stocked
  shelf, every bottle or jar that can be picked up independently must be its own
  rigid `Part` with its own collision, mass, and physics material. Use a
  `FLOATING` placement edge for each item; do not merge a row into one body or
  weld ordinary stock to the shelf.
