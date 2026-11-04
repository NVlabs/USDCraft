# Shelving and Panel Furniture

## Purpose

Use this page for retail/storage shelving and board-built (panel) furniture:
supermarket gondolas, storage racks, bookcases, cabinets, nightstands,
sideboards, desks, and tables. These helpers produce the rigid structure;
doors, drawers, and other moving parts are authored as separate parts and
hinged onto it.

## Import

```python
from sdk import (
    ShelvingUnitGeometry,
    CabinetCarcassGeometry,
    PanelGeometry,
    TaperedLegGeometry,
    BarHandleGeometry,
    BridgeHandleGeometry,
)
```

All Z-up, floor at z=0 (PanelGeometry and the handle helpers are exceptions
noted below), dimensions in meters.

## `ShelvingUnitGeometry` — racks and gondolas

```python
ShelvingUnitGeometry(
    width: float = 0.9,
    depth: float = 0.4,
    height: float = 1.8,
    *,
    shelf_count: int = 4,                 # evenly spaced, or:
    shelf_heights: list[float] | None = None,  # explicit board underside heights
    shelf_thickness: float = 0.028,
    post_size: float = 0.045,
    back_panel: bool = False,             # True for supermarket gondolas
    kick_height: float = 0.0,             # base plinth
    shelf_lip: float = 0.0,               # front retaining lip (rises above the board)
    top_shelf: bool = True,
)
```

The instance exposes `shelf_levels` (board underside Z values) — place
products at `level + shelf_thickness`:

```python
unit = ShelvingUnitGeometry(back_panel=True, kick_height=0.1, shelf_lip=0.03)
surface_z = unit.shelf_levels[1] + 0.028  # second shelf surface
```

For simulator-ready stock, use the shelving geometry as both visual and
collision geometry, and place each product's lowest collision point exactly at
`surface_z`. Each independently graspable product is a separate rigid `Part`
with an `ArticulationType.FLOATING` placement edge, explicit mass, collision,
rigid-body properties, and a frictional physics material. The FLOATING edge
places the item in the SDK tree but exports no USD joint; gravity, contact, and
friction support it. Rows and stacks are layouts, not rigid-body grouping.
Never merge a removable row or attach ordinary stock with `FIXED`.

Estimate each product's mass using a representative primitive and material density.
Keep its explicit `Inertial` and contact material; see `20_core_types.md`.

Recipes: supermarket gondola `back_panel=True, kick_height=0.1,
shelf_lip=0.03`; warehouse rack `post_size=0.06, shelf_thickness=0.04`;
bookcase `width=0.8, depth=0.3, back_panel=True`.

## `CabinetCarcassGeometry` — cabinet bodies

```python
CabinetCarcassGeometry(
    width: float = 0.6,
    depth: float = 0.45,
    height: float = 0.75,
    *,
    panel_thickness: float = 0.018,
    back_panel: bool = True,
    shelf_count: int = 0,        # fixed interior shelves
    kick_height: float = 0.0,
)
```

Open front faces **-Y**. Exposes `shelf_levels`, `interior_width`, and
`interior_depth` for sizing doors/drawer boxes. A hinged door is a separate
part: a `PanelGeometry` sized to the front opening, articulated with a
revolute joint along its edge; add a `BarHandleGeometry` visual to it.

## `PanelGeometry` — boards

```python
PanelGeometry(width, height, thickness, *, edge_radius=0.002)
```

A board in the XY plane, thickness along Z, centered at the origin — rotate
and translate into place. Use for doors, table tops, drawer fronts, and any
custom panel work. Choose `edge_radius` from the required edge shape and panel thickness;
the 2 mm default is a starting value.

## `TaperedLegGeometry` — round legs

```python
TaperedLegGeometry(
    height=0.4, top_radius=0.022, bottom_radius=0.012,
    *, curve=0.0, foot_height=0.0,
)
```

Floor at z=0. `curve > 0` bows the silhouette outward (mid-century style),
`curve < 0` hollows it. For splayed legs, rotate a few degrees before
translating under the corners of a table top.

## `BarHandleGeometry` — pulls

```python
BarHandleGeometry(length=0.16, *, radius=0.005, standoff=0.03, style="bar")
```

`style="bar"` is a modern rail handle on two posts; `style="arch"` is a
smooth bow. Grip runs along X, posts land on the mounting face at z=0 —
rotate so -Z points into the door/drawer front.

For a broad, architectural arch such as a lid or appliance pull, prefer the
constrained bridge family instead of drawing an arbitrary spline:

```python
BridgeHandleGeometry(
    mount_span=0.24,
    rise=0.10,
    section_width=0.025,
    section_depth=0.035,
    shoulder_radius=0.045,
    crown_height=0.012,
)
```

It guarantees a bilateral X symmetry plane, two contacts on z=0, circular
shoulders, and either a flat top (`crown_height=0`) or a symmetric shallow crown.
Place the part by its symmetry center and mounting plane; do not independently
move the two feet.

## Realism checklist

- Wood: `roughness=0.45-0.6, clearcoat=0.15` for lacquered, plain
  `roughness=0.7` for raw; tint sides/back slightly darker than the front.
- Metal shelving: `metallic=0.85, roughness=0.4`; powder-coated retail
  shelving is *not* metallic: `metallic=0.0, roughness=0.5`.
- Assign distinct materials where the required appearance differs between
  carcass, panels and handles.
- Stock shelves with `BottleGeometry`/`JarGeometry`/`BoxGeometry` products
  placed at `shelf_levels` for supermarket scenes; keep each removable product
  as an independent contact-supported rigid body.
