# Common compound geometry

Use these helpers when the reference contains a recognizable compound structure. They keep
the shape parameterized and named, while the surrounding `Part` and articulation hierarchy
remain under your control.

```python
from sdk import (
    AFrameGeometry,
    ArticulationType,
    BeltSeatGeometry,
    BodyModel,
    CasterForkGeometry,
    CasterGeometry,
    FlexibleHoseGeometry,
    GableRoofGeometry,
    HangingChainGeometry,
    LadderGeometry,
    MoldedSeatGeometry,
    Origin,
    PostBeamBracketGeometry,
    SlideChuteGeometry,
    mesh_from_geometry,
)
```

Fit these families to the required shape and attachment points. Do not
non-uniformly scale finished triangles to an AABB; that deforms rails, links, and cross-sections.

## Anchor-fitted frames

`LadderGeometry(width=..., rise=..., run=..., rail_section=..., rung_count=...)` starts at local
`(0, 0, 0)` and ends at `(0, run, rise)`. `AFrameGeometry(height=..., footprint_width=...,
leg_section=..., spreader_height=...)` places its feet on Z=0 and apex at `(0, 0, height)`.
Place these local frames at the required base/feet anchors; keep the supported beam separate.

## Flexible hoses

`FlexibleHoseGeometry(control_points, radius=...)` produces a smooth **static** tube through the
observed route. For USDCraft Authors, use this static-rest representation for extinguisher
hoses, appliance cables, and similar flexible members. It does not provide bending, draping, load
transfer, or dynamic contact in a simulator; current USDCraft does not approximate those
behaviors with a rigid-body joint chain.

Author the hose as its own `BodyModel.FLEXIBLE_1D` part and attach it to the rigid assembly with a
`FIXED` articulation:

```python
hose = model.part("hose", body_model=BodyModel.FLEXIBLE_1D)
hose.visual(
    mesh_from_geometry(
        FlexibleHoseGeometry(control_points=hose_route_local, radius=hose_radius),
        "extinguisher_hose",
    ),
    name="hose_tube",
)
model.articulation(
    "body_to_hose",
    ArticulationType.FIXED,
    parent=body,
    child=hose,
    origin=Origin(xyz=hose_mount),
)
```

Do not set `inertial` or `rigid_body` on the hose. `BodyModel.FLEXIBLE_1D` and the geometry helper
describe one reference pose; USDCraft does not simulate deformation. URDF/USDC preserve the
hose's derived collision geometry as a `deformable_contact` rest-shape inspection surface without
claiming a live physics collider. This remains the required USDCraft representation even when
ordinary use would flex, carry, route, or pull the member. If genuine distributed behavior is an
explicit requirement, report the current USDCraft limitation instead of building a rigid-segment
joint chain with either the helper or hand-authored code. Put rigid nozzles, ferrules,
couplings, valves, and end-handles in separate `BodyModel.RIGID` parts; attach them with `FIXED`
when their modeled relationship is fixed. One genuinely meaningful low-order joint may instead
connect rigid components when required for normal use, but it must never segment the hose itself.

## Casters

`CasterGeometry(wheel_radius=..., wheel_width=...)` includes a wheel, axle, fork, bridge, and
stem. It is a compact visual family, not a multi-joint mechanism. If swivel and wheel rotation
are primary requested motions, split those members into separate parts and add the real joints.

Use `CasterForkGeometry(wheel_radius=..., wheel_width=..., trail_offset=...)` when metal and tire
need different materials or the wheel is a separately articulated part. Its wheel center is local
`(0, 0, wheel_radius)`, axle is X-aligned, and the stem trails along +Y. Do not use one vertical box
as a fork: two cheeks, a swept yoke, axle, race, and stem are important silhouette evidence.
For anchor fitting record `wheel_center`, `wheel_bottom`, `axle_left`, `axle_right`, and
`stem_mount`.

## Molded seats

`MoldedSeatGeometry(width=..., depth=..., back_height=..., thickness=..., seat_sag=...,
back_recline=...)` creates a continuous shallow pan/back shell with rounded side rims. Local X is
width, +Y is toward the back, and the seat surface is near Z=0. Use it for cart child seats,
high-chair inserts, stadium shells, and compact bucket seats. Intersecting flat seat/back slabs are
not an equivalent substitute when the image shows a molded transition.
For anchor fitting record `seat_front`, `seat_back`, and `back_top` attachment points.

## Swing seats and static chain visuals

`BeltSeatGeometry(width=..., depth=..., thickness=..., sag=...)` preserves the shallow symmetric
sag of a flexible belt seat. `HangingChainGeometry(length=..., link_count=..., link_radius=...,
wire_radius=...)` creates alternating visible links in one authored pose. The latter is presentation
geometry only: it has no per-link rigid bodies, inertias, joints, or runtime chain dynamics. As a
USDCraft Author, place each flexible chain run in one `BodyModel.FLEXIBLE_1D` Part attached with
`FIXED`; do not turn its visible links into rigid bodies and joints. If independently moving or
load-bearing chain dynamics are required, report that current USDCraft limitation explicitly.

## Slide chutes

`SlideChuteGeometry(width=..., length=..., drop=..., thickness=..., rail_radius=...)` creates a
smooth descending bed with continuous side rails. Its local top begins at `(0, 0, 0)` and the
chute extends along positive Y while descending in Z. Place that local frame at the real platform
edge; do not replace the negative space below a slide with a solid wedge.

## Structural brackets and gable roofs

`PostBeamBracketGeometry(member_width=..., member_height=..., plate_length=...,
plate_thickness=..., bolt_radius=...)` is an open saddle with two side plates, a lower plate, and
transverse bolts. The supported member runs along local X. Use it at exposed playground,
pergola, rack, and furniture-frame joints; keep the timber or beam separate.
For anchor fitting record `joint_center`, `beam_entry`, and `support_entry`.

`GableRoofGeometry(width=..., depth=..., rise=..., panel_thickness=..., overhang=...)` creates two
pitched panels, a ridge cap, and eave fascia. Local Z=0 is the eave plane and roof depth is Y. It
is preferable to two intersecting boxes when a playhouse, shed, kiosk, or pet house needs a
readable ridge and finished eaves.
For anchor fitting record `ridge_start`, `ridge_end`, `left_eave`, and `right_eave`.

## Rounded doors and integrated pulls

Compose a rounded door from `RoundedBoxGeometry((width, depth, height), corner_radius,
round_axis="y")`, a separate liner or gasket, and `BarHandleGeometry`/`BridgeHandleGeometry`.
Rotate the handle so its mounting normal matches the door face. Use `carve_pocket(...)` only when
the image clearly shows a recessed pull; a dark rectangular decal is not a supported handle.
For a `rounded_panel` family record `hinge_bottom`, `hinge_top`, and `free_edge_mid`.

The geometry-family helpers return `MeshGeometry`. Convert with `mesh_from_geometry(...)`
and assign semantic visual names. Apply each helper in the owning Part frame.
