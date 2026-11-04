## Direct SDK/downstream-only segmented flexible members

> **Not a USDCraft authoring pattern.** A USDCraft Author (Generation or Reconstruction) must never
> use `add_segmented_flexible_member(...)`, or hand-author an equivalent rigid-body/joint chain, for
> a hose, rope, cable, cord, wire, chain, strap, belt, flexible suspension, or comparable member.
> USDCraft Authors use one `BodyModel.FLEXIBLE_1D` static-rest Part attached with `FIXED`; genuine
> distributed dynamics are currently unsupported. The API below remains documented solely for
> callers directly using the SDK or a downstream application outside the USDCraft Author tasks.

For such direct SDK/downstream callers, `add_segmented_flexible_member(...)` can construct an open,
single-ended chain-like member whose behavior includes bending, gravity, contact, or payload motion.
This is a discrete articulated-rigid-body approximation, not a deformable-body solver. Every segment
is a `BodyModel.RIGID` part with explicit mass and inertia, one collidable `Box`, `Cylinder`, or
`Sphere` primitive, and one native joint. Choose the lowest segment count that preserves the required
motion; large numbers of tiny colliding bodies increase simulator cost without making the asset more
physically meaningful.

For direct SDK/downstream use, if the member hangs from a stationary beam, ceiling, hoist, or
playground frame, construct the model with `base_mode=BaseMode.FIXED` and keep a continuous rigid
load path from the attachment to the root. USDC authors that fixed-root constraint. Standard URDF
does not encode this reusable-asset root mode, so the target URDF loader must fix the root when it
instantiates the asset.

Do not discretize decorative rings one-for-one into rigid bodies. Choose segment length from the
smallest bending scale the task must resolve, then benchmark the lowest useful count. A practical
starting sweep is often 6–24 segments per open member, but this is neither a hard limit nor a
performance guarantee. In USDC, each joint explicitly disables collision between its adjacent
parent and child; non-adjacent segment collision pairs remain. Runtime cost still grows with rigid
bodies, joints, and candidate collision pairs, so validate segment count in the target simulator
before making stability or performance claims.

```python
STEEL_DENSITY = 7850.0
segment_shape = Cylinder(radius=0.008, length=0.08)
spec = SegmentedFlexibleMemberSpec(
    link_count=12,
    segment_geometry=segment_shape,
    segment_mass=estimate_mass_from_density(
        segment_shape,
        STEEL_DENSITY,
        fill_fraction=0.2,  # open-link proxy inside the cylinder envelope
    ),
    rest_axis=(0.0, 0.0, -1.0),
    joint=SpatialSegmentJoint(cone_deg=55.0),
)
member = add_segmented_flexible_member(
    model,
    "hoist_chain",
    support,
    spec=spec,
    attachment_origin=Origin(xyz=chain_mount),
    material="steel",
    physics_material="chain_contact",
)
```

The optional visual and physics material names in this example must be registered on `model` before
the helper call.

`SpatialSegmentJoint` creates spherical segment joints. They are native in USDC; standard URDF has
no spherical joint, so the URDF export is a floating-joint fallback with `usdcraft` metadata and is
not dynamically equivalent. Portable non-zero damping is not available for this spatial mode.
Use `PlanarSegmentJoint(axis=..., lower=..., upper=..., damping=..., friction=...)` when one bending
plane is an honest model: its revolute joints and limits are represented in both URDF and USDC.
URDF writes joint dynamics directly. Shared USD authoring represents passive damping with a
zero-stiffness DriveAPI and converts the SDK per-radian coefficient to USD's per-degree
representation. The flattened PhysX entry retains that DriveAPI and authors
the helper's deliberate legacy `PhysxJointFrictionCoefficient(friction)` as
`physxJoint:jointFriction`; the Newton overlay authors passive damping as native `mjc:damping`
without an actuator, uses the helper's legacy `MotionProperties.friction` fallback for
`mjc:frictionloss`, and removes the applied DriveAPI token while leaving its properties as inactive
metadata. Isaac Sim 6 has no MJC mapping for the authored velocity limit,
so Newton export preserves that SDK value in `usdcraft` custom data and records that fact as an
informational note in the compile report.
Spatial spherical joints have only static Newton validation. Any runtime-support claim requires
separate downstream evaluation; it is not part of USDCraft acceptance.

The returned member exposes `head`, `parts`, `joints`, `tail`, and
`tail_attachment_origin`. Attach a separately authored rigid payload to `member.tail` with a real
articulation at `member.tail_attachment_origin`. The payload geometry must meet that endpoint in
the rest pose; the decorative visual is not the attachment contract.

Use `spec.total_length` for the attachment-to-tail rest length. It is
`link_count * segment_pitch`, not `link_count * segment_geometry.length`, because the safe pitch
includes joint clearance. When fitting a requested endpoint distance, adjust link count and body
length against this resolved value instead of scaling the finished chain afterward.

The current articulation model is a tree: each part can have only one parent. It therefore cannot
represent a closed loop or a payload constrained by multiple flexible members, such as two chains
attached from one beam to the same swing seat or a closed belt. Do not approximate such topology by
adding a second visual chain and claiming that both chains are simulated. Use a clearly declared
reduced-order carrier when that abstraction is sufficient, or report the requested mechanics as
unsupported. See `30_articulated_object.md` for the topology boundary.

