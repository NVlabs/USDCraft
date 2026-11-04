# ArticulatedObject

Use `ArticulatedObject` to assemble Parts, visuals, materials and physical relationships.
Import public names from `sdk`. Type details and joint-physics calculations are in
`20_core_types.md`; diagnostics and exact relationship checks are in `80_testing.md`.

## Assembly and physical ownership

Choose the relationship from the target's normal use:

| Relationship | Representation |
| --- | --- |
| Rigidly attached detail with no independent motion | Named visuals in the same rigid Part |
| Distinct rigid bodies that are permanently attached | Separate Parts joined by `FIXED` when that body boundary is useful |
| Hinge, slide, axle or supported constrained motion | Separate Parts with the corresponding articulation |
| Freely separable/contact-supported object | Its own rigid Part and a `FLOATING` placement edge |
| Flexible linear member | One static-rest `FLEXIBLE_1D` Part, attached with `FIXED` |

Material changes, decorative seams and separate mesh constructors do not require separate bodies.
Keep required independent motion; do not merge a movable lid or drawer into its parent to avoid
checks. Conversely, do not add FIXED bodies solely to represent every fastener, material or surface.
One rigid Part may have several named visuals and materials.

Each rigid Part needs explicit mass, inertia proxy and contact-material binding. A Part containing
several fixed visual details uses one coherent mass estimate. Non-rigid Parts omit inertial and
rigid-body properties and keep rigid fittings in separate rigid Parts.

Use one root in the authoring graph. Give a standalone asset explicit `BaseMode.FLOATING`.
`model.meta["mode"]="static"` means no constrained motion axes, not a fixed or kinematic root;
use `"articulated"` for an object with constrained moving parts. FLOATING placement edges remain
compatible with static mode.

## Construction

```text
ArticulatedObject(name, parts=[], articulations=[], materials=[], physics_materials=[],
                  base_mode=BaseMode.FLOATING, meta={}, assets=None)
model.part(name, *, visuals=None, inertial=None, rigid_body=None,
           body_model=BodyModel.RIGID, meta=None) -> Part
part.visual(geometry, *, origin=None, material=None, color=None, name=None,
            collidable=True) -> Visual
part.get_visual(name) -> Visual
```

The model owns its registered Parts and materials. Runtime-managed assets are available to newly
created Parts. Use `mesh_from_geometry`, `mesh_from_input` or `mesh_from_cadquery` for permitted
mesh assets; a `MeshGeometry` constructor result is not itself a `Part.visual` descriptor.
Name visuals that need a distinct material, local measurement or exact test.

```text
model.articulation(name, articulation_type, parent, child, *, origin=None, axis=None,
                   cone_deg=None, motion_limits=None, motion_properties=None,
                   newton_joint_solver_tuning=None, passive_spring=None,
                   physx_joint_resistance=None, physx_armature=None, drive=None,
                   mimic=None, meta=None) -> Articulation
```

`parent` and `child` accept registered Part objects or their names. `origin=None` means `Origin()`;
`axis=None` resolves to local +Z for joints that use an axis. Give moving joints an explicit unit
axis so direction is clear. The method appends the articulation and returns it.

### Joint selection

| Type | Required motion declaration | Meaning |
| --- | --- | --- |
| `REVOLUTE` | Unit axis; positive effort/velocity; finite lower/upper containing 0 | Bounded hinge |
| `CONTINUOUS` | Unit axis; positive effort/velocity; no lower/upper | Unlimited rotation |
| `PRISMATIC` | Unit axis; positive effort/velocity; finite lower/upper containing 0 | Bounded translation |
| `SPHERICAL` | Nonzero rest axis and `cone_deg` in (0, 180] | Three rotational degrees; swing cone, free twist |
| `FIXED` | No scalar limits | Permanent rigid attachment or non-rigid rest-pose attachment |
| `FLOATING` | No axis, scalar limits or motion properties | Unconstrained rigid-body placement |

Spherical joints do not take scalar damping/friction, drives or mimic. FIXED/FLOATING edges do
not take scalar motion physics. Any edge touching a non-rigid Part must be FIXED. D6, multi-parent
constraints and closed-loop mechanisms are unsupported; use a clearly stated meaningful reduced
model only when it preserves the requested behavior.

Every scalar joint needs explicit `MotionProperties(damping=..., friction=...)`, separate
`physx_joint_resistance`, and positive effort/velocity limits. The friction value remains required
when PhysX resistance is declared. Use `PhysxJointFriction` for absolute effort or
`PhysxJointFrictionCoefficient` for an intentional load-proportional bearing.
Use `meta["gravity_hold"]` to state pose-hold intent and
follow the load/tuning calculation in core types.

A passive spring and active drive are mutually exclusive. With a drive, set
`MotionProperties.damping=0.0` and put active damping in `JointDrive.damping`.
`JointDrive.max_force` is finite, non-negative and <= `MotionLimits.effort`; `None` uses that
limit. Omit a drive for a hand-operated passive mechanism. `physx_armature` is optional PhysX
joint-space inertia, not a generic friction or pose-hold parameter.

## Frames and motion direction

1. An articulation `origin` maps its joint frame into the parent Part frame.
2. At q=0, the child Part frame coincides with that joint frame.
3. The child's visual origins are local to its own frame; avoid applying the parent-to-joint
   translation a second time to child geometry.
4. `axis` is expressed in the joint frame. `origin.rpy` rotates it relative to the parent.
5. Positive angular q follows the right-hand rule around +axis; positive prismatic q translates
   along +axis. SDK angular values are radians, linear values metres.

Choose axis sign from the required positive motion. If a lid extends along local +X from its
hinge, axis `(0, -1, 0)` lifts its free edge toward +Z. A mirrored lid extending along local -X
uses `(0, 1, 0)` for the same upward opening. A drawer extending along world +X uses local
`(1, 0, 0)` only when its joint frame is aligned with the world. Set coherent limits and verify
the intended pose; swapping lower/upper is not a repair for a wrong axis.

This fragment assumes existing `cabinet` and `drawer` Parts with authored visuals and physics,
an aligned horizontal joint frame, and geometry that retains insertion over 0.28 m of travel:

```python
from sdk import ArticulationType, MotionLimits, MotionProperties, Origin, PhysxJointFriction
slide = model.articulation(
    "cabinet_to_drawer", ArticulationType.PRISMATIC, cabinet, drawer,
    origin=Origin(xyz=(0.0, 0.0, 0.12)), axis=(1.0, 0.0, 0.0),
    motion_limits=MotionLimits(effort=40.0, velocity=0.25, lower=0.0, upper=0.28),
    motion_properties=MotionProperties(damping=5.0, friction=4.5),  # Newton/URDF slide drag, N
    physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(
        6.0, dynamic_ratio=0.75,
    ),
    meta={"gravity_hold": False},  # horizontal slide; gravity acts perpendicular to the axis
)
```

The effort and travel here are example design inputs. Derive them from the actual mechanism and
available evidence; do not copy the values into a different object without checking the fit.

### Mounted geometry

Place a mechanical joint where real parent and child geometry meet. A hinge pin/barrel, rigid
neck or bracket can supply the mount when the major surfaces are separated. Add a bridge only
when such a structural member is justified; do not invent a rod to satisfy a numeric diagnostic.

Exact contact is preferable to artificial penetration. If a simplified mount intentionally
interpenetrates, declare only that named interface with `allow_overlap(...)` and test its decisive
clearance/contact. A joint-origin distance heuristic uses an absolute tolerance and is not a
universal mount-distance rule; inspect the actual geometry before moving a valid joint.

### Retained insertion

A slider's hidden member must remain engaged at maximum travel:

`member length >= exposed length at full extension + retained insertion`.

Place the joint at a meaningful sleeve entry or seating plane, extend child geometry into the
hidden direction when needed, and set travel from the remaining engagement. Check transverse fit
and longitudinal retained overlap at rest and extension. A projected overlap is not proof of a
clear cavity or absence of collisions; use the exact named interface and its intended free space.

### Contact-supported and separable objects

A product on a shelf or one chair beside a table remains independently manipulable. Use one
rigid Part per separable object and FLOATING edges to maintain the single-root authoring tree.
Those edges emit no native USD joint; URDF uses floating joints to preserve the hierarchy.
Do not weld separable items to their supports with FIXED.

At q=0, put each contact surface against its actual support without penetration. A floor-less
reusable arrangement can reference the external ground plane; justify only genuinely independent
members with `allow_isolated_part`. That allowance does not supply a missing shelf or suspend an
unsupported prop.

### Flexible members and reduced mechanisms

Represent a hose, cable, strap or chain as one static-rest flexible Part, with separate rigid
fittings where necessary. It does not simulate distributed tension, gravity or contact response.
One parent articulation per Part excludes loops or a payload constrained by two branches.
A second visual tether supplies appearance, not a second physical constraint.

For a reduced swing mechanism, a meaningful carrier joint may move the rigid seat relative to
its support; displayed flexible tethers retain their declared static-rest status. Do not segment
them into a rigid-body joint stack or claim independent chain dynamics.

## Materials and lookup

`model.material(name, **appearance_fields)` accepts all `Material` appearance fields listed in
core types, including PBR and texture inputs, and registers the result. Use `rgba` or `color`,
not both. `model.physics_material(name, static_friction=..., dynamic_friction=...,
restitution=..., density=...)` registers contact material independently.

```text
model.get_part(name_or_part) -> Part
model.get_articulation(name_or_joint) -> Articulation
model.root_parts() -> list[Part]
model.validate(*, strict=True, strict_mesh_paths=False) -> None
```

Lookup raises `ValidationError` for an unknown name. `root_parts()` returns Parts that are not
children of any articulation. Validation checks names, references, geometry/material legality,
parent uniqueness and rooted graph structure. `strict_mesh_paths=True` also checks mesh files.
Model validity does not establish target fidelity, usable motion or runtime stability.

## Coupled and non-scalar poses

`mimic=Mimic(source_name, multiplier=..., offset=...)` gives a scalar follower
`q_follower = multiplier * q_source + offset`. Use compatible motion domains with no mimic cycle;
pose the source joint. Newton declares native mimic; PhysX retains metadata without verified
native enforcement. A mimic relation is not a nonlinear or closed-loop constraint.

Spherical USD joints retain three rotational degrees with free twist. Their URDF representation
is a semantic fallback. SDK tests use a deterministic one-dimensional swing slice; it does not
exercise every spherical orientation. FLOATING test poses use `Origin(xyz=..., rpy=...)` in the
placement frame rather than a scalar. The testing reference defines both pose interfaces.
