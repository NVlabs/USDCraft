# Core Types

Import the types below from `sdk`. Construction and motion frames are defined in
`30_articulated_object.md`; exact geometry assertions are in `80_testing.md`.

## Units and numeric values

Distances are metres, mass is kilograms, and angular positions/velocities are radians and rad/s.
Angular spring/drive gains use N·m/rad and N·m·s/rad; linear gains use N/m and N·s/m.
`Origin.rpy=(roll, pitch, yaw)` composes `Rz(yaw) @ Ry(pitch) @ Rx(roll)`. Cylinders point along
local +Z. Numeric fields must be finite; dimensions, mass and required effort/velocity limits
must be positive. Set SDK values in these units; exporters handle USD angular conversions.

## Transforms

### `Origin`

```python
Origin(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
)
```

- `xyz`: translation in meters.
- `rpy`: rotation in radians, composed as `R = Rz(yaw) * Ry(pitch) * Rx(roll)`.
- Both fields must contain exactly 3 finite numbers.

## Geometry Descriptors

Geometry descriptors describe shape only. Put transforms on the containing
`Visual` or `Collision` via `origin=Origin(...)`.

### `Box`

```python
Box(size: tuple[float, float, float])
```

- `size`: finite positive full extents `(sx, sy, sz)` in meters.

### `Cylinder`

```python
Cylinder(radius: float, length: float)
```

- `radius`: finite positive cylinder radius in meters.
- `length`: finite positive full cylinder length along local `+Z`.
- `height`: accepted as a compatibility alias for `length`, but prefer `length`
  in new code.

### `Sphere`

```python
Sphere(radius: float)
```

- `radius`: finite and positive sphere radius in meters.

### `Mesh`

```python
Mesh(
    filename: str | os.PathLike[str] | None = None,
    name: str | None = None,
    scale: tuple[float, float, float] | None = None,
    source_geometry: Box | Cylinder | Sphere | None = None,
    source_transform: tuple[tuple[float, float, float, float], ...] | None = None,
    materialized_path: str | None = None,
    normal_crease_deg: float | None = None,
)
```

- `filename`: absolute or asset-relative mesh path; either this or `name` is required.
- `name`: logical name for a managed/materialized mesh when no concrete filename is available yet.
- `scale`: optional finite positive per-axis scale.
- `source_geometry`: optional primitive provenance for meshes that were
  generated from a primitive.
- `source_transform`: optional provenance transform. This requires
  `source_geometry`; all 16 values must be finite.
- `normal_crease_deg`: visual-only face-normal smoothing threshold in degrees,
  from `0` (fully hard) to `180` (fully smooth); `None` keeps the established
  60-degree default. It does not alter geometry or collisions.

When provided, `filename` and `name` must be non-empty.

`normal_crease_deg` affects shading, not geometry or collision. `MeshGeometry` provides
`set_normal_crease_deg()`, `use_hard_normals()` and `use_smooth_normals()`; set the constructor
field on a `Mesh` descriptor. `Mesh` is immutable: use
`dataclasses.replace(mesh, normal_crease_deg=30.0)` to obtain a modified descriptor.
`materialized_path` is managed by the asset system.

## Materials and visuals

```text
Material(name, rgba=None, texture=None, *, color=None, metallic=None, roughness=None,
         emissive=None, transmission=None, ior=None, clearcoat=None,
         clearcoat_roughness=None, base_color_texture=None, metallic_texture=None,
         roughness_texture=None, normal_texture=None, occlusion_texture=None,
         emissive_texture=None, opacity_texture=None)
TextureInput(file, uv_set="st", wrap_s="repeat", wrap_t="repeat",
             source_color_space="auto", channel="rgb")
Visual(geometry, origin=Origin(), material=None, name=None, collidable=True)
```

`model.material(name, **fields)` registers a material and accepts the same appearance fields.
Use either `rgba` or its `color` alias, not both. RGB values are display-referred sRGB in [0, 1];
three components imply alpha=1. Alpha is linear opacity. `emissive` is linear RGB and may exceed 1.
A visual accepts a `Material` object or registered name. `part.visual(color=...)` aliases
`material`; choose one field. Assign materials per visual; mesh regions
requiring different materials need separate visuals.

| Field | Meaning and starting range |
| --- | --- |
| `metallic` | [0, 1]; dielectric 0, bare metal about 0.9–1 |
| `roughness` | [0, 1]; polished 0.05–0.2, satin 0.3–0.6, matte 0.7–1 |
| `transmission` | [0, 1]; transmissive glass/plastic, usually with alpha=1 |
| `ior` | At least 1; glass about 1.45, acrylic about 1.49 |
| `clearcoat` | [0, 1]; clear outer coating |
| `clearcoat_roughness` | [0, 1]; defaults to 0.2 when clearcoat is specified alone |

These are material hypotheses, not target dimensions or universal appearance recipes. For lightly
tinted glass, use a light base tint and transmission rather than making the base color nearly black.
If image feedback is available, assess lighting, background and rendering support before changing
a material to explain a dark preview. Numerical PBR values do not guarantee the observed appearance.

Each `*_texture` field accepts a path or `TextureInput`. Color textures use `sRGB`, data and normal
textures use `raw`. Packed ORM channels can share one file (`r` occlusion, `g` roughness, `b`
metallic). `texture` is the base-color shortcut. For example:

```python
from sdk import TextureInput
roughness_map = TextureInput("textures/packed_orm.png", source_color_space="raw", channel="g")
```

## Mass and inertia

```text
Inertia(ixx, ixy, ixz, iyy, iyz, izz)
Inertial(mass, inertia, origin=Origin())
Inertial.from_geometry(geometry, mass, *, origin=None) -> Inertial
estimate_mass_from_density(geometry, density, fill_fraction=1.0) -> float
```

`from_geometry` and `estimate_mass_from_density` accept primitive `Box`, `Cylinder` or `Sphere`,
not `Mesh`. Use a representative physical proxy. Mass is positive and density is kg/m³;
`0 < fill_fraction <= 1`. The estimate is `primitive_volume * density * fill_fraction`.
For thin/hollow/slatted construction, justify the fraction or use area × thickness × density.
The helper does not infer wall thickness, mesh volume or compound centre of mass.

```python
from sdk import Box, Inertial, estimate_mass_from_density
seat_proxy = Box((0.48, 0.25, 0.055))
wood_density = 650.0
seat_mass = estimate_mass_from_density(seat_proxy, wood_density, fill_fraction=0.45)
seat_inertial = Inertial.from_geometry(seat_proxy, mass=seat_mass)
```

Give every rigid Part explicit mass and contact properties. Derive empty-asset mass unless a loaded
state is requested. Reuse major geometry dimensions and nearby named density/fill assumptions;
avoid unsupported mass literals and unnecessary high-precision compound calculations.
`PhysicsMaterial.density` does not assign `Inertial.mass`. Runtime inertia may be derived from
colliders; do not distort geometry, mass or COM to silence a source-proxy diagnostic.

## Contact and body properties

```text
PhysicsMaterial(name, static_friction=0.5, dynamic_friction=0.5, restitution=0.0, density=None)
RigidBodyProperties(enabled=True, kinematic=False, physics_material=None,
                    collision_approximation="convexDecomposition")
```

Register contact material with `model.physics_material(name, **fields)` and bind its name through
`RigidBodyProperties(physics_material="name")`. Contact material and visual PBR are independent.
Require `static_friction >= dynamic_friction >= 0`; restitution is [0, 1], density positive if set.
Representative static/dynamic pairs: painted metal 0.40/0.30, hard plastic 0.50/0.35,
wood 0.55/0.40, rubber grip 0.90/0.70, low-friction guide 0.15/0.10. Choose by contact role.

`enabled` controls rigid-body participation; `kinematic` declares externally prescribed motion.
A lack of internal joints does not make an object kinematic. Use the root `BaseMode` and task's
physical intent instead of setting kinematic flags to keep an object still.

Mesh collision approximation defaults to `convexDecomposition`; disconnected components receive
separate colliders. Use `convexHull` only for connected components intended to behave as convex
solids. Keep physical surfaces collidable, including non-rigid rest-shape inspection surfaces.
Use visual-only graphics for decals; `collidable=False` is not a repair for a contact defect.
Author visuals and materialize procedural geometry as `Mesh`; do not fill `Part.collisions`
in the authored source model. Collisions are derived from those visuals.

### `BodyModel`

- `RIGID`: independently simulated rigid Part with `Inertial` and `RigidBodyProperties`.
- `FLEXIBLE_1D`, `SURFACE_DEFORMABLE`, `VOLUME_DEFORMABLE`: typed static-rest presentations.
  Omit `inertial` and `rigid_body`; any edge involving a non-rigid Part must be `FIXED`.
  Rest-shape collision geometry remains inspectable, but live deformation and load transfer are
  not simulated.

Represent each hose, rope, cable, chain, strap or belt as one `FLEXIBLE_1D` Part with a plausible
rest shape. Keep rigid fittings separate. A mesh named chain is not a dynamic chain. Distributed
flexible dynamics are outside the Author contract: do not build serial rigid segments or use
`add_segmented_flexible_member` to imply that behavior. Report a required unsupported dynamic
behavior rather than substituting static geometry while claiming equivalence.

## Scalar joint physics

```text
MotionLimits(effort=1.0, velocity=1.0, lower=None, upper=None)
MotionProperties(damping=None, friction=None)
PhysxJointFriction(static_friction_effort, dynamic_friction_effort,
                   viscous_friction_coefficient=0.0)
PhysxJointFrictionCoefficient(coefficient)
PassiveJointSpring(stiffness, reference=0.0)
JointDrive(mode="force", target_position=None, target_velocity=None,
           stiffness=0.0, damping=0.0, max_force=None)
NewtonJointSolverTuning(solref_friction=None, solimp_friction=None)
```

Author positive effort/velocity and explicit damping/friction for each scalar joint. Bounded
revolute/prismatic limits must contain q=0; continuous joints have no position bounds.
Angular effort is N·m; prismatic effort is N. `MotionProperties.damping` is passive viscous
resistance. Select `physx_joint_resistance` explicitly for PhysX. Absolute `PhysxJointFriction` static effort
also supplies Newton's `mjc:frictionloss`; its viscosity adds to Newton damping. Newton uses
`MotionProperties.friction` only when absolute effort resistance is absent. Set that field
deliberately for URDF and the Newton fallback; a load-proportional coefficient has no equivalent
Newton mapping. Separate static/dynamic thresholds are not preserved in Newton.

Select one explicit PhysX resistance model:

- `PhysxJointFriction`: absolute static/dynamic effort with dynamic <= static, plus optional
  non-negative viscosity. Use for predictable breakaway/hold behavior.
- `PhysxJointFrictionCoefficient`: dimensionless load-proportional bearing/bushing/shaft drag.
  Use only when that payload dependence is intended. It is mutually exclusive with the effort form.

```python
from sdk import MotionProperties, PhysxJointFriction
# Passive rotary joint: URDF friction and absolute resistance (also mapped to Newton).
passive_motion = MotionProperties(damping=0.01, friction=0.03)
passive_resistance = PhysxJointFriction.from_breakaway_effort(0.04, dynamic_ratio=0.75)
```

The example values are placeholders for a load estimate. `stay_put` sizes dry friction; it does
not add a spring, drive or Newton tuning.

### Gravity hold

Set joint `meta={"gravity_hold": True}` when the default mechanism must hold any permitted pose
against gravity; use `False` for deliberate gravity-driven motion or real modeled contact support.
For each pose q across the permitted range, sum loads from mechanically connected downstream
bodies, holding other moving joints at q=0 and excluding independent `FLOATING` branches:

- Angular load: `axis_world · ((COM_world - joint_world) × mass * gravity)`.
- Prismatic load: `axis_world · (mass * gravity)`.

Respect the joint frame rotation. Account for signed spring restoring effort
`stiffness * (reference - q)` and any force-saturated force-mode drive. Size dry friction from the
maximum absolute signed residual, not the object's full height or an unprojected bounding box.
A vertical hinge or horizontal slide has zero gravitational axis load. Include payload only when
it belongs to the requested/default modeled state.

For positive residual held by dry friction, explicitly add:

```python
from sdk import NewtonJointSolverTuning
hold_tuning = NewtonJointSolverTuning(solimp_friction=(0.999, 0.9999, 0.001, 0.5, 2.0))
```

Friction magnitude and Newton constraint impedance are independent. The task's pose-hold policy
requires this tuning for positive friction-held residual; a zero residual or `gravity_hold=False`
does not. This is an authoring policy, not a runtime guarantee.

`solref_friction` has two values: both positive `(timeconst, dampratio)` or both non-positive
`(-stiffness, -damping)`. `solimp_friction=(d0, dwidth, width, midpoint, power)` requires d0, dwidth
and midpoint in [0, 1], width > 0, power >= 1. There is no d0/dwidth ordering requirement. For dry
friction, d0 controls resting impedance and dwidth affects damping; neither sets friction effort.

### Passive return and active control

Use `PassiveJointSpring` for passive return to equilibrium: positive stiffness, reference in
radians or metres, damping in `MotionProperties.damping`. A spring and `JointDrive` are mutually
exclusive. Ordinary hand-operated doors, lids, knobs and drawers omit a drive unless an active
actuator is part of the target mechanism.

For an active drive, set `MotionProperties.damping=0.0` and put active damping in
`JointDrive.damping`. `max_force` must be finite, non-negative and <= `MotionLimits.effort`;
`None` uses that effort limit. Angular targets/gains remain in radians, not USD degrees.
`mode` is `"force"` or `"acceleration"`; keep explicit backend limitations below in mind.

```python
from sdk import JointDrive, MotionLimits, MotionProperties, PhysxJointFriction
# Example rotary actuator settings; derive limits, friction and gains for the mechanism.
active_limits = MotionLimits(effort=0.5, velocity=1.0, lower=0.0, upper=1.2)
active_motion = MotionProperties(damping=0.0, friction=0.02)
active_resistance = PhysxJointFriction.from_breakaway_effort(0.03, dynamic_ratio=0.8)
active_drive = JointDrive(target_position=0.0, stiffness=0.5, damping=0.05, max_force=0.5)
```

Pass these as `motion_limits`, `motion_properties`, `physx_joint_resistance` and `drive`
to the articulation. See `30_articulated_object.md` for a complete passive joint declaration.

### PhysX armature

`physx_armature` is non-negative joint-space inertia: kg·m² for angular joints, kg for slides.
Use the SDK field, not generated USD attributes. A real motor may contribute reflected rotor
inertia `J * G**2`. A reported extreme-ratio passive joint may instead need numerical compensation;
identify that intent in a comment and leave ordinary unflagged passive joints at zero.

The source-proxy screen uses declared source inertias projected through each link COM onto the
joint's q=0 world axis, or masses for a prismatic joint, within the native articulation component.
For the reported provisional 100:1 target, total compensation is
`max(0, J_max / 100 - J_child)` or `max(0, m_max / 100 - m_child)`, not an increment added twice.
This limited screen excludes parallel-axis, subtree, contact and collider-derived runtime effects.
It is not a verified stability threshold. Preserve plausible geometry and mass; validate behavior
in the intended runtime before claiming stability. This field has no automatic Newton mapping.

### Backend behavior relevant to authoring

| Feature | PhysX | Newton |
| --- | --- | --- |
| Passive damping/spring | Drive representation | Native joint damping/spring; no actuator |
| Active force drive | Position/velocity targets and gains | Verified static PD and positive-damping zero-stiffness forms |
| Velocity limit | Authored velocity limit | Metadata only; no enforced mapping |
| Active target velocity, acceleration mode, pure-P | Drive behavior | Requires runtime control/tuning; static equivalent is unverified |
| Absolute resistance | Separate static/dynamic/viscous values | Static effort becomes one friction-loss threshold; viscosity adds to damping |
| Load-proportional coefficient | Supported resistance model | Coefficient has no equivalent; legacy friction fallback is independent |
| Armature | Explicit joint-space inertia | Not automatically copied |
| Mimic | Metadata only | Native mimic declaration |

These are export capabilities, not evidence of simulator import or observed motion.

## Parts, articulations and poses

```text
Part(name, visuals=[], collisions=[], inertial=None, rigid_body=None,
     body_model=BodyModel.RIGID, meta={}, assets=None)
part.visual(geometry, *, origin=None, material=None, color=None, name=None,
            collidable=True) -> Visual
part.get_visual(name) -> Visual
Mimic(joint, multiplier=1.0, offset=0.0)
```

Prefer `model.part(...)` and `model.articulation(...)`; their full signatures and relationships
are in `30_articulated_object.md`. `part.visual` accepts a registered material name or object.
A `MeshGeometry` must first become a managed `Mesh`. Reading public `part.visuals`, `part.inertial`,
`part.rigid_body`, `part.body_model`, `part.meta` and named visual geometry/origin/material is supported.

`ArticulationType` has `REVOLUTE`, `CONTINUOUS`, `PRISMATIC`, `SPHERICAL`, `FIXED` and `FLOATING`.
Scalar poses are radians or metres. `SPHERICAL` requires `axis` and `cone_deg` in (0, 180]; the cone
bounds swing, not twist. Native USD keeps three rotational degrees of freedom; URDF is a semantic
fallback. `FLOATING` is an unconstrained placement edge, emits no USD joint, and takes an `Origin`
pose rather than a scalar. D6 and closed-loop constraints are unsupported.

Mimic followers satisfy `q_follower = multiplier * q_source + offset`; pose the source, not the
follower. Source/follower must have compatible scalar motion domains, without a mimic cycle.
Read joint `name`, `articulation_type` (`joint_type` alias), `parent`, `child`, `origin`, `axis`,
`cone_deg`, `motion_limits` (`limit`), `motion_properties` (`dynamics`), `newton_joint_solver_tuning`,
`passive_spring`, `physx_joint_resistance`, `physx_armature`, `drive`, `mimic` and `meta` directly.
Read `MotionLimits.effort`, `velocity`, `lower`, `upper`; optional bounds can be `None`.

## Resizing

```text
scale_geometry_to_size(geometry, target_size, *, mode="stretch", asset_root=None, filename=None)
```

`target_size=(sx, sy, sz)` accepts positive floats or `None` per axis. `stretch` matches extents;
`uniform` preserves proportions. Use `filename` if a primitive must become a mesh, such as a
nonuniformly scaled sphere. Meshes can use an explicit `asset_root`; runtime-managed asset paths
are described in `40_assets.md`. Keep the task's declared global scale and frame.
