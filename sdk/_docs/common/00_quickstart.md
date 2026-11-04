# SDK Quickstart

Use the public `sdk` package to author the staged `model.py`. The runtime task defines the target,
evidence authority, available tools and workspace. Keep object construction in the source model;
derived USD, URDF and collision assets are build outputs.

## Script contract

Define `build_object_model() -> ArticulatedObject`, `run_tests() -> TestReport`, and
`object_model = build_object_model()`. Import public symbols directly from `sdk`; documentation
filenames are topics, not Python submodule names.

Use metres, kilograms and radians. `Origin.rpy` is roll, pitch, yaw with
`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`. Give a standalone asset an explicit
`base_mode=BaseMode.FLOATING`; scene fixation and gravity belong to its consumer.

## Minimal executable model

This example defines a single rigid object. Its dimensions and density are example inputs.

```python
from sdk import (
    ArticulatedObject, BaseMode, Box, Inertial, Origin, RigidBodyProperties,
    TestContext, TestReport, estimate_mass_from_density,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="wood_block", base_mode=BaseMode.FLOATING)
    model.meta["mode"] = "static"
    size = (0.20, 0.12, 0.04)
    shape = Box(size)
    wood_density = 650.0  # kg/m³; solid wood proxy
    mass = estimate_mass_from_density(shape, wood_density)
    model.physics_material(
        "wood_contact", static_friction=0.55, dynamic_friction=0.40,
        restitution=0.05, density=wood_density,
    )
    finish = model.material("wood", rgba=(0.55, 0.32, 0.14, 1.0), roughness=0.65)
    body = model.part(
        "body", inertial=Inertial.from_geometry(
            shape, mass=mass, origin=Origin(xyz=(0.0, 0.0, size[2] / 2)),
        ),
        rigid_body=RigidBodyProperties(physics_material="wood_contact"),
    )
    body.visual(shape, origin=Origin(xyz=(0.0, 0.0, size[2] / 2)),
                material=finish, name="block")
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    bounds = ctx.part_world_aabb("body")
    ctx.check("block rests on ground", bounds is not None and abs(bounds[0][2]) < 1e-6)
    return ctx.report()


object_model = build_object_model()
```

## Geometry and assets

Part.visual accepts Box, Cylinder, Sphere or managed Mesh descriptors. Materialize procedural
MeshGeometry with `mesh_from_geometry(geometry, "logical_name")`; materialize CadQuery shapes
with mesh_from_cadquery (read its exact signature). Use mesh_from_input only for task-permitted
assets: target evidence is not implicitly an authored asset. Asset paths are runtime-managed;
use TestContext(object_model) without inventing asset roots. See `40_assets.md` for ownership.

Choose the construction from the needed operation, then read that reference's exact API:

| Operation | Reference under `sdk/_docs/` |
| --- | --- |
| Filleted solids, thin shells, cut-through holes or boolean cavities | `cadquery/35_cadquery.md` — CadQuery operations and `mesh_from_cadquery` |
| Rounded profiles and built-in procedural mesh constructors | `base/40_mesh_geometry.md` — `RoundedBoxGeometry`, `mesh_from_geometry` |
| Deliberately varying cross-sections | `base/46_section_lofts.md` |
| Sculpted organic surfaces or mesh smoothing | `base/50_organic_surfaces.md` |
| Printed text, labels and surface graphics | `base/54_decals_and_labels.md` — `TextureCanvas`, `Part.decal` |

Round a solid with fillets when that describes the shape; use lofts or smoothing when changing
sections or organic curvature require them. Build a simple version before surveying alternatives.

## Parts, joints and frames

Use one rooted tree: every non-root Part has exactly one parent relation. Keep rigid details in
one Part; separate constrained moving bodies use joints. A freely separable/contact-supported
object uses its own rigid Part and a `FLOATING` placement edge, not a fixed weld. Use
`model.meta["mode"]="articulated"` for constrained motion, `"static"` otherwise; a static object
can still have a floating root and FLOATING placement edges. Closed loops and multiple parents
are unsupported.

Common construction signatures (all symbols are public imports from `sdk`):

```text
model.part(name, *, visuals=None, inertial=None, rigid_body=None,
           body_model=BodyModel.RIGID, meta=None) -> Part
part.visual(geometry, *, origin=None, material=None, color=None, name=None,
            collidable=True) -> Visual
model.articulation(name, articulation_type, parent, child, *, origin=None, axis=None,
                   cone_deg=None, motion_limits=None, motion_properties=None,
                   newton_joint_solver_tuning=None, passive_spring=None,
                   physx_joint_resistance=None, physx_armature=None, drive=None,
                   mimic=None, meta=None) -> Articulation
```

Parent/child accept registered Part objects or names. `origin` locates the joint in the parent
frame. At q=0 the child frame coincides with that joint frame; put child visuals in child-local
coordinates, not world coordinates. `axis` is a unit vector in the joint frame. Rotation uses
the right-hand rule; translation follows the positive axis. Visual origins transform geometry
within a Part, independently of the joint origin. Check opening direction with a posed query.

| ArticulationType | Declaration |
| --- | --- |
| REVOLUTE | Bounded hinge: axis, effort/velocity, finite lower/upper containing zero |
| CONTINUOUS | Unlimited rotation: axis, effort/velocity; no position bounds |
| PRISMATIC | Bounded slide: axis, effort/velocity, finite lower/upper containing zero |
| FIXED | Permanent attachment; no scalar motion properties |
| FLOATING | Unconstrained placement; an Origin pose; no scalar axis/limits |
| SPHERICAL | Nonzero axis and cone_deg in (0, 180]; swing cone with free twist |

This fragment joins existing body/door Parts using modeled `hinge_xyz` and `door_handle_radius`.
The vertical hinge has zero gravity effort, requiring no positive-load Newton tuning.
Choose limits and resistance for the actual mechanism.

```python
from sdk import ArticulationType, MotionLimits, MotionProperties, PhysxJointFriction
handle_force = 2.0  # N; example manual breakaway force
breakaway = handle_force * door_handle_radius
model.articulation(
    "door_hinge", ArticulationType.REVOLUTE, "body", "door",
    origin=Origin(xyz=hinge_xyz), axis=(0.0, 0.0, 1.0),
    motion_limits=MotionLimits(effort=5.0, velocity=1.0, lower=0.0, upper=1.5),
    motion_properties=MotionProperties(damping=0.02, friction=breakaway),
    physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(
        breakaway, dynamic_ratio=0.75,
    ),
    meta={"gravity_hold": True},
)
```

Non-rigid `FLEXIBLE_1D`, `SURFACE_DEFORMABLE` and `VOLUME_DEFORMABLE` Parts describe static rest
shapes: omit inertial/rigid_body; all incident relations are FIXED. Keep rigid fittings separate.
Read the task's mechanical policy for flexible-member scope and joint-physics requirements.

## Type essentials

```text
Box(size)                         # full (x, y, z) extents, centered at origin
Cylinder(radius, length)           # full length along local Z, centered at origin
Sphere(radius)
Origin(xyz=(0, 0, 0), rpy=(0, 0, 0))
Inertial.from_geometry(geometry, mass, *, origin=None)
estimate_mass_from_density(geometry, density, fill_fraction=1.0)
RigidBodyProperties(enabled=True, kinematic=False, physics_material=None,
                    collision_approximation="convexDecomposition")
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

Values must be finite; dimensions/mass/effort/velocity positive. Mass helpers accept primitives,
not Mesh; fill_fraction is in (0,1]. PhysicsMaterial.density does not assign mass. No internal
joints does not imply a kinematic body.

Register contact using model.physics_material and bind its name through RigidBodyProperties.
Register appearance separately: `model.material(name, rgba=..., metallic=..., roughness=...)`.
Visuals accept Material objects or names. RGB is sRGB [0,1], alpha linear opacity; choose rgba or
color, not both. Metallic/roughness are [0,1]. Glass uses transmission [0,1], ior >=1, usually
alpha=1. Separate visuals for distinct mesh-region materials. Texture rules: `20_core_types.md`.

Author visuals; collision geometry is derived. Keep physical surfaces collidable; decals may opt
out. convexDecomposition is the mesh default; convexHull is for intended convex components.
Mesh is immutable (use dataclasses.replace); MeshGeometry is editable before materialization.

## Focused tests

Build a coarse working model first. In run_tests, test support, usable openings, travel direction,
retention and load coverage; chosen counts alone prove none of these. Return ctx.report() and run
the task's check command/tool for shared diagnostics and these tests.

```text
ctx = TestContext(object_model, seed=0)
ctx.check(name, ok, details="") -> bool
ctx.pose(joint_positions=None, **joint_name_positions)  # context manager
ctx.part_world_aabb(part) -> (minimum_xyz, maximum_xyz) | None
ctx.part_element_world_aabb(part, *, elem) -> (minimum_xyz, maximum_xyz) | None
ctx.part_world_position(part) -> xyz | None
ctx.expect_contact(link_a, link_b, *, contact_tol=1e-6,
                   elem_a=None, elem_b=None, name=None) -> bool
ctx.expect_gap(positive_link, negative_link, *, axis, min_gap=None, max_gap=None,
               max_penetration=None, positive_elem=None, negative_elem=None,
               elem_a=None, elem_b=None, name=None) -> bool
ctx.expect_overlap(link_a, link_b, *, axes="xy", min_overlap=0.0,
                   elem_a=None, elem_b=None, name=None) -> bool
ctx.expect_within(inner_link, outer_link, *, axes="xy", margin=0.0,
                  inner_elem=None, outer_elem=None, elem_a=None, elem_b=None,
                  name=None) -> bool
ctx.allow_overlap(link_a, link_b, *, reason, elem_a=None, elem_b=None)
ctx.allow_isolated_part(part, *, reason)
ctx.report() -> TestReport
```

Arguments accept Part names or objects; element selectors name visuals. Guard `None` query
results. `ctx.check` takes the name before the condition. Gap is positive-geometry minimum minus
negative-geometry maximum along world axis x/y/z. Positive `expect_within.margin` permits slack
outside the enclosing bounds; it does not require internal clearance. Whole-Part AABBs cannot
prove that a cavity is hollow; query named walls and check usable gaps. Projected overlap is not
3D contact. `expect_contact` establishes proximity; inspect penetration separately.

Use `with ctx.pose({"door_hinge": 1.0}):` for temporary radians/metres poses; queries inside use
that world pose and restore on exit. Test decisive interior/end poses, not only q=0. Mimic
followers derive from their source, so pose the source. Sampled checks do not prove continuous
collision-free motion.

An overlap allowance describes a real intentional interface, scoped to named elements for moving
parts; a broad Part-pair exemption can hide collisions across travel. Containment alone does not
need an allowance. Isolation allowances require a genuinely independent/contact-supported member
or justified ground arrangement; they cannot repair missing mounts. For disconnected-geometry
classifications, exact probes, spherical/FLOATING poses or other assertions, read `80_testing.md`.

## Further references

Use the task inventory's exact paths and search-returned examples. Full types/materials/backends:
`20_core_types.md`; assembly/mount/mimic helpers: `30_articulated_object.md`; assertions/diagnostics:
`80_testing.md`; managed assets: `40_assets.md`; mounting: `50_placement.md`; probes:
`70_probe_tooling.md`. Focused base/ and cadquery/ pages provide geometry families; prefer a fitting
family constructor before low-level meshes. Product evidence and observations are defined by the task.
