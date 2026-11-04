# CadQuery assemblies

Use `cq.Assembly` to position reusable geometry components before mesh export.
Locations and solved constraints describe a static arrangement. Define moving
Parts and joints separately through `ArticulatedObject`.

## Explicit locations

All dimensions below are metres; CadQuery rotation angles are degrees.

```python
import cadquery as cq
from sdk import mesh_components_from_cadquery

plate = cq.Workplane("XY").box(0.12, 0.08, 0.01)
boss = cq.Workplane("XY").circle(0.01).extrude(0.02)
assembly = cq.Assembly(name="mount")
assembly.add(plate, name="plate")
assembly.add(boss, name="boss", loc=cq.Location(cq.Vector(0.03, 0, 0.005)))
meshes = mesh_components_from_cadquery(assembly, "mount")
```

Child locations are relative to their parent assembly; nested transforms compose.
SDK component export bakes those locations into mesh coordinates. Do not apply the
same location a second time to the resulting Visual. Export different material
regions separately and assign SDK materials; CadQuery colors are not transferred.

## Constraints

Use `.constrain(...)` then `.solve()` before export. Fix one reference component to
remove free global motion. Use explicit locations when the placement is already
known; an underconstrained solve can choose an unintended orientation.

```python
assembly.constrain("plate", "Fixed")
assembly.constrain("plate@faces@>Z", "boss@faces@<Z", "Plane")
assembly.solve()
```

Query strings can select named components, tagged geometry or faces/edges, such as
`"plate@faces@>Z"`. Inspect the resulting locations and contacts after solving.
Assembly constraints do not become runtime SDK joints, drives or contact laws.

## Constraint reference

| Constraint | Effect |
| --- | --- |
| `Fixed` | Keeps one component at its initial location |
| `Point` | Aligns centers; optional scalar distance has no fixed direction |
| `Axis` | Aligns directions; default 180 degrees mates opposing normals, 0 makes them parallel |
| `Plane` | Combines point and axis constraints for mating faces |
| `PointInPlane` | Places a point on a plane, optionally at a signed offset |
| `PointOnLine` | Places a point on a line |
| `FixedPoint` | Sets a point to explicit coordinates |
| `FixedRotation` | Constrains component rotation with Euler angles in degrees |
| `FixedAxis` | Aligns a selected direction to an explicit vector |

Select the real mating face/edge, not an unrelated geometric center. Constraining
one face does not necessarily remove in-plane translation or rotation. Add only
the constraints needed to specify the intended static arrangement, solve, and
check that the result is consistent before exporting.
