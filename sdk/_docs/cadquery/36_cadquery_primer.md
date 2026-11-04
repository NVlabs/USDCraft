# CadQuery concepts

Use CadQuery for solid geometry with precise profiles, booleans, holes and fillets.
Attach the result using `mesh_from_cadquery(...)`; the SDK owns Parts and joints.
CadQuery dimensions are unitless. Author in metres or apply one `unit_scale` at
export as described in `35_cadquery.md`. CadQuery rotation angles use degrees.

## Topology

| Entity | Meaning |
| --- | --- |
| Vertex | Point |
| Edge | Curve between vertices |
| Wire | Connected edges |
| Face | Bounded surface, optionally with holes |
| Shell | Connected faces |
| Solid | Closed volume |
| Compound | Collection of shapes; it need not be one connected solid |

A closed-looking wire is not a solid. Extrude a closed profile or assemble closed
faces before using the result as a solid. Check `shape.isValid()` after difficult
operations; validity alone does not establish that a part has the desired volume.

## Workplane and direct shapes

```python
import cadquery as cq

workplane = cq.Workplane("XY").box(0.10, 0.06, 0.02)
solid = workplane.val()
parts = workplane.vals()
wrapped = cq.Workplane(obj=solid)
```

`val()` returns the first item on the current stack. After selecting edges or
drawing a wire, that item may not be the modeled solid. Use `findSolid()` for the
context solid; use `vals()` when multiple selected shapes matter. `newObject(...)`
replaces the stack in a linked Workplane; it does not fuse the shapes.

Workplane operations retain a parent chain. Direct `Shape` operations return shapes
without that modeling history. Prefer the higher-level API that expresses the needed
geometry; internal kernel objects are unnecessary for ordinary solid authoring.

## Sketches and assemblies

`cq.Sketch` combines 2D faces or connected edges. Reset selections before another
independent operation, and call `assemble()` after building an edge sketch.
See `38_cadquery_sketch.md` for holes, offsets and Workplane integration.

`cq.Assembly` places named components and can solve static constraints. It does not
define an SDK articulation or a simulated mechanism. See `39_cadquery_assembly.md`.

Some CadQuery methods use positional overload dispatch. For example, pass points
positionally to `Sketch.arc(...)`; keyword names alone may not select an overload.
Use the documented overload for the type of geometry being constructed.

## Choosing a construction

- Repeated holes: create a point array on one stable Workplane.
- Hollow case: subtract an inner solid or remove selected faces with `shell(...)`.
- Recess: subtract material at the intended surface; keep its floor and wall thickness.
- Rounded manufactured edges: select the intended edges and fillet within local thickness.
- Curved transition: loft corresponding profiles or sweep along a controlled path.

Keep measured dimensions and important datums explicit. Inspect local bounds after
geometry operations and after SDK placement to distinguish a shape error from a
coordinate error. The API index is `39c_cadquery_api_ref.md`.
