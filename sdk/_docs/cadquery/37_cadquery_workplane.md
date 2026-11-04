# Workplane — CadQuery Documentation

A Workplane has a local XY drawing plane, an origin, a normal and a stack of
selected shapes. Build profiles in that local plane, then extrude, revolve, loft
or sweep. The name `"XY"` identifies the plane; it does not declare the object front.

CadQuery angles use degrees. Geometry is unitless: use metres consistently, or
convert once with `unit_scale` at SDK export (`35_cadquery.md`).

## Common Pitfall: Repeated Face Workplanes

When you create a new workplane from a selected face, CadQuery does not always keep the origin at the global model origin. By default it projects the previous workplane origin onto the selected face. That means repeated patterns like:

```python
for angle in angles:
    body = (
        body.faces(">Z")
        .workplane()
        .transformed(rotate=(0, 0, angle))
        .center(radius, 0)
        .circle(hole_radius)
        .cutThruAll()
    )
```

can drift after the first feature. The second and later cuts may be positioned relative to the previous cut's local frame instead of the model center.

For repeated radial or symmetric features, prefer one of these patterns:

- Build cutters from a fixed global workplane such as `cq.Workplane("XY")`, then subtract them with `body.cut(cutter)`.
- Tag and reuse a stable base workplane instead of recreating a fresh face workplane inside the loop.
- Use `centerOption=` intentionally when creating a face workplane if you need a specific face-local origin.

## Selectors

Selectors allow you to select one or more features, in order to define new features. As an example, you might extrude a box, and then select the top face as the location for a new feature. Or, you might extrude a box, and then select all of the vertical edges so that you can apply a fillet to them.

You can select Vertices, Edges, Faces, Solids, and Wires using selectors.

## Construction Geometry

Construction geometry are features that are not part of the object, but are only defined to aid in building the object. A common example might be to define a rectangle, and then use the corners to define the location of a set of holes.

Most CadQuery construction methods provide a `forConstruction` keyword, which creates a feature that will only be used to locate other features.

## The Stack

As you work in CadQuery, each operation returns a new Workplane object with the result of that operation. Each Workplane object has a list of objects, and a reference to its parent.

You can always go backwards to older operations by removing the current object from the stack. For example:

```python
Workplane(obj=someObject).faces(">Z").first().vertices()
```

returns a CadQuery object that contains all of the vertices on the highest face of someObject. But you can always move backwards in the stack to get the face as well:

```python
Workplane(obj=someObject).faces(">Z").first().vertices().end()
```

## Chaining

All Workplane methods return another Workplane object, so that you can chain the methods together fluently. Use the core Workplane methods to get at the objects that were created.

Each time a new Workplane object is produced during these chained calls, it has a `parent` attribute that points to the Workplane object that created it. Several CadQuery methods search this parent chain, for example when searching for the context solid. You can also give a Workplane object a tag, and further down your chain of calls you can refer back to this particular object using its tag.

## The Context Solid

Most of the time, you are building a single object, and adding features to that single object. CadQuery watches your operations, and defines the first solid object created as the 'context solid'. After that, any features you create are automatically combined (unless you specify otherwise) with that solid. This happens even if the solid was created a long way up in the stack. For example:

```python
Workplane("XY").box(0.10, 0.08, 0.03).faces(">Z").workplane().circle(0.01).extrude(0.01)
```

Creates a box with a cylindrical boss extending from its top face. It was not necessary to manually combine the cylinder created by extruding the circle with the box, because the default behavior for extrude is to combine the result with the context solid. The `hole()` method works similarly – CadQuery presumes that you want to subtract the hole from the context solid.

If you want to avoid this, you can specify `combine=False`, and CadQuery will create the solid separately.

## Iteration

CAD models often have repeated geometry, and it's really annoying to resort to for loops to construct features. Many CadQuery methods operate automatically on each element on the stack, so that you don't have to write loops. For example, this:

```python
Workplane("XY").box(1, 2, 3).faces(">Z").vertices().circle(0.5)
```

Will actually create 4 circles, because `vertices()` selects 4 vertices of a rectangular face, and the `circle()` method iterates on each member of the stack.

This is really useful to remember when you author your own plugins. `cadquery.Workplane.each()` is useful for this purpose.
