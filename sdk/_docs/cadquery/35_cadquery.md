# CadQuery Helpers

Use CadQuery to construct precise solids, then attach their meshes to SDK Parts.
CadQuery assemblies arrange geometry; SDK articulations define motion.

## Import

```python
import cadquery as cq

from sdk import (
    cadquery_local_aabb,
    export_cadquery_components,
    export_cadquery_mesh,
    mesh_components_from_cadquery,
    mesh_from_cadquery,
    save_cadquery_obj,
    tessellate_cadquery,
)
```

## Recommended Surface

- `cadquery_local_aabb(...)`
- `tessellate_cadquery(...)`
- `export_cadquery_mesh(...)`
- `export_cadquery_components(...)`
- `mesh_from_cadquery(...)`
- `mesh_components_from_cadquery(...)`

Compatibility helper:

- `save_cadquery_obj(...)`

## Units

CadQuery is unitless. In `sdk`, authored geometry must land in meters.

- If the CadQuery model is already authored in meters, keep `unit_scale=1.0`.
- If it is authored in millimeters, either convert the literals to meters or
  pass `unit_scale=0.001` exactly once at export time.
- `unit_scale` applies to the entire tessellated result, including solved
  assembly/component locations.

## Helper Reference

### `tessellate_cadquery(...)`

```python
tessellate_cadquery(
    model: object,
    *,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
    unit_scale: float = 1.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]
```

- Returns raw vertices and triangle faces.
- Use this when you need tessellated geometry directly rather than an exported
  OBJ.

### `cadquery_local_aabb(...)`

```python
cadquery_local_aabb(
    model: object,
    *,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
    unit_scale: float = 1.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]
```

- Returns the local axis-aligned bounding box of the tessellated CadQuery
  result.
- This is useful when you need a size/placement probe before attaching the mesh
  into an `ArticulatedObject`.

### `export_cadquery_mesh(...)`

```python
export_cadquery_mesh(
    model: object,
    name: str,
    *,
    assets=None,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
    unit_scale: float = 1.0,
) -> CadQueryMeshExport
```

```python
CadQueryMeshExport(
    mesh: Mesh,
    local_aabb,
    center_xyz,
    size_xyz,
)
```

- Materializes the OBJ internally and returns the managed `sdk.Mesh` plus local
  bounds.
- `name`: stable logical mesh name such as `"door_panel"`.
- `assets`: optional explicit asset owner or root. Read
  `../common/40_assets.md` when you need stable on-disk paths outside the
  managed harness.
- `tolerance`: linear tessellation tolerance.
- `angular_tolerance`: angular tessellation tolerance in radians.
- `unit_scale`: scales both tessellated geometry and CadQuery assembly
  locations.

### `export_cadquery_components(...)`

```python
export_cadquery_components(
    model: object,
    name: str,
    *,
    assets=None,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
    unit_scale: float = 1.0,
) -> list[CadQueryMeshExport]
```

- Splits a CadQuery multi-shape workplane or assembly into one exported mesh per
  component.
- Uses numbered managed names such as `"pair__component_001.obj"` under the
  chosen logical name.
- Returns one `CadQueryMeshExport` per resolved component in CadQuery order.

### `mesh_from_cadquery(...)`

```python
mesh_from_cadquery(
    model: object,
    name: str,
    *,
    assets=None,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
    unit_scale: float = 1.0,
) -> Mesh
```

- This is the normal helper for attaching a CadQuery-authored visual to a part.
- Parameters match `export_cadquery_mesh(...)`, but only the managed `Mesh` is
  returned.

### `mesh_components_from_cadquery(...)`

```python
mesh_components_from_cadquery(
    model: object,
    name: str,
    *,
    assets=None,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
    unit_scale: float = 1.0,
) -> list[Mesh]
```

- Component-only convenience wrapper over `export_cadquery_components(...)`.
- Use this when one CadQuery source should become multiple separate mesh visuals
  or subfeatures.

### `save_cadquery_obj(...)`

```python
save_cadquery_obj(
    model: object,
    name: str,
    *,
    assets=None,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
    unit_scale: float = 1.0,
) -> Path
```

- Public compatibility helper that returns the materialized OBJ path directly.
- Prefer `export_cadquery_mesh(...)` or `mesh_from_cadquery(...)` for new code
  when you want the managed `Mesh` object.

## Attach geometry in the Part frame

This fragment assumes an existing `door` Part with its mass, contact material and
articulation authored as described in the four foundational pages.

```python
import cadquery as cq
from sdk import mesh_from_cadquery

door_shape = (
    cq.Workplane("XY")
    .box(0.58, 0.02, 0.78)
    .edges("|Z").fillet(0.008)
)
door.visual(mesh_from_cadquery(door_shape, "door", assets=door.assets), name="panel")
```

- Author a shape in its Part frame or apply `Visual.origin` explicitly. Export
  preserves the local frame and does not infer hinge pivots or recenter geometry.
- `Shape`, `Workplane` and `Assembly` are accepted by these helpers. Solve any
  constrained assembly before exporting; export does not call `solve()`.
- `mesh_from_cadquery` returns one combined mesh. Use the component helpers for
  separate visuals or material regions. Colors, face materials and texture
  coordinates from CadQuery are not preserved; assign SDK materials to visuals.
- Keep the mass, collision/contact properties and joints explicit on SDK Parts.
- Reuse stable workplanes for repeated features; see `37_cadquery_workplane.md`.

## See Also

- `../common/80_testing.md` for the shared testing API
- `../common/00_quickstart.md` for the overall script contract
- `39d_cadquery_gears.md` for the vendored gear builders and Workplane gear
  plugin helpers
- `../common/40_assets.md` for explicit asset-root control
