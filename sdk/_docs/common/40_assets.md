# Asset roots

Use stable logical names for managed meshes and one coherent local asset root per model.

| Value/helper | Result and asset ownership |
| --- | --- |
| `MeshGeometry` | Mutable procedural geometry; edit it before conversion to `Mesh`. |
| `mesh_from_geometry(geometry, "body")` | Returns a managed `Mesh`; accepts geometry and name, with no `assets` argument. |
| `mesh_from_cadquery(shape, "body", assets=assets)` | Returns a `Mesh`; accepts an explicit `AssetContext` for materialization. |
| `Mesh(filename=...)` / `Mesh(name=...)` | Immutable descriptor; provide a filename or logical name. Use `dataclasses.replace` to change descriptor fields. |

The model's `assets` context owns managed mesh resolution. When a geometry transform converts a
primitive to `Mesh`, supply the helper's `mesh_name` or filename argument. Read the exact helper
signature before passing asset options.

```python
from sdk import AssetContext, ArticulatedObject, BaseMode

assets = AssetContext.from_script(__file__)
model = ArticulatedObject("asset", assets=assets, base_mode=BaseMode.FLOATING)
```

## API

```text
AssetContext(root: str | Path, mesh_subdir: str = "assets/meshes")
AssetContext.from_script(script_path: str | Path) -> AssetContext
assets.asset_root -> Path
assets.mesh_dir -> Path
assets.ensure_mesh_dir() -> Path
assets.mesh_path(filename: str | Path, *, ensure_dir: bool = True) -> Path
assets.mesh_ref(filename: str | Path) -> str
```

`from_script` uses the script's parent directory. Relative mesh references resolve
from `asset_root`; `mesh_subdir` is relative to it. `mesh_path` returns a writable
path, while `mesh_ref` returns the model-facing relative reference, such as
`"assets/meshes/body.obj"`.

The CadQuery export helpers accept `assets=assets` when materializing geometry.
See `../cadquery/35_cadquery.md`. `TestContext(model)` uses the model's asset root.
Keep dependencies inside the workspace permitted by the task.
