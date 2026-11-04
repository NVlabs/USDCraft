# Public SDK documentation map

`sdk/` is the canonical Python authoring surface used by Generation, Reconstruction, standalone
trusted models, and downstream SDK callers. This README is a repository index. It is not
part of an Author's automatically declared documentation inventory unless it is explicitly added to
[`sdk/_profiles.py`](_profiles.py).

## Public surface and implementation

- `sdk` is the supported public import namespace; it has no public submodules.
- [`_profiles.py`](_profiles.py) is the single exact scaffold/document inventory used by Generation
  and Reconstruction prompt builders and by the Provider documentation tools.
- [`_docs/`](_docs/) contains public authoring documentation.
- [`_examples/`](_examples/) contains discoverable examples; callers enumerate exact paths before
  opening a relevant example.
- [`downstream_examples/`](downstream_examples/) contains direct SDK demonstrations for fixed
  scene supports and segmented dynamics. It is outside Author documentation search/read roots.
- [`_core/`](_core/) contains the base SDK implementation, URDF/USD exporters, testing helpers,
  and built-in mesh operations.
- [`_extensions/cadquery/`](_extensions/cadquery/) contains the CadQuery extension and the vendored
  gear generator.

The `_core/mesh.py` import facade re-exports geometry helpers, while geometry implementations call their
owning domain modules directly. Internal tests patch helpers in `sdk/_core/_mesh/`, not private
names on the facade; facade imports do not select runtime geometry behavior.

Prompt policy belongs to the Generation or Reconstruction runtime, not to the SDK: the physical-
authoring policy both products render into their Author tasks lives in
[`usdcraft/pipeline/policy.py`](../usdcraft/pipeline/policy.py). Repository architecture,
Storage, Viewer, release notes, and product evidence rules must not be added to the SDK Author
inventory merely to move them out of root instructions, and the Author-facing documents under
`_docs/` and `_examples/` name neither the Viewer nor a single product where a rule binds every
USDCraft Author.

## Documentation routing

Start with [`_docs/common/00_quickstart.md`](_docs/common/00_quickstart.md). The profile provides
exact read-only paths; it is a routing map, not an instruction to read every document. An Author
should open only the subset relevant to the current model and use the exact example inventory rather
than guess filenames.

Common routes include:

- [`_docs/common/20_core_types.md`](_docs/common/20_core_types.md) — units, geometry/material/physics
  types and backend capabilities relevant to authoring.
- [`_docs/common/30_articulated_object.md`](_docs/common/30_articulated_object.md) — Parts,
  relationships, frames, roots, and joint rules.
- [`_docs/common/40_assets.md`](_docs/common/40_assets.md) — managed local dependencies.
- [`_docs/common/50_placement.md`](_docs/common/50_placement.md) — placement and surface mounting.
- [`_docs/common/70_probe_tooling.md`](_docs/common/70_probe_tooling.md) — authoring probes.
- [`_docs/common/80_testing.md`](_docs/common/80_testing.md) — `run_tests()` and assertions.
- [`_docs/base/`](_docs/base/) — built-in procedural geometry families.
- [`_docs/cadquery/`](_docs/cadquery/) — CadQuery authoring and reference material.

`SdkProfile.docs_full` is the complete declared routing inventory. Adding, removing, or
renaming public documentation requires updating `_profiles.py`, prompt/profile tests, and any
affected examples. A repository component README should not be added to that inventory unless the
asset Author genuinely needs it.

Generation and Reconstruction render the exact `docs_full` paths into their tasks on both arms.
The compact `_docs/common/00_quickstart.md` is preloaded verbatim through
`usdcraft/pipeline/sdk_context.py`. It includes an executable model, common signatures, frames and
focused tests. Full core types, assembly, testing, category references and examples remain on demand.
The inventory states the repository root once and lists canonical repo-relative paths. These files
address the runtime Author and contain current APIs and actionable constraints, not change history,
transport setup or Host acceptance internals. The Provider arms search the same inventory and
limit reads to `sdk/_docs`, `sdk/_examples`, and the staged model surface.

## Export documentation boundary

Author-facing PhysX/Newton field semantics and known degraded mappings remain in
`_docs/common/20_core_types.md` and `_docs/common/30_articulated_object.md`. Mechanical compile,
export validation, Host/no-render behavior, and artifact claims belong to
[`usdcraft/compile/README.md`](../usdcraft/compile/README.md). Canonical record and materialization
rules belong to [`storage/README.md`](../storage/README.md).

## Tests

Parametric knobs use an analytic cylinder when their circular loft sections have identical
radii. This avoids an OCC spline/fillet/scallop degeneracy without removing the authored grip or
indicator. Shared patterned CadQuery cuts check the raw Boolean solid before same-domain cleanup;
an invalid solid raises repair feedback instead of entering native cleanup with corrupt topology.
The bounded native regressions live in `tests/sdk/test_knob_boolean_regression.py`.

The SDK suite covers public imports, profiles, geometry, testing helpers, dependencies, URDF, USD,
backend vendor tokens, finite serialization, and examples:

```bash
uv run pytest -q tests/sdk
```

Documentation inventory coverage is centered in `tests/sdk/test_profiles.py`; Generation and
Reconstruction prompt tests verify that their declared SDK routing stays aligned. The example
derivation tests (`tests/sdk/test_base_example_derivations.py`,
`tests/sdk/test_cadquery_example_derivations.py`) execute the code blocks of the examples they
list; when an example is added or changed, add it there so it keeps building.
