# Compile and export

This package owns executable-model checks, timeout-bounded compilation, strict URDF/USD export
validation, and compile feedback. It is a mechanical component shared by Generation and Reconstruction; it does not own either product's
request semantics, authoring prompt, record schema, or promotion decision.

## Public entry points

```bash
uv run usdcraft check /absolute/model.py
uv run usdcraft compile /absolute/model.py --output /absolute/build
uv run usdcraft compile /absolute/model.py --output /absolute/build --no-render
```

`--no-render` skips presentation renders. By default an existing output directory is removed before
compilation; `--no-clean` keeps existing contents.

`usdcraft check` runs model-authored `run_tests()` and compiler-provided authoring diagnostics. It
returns repair feedback but creates no build directory, compile report, canonical record, or
promotion result. Generation and Reconstruction Authors use it while `model.py` is still editable;
the final Host deliberately does not replay it as a semantic gate.

`usdcraft compile` executes trusted Python, exports the model, validates the complete mechanical
artifact, and writes `compile_report.json`. Standalone compilation permits every SDK-supported root
mode. Product Host calls additionally require a floating asset root and set `render=False`.
Generation authoring compile and Reconstruction authoring compile/re-encode also use the no-render
path. Standalone rendering is optional presentation output and a render failure is a warning, not an
export failure.

`model.py` is executable code. Compilation always runs it in a worker process bounded by
`USDCRAFT_COMPILE_TIMEOUT_SECONDS` (finite positive seconds; unset or empty means 300). Any other
value, including zero or a negative number, is a configuration error; there is no in-process mode.
`compile_timeout_seconds()` in [`compiler.py`](compiler.py) is the single parser, also used by the
Provider pre-check before any model call. The worker timeout contains accidental nontermination; it
is not an operating-system sandbox, memory limit, network boundary, or permission boundary.

## Mechanical artifact contract

One successful current compile produces the Asset Structure 3.0
`minimal_shared_physics` profile:

```text
model.urdf
model.usdz
compile_report.json
isaac/
  asset.usda
  physics.usdc
  physx.usda
  mujoco.usda
  model.usdc
  model_newton.usdc
  textures/...          # only when referenced
assets/...              # managed URDF mesh dependencies, when referenced
<relative paths>/...    # other URDF dependencies, preserving their authored paths
renders/...             # optional standalone presentation output
```

`isaac/model.usdc` is the independently openable flattened PhysX entry.
`isaac/model_newton.usdc` is the independently openable flattened Newton entry. The modular stack
is `asset.usda` plus shared `physics.usdc` and thin `physx.usda` / `mujoco.usda` overlays.
`Physics=none` intentionally loads no payload in this minimal profile. Reusable asset layers do not
author a scene-level `PhysicsScene`.

The compile path mechanically verifies, among other things:

- model execution and strict SDK/export structure;
- required URDF, USDZ, Asset Structure, and flattened backend files;
- stage opening, `/Asset`, Z-up, units, layer namespaces, and variant composition;
- backend schema separation, legal physics topology, and exact articulation-root bindings;
- finite numeric values actually serialized into URDF, authored USD attributes, and custom data;
- safe relative URDF references and complete contained USD dependency closures;
- file and dependency hashes recorded in the schema-2 report;
- the floating root when the calling Generation or Reconstruction Host requests it.

The canonical file/profile/check constants and materialization validators live in
[`storage/artifact_contract.py`](../../storage/artifact_contract.py). Product persistence must run
that storage-owned contract again over the staged record and materialization before publication.
Compile copies every safe relative URDF dependency from the model's asset directory, records its
hash and uses the storage-owned URDF dependency validator before reporting success. Missing files
and collisions with compiler-owned outputs are returned as repairable compile errors.

## Backend documentation boundary

Public SDK authoring semantics, units, output layout, drive/friction conversion, PhysX armature,
Newton solver tuning, mimic, and known degraded mappings are documented for Authors in:

- [`sdk/_docs/common/20_core_types.md`](../../sdk/_docs/common/20_core_types.md)
- [`sdk/_docs/common/30_articulated_object.md`](../../sdk/_docs/common/30_articulated_object.md)

The exporter implementation is
[`sdk/_core/_usd_export.py`](../../sdk/_core/_usd_export.py), with the PhysX/Newton backend layers
and asset interface in [`_usd_backend_layers.py`](../../sdk/_core/_usd_backend_layers.py), strict
post-export validation in [`_usd_validate.py`](../../sdk/_core/_usd_validate.py) and shared layer
names in [`_usd_common.py`](../../sdk/_core/_usd_common.py). Keep implementation, SDK docs,
strict validation, and regression tests aligned. Do not duplicate those author-facing field rules in
repository entry instructions.

The local checks establish static writability, composition, dependency closure, structural legality,
and recorded evidence. They do not establish an Isaac Sim import, extension startup, solver step,
stability, performance, or behavior result; runtime validation happens downstream, outside this
repository.

## Source map

- [`command.py`](command.py) — `check_model`, `compile_model`, artifact assembly, closures, hashes,
  optional renders, and report writing.
- [`compiler.py`](compiler.py) — model execution, SDK diagnostics, URDF/USD exporter invocation, and
  compile signals.
- [`physics_checks.py`](physics_checks.py) — compiler-owned generated-physics checks (movable
  bodies, PhysX armature stability, gravity-hold coverage).
- [`test_report.py`](test_report.py) — `run_tests()` invocation and merging with the compiler-owned
  baseline report.
- [`mp_utils.py`](mp_utils.py) — worker-process context (always the `spawn` start method).
- [`models.py`](models.py) and [`feedback.py`](feedback.py) — structured report/repair feedback.

## Tests

Primary regression coverage lives in:

- `tests/compile/test_command.py`
- `tests/compile/test_feedback_locations.py`
- `tests/compile/test_finite_contract.py`
- `tests/sdk/test_usd_export.py`
- `tests/sdk/test_finite_serialization.py`
- `tests/sdk/test_urdf_material_export.py`

Run the focused suite with:

```bash
uv run pytest -q tests/compile tests/sdk/test_usd_export.py \
  tests/sdk/test_finite_serialization.py tests/sdk/test_urdf_material_export.py
```
