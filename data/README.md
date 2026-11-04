# Local data layout

`data/` contains canonical records, derived materializations, disposable or recoverable local
workflow state, and optional user-built export packages. Generated contents beneath these roots are
intentionally ignored by Git. Code and documentation must not treat an ignored directory scan as a
canonical record index.

## Roots

```text
data/
  records/                               # canonical Generation and Reconstruction records
  cache/
    record_materialization/              # validated derived artifacts by record ID
  local/
    generation_runs/                     # disposable/retained Generation workspaces
    generation_deadlines/                # per-row Batch supervisor: request, progress, stop snapshots
    reconstruction_runs/                 # Reconstruction shared preprocessing/evidence workspaces
    reconstruction_deadlines/            # per-row Batch supervisor: request, progress, stop snapshots
    reconstruction_author_runs/          # isolated Reconstruction Author workspaces
    promotion_staging/                   # recoverable canonical publication transactions
    materialization_replacement_staging/ # exact-record cache replacement transactions
    record_deletion_staging/             # recoverable explicit deletions
    locks/records/                       # per-record process locks
  exports/                               # optional non-canonical downstream packages
```

`usdcraft init` (also run by `usdcraft setup`) creates the ordinary record, cache, Generation, and
Reconstruction run roots and `materialization_replacement_staging/`. The other transaction roots and
the Batch `*_deadlines/` roots are created lazily by the operation that owns them.

`data/records/` is the canonical discovery root. `data/cache/record_materialization/` is derived
state bound back to one exact record/revision. Nothing under `data/local/` or `data/exports/` is a
canonical record, and those directories must not be used to infer record existence or success.

See [`storage/README.md`](../storage/README.md) for record, materialization, validation, mutation, and
recovery contracts.

## Optional record export packages

USDCraft has no CLI command for packaging several records for Isaac Sim. If you want such a package,
you can assemble it by hand under the ignored `data/exports/` root. Never modify canonical records or
their materializations while packaging.

1. Pick the exact record IDs (to find the latest ones, sort by `record.json` `created_at`, not by
   directory names or file times) and check that each
   `data/cache/record_materialization/<record-id>/compile_report.json` reports `status: success`.
2. Choose the backend entry per record: `isaac/model.usdc` (PhysX) and/or
   `isaac/model_newton.usdc` (Newton). Do not substitute `isaac/asset.usda`, a backend overlay,
   `isaac/physics.usdc` or `model.usdz` for the flattened runtime entry.
3. Keep a separate tree per record, e.g. `data/exports/<package>/<record-id>/isaac/model.usdc`.
   Never merge different records' `isaac/` trees: names such as `physics.usdc`, `textures/` or
   `mujoco.usda` are record-local.
4. For each selected entry, run `pxr.UsdUtils.ComputeAllDependencies` and copy every external
   dependency (textures and other assets) beneath that record's package folder, preserving its path
   relative to the materialization root. Reject unresolved, absolute, escaping or missing
   dependencies, and never overwrite differing content.
5. Verify the package: compare source and destination SHA-256 hashes, open every packaged stage with
   `pxr.Usd.Stage.Open` (default prim `/Asset`, Z-up, metre and kilogram units), and rerun dependency
   discovery from the packaged location expecting zero unresolved or missing dependencies.

A package note should state that the USDCraft static artifact gate does not prove Isaac Sim import,
runtime startup, simulation stability or sim-readiness for either backend. Exported packages and any
downstream results do not change the source record, its compile report or canonical validity.
