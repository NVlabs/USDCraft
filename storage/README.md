# Canonical storage

`storage/` owns the shared on-disk record/materialization contract, identifiers, path containment,
artifact validation, record locks, canonical promotion, rating and deletion
mutations, materialization replacement, and transaction recovery. Generation and Reconstruction
assemble different immutable revision contents, but they publish through the same storage
primitives.

Storage is deterministic application code. It does not start an Author, reinterpret semantic
quality, compile on read, select a best sample, or turn Viewer previewability into canonical
validity.

## Canonical layout

Every canonical record has mutable record-level metadata and immutable revision contents:

```text
data/records/<record-id>/
  record.json
  collections/workbench.json
  revisions/<revision-id>/
    revision.json
    model.py
    provenance.json
    inputs/...
    traces/...
```

A Generation revision additionally uses `prompt.txt` and `generation.json`; an optional inspiration
image is retained under `inputs/`. A Reconstruction revision instead uses `request.txt`,
`reconstruction.json` and `inputs/manifest.json`. `reconstruction.json` `input.guidance` records
the user's optional Reconstruction guidance inline: `null` for a run without guidance, otherwise the
normalized non-empty text (the contract rejects any other value; records promoted before the field
existed omit it).
Its input tree preserves the original source dependency closure with its approved-input snapshot
manifest, preprocessing/audit provenance, the exact Author-context snapshot, the semantic image
(provenance role `reconstruction_target`), AXIAL, initial scaffold, and source-overlay
carrier according to the storage-owned manifest. The Author inputs are exactly the normalized source
mesh, semantic image, scaffold, AXIAL and Author evidence manifest; raw source closures and
preprocessing internals remain Host-only inputs. A revision carries no Author notes file and no
sample index: each run publishes exactly one record from one Author.
In-session source/candidate render and measurement files are archived under
`traces/author_feedback/` with a path/hash index, separate from initial target evidence. Input
manifests declare `source_view_policy: author_requested`: the only initial Author image is the
semantic image, source views are rendered only on the Author's request, and the staged source
carrier is always snapshotted. The source-overlay carrier manifest is the
file key `source_overlay_manifest` (`inputs/preprocessing/source_overlay_manifest.json`, bound by
`channels.source_overlay.manifest_key`) with the audit channel `source_overlay` under
`inputs/preprocessing/audit/source_overlay`; it holds geometry bindings only, never camera images.

Derived artifacts are stored separately:

```text
data/cache/record_materialization/<record-id>/
  model.urdf
  model.usdz
  compile_report.json
  isaac/...
  assets/...
  renders/...          # optional non-canonical presentation output
```

The active source revision remains authoritative. A materialization is compiler-produced derived
state and is never an independently edited source model.

## Current contracts

[`artifact_contract.py`](artifact_contract.py) is the executable source of truth for schema/profile
constants, required artifacts/checks, hashes, dependency closures, input binding, and displayability.
The current contracts include:

- Generation manifest schema 3, compile-report schema 2, validation profile `generation`;
- Reconstruction manifest schema 1, product profile `reconstruction`, input manifest
  `reconstruction_input_manifest`, and the shared schema-2 / `generation` mechanical compile
  evidence;
- provenance schema 3 for lightweight runtime/source metadata without a clean-worktree publication
  requirement;
- Asset Structure 3.0 profile `minimal_shared_physics`, flattened PhysX and Newton entries, and
  current required backend evidence.

The validator accepts exactly these current formats, as written by the current Generation and
Reconstruction writers: records must declare `agent_harness` (`codex` with provider `codex-cli`, or
`provider` with the endpoint that served it as provider, `nvidia` or `openai`) and the matching
identity in every projection. Records from
earlier schema versions are reported as invalid; missing evidence is never fabricated and a record
is never silently upgraded.

Canonical validity and Viewer previewability are separate projections. Storage classifies a
materialization as exactly one of:

- `available` — the complete applicable contract passes;
- `contract_invalid` — the contract fails, but a saved `model.urdf` exists and the report does not
  explicitly say compilation failed;
- `missing` — the URDF needed for ordinary preview is absent;
- `compile_failed` — the compile report explicitly records failure.

`contract_invalid` may be displayed read-only, with its structured `{path, message}` issues still
visible. That never makes it canonical-valid or simulator-ready. `missing` and `compile_failed`
block ordinary model preview.

## Publication and mutations

[`canonical_promotion.py`](canonical_promotion.py) publishes one fully staged record plus one fully
staged materialization. Generation and Reconstruction product persistence own their revision
contents; the shared promotion owns conflict detection, locks, transaction identity, commit,
rollback, and recovery. Canonical paths are not mutated until the staged product-specific contract
has passed.

Reconstruction retains its atomic approved-input snapshot manifest and Storage cross-checks the
canonical copied source topology, semantic image, scaffold, hashes, and sizes against it.

Revision contents are immutable after publication. The supported control mutations are:

- an atomic 1–5 rating update (or clearing the rating) in `record.json`, for Generation and
  Reconstruction records alike (each caller names the pipeline it is bound to);
- permanently deleting one explicitly confirmed record and its materialization through a recoverable
  transaction, likewise for both canonical pipelines;
- the Generation-only, explicitly user-invoked `usdcraft rematerialize <record-id>` workflow,
  which may atomically replace the derived cache of a current-format record (schema-3
  `generation` manifest, schema-2 compile report, one consistent `promotion_transaction_id`
  across `record.json`, `generation.json` and the report) without editing its canonical revision.

Transactions use per-record locks and dedicated ignored staging roots:

- `data/local/promotion_staging/`
- `data/local/record_deletion_staging/`
- `data/local/materialization_replacement_staging/`
- `data/local/locks/records/`

These roots exist only for the named mutation and recovery protocol. They are never record indexes,
Viewer discovery sources, hydration queues, or permission to scan arbitrary local state. Promotion
and deletion recovery are process-crash recovery, not a claim of power-loss durability. Current
materialization replacement requires same-filesystem atomic exchange support on macOS or Linux.

## Component boundary and source map

- [`layout.py`](layout.py) — canonical paths.
- [`repo.py`](repo.py) — contained filesystem and JSON/text operations.
- [`identifiers.py`](identifiers.py) and [`revisions.py`](revisions.py) — IDs and active revision
  resolution.
- [`artifact_contract.py`](artifact_contract.py) — Generation/Reconstruction artifact validation and
  Viewer display classification.
- [`canonical_promotion.py`](canonical_promotion.py) — shared staged publication.
- [`record_locks.py`](record_locks.py) — re-entrant per-record locking.
- [`record_mutations.py`](record_mutations.py) — rating, deletion, promotion manifests, and recovery.
- [`materialization_mutations.py`](materialization_mutations.py) — exact-record cache exchange and
  recovery.
- [`collections.py`](collections.py) — canonical Workbench collection entries.
- `usdcraft/generation/persist.py` and `usdcraft/reconstruction/persist.py` — product-specific record
  assembly before shared promotion.

Viewer API code may project canonical files through narrow allowlists and may delegate an authorized
mutation to Storage. It must not reproduce contract/hash logic or compile, repair, hydrate, or
rematerialize during reads. See [`viewer/README.md`](../viewer/README.md).

## Tests

Primary regression coverage lives in:

- `tests/storage/test_promotion_recovery.py`
- `tests/storage/test_record_locks.py`
- `tests/storage/test_record_mutations.py`
- `tests/generation/test_persist.py`
- `tests/generation/test_rematerialize.py`
- `tests/reconstruction/test_reconstruction_persist.py`
- `tests/viewer/test_api.py`
- `tests/viewer/test_reconstruction_records.py`

Run the focused suite with:

```bash
uv run pytest -q tests/storage tests/generation/test_persist.py \
  tests/generation/test_rematerialize.py \
  tests/reconstruction/test_reconstruction_persist.py \
  tests/viewer/test_api.py tests/viewer/test_reconstruction_records.py
```

## Explicit validation

`validate_data_format()` is the Storage entry point used by both validation CLI commands. It
returns data-format errors and the materialization display projection from the same check.
`record_ids=None` enumerates raw record directories; `record_ids=[]` selects nothing. A product
filter skips readable records of the other product while reporting unreadable or unknown headers.
Explicit wrong-product ids fail. Missing Workbench registrations are validation errors rather
than a discovery filter. A selected materialization is validated once per invocation.

`generation-validate` and `reconstruction-validate` return 0 only when all requested checks pass,
1 for invalid data (even when a saved model can still preview), and 2 for invalid argument choices.
These manual commands do not depend on Viewer listing or cached validity and do not write records.
