# Viewer

Viewer is a local inspection application over canonical Generation records and canonical
Reconstruction records. `viewer/api/` owns the FastAPI projections and contained file resolution;
`viewer/web/` owns the React/Three.js interface. Neither layer is a compiler, canonical validator,
reconstruction evaluator, or semantic reviewer.

Start the application with:

```bash
uv run usdcraft viewer
```

The default bind is `127.0.0.1:8770`; `--port` changes the port and `--rebuild` forces a frontend
rebuild, `--host` changes the bind address, and `--read-only` serves a review copy (see below). The
application has no authentication; a non-loopback deployment needs an external access-control
boundary.

Every request must carry a loopback `Host` header — `localhost`, `127.0.0.1` or `[::1]`, any
port — or a host listed in `USDCRAFT_VIEWER_ALLOWED_HOSTS` (comma-separated; `usdcraft viewer`
adds its `--host` value there); anything else answers 400 (`api/host_guard.py`). This keeps
a DNS-rebound web page, which reaches the local port under its own domain name, away from the
unauthenticated API. A wildcard bind (`0.0.0.0`, `::`) cannot be reduced to a Host allowlist and
turns the check off with a startup warning; that deployment relies on the external boundary above.

## Application modes

The frontend has two explicit modes, selected in the header and mirrored in the URL (`?mode=`):

1. **Generation Viewer** — the default canonical Generation Workbench.
2. **Recon Viewer** — canonical records whose `record.json` declares `pipeline=reconstruction`; it
   reuses the compiled-model Workbench and adds Reconstruction-only surfaces.

Canonical record discovery comes from `data/records/`. An unknown or absent `?mode=` opens the
Generation Workbench.

Each mode only fetches its own product's record list: Recon Viewer never issues the Generation
`GET /api/bootstrap` scan, and the Generation Workbench never issues
`GET /api/reconstruction-records`, so loading one product's records never waits on the other
product's disk scan.

## Look and layout

The Viewer has a single light theme; every colour is a CSS token in `web/src/index.css`, and the
Three.js viewport uses the matching light background and grid colours
(`web/src/components/viewer3d/lighting.ts`). Headless review renders (`web/render.html`) keep their
own pinned backdrop (`REVIEW_BACKGROUND` in `web/src/render/review-profiles.ts`). The type is Sora
with IBM Plex Mono for identifiers and readings, served locally from the OFL-licensed
`@fontsource/sora` and `@fontsource/ibm-plex-mono` packages, so the Viewer makes no third-party font
requests. An NVIDIA-green family is the single product accent in both modes, with amber and red reserved for warnings. The header carries the generated
USDCraft layered-scene mark, wordmark and a two-product switch (Generation / Reconstruction); it
does not add storage-state slogans to the product navigation. The inspector tabs are named for what
they show — **Model**, **View**, **Source**, **Evidence**, **Diagnostics**, **Runtime**. The
Generation record list keeps its browsing rows deliberately sparse: asset title and authoring model,
followed by the manual star-rating control. Record ids, runtime status, effort, turn count, cost and timestamps stay out of the list
and remain available in Runtime metadata when needed. The
Reconstruction record list switches between a card list and a dense table (author, model,
scale, AXIAL recall, exterior excess, rating, updated). Record ids and the
constant `reconstruction_ready` state are omitted from the browsing surface; they remain available
in record metadata and destructive confirmation. Recall is shown as a prominent percentage with a
60% visual review split, and an exterior-excess reading above 1.6% of the candidate's columns is
highlighted (smaller shares are the fringe every smooth member leaves around a lumpy source). The viewport
shows the loaded model's width × depth × height in metres so a mis-scaled record is visible at a
glance. Search stays visible in both record libraries; the denser filter controls are collapsed by
default and retain an active filter count when closed. The fonts are loaded from Google Fonts at
runtime; offline the system fallbacks apply.

Both record lists fold batch runs: records whose `runtime_report.batch.batch_id` matches (one
`generation-batch` or `reconstruction-batch` CSV run) collapse into one group header showing the
CSV stem, the UTC start, the record count, the mean AXIAL recall (Reconstruction) and the ratings.
Groups start collapsed except the one holding the selected record; a click toggles each group, in
the card list and the dense table alike. Single-run records stay flat, and filters apply to the
records before grouping (`web/src/lib/batch-groups.ts`).

Reconstruction titles are shown exactly as stored; run settings and batch repeat numbers are not
appended to them. Cards and table titles also show an approximate run-duration badge from the first recorded stage to record creation, excluding
queue time and the final persistence tail; unavailable timing is omitted.

## Read and mutation boundary

Artifact, compile-result, trace, input, and materialized-file requests are read-only. Resolution is
contained beneath allowlisted record/revision/materialization roots and never triggers compilation,
hydration, repair, promotion, or `usdcraft rematerialize`.

Canonical validity comes only from [`storage/artifact_contract.py`](../storage/artifact_contract.py).
Viewer presents its structured issues and the four storage classifications:

- `available`
- `contract_invalid`
- `missing`
- `compile_failed`

`contract_invalid` remains visibly invalid but can preview a saved URDF through the narrow read-only
allowlist. `missing` and `compile_failed` block ordinary compiled-model display. Previewability is
not Host success, artifact-package validity, or simulator readiness.

Both Workbenches expose the same two storage-owned mutations, each bound to its own pipeline:
rating update/clear (`PUT /api/records/{id}/rating`, `PUT /api/reconstruction-records/{id}/rating`)
and explicitly confirmed permanent deletion (`DELETE` on the same record routes). A Generation route
never touches a Reconstruction record or the reverse. Every other route is a read-only projection.
`usdcraft viewer --read-only` (or `USDCRAFT_VIEWER_READ_ONLY=1` for a directly started app) turns
the whole Viewer into a projection: the four mutation routes answer 403, `GET /api/status` reports
`read_only: true`, and the frontend hides the delete buttons and freezes the star ratings. This is
the mode for a deployed review copy; the default local run stays writable. Filters, sort order and
the current selection are frontend presentation state, not another record index.

### Lazy canonical validation

Listing or refreshing either record library (`GET /api/records` and `GET /api/bootstrap` for
Generation, `GET /api/reconstruction-records` for Reconstruction) never re-runs the full canonical
contract for every record: that check does hash and dependency-closure work that scales with the
number of records and files on disk. Each row instead reuses a process-lifetime cache of the last
real `storage/artifact_contract.py` result (`viewer/api/validation_cache.py`), keyed on the
record's active revision id and a sorted per-file stat signature of its materialization directory
(relative path, size, mtime_ns, ctime_ns). This detects ordinary renames, replacements and equal-size
edits even when another file has the latest mtime. It reads no file contents and is not a
cryptographic integrity guarantee. A scan error makes the cache unavailable rather than trusting
an incomplete signature. A row with no matching cached result falls back to the two cheap,
hash-free objective signals (`compile_failed`, a missing `model.urdf`) and otherwise reports
`materialization_status: "not_validated"` rather than guessing `available`; the summary also
carries `materialization_validated: bool`. Opening a record's detail view or updating its rating
obtains a full-check result, reusing the cache when its signature matches. Each store owns its own
cache instance; the products do not share or evict each other's entries.

`usdcraft reconstruction-validate <record-id>` and `usdcraft generation-validate <record-id>`
(or `--all`) run explicit Storage validation, bypassing Viewer enumeration and caches. `--all`
scans raw record directories, skips readable records of the other product and reports unreadable
or unknown product headers; missing Workbench registrations cannot hide records. An explicit
wrong-product id is an error. An empty selection stays empty. The data-format validation returns
its materialization classification so each selected materialization is checked once, rather than
running Viewer validation and then repeating the contract check. Exit 0 means the requested
checks passed; invalid records, including previewable `contract_invalid` results, exit 1.
Previewability remains separate and unchanged. These commands do not refresh a running Viewer's
in-process cache or mutate records. Record lists never trigger these checks automatically.

The Reconstruction list additionally keeps a process-lifetime cache of each finished row, keyed on
the parsed `record.json` content (rating and `updated_at` live there) and the same stat-only
materialization signature. Everything else a row reads — `reconstruction.json`, the input manifest,
provenance, cost, traces — sits in an immutable promoted revision. This assumes revision contents
are not edited in place and uses the metadata signature above; the four
materialization fields and `candidate_available` are re-derived on
every listing so a validation completed since (opening the record, an explicit validate) shows up
immediately. Neither list computes a hash: every input manifest already declares each file's
`sha256`, and the list only counts and classifies declared inputs. Declared input paths are still
resolved against the revision with `realpath` containment, memoized per declared directory rather
than repeated per file, and both lists read their independent per-record files on a small thread
pool. Rows, their order and their values are identical to a sequential, uncached listing.

## Canonical model Workbench

Generation Viewer and Recon Viewer share the same compiled `model.py` result and the same primary
inspection surfaces:

- **Model** — the request text, scalar joint controls, and the compiled model downloads (the
  PhysX `isaac/model.usdc` as the default button, the shared `model.usdz` beside it; both are
  served by the existing read-only file routes and offered only for an `available`
  materialization);
- **View** — presentation controls, source overlay (Reconstruction) and snapshots;
- **Source** — canonical `model.py` and materialized `model.urdf`;
- **Evidence** — the declared inputs: for Generation the one inspiration image (or the statement
  that the request was text-only), for Reconstruction the primary input groups below;
- **Runtime** — structured identity, **Run settings**, usage and read-only artifact information;
  it carries no input evidence. Run settings (`web/src/components/inspector/RunSettings.tsx`,
  rows from `web/src/lib/run-settings.ts`) is one compact section for both products: arm
  (Provider/Codex), wire protocol (Responses or Chat Completions API for Provider, Codex CLI, with
  the recorded runtime source), model, reasoning effort, image observation (external inventory,
  author-only, disabled, or no image), turns against the Provider turn budget (`n / max`; Codex:
  no turn limit) and batch row. Reconstruction adds guidance (none / provided, expandable; read
  from `reconstruction.json` `input.guidance`, "not recorded" for a record without the field),
  the request prompt (default, or custom with the text expandable), scale source with
  `meters_per_unit`, and whether mesh cleanup ran or was skipped. It reads only what the record already stores.

Revolute, continuous, prismatic, and mimic relationships are interactive. Spherical relationships
are statically inspectable but do not have interactive controls. D6 is not supported by the SDK.
Viewer gestures and overlays are inspection aids, not dynamics evidence.

## Recon Viewer additions

The Reconstruction record list carries the same star rating and delete controls as Generation and
a filter bar over source (Codex/Provider), model, scale provenance, batch, rating and AXIAL
recall band, with sorting by date, recall or rating. Each card prioritizes the asset title,
authoring model, scale provenance and last `check_axial` readings (recall and exterior
excess).

Recon Viewer adds two tabs:

- **Inputs** — the nonduplicated user-facing evidence groups: **Source Mesh** (the original
  declared source), **Semantic Input Image** (the one declared semantic channel) and **AXIAL**
  (the frozen AXIAL text). Source views the Author requested during the run are authoring
  feedback, not initial inputs. Subject-isolation contact sheets, component-ID renders, mesh-preprocess previews,
  manifests, and other Host preprocessing audits remain preserved in the canonical revision for
  provenance; they are not additional Author inputs and are not rendered as extra Inputs cards.
  Inputs are read only from the current `reconstruction_input_manifest` evidence manifest; the
  Viewer does not interpret record formats written by earlier versions.
- **Diagnostics** — the record-level `check_axial` summary (recall of the first and last call, cell
  counts, exterior excess, surface fit, control and opening placement counts), how the run's metres
  were decided (declared, embedded, format contract or the category estimate with its rationale),
  run timing (tool calls, Author elapsed time, stage timings; the run's settings, turn count and
  batch membership are in Runtime → Run settings rather than repeated here), the asset-specific
  compile warnings and the informational backend notes. The
  AXIAL summary is read from `runtime_report.axial_diagnostic`, falling back to the same reduction
  of the `provider_events.jsonl` trace; Codex records report no diagnostic.
  The tab also carries the **Compare** picker and the **Author timeline** described below.

**Compare** puts another Reconstruction record's candidate beside the selected one, both at q=0 in
their own viewport, each captioned with source arm, last recall, exterior excess and rating.
Records sharing the same batch row are listed first, so two runs of one CSV row are one click
apart. Closing the comparison returns to the ordinary viewport.

**Author timeline** (Provider records only) reads `provider_events.jsonl` through
`GET /api/reconstruction-records/{id}/author-timeline` and shows one row per Author turn: the
tools called, failure marks, the candidate revision, model seconds and the `check_axial` recall
of that turn, with totals for calls, patches, compiles, failed calls, model versus tool time and
the recall series. Codex records have no per-turn tool trace and say so.

**Source overlay** is a Reconstruction-only, default-off option inside View. When enabled, it
temporarily displays the post-isolation normalized source and the compiled candidate in their shared
q=0 frame with simple visibility/opacity controls. A third switch, **check_axial readings**, draws
the last `check_axial` call's geometry from
`GET /api/reconstruction-records/{id}/axial-geometry` in the same frame: exterior-excess stack
slices as amber wire boxes. The overlay performs no alignment, fitting, scoring,
compilation, or repair. Leaving the overlay returns to the ordinary compiled candidate viewport.

The Reconstruction Diagnostics card exposes the last AXIAL measurement's source/candidate binding:
`current`, `stale`, or `unverified` (no hashes recorded). This is a provenance reading; it does not
change canonical validity or trigger a rebuild. Bindings hash authored USD layer content and
dependency asset bytes, so crate serialization alone does not mark a recompile stale. A differing
binding does not prove different geometry.

## Source map

- [`api/app.py`](api/app.py) — application wiring and route registration.
- [`api/store.py`](api/store.py) — canonical Generation projection.
- [`api/reconstruction_records.py`](api/reconstruction_records.py) — canonical Reconstruction
  projection: Inputs mapping, run-setting/scale/batch fields, the AXIAL
  diagnostic summary (`usdcraft/reconstruction/axial_summary.py` is the one reduction shared with
  the Host), the Author timeline and the `check_axial` geometry reductions of
  `provider_events.jsonl`.
- [`api/run_settings.py`](api/run_settings.py) — read-only run-setting derivations shared by both
  stores (Provider wire protocol, image observation, turn budget, mesh cleanup, default request,
  recorded guidance).
- [`api/host_guard.py`](api/host_guard.py) — the loopback/explicit `Host` allowlist.
- [`api/file_resolver.py`](api/file_resolver.py) — contained canonical file/materialization access.
- [`api/routes/`](api/routes/) — HTTP boundary (endpoint list below).
- [`web/src/components/model-viewer/`](web/src/components/model-viewer/) — shared compiled-model
  Workbench.
- [`web/src/components/browser/`](web/src/components/browser/) — Generation record list, filters,
  the shared `StarRating` control and the shared `BatchGroupHeader` / expansion hook.
- [`web/src/components/reconstruction/`](web/src/components/reconstruction/) — canonical Recon
  record list, filters, Inputs, Diagnostics and overlay.
- [`web/src/lib/reconstruction-filters.ts`](web/src/lib/reconstruction-filters.ts),
  [`web/src/lib/reconstruction-diagnostics.ts`](web/src/lib/reconstruction-diagnostics.ts),
  [`web/src/lib/author-timeline.ts`](web/src/lib/author-timeline.ts) and
  [`web/src/lib/axial-geometry.ts`](web/src/lib/axial-geometry.ts) — pure filter/sort,
  diagnostics, timeline grouping and face-UV→world geometry logic (unit-tested).
- [`web/src/lib/app-mode.ts`](web/src/lib/app-mode.ts) — the two-mode URL contract.

## API endpoints

All routes are `GET` except the two rating (`PUT`) and two deletion (`DELETE`) mutations.

- Status: `/health`, `/api/status`, `/api/bootstrap`, `/api/collections/workbench`.
- Generation records: `/api/records`, `/api/records/{id}`, `/api/records/{id}/summary`,
  `/api/records/{id}/history`, `/api/records/{id}/input-image`, `PUT /api/records/{id}/rating`,
  `DELETE /api/records/{id}`.
- Generation files: `/api/records/{id}/text/{path}`, `/api/records/{id}/files/{path}`,
  `/api/records/{id}/traces/{path}` and the same three under
  `/api/records/{id}/revisions/{revision_id}/`.
- Reconstruction records: `/api/reconstruction-records`, `/api/reconstruction-records/{id}`,
  `/api/reconstruction-records/{id}/author-timeline`,
  `/api/reconstruction-records/{id}/axial-geometry`,
  `/api/reconstruction-records/{id}/text/{path}`, `/api/reconstruction-records/{id}/files/{path}`,
  `/api/reconstruction-records/{id}/materialized/{path}`,
  `PUT /api/reconstruction-records/{id}/rating`, `DELETE /api/reconstruction-records/{id}`.

## Tests

Backend coverage:

```bash
uv run pytest -q tests/viewer
```

Frontend coverage and static validation:

```bash
npm --prefix viewer/web run test
npm --prefix viewer/web run lint
npm --prefix viewer/web run typecheck
npm --prefix viewer/web run build
```
