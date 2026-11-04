# Reconstruction

This directory owns the USDCraft Reconstruction pipeline: one user-supplied source asset plus one
semantic image become one canonical, editable, simulator-ready record. Two execution arms share one
core — the same evidence doctrine, Author task, chain, record shape and harness, one Author per
request:

- the **Provider arm** — `usdcraft reconstruction` and `reconstruction-batch`, direct; and
- the **Codex arm** — `usdcraft reconstruction-codex` (preflight/confirmation by default, or
  `--direct`) and `reconstruction-codex-batch` (direct CSV execution).

The chain is freeze → mesh preprocess → optional image observation → one Author → mechanical Host
gate → persist. The Author interprets the semantic image for identity, shape character and
function and uses reliable source measurements for dimensions; there is no independent visual
critic. Reconstruction is not an alias for Generation. What each evidence channel and AXIAL section
may establish is specified in [EVIDENCE.md](EVIDENCE.md).

## Shared harness and capability switches

All models and both arms use the same Author task: the Author chooses construction,
measurement and repair order, with no fixed construction branch or AXIAL repair-count limit. Every
run has direct mesh measurement (`measure_mesh`), on-demand source rendering (`render_source`),
candidate rendering (`render_candidate`) and local part comparison (`compare_part`). The remaining
switches are:

| Switch | Default | Scope |
| --- | --- | --- |
| `--[no-]image-observation` | on | Independent advisory image inventory; the semantic image is always attached |
| `--max-turns N` | 80 | Provider Author turn budget, integer 1–200; Codex does not enforce it |

The policy is recorded in the request, workspace and `runtime.harness`, and bound into the Codex
fingerprint; it governs the observation stage and the Provider turn budget, not an OS sandbox.
With `--no-image-observation` the Author still receives the semantic image and the report records `image_observation=disabled`.

```bash
uv run usdcraft reconstruction /abs/scan.glb --image /abs/photo.jpg --model <model> --reasoning <effort>
uv run usdcraft reconstruction /abs/scan.glb --image /abs/photo.jpg --no-image-observation --max-turns 40
uv run usdcraft reconstruction-codex /abs/scan.glb --image /abs/photo.jpg --model <model> --reasoning <effort> --direct
```

The initial Author message attaches only the semantic image. When a shape or occlusion question
needs another camera, the Author requests a source view (Provider `render_source(azimuth,
elevation)`; Codex `recon-render-source`). Views render the frozen geometry-only source carrier in
normalized metre/Z-up coordinates, need no compile, and report renderer capabilities and fallback;
a render failure is tool feedback. Preprocessing generates no fixed source images. The carrier
preserves available appearance for on-demand rendering and Viewer overlays (components share
immutable texture materials; each owns its UV array). Source views describe the target; candidate
views describe output; both are archived with the Author trace.

Provider function tools and Codex shell commands are transport adapters over one task and the
shared `author_tools.py` operations. `lifecycle.py` owns the mechanical compile, diagnostic
reduction, trace archival, promotion and cleanup for both arms; preprocessing and observation are
shared in `command.py`. Transport differences (NVIDIA credentials and API attempts; Codex CLI
runtime, isolation and authorization) never select different tasks or acceptance rules. Both arms
preload the same SDK quickstart (`usdcraft/pipeline/sdk_context.py`); other SDK references stay on
demand. SDK references define authoring APIs; evidence authority and tools are owned by this task.

### Scale

Both arms settle suspected unit-normalized scale before preprocessing: a source whose units are
assumed (no unit metadata) and whose longest side is within 2.5% of 1.0 or 2.0 receives one
clean-context image-based estimate of the subject's real longest dimension (NVIDIA call on
Provider, Codex vision call on Codex; one retry). The quotient becomes the run's `meters_per_unit`,
tagged `estimated_from_semantic_image` in `request.json`, the source manifest and
`runtime_report.scale`, and the Author task carries a SCALE paragraph. An explicit
`--meters-per-unit` skips it; failure keeps metres with a warning. On the Codex arm the estimate runs
after confirmation and before the freeze as a Host derivation, not a new approval. See
[`scale_estimate.py`](scale_estimate.py).

## Provider arm

The Provider arm executes immediately; there is no preflight, fingerprint or confirmation turn. It
accepts the source, `--image`, `--prompt`, `--guidance`, `--label`, `--up-axis`, `--front-axis`,
`--meters-per-unit`, `--[no-]image-observation`, `--mesh-cleanup`, `--max-turns` (Author turn budget,
default 80), `--model`, `--reasoning`, `--timeout` (deadline for each Provider HTTP request, default
900 s),
`--api-style responses|chat-completions` (default `responses`), `--keep-workspace` and `--json`.
Time and turn limits are Provider-only; the Codex arm has none. Use `--reasoning provider-default` to omit reasoning fields. The API style applies to
the Author, image observation, scale estimation and subject isolation, and is recorded in the
runtime report. `--mesh-cleanup` (default off) starts the same optional Subject-Isolation Agent as
the Codex arm through the Provider transport; without it the Host freezes the deterministic retain-all
declaration. See [Mesh cleanup](#mesh-cleanup-subject-isolation-agent).

A Provider record carries `agent_harness: provider`, its endpoint as `provider` (`openai`, or
`nvidia` when only `NVIDIA_API_KEY` is configured), `runtime.harness` and `provider_*` traces.
Its Author workspace is `data/local/reconstruction_author_runs/<run_id>/`; frozen inputs and
derived evidence stay under `data/local/reconstruction_runs/<run_id>/`. Entry point:
[`provider_command.py`](provider_command.py).

### Provider CSV batch

`usdcraft reconstruction-batch <csv>` runs each row through the same primitive on the shared
coordinator (`usdcraft/pipeline/batch.py`); the row contract and executor are in
[`provider_batch.py`](provider_batch.py). Batch-wide options are `--model`, `--reasoning`,
`--timeout`, `--api-style`, `--concurrency` (1–32, default 1), `--keep-workspace`, `--json`,
`--validate-only` and `--run-timeout` (default 1920 s, `0` = no limit), with the same semantics as
`generation-batch`. The CSV columns
are listed under [Reconstruction CSV](#reconstruction-csv).

```bash
uv run usdcraft reconstruction-batch rows.csv --validate-only
uv run usdcraft reconstruction-batch rows.csv --concurrency 4 --json > results.json
```

## Reconstruction CSV

Both batch commands read the same schema. Unknown columns and surplus cells are rejected so a typo
cannot silently fall back to defaults; the error names the unknown columns and lists the allowed
ones. Relative `source_path`/`image_path` values resolve from the
CSV directory. Empty optional cells use the defaults.

| Column | Required | Meaning |
| --- | --- | --- |
| `row_id` | yes | Unique row identifier |
| `source_path` | yes | Source mesh, USD scene or point cloud |
| `image_path` | yes | The semantic image |
| `prompt` | no | Request text (`--prompt`) |
| `guidance` | no | User guidance (`--guidance`) |
| `label` | no | Display name (`--label`) |
| `mesh_cleanup` | no | `true`/`false`/`1`/`0` (default false): run the Subject-Isolation Agent (`--mesh-cleanup`); both arms |
| `image_observation` | no | `true`/`false`/`1`/`0` (default true) |
| `max_turns` | no | Provider turn budget, integer 1–200; Provider only (Codex batch rejects a nonempty cell) |
| `model`, `reasoning` | no | Per-row runtime override; both cells together or neither |
| `api_style` | no | `responses` or `chat-completions`; Provider only (Codex batch rejects a nonempty cell) |
| `up_axis`, `front_axis`, `meters_per_unit` | no | Frame declarations, as on the single-item command |

A nonempty `model`/`reasoning` pair overrides the CLI defaults for that row; blank pairs use the CLI
defaults or the arm's normal runtime resolution. All effective pairs are resolved before rows
start; mixed-runtime batches display `mixed` in the progress header. `--validate-only` checks the
CSV contract without model calls, credentials or workspace writes; it does not decode sources or
images or test runtime availability. Batch progress identifies `Provider Reconstruction Batch` or
`Codex Reconstruction Batch` in terminal and redirected output.

**Run deadline (Provider only).** `reconstruction-batch --run-timeout SECONDS` (POSIX; default
1920) limits each executing row's total wall-clock time from process startup through persistence,
excluding queue time; it is separate from `--timeout`. `--run-timeout 0` removes the limit, and the
rows then run in the calling process without a supervisor. At expiry the supervisor (the shared
[`../pipeline/deadline.py`](../pipeline/deadline.py), also used by `generation-batch`) stops the
row's process group including descendants, marks it failed with `status=timeout`, retains its
workspaces and writes `data/local/reconstruction_deadlines/<run_id>/snapshot.json` (last progress,
elapsed time, request and workspace paths). Completion counts only if its timestamp is within the
deadline; a timeout never counts as success even if publication raced the stop, transaction state
is left to Storage recovery, and existing records are never rewritten. Timed-out rows are not
retried. `reconstruction-codex-batch` has no run deadline, but its rows run under the same
supervisor without a clock so Ctrl-C stops each row's Codex processes; interrupted JSON lists
finalized, cancelled and in-flight rows.

## Codex arm

The Codex arm runs the same Reconstruction through the Codex CLI. It accepts the source, `--image`,
`--prompt`, `--guidance`, `--label`, `--up-axis`, `--front-axis`, `--meters-per-unit`,
`--[no-]image-observation`, `--mesh-cleanup`, `--model`/`--reasoning` (paired; when both are
omitted inside a Codex task, the root task's model and reasoning effort are inherited from Codex's
local thread state, and outside Codex the command fails instead of guessing), `--keep-workspace`,
`--json` and either `--direct` or `--confirm-reconstruction <fingerprint>`. The Codex arm has no time
or turn limit: there is no `--timeout`, `--run-timeout` or `--max-turns`, and the request, the
fingerprint and the record carry no timeout or turn budget. `--mesh-cleanup` starts the
optional Subject-Isolation Agent exactly as on the Provider arm (bound into the fingerprint).

Codex execution needs the `codex` executable on `PATH` or `USDCRAFT_CODEX_BIN` set to its path;
batch checks this before starting rows (it does not test login or model access).

### Direct execution and CSV batch

`--direct` validates and executes immediately, without a preflight summary or fingerprint. It is
mutually exclusive with `--confirm-reconstruction`. Input decoding, hash binding, scale, freeze,
Author, mechanical gate and persistence reuse the confirmed chain.

```bash
uv run usdcraft reconstruction-codex /abs/scan.glb --image /abs/photo.jpg \
  --model <model> --reasoning <effort> --no-image-observation --direct
```

`reconstruction-codex-batch` executes the same direct chain once per row of the
[Reconstruction CSV](#reconstruction-csv), each in a separate worker process on POSIX hosts. A
row's `mesh_cleanup` cell selects the optional Subject-Isolation Agent as on `reconstruction-batch`
(default off). Model/reasoning can
come from a paired CSV override, paired CLI defaults or the calling Codex task; a CSV with pairs on
every row needs neither. Each row has its own Author conversation. The Provider-only `max_turns` and
`api_style` cells are rejected (also by `--validate-only`). Results identify `provider=codex-cli`;
canonical runtime metadata retains direct authorization, batch and row IDs and the effective
harness policy (`image_observation` only; no `max_turns`).

```csv
row_id,source_path,image_path,image_observation,label
kettle_01,/abs/scan.glb,/abs/photo.jpg,false,kettle
```

```bash
uv run usdcraft reconstruction-codex-batch rows.csv --validate-only
uv run usdcraft reconstruction-codex-batch rows.csv \
  --model <model> --reasoning <effort> --concurrency 4 --json > results.json
```

### Preflight and confirmation

Without `--direct`, the first invocation is a zero-write preflight:

```bash
uv run usdcraft reconstruction-codex /abs/scan.glb --image /abs/photo.jpg --front-axis=-Y
```

It resolves the source dependency closure, decodes the semantic image, validates the scaffold,
frame and Codex runtime, and prints one compact summary: source, semantic image, request text,
optional label/guidance, frame, mesh-cleanup state, capability switches (`harness`), model,
reasoning effort and fingerprint. The fingerprint also binds the Author evidence contract: one
attached image (the semantic image), AXIAL, no initial source views and on-demand source views. It creates no workspace, starts no Codex process and creates
no record. After review, the same request runs with the exact token:

```bash
uv run usdcraft reconstruction-codex /abs/scan.glb --image /abs/photo.jpg --front-axis=-Y \
  --confirm-reconstruction <approval_fingerprint>
```

Every fingerprinted field — source dependencies, image, frame, request text, guidance, label,
mesh-cleanup choice, capability switches and model/reasoning pair — must be unchanged; any change
needs a new preflight. The fingerprint is a stateless human-intent confirmation token, not a source
seal or security boundary.

### Codex tool commands

The Codex Author works in `data/local/reconstruction_author_runs/<run_id>/` with the semantic image
attached and `inputs/axial_brief.txt` staged beside the full AXIAL text, using shell equivalents of
the Provider tools:

| Provider tool | Codex command |
| --- | --- |
| `check_model` | `python -m usdcraft.cli check <model.py>` |
| `compile_model` | `python -m usdcraft.cli compile <model.py> --output <build> --no-render` |
| `check_axial` | `python -m usdcraft.cli recon-check-axial <axial> <candidate.usdc> --source <normalized source> --payload-dir <workspace>/axial_checks` |
| `measure_face` | `python -m usdcraft.cli recon-measure-face <axial> --face=<face> [--z-band LO HI]` |
| `measure_mesh` | `usdcraft recon-measure-mesh inputs/source_mesh.usdc --output author_feedback/mesh_001 [--bounds xmin ymin zmin xmax ymax zmax]` |
| `render_source` | `usdcraft recon-render-source inputs/source_render --output author_feedback/source_001 --azimuth 45 --elevation 30` |
| `render_candidate` | `usdcraft recon-render build --output author_feedback/view_001 --azimuth 45 --elevation 30 [--joint NAME=VALUE]` |
| `compare_part` | `usdcraft recon-compare-part inputs/source_mesh.usdc build --part NAME --face=-Y --bounds xmin ymin zmin xmax ymax zmax --output author_feedback/part_001` |
| `finish_reconstruction` | the schema-conforming `complete`/`blocked` final response |

`recon-check-axial --source` prints the same mesh-to-mesh audit as the Provider tool; each
`--payload-dir` payload (and each call's identity and closure hashes) is reduced by the Host to
`runtime_report.axial_diagnostic`. The Author may also inspect `inputs/source_mesh.usdc` with
Python/OpenUSD. `recon-render` sends a bounded file request to a run-scoped Host renderer (the
existing Chromium renderer), active only during the Author session and executing no commands.

Codex has no turn budget and no run clock; the Author, the Subject-Isolation Agent, the scale
estimate, image observation and its review run without a timeout (one retry for the initial
observation). Observation attempt logs (`.codex-run/stdout.jsonl`, `stderr.log`) end in
`upstream/image_observation/` of the run workspace.

A Codex record carries `agent_harness: codex`, `provider: codex-cli`, `codex_*` traces,
`runtime.harness` (`image_observation` only, without `max_turns`), `runtime_report.scale`,
`axial_diagnostic` and `preloaded_docs`. Every single-item command of both products prints the same
result block (`status`, `pipeline`, `provider`, `run_id`, `record_id`, `record_dir`, `model`,
optional `workspace_dir` and `warnings`, `viewer`) or one JSON object with the same keys under
`--json`.

## Request text and record title

Every request carries one request text — what to reconstruct. `--prompt` supplies it; otherwise a
deterministic default request naming the primary subject is used ("Reconstruct the … shown in the
semantic image and the source mesh as a complete, functionally usable, simulator-ready 3D asset.").
The text enters the Author task as the REQUEST block with the highest authority for the subject's
category and every function, count and must-have it states; the semantic image and the brief keep
their authority over visible identity and measured geometry. The record title is `--label` or the
request text; the text is archived as `request.txt` and shown as the prompt preview. `--guidance` is
the separate user-supplied supplemental constraint (CSV column `guidance` in batch). Both arms record
it in the canonical record as `reconstruction.json` `input.guidance` (`null` without guidance), the
field the Viewer reads; the archived Author task is never parsed for it.

## Product boundary

One request reconstructs one primary subject from one source mesh, USD scene or point cloud; one
semantic image (role `reconstruction_target`); frame metadata or declarations (or the scale
estimate); and optional guidance, label and mesh cleanup. There is no image-only
Reconstruction. The Host derives geometry evidence; it does not reinterpret the request, add a
case-specific correction, choose a best result or act as a semantic reviewer. Generation images
become Reconstruction evidence only when passed as `--image`.

## Accepted source and frame contract

`source.py` accepts USD (`.usd`, `.usda`, `.usdc`, `.usdz`), glTF (`.glb`, `.gltf`), OBJ, STL, PLY,
OFF and XYZ. Referenced USD layers and glTF assets are part of the source closure. A USDZ package is
one self-contained file: the closure records the package after verifying each referenced member
exists inside it.

The freeze atomically copies the closure, semantic image and initial scaffold under the run's
`approved_inputs/` before any Agent starts, preserving relative topology. The Host reopens the
copied entry and requires the resolved closure, hashes and sizes to match and remain under the
copied tree; escaping dependencies fail closed. Every later read uses the frozen paths.

Normalization expresses the source in metre-scale, Z-up `recon_world` without using geometry to
guess units, up, front, orientation or occupancy. glTF uses +Y up and one metre per unit unless
overridden; USD may use authored `upAxis` and `metersPerUnit`; formats without trustworthy
metadata require `--up-axis` and `--meters-per-unit`; suspected unit-normalized sources take the
scale estimate. Front is optional: if unresolved, deterministic yaw is preserved, the evidence
records it as unresolved, and the Author resolves orientation from the image and source geometry.
Subject isolation happens before any rotation, recenter, ground snap, autoscale or AXIAL
construction. Source topology is surface-only evidence.

## Pipeline stages

| Stage | Executor | What it does |
| --- | --- | --- |
| 1. Command | User + Host | Provider: direct. Codex: preflight/confirmation, `--direct`, or CSV batch |
| 2. Scale | Host + one vision call | Unit-normalized source → `meters_per_unit` from the semantic image; skipped for a declared scale |
| 3. Freeze | Host | Atomic snapshot of source closure, semantic image and scaffold |
| 4. Subject isolation | Optional Agent (`--mesh-cleanup`, both arms) | Retained-component decision, or deterministic retain-all |
| 5. Mesh preprocess | Host | Normalized source, `AXIAL`, voxel sidecar, geometry-only source carrier (the brief is derived at workspace staging) |
| 6. Image observation | Observation Agent (fresh context) | Schema-validated inventory from the semantic image alone; skipped when disabled |
| 7. Authoring | Author | Mesh inspection, editable model, check/compile, AXIAL diagnostics, candidate inspection and repair |
| 8. Host gate | Host | Independent clean no-render compile and mechanical validation |
| 9. Persist | Host | Atomic promotion of the one valid model to a canonical record |

Timed stages (`usdcraft/stagetime.py`) are `freeze`, `subject_isolation`, `mesh_preprocess`,
`image_observation`, `author`, `host_gate` and `persist`. Each appends events to
`stage_trace.jsonl` in the run workspace (archived per record under
`revisions/<rev>/traces/stage_trace.jsonl`), and the table is persisted as `stage_timings` in the
runtime report. Timing is descriptive, never a gate; preflight and confirmation are not timed.

## Runtime roles

### Subject-Isolation Agent

Optional mesh-artifact cleanup, run only with `--mesh-cleanup` (or a true `mesh_cleanup` CSV cell)
on either arm; default off: it may exclude whole source components as clear background, incidental
or outlier evidence, and nothing else. Ambiguous components are retained; its artifacts are Host
audit data never passed to the Author. See [EVIDENCE.md](EVIDENCE.md#subject-isolation).

#### Mesh cleanup (Subject-Isolation Agent)

Both arms share `isolation_agent.py`: the component inventory, the two isolation contact sheets,
the task text, the decision schema, the decision validation and the published
`subject_isolation.json` manifest and audit tree. Only the transport differs:

| | Codex arm | Provider arm (`provider_isolation.py`) |
| --- | --- | --- |
| Call | one ephemeral read-only `codex exec` with an output schema | one Provider request with one strict `submit_subject_isolation` tool; manifest `transport` names the endpoint (`nvidia_provider` or `openai_provider`) |
| Evidence | `TASK.md` on stdin; the Agent reads `component_inventory.json` from its workspace; two `--image` sheets | the same `TASK.md` text, then the same inventory bytes, then the same two sheets in the same order |
| Task wording | "clean-context Codex Subject-Isolation Agent" | "clean-context Subject-Isolation Agent" (transport name only) |
| Time limit | none | the Provider per-request `--timeout` |
| Audit logs | `agent/.codex-run/{final.json,stdout.jsonl,stderr.log}` | `agent/.provider-run/{final.json,response.json}` |

Neither transport retries: a failed call, a missing response, an invalid decision (unknown or
repeated selection, invalid disposition, a blocked result that excludes, exclusion of the whole
source) fails the run, and a `blocked` outcome fails preprocessing. The manifest `agent` block
records the model, reasoning effort and transport facts (`arm: provider`, `api_style`,
`request_timeout_seconds` on the Provider; Codex feature/sandbox facts on Codex). Both arms record
the request's `mesh_cleanup.requested` state in `request.json` and the record.

### Image-Observation Agent

One fresh clean-context vision call per run (`observation.py`, an adapter over the shared engine
`usdcraft/pipeline/observation.py` that Generation reuses). Its context is exactly the semantic
image, its task and the response schema. It returns one concise, schema-validated inventory — part
names, visible counts, mobility, granularity, categorical geometry cues (never numbers),
arrangement, affordances, occlusion, duplicate/mirror references, `carried_as` (`none`,
`displayed` or `enclosed`; surface graphics are never entries), `attaches_to` under a
reference-ordering contract (each entry attaches to an earlier one), and bounded
`visibility: inferred` completions — with grouped repeats and no target part count.

A second clean-context reviewer checks it against the same image under a deterministic merge:
additions and sharpening are accepted, counts may only grow, renames are rejected, dropping an entry
by omission is rejected (the original is restored and that round's additions forfeited), and
removal needs an explicit `removals` entry with a reason, honored only when nothing surviving
references it. A failed review keeps the first inventory. If observation fails, blocks or is invalid
twice, the run degrades to the Author-only reading with a warning (`image_observation=author_only`).
The inventory is advisory, never a gate or part of the fingerprint, and is archived as
`traces/image_observation.json` and `traces/image_observation_manifest.json`.

### Reconstruction Author and Host

One Author per run receives shared rules, the preloaded quickstart and SDK inventory, then the
request, guidance, scale, evidence paths, optional observation, the semantic image, compact AXIAL
brief, evidence manifest and scaffold (Provider places cache boundaries before the run inputs and
before the image). The raw source closure, isolation data, voxel sidecars, prior attempts,
evaluator state and ground truth stay outside its context; these are context protocols, not OS
isolation. `model.py` is the editable source; candidate USD/URDF are rebuilt outputs, and candidate
images are feedback about the output, never target evidence.

The Host owns validation, scale estimation, preprocessing, staging, process execution, clean
compilation, materialization, persistence, cleanup and recovery. Its final gate is narrow and
mechanical; semantic interpretation, Part selection and functional completeness belong to the Author.

## Authoring loop

The Author plans a compact model, obtains useful measurements, builds an early executable
candidate, inspects discrepancies and views its candidate, then repairs supported defects. AXIAL
findings are diagnostic questions; recall and automatic matching are not completion thresholds, and
zero numbered discrepancies is not a quality verdict. Consequential local residuals are repaired or
explained in comments and the closing summary. The final inspection on the latest compiled geometry
checks openings, thin surfaces, visible overlap and that a control's assembly moves together.

The Author reviews functional mobility before construction and after repair (fixed structure,
carried attachments, operating joints, detachable parts, unresolved relations, setup adjustments);
family prompts suggest what to inspect, not mandatory parts or joints. An explicit request for
rolling support can require a plausible mechanism the scan omits; a generic category does not. An
image-only inventory or exterior envelope does not prove an empty interior. Repair preserves
function: FIXED/FLOATING substitution, deleting a member or collapsing travel is not a collision
fix. Interacting joints are checked together in one pose mapping (nested `ctx.pose` calls replace
rather than accumulate). These are Author instructions, not Host gates or runtime guarantees.

### Author tools

Provider edits `model.py` with `write_model(content)` (first construction or full replacement) and
`apply_patch` (local edits with exact context); both write atomically, report syntax issues and
advance the revision only on change. A failed patch leaves source and revision unchanged.

- **`measure_mesh`**: selected-vertex bounds, quantiles, a vertex-weighted plane fit, three
  orthographic projections, up to 4096 XYZ samples and anonymous component rows (16 of each
  previewed, the rest in `detail_path`, `components_truncated` explicit); at most two million
  source vertices. Bounds select vertices, not solids or parts; check fit residual and eigenvalues
  before trusting a normal. CLI `--bounds` takes six separate numbers
  (`--bounds -0.10 -0.20 0.00 0.10 0.20 0.30`).
- **`measure_face`**: AXIAL-derived per-face queries; not remeasurement.
- **`render_candidate`**: compiled URDF visual geometry; degrees for cameras, radians/metres for
  joints; kinematic posing only. Render failure is a tool limitation, not a gate.
- **`compare_part`**: one compiled q=0 link (optionally one element) against a selected source
  region and face — coverage, first-surface residuals and an overlay; the Author owns
  correspondence. AXIAL local patches include a six-number source ROI ready for these queries.

Each CLI query uses its own directory under `author_feedback/`, retained under canonical traces
with a path/hash index; other Codex outputs need `--keep-workspace`. Provider conversations receive
compact AXIAL text; full results stay in traces and feedback files.

### Candidate checkpoints and revision coverage

Provider Authors use `validate_model` as the default checkpoint (check, compile, AXIAL and q=0
render in one call). A model edit invalidates the compiled revision; `compare_part`, `check_axial`
and `render_candidate` require a successful `compile_model` or `validate_model` for the current
revision. `ok` reports check/compile success; `diagnostics_complete` separately reports diagnostic
execution. Replies include `validation_state`: the model revision and the revision/status
(`current`, `stale`, `failed`, `not_run`) of the latest check, compile, AXIAL and render, preserved
in the finish summary and `runtime_report.author_validation`. It is execution coverage, not
semantic acceptance, and adds no gate. Codex's closing summary identifies final checks and any
earlier-revision views. The final AXIAL measurement is labelled `current`, `stale` or `unverified`
against the final authored USD (shown on the Viewer Diagnostics card; binding only, never quality).

## Failure and retention

A run promotes one record or none. A blocked Author, failed Host gate or failed promotion retains
the frozen inputs, derived evidence and Author workspace and reports `run_id` and `workspace_dir`;
a successful run removes both workspaces unless `--keep-workspace`. There is no hidden evaluator or
Host semantic reviewer.

## Records and Viewer

`persist.py` stores each run's single result through the shared canonical promotion transaction.
A record includes the request and runtime, `model.py`, Author/Host traces, materialized artifacts,
the exact Author context (including the normalized source mesh, always an Author input), source
dependency closure, the semantic image and its fixed role `reconstruction_target`, AXIAL, the
geometry-only source-overlay manifest (`source_overlay_manifest`, audit channel `source_overlay`),
Author-requested feedback, preprocessing manifests and isolation/mesh-preprocess/source-overlay
audit closures. A record has no sample index and no Author notes file. Retaining Host audit data does not make it Author evidence. The approved-input
manifest is cross-checked against canonical source topology, image, scaffold, hashes and sizes.

The Viewer reads the canonical manifest rather than crawling `inputs/`; see
[viewer/README.md](../../viewer/README.md). Source/candidate overlay is an optional Render mode;
nothing in the Viewer is a promotion score. Storage schemas belong to
[storage/README.md](../../storage/README.md).

## Code ownership map

| Concern | Implementation |
| --- | --- |
| Provider entry, tools and batch rows | `provider_command.py`, `provider_tools.py`, `provider_batch.py` |
| Codex preflight/direct orchestration and shared preprocessing; Codex batch | `command.py`, `codex_batch.py` |
| Batch run deadline supervisor | `provider_deadline.py` (Reconstruction worker over the shared `../pipeline/deadline.py`) |
| Request validation, fingerprint payload, capability switches | `contract.py`, `harness_policy.py` |
| Scale estimate | `scale_estimate.py` |
| Author task and default request | `prompt.py`, `default_prompt.py` |
| Workspace staging; Codex invocation | `workspace.py`, `author.py` |
| Mechanical completion, runtime metadata, promotion, cleanup | `lifecycle.py` |
| Source formats, closure, frame decode | `source.py` |
| Subject isolation | `isolation_agent.py`, `isolation.py`, `isolation_render.py` |
| Mesh preprocess; source carrier and render session | `mesh_preprocess.py`, `source_overlay.py`, `orientation.py` |
| Image observation | `observation.py` |
| AXIAL and derived sections | `axial.py`, `face_features.py`, `face_openings.py`, `_face_grid.py`, `interior_islands.py`, `axial_brief.py` |
| Shared measurement, rendering, part comparison | `author_tools.py`, `part_comparison.py`, `render_service.py` |
| `check_axial` audit, re-encode and summary | `provider_axial_audit.py`, `local_surface_audit.py`, `diagnostics.py`, `axial_summary.py` |
| Codex diagnostic commands and binding | `diagnostic_cli.py`, `face_measure.py`, `diagnostic_binding.py` |
| Record assembly | `persist.py` |
| CLI arguments | `usdcraft/cli.py` |

## Tests

Run `uv run pytest -q tests/reconstruction`. `test_reconstruction_provider.py` covers the Provider
arm, `test_reconstruction_codex_author.py` the Codex arm, `test_reconstruction_batch.py` and
`test_codex_batch.py` the CSV contract, and `test_harness_policy.py` the capability switches; the
evidence tests are listed in [EVIDENCE.md](EVIDENCE.md#verification-map).
