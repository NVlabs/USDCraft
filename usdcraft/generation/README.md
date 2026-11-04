# Generation

This directory owns the Generation product flow. It normalizes one Generation request, stages one
author workspace, builds the shared Author task, runs either the Codex or Provider arm, calls
the shared mechanical Host gate, and promotes a valid result through Storage.

This page documents the component boundary and routes readers to the implementation that enforces
it. It is not an SDK API reference, USD/backend manual, or Storage schema specification.

## Sources of truth

Use the narrowest source that owns the behavior:

- [`usdcraft/pipeline/policy.py`](../pipeline/policy.py) is the physical-authoring policy shared
  with Reconstruction (the flexible-member rule, explicit mass and materials, contact presets, joint
  limits and resistance, gravity hold, drives, the radians contract, backend capability facts): one
  text, rendered into the Generation DELIVERABLE list and the Reconstruction SIMULATOR-READY
  MECHANICAL AUTHORING section with the product named where a policy is attributed. Product
  semantics never live there.
- [`prompt.py`](prompt.py) is the runtime Generation Author contract. Both execution arms call
  `build_generation_prompt(...)`; Authoring policy must live there rather than in root repository
  instructions.
- [`contract.py`](contract.py), [`default_prompt.py`](default_prompt.py), and
  [`command.py`](command.py) own Codex request normalization, preflight binding, approval, and
  execution orchestration.
- [`../pipeline/provider/README.md`](../pipeline/provider/README.md) and the files under
  [`../pipeline/provider/`](../pipeline/provider/) own the Provider transport and bounded harness
  adapter. [`provider_command.py`](provider_command.py) owns its direct per-item entry point, and
  [`provider_batch.py`](provider_batch.py) owns the Batch CSV row contract.
- [`../compile/README.md`](../compile/README.md) and [`../compile/command.py`](../compile/command.py)
  own authoring-only `check`, no-render compilation, and the final mechanical export gate.
- [`../../storage/README.md`](../../storage/README.md),
  [`../../storage/artifact_contract.py`](../../storage/artifact_contract.py), and
  [`persist.py`](persist.py) own canonical artifact validation and promotion.
- [`../../sdk/_docs/common/00_quickstart.md`](../../sdk/_docs/common/00_quickstart.md) is the SDK
  documentation entry point. Public APIs, authoring types, tests, and backend mappings belong in the
  relevant SDK pages, not here.
- [`../../README.md`](../../README.md#where-results-are-stored) owns the stored-data and
  trusted-code summary, [`../../viewer/README.md`](../../viewer/README.md) owns Viewer access, and
  [`../../.env.example`](../../.env.example) owns Provider credentials.

When prose disagrees with executable behavior, fix the owning code, focused tests, and this component
page together. Do not copy a policy into several repository-wide instruction files.

## Product inputs

Generation accepts exactly these user-facing modes:

1. explicit text;
2. explicit text plus at most one image; or
3. one image with no explicit text.

Internally, image-only input is normalized to a non-empty standard Generation prompt before a
request object is created. That standard prompt names the primary subject shown in the image and
asks the Author to reconstruct it as a complete, functionally usable, simulator-ready asset; the
Author identifies the subject. The resulting internal input kind is `text_image`, not a third
pipeline kind.

The image role field is always `inspiration`, but with an image attached the goal is the depicted
individual object itself, not a design brief (see "Confirmed Codex execution" below). Explicit text
controls the category, functions, counts, or must-haves that it actually states. The Author
completes occluded structure and anything the image cannot establish with ordinary category
knowledge, and prioritizes functional completeness, coherent geometry, and usable paths over copying
perspective or occlusion.

A request with neither text nor image fails before authoring. The same normalizer is used by Codex,
Provider single-item, and every Provider Batch row. Any supplied image must be a fully decodable
raster; invalid image bytes fail before workspace creation or Provider transport.

## Codex arm: two-turn interface, direct execution and CSV batch

By default `usdcraft generation-codex` is a two-turn operation (preflight, then confirmation).
`--direct` and `generation-codex-batch` are explicit user selections of immediate execution; see
[Direct execution and CSV batch](#direct-execution-and-csv-batch).

### Turn 1: zero-write preflight

Run the command without `--confirm-generation`:

```bash
uv run usdcraft generation-codex "Create a functional articulated desk lamp."
```

or with one inspiration image:

```bash
uv run usdcraft generation-codex \
  --image /absolute/reference.png \
  "Create a complete top-freezer refrigerator inspired by this design."
```

Inside a Codex task, when `--model` and `--reasoning` are both omitted, the preflight inherits the
root user task's model and reasoning effort (read from Codex's local thread state for the current
`CODEX_THREAD_ID`); the inherited pair is bound into the fingerprint like an explicit one. A normal
terminal, CI job, or external harness must pass `--model` and `--reasoning` together; without a
Codex thread the preflight fails instead of guessing a default.

Preflight resolves and validates the request, runtime, scaffold, and optional image and computes a
stateless approval fingerprint. It creates no author workspace or canonical record and starts no
Codex process. The normal approval surface contains exactly:

- normalized prompt;
- model;
- reasoning effort;
- input mode;
- absolute image path when present; and
- approval fingerprint.

`--json` is a diagnostic view of the complete plan, not the ordinary approval response. The outer
repository agent must show the compact summary once, ask the user to approve or cancel, and end that
turn. It must not infer approval or execute confirmation in the same turn.

### Turn 2: exact confirmation

Only a later explicit approval authorizes rerunning the same inputs with the exact fingerprint:

```bash
uv run usdcraft generation-codex \
  "Create a functional articulated desk lamp." \
  --confirm-generation <approved-fingerprint>
```

Changing any bound input or execution option requires a new preflight. The fingerprint is a
human-intent confirmation token, not a security boundary, source attestation, replay ledger, or full
repository/environment seal.

Other options: `--label` sets a concise object name for the canonical record and `--keep-workspace`
retains a successful workspace. Time and turn limits are Provider-only: the Codex arm has no time or
turn limit, so `generation-codex` has no `--timeout` and its preflight binds none; this includes its
optional image-observation call and review.

Do not author a canonical asset or create a record by hand. A successful record exists only after the
confirmed run passes the mechanical Host and Storage promotion path.

### Direct execution and CSV batch

`--direct` validates the inputs and executes immediately, without a preflight summary or
fingerprint. It is mutually exclusive with `--confirm-generation`. Input decoding, hash binding,
workspace staging, image observation, Author, mechanical gate and persistence reuse the confirmed
chain; the runtime report records `authorization: direct` (a confirmed run records `fingerprint`).

```bash
uv run usdcraft generation-codex "Create a functional articulated desk lamp." \
  --model <model> --reasoning <effort> --direct
```

`generation-codex-batch <csv>` executes the same direct chain once per row of the
[Generation CSV](#generation-csv), each in a separate supervised worker process on POSIX hosts
(no time limit; Ctrl-C stops each row's Codex processes and interrupted JSON lists finalized,
cancelled and in-flight rows). Batch-wide options are `--model`/`--reasoning` (paired row
defaults), `--concurrency` (1–32, default 1), `--validate-only`, `--keep-workspace`, `--json` and
`--repo-root`; there is no `--timeout`, `--run-timeout` or `--api-style`. Model/reasoning come from
a paired CSV override, the paired CLI defaults or the calling Codex task; all effective pairs are
resolved before rows start. The Provider-only `api_style` cell must be empty (a nonempty cell is
rejected, also by `--validate-only`). Each row has its own Author conversation; results identify
`provider=codex-cli`, and the runtime report records direct authorization and the batch and row
IDs. The `codex` executable must be on `PATH` or named by `USDCRAFT_CODEX_BIN`; the batch checks
this before starting rows.

```bash
uv run usdcraft generation-codex-batch rows.csv --validate-only
uv run usdcraft generation-codex-batch rows.csv \
  --model <model> --reasoning <effort> --concurrency 4 --json > results.json
```

## Confirmed Codex execution

[`execute_generation()`](command.py) performs the confirmed (or explicitly direct) item in this
order:

1. compare the supplied fingerprint with the freshly resolved preflight (direct execution skips
   only this comparison);
2. recover any prior incomplete Storage promotion transaction;
3. create one non-git workspace under `data/local/generation_runs/`;
4. copy the approved scaffold and optional image, then verify their approved hashes;
5. with an image, run image observation (below);
6. build stable absolute check, no-render compile and candidate-render commands;
7. build the task with [`build_generation_prompt()`](prompt.py);
8. run one ephemeral `codex exec` with its run-scoped Host renderer in that workspace;
9. accept only a schema-valid `complete` or `blocked` final message;
10. after `complete`, independently clean-compile with rendering disabled and require a floating
    Generation root;
11. validate and atomically promote one canonical record through [`persist.py`](persist.py).

With a reference image attached, the Generation goal is the depicted individual object — the same
individual, not merely a member of its category; the Author matches the individual's parts,
proportions, pose, and arrangement wherever the image shows them with reasonable confidence, and
completes the rest from category knowledge without chasing what the image cannot establish.

With an image, both arms use the shared observation engine through [`observation.py`](observation.py).
The first image-only inventory and its independent inventory review supply qualitative cues and
bounded inferred structure. This is input observation, not candidate evaluation. Names, counts,
grouping and attachments are advisory: the Author may verify and revise them against the request
and image. The Author records consequential estimated dimensions, proportions, pose and scale as
named parameters, distinguishing visible cues from assumptions. A single view does not establish
exact 3D ratios or angles. Codex observation calls have no time limit; Provider observation requests
use the Provider per-request `--timeout`. A failed observation degrades to an Author-only reading
with a warning; text-only runs skip observation. Inventory and manifest are archived under `traces/`.

`blocked` is reserved by the task for a concrete request or SDK limitation that cannot be repaired by
editing `model.py`. Python, SDK/API, check, export, or USD errors that can be repaired belong in the
same Author loop. A blocked or failed run creates no successful record and retains its workspace for
diagnosis. A successful workspace is removed unless `--keep-workspace` was approved.

Both arms time their Host-visible stages with the shared [`usdcraft/stagetime.py`](../stagetime.py):
freeze, image observation (only with an image), author (including tool feedback and repair), host
gate, persist (Codex preflight and confirmation are not timed). The Codex arm logs each stage
start/finish to stderr; the Provider arm reports through its progress display. Both append live
events to `stage_trace.jsonl` in the run workspace (retained on failure, archived in the record's
`traces/`) and persist the per-stage table as `stage_timings` in the runtime report. Timing is
descriptive metadata, never a gate.

## Author context and task policy

The generated text from [`prompt.py`](prompt.py) is the complete Generation Author policy. It
contains the declared request, input authority, deliverable, relevant authoring rules, SDK routing,
check/compile commands, workflow, and completion boundary. Changes that are meant to affect authored
assets must update this runtime task and its tests; placing prose only in `AGENTS.md` or this README
does not change the Author and must not be treated as an implementation.

The root `AGENTS.md` is an outer repository-agent command interface and knowledge index. It must not
be injected into an ephemeral Generation Author. The Codex launcher disables project-document
discovery for the author process (`project_doc_max_bytes=0`), ignores the user's personal Codex
configuration (`--ignore-user-config`) and disables the clean-context feature set shared with the
Reconstruction Author (`CLEAN_CONTEXT_DISABLED_CODEX_FEATURES` in
[`../pipeline/codex.py`](../pipeline/codex.py): sub-agents, plugins, apps, browser and computer use,
hooks, image generation, skill discovery and MCP elicitation), exactly as the Reconstruction Codex
Author does: one Author works from its task and workspace alone. The Provider harness naturally
receives only the generated task and its bounded tools. This prevents outer preflight instructions,
development guidance, or unrelated product context from entering one Author and keeps the two arms'
declared context aligned. Repository instruction edits therefore never constitute an Author prompt
change.

Both arms' tasks preload the compact SDK quickstart and list the full SDK document inventory as an
on-demand routing map; see [SDK context preload](../pipeline/README.md#sdk-context-preload). A
shell-enabled Codex Author enumerates exact example paths before opening one; Provider exposes
bounded `search_docs`, `find_examples`, and `read_file` equivalents. Both require consequential
defects to be addressed and the latest compiled candidate inspected before completion. Passing
check/compile is necessary but does not establish semantic or visual quality. Provider has a
64-turn budget with turn-budget notes in the last eight turns; Authors reserve room for inspection,
repair and finishing.

Do not put Generation pipeline documentation under `sdk/_docs/*/*.md`. Every Markdown page there is
part of the machine-checked Author SDK inventory.

## Authoring loop and Host boundary

Both arms use one Author for planning, construction, inspection and repair, for text-only and image
runs. Build a coarse executable candidate early. Choose body grouping and construction order from
function and supported image cues; an observation entry need not become a rigid body. Use focused
motion/clearance checks and no-render compilation, then inspect actual compiled visual geometry at
useful camera and joint poses. Repair consequential shape, placement, material, opening, overlap,
support and articulation defects; compile and reinspect after changes. A render failure is an
explicit tool limitation to report, never a new semantic gate.

For image runs, the Author establishes approximate object-frame landmarks and composes parent/child
transforms to obtain the intended rest pose. Candidate inspection first compares q=0 from a
reasonably similar camera against the reference: major junction heights, member directions,
functional-end facing, silhouette and relative part sizes. Supported differences lead to named
parameter/transform corrections and reinspection at that view; changing camera or joint pose alone
is not a repair. Uncertain differences may be retained with an explanation. Significant corrections
or uncertainty belong in model comments and the closing summary, with no mandatory number of defects,
repair rounds or semantic completion gate. Surface flicker/overlap remains visual work even when
collision allowances are justified.

Provider uses `write_model(content)` for first construction and whole-file rewrites, and small
patches against exact current source for local changes, with a focused read when needed. It accepts
source only, not a target path; syntax errors remain repair feedback. Both write paths retain
revision/check/compile freshness.

Provider's `validate_model` runs check, compile and a default-pose candidate render in one call.
Check or compile failure stops the checkpoint; rendering failure is reported separately and does not
block mechanically valid completion. Individual tools remain available for focused repair. Tool
responses, completion summaries and runtime metadata record coverage against the current model
revision; editing invalidates earlier successful coverage.

Provider exposes `render_candidate(azimuth, elevation, joints)` through
[`provider_tools.py`](provider_tools.py), delivering the original reference first (when present),
then candidate images, to the same Author history; `attachment_roles` identifies their order. It
requires the current model revision to have compiled. Codex uses:

```bash
usdcraft generation-render build --output author_feedback/view_001 --azimuth 45 --elevation 30
usdcraft generation-render build --output author_feedback/view_002 --joint hinge=0.5
```

Camera angles are degrees; joint positions are radians/metres. The Author opens the returned images;
a path alone is not inspection. [`candidate_render.py`](candidate_render.py) supplies the common
Generation rendering operation. [`render_service.py`](render_service.py) runs the renderer on the
Host during Codex authoring, accepting bounded file requests within this workspace. The Author does
not need browser or socket permissions. Standalone `generation-render` uses the same renderer
directly. Candidate views are output feedback, never new target evidence, and kinematic posing is
not a physics simulation. Renderer backend and fallback information remain explicit.

Both arms retain candidate feedback under `traces/author_feedback/` with
`author_feedback_index.json` path/hash entries. Runtime metadata records candidate-render
availability and Provider call counts; it does not certify that the Author actually inspected an
image. SDK and backend details remain in the public SDK inventory.

After Author completion, the deterministic Host runs a fresh clean no-render compile. It executes and
strictly exports the trusted staged model, requires the current output set and floating Generation
root, verifies static structure and dependency closure, and writes compile evidence. It does not rerun
`run_tests()` or make a second semantic, visual, realism, score, or simulator-runtime judgment. The
exact mechanical contract belongs to [`../compile/README.md`](../compile/README.md) and
[`../../storage/README.md`](../../storage/README.md).

Codex and Provider call the same final `compile_model(..., clean=True, render=False,
require_generation_root_floating=True)` and the same [`persist_generation()`](persist.py). Provider
does not own a parallel Host or promotion implementation.

## Canonical promotion

[`persist.py`](persist.py) identifies the authoring arm but shares one Generation record contract. It:

- validates the fresh compile output before staging;
- persists the normalized prompt, optional inspiration image, final `model.py`, runtime provenance,
  and the arm's bounded traces;
- writes lightweight provenance (schema 3); repository worktree state is not a publication
  constraint;
- writes the arm identity (`agent_harness` plus `provider`: `codex`/`codex-cli`, or `provider` with
  its endpoint `nvidia`/`openai`) into every identity projection and one
  `promotion_transaction_id` into `record.json`, `generation.json` and the compile report, as
  Storage requires of a current record;
- stages the materialization and immutable source revision;
- revalidates hashes, dependency closure, manifest/report binding, and required artifacts in their
  canonical layout; and
- commits through Storage's recoverable promotion transaction.

Viewer and offline rematerialization are consumers of this canonical state, not alternate Generation
or promotion routes. See [`../../storage/README.md`](../../storage/README.md) and the top-level
[`../../README.md`](../../README.md).

## Provider arm

The Provider arm (`usdcraft generation`, `usdcraft generation-batch`) reuses request semantics,
[`workspace.py`](workspace.py), [`prompt.py`](prompt.py), authoring check/compile behavior, final
Host compile, and [`persist.py`](persist.py). It differs only at the adapter surfaces: runtime
configuration, transport, bounded tools, harness memory, direct entry point, Batch orchestration,
and progress presentation.

Provider single-item and Batch execute directly; neither uses the Codex preflight or fingerprint.
Both accept `--model`, `--reasoning`, `--timeout` (deadline for each Provider HTTP request, default
900 s) and `--api-style responses|chat-completions` (default `responses`; it applies to the Author
and to image observation and is recorded in the runtime report). The Provider Author has a fixed
64-turn budget. See [`../pipeline/provider/README.md`](../pipeline/provider/README.md) for the exact
boundary.

### Generation CSV

`usdcraft generation-batch <csv>` runs each row through the same primitive on the shared coordinator.
Batch-wide options are `--model`, `--reasoning`, `--timeout`, `--api-style`, `--concurrency` (1–32),
`--run-timeout`, `--validate-only`, `--keep-workspace` and `--json`, with the same semantics as
`reconstruction-batch`. `generation-codex-batch` reads the same CSV contract; see
[Direct execution and CSV batch](#direct-execution-and-csv-batch).

| Column | Required | Meaning |
| --- | --- | --- |
| `row_id` | yes | Unique row identifier |
| `prompt` | yes (may be empty) | Request text; empty with an image selects the image-only default |
| `image_path` | yes (may be empty) | Optional inspiration image, relative to the CSV directory |
| `label` | no | Display name (`--label`) |
| `model`, `reasoning` | no | Per-row runtime override; both cells together or neither |
| `api_style` | no | `responses` or `chat-completions`; overrides `--api-style` for the row; Provider only (Codex batch rejects a nonempty cell) |

Unknown columns and surplus cells are rejected exactly as in the Reconstruction CSV, with an error
naming the unknown columns and listing the allowed ones. All effective runtime pairs are resolved
before rows start (mixed batches display `mixed`). `--validate-only` checks the CSV contract only:
no runtime resolution, credentials, model calls or workspace writes, and it does not decode images.
`--run-timeout SECONDS` (POSIX; default 1920) runs each row in a supervised worker process group
with a total wall-clock deadline from worker startup through persistence (`--run-timeout 0` removes
the limit and runs rows in the calling process); a timed-out row fails with
`status=timeout`, keeps its workspace and writes
`data/local/generation_deadlines/<run_id>/snapshot.json`. The supervisor is the shared
[`../pipeline/deadline.py`](../pipeline/deadline.py) also used by Reconstruction.

## Source map

| Concern | Canonical implementation |
| --- | --- |
| Input normalization | [`default_prompt.py`](default_prompt.py) |
| Codex request and preflight schema | [`contract.py`](contract.py) |
| Runtime model/effort selection | [`runtime.py`](../pipeline/runtime.py) |
| Codex preflight and confirmed/direct orchestration | [`command.py`](command.py) |
| Codex CSV batch | [`codex_batch.py`](codex_batch.py), [`provider_batch.py`](provider_batch.py) (CSV contract and supervised worker) |
| Shared Author task | [`prompt.py`](prompt.py) |
| Image observation adapter | [`observation.py`](observation.py) |
| Candidate rendering and Codex Host service | [`candidate_render.py`](candidate_render.py), [`render_service.py`](render_service.py) |
| Generation Provider tool adapter | [`provider_tools.py`](provider_tools.py) |
| Workspace staging | [`workspace.py`](workspace.py) |
| Codex process and enriched trace | [`codex.py`](codex.py), [`../pipeline/codex.py`](../pipeline/codex.py), [`trace.py`](../pipeline/trace.py) |
| Provider direct item | [`provider_command.py`](provider_command.py) |
| Provider Batch | [`provider_batch.py`](provider_batch.py), [`../pipeline/batch.py`](../pipeline/batch.py), [`../pipeline/deadline.py`](../pipeline/deadline.py) |
| Final compile/Host | [`../compile/command.py`](../compile/command.py) |
| Canonical record adapter | [`persist.py`](persist.py) |
| Artifact and promotion contract | [`../../storage/artifact_contract.py`](../../storage/artifact_contract.py), [`../../storage/canonical_promotion.py`](../../storage/canonical_promotion.py) |
| Explicit derived-cache repair | [`rematerialize.py`](rematerialize.py) |

## Tests

Focused Generation coverage is under [`../../tests/generation/`](../../tests/generation/):

- `test_contract.py`, `test_command.py`, `test_cli.py`, and `test_codex.py` cover request binding,
  two-turn behavior, execution, process context, and failure retention;
- `test_codex_batch.py` covers `generation-codex --direct` and `generation-codex-batch` (CSV
  contract, runtime resolution, supervised direct rows) with a fake Codex runner;
- `test_provider_writes.py` covers whole-file write recovery, atomic failure, bounded inputs, syntax
  feedback, revision freshness and isolation from the shared executor;
- `test_candidate_render.py` covers candidate poses, same-session Provider image delivery and
  repair, stale compile rejection, render failure recovery, and the Codex Host transport;
- `test_prompt.py` locks the shared task body, input authority, SDK routing, and arm-specific reference
  access;
- `test_generation_observation.py` covers image observation;
- `test_workspace.py`, `test_persist.py`, and `test_rematerialize.py` cover staging, canonical
  persistence, and derived-cache repair;
- `test_provider_command.py`, `test_provider_contract.py`, `provider/test_harness.py`,
  `provider/test_tools.py`, `provider/test_client.py`, and `provider/test_chat.py` cover the Provider
  adapter;
- `test_provider_batch.py` and the display tests cover Batch isolation, interruption, and fail-open
  presentation.

Run the focused suite after changing this component:

```bash
uv run pytest -q tests/generation
uv run ruff check usdcraft/generation tests/generation
```

Changes to compile, SDK, Storage, or Viewer contracts also require their owning focused suites.
