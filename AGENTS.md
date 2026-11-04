# USDCraft repository agent entrypoint

This file is the entry contract for a **repository agent** working in this checkout: it routes
tasks to the component that owns the relevant behavior and defines how to invoke the public
Generation and Reconstruction workflows for the user. This file is intentionally not an Author
prompt, SDK manual, evidence specification, backend handbook or storage schema.

Executable source and focused tests are the final authority. Component-local documentation explains
ownership and invariants. The root `README.md` is the user quickstart. Read only the documentation
routed by the current task instead of loading every repository document.

## Product and role boundary

USDCraft has three separate product directions:

- **Generation** creates a new asset from text, one inspiration image, or both.
- **Reconstruction** reconstructs one declared source instance from typed evidence and creates
  canonical records through its Provider arm (default) or its Codex arm.
- **Edit** is not implemented.

The image-observation task contract body is single-sourced in `usdcraft/pipeline/observation.py`; each
product supplies only its own semantic profile (agent identity, attached evidence, reading
authority, dimension ownership, render handling). Sharing that contract body never merges product
semantics: do not move Generation request semantics, Reconstruction evidence authority or runtime
agent context into the shared wording or into a generic Observation workflow. Never
silently reuse a Generation image as Reconstruction evidence; the user must explicitly supply it
to Reconstruction as the subject's semantic image.

Runtime Generation Authors, Reconstruction Authors, Image-Observation Agents and Subject-Isolation
Agents author, observe or classify one staged asset under their exact task prompts; they do not
redesign this repository. The deterministic Host stages, compiles, exports, validates, persists and
recovers. Storage and Viewer are application components, not
semantic reviewers. Product-launched Codex Author and isolation processes intentionally do not
receive this repository entrypoint.

## User command interface

Time and turn limits are Provider-only: Provider `--timeout` (each Provider HTTP request, default
900 s), batch `--run-timeout` (each row's total time, default 1920 s; `0` removes the limit) and
Reconstruction `--max-turns` (Generation has a fixed 64-turn budget). The Codex arm has no time or
turn limit and accepts none of these options. Pass them only when the user supplies them. A shared
CSV may contain Provider-only columns (`api_style`, and `max_turns` for Reconstruction), but for
the Codex batch commands those cells must be empty; a non-empty value is rejected as
Provider-only. Apart from these limits, the Provider and Codex arms of each product accept the same
options, including Reconstruction `--mesh-cleanup`. Every run command, single or batch, keeps a
successful run's working directory only with `--keep-workspace`.

### Installation (`uv sync`, `usdcraft setup`)

When the user asks to install or set up USDCraft, check the prerequisites first: macOS or Linux,
[`uv`](https://docs.astral.sh/uv/), and Node.js ^20.19 or >=22.12 with npm (`uv` provides Python
3.12). If one is missing, report it with its install link and stop; do not install system packages
or run remote install scripts. Otherwise run, from the repository root:

```bash
uv sync
uv run usdcraft setup
```

`usdcraft setup` installs Playwright Chromium, builds the viewer and creates the `data/` layout. On
Ubuntu, if Chromium reports missing system libraries, tell the user to run
`uv run playwright install-deps chromium` (it needs administrator rights). Finish by reporting that
the Codex workflows are ready, and that the API workflows need `OPENAI_API_KEY` in `.env` (copied
from `.env.example`). The user sets the key themselves: never ask for its value or write it.

### Generation through the Provider (OpenAI or NVIDIA API) (`usdcraft generation`)

`usdcraft generation` is the direct-execution Provider arm and the default entry point: it starts
immediately with no preflight or fingerprint, runs one Author, and promotes one record. Use it only
when the user explicitly asks to generate; pass exactly the text, `--image` and `--label`
the user supplies. Both input modes use one Author with candidate inspection and repair.
`generation-batch` runs CSV rows through the same primitive; use it only when the user explicitly
selects that interface. Report the printed `record_id`, warnings and viewer
line. See `usdcraft/pipeline/provider/README.md` for the shared primitive and boundaries.

### Generation through Codex (`usdcraft generation-codex`)

When the user explicitly selects direct execution, use `generation-codex --direct` with the
supplied inputs. This authorizes immediate execution after input validation without a fingerprint
exchange. When the user selects CSV batch, use `generation-codex-batch <csv>` (the same CSV
columns as `generation-batch`); each row runs the same direct Codex primitive, with default
`--model`/`--reasoning` (a per-row CSV pair overrides both). These explicit interfaces are
exceptions to the default protocol below. Never add `--direct` merely to bypass an unapproved
default preflight.

When the user explicitly asks for the Codex arm without selecting one of those interfaces:

1. Read `usdcraft/generation/README.md` and run the requested command **without**
   `--confirm-generation`. This is the zero-write preflight and must not start Codex or create a
   workspace/record.
2. Report the compact approval summary exactly once: normalized prompt, resolved model, reasoning
   effort, input mode, absolute image path when present, and fingerprint. Do not paste the diagnostic
   `--json` payload or repeat internal roots, hashes, schemas, executables or output layers.
3. Ask the user to approve or cancel, then end the turn. Never infer approval from the original
   request and never confirm in the same turn as preflight.
4. Only after a later explicit approval, rerun the same inputs with
   `--confirm-generation <fingerprint>`. Any bound-input change requires a new preflight.

Valid inputs are text, text plus at most one inspiration image, or one image with no explicit text.
An empty request is invalid. Inside a Codex task, inherit that root task's model and reasoning
effort; outside Codex, provide `--model` and `--reasoning` together. Do not create canonical records
by hand. Both arms use one Author with candidate inspection and repair.

### Reconstruction through the Provider (OpenAI or NVIDIA API) (`usdcraft reconstruction`)

`usdcraft reconstruction` is the direct-execution Provider arm: it starts immediately with no
preflight or fingerprint, runs one Author, and promotes one record. Use it only when the user
explicitly asks to reconstruct; pass exactly the source, `--image`, and any `--prompt`, capability
flags, frame, guidance, label or `--mesh-cleanup` the user supplies. Every model uses the same shared harness, with no
independent critic. The source mesh is always the metric authority for each attribute it reliably
resolves. The image fixes identity, shape character, pose categories and required visible parts,
including those absent from the brief; unresolved metrics are constrained hypotheses. Both Authors
always have AXIAL, measure_face, direct mesh measurement, on-demand source rendering and candidate
rendering. `--[no-]image-observation` controls the advisory observation. The Author starts with only
the semantic image; source views are requested through `render_source` (Provider) or
`recon-render-source` (Codex). Provider `--max-turns` sets the Author
budget (80 by default). Image-observation inventories are advisory. A source without units whose
longest side sits on 1.0 or 2.0 is treated as unit-normalized: the arm estimates its real scale
from the semantic image and records the estimate; pass `--meters-per-unit` when the user knows the
scale. Without `--prompt` the Host supplies the same deterministic default request text as
Generation (a sentence naming the primary subject); do not compose one for the user.
`--mesh-cleanup` (default off) runs the optional Subject-Isolation Agent through the Provider transport
exactly as on the Codex arm; pass it only when the user asks for mesh cleanup. Report the printed
`record_id`, warnings and viewer line.
`reconstruction-batch` runs CSV rows through the same primitive (columns: see
`usdcraft/reconstruction/README.md#reconstruction-csv`); use it only when the user
explicitly selects that interface, and preserve the user-supplied switch settings. A CSV
`mesh_cleanup` cell (true/false, default off) selects mesh cleanup per row on both batch commands.

### Reconstruction through Codex (`usdcraft reconstruction-codex`)

When the user explicitly selects direct execution, use `reconstruction-codex --direct` with the
supplied inputs. This authorizes immediate execution after input validation without a fingerprint
exchange. When the user selects CSV batch, use `reconstruction-codex-batch <csv>`; each row runs
the same direct Codex primitive, with default `--model`/`--reasoning` (a per-row CSV pair
overrides both). Without an explicit pair, a Codex command run inside a Codex task inherits that
root task's model and reasoning effort; outside Codex, pass both.
These explicit interfaces are exceptions to the default protocol below. Never add `--direct`
merely to bypass an unapproved default preflight.

When the user explicitly asks for the Codex arm:

1. Read `usdcraft/reconstruction/README.md` and run the requested command **without**
   `--confirm-reconstruction`. This is a zero-write Host preflight; it starts neither isolation nor
   authoring and creates no workspace/record. Pass exactly the source, `--image`, and any
   `--prompt`, capability flags, frame, guidance, label or `--mesh-cleanup` the user supplies.
2. Report one compact approval summary containing source, semantic image, request text, declared
   frame, mesh-cleanup state, resolved capability switches, model, reasoning effort, guidance and
   label when present, and fingerprint.
3. Ask the user to approve or cancel, then end the turn. Do not infer approval or execute a
   confirmed run in the preflight turn.
4. Only a later exact `--confirm-reconstruction <fingerprint>` may start the automatic chain. Any
   bound-input or option change requires a new preflight.

Confirmation authorizes the Provider arm's chain under Codex transport: the scale estimate for a
unit-normalized mesh (a Host derivation recorded in the request, never a new approval), the freeze,
optional isolation, mesh preprocess, image observation, one Codex Author over the same task
doctrine, the Host gate and one `reconstruction_ready` record. There is no hidden evaluator or
semantic Host reviewer. Never create a Reconstruction record by hand. Report the printed
`record_id`, warnings and viewer line.

The semantic image is the reconstruction target's semantic record on both arms; it remains distinct
from AXIAL and the mesh reference views. Read `usdcraft/reconstruction/EVIDENCE.md` only for
evidence-contract work. Evaluation ground truth and case mappings must never enter the runtime
prompt.

## Knowledge routing

Read the smallest matching set before changing a component:

| Task | Read first | Executable authority |
| --- | --- | --- |
| User setup and command examples | `README.md` | `usdcraft/cli.py` |
| Cross-product or Host-boundary design | `docs/architecture.md` | Shared prompts, compile and storage tests |
| Shared Author pipeline / execution / tools | `usdcraft/pipeline/README.md` | `usdcraft/pipeline/`, `usdcraft/render/service.py` |
| Shared Author-task blocks (physical-authoring policy, observation contract body) | docstrings of `usdcraft/pipeline/policy.py`, `usdcraft/pipeline/observation.py` | same files |
| Generation / Codex arm | `usdcraft/generation/README.md` | `usdcraft/generation/` |
| Provider (OpenAI or NVIDIA API) / Batch | `usdcraft/pipeline/provider/README.md` | `usdcraft/pipeline/provider/`, `usdcraft/pipeline/batch.py`, product `provider_batch.py` |
| Reconstruction | `usdcraft/reconstruction/README.md` | `usdcraft/reconstruction/` |
| Reconstruction evidence / AXIAL | `usdcraft/reconstruction/EVIDENCE.md` | `prompt.py`, `axial.py`, `diagnostics.py` |
| Public authoring SDK | `sdk/README.md`, then only relevant `sdk/_docs/**` | `sdk/_profiles.py`, `sdk/_core/` |
| Check, compile, USD/export/backends | `usdcraft/compile/README.md` | `usdcraft/compile/`, SDK exporters/tests |
| Canonical records / promotion / recovery | `storage/README.md` | `storage/` and product `persist.py` |
| Viewer/API/UI | `viewer/README.md` | `viewer/api/`, `viewer/web/src/` |
| Local data / optional export packages | `data/README.md` | layout and validation code named there |
| Security and credential boundaries | `README.md` (Where results are stored), `viewer/README.md` (Viewer access), `.env.example` (credentials), `usdcraft/pipeline/provider/README.md` (Transport and security) | transport/worker implementation |

SDK runtime inventories are machine-owned by `sdk/_profiles.py`. Do not put repository, Storage,
Viewer or release instructions into `sdk/_docs`, because those files are exposed to asset Authors.
Do not treat ignored local outputs as current executable contracts.

## Modifying this repository

Cross-product and Author/Host boundary rules live in `docs/architecture.md`. Update the component
document that owns a changed behavior, and keep repository, Storage, Viewer and release notes out
of `sdk/_docs`.
