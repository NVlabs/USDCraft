# USDCraft architecture boundaries

This document owns the small set of cross-component rules that cannot live in only Generation,
Reconstruction, Compile, Storage or Viewer. It is a contributor reference, not a runtime Author
prompt or user command guide.

## Product separation

Generation creates a new asset from a design request. Reconstruction rebuilds one declared source
instance under an explicit evidence contract. Edit is not implemented. Their request semantics,
image roles, evidence authority and agent contexts must not be merged into a generic Observation
state machine.

Runtime Authors, Image-Observation Agents and the isolation-only Agent operate on one staged asset
under their generated tasks; none of them changes repository code. The Host is deterministic
orchestration; Storage and Viewer are application components. The Host/Viewer cannot become a
semantic asset reviewer.

## Author and Host boundary

Semantic and functional quality is instruction-first. The applicable Author repairs category,
function, support/contact, affordance, appearance, articulation and plausible physics while staged
`model.py` is editable, using focused tests and diagnostics.

The final Host gate is narrow and mechanically reproducible: executable/exportable source,
schema/structure, finite numeric legality, required files, dependency closure, hashes and canonical
binding. A new fail-closed blocker must fit that boundary. Subjective quality, realism, appearance,
mass plausibility, sampled-pose interpretation and rendered similarity remain authoring feedback;
they do not justify a second reviewer, score, best-candidate selector or multi-agent approval layer.

Static checks establish only the evidence they actually measure. They do not establish Isaac Sim
import, runtime startup, solver stability or behavior, and a timeout-bounded Python worker is not an
operating-system security sandbox.

## Shared core and execution arms

`usdcraft/pipeline/` owns shared Author context, execution contracts, Codex process handling,
Provider transport and tool loop, bounded common tools, observation execution and batch scheduling.
Generation and Reconstruction depend on this package; it imports neither product. Their request
types, preparation, evidence roles, additional tools and record assembly remain product-owned.
Shared completion tools take an explicit product name instead of defaulting to Generation.
The shared final Host compile lives in `pipeline/execution.py`; canonical validation and promotion
continue to use Storage. `render/service.py` owns the common render request queue and lifecycle,
with each product providing its renderer and workspace policy.

See [`pipeline/README.md`](../usdcraft/pipeline/README.md) for the implementation map.

Generation's Codex and Provider arms read the shared request normalizer, Author task, SDK
inventory, authoring diagnostics, no-render compile, final Host and persistence sources in place.
Its Codex arm has the same entry points as Reconstruction's: `usdcraft generation-codex` (default
two-turn approval or explicit `--direct`) and `generation-codex-batch` (direct CSV rows).
The physical-authoring policy (the flexible-member rule, mass, materials, contact presets, joint
limits and resistance, gravity hold, drives, radians, backend capability facts) is one text in
`usdcraft/pipeline/policy.py` rendered into both products' Author tasks; like the shared
observation contract body, it carries no product semantics.
Provider-specific code is limited to its entrypoint, transport, bounded tool/harness adapter,
Batch orchestration and presentation. It must not duplicate product semantics, the Host gate or
canonical promotion.

Reconstruction has the same two arms as Generation over one shared core: the Provider arm
(`usdcraft reconstruction`, direct execution, one Author per run) and the Codex arm
(`usdcraft reconstruction-codex`, default two-turn approval or explicit `--direct`;
`reconstruction-codex-batch` authorizes direct CSV rows; Codex shell transport). Both arms share
one task doctrine, chain and record shape: the source mesh is always reference evidence, direct
source measurements support metrics, AXIAL feature interpretations are revisable, the image fixes
identity, shape and pose category, and unresolved metric attributes remain constrained hypotheses.
Reconstruction and Generation never share hidden Author state or silently rebind one product's
input as another product's evidence.

Reconstruction authorization (fingerprint confirmation or explicit direct execution) is
followed by an atomic input snapshot before any Agent: complete
source closure with relative topology, semantic image, scaffold, and a hashed manifest. Reopening the
copied source must resolve exactly that self-contained closure. Preprocessing, the Author and
canonical persistence consume only the snapshot. This mechanical freeze does not merge Generation
inspiration semantics with Reconstruction evidence authority and does not permit image-only Recon.

Reconstruction runs user command → scale/freeze → optional isolation → mesh preprocess →
optional independent image observation → one Author → Host gate →
persist/export. Geometry diagnosis, direct mesh queries and candidate visual inspection/repair
all stay inside the Author session, with no new semantic agent. The Author can revise observation
names, counts and grouping. Codex may inspect the staged normalized mesh with local geometry tools;
Provider uses bounded queries. Both arms share `reconstruction/author_tools.py`, and the Provider
harness delivers tool-created images back to that same conversation after function results.
Raw source/isolation audit state and other runs stay excluded. Candidate renders are output
feedback, not evidence about the target. The `--[no-]image-observation` switch binds the advisory
observation input; mesh measurement, on-demand source rendering and candidate rendering are always
available. Every model uses the same flexible construction and inspection workflow, with no
semantic reviewer.

Generation has no source mesh or AXIAL. Its chain is user command → freeze → optional image
observation → one Author with check/compile, candidate rendering and in-session repair → mechanical
Host gate → persist/export. Both Codex and Provider use this workflow for text and image runs;
there is no independent visual critic. Candidate feedback describes output, never target
evidence. Single-image dimensional estimates and the real-world scale anchor are Author hypotheses,
not source measurements. The observation inventory is advisory and may be revised; failure degrades
to an Author-only reading. Its shared engine is `usdcraft/pipeline/observation.py` with separate
product profiles. Generation's render adapters live under `generation/` and use the existing renderer;
Reconstruction's evidence tools, capability controls and lifecycle stay separate.

## Documentation and evidence ownership

Executable code and focused tests are authoritative. Component-local documentation describes the
current contract. The root README is a user quickstart and `AGENTS.md` is the repository-agent
router. Runtime Author policies live in their task builders;
changing repository prose alone cannot change an Author.

Public SDK documents are part of a machine-defined Author inventory. Do not place Storage, Viewer,
release or repository-development instructions under `sdk/_docs`. Ignored local output never
changes how the product behaves.

Reconstruction's task and authoring operations are shared across Provider function tools and
Codex shell commands. `reconstruction/lifecycle.py` owns both arms' final mechanical compile,
runtime diagnostic reduction, evidence/trace assembly, promotion and cleanup. Per-arm runtime
selection, model communication, approval and cancellation remain adapters.
