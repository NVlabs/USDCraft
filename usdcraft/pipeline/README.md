# Shared authoring pipeline

Generation and Reconstruction prepare distinct inputs and tasks, then use this package to execute
one Author, access common authoring tools and compile the final model. This package imports neither
product. Product adapters provide evidence, additional tools, completion names and persistence.

| Responsibility | Implementation |
| --- | --- |
| SDK foundations and document inventory | `sdk_context.py`; inventory remains owned by `sdk/_profiles.py` |
| Shared physical authoring rules | `policy.py` |
| Image-observation contract and execution (Codex calls have no time limit; Provider requests keep the per-request timeout) | `observation.py`; product adapters supply semantic profiles |
| Execution result/runtime types | `contracts.py` |
| Codex process execution and workspace traces | `codex.py`, `trace.py`, `runtime.py` |
| Provider (OpenAI or NVIDIA API) transport, conversation loop and usage | `provider/`; see [Provider runtime](provider/README.md) |
| Read/search/write/patch/check/compile and revision freshness | `tools.py`; products compose their completion tool |
| Author checks, isolated Provider compilation and final Host compilation | `execution.py` |
| Batch scheduling, shared CSV column/runtime checks, progress and display | `batch.py`, `batch_progress.py`, `batch_display.py`, `single_display.py`, `display_common.py` |
| Per-row Batch supervisor (Provider `--run-timeout`, default 1920 s; Codex batch rows without a clock; Ctrl-C stop) | `deadline.py` |
| File hashes, timestamps, git commit and JSON writes | `hashing.py`, `files.py` (the image-only Generation request text lives in `usdcraft/generation/default_prompt.py`) |

The shared final compile is clean, render-free and requires a floating root. Product lifecycle
code assembles its record and evidence; Storage owns canonical validation and promotion transactions.
Shared candidate-render queue mechanics live in `usdcraft/render/service.py`. Each product binds
its renderer, workspace environment, path requirements and initialization policy explicitly.
Provider usage and request-composition telemetry are described in the
[Provider runtime](provider/README.md).

Keep source normalization, AXIAL, mesh measurements, source comparison, capability switches and
Reconstruction evidence interpretation in `usdcraft/reconstruction/`. Generation owns its request
and inspiration-image semantics. Neither product imports the other, and neither contributes a
semantic acceptance rule to the deterministic Host.

## SDK context preload

This section is the authoritative description of the SDK context given to every Author, in both
products and both execution arms.

- `sdk_context.py` embeds the canonical compact quickstart,
  [`sdk/_docs/common/00_quickstart.md`](../../sdk/_docs/common/00_quickstart.md), verbatim in the
  task's PRELOADED SDK REFERENCES section. Runtime `preloaded_docs` records exactly the embedded
  files.
- The complete SDK document inventory is rendered from `sdk/_profiles.py` with repo-relative paths
  under one declared repository root and marks the embedded file as preloaded. Provider tools
  resolve those paths directly; shell-enabled Codex Authors join them to the declared root.
- Full type, assembly and testing references stay in the inventory and are read on demand. Listing
  a path is not reading it; Authors open only the subset relevant to the asset.
- Shared mechanical policy keeps consequential requirements inline; detailed API/export
  explanations live in SDK references.

SDK routing in the quickstart maps concrete operations (solid fillets/booleans, procedural rounded
profiles, lofts, organic smoothing, graphics) to exact references. Shared `search_docs` matches
whole words and API identifiers, uses headings/path terms for routing, and returns one best line
per document within the requested limit. This avoids substring matches such as text/context and
repeated hits crowding out another relevant reference. It does not add a preloaded document.

## Initial cache boundary

Generation and Reconstruction place shared task rules, the full preloaded Quickstart and the
complete SDK reference inventory before instance-specific inputs. This ordering does not shorten
the instructions or documentation.

Provider `run(cache_prefix=...)` optionally splits an exact task prefix into a separate text block
with an explicit cache breakpoint. The remainder and images stay in the same request; concatenating
text blocks reproduces the task. Product adapters choose this boundary; the shared harness does not
interpret product instructions. Later image feedback keeps its own text breakpoint. Cache hits are
not guaranteed.

## Candidate checkpoint

Both Provider adapters use the shared candidate checkpoint (`validate_model`) and revision coverage
tracker. The checkpoint runs check and compile once and stops on check/compile failure; product
diagnostics that follow remain advisory. Generation adds candidate rendering; Reconstruction adds
AXIAL and candidate rendering. Completion reports current, stale, failed, disabled or unrun coverage
without claiming visual acceptance.
