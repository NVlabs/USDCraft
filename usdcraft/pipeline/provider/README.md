# Shared Provider runtime (OpenAI or NVIDIA API)

This directory owns the Provider transport for the NVIDIA inference API or the OpenAI API,
conversation execution and telemetry for both Generation and Reconstruction. Product entrypoints supply staged inputs, task text, tools, completion names
and progress labels. Shared bounded tools and final compilation live in the parent pipeline package.
The Generation-specific examples below describe its adapter; Reconstruction uses the same runtime
with its own evidence and tool extensions.

Read [`../../generation/README.md`](../../generation/README.md) first for the Generation product contract.
For Reconstruction, use [`../../reconstruction/README.md`](../../reconstruction/README.md).
Its source preparation, AXIAL operations and evidence permissions remain in the Reconstruction
adapter; neither this transport nor the common tools import either product implementation.

## Boundary

Generation and Reconstruction both support explicit `responses` (default) and `chat-completions`
wire protocols (`--api-style`, or a CSV `api_style` cell per Batch row) through the same client,
retry/cancellation boundary and canonical harness history. Chat mapping lives in `chat.py`: it retains the native assistant message (including provider reasoning fields)
for replay and exposes normalized calls/text to the existing tools. Image observation, scale
estimation and the optional Reconstruction Subject-Isolation Agent (`--mesh-cleanup`, default off)
use the same configured client, model, reasoning effort, wire protocol and per-request timeout. `provider-default` reasoning omits the reasoning
fields; other efforts are sent as `reasoning.effort` for Responses or `reasoning_effort` for Chat.
The protocol is explicit, with no model-name inference or fallback to another endpoint.

Provider-specific code is limited to:

- resolving the OpenAI or NVIDIA runtime and credential configuration;
- serializing and sending Responses or Chat Completions requests;
- retaining one item's complete local conversation history;
- exposing bounded equivalents of the shared authoring surfaces;
- adapting structured completion to the common Generation result;
- single-item and Batch progress presentation; and
- Batch scheduling, row failure isolation, and cooperative cancellation.

It must not fork the shared semantic prompt, duplicate the mechanical Host, introduce an independent
semantic reviewer, or implement a separate record/promotion contract.

The harness and tool executor are product-neutral: `ProviderHarness` accepts a product's tool
schemas, progress label and several staged images, and `ProviderToolExecutor` is subclassed with
extra readable evidence files and product tools. Reconstruction's Provider arm
(`usdcraft/reconstruction/provider_command.py`, `provider_tools.py`) reuses them this way; see
`usdcraft/reconstruction/README.md`.

SDK APIs and PhysX/Newton authoring rules belong in
[`../../../sdk/_docs/`](../../../sdk/_docs/). Mechanical export belongs in
[`../../compile/README.md`](../../compile/README.md), canonical state in
[`../../../storage/README.md`](../../../storage/README.md), trusted-code policy in
[`../../../README.md`](../../../README.md#where-results-are-stored), and credentials in
[`../../../.env.example`](../../../.env.example) and [Transport and security](#transport-and-security).

## Direct single-item flow

`usdcraft generation` executes immediately. It has no Codex preflight, approval fingerprint, or
later confirmation turn.

```bash
uv run usdcraft generation \
  --image /absolute/reference.png \
  "Create a complete top-freezer refrigerator inspired by this design."
```

[`execute_provider_generation()`](../../generation/provider_command.py) performs one item:

1. validate the repository and non-forked compile-worker boundary;
2. validate the optional raster image, then normalize explicit text or image-only input with the
   shared Generation normalizer;
3. resolve the Provider model, reasoning, request timeout, request-size limit, and credentials;
4. recover prior incomplete Storage promotions and create one shared Generation workspace;
5. build the same task-policy body from [`../../generation/prompt.py`](../../generation/prompt.py), selecting only the Provider
   identity and bounded reference-access wording;
6. start one [`ProviderHarness`](harness.py) with a new client, Generation tool executor and local
   history; that same Author can render its compiled candidate and repair it;
7. require a structured `complete` or `blocked` through `finish_generation`;
8. after `complete`, call the same fresh no-render floating-root Host compile used by Codex; and
9. call the same [`persist_generation()`](../../generation/persist.py) with Provider authoring identity.

`--model`, `--reasoning`, `--timeout` and `--api-style` override the configured Provider runtime for
this item, and `--label` sets a concise object name for the canonical record. `--reasoning` accepts
the typical efforts `low` (default), `medium`, `high` and `xhigh` verbatim, or `provider-default`,
which omits the reasoning field for models that reject an explicit effort. An item failure retains its
run/workspace identity and creates no successful record. A successful workspace is removed unless
`--keep-workspace` was requested.

Provider executes the concrete supplied input directly. It does not perform Codex's approved
scaffold/image hash comparison because there is no approval interval or fingerprint.

## Shared task, distinct access adapter

[`../../generation/prompt.py`](../../generation/prompt.py) is the runtime Author contract for both arms. Provider calls it with:

- `agent_name="Provider"`;
- `reference_access="provider_tools"`;
- `check_command="check_model"`; and
- `compile_command="compile_model"`.

The semantic and authoring body stays shared. Only identity, available command names, and reference
discovery wording differ. Provider must not maintain a copied prompt snapshot.

Provider sends only the task produced by `prompt.py`; the root `AGENTS.md` never enters the Author
context (see [Author context and task policy](../../generation/README.md#author-context-and-task-policy)).

## Local conversation memory

Each image-bearing user message places an explicit `prompt_cache_breakpoint` on its text block
before the images. This applies to initial task images and tool feedback images; text-only
messages are unchanged. Implicit caching remains enabled. The marker preserves a
candidate reusable prefix but does not guarantee a Provider cache hit or full image-cache reuse.

Generation and Reconstruction also supply an exact public task prefix through
`run(cache_prefix=...)`. The harness places that prefix in a separate text block with its own explicit breakpoint, including
for text-only input. Concatenating the blocks retains the complete task; instance-specific text
and image attachments follow the shared prefix. This does not enable shared conversation state.

[`ProviderHarness`](harness.py) owns one in-memory Responses history per item. The first local history
entry contains the complete generated task and optional staged image; later entries retain provider
output, function calls, and function results. The complete local history is replayed on every Provider
turn.

Provider-side storage and `previous_response_id` are not canonical memory. The client uses
`store=false`, and neither a direct item nor a Batch row may resume or share another item's local
history, client, workspace, or record identity.

The serialized full-history request is checked against a configurable byte limit before transport.
An oversized request fails locally with safe byte counts. History is never silently truncated,
summarized, or mutated to fit the limit.

## Bounded Author tools

[`../tools.py`](../tools.py) supplies shared reading, whole-file writing, exact patches and
check/compile tools. Generation adds the candidate checkpoint, candidate rendering and its completion
tool in [`../../generation/provider_tools.py`](../../generation/provider_tools.py):

- `read_file` — staged `model.py` or an exact file under `sdk/_docs`/`sdk/_examples`;
- `search_docs` — bounded lexical results from the canonical SDK document inventory;
- `find_examples` — bounded relevant SDK examples and exact paths;
- `write_model` — complete Python source to staged `model.py` only (up to 262,144 characters);
- `apply_patch` — an exact local update to staged `model.py`, with current context or removed lines in every hunk;
- `check_model` — shared authoring-only tests and diagnostics;
- `compile_model` — shared compile with rendering disabled;
- `validate_model` — one checkpoint call: check, compile and a default-pose candidate render;
- `render_candidate` — current compiled visual geometry at requested camera/joint poses; and
- `finish_generation` — structured `complete` or `blocked`.

There is no shell, arbitrary repository read, arbitrary filesystem write, or direct record mutation.
Tool results are returned to the same item's local history. Use `write_model(content)` for first
construction or a whole-file rewrite; it needs no old scaffold context. Writes are atomic, report
syntax problems as editable feedback, and invalidate prior check/compile success only when content
changes. Failed patches leave the file and revision unchanged and return a reread/whole-write
recovery hint with the failure reason. Both Generation and Reconstruction reuse these editing
operations. Reconstruction additionally reports whether check/compile/AXIAL/render results cover
the current revision. Whole-file replacement removes the old scaffold; local patches require exact
current text, with a focused read only when that text is unknown or a match failed.

## In-session candidate inspection

Both text and image runs use one Author conversation for construction, checks, no-render compilation,
candidate inspection and repair. `render_candidate` requires compilation of the current revision;
its returned images enter the same conversation after function results. On image runs the staged
original reference is reattached before the candidate, with `attachment_roles` declaring the order.
The input remains inspiration; it is not generated feedback or new evidence. Camera angles are degrees;
joint positions are radians/metres. The tool reports backend capabilities and fallback information,
and rendering errors are ordinary diagnostic feedback. They do not invalidate mechanical completion
or start another agent. The Author opens and interprets candidate views, makes supported repairs and
rechecks/recompiles before completing. Candidate views describe output, never new target evidence.

SDK context follows the shared [SDK context preload](../README.md#sdk-context-preload). The Author
owns consequential semantic and visual checks rather than ending immediately on its first successful
compile. Its budget is 64 turns; tool results in the last eight turns carry `turns_remaining` and a
note to reserve time for completion. No separate review or repair budget exists.

Candidate feedback is retained under `traces/author_feedback/` with a path/hash index.

Each `provider.tool.completed` trace event retains the call's model-authored arguments, its
duration, and — for authoring tools and every failed call — the complete local result payload, so a
run's authoring and repair sequence (including in-session candidate repairs) can be replayed from
the retained trace alone. Successful `read_file`/`search_docs`/`find_examples` results omit their bodies because they are reproducible
from the repository plus the recorded arguments.

Each `provider.response.completed` event retains the Provider's reported input, output and total
tokens, plus `cached_input_tokens`, `cache_write_input_tokens` and `reasoning_output_tokens` when
the endpoint reports the corresponding nested usage details. Explicit zero is retained; missing
or invalid counters remain unknown. These detail counters are subdivisions, not extra tokens to
add to input/output totals. Final `usage` includes a counter only when every successful response
reported it; `usage_reported_turns` gives coverage per counter. Per-turn events retain partial
coverage. Failed requests and unreported retry costs cannot be reconstructed from these counters.

Before each logical Author request, `provider.request.composition` records the exact compact JSON
byte sizes of the `input` and `tools` fields, input item counts, schema/source hashes and source
revision. It groups the initial task, reasoning state, tool calls/results by tool name, and later
messages, separating previously sent items from newly appended items. New item descriptors are
recorded once with indexes and hashes. Text, image URL and encrypted reasoning byte counts are
separate. The measurement uses the actual model-visible tool result, which can be smaller than
the retained diagnostic payload. No prompt text, document body, image data or encrypted reasoning
content is copied into composition metadata. Telemetry failures do not interrupt authoring.

These are byte measurements, not component token charges: the Provider exposes cache counts for
the whole turn, not which document or tool result was cached. Previously sent input is not
necessarily a cache hit. Initial task text includes preloaded SDK documents; it is one measured
block. The JSON field sizes exclude other request fields and HTTP overhead. Full canonical history,
tool schemas, the Author loop and tool behavior remain unchanged.

Every write or patch that changes source increments the staged revision and invalidates prior
check/compile success. A no-op preserves that state.
`finish_generation(status="complete")` is accepted only after at least one real edit and successful
`check_model` and `compile_model` calls for that exact current revision. This freshness check is a
Provider harness adapter invariant. It does not replace the later independent Host compile.

## Transport and security

[`client.py`](client.py) reads only explicit Provider values from the repository `.env`, without
dotenv interpolation or copying the API key into `os.environ` or tool arguments. `OPENAI_API_KEY`
selects the OpenAI API, the default endpoint (`OPENAI_MODEL`, default `gpt-6-astra`;
`OPENAI_BASE_URL`, default `https://api.openai.com/v1`). `NVIDIA_API_KEY` alone (no
`OPENAI_API_KEY`) selects the alternative NVIDIA inference API (`NVIDIA_MODEL`, `NVIDIA_BASE_URL`);
when both keys are set, `OPENAI_API_KEY` wins. Both send the same requests; the recorded
`runtime_source` names the endpoint and protocol (for example `openai-responses-api`), and the
endpoint is also the Provider-arm identity: records, single-run results and batch summaries carry
`provider: openai` or `provider: nvidia` (`resolve_provider_endpoint()` reads it from `.env`
without credentials and reports `openai` when no key is configured; the Codex arm stays
`codex-cli`). CLI overrides may select model, reasoning, wire protocol and an
optional per-request deadline; the API key is never a CLI argument. The per-request byte limit is
`USDCRAFT_MAX_REQUEST_BYTES` in `.env` (legacy alias `NVIDIA_MAX_REQUEST_BYTES`, used only when the
new name is absent).

Time and turn limits are Provider-only; the Codex arm has no time or turn limit. A single Provider
item has no run clock: its bound is the harness turn budget (Generation 64 turns; Reconstruction
`--max-turns`, default 80), including in-session inspection and repair. Batch `--run-timeout`
bounds each row's total time in a supervised worker process (default 1920 s, `0` for no limit;
see Batch below).
The per-request deadline (`--timeout`, default 900 s) is a transport hang guard: it covers
one entire HTTP attempt — connection, headers, and the complete success or error body — and a
timed-out attempt is retried from scratch up to `max_retries` more times before the request fails.
One attempt is one whole model turn, so the deadline must stay above the longest plausible turn;
`timeout_seconds=None` disables it programmatically. Retry waits are bounded and cooperative
cancellation can stop an active default HTTP wait. Arbitrary provider error bodies are not copied
into retained exceptions or traces; only an allowlisted failure category, HTTP status, and bounded
request-size metadata may be retained.

Provider requires a positive compile-worker timeout, and compile workers always use the `spawn`
start method. This keeps generated `model.py` out of the credential-holding process and its
memory, but it is a process boundary rather than an OS security sandbox.
Generated code remains trusted local code. The user-facing trusted-code note is in
[`../../../README.md`](../../../README.md#where-results-are-stored), and environment names/defaults
are in [`../../../.env.example`](../../../.env.example).

## Final Host and persistence

Provider's authoring `compile_model` tool always uses `render=False`. After the harness returns
`complete`, [`../../generation/provider_command.py`](../../generation/provider_command.py) independently invokes the common Host:

```text
compile_model(
    staged_model,
    staged_build,
    clean=True,
    render=False,
    require_generation_root_floating=True,
)
```

It then calls [`../../generation/persist.py`](../../generation/persist.py), which uses the same artifact contract and Storage
transaction as Codex, differing only in persisted authoring identity and trace filenames. Provider
progress callbacks, token usage, and harness completion do not grant promotion authority.

See [`../../compile/README.md`](../../compile/README.md) and
[`../../../storage/README.md`](../../../storage/README.md) for the exact mechanical and canonical
contracts.

## Batch

`usdcraft generation-batch` is the Provider batch command (`generation-codex-batch` runs the same CSV
contract on the Codex arm; see the [Generation guide](../../generation/README.md)). The CSV requires `row_id`, `prompt`, and `image_path`;
`label`, a paired `model`/`reasoning` override and `api_style` are optional, and any other column is
rejected with the list of allowed columns (the same rule as the Reconstruction CSV). A row may contain
text, one image, or both; relative image paths resolve from the CSV directory. `--concurrency` sets
the number of concurrent rows (1-32, default 1); `--model`, `--reasoning`, `--timeout` and
`--api-style` are row defaults, and `--keep-workspace` retains successful row workspaces.
`--validate-only` checks the CSV contract only, with no runtime resolution, model call or write.
`--run-timeout` (default 1920 s) runs each row in the shared supervised worker
([`../deadline.py`](../deadline.py)) with a total per-row deadline, exactly as for
`reconstruction-batch`; `--run-timeout 0` removes the limit and runs rows in the calling process.

[`load_provider_batch()`](../../generation/provider_batch.py) validates the Generation CSV shape and columns
once, on top of the shared CSV reader (duplicate headers, required columns, row identity, path
resolution) in [`usdcraft/pipeline/batch.py`](../batch.py). Prompt/image semantic
validity remains inside the same direct primitive used by single-item execution.

[`execute_provider_batch()`](../../generation/provider_batch.py) contributes only the row executor, which calls
`execute_provider_generation(...)` directly, so Batch cannot acquire a different prompt, tools,
Host, or promotion path. The bounded thread pool, fixed slots, per-row failure isolation,
CSV-ordered results, progress events and the Ctrl-C snapshot live in the product-neutral
coordinator `run_provider_batch()` in `usdcraft/pipeline/batch.py`; `usdcraft reconstruction-batch`
(`usdcraft/reconstruction/provider_batch.py`) reuses that coordinator with its own row contract
(its optional `mesh_cleanup` column, `true`/`false`/`1`/`0`, default off, selects the
Subject-Isolation Agent per row on both `reconstruction-batch` and `reconstruction-codex-batch`; see
the [Reconstruction CSV](../../reconstruction/README.md#reconstruction-csv)).
Rows own independent clients, histories, workspaces, traces, and canonical record identities.
Worker-local failures are recorded per row, other rows continue, and the result preserves original
CSV order. Successful rows remain canonical even when siblings fail.

Batch has no per-row or batch-level preflight/fingerprint exchange. Concurrency is local scheduling,
not shared Author context or multi-agent review.

On Ctrl-C, the coordinator stops new submissions, cancels pending futures, sets one cooperative
cancellation event for active items, reports completed partial results plus canceled/in-flight row
IDs, and exits with status 130. The default HTTP client can cancel an active request; Python cannot
safely kill a running worker thread, so a tool, Host compile, or promotion stops or completes only at
its next safe boundary.

## Progress presentation

Single-item and Batch displays subscribe to sanitized, fail-open callbacks. They may show runtime
identity, turn/row state, elapsed time, tool name/status, token counts, Host phase, and run/record
identity. They must not show prompt/image content, reasoning, tool arguments/results, API keys, or
provider error bodies.

Interactive TTY rendering, plain non-TTY/CI/JSON fallback, renderer threads, and terminal writes are
presentation only. A display failure cannot alter authoring, validation, cancellation semantics, or
promotion, and JSON stdout remains one clean result object.

Owning modules are [`../single_display.py`](../single_display.py),
[`../batch_display.py`](../batch_display.py),
[`../display_common.py`](../display_common.py), and
[`../batch_progress.py`](../batch_progress.py).

## Source map

| Concern | Canonical implementation |
| --- | --- |
| Direct Provider orchestration | [`../../generation/provider_command.py`](../../generation/provider_command.py) |
| Provider request/runtime contract | [`../../generation/provider_contract.py`](../../generation/provider_contract.py) |
| Provider (OpenAI or NVIDIA API) configuration and transport | [`client.py`](client.py) |
| Local full-history loop | [`harness.py`](harness.py) |
| Bounded tool surface and freshness | [`../tools.py`](../tools.py) |
| Progress callback normalization | [`progress.py`](progress.py) |
| Batch CSV and scheduler | [`../../generation/provider_batch.py`](../../generation/provider_batch.py), [`../batch.py`](../batch.py) |
| Per-row deadline supervisor | [`../deadline.py`](../deadline.py) |
| Batch progress event contract | [`../batch_progress.py`](../batch_progress.py) |
| Single/Batch display | [`../single_display.py`](../single_display.py), [`../batch_display.py`](../batch_display.py) |
| Shared Generation task | [`../../generation/prompt.py`](../../generation/prompt.py) |
| Shared final Host | [`../../compile/command.py`](../../compile/command.py) |
| Shared persistence | [`../../generation/persist.py`](../../generation/persist.py) |

## Tests

Provider-focused coverage lives under
[`../../../tests/generation/`](../../../tests/generation/):

- `test_provider_command.py` covers the direct primitive and common final Host;
- `provider/test_client.py` covers configuration, request limits, timeout, retry, cancellation, and
  safe error diagnostics;
- `provider/test_harness.py` covers local history, tool turns, finish, tracing, and cancellation;
- `provider/test_tools.py` covers path bounds, patching, SDK search, check/compile freshness, and
  completion;
- `test_provider_batch.py` covers CSV validation, same-primitive execution, ordering, failure
  isolation, concurrency, and interruption;
- `test_provider_single_display.py` and `test_provider_batch_display.py` cover sanitized fail-open
  presentation; and
- `test_prompt.py` checks that Codex and Provider use the same policy body with only the intended
  identity/reference-access adaptation.

Run the focused suite after Provider-specific changes:

```bash
uv run pytest -q \
  tests/generation/test_provider_command.py \
  tests/generation/test_provider_batch.py \
  tests/generation/test_provider_single_display.py \
  tests/generation/test_provider_batch_display.py \
  tests/generation/provider
uv run ruff check usdcraft/pipeline usdcraft/generation/provider_*.py tests/generation
```

Run the whole Generation suite (`uv run pytest -q tests/generation`) after changes that touch shared
task, tools, Host, or persistence behavior.
