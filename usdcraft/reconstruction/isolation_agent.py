# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""One explicit clean-context run for Reconstruction subject isolation (optional mesh cleanup).

The runtime Agent sees exactly two isolation-only contact sheets, one path-free component inventory,
and the exact task/schema staged in a fresh non-git workspace.  It never receives the semantic
image, AXIAL, source mesh, SDK, prior attempts, candidate output, Viewer state, evaluator
feedback, or the later Reconstruction Agent task.  Filesystem isolation is an evidence protocol,
not an OS security boundary; the prompt and retained context manifest make the allowlist auditable.

Both execution arms share this module's task text, evidence rendering, decision schema, decision
validation and published manifest; only the transport differs. :func:`run_subject_isolation` runs
the Agent through one ``codex exec`` process; the Provider arm supplies its Provider transport from
:mod:`usdcraft.reconstruction.provider_isolation` through :func:`run_subject_isolation_with_transport`.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from usdcraft.pipeline.codex import (
    CLEAN_CONTEXT_DISABLED_CODEX_FEATURES,
    CODEX_PROJECT_DOC_CONFIG,
)
from usdcraft.pipeline.hashing import sha256_file as _sha256
from usdcraft.pipeline.runtime import current_main_codex_runtime
from usdcraft.reconstruction.isolation import (
    SUBJECT_ISOLATION_SCHEMA,
    SubjectIsolationDecision,
    SubjectIsolationError,
    apply_subject_isolation,
    build_subject_isolation_inventory,
    isolation_selection_lookup,
    normalized_scene_identity_payload,
    normalized_scene_identity_sha256,
)
from usdcraft.reconstruction.isolation_render import (
    IsolationRenderEvidence,
    render_subject_isolation_evidence,
)
from usdcraft.reconstruction.orientation import BundleBuilder, SessionFactory
from usdcraft.reconstruction.source import (
    SourceInput,
    load_normalized_source,
    write_normalized_source,
)
from usdcraft.values import normalize_reasoning_level

ISOLATION_AGENT_RESULT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["complete", "blocked"]},
        "decisions": {
            "type": "array",
            "maxItems": 512,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "selection_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "disposition": {
                        "type": "string",
                        "enum": ["exclude_background_or_incidental", "retain_unresolved"],
                    },
                    "reason": {"type": "string", "minLength": 1, "maxLength": 800},
                    "visual_evidence": {"type": "string", "minLength": 1, "maxLength": 800},
                },
                "required": ["selection_id", "disposition", "reason", "visual_evidence"],
            },
        },
        "summary": {"type": "string", "minLength": 1, "maxLength": 1600},
    },
    "required": ["status", "decisions", "summary"],
}
ISOLATION_DISABLED_CODEX_FEATURES: tuple[str, ...] = CLEAN_CONTEXT_DISABLED_CODEX_FEATURES
CODEX_RUN_DIRNAME = ".codex-run"


class SubjectIsolationAgentError(SubjectIsolationError):
    """The clean-context isolation Agent failed or returned an unsafe decision."""


@dataclass(frozen=True, slots=True)
class IsolationRuntimeSelection:
    model_id: str
    reasoning_effort: str
    runtime_source: str
    selection_source: str


@dataclass(frozen=True, slots=True)
class IsolationProcessResult:
    returncode: int
    stdout: str
    stderr: str


class IsolationProcessRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        stdin_text: str,
        cwd: Path,
        env: Mapping[str, str],
    ) -> IsolationProcessResult: ...


@dataclass(frozen=True, slots=True)
class SubjectIsolationArtifacts:
    root: Path
    status: str
    manifest: Path
    inventory: Path
    native_contact_sheet: Path
    component_id_contact_sheet: Path
    retained_source: Path
    retained_source_manifest: Path


def _normalized_nonempty(value: object, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{label} must not contain control characters")
    return normalized


def resolve_isolation_runtime(
    *,
    model: str | None,
    reasoning_effort: str | None,
    environ: Mapping[str, str] | None = None,
) -> IsolationRuntimeSelection:
    """Resolve the one isolation subagent runtime without adopting Generation semantics."""

    if (model is None) != (reasoning_effort is None):
        raise ValueError("--model and --reasoning must be provided together")
    if model is not None and reasoning_effort is not None:
        return IsolationRuntimeSelection(
            model_id=_normalized_nonempty(model, label="Codex model"),
            reasoning_effort=normalize_reasoning_level(
                _normalized_nonempty(reasoning_effort, label="Codex reasoning effort")
            ),
            runtime_source="explicit_reconstruction_isolation_request",
            selection_source="explicit_reconstruction_isolation_request",
        )
    inherited = current_main_codex_runtime(environ=environ)
    return IsolationRuntimeSelection(
        model_id=inherited.model_id,
        reasoning_effort=inherited.reasoning_effort,
        runtime_source=inherited.runtime_source,
        selection_source="inherited_root_for_reconstruction_isolation",
    )


def build_subject_isolation_task(
    *,
    inventory_sha256: str,
    agent_name: str = "Codex",
) -> str:
    """Build the complete task for the isolation-only clean-context subagent.

    ``agent_name`` names the transport in the agent's self-description (``"Codex"`` for the Codex
    arm; empty for the Provider arm, whose agent is simply the Subject-Isolation Agent), exactly as
    the image-observation task does. Every other word is shared by both arms.
    """

    agent = f"{agent_name.strip()} " if agent_name.strip() else ""
    return f"""You are the one clean-context {agent}Subject-Isolation Agent for a USDCraft Reconstruction run.

Your only job is to decide which already-authored complete source-surface components belong to the
primary subject and which are clearly background, an incidental support surface, scene dressing, or
detached reconstruction flypoints. You are not the later Reconstruction Agent and you must not
author, repair, simplify, rotate, recenter, rescale, name Parts, infer joints, or infer a solid.

ALLOWED EVIDENCE
- `component_inventory.json` (expected inventory sha256: `{inventory_sha256}`)
- the attached `native_contact_sheet.png`
- the attached `component_id_contact_sheet.png`
- this task and the final response schema

No semantic input image is provided. The native-appearance sheet is not a semantic-image channel;
it and the component-ID sheet are two presentations of the same raw source surfaces, solely for
subject isolation. Do not inspect any source/normalized mesh, AXIAL, voxel data, SDK, Recon
authoring task, prior attempt, candidate result, Viewer/evaluator feedback, GT code, repository,
network source, or other filesystem path. Do not delegate or start another agent.

SURFACE AND SELECTION CONTRACT
- Every mesh/point component is surface-only evidence. Closed topology never proves material fill,
  inside/outside, free space, wall thickness, cavity contents, or trusted solidity.
- You may select only exact `selection_id` values from the inventory. Every selection expands to
  complete components. You cannot clip triangles/points, erase a patch inside one component, fill a
  hole, repair geometry, or create new geometry.
- Keep the complete coherent primary subject, including small, thin, detached, suspended, or
  disconnected-looking pieces that plausibly belong to it. Size alone, triangle count alone, open
  topology, weak contact, or being separated from the largest component is never enough to exclude.
- A strict empty-band group is only a convenient component set. The gap does not prove empty space,
  background, or subject identity. Inspect both sheets and the inventory before using the group.
- Exclude a selection only with high-confidence evidence that all expanded components are outside
  the primary subject: for example a table/ground/background fragment, unrelated scene dressing, or
  a far detached flypoint cloud. If a group mixes subject and background, do not exclude the group;
  use exact component selections only when those are clear.
- Unspecified components are retained. Use `retain_unresolved` for a suspected contaminant that is
  not safe to exclude. Unresolved selections are also retained. Conservative retention is correct;
  never force an exclusion merely to make the mesh look clean.
- Do not use a category guess to idealize the source or remove an unusual but coherent subject
  structure. This stage answers only subject versus non-subject, not whether subject geometry is a
  later local reconstruction artifact.

OUTPUT
Return only the schema-conforming JSON response. Each decision must use one exact inventory
`selection_id`, one disposition, a concrete reason, and the multiview evidence supporting it.
Do not repeat the same selection id. Use `status: complete` even when the safe result excludes
nothing. Use `status: blocked` only when the evidence itself cannot be read or no primary subject
can be identified; a blocked result must not exclude anything.
"""


def build_isolation_codex_command(
    runtime: IsolationRuntimeSelection,
    workspace: Path,
    *,
    executable: str = "codex",
) -> tuple[str, ...]:
    schema = workspace / CODEX_RUN_DIRNAME / "final.schema.json"
    final = workspace / CODEX_RUN_DIRNAME / "final.json"
    native = workspace / "native_contact_sheet.png"
    component_id = workspace / "component_id_contact_sheet.png"
    for path in (schema, native, component_id):
        if not path.is_file():
            raise SubjectIsolationAgentError(f"isolation Agent workspace is missing {path.name}")
    command = [
        executable,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--json",
    ]
    for feature in ISOLATION_DISABLED_CODEX_FEATURES:
        command.extend(("--disable", feature))
    command.extend(
        (
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(final),
            "-C",
            str(workspace),
            "--model",
            runtime.model_id,
            "-c",
            f"model_reasoning_effort={json.dumps(runtime.reasoning_effort, ensure_ascii=False)}",
            "-c",
            CODEX_PROJECT_DOC_CONFIG,
            "--image",
            str(native),
            "--image",
            str(component_id),
            "-",
        )
    )
    return tuple(command)


def _default_process_runner(
    command: Sequence[str],
    *,
    stdin_text: str,
    cwd: Path,
    env: Mapping[str, str],
) -> IsolationProcessResult:
    """Run one Codex child to completion; the Codex arm has no time limit."""

    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=dict(env),
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(stdin_text)
    except BaseException:
        # The child runs in its own session, so an interrupt never reaches it: stop the group.
        with suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
        with suppress(OSError):
            process.kill()
        process.communicate()
        raise
    return IsolationProcessResult(
        returncode=int(process.returncode),
        stdout=stdout,
        stderr=stderr,
    )


@dataclass(frozen=True, slots=True)
class IsolationTransport:
    """The transport-specific surface of one subject-isolation Agent run.

    The shared core stages the same workspace, builds the same task, validates the same decision and
    publishes the same manifest for every transport. A transport contributes only how the staged
    task and attached evidence reach the model and how the final response returns:

    - ``invoke(workspace, task_text)`` must write the Agent's final response verbatim to
      ``<workspace>/<run_dir_name>/final.json`` (or raise :class:`SubjectIsolationAgentError`) and
      returns its transport log artifacts as ``{manifest_key: workspace-relative path}``;
    - ``context_fields`` / ``agent_fields`` are the transport facts recorded in the retained context
      manifest and in the manifest's ``agent`` block.
    """

    agent_name: str
    run_dir_name: str
    response_label: str
    runtime: IsolationRuntimeSelection
    invoke: Callable[[Path, str], Mapping[str, str]]
    context_fields: Mapping[str, object] = field(default_factory=dict)
    agent_fields: Mapping[str, object] = field(default_factory=dict)


def _write_agent_workspace(
    workspace: Path,
    *,
    inventory_path: Path,
    evidence: IsolationRenderEvidence,
    agent_name: str,
    run_dir_name: str,
) -> Path:
    if not workspace.is_dir() or any(workspace.iterdir()):
        raise SubjectIsolationAgentError("isolation Agent workspace must be a new empty directory")
    run_dir = workspace / run_dir_name
    run_dir.mkdir()
    staged_inventory = workspace / "component_inventory.json"
    native = workspace / "native_contact_sheet.png"
    component_id = workspace / "component_id_contact_sheet.png"
    shutil.copy2(inventory_path, staged_inventory)
    shutil.copy2(evidence.native_contact_sheet, native)
    shutil.copy2(evidence.component_id_contact_sheet, component_id)
    task = build_subject_isolation_task(
        inventory_sha256=_sha256(staged_inventory),
        agent_name=agent_name,
    )
    (workspace / "TASK.md").write_text(task.rstrip() + "\n", encoding="utf-8")
    (run_dir / "final.schema.json").write_text(
        json.dumps(ISOLATION_AGENT_RESULT_SCHEMA, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return workspace / "TASK.md"


def _agent_context_payload(transport: IsolationTransport) -> dict[str, object]:
    return {
        "schema": "reconstruction_subject_isolation_agent_context",
        "context": "clean_subject_isolation_only",
        "allowed_evidence": [
            "TASK.md",
            "component_inventory.json",
            "native_contact_sheet.png",
            "component_id_contact_sheet.png",
            f"{transport.run_dir_name}/final.schema.json",
        ],
        "attached_images": [
            "native_contact_sheet.png",
            "component_id_contact_sheet.png",
        ],
        "semantic_image": False,
        "source_mesh": False,
        "axial": False,
        "sdk": False,
        "prior_attempts": False,
        "candidate_or_evaluator_feedback": False,
        "reconstruction_agent_input": False,
        "filesystem_isolation": "evidence_protocol_not_os_security_boundary",
        **dict(transport.context_fields),
    }


def _parse_agent_result(path: Path, *, label: str = "Codex") -> dict[str, object]:
    if not path.is_file():
        raise SubjectIsolationAgentError(
            f"{label} isolation Agent did not write its final response"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubjectIsolationAgentError(
            f"{label} isolation Agent response is invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise SubjectIsolationAgentError(f"{label} isolation Agent response must be a JSON object")
    if set(payload) != {"status", "decisions", "summary"}:
        raise SubjectIsolationAgentError(f"{label} isolation Agent response has unexpected fields")
    if payload.get("status") not in {"complete", "blocked"}:
        raise SubjectIsolationAgentError(f"{label} isolation Agent status is invalid")
    if not isinstance(payload.get("summary"), str) or not str(payload["summary"]).strip():
        raise SubjectIsolationAgentError(f"{label} isolation Agent summary is missing")
    if not isinstance(payload.get("decisions"), list):
        raise SubjectIsolationAgentError(f"{label} isolation Agent decisions must be an array")
    return payload


def _expanded_selection(
    *,
    scene_component_ids: tuple[str, ...],
    inventory: Mapping[str, object],
    result: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], list[dict[str, object]]]:
    lookup = isolation_selection_lookup(inventory)
    raw_decisions = result.get("decisions")
    assert isinstance(raw_decisions, list)
    seen: set[str] = set()
    excluded: set[str] = set()
    unresolved: set[str] = set()
    audited: list[dict[str, object]] = []
    for raw in raw_decisions:
        if not isinstance(raw, Mapping):
            raise SubjectIsolationAgentError("isolation Agent decision must be an object")
        if set(raw) != {"selection_id", "disposition", "reason", "visual_evidence"}:
            raise SubjectIsolationAgentError("isolation Agent decision has unexpected fields")
        selection_id = raw.get("selection_id")
        disposition = raw.get("disposition")
        reason = raw.get("reason")
        evidence = raw.get("visual_evidence")
        if not isinstance(selection_id, str) or selection_id not in lookup:
            raise SubjectIsolationAgentError(
                f"isolation Agent used unknown selection id {selection_id!r}"
            )
        if selection_id in seen:
            raise SubjectIsolationAgentError(
                f"isolation Agent repeated selection id {selection_id!r}"
            )
        seen.add(selection_id)
        if disposition not in {"exclude_background_or_incidental", "retain_unresolved"}:
            raise SubjectIsolationAgentError(
                f"isolation Agent used invalid disposition {disposition!r}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise SubjectIsolationAgentError("isolation Agent decision reason is missing")
        if not isinstance(evidence, str) or not evidence.strip():
            raise SubjectIsolationAgentError("isolation Agent decision visual evidence is missing")
        expanded = set(lookup[selection_id])
        if disposition == "exclude_background_or_incidental":
            excluded.update(expanded)
        else:
            unresolved.update(expanded)
        audited.append(
            {
                "selection_id": selection_id,
                "disposition": disposition,
                "expanded_component_ids": list(lookup[selection_id]),
                "reason": reason.strip(),
                "visual_evidence": evidence.strip(),
            }
        )
    if result.get("status") == "blocked" and excluded:
        raise SubjectIsolationAgentError(
            "a blocked isolation Agent result must not exclude surfaces"
        )
    overlap = excluded & unresolved
    if overlap:
        raise SubjectIsolationAgentError(
            "isolation Agent selections conflict between exclusion and unresolved retention: "
            + ", ".join(sorted(overlap))
        )
    if result.get("status") == "blocked":
        unresolved.update(scene_component_ids)
    retained = tuple(
        component_id for component_id in scene_component_ids if component_id not in excluded
    )
    excluded_ordered = tuple(
        component_id for component_id in scene_component_ids if component_id in excluded
    )
    unresolved_ordered = tuple(
        component_id for component_id in scene_component_ids if component_id in unresolved
    )
    if not retained:
        raise SubjectIsolationAgentError("isolation Agent attempted to exclude the entire source")
    return retained, excluded_ordered, unresolved_ordered, audited


def _artifact(path: Path, *, root: Path, isolation_agent_input: bool) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
        "isolation_agent_input": isolation_agent_input,
        "reconstruction_agent_input": False,
    }


def run_subject_isolation_with_transport(
    source_input: SourceInput,
    output_dir: Path,
    *,
    transport: IsolationTransport,
    bundle_builder: BundleBuilder | None = None,
    session_factory: SessionFactory | None = None,
) -> SubjectIsolationArtifacts:
    """Run and freeze one clean-context subject-isolation decision through ``transport``.

    The inventory, both contact sheets, the task text, the decision schema, the decision
    validation and the published manifest are identical on every transport.
    """

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise SubjectIsolationError(f"output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = transport.runtime
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    agent_workspace = Path(tempfile.mkdtemp(prefix="usdcraft-recon-isolation-agent-"))
    started = time.monotonic()
    try:
        scene = load_normalized_source(source_input)
        source_identity = normalized_scene_identity_sha256(scene)
        inventory = build_subject_isolation_inventory(scene)
        inventory_path = staging / "component_inventory.json"
        inventory_path.write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evidence = render_subject_isolation_evidence(
            scene,
            inventory,
            staging / "evidence",
            bundle_builder=bundle_builder,
            session_factory=session_factory,
        )
        task_path = _write_agent_workspace(
            agent_workspace,
            inventory_path=inventory_path,
            evidence=evidence,
            agent_name=transport.agent_name,
            run_dir_name=transport.run_dir_name,
        )
        task_text = task_path.read_text(encoding="utf-8")
        log_artifacts = dict(transport.invoke(agent_workspace, task_text))
        result = _parse_agent_result(
            agent_workspace / transport.run_dir_name / "final.json",
            label=transport.response_label,
        )
        component_ids = tuple(component.component_id for component in scene.components)
        retained, excluded, unresolved, audited_decisions = _expanded_selection(
            scene_component_ids=component_ids,
            inventory=inventory,
            result=result,
        )
        (agent_workspace / "context_manifest.json").write_text(
            json.dumps(
                _agent_context_payload(transport),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        agent_audit = staging / "agent"
        shutil.copytree(agent_workspace, agent_audit)
        status = "ready" if result["status"] == "complete" else "blocked"
        provisional = SubjectIsolationDecision(
            manifest_path=staging / "subject_isolation.json",
            source_identity_sha256=source_identity,
            retained_component_ids=retained,
            excluded_component_ids=excluded,
            unresolved_component_ids=unresolved,
            upstream_agent_run=True,
            upstream_agent_context="clean_subject_isolation_only",
        )
        retained_scene = apply_subject_isolation(scene, provisional)
        retained_artifacts = write_normalized_source(retained_scene, staging / "retained_source")
        elapsed = time.monotonic() - started
        manifest_path = staging / "subject_isolation.json"
        artifacts = {
            "inventory": _artifact(inventory_path, root=staging, isolation_agent_input=True),
            "native_contact_sheet": _artifact(
                evidence.native_contact_sheet,
                root=staging,
                isolation_agent_input=True,
            ),
            "component_id_contact_sheet": _artifact(
                evidence.component_id_contact_sheet,
                root=staging,
                isolation_agent_input=True,
            ),
            "native_render_manifest": _artifact(
                evidence.native_render_manifest,
                root=staging,
                isolation_agent_input=False,
            ),
            "component_id_render_manifest": _artifact(
                evidence.component_id_render_manifest,
                root=staging,
                isolation_agent_input=False,
            ),
            "agent_task": _artifact(
                agent_audit / "TASK.md", root=staging, isolation_agent_input=True
            ),
            "agent_context_manifest": _artifact(
                agent_audit / "context_manifest.json",
                root=staging,
                isolation_agent_input=False,
            ),
            "agent_final": _artifact(
                agent_audit / transport.run_dir_name / "final.json",
                root=staging,
                isolation_agent_input=False,
            ),
            **{
                key: _artifact(agent_audit / relative, root=staging, isolation_agent_input=False)
                for key, relative in log_artifacts.items()
            },
            "retained_source": _artifact(
                retained_artifacts.normalized_usdc,
                root=staging,
                isolation_agent_input=False,
            ),
            "retained_source_manifest": _artifact(
                retained_artifacts.source_manifest,
                root=staging,
                isolation_agent_input=False,
            ),
        }
        for optional_name in ("timeline.jsonl", "trace_errors.log"):
            optional_path = agent_audit / transport.run_dir_name / optional_name
            if optional_path.is_file():
                artifacts[f"agent_{optional_name.replace('.', '_')}"] = _artifact(
                    optional_path,
                    root=staging,
                    isolation_agent_input=False,
                )
        manifest = {
            "schema": SUBJECT_ISOLATION_SCHEMA,
            "status": status,
            "scope": "reconstruction_subject_isolation_preprocessing",
            "canonical_record": False,
            "source_identity_sha256": source_identity,
            "source_identity": normalized_scene_identity_payload(scene),
            "processing_order": [
                "lossless_declared_frame_decode",
                "subject_isolation",
                "subject_rotation_recenter_or_scale_if_later_declared",
                "normalized_source_snapshot",
                "axial_and_source_overlay",
            ],
            "selection": {
                "retained_component_ids": list(retained),
                "excluded_component_ids": list(excluded),
                "unresolved_component_ids": list(unresolved),
                "retained_component_count": len(retained),
                "excluded_component_count": len(excluded),
                "unresolved_component_count": len(unresolved),
                "decision_units": audited_decisions,
                "retained_samples_unchanged": True,
                "triangle_or_point_clipping": False,
                "repair_fill_or_inferred_surface": False,
                "excluded_surface_semantics": "not_subject_not_proven_empty_space",
            },
            "agent": {
                "run": True,
                "context": "clean_subject_isolation_only",
                **dict(transport.agent_fields),
                "model": runtime.model_id,
                "reasoning_effort": runtime.reasoning_effort,
                "runtime_source": runtime.runtime_source,
                "selection_source": runtime.selection_source,
                "attached_image_count": 2,
                "semantic_image": False,
                "axial": False,
                "source_mesh": False,
                "sdk": False,
                "later_reconstruction_context": False,
                "outcome": result["status"],
                "summary": str(result["summary"]).strip(),
                "elapsed_seconds": elapsed,
            },
            "downstream_contract": {
                "reconstruction_agent_input": False,
                "recon_observe_requires_ready_frozen_decision": True,
                "same_retained_scene_for_normalized_source_axial_renders_and_evaluation": True,
                "render_only_or_axial_only_filtering": False,
            },
            "artifacts": artifacts,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(agent_workspace, ignore_errors=True)

    return SubjectIsolationArtifacts(
        root=output,
        status=status,
        manifest=output / "subject_isolation.json",
        inventory=output / "component_inventory.json",
        native_contact_sheet=output / "evidence" / "native_contact_sheet.png",
        component_id_contact_sheet=output / "evidence" / "component_id_contact_sheet.png",
        retained_source=output / "retained_source" / "normalized.usdc",
        retained_source_manifest=output / "retained_source" / "source_manifest.json",
    )


def run_subject_isolation(
    source_input: SourceInput,
    output_dir: Path,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    executable: str | None = None,
    environ: Mapping[str, str] | None = None,
    bundle_builder: BundleBuilder | None = None,
    session_factory: SessionFactory | None = None,
    process_runner: IsolationProcessRunner = _default_process_runner,
) -> SubjectIsolationArtifacts:
    """Run and freeze one explicit clean-context subject-isolation decision through Codex.

    The Codex arm has no time limit: the Agent process runs until it exits.
    """

    runtime = resolve_isolation_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        environ=environ,
    )
    selected_environ = os.environ if environ is None else environ

    def invoke(workspace: Path, task_text: str) -> Mapping[str, str]:
        command = build_isolation_codex_command(
            runtime,
            workspace,
            executable=(
                executable or selected_environ.get("USDCRAFT_CODEX_BIN", "").strip() or "codex"
            ),
        )
        process_env = dict(selected_environ)
        process_env["PYTHONUNBUFFERED"] = "1"
        try:
            process = process_runner(
                command,
                stdin_text=task_text,
                cwd=workspace,
                env=process_env,
            )
        except FileNotFoundError as exc:
            raise SubjectIsolationAgentError(
                "subject isolation requires the `codex` executable; install/login to Codex CLI "
                "or set USDCRAFT_CODEX_BIN"
            ) from exc
        codex_dir = workspace / CODEX_RUN_DIRNAME
        (codex_dir / "stdout.jsonl").write_text(process.stdout, encoding="utf-8")
        (codex_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0:
            details = (process.stderr or process.stdout or "(no output)").strip()
            raise SubjectIsolationAgentError(
                f"subject-isolation Agent exited with status {process.returncode}: "
                f"{details[-4000:]}"
            )
        return {
            "agent_stdout": f"{CODEX_RUN_DIRNAME}/stdout.jsonl",
            "agent_stderr": f"{CODEX_RUN_DIRNAME}/stderr.log",
        }

    codex_context: dict[str, object] = {
        "user_config_loaded": False,
        "project_documents_loaded": False,
        "disabled_codex_features": list(ISOLATION_DISABLED_CODEX_FEATURES),
    }
    return run_subject_isolation_with_transport(
        source_input,
        output_dir,
        transport=IsolationTransport(
            agent_name="Codex",
            run_dir_name=CODEX_RUN_DIRNAME,
            response_label="Codex",
            runtime=runtime,
            invoke=invoke,
            context_fields=codex_context,
            agent_fields={"ephemeral": True, "sandbox": "read-only", **codex_context},
        ),
        bundle_builder=bundle_builder,
        session_factory=session_factory,
    )


__all__ = [
    "CODEX_RUN_DIRNAME",
    "ISOLATION_AGENT_RESULT_SCHEMA",
    "IsolationProcessResult",
    "IsolationProcessRunner",
    "IsolationRuntimeSelection",
    "IsolationTransport",
    "SubjectIsolationAgentError",
    "SubjectIsolationArtifacts",
    "build_isolation_codex_command",
    "build_subject_isolation_task",
    "resolve_isolation_runtime",
    "run_subject_isolation",
    "run_subject_isolation_with_transport",
]
