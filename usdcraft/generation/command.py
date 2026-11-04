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

from __future__ import annotations

import hmac
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from storage.record_mutations import recover_pending_promotions
from storage.repo import StorageRepo
from usdcraft._images import validate_decodable_image
from usdcraft.compile import compile_model
from usdcraft.generation.codex import run_codex
from usdcraft.generation.contract import GenerationPreflight, GenerationRequest
from usdcraft.generation.default_prompt import resolve_generation_prompt
from usdcraft.generation.observation import (
    ImageObservationError,
    render_inventory_for_prompt,
    run_generation_image_observation,
)
from usdcraft.generation.persist import REVISION_ID, persist_generation
from usdcraft.generation.prompt import build_generation_prompt
from usdcraft.generation.workspace import GenerationWorkspace, create_generation_workspace
from usdcraft.pipeline.contracts import CodexCompletionStatus
from usdcraft.pipeline.execution import compile_author_output
from usdcraft.pipeline.hashing import sha256_bytes, sha256_file
from usdcraft.pipeline.runtime import _codex_executable, resolve_codex_runtime
from usdcraft.pipeline.sdk_context import PRELOADED_SDK_DOC_PATHS
from usdcraft.stagetime import STAGE_TRACE_FILENAME, StageTimer


class GenerationBlockedError(RuntimeError):
    def __init__(self, summary: str, workspace_dir: Path) -> None:
        super().__init__(f"Codex reported a blocker: {summary} (workspace: {workspace_dir})")
        self.summary = summary
        self.workspace_dir = workspace_dir


class GenerationApprovalError(RuntimeError):
    pass


class GenerationInputChangedError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class GenerationFailureContext:
    """Diagnostic identity for a failure that happened after workspace creation."""

    run_id: str
    workspace_dir: Path


@dataclass(slots=True, frozen=True)
class GenerationResult:
    status: str
    record_id: str
    record_dir: Path
    model_path: Path
    workspace_dir: Path | None
    warnings: tuple[str, ...]
    run_id: str | None = None
    # The arm's provider identity: ``codex-cli`` (Codex arm) or the Provider endpoint
    # (``nvidia`` or ``openai``).
    provider: str = "codex-cli"

    def to_dict(self) -> dict[str, Any]:
        """The single-item result keys shared with Reconstruction (both arms)."""

        return {
            "status": self.status,
            "pipeline": "generation",
            "provider": self.provider,
            "run_id": self.run_id,
            "record_id": self.record_id,
            "record_dir": str(self.record_dir),
            "model_path": str(self.model_path),
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir is not None else None,
            "warnings": list(self.warnings),
        }


def fingerprint_generation_preflight(preflight: GenerationPreflight) -> str:
    payload = json.dumps(
        preflight.fingerprint_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def preflight_generation(
    *,
    repo_root: Path,
    prompt: str | None = None,
    image: Path | None = None,
    model: str | None = None,
    reasoning: str | None = None,
    label: str | None = None,
    keep_workspace: bool = False,
    scaffold_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    codex_home: Path | None = None,
) -> GenerationPreflight:
    """Build one fully resolved, zero-write Generation plan for user approval."""

    repo_root = Path(repo_root).expanduser().resolve()
    resolved_image = Path(image).expanduser().resolve() if image is not None else None
    if resolved_image is not None:
        validate_decodable_image(resolved_image, label="Generation inspiration image")
    resolved_prompt = resolve_generation_prompt(
        prompt=prompt,
        image_path=resolved_image,
    )
    runtime = resolve_codex_runtime(
        model=model,
        reasoning_effort=reasoning,
        environ=environ,
        codex_home=codex_home,
    )
    request = GenerationRequest(
        prompt=resolved_prompt,
        repo_root=repo_root,
        image_path=resolved_image,
        model=runtime.model_id,
        reasoning_effort=runtime.reasoning_effort,
        runtime_source=runtime.runtime_source,
        selection_source=runtime.selection_source,
    )
    resolved_scaffold = Path(scaffold_path or repo_root / "scaffold.py").expanduser().resolve()
    if not resolved_scaffold.is_file():
        raise FileNotFoundError(f"Generation scaffold not found: {resolved_scaffold}")
    normalized_label = str(label).strip() if label is not None else None
    if normalized_label == "":
        normalized_label = None
    image_sha256 = sha256_file(request.image_path) if request.image_path is not None else None
    draft = GenerationPreflight(
        request=request,
        label=normalized_label,
        keep_workspace=bool(keep_workspace),
        scaffold_path=resolved_scaffold,
        scaffold_sha256=sha256_file(resolved_scaffold),
        image_sha256=image_sha256,
        codex_executable=_codex_executable(environ),
        python_executable=str(Path(sys.executable).expanduser().absolute()),
        fingerprint="",
    )
    return replace(draft, fingerprint=fingerprint_generation_preflight(draft))


def _verify_staged_inputs(
    preflight: GenerationPreflight,
    *,
    workspace: GenerationWorkspace,
) -> None:
    if sha256_file(workspace.model_path) != preflight.scaffold_sha256:
        raise GenerationInputChangedError(
            "Generation scaffold changed after approval; run preflight again."
        )
    if preflight.image_sha256 is None:
        if workspace.input_image_path is not None:
            raise GenerationInputChangedError(
                "Generation input mode changed after approval; run preflight again."
            )
        return
    if workspace.input_image_path is None or not workspace.input_image_path.is_file():
        raise GenerationInputChangedError(
            "Generation inspiration image disappeared after approval; run preflight again."
        )
    if sha256_file(workspace.input_image_path) != preflight.image_sha256:
        raise GenerationInputChangedError(
            "Generation inspiration image changed after approval; run preflight again."
        )


def _attach_failure_context(
    exc: BaseException,
    *,
    workspace: GenerationWorkspace,
) -> None:
    """Annotate an existing exception without replacing its type, cause, or traceback."""

    workspace_dir = workspace.root
    try:
        setattr(exc, "generation_run_id", workspace.run_id)
        setattr(exc, "generation_workspace_dir", workspace_dir)
    except (AttributeError, TypeError):
        # Exception instances normally allow diagnostic attributes. Keep a standard note as a
        # defensive fallback for a custom exception that deliberately rejects them.
        exc.add_note(f"USDCraft Generation run_id: {workspace.run_id}")
        exc.add_note(f"USDCraft Generation workspace_dir: {workspace_dir}")


def generation_failure_context(exc: BaseException) -> GenerationFailureContext | None:
    """Return post-workspace diagnostic context attached by :func:`execute_generation`."""

    run_id = getattr(exc, "generation_run_id", None)
    workspace_dir = getattr(exc, "generation_workspace_dir", None)
    if isinstance(run_id, str) and run_id and isinstance(workspace_dir, Path):
        return GenerationFailureContext(run_id=run_id, workspace_dir=workspace_dir)

    note_run_id: str | None = None
    note_workspace: Path | None = None
    for note in getattr(exc, "__notes__", ()):
        if not isinstance(note, str):
            continue
        if note.startswith("USDCraft Generation run_id: "):
            note_run_id = note.removeprefix("USDCraft Generation run_id: ").strip()
        elif note.startswith("USDCraft Generation workspace_dir: "):
            value = note.removeprefix("USDCraft Generation workspace_dir: ").strip()
            if value:
                note_workspace = Path(value)
    if note_run_id and note_workspace is not None:
        return GenerationFailureContext(run_id=note_run_id, workspace_dir=note_workspace)
    return None


def execute_generation(
    *,
    preflight: GenerationPreflight,
    approved_fingerprint: str | None = None,
    direct: bool = False,
    run_id: str | None = None,
    batch_context: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> GenerationResult:
    """Execute one confirmed or explicitly direct preflight and promote only validated output.

    ``direct=True`` is the user's explicit authorization of immediate execution after input
    validation (``generation-codex --direct`` and every ``generation-codex-batch`` row); it
    replaces the fingerprint exchange and cannot be combined with ``approved_fingerprint``.
    """

    if direct and approved_fingerprint is not None:
        raise GenerationApprovalError(
            "Use direct execution or a confirmation fingerprint, not both."
        )
    expected_fingerprint = fingerprint_generation_preflight(preflight)
    if not hmac.compare_digest(preflight.fingerprint, expected_fingerprint):
        raise GenerationApprovalError("Generation preflight is internally inconsistent.")
    candidate = expected_fingerprint if direct else str(approved_fingerprint or "").strip()
    if not hmac.compare_digest(candidate, expected_fingerprint):
        raise GenerationApprovalError(
            "Generation approval does not match the current preflight; run preflight again."
        )

    request = preflight.request
    # Roll back any promotion a previous crashed run left half-committed before starting.
    promotion_recovery = recover_pending_promotions(StorageRepo(request.repo_root))
    stage_timer = (
        StageTimer(pipeline="Generation Codex", log=progress)
        if progress is not None
        else StageTimer(pipeline="Generation Codex")
    )
    with stage_timer.stage("freeze"):
        workspace = create_generation_workspace(
            request,
            scaffold_path=preflight.scaffold_path,
            run_id=run_id,
        )
    stage_timer.attach_trace_path(workspace.root / STAGE_TRACE_FILENAME)
    try:
        try:
            _verify_staged_inputs(preflight, workspace=workspace)
        except BaseException:
            # An approval-input mismatch is the one post-creation failure whose untrusted staged
            # copy is removed. It still receives the run identity and absolute attempted path.
            shutil.rmtree(workspace.root, ignore_errors=True)
            raise

        # Image observation is one clean-context vision reading of the staged inspiration image.
        # It is an external first reading for the Author, never binding: a failed or blocked
        # observation degrades the run to the previous Author-only reading with a warning, and
        # text-only runs skip the stage entirely.
        image_observation = None
        observation_warnings: list[str] = []
        if workspace.input_image_path is not None:
            try:
                with stage_timer.stage("image_observation"):
                    image_observation = run_generation_image_observation(
                        image_path=workspace.input_image_path,
                        output_dir=workspace.root / "image_observation",
                        model=request.runtime.model_id,
                        reasoning_effort=request.runtime.reasoning_effort,
                        executable=preflight.codex_executable,
                    )
                shutil.copy2(
                    image_observation.inventory_path,
                    workspace.root / "image_observation.json",
                )
                shutil.copy2(
                    image_observation.manifest_path,
                    workspace.root / "image_observation_manifest.json",
                )
            except ImageObservationError as exc:
                observation_warnings.append(
                    f"Image observation degraded to the Author-only reading: {exc}"
                )

        # Use the exact Python environment that launched USDCraft. The Codex process inherits
        # PYTHONPATH from the runner, so this works from the isolated staging directory without
        # installing or copying another runtime.
        compile_command = (
            preflight.python_executable,
            "-m",
            "usdcraft.cli",
            "compile",
            str(workspace.model_path),
            "--output",
            str(workspace.build_dir),
            "--no-render",
        )
        check_command = (
            preflight.python_executable,
            "-m",
            "usdcraft.cli",
            "check",
            str(workspace.model_path),
        )
        task = build_generation_prompt(
            request,
            workspace,
            check_command=check_command,
            compile_command=compile_command,
            render_command=(preflight.python_executable, "-m", "usdcraft.cli", "generation-render"),
            image_observation_text=(
                render_inventory_for_prompt(image_observation.inventory)
                if image_observation is not None
                else None
            ),
        )
        with stage_timer.stage("author"):
            codex_result = run_codex(
                request,
                workspace,
                prompt=task,
                executable=preflight.codex_executable,
            )
        if codex_result.final_message.status is CodexCompletionStatus.BLOCKED:
            raise GenerationBlockedError(codex_result.final_message.summary, workspace.root)

        # Both the Agent's authoring compile and this final Host pass disable rendering. Agent-side
        # compile is repair feedback; this clean pass is the mechanical/structural promotion gate.
        artifacts = compile_author_output(workspace, stage_timer, compile_runner=compile_model)
        runtime_report = {
            "returncode": codex_result.returncode,
            "elapsed_seconds": round(codex_result.elapsed_seconds, 6),
            "final_status": codex_result.final_message.status.value,
            "final_summary": codex_result.final_message.summary,
            "stage_timings": stage_timer.report(),
            "authorization": "direct" if direct else "fingerprint",
            # Batch membership when `generation-codex-batch` ran this item; None for a single run.
            "batch": dict(batch_context) if batch_context is not None else None,
            "candidate_render": {"enabled": True},
            "preloaded_docs": list(PRELOADED_SDK_DOC_PATHS),
            "image_observation": (
                "external_inventory"
                if image_observation is not None
                else ("author_only" if workspace.input_image_path is not None else "no_image")
            ),
        }
        with stage_timer.stage("persist"):
            record_dir = persist_generation(
                repo_root=request.repo_root,
                run_id=workspace.run_id,
                prompt=request.prompt,
                image_path=workspace.input_image_path,
                label=preflight.label,
                runtime=request.runtime,
                workspace_dir=workspace.root,
                runtime_report=runtime_report,
                artifacts=artifacts,
            )
        model_path = record_dir / "revisions" / REVISION_ID / "model.py"
        warnings = list(artifacts.warnings)
        warnings.extend(observation_warnings)
        for message in promotion_recovery.errors:
            warnings.append(f"Promotion recovery: {message}")
        retained_workspace: Path | None = workspace.root
        if not preflight.keep_workspace:
            # The canonical record already exists; a cleanup failure must not fail the command.
            try:
                shutil.rmtree(workspace.root)
                retained_workspace = None
            except OSError as exc:
                warnings.append(
                    f"Promotion succeeded but the staging workspace could not be removed: "
                    f"{workspace.root} ({exc})"
                )

        return GenerationResult(
            status="generation_ready",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=model_path,
            workspace_dir=retained_workspace,
            warnings=tuple(warnings),
            run_id=workspace.run_id,
        )
    except BaseException as exc:
        _attach_failure_context(exc, workspace=workspace)
        raise


__all__ = [
    "GenerationApprovalError",
    "GenerationBlockedError",
    "GenerationFailureContext",
    "GenerationInputChangedError",
    "GenerationResult",
    "execute_generation",
    "fingerprint_generation_preflight",
    "generation_failure_context",
    "preflight_generation",
]
