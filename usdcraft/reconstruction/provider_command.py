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

"""Direct NVIDIA Provider execution of one Reconstruction request.

The Provider arm reuses the Reconstruction request contract, the atomic input freeze, the
optional Subject-Isolation Agent (`--mesh-cleanup`, default off), the deterministic mesh-preprocess
stage, the shared image-observation engine, the Author task, the shared mechanical Host gate and
canonical promotion. It differs only at the arm surfaces: direct execution without the Codex
preflight/fingerprint exchange, the Provider transport (for the Author, image observation, the scale
estimate and subject isolation alike), the bounded Provider tools (plus the Reconstruction evidence
reads and the AXIAL diagnostic), the Provider-only time/turn limits, one Author per run and no
notes file. The run finishes at the mechanical Host gate without an independent visual critic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Any, Mapping, cast

from storage.record_mutations import recover_pending_promotions
from storage.repo import StorageRepo
from usdcraft._images import validate_decodable_image
from usdcraft.compile import compile_model
from usdcraft.pipeline.contracts import ProviderRuntimeSelection
from usdcraft.pipeline.execution import (
    _provider_check_runner,
    _provider_compile_runner,
    _provider_selection_source,
    _require_isolated_provider_compile_worker,
    compile_author_output,
)
from usdcraft.pipeline.provider import (
    NvidiaProviderConfig,
    NvidiaResponsesClient,
    ProviderHarness,
    ProviderHarnessCancelledError,
)
from usdcraft.pipeline.provider.progress import (
    ProviderProgressCallback,
    emit_provider_progress,
)
from usdcraft.reconstruction.command import (
    ReconstructionBlockedError,
    _attach_reconstruction_failure_context,
    _build_reconstruction_preprocessing,
    fingerprint_reconstruction_preflight,
)
from usdcraft.reconstruction.contract import (
    ReconstructionInputIdentity,
    ReconstructionPreflight,
    ReconstructionRequest,
)
from usdcraft.reconstruction.default_prompt import resolve_reconstruction_prompt
from usdcraft.reconstruction.lifecycle import (
    author_runtime_metadata,
    cleanup_completed_workspaces,
    promote_author_output,
)
from usdcraft.reconstruction.observation import (
    render_inventory_for_prompt,
    run_image_observation_via_provider,
)
from usdcraft.reconstruction.persist import (
    REVISION_ID,
    persist_reconstruction,
    provider_reconstruction_authoring,
)
from usdcraft.reconstruction.prompt import (
    build_reconstruction_author_prompt,
    reconstruction_cache_prefix,
)
from usdcraft.reconstruction.provider_isolation import run_subject_isolation_via_provider
from usdcraft.reconstruction.provider_tools import (
    ReconstructionToolExecutor,
    build_reconstruction_tool_schemas,
)
from usdcraft.reconstruction.scale_estimate import (
    ESTIMATED_SCALE_SOURCE,
    ScaleEstimate,
    ScaleEstimateError,
    ScaleInspection,
    estimate_subject_scale,
    inspect_source_scale,
    meters_per_unit_from_estimate,
    scale_note_for_author,
    scale_report,
)
from usdcraft.reconstruction.source import FrameValueSource, SourceInput, load_normalized_source
from usdcraft.reconstruction.workspace import (
    create_reconstruction_author_workspace,
    create_reconstruction_run_workspace,
)
from usdcraft.stagetime import STAGE_TRACE_FILENAME, StageTimer

PIPELINE_LABEL = "Reconstruction Provider"
PROGRESS_LABEL = "Provider Reconstruction"


class ProviderReconstructionCancelledError(RuntimeError):
    """A coordinator requested cancellation before canonical promotion."""


@dataclass(slots=True, frozen=True)
class ProviderReconstructionResult:
    status: str
    run_id: str
    record_id: str | None
    record_dir: Path | None
    model_path: Path | None
    workspace_dir: Path | None
    warnings: tuple[str, ...] = ()
    # The Provider endpoint that served the run (`openai` or `nvidia`); the record carries the same.
    provider: str = "openai"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "pipeline": "reconstruction",
            "provider": self.provider,
            "run_id": self.run_id,
            "record_id": self.record_id,
            "record_dir": str(self.record_dir) if self.record_dir is not None else None,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir is not None else None,
            "warnings": list(self.warnings),
        }


def _raise_if_cancelled(cancellation_event: Event | None) -> None:
    if cancellation_event is not None and cancellation_event.is_set():
        raise ProviderReconstructionCancelledError(
            "Provider Reconstruction was cancelled by the local coordinator."
        )


def _source_scale_inspection(
    source: Path,
    *,
    up_axis: str | None,
    front_axis: str | None,
    meters_per_unit: float | None,
) -> ScaleInspection:
    """Load the source once under the user's frame declarations and inspect its scale."""

    return inspect_source_scale(
        load_normalized_source(
            SourceInput(
                source, up_axis=up_axis, front_axis=front_axis, meters_per_unit=meters_per_unit
            )
        )
    )


def resolve_provider_reconstruction_preflight(
    *,
    repo_root: Path,
    source: Path,
    semantic_image: Path,
    config: NvidiaProviderConfig,
    selection_source: str,
    up_axis: str | None = None,
    front_axis: str | None = None,
    meters_per_unit: float | None = None,
    guidance: str | None = None,
    prompt: str | None = None,
    label: str | None = None,
    keep_workspace: bool = False,
    scaffold_path: Path | None = None,
    meters_per_unit_source: str | None = None,
    scale_estimate: Mapping[str, Any] | None = None,
    image_observation: bool = True,
    max_turns: int | None = None,
    mesh_cleanup: bool = False,
) -> ReconstructionPreflight:
    """Resolve and validate the request inputs for direct Provider execution.

    This is the same input resolution the Codex arm performs before approval — source closure,
    decodable semantic image, scaffold and frame declaration — without a Codex executable or an
    approval exchange. The semantic image is the reconstruction target's semantic record
    (`reconstruction_target`); the Provider arm does not distinguish where it came from. The
    fingerprint is retained as the request identity written into the run workspace; it
    authorizes nothing.
    """

    resolved_repo = Path(repo_root).expanduser().resolve()
    request = ReconstructionRequest(
        image_observation=image_observation,
        max_turns=max_turns,
        repo_root=resolved_repo,
        source_path=source,
        semantic_image_path=semantic_image,
        up_axis=up_axis,
        front_axis=front_axis,
        meters_per_unit=meters_per_unit,
        meters_per_unit_source=meters_per_unit_source,
        scale_estimate=scale_estimate,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        runtime_source=config.runtime_source,
        selection_source=selection_source,
        timeout_seconds=config.timeout_seconds,
        label=label,
        guidance=guidance,
        prompt=prompt,
        mesh_cleanup=bool(mesh_cleanup),
    )
    validate_decodable_image(request.semantic_image_path, label="Reconstruction semantic image")
    scene = load_normalized_source(
        SourceInput(
            request.source_path,
            up_axis=request.up_axis,
            front_axis=request.front_axis,
            meters_per_unit=request.meters_per_unit,
            meters_per_unit_source=cast("FrameValueSource | None", request.meters_per_unit_source),
        )
    )
    source_dependencies = tuple(
        ReconstructionInputIdentity(
            path=dependency.path,
            sha256=dependency.sha256,
            byte_count=dependency.byte_count,
        )
        for dependency in scene.source_dependencies
    )
    resolved_scaffold = Path(scaffold_path or resolved_repo / "scaffold.py").expanduser().resolve()
    draft = ReconstructionPreflight(
        request=request,
        source_dependencies=source_dependencies,
        semantic_image=ReconstructionInputIdentity.from_path(request.semantic_image_path),
        scaffold=ReconstructionInputIdentity.from_path(resolved_scaffold),
        codex_executable="",
        python_executable=str(Path(sys.executable).expanduser().absolute()),
        keep_workspace=bool(keep_workspace),
        fingerprint="",
    )
    return replace(draft, fingerprint=fingerprint_reconstruction_preflight(draft))


def execute_provider_reconstruction(
    *,
    repo_root: Path,
    source: Path,
    semantic_image: Path,
    up_axis: str | None = None,
    front_axis: str | None = None,
    meters_per_unit: float | None = None,
    guidance: str | None = None,
    prompt: str | None = None,
    label: str | None = None,
    model: str | None = None,
    reasoning: str | None = None,
    timeout_seconds: float | None = None,
    keep_workspace: bool = False,
    progress: ProviderProgressCallback | None = None,
    cancellation_event: Event | None = None,
    environ: Mapping[str, str] | None = None,
    batch_context: Mapping[str, object] | None = None,
    image_observation: bool = True,
    max_turns: int | None = None,
    run_id: str | None = None,
    api_style: str = "responses",
    mesh_cleanup: bool = False,
) -> ProviderReconstructionResult:
    """Directly execute one Reconstruction request through the NVIDIA Provider.

    The shared task exposes the user's selected Author tool capabilities.
    Reliable source metrics remain authoritative; unresolved attributes are
    constrained hypotheses. No independent critic runs after Author completion.
    ``mesh_cleanup`` (default off) runs the shared Subject-Isolation Agent through the NVIDIA
    transport before mesh preprocessing; otherwise the Host freezes the retain-all declaration.
    """

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"USDCraft repository root is not a directory: {root}")
    _raise_if_cancelled(cancellation_event)
    _require_isolated_provider_compile_worker(environ)
    config_kwargs: dict[str, Any] = {"model": model, "reasoning_effort": reasoning}
    if api_style != "responses":
        config_kwargs["api_style"] = api_style
    if timeout_seconds is not None:
        config_kwargs["timeout_seconds"] = float(timeout_seconds)
    config = NvidiaProviderConfig.from_repo(root, **config_kwargs)
    from usdcraft.reconstruction.harness_policy import resolve_harness_policy

    resolve_harness_policy(
        image_observation=image_observation,
        max_turns=max_turns,
    )
    selection_source = _provider_selection_source(model=model, reasoning=reasoning)

    # Scale first: a mesh without units whose longest side sits on a normalizer's target (1.0 or
    # 2.0) is not in metres. One vision call over the semantic image estimates the subject's real
    # longest dimension, and the quotient becomes this run's meters_per_unit before any evidence
    # is built, so AXIAL, the Author, the physics and the export all work at real scale. An
    # explicit declaration skips the estimate.
    scale_inspection = _source_scale_inspection(
        Path(source), up_axis=up_axis, front_axis=front_axis, meters_per_unit=meters_per_unit
    )
    resolved_meters_per_unit = meters_per_unit
    scale_source: str | None = None
    scale_estimate: ScaleEstimate | None = None
    scale_warnings: list[str] = []
    if meters_per_unit is None and scale_inspection.suspected_unit_normalized:
        emit_provider_progress(
            progress,
            f'{PROGRESS_LABEL} | activity="estimating scale" | detail='
            f'"longest side {scale_inspection.longest_extent_units:.3f} units, no unit metadata"',
        )
        semantic_path = Path(semantic_image).expanduser().resolve()
        try:
            scale_estimate = estimate_subject_scale(
                semantic_image_path=semantic_path,
                request_text=resolve_reconstruction_prompt(prompt=prompt),
                label=label,
                config=config,
                cancellation_event=cancellation_event,
            )
        except ScaleEstimateError as exc:
            scale_warnings.append(
                "Scale estimate failed; the source's "
                f"{scale_inspection.longest_extent_units:.3f}-unit longest side is treated as "
                f"metres and the record is authored at that size: {exc}"
            )
            emit_provider_progress(
                progress,
                f'{PROGRESS_LABEL} | activity="scale estimate degraded" | '
                f"error_type={type(exc).__name__}",
            )
        else:
            resolved_meters_per_unit = meters_per_unit_from_estimate(
                scale_inspection, scale_estimate
            )
            scale_source = ESTIMATED_SCALE_SOURCE
            emit_provider_progress(
                progress,
                f'{PROGRESS_LABEL} | activity="scale estimated" | detail='
                f'"{scale_estimate.category}: longest side {scale_estimate.longest_dimension_m:.2f} m '
                f"-> meters_per_unit {resolved_meters_per_unit:.4f} "
                f'({scale_estimate.confidence} confidence)"',
            )

    preflight = resolve_provider_reconstruction_preflight(
        image_observation=image_observation,
        max_turns=max_turns,
        repo_root=root,
        source=source,
        semantic_image=semantic_image,
        config=config,
        selection_source=selection_source,
        up_axis=up_axis,
        front_axis=front_axis,
        meters_per_unit=resolved_meters_per_unit,
        meters_per_unit_source=scale_source,
        scale_estimate=scale_estimate.to_dict() if scale_estimate is not None else None,
        guidance=guidance,
        prompt=prompt,
        label=label,
        keep_workspace=keep_workspace,
        mesh_cleanup=mesh_cleanup,
    )
    runtime = ProviderRuntimeSelection(
        model_id=config.model,
        reasoning_effort=config.reasoning_effort,
        runtime_source=preflight.request.runtime_source,
        selection_source=selection_source,
    )
    emit_provider_progress(
        progress,
        f"{PROGRESS_LABEL} | activity=starting | model={config.model!r} | "
        f"effort={config.reasoning_effort!r} | max_turns={preflight.request.harness_policy.max_turns}",
    )

    # The stage timer narrates every Host stage (freeze, mesh preprocess, image observation,
    # author, host gate, persist) into the same progress display as the harness.
    stage_timer = StageTimer(
        pipeline=PIPELINE_LABEL,
        log=lambda message: emit_provider_progress(progress, message),
        log_label=PROGRESS_LABEL,
    )
    with stage_timer.stage("freeze"):
        run_workspace = (
            create_reconstruction_run_workspace(preflight)
            if run_id is None
            else create_reconstruction_run_workspace(preflight, run_id=run_id)
        )
    stage_timer.attach_trace_path(run_workspace.root / STAGE_TRACE_FILENAME)
    emit_provider_progress(
        progress,
        f'{PROGRESS_LABEL} | activity="workspace ready" | run_id={run_workspace.run_id!r}',
    )
    warnings: list[str] = list(scale_warnings)
    try:
        recovery = recover_pending_promotions(StorageRepo(root))
        warnings.extend(f"Promotion recovery: {message}" for message in recovery.errors)
        _raise_if_cancelled(cancellation_event)

        def observe(**kwargs: Any):
            return run_image_observation_via_provider(
                config=config,
                cancellation_event=cancellation_event,
                **kwargs,
            )

        def isolate(**kwargs: Any):
            return run_subject_isolation_via_provider(
                config=config,
                cancellation_event=cancellation_event,
                **kwargs,
            )

        preprocessing = _build_reconstruction_preprocessing(
            preflight,
            run_workspace,
            stage_timer=stage_timer,
            image_observation_runner=observe,
            isolation_runner=isolate,
        )
        warnings.extend(preprocessing.warnings)
        _raise_if_cancelled(cancellation_event)

        # One Author per run: its workspace is the run's author root itself, as in Generation.
        workspace = create_reconstruction_author_workspace(
            preflight,
            run_workspace,
            preprocessing.evidence,
            root=run_workspace.author_workspace_base,
            author_brief=True,
            # Source views are requested by the Author; only the semantic image is attached.
        )
        task = build_reconstruction_author_prompt(
            workspace=workspace,
            user_guidance=preflight.request.guidance,
            check_command="check_model",
            compile_command="compile_model",
            axial_reencode_command="check_axial",
            image_observation_text=(
                render_inventory_for_prompt(preprocessing.image_observation.inventory)
                if preprocessing.image_observation is not None
                else None
            ),
            reference_access="provider_tools",
            request_text=preflight.request.prompt,
            scale_note=(
                scale_note_for_author(scale_inspection, scale_estimate)
                if scale_estimate is not None
                else None
            ),
        )
        client = NvidiaResponsesClient(config, cancellation_event=cancellation_event)
        executor = ReconstructionToolExecutor(
            workspace,
            check_runner=_provider_check_runner,
            compile_runner=_provider_compile_runner,
            source_usd_path=preprocessing.evidence.normalized_source_path,
        )
        harness = ProviderHarness(
            workspace=workspace,
            client=client,
            tool_executor=executor,
            # The Provider arm always resolves a turn budget (turn_budget=True).
            max_turns=cast(int, preflight.request.harness_policy.max_turns),
            progress=progress,
            cancellation_event=cancellation_event,
            tool_schemas=build_reconstruction_tool_schemas(),
            progress_label=PROGRESS_LABEL,
        )
        try:
            with stage_timer.stage("author"):
                harness_result = harness.run(
                    prompt=task,
                    image_paths=workspace.attached_images,
                    cache_prefix=reconstruction_cache_prefix(task),
                )
        except ProviderHarnessCancelledError as exc:
            raise ProviderReconstructionCancelledError(str(exc)) from exc
        if harness_result.status == "blocked":
            raise ReconstructionBlockedError(harness_result.summary, workspace.root)
        _raise_if_cancelled(cancellation_event)

        artifacts = compile_author_output(workspace, stage_timer, compile_runner=compile_model)
        _raise_if_cancelled(cancellation_event)

        runtime_report: dict[str, Any] = {
            **({"api_style": config.api_style} if config.api_style != "responses" else {}),
            **harness_result.runtime_report(),
            **author_runtime_metadata(
                preflight=preflight,
                workspace=workspace,
                evidence=preprocessing.evidence,
                image_observation=preprocessing.image_observation,
                timer=stage_timer,
                axial_payloads=executor.axial_payloads,
            ),
            "authorization": "direct",
            # Batch membership (batch_id, csv, row_id, ...) when `reconstruction-batch` ran this
            # row; None for a single-item run. Recorded for grouping in the Viewer only.
            "batch": dict(batch_context) if batch_context else None,
            "scale": scale_report(
                scale_inspection,
                meters_per_unit=float(
                    preflight.request.meters_per_unit
                    if preflight.request.meters_per_unit is not None
                    else 1.0
                ),
                source=scale_source
                or (
                    "explicit"
                    if meters_per_unit is not None
                    else scale_inspection.meters_per_unit_source
                ),
                estimate=scale_estimate,
            ),
            "author_feedback_calls": executor.feedback_call_count,
            "author_validation": executor.validation_state(),
        }
        record_dir = promote_author_output(
            preflight=preflight,
            run_workspace=run_workspace,
            workspace=workspace,
            evidence=preprocessing.evidence,
            image_observation=preprocessing.image_observation,
            runtime=runtime,
            runtime_report=runtime_report,
            artifacts=artifacts,
            authoring=provider_reconstruction_authoring(config.endpoint),
            timer=stage_timer,
            persist_runner=persist_reconstruction,
        )
        warnings.extend(artifacts.warnings)
        emit_provider_progress(
            progress,
            f'{PROGRESS_LABEL} | activity="canonical record promoted" | '
            f"record_id={record_dir.name!r}",
        )
    except BaseException as exc:
        emit_provider_progress(
            progress,
            f"{PROGRESS_LABEL} | activity=failed | error_type={type(exc).__name__}",
        )
        _attach_reconstruction_failure_context(
            exc,
            run_workspace=run_workspace,
            workspace_dir=run_workspace.author_workspace_base,
        )
        raise

    retained = cleanup_completed_workspaces(
        run_workspace, workspace, keep_workspace=keep_workspace, warnings=warnings
    )
    emit_provider_progress(
        progress,
        f"{PROGRESS_LABEL} | activity=completed | status=reconstruction_ready | "
        f"record_id={record_dir.name!r}",
    )
    return ProviderReconstructionResult(
        status="reconstruction_ready",
        run_id=run_workspace.run_id,
        record_id=record_dir.name,
        record_dir=record_dir,
        model_path=record_dir / "revisions" / REVISION_ID / "model.py",
        workspace_dir=retained,
        warnings=tuple(warnings),
        provider=config.endpoint,
    )


__all__ = [
    "PIPELINE_LABEL",
    "PROGRESS_LABEL",
    "ProviderReconstructionCancelledError",
    "ProviderReconstructionResult",
    "execute_provider_reconstruction",
    "resolve_provider_reconstruction_preflight",
]
