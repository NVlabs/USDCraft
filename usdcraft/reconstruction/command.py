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
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from storage.record_mutations import recover_pending_promotions
from storage.repo import StorageRepo
from usdcraft._images import validate_decodable_image
from usdcraft.compile import compile_model
from usdcraft.compile.command import CompileArtifacts
from usdcraft.pipeline.codex import ProcessRunner
from usdcraft.pipeline.contracts import CodexCompletionStatus, CodexRunResult, CodexRuntimeSelection
from usdcraft.pipeline.execution import compile_author_output
from usdcraft.pipeline.hashing import sha256_bytes, sha256_file
from usdcraft.pipeline.runtime import _codex_executable, current_main_codex_runtime
from usdcraft.reconstruction.author import run_reconstruction_author
from usdcraft.reconstruction.contract import (
    ReconstructionEvidenceBundle,
    ReconstructionInputIdentity,
    ReconstructionPreflight,
    ReconstructionRequest,
)
from usdcraft.reconstruction.diagnostic_cli import read_axial_payloads
from usdcraft.reconstruction.isolation import write_retain_all_subject_isolation
from usdcraft.reconstruction.isolation_agent import (
    SubjectIsolationArtifacts,
    run_subject_isolation,
)
from usdcraft.reconstruction.lifecycle import (
    author_runtime_metadata,
    cleanup_completed_workspaces,
    promote_author_output,
)
from usdcraft.reconstruction.mesh_preprocess import MeshPreprocessResult, run_mesh_preprocess
from usdcraft.reconstruction.observation import (
    ImageObservationArtifacts,
    ImageObservationError,
    render_inventory_for_prompt,
    run_image_observation,
)
from usdcraft.reconstruction.persist import (
    CODEX_RECONSTRUCTION_AUTHORING,
    REVISION_ID,
    persist_reconstruction,
)
from usdcraft.reconstruction.prompt import build_reconstruction_author_prompt
from usdcraft.reconstruction.scale_estimate import (
    ESTIMATED_SCALE_SOURCE,
    ScaleEstimate,
    ScaleEstimateError,
    ScaleInspection,
    estimate_subject_scale_via_codex,
    inspect_source_scale,
    meters_per_unit_from_estimate,
    scale_note_for_author,
    scale_report,
)
from usdcraft.reconstruction.source import FrameValueSource, SourceInput, load_normalized_source
from usdcraft.reconstruction.workspace import (
    ReconstructionAuthorWorkspace,
    ReconstructionRunWorkspace,
    create_reconstruction_author_workspace,
    create_reconstruction_run_workspace,
)
from usdcraft.stagetime import STAGE_TRACE_FILENAME, StageTimer

PIPELINE_LABEL = "Reconstruction Codex"
AXIAL_PAYLOAD_DIRNAME = "axial_checks"


class ReconstructionApprovalError(RuntimeError):
    pass


class ReconstructionInputChangedError(RuntimeError):
    pass


class ReconstructionBlockedError(RuntimeError):
    def __init__(self, summary: str, workspace_dir: Path) -> None:
        super().__init__(f"Codex Reconstruction reported a blocker: {summary} ({workspace_dir})")
        self.summary = summary
        self.workspace_dir = workspace_dir


class ReconstructionPreprocessingBlockedError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class ReconstructionPreprocessingResult:
    isolation: SubjectIsolationArtifacts | None
    mesh_preprocess: MeshPreprocessResult
    evidence: ReconstructionEvidenceBundle
    image_observation: ImageObservationArtifacts | None = None
    warnings: tuple[str, ...] = ()


def _resolve_runtime(
    *,
    model: str | None,
    reasoning: str | None,
    environ: Mapping[str, str] | None,
    codex_home: Path | None,
) -> CodexRuntimeSelection:
    if (model is None) != (reasoning is None):
        raise ValueError("--model and --reasoning must be provided together.")
    if model is not None and reasoning is not None:
        return CodexRuntimeSelection(
            model_id=model,
            reasoning_effort=reasoning,
            runtime_source="explicit_reconstruction_request",
            selection_source="explicit_reconstruction_request",
        )
    inherited = current_main_codex_runtime(environ=environ, codex_home=codex_home)
    return CodexRuntimeSelection(
        model_id=inherited.model_id,
        reasoning_effort=inherited.reasoning_effort,
        runtime_source=inherited.runtime_source,
        selection_source="inherited_root_for_reconstruction",
    )


def fingerprint_reconstruction_preflight(preflight: ReconstructionPreflight) -> str:
    payload = json.dumps(
        preflight.fingerprint_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def _verify_approved_inputs(preflight: ReconstructionPreflight) -> None:
    for identity in (*preflight.source_dependencies, preflight.semantic_image, preflight.scaffold):
        if not identity.path.is_file():
            raise ReconstructionInputChangedError(
                f"Approved Reconstruction input disappeared: {identity.path}"
            )
        if identity.path.stat().st_size != identity.byte_count:
            raise ReconstructionInputChangedError(
                f"Approved Reconstruction input size changed: {identity.path}"
            )
        if sha256_file(identity.path) != identity.sha256:
            raise ReconstructionInputChangedError(
                f"Approved Reconstruction input changed: {identity.path}"
            )


def verify_reconstruction_approval(
    preflight: ReconstructionPreflight,
    *,
    approved_fingerprint: str,
) -> None:
    expected = fingerprint_reconstruction_preflight(preflight)
    if not hmac.compare_digest(preflight.fingerprint, expected):
        raise ReconstructionApprovalError("Reconstruction preflight is internally inconsistent.")
    if not hmac.compare_digest(str(approved_fingerprint or "").strip(), expected):
        raise ReconstructionApprovalError(
            "Reconstruction approval does not match the current preflight; run preflight again."
        )
    _verify_approved_inputs(preflight)


def begin_reconstruction_run(
    preflight: ReconstructionPreflight,
    *,
    approved_fingerprint: str,
    run_id: str | None = None,
) -> ReconstructionRunWorkspace:
    """Verify the later-turn approval and create the private upstream run root."""

    verify_reconstruction_approval(
        preflight,
        approved_fingerprint=approved_fingerprint,
    )
    return create_reconstruction_run_workspace(
        preflight,
        run_id=run_id,
    )


def _require_upstream_evidence(
    preflight: ReconstructionPreflight,
    run_workspace: ReconstructionRunWorkspace,
    evidence: ReconstructionEvidenceBundle,
) -> None:
    frozen = run_workspace.frozen_inputs
    if evidence.original_source_path != frozen.source_path:
        raise ReconstructionInputChangedError(
            "Reconstruction evidence does not reference the frozen approved source."
        )
    observed_dependencies = tuple(
        ReconstructionInputIdentity.from_path(path)
        for path in evidence.original_source_dependencies
    )
    if observed_dependencies != frozen.source_dependencies:
        raise ReconstructionInputChangedError(
            "Reconstruction evidence source closure differs from the frozen approved closure."
        )
    if evidence.original_source_tree_root != frozen.source_tree_root:
        raise ReconstructionInputChangedError(
            "Reconstruction evidence does not reference the frozen source tree."
        )
    if evidence.approved_inputs_manifest_path != frozen.manifest_path:
        raise ReconstructionInputChangedError(
            "Reconstruction evidence does not reference the frozen approved-input manifest."
        )
    if (
        evidence.approved_inputs_manifest_sha256 != frozen.manifest_sha256
        or evidence.approved_inputs_manifest_byte_count != frozen.manifest_byte_count
    ):
        raise ReconstructionInputChangedError(
            "Reconstruction evidence does not preserve the approved-input manifest identity."
        )
    if evidence.semantic_image_path != frozen.semantic_image_path:
        raise ReconstructionInputChangedError(
            "Reconstruction evidence does not reference the frozen semantic image."
        )
    if evidence.scaffold_path != frozen.scaffold_path:
        raise ReconstructionInputChangedError(
            "Reconstruction evidence does not reference the frozen scaffold."
        )
    if sha256_file(evidence.semantic_image_path) != preflight.semantic_image.sha256:
        raise ReconstructionInputChangedError(
            "Frozen Reconstruction evidence semantic image differs from the approved preflight."
        )

    upstream_paths = (
        evidence.axial_path,
        evidence.normalized_source_path,
        evidence.source_manifest_path,
        evidence.subject_isolation_manifest_path,
        evidence.subject_isolation_binding_path,
        evidence.mesh_preprocess_manifest_path,
        evidence.source_overlay_manifest_path,
        evidence.source_overlay_urdf_path,
        evidence.source_overlay_glb_path,
        *evidence.source_overlay_dependency_paths,
        *((evidence.voxel_evidence_path,) if evidence.voxel_evidence_path is not None else ()),
    )
    upstream_root = run_workspace.upstream_dir.resolve()
    for path in upstream_paths:
        try:
            path.resolve().relative_to(upstream_root)
        except ValueError as exc:
            raise ValueError(
                f"Derived Reconstruction evidence must stay inside the run upstream root: {path}"
            ) from exc


def _verify_frozen_inputs(
    preflight: ReconstructionPreflight,
    run_workspace: ReconstructionRunWorkspace,
) -> None:
    frozen = run_workspace.frozen_inputs
    for item in frozen.source_files:
        observed = ReconstructionInputIdentity.from_path(item.staged.path)
        if observed != item.staged:
            raise ReconstructionInputChangedError(
                f"Frozen Reconstruction source changed after approval: {item.staged.path}"
            )
    for path, expected, label in (
        (frozen.semantic_image_path, preflight.semantic_image, "semantic image"),
        (frozen.scaffold_path, preflight.scaffold, "scaffold"),
    ):
        observed = ReconstructionInputIdentity.from_path(path)
        if observed.sha256 != expected.sha256 or observed.byte_count != expected.byte_count:
            raise ReconstructionInputChangedError(
                f"Frozen Reconstruction {label} changed after approval: {path}"
            )
    if not frozen.manifest_path.is_file():
        raise ReconstructionInputChangedError(
            f"Frozen Reconstruction input manifest disappeared: {frozen.manifest_path}"
        )
    if (
        frozen.manifest_path.stat().st_size != frozen.manifest_byte_count
        or sha256_file(frozen.manifest_path) != frozen.manifest_sha256
    ):
        raise ReconstructionInputChangedError(
            f"Frozen Reconstruction input manifest changed: {frozen.manifest_path}"
        )


def _source_input(request: ReconstructionRequest, *, path: Path) -> SourceInput:
    return SourceInput(
        path,
        up_axis=request.up_axis,
        front_axis=request.front_axis,
        meters_per_unit=request.meters_per_unit,
        meters_per_unit_source=cast("FrameValueSource | None", request.meters_per_unit_source),
    )


def _build_reconstruction_preprocessing(
    preflight: ReconstructionPreflight,
    run_workspace: ReconstructionRunWorkspace,
    *,
    stage_timer: StageTimer | None = None,
    image_observation_runner: Callable[..., ImageObservationArtifacts] | None = None,
    isolation_runner: Callable[..., SubjectIsolationArtifacts] | None = None,
) -> ReconstructionPreprocessingResult:
    """Build the one frozen preprocessing/evidence bundle for the run's single Author.

    ``image_observation_runner`` lets an execution arm supply its own observation transport; it
    receives the frozen semantic image and the output directory (the mesh renders never enter
    the observation reading). Without it the Codex observation Agent runs.
    ``isolation_runner`` likewise supplies the Subject-Isolation transport used when the request
    asks for mesh cleanup; it receives the frozen source input and the output directory. Without
    it the Codex isolation Agent runs. The task, evidence, validation and manifest are shared.
    Source geometry is exported for on-demand Author rendering and Viewer overlays; no images
    are generated during preprocessing.
    """

    timer = stage_timer or StageTimer(pipeline=PIPELINE_LABEL, log=None)
    _verify_frozen_inputs(preflight, run_workspace)
    request = preflight.request
    source_input = _source_input(request, path=run_workspace.frozen_inputs.source_path)
    isolation: SubjectIsolationArtifacts | None = None
    with timer.stage(
        "subject_isolation",
        detail="agent" if request.mesh_cleanup else "skipped_retain_all",
    ):
        if request.mesh_cleanup:
            if isolation_runner is not None:
                isolation = isolation_runner(
                    source_input=source_input,
                    output_dir=run_workspace.upstream_dir / "subject_isolation",
                )
            else:
                isolation = run_subject_isolation(
                    source_input,
                    run_workspace.upstream_dir / "subject_isolation",
                    model=request.runtime.model_id,
                    reasoning_effort=request.runtime.reasoning_effort,
                    executable=preflight.codex_executable,
                )
            if isolation.status != "ready":
                raise ReconstructionPreprocessingBlockedError(
                    "Subject-Isolation Agent did not produce a ready retained scene."
                )
            isolation_manifest = isolation.manifest
        else:
            # Mesh cleanup is optional: without an explicit user request the Host freezes a
            # deterministic retain-all declaration instead of starting the isolation Agent.
            isolation_manifest = write_retain_all_subject_isolation(
                load_normalized_source(source_input),
                run_workspace.upstream_dir / "subject_isolation",
            )

    with timer.stage("mesh_preprocess"):
        mesh_preprocess = run_mesh_preprocess(
            source_input,
            run_workspace.upstream_dir,
            subject_isolation=isolation_manifest,
        )

    # Image observation is one clean-context vision reading before the Author. It is an
    # external first reading for the Author, never evidence: a failed or blocked observation
    # degrades the run to the previous Author-only reading with a warning instead of failing.
    image_observation: ImageObservationArtifacts | None = None
    preprocessing_warnings: list[str] = []
    if request.harness_policy.image_observation:
        try:
            with timer.stage("image_observation"):
                if image_observation_runner is not None:
                    image_observation = image_observation_runner(
                        semantic_image_path=run_workspace.frozen_inputs.semantic_image_path,
                        output_dir=run_workspace.upstream_dir / "image_observation",
                    )
                else:
                    image_observation = run_image_observation(
                        semantic_image_path=run_workspace.frozen_inputs.semantic_image_path,
                        output_dir=run_workspace.upstream_dir / "image_observation",
                        model=request.runtime.model_id,
                        reasoning_effort=request.runtime.reasoning_effort,
                        executable=preflight.codex_executable,
                    )
        except ImageObservationError as exc:
            preprocessing_warnings.append(
                f"Image observation degraded to the Author-only reading: {exc}"
            )
    case = mesh_preprocess.case
    source_overlay = mesh_preprocess.source_overlay
    evidence = ReconstructionEvidenceBundle(
        semantic_image_path=run_workspace.frozen_inputs.semantic_image_path,
        scaffold_path=run_workspace.frozen_inputs.scaffold_path,
        axial_path=case.axial,
        original_source_path=run_workspace.frozen_inputs.source_path,
        original_source_dependencies=tuple(
            identity.path for identity in run_workspace.frozen_inputs.source_dependencies
        ),
        original_source_tree_root=run_workspace.frozen_inputs.source_tree_root,
        approved_inputs_manifest_path=run_workspace.frozen_inputs.manifest_path,
        approved_inputs_manifest_sha256=run_workspace.frozen_inputs.manifest_sha256,
        approved_inputs_manifest_byte_count=(run_workspace.frozen_inputs.manifest_byte_count),
        normalized_source_path=case.normalized_source,
        source_manifest_path=case.source_manifest,
        subject_isolation_manifest_path=isolation_manifest,
        subject_isolation_binding_path=case.subject_isolation,
        mesh_preprocess_manifest_path=case.case_manifest,
        source_overlay_manifest_path=source_overlay.manifest,
        source_overlay_urdf_path=source_overlay.carrier_urdf,
        source_overlay_glb_path=source_overlay.carrier_glb,
        voxel_evidence_path=case.voxel_evidence,
    )
    return ReconstructionPreprocessingResult(
        isolation=isolation,
        mesh_preprocess=mesh_preprocess,
        evidence=evidence,
        image_observation=image_observation,
        warnings=tuple(preprocessing_warnings),
    )


@dataclass(slots=True, frozen=True)
class ReconstructionFailureContext:
    run_id: str
    workspace_dir: Path


@dataclass(slots=True, frozen=True)
class ReconstructionAuthorResult:
    """One completed, Host-compiled Author run awaiting canonical persistence."""

    status: str
    run_id: str
    run_workspace: ReconstructionRunWorkspace
    author_workspace: ReconstructionAuthorWorkspace
    codex_result: CodexRunResult
    artifacts: CompileArtifacts
    runtime_report: dict[str, object]
    warnings: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ReconstructionResult:
    """The Codex arm's result: one record, printed with the same keys as the Provider arm."""

    status: str
    run_id: str
    record_id: str | None
    record_dir: Path | None
    model_path: Path | None
    workspace_dir: Path | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "pipeline": "reconstruction",
            "provider": "codex-cli",
            "run_id": self.run_id,
            "record_id": self.record_id,
            "record_dir": str(self.record_dir) if self.record_dir is not None else None,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir is not None else None,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True, frozen=True)
class ReconstructionScaleResolution:
    """How the run's metres were decided before the freeze (see ``scale_estimate``)."""

    preflight: ReconstructionPreflight
    inspection: ScaleInspection
    estimate: ScaleEstimate | None
    source: str
    warnings: tuple[str, ...] = ()

    @property
    def author_note(self) -> str | None:
        if self.estimate is None:
            return None
        return scale_note_for_author(self.inspection, self.estimate)

    def report(self) -> dict[str, Any]:
        request = self.preflight.request
        return scale_report(
            self.inspection,
            meters_per_unit=float(
                request.meters_per_unit if request.meters_per_unit is not None else 1.0
            ),
            source=self.source,
            estimate=self.estimate,
        )


def preflight_reconstruction(
    *,
    repo_root: Path,
    source: Path,
    semantic_image: Path,
    up_axis: str | None = None,
    front_axis: str | None = None,
    meters_per_unit: float | None = None,
    guidance: str | None = None,
    prompt: str | None = None,
    mesh_cleanup: bool = False,
    model: str | None = None,
    reasoning: str | None = None,
    label: str | None = None,
    keep_workspace: bool = False,
    scaffold_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    codex_home: Path | None = None,
    image_observation: bool = True,
) -> ReconstructionPreflight:
    """Build one zero-write Codex-arm Reconstruction plan and validate the complete source closure.

    The semantic image is the reconstruction target's semantic record (role
    ``reconstruction_target``, as on the Provider arm); one Author runs per request. The Codex arm
    has no time or turn limit, so the request binds neither.
    """

    resolved_repo = Path(repo_root).expanduser().resolve()
    runtime = _resolve_runtime(
        model=model,
        reasoning=reasoning,
        environ=environ,
        codex_home=codex_home,
    )
    request = ReconstructionRequest(
        image_observation=image_observation,
        turn_budget=False,
        repo_root=resolved_repo,
        source_path=source,
        semantic_image_path=semantic_image,
        up_axis=up_axis,
        front_axis=front_axis,
        meters_per_unit=meters_per_unit,
        model=runtime.model_id,
        reasoning_effort=runtime.reasoning_effort,
        runtime_source=runtime.runtime_source,
        selection_source=runtime.selection_source,
        label=label,
        guidance=guidance,
        prompt=prompt,
        mesh_cleanup=bool(mesh_cleanup),
    )
    validate_decodable_image(
        request.semantic_image_path,
        label="Reconstruction semantic image",
    )
    scene = load_normalized_source(_source_input(request, path=request.source_path))
    source_dependencies = tuple(
        ReconstructionInputIdentity(
            path=dependency.path,
            sha256=dependency.sha256,
            byte_count=dependency.byte_count,
        )
        for dependency in scene.source_dependencies
    )
    resolved_scaffold = Path(scaffold_path or resolved_repo / "scaffold.py").expanduser().resolve()
    scaffold = ReconstructionInputIdentity.from_path(resolved_scaffold)
    semantic_identity = ReconstructionInputIdentity.from_path(request.semantic_image_path)
    draft = ReconstructionPreflight(
        request=request,
        source_dependencies=source_dependencies,
        semantic_image=semantic_identity,
        scaffold=scaffold,
        codex_executable=_codex_executable(environ),
        python_executable=str(Path(sys.executable).expanduser().absolute()),
        keep_workspace=bool(keep_workspace),
        fingerprint="",
    )
    return replace(draft, fingerprint=fingerprint_reconstruction_preflight(draft))


def resolve_reconstruction_scale(
    preflight: ReconstructionPreflight,
    *,
    environ: Mapping[str, str] | None = None,
    process_runner: Callable[..., Any] | None = None,
) -> ReconstructionScaleResolution:
    """Settle the source's scale after approval and before the freeze.

    A source without unit metadata whose longest side sits on 1.0 or 2.0 is suspected
    unit-normalized; one clean-context Codex vision call over the semantic image and the request
    text estimates the subject's real longest dimension, and the quotient becomes the run's
    ``meters_per_unit``. The estimate is a Host derivation: it is written into the request that the
    freeze records (``meters_per_unit_source`` = ``estimated_from_semantic_image``) and into
    ``runtime_report.scale``, and the fingerprint stays the user's approved token. An explicit
    declaration skips the step; a failed estimate keeps metres with a warning.
    """

    request = preflight.request
    inspection = inspect_source_scale(
        load_normalized_source(_source_input(request, path=request.source_path))
    )
    if request.meters_per_unit is not None:
        return ReconstructionScaleResolution(preflight, inspection, None, "explicit")
    if not inspection.suspected_unit_normalized:
        return ReconstructionScaleResolution(
            preflight, inspection, None, inspection.meters_per_unit_source
        )
    try:
        estimate = estimate_subject_scale_via_codex(
            semantic_image_path=request.semantic_image_path,
            request_text=request.prompt or "",
            label=request.label,
            model=request.model,
            reasoning_effort=request.reasoning_effort,
            executable=preflight.codex_executable,
            environ=environ,
            process_runner=process_runner,
        )
    except ScaleEstimateError as exc:
        warning = (
            "Scale estimate failed; the source's "
            f"{inspection.longest_extent_units:.3f}-unit longest side is treated as metres and "
            f"the record is authored at that size: {exc}"
        )
        return ReconstructionScaleResolution(
            preflight, inspection, None, inspection.meters_per_unit_source, (warning,)
        )
    resolved_request = replace(
        request,
        meters_per_unit=meters_per_unit_from_estimate(inspection, estimate),
        meters_per_unit_source=ESTIMATED_SCALE_SOURCE,
        scale_estimate=estimate.to_dict(),
    )
    return ReconstructionScaleResolution(
        replace(preflight, request=resolved_request), inspection, estimate, ESTIMATED_SCALE_SOURCE
    )


def execute_reconstruction_author(
    *,
    preflight: ReconstructionPreflight,
    run_workspace: ReconstructionRunWorkspace,
    evidence: ReconstructionEvidenceBundle,
    image_observation: ImageObservationArtifacts | None = None,
    process_runner: ProcessRunner | None = None,
    scale_note: str | None = None,
    stage_timer: StageTimer | None = None,
) -> ReconstructionAuthorResult:
    """Run the one Codex Author and the Host gate; canonical persistence is the next step.

    The Author's workspace is the run's author root itself (as on the Provider arm): the brief,
    the semantic image and the four mesh reference views are staged, and the check, compile,
    AXIAL-audit and per-face-measurement commands are the shell equivalents of the Provider tools.
    """

    _verify_frozen_inputs(preflight, run_workspace)
    _require_upstream_evidence(preflight, run_workspace, evidence)
    workspace = create_reconstruction_author_workspace(
        preflight,
        run_workspace,
        evidence,
        root=run_workspace.author_workspace_base,
        author_brief=True,
    )
    python = preflight.python_executable
    check_command = (python, "-m", "usdcraft.cli", "check", str(workspace.model_path))
    compile_command = (
        python,
        "-m",
        "usdcraft.cli",
        "compile",
        str(workspace.model_path),
        "--output",
        str(workspace.build_dir),
        "--no-render",
    )
    payload_dir = workspace.root / AXIAL_PAYLOAD_DIRNAME
    axial_command = (
        python,
        "-m",
        "usdcraft.cli",
        "recon-check-axial",
        str(workspace.axial_path),
        str(workspace.build_dir / "isaac" / "model.usdc"),
        "--source",
        str(evidence.normalized_source_path),
        "--payload-dir",
        str(payload_dir),
    )
    measure_command = (
        python,
        "-m",
        "usdcraft.cli",
        "recon-measure-face",
        str(workspace.axial_path),
    )
    task = build_reconstruction_author_prompt(
        workspace=workspace,
        user_guidance=preflight.request.guidance,
        check_command=check_command,
        compile_command=compile_command,
        axial_reencode_command=axial_command,
        measure_command=measure_command,
        image_observation_text=(
            render_inventory_for_prompt(image_observation.inventory)
            if image_observation is not None
            else None
        ),
        request_text=preflight.request.prompt,
        scale_note=scale_note,
    )
    timer = stage_timer or StageTimer(
        pipeline=PIPELINE_LABEL, trace_path=run_workspace.root / STAGE_TRACE_FILENAME
    )
    with timer.stage("author"):
        codex_result = run_reconstruction_author(
            preflight.request,
            workspace,
            prompt=task,
            source_usd_path=evidence.normalized_source_path,
            executable=preflight.codex_executable,
            process_runner=process_runner,
        )
    if codex_result.final_message.status is CodexCompletionStatus.BLOCKED:
        raise ReconstructionBlockedError(codex_result.final_message.summary, workspace.root)

    artifacts = compile_author_output(workspace, timer, compile_runner=compile_model)
    runtime_report: dict[str, object] = {
        "returncode": codex_result.returncode,
        "elapsed_seconds": round(codex_result.elapsed_seconds, 6),
        "final_status": codex_result.final_message.status.value,
        "final_summary": codex_result.final_message.summary,
        **author_runtime_metadata(
            preflight=preflight,
            workspace=workspace,
            evidence=evidence,
            image_observation=image_observation,
            timer=timer,
            axial_payloads=read_axial_payloads(payload_dir),
        ),
        "batch": None,
    }
    return ReconstructionAuthorResult(
        status="compiled_pending_promotion",
        run_id=run_workspace.run_id,
        run_workspace=run_workspace,
        author_workspace=workspace,
        codex_result=codex_result,
        artifacts=artifacts,
        runtime_report=runtime_report,
        warnings=tuple(artifacts.warnings),
    )


def _attach_reconstruction_failure_context(
    exc: BaseException,
    *,
    run_workspace: ReconstructionRunWorkspace,
    workspace_dir: Path | None = None,
) -> None:
    if workspace_dir is None:
        workspace_dir = run_workspace.root
    try:
        setattr(exc, "reconstruction_run_id", run_workspace.run_id)
        setattr(exc, "reconstruction_workspace_dir", workspace_dir)
    except (AttributeError, TypeError):
        exc.add_note(f"USDCraft Reconstruction run_id: {run_workspace.run_id}")
        exc.add_note(f"USDCraft Reconstruction workspace_dir: {workspace_dir}")


def reconstruction_failure_context(
    exc: BaseException,
) -> ReconstructionFailureContext | None:
    run_id = getattr(exc, "reconstruction_run_id", None)
    workspace_dir = getattr(exc, "reconstruction_workspace_dir", None)
    if isinstance(run_id, str) and run_id and isinstance(workspace_dir, Path):
        return ReconstructionFailureContext(run_id=run_id, workspace_dir=workspace_dir)
    return None


def execute_reconstruction(
    *,
    preflight: ReconstructionPreflight,
    approved_fingerprint: str | None = None,
    direct: bool = False,
    run_id: str | None = None,
    batch_context: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    environ: Mapping[str, str] | None = None,
    author_process_runner: ProcessRunner | None = None,
    scale_process_runner: Callable[..., Any] | None = None,
) -> ReconstructionResult:
    """Execute a confirmed or explicitly direct Codex request through the same chain.

    The chain after the approval check is the Provider arm's: scale estimate → freeze → mesh
    preprocess → image observation → authoring → Host gate → persist.
    """

    if direct and approved_fingerprint is not None:
        raise ReconstructionApprovalError(
            "Use direct execution or a confirmation fingerprint, not both."
        )
    verify_reconstruction_approval(
        preflight,
        approved_fingerprint=preflight.fingerprint if direct else (approved_fingerprint or ""),
    )
    stage_timer = (
        StageTimer(pipeline=PIPELINE_LABEL, log=progress)
        if progress is not None
        else StageTimer(pipeline=PIPELINE_LABEL)
    )
    scale = resolve_reconstruction_scale(
        preflight, environ=environ, process_runner=scale_process_runner
    )
    resolved = scale.preflight
    with stage_timer.stage("freeze"):
        run_workspace = create_reconstruction_run_workspace(resolved, run_id=run_id)
    stage_timer.attach_trace_path(run_workspace.root / STAGE_TRACE_FILENAME)
    warnings: list[str] = list(scale.warnings)
    try:
        recovery = recover_pending_promotions(StorageRepo(resolved.request.repo_root))
        warnings.extend(f"Promotion recovery: {message}" for message in recovery.errors)
        preprocessing = _build_reconstruction_preprocessing(
            resolved,
            run_workspace,
            stage_timer=stage_timer,
        )
        warnings.extend(preprocessing.warnings)
        author = execute_reconstruction_author(
            preflight=resolved,
            run_workspace=run_workspace,
            evidence=preprocessing.evidence,
            image_observation=preprocessing.image_observation,
            process_runner=author_process_runner,
            scale_note=scale.author_note,
            stage_timer=stage_timer,
        )
        workspace = author.author_workspace
        runtime_report = {
            **author.runtime_report,
            "scale": scale.report(),
            "authorization": "direct" if direct else "fingerprint",
            "batch": dict(batch_context) if batch_context is not None else None,
        }
        record_dir = promote_author_output(
            preflight=resolved,
            run_workspace=run_workspace,
            workspace=workspace,
            evidence=preprocessing.evidence,
            image_observation=preprocessing.image_observation,
            runtime=resolved.request.runtime,
            runtime_report=runtime_report,
            artifacts=author.artifacts,
            authoring=CODEX_RECONSTRUCTION_AUTHORING,
            timer=stage_timer,
            persist_runner=persist_reconstruction,
        )
        warnings.extend(author.warnings)
    except BaseException as exc:
        _attach_reconstruction_failure_context(
            exc,
            run_workspace=run_workspace,
            workspace_dir=run_workspace.author_workspace_base,
        )
        raise

    retained = cleanup_completed_workspaces(
        run_workspace, workspace, keep_workspace=resolved.keep_workspace, warnings=warnings
    )
    return ReconstructionResult(
        status="reconstruction_ready",
        run_id=run_workspace.run_id,
        record_id=record_dir.name,
        record_dir=record_dir,
        model_path=record_dir / "revisions" / REVISION_ID / "model.py",
        workspace_dir=retained,
        warnings=tuple(warnings),
    )


__all__ = [
    "AXIAL_PAYLOAD_DIRNAME",
    "PIPELINE_LABEL",
    "ReconstructionApprovalError",
    "ReconstructionAuthorResult",
    "ReconstructionBlockedError",
    "ReconstructionFailureContext",
    "ReconstructionInputChangedError",
    "ReconstructionPreprocessingBlockedError",
    "ReconstructionPreprocessingResult",
    "ReconstructionResult",
    "ReconstructionScaleResolution",
    "begin_reconstruction_run",
    "execute_reconstruction",
    "execute_reconstruction_author",
    "fingerprint_reconstruction_preflight",
    "preflight_reconstruction",
    "reconstruction_failure_context",
    "resolve_reconstruction_scale",
    "verify_reconstruction_approval",
]
