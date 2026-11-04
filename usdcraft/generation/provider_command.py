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

import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from threading import Event
from typing import Any

from storage.record_mutations import recover_pending_promotions
from storage.repo import StorageRepo
from usdcraft._images import validate_decodable_image
from usdcraft.compile import compile_model
from usdcraft.generation.command import (
    GenerationResult,
    _attach_failure_context,
)
from usdcraft.generation.default_prompt import resolve_generation_prompt
from usdcraft.generation.observation import (
    ImageObservationError,
    render_inventory_for_prompt,
    run_generation_image_observation_via_provider,
)
from usdcraft.generation.persist import (
    REVISION_ID,
    persist_generation,
    provider_authoring_identity,
)
from usdcraft.generation.prompt import build_generation_prompt, generation_cache_prefix
from usdcraft.generation.provider_contract import ProviderGenerationRequest
from usdcraft.generation.provider_tools import GenerationToolExecutor, build_generation_tool_schemas
from usdcraft.generation.workspace import create_generation_workspace
from usdcraft.pipeline.contracts import ProviderRuntimeSelection
from usdcraft.pipeline.execution import (
    _provider_check_runner,
    _provider_compile_runner,
    _provider_selection_source,
    _require_isolated_provider_compile_worker,
    compile_author_output,
)
from usdcraft.pipeline.provider import (
    DEFAULT_PROVIDER_MAX_TURNS,
    NvidiaProviderConfig,
    NvidiaResponsesClient,
    ProviderHarness,
    ProviderProgressCallback,
)
from usdcraft.pipeline.provider.progress import emit_provider_progress
from usdcraft.pipeline.sdk_context import PRELOADED_SDK_DOC_PATHS
from usdcraft.stagetime import STAGE_TRACE_FILENAME, StageTimer


class ProviderGenerationBlockedError(RuntimeError):
    def __init__(self, summary: str, workspace_dir: Path) -> None:
        super().__init__(f"Provider reported a blocker: {summary} (workspace: {workspace_dir})")
        self.summary = summary
        self.workspace_dir = workspace_dir


class ProviderGenerationCancelledError(RuntimeError):
    """A Batch coordinator requested cancellation before canonical promotion."""


def _raise_if_provider_cancelled(cancellation_event: Event | None) -> None:
    if cancellation_event is not None and cancellation_event.is_set():
        raise ProviderGenerationCancelledError(
            "Provider Generation was cancelled by the local Batch coordinator."
        )


PROGRESS_LABEL = "Provider Generation"


def execute_provider_generation(
    *,
    repo_root: Path,
    prompt: str | None = None,
    image: Path | None = None,
    model: str | None = None,
    reasoning: str | None = None,
    timeout_seconds: float | None = None,
    label: str | None = None,
    keep_workspace: bool = False,
    progress: ProviderProgressCallback | None = None,
    cancellation_event: Event | None = None,
    batch_context: Mapping[str, object] | None = None,
    run_id: str | None = None,
    api_style: str = "responses",
) -> GenerationResult:
    """Directly execute one NVIDIA Provider item and use the shared Host promotion path."""

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"USDCraft repository root is not a directory: {root}")
    _raise_if_provider_cancelled(cancellation_event)
    _require_isolated_provider_compile_worker()
    resolved_image = Path(image).expanduser().resolve() if image is not None else None
    if resolved_image is not None:
        validate_decodable_image(resolved_image, label="Generation inspiration image")
    resolved_prompt = resolve_generation_prompt(
        prompt=prompt,
        image_path=resolved_image,
    )
    config_kwargs: dict[str, Any] = {
        "model": model,
        "reasoning_effort": reasoning,
    }
    if api_style != "responses":
        config_kwargs["api_style"] = api_style
    if timeout_seconds is not None:
        config_kwargs["timeout_seconds"] = float(timeout_seconds)
    config = NvidiaProviderConfig.from_repo(root, **config_kwargs)
    runtime = ProviderRuntimeSelection(
        model_id=config.model,
        reasoning_effort=config.reasoning_effort,
        runtime_source=config.runtime_source,
        selection_source=_provider_selection_source(model=model, reasoning=reasoning),
    )
    generation_started = time.monotonic()
    emit_provider_progress(
        progress,
        f"{PROGRESS_LABEL} | activity=starting | model={config.model!r} | "
        f"effort={config.reasoning_effort!r} | max_turns={DEFAULT_PROVIDER_MAX_TURNS}",
    )
    request = ProviderGenerationRequest(
        prompt=resolved_prompt,
        repo_root=root,
        image_path=resolved_image,
        runtime=runtime,
        timeout_seconds=config.timeout_seconds,
        endpoint=config.endpoint,
    )
    normalized_label = str(label).strip() if label is not None else None
    if normalized_label == "":
        normalized_label = None
    # Recovery is the same storage-owned transaction recovery used by confirmed Codex Generation.
    _raise_if_provider_cancelled(cancellation_event)
    promotion_recovery = recover_pending_promotions(StorageRepo(root))
    # The stage timer narrates every Host stage (freeze, image observation, author,
    # host gate, persist) into the same progress display as the harness and writes the uniform
    # stage_timings report plus the stage_trace.jsonl sidecar shared with the Codex arm.
    stage_timer = StageTimer(
        pipeline="Generation Provider",
        log=lambda message: emit_provider_progress(progress, message),
        log_label=PROGRESS_LABEL,
    )
    with stage_timer.stage("freeze"):
        workspace = (
            create_generation_workspace(request)
            if run_id is None
            else create_generation_workspace(request, run_id=run_id)
        )
    stage_timer.attach_trace_path(workspace.root / STAGE_TRACE_FILENAME)
    emit_provider_progress(
        progress,
        f'{PROGRESS_LABEL} | activity="workspace ready" | run_id={workspace.run_id!r}',
    )
    try:
        _raise_if_provider_cancelled(cancellation_event)
        # Image observation is one clean-context vision reading of the staged inspiration image
        # through the same Provider transport. It is an external first reading for the Author,
        # never binding: a failed or blocked observation degrades the run to the Author-only
        # reading with a warning, and text-only runs skip the stage entirely.
        image_observation = None
        observation_warnings: list[str] = []
        if workspace.input_image_path is not None:
            try:
                with stage_timer.stage("image_observation"):
                    image_observation = run_generation_image_observation_via_provider(
                        image_path=workspace.input_image_path,
                        output_dir=workspace.root / "image_observation",
                        config=config,
                        cancellation_event=cancellation_event,
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
                emit_provider_progress(
                    progress,
                    f'{PROGRESS_LABEL} | activity="image observation degraded" | '
                    f"error_type={type(exc).__name__}",
                )
            _raise_if_provider_cancelled(cancellation_event)
        task = build_generation_prompt(
            request,
            workspace,
            check_command="check_model",
            compile_command="compile_model",
            agent_name="Provider",
            reference_access="provider_tools",
            turn_budget=DEFAULT_PROVIDER_MAX_TURNS,
            image_observation_text=(
                render_inventory_for_prompt(image_observation.inventory)
                if image_observation is not None
                else None
            ),
        )
        tool_executor = GenerationToolExecutor(
            workspace,
            check_runner=_provider_check_runner,
            compile_runner=_provider_compile_runner,
        )
        harness = ProviderHarness(
            progress_label="Provider Generation",
            workspace=workspace,
            client=NvidiaResponsesClient(config, cancellation_event=cancellation_event),
            tool_executor=tool_executor,
            tool_schemas=build_generation_tool_schemas(),
            max_turns=DEFAULT_PROVIDER_MAX_TURNS,
            progress=progress,
            cancellation_event=cancellation_event,
        )
        with stage_timer.stage("author"):
            harness_result = harness.run(
                prompt=task,
                image_path=workspace.input_image_path,
                cache_prefix=generation_cache_prefix(task),
            )
        _raise_if_provider_cancelled(cancellation_event)
        if harness_result.status == "blocked":
            raise ProviderGenerationBlockedError(harness_result.summary, workspace.root)

        # This is the same one clean mechanical/structural Host gate used by Codex Generation.
        artifacts = compile_author_output(workspace, stage_timer, compile_runner=compile_model)
        _raise_if_provider_cancelled(cancellation_event)
        runtime_report = {
            **({"api_style": config.api_style} if config.api_style != "responses" else {}),
            **harness_result.runtime_report(),
            "stage_timings": stage_timer.report(),
            # Batch membership (batch_id, csv, row_id, ...) when a batch command ran this item;
            # None for a single-item run. Recorded for grouping in the Viewer only.
            "batch": dict(batch_context) if batch_context else None,
            "preloaded_docs": list(PRELOADED_SDK_DOC_PATHS),
            "image_observation": (
                "external_inventory"
                if image_observation is not None
                else ("author_only" if workspace.input_image_path is not None else "no_image")
            ),
            "candidate_render": {"enabled": True, "calls": tool_executor.render_calls},
            "author_validation": tool_executor.validation_state(),
        }
        with stage_timer.stage("persist"):
            record_dir = persist_generation(
                repo_root=root,
                run_id=workspace.run_id,
                prompt=request.prompt,
                image_path=workspace.input_image_path,
                label=normalized_label,
                runtime=request.runtime,
                workspace_dir=workspace.root,
                runtime_report=runtime_report,
                artifacts=artifacts,
                authoring_identity=provider_authoring_identity(config.endpoint),
            )
        emit_provider_progress(
            progress,
            f'{PROGRESS_LABEL} | activity="canonical record promoted" | '
            f"record_id={record_dir.name!r}",
        )
        model_path = record_dir / "revisions" / REVISION_ID / "model.py"
        warnings = list(artifacts.warnings)
        warnings.extend(observation_warnings)
        for message in promotion_recovery.errors:
            warnings.append(f"Promotion recovery: {message}")

        retained_workspace: Path | None = workspace.root
        if not keep_workspace:
            try:
                shutil.rmtree(workspace.root)
                retained_workspace = None
            except OSError as exc:
                warnings.append(
                    "Promotion succeeded but the staging workspace could not be removed: "
                    f"{workspace.root} ({exc})"
                )
        emit_provider_progress(
            progress,
            f"{PROGRESS_LABEL} | activity=completed | status=generation_ready | "
            f"record_id={record_dir.name!r} | elapsed={time.monotonic() - generation_started:.2f}s",
        )
        return GenerationResult(
            status="generation_ready",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=model_path,
            workspace_dir=retained_workspace,
            warnings=tuple(warnings),
            run_id=workspace.run_id,
            provider=config.endpoint,
        )
    except BaseException as exc:
        emit_provider_progress(
            progress,
            f"{PROGRESS_LABEL} | activity=failed | error_type={type(exc).__name__} | "
            f"elapsed={time.monotonic() - generation_started:.2f}s",
        )
        _attach_failure_context(exc, workspace=workspace)
        raise


__all__ = [
    "ProviderGenerationBlockedError",
    "ProviderGenerationCancelledError",
    "execute_provider_generation",
]
