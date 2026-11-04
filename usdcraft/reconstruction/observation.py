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

"""Reconstruction adapter for the shared image-observation stage.

After the deterministic mesh-preprocess stage, formal Reconstruction runs one fresh-context vision
call that reads only the declared semantic image — the photographed record of the subject being
reconstructed; the source mesh is a separately produced approximation and never enters this
reading — and returns one schema-validated semantic inventory of the primary subject. The engine —
schema, deterministic validation, prompt rendering, publishing and the Codex transport — lives in
:mod:`usdcraft.pipeline.observation`; this module owns only the Reconstruction task wording, image
set and scope strings. The inventory is an auditable external first reading for the
Reconstruction Author, never evidence authority: it adds no visibility, proves no absence, and the
semantic image wins every conflict. A failed or blocked observation degrades the run to an
Author-only reading with a warning; it never blocks the run and never enters the approval
fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any, Mapping, Sequence

from usdcraft.pipeline.observation import (
    OBSERVATION_AGENT_RESULT_SCHEMA,
    ImageObservationArtifacts,
    ImageObservationError,
    ObservationProcessResult,
    ObservationProcessRunner,
    ObservationTaskProfile,
    _default_process_runner,
    build_observation_codex_command,
    build_observation_review_task,
    build_observation_task,
    parse_observation_result,
    render_inventory_for_prompt,
    run_image_observation_via_codex,
)
from usdcraft.pipeline.observation import (
    resolve_observation_runtime as _resolve_shared_observation_runtime,
)
from usdcraft.pipeline.observation import (
    run_image_observation_via_provider as _run_image_observation_via_provider,
)
from usdcraft.reconstruction.contract import SEMANTIC_PROVENANCE_ROLE

OBSERVATION_MANIFEST_SCHEMA = "reconstruction_image_observation"


def resolve_observation_runtime(
    *,
    model: str | None,
    reasoning_effort: str | None,
    environ: Mapping[str, str] | None = None,
):
    """Resolve the Reconstruction observation runtime through the shared engine."""

    return _resolve_shared_observation_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        product="reconstruction",
        environ=environ,
    )


def _observation_profile(*, agent_name: str = "Codex") -> ObservationTaskProfile:
    """Build the Reconstruction semantic parameters for the shared observation task text.

    ``agent_name`` names the transport in the agent's self-description (``"Codex"`` for the Codex
    arm; empty for the Provider arm, whose agent is simply the Image-Observation Agent). The
    semantic image is the reconstruction target's semantic record, so its evidence bullet names
    the image as the subject's semantic record.
    """

    agent = f"{agent_name.strip()} " if agent_name.strip() else ""
    semantic_bullet = (
        "- Attached image 1 is the single semantic image of the subject being reconstructed."
        " It is\n  the only semantic channel."
    )
    return ObservationTaskProfile(
        task_preamble=f"""You are the one clean-context {agent}Image-Observation Agent for a Reconstruction run.

Your only job is to produce one auditable semantic inventory of the primary subject from the
attached semantic image: its visible Parts, the bounded completion of parts the subject needs but
the image cannot show, and, for each Part, the qualitative geometry cues the image shows. You
are not the later Reconstruction Author: you do not write model code, judge any candidate, or
decide what gets modeled. A later Author interprets your inventory alongside the image and source geometry and may
revise names, counts, grouping and proposed attachments. Your reading is advisory, never authority.""",
        review_preamble=f"""You are the one clean-context {agent}Image-Observation Reviewer for a Reconstruction run.

A first Observation Agent produced the semantic inventory at the end of this task from the same
attached image. Your only job is to review it against attached image 1 and return the full
corrected inventory.""",
        evidence_bullets=(semantic_bullet,),
        review_evidence_bullets=(semantic_bullet,),
        prohibitions_paragraph=(
            "Do not inspect any source or normalized mesh, AXIAL or voxel data, SDK,"
            " Reconstruction authoring\ntask, prior attempt, candidate result, Viewer or evaluator"
            " feedback, GT code, repository, network\nsource, or other filesystem path. Do not"
            " delegate or start another agent."
        ),
        views="the image",
        views_show="the image shows",
        views_plainly_show="the image plainly shows",
        dimension_authority_clause=(
            "the Author obtains metric support from source mesh tools; an image-only numeric "
            "estimate here would anchor its construction without measured support."
        ),
        completion_trailing_sentence=(
            " Every `inferred` entry is a hypothesis for the Author, never evidence."
        ),
        review_category_knowledge_bullet=(
            "- introduce Parts from category knowledge (except the bounded-completion entries\n"
            "  allowed above)."
        ),
        blocked_subject="the semantic image",
    )


def build_image_observation_task(*, agent_name: str = "Codex") -> str:
    """Build the complete task for the observation-only clean-context subagent."""

    return build_observation_task(_observation_profile(agent_name=agent_name))


def build_image_observation_review_task(
    *,
    inventory_text: str,
    agent_name: str = "Codex",
) -> str:
    """Build the add-and-sharpen review task for the observation-review clean-context subagent."""

    return build_observation_review_task(
        _observation_profile(agent_name=agent_name),
        inventory_text=inventory_text,
    )


def run_image_observation(
    *,
    semantic_image_path: Path,
    output_dir: Path,
    model: str | None = None,
    reasoning_effort: str | None = None,
    executable: str | None = None,
    environ: Mapping[str, str] | None = None,
    process_runner: ObservationProcessRunner = _default_process_runner,
) -> ImageObservationArtifacts:
    """Run one clean-context Reconstruction observation Agent through the shared Codex engine.

    Retries the complete call once on an invalid response. Raises ``ImageObservationError`` when
    the Agent fails, blocks, or returns an invalid inventory twice; callers degrade to the
    Author-only reading and never fail the run for it. The call has no time limit.
    """

    def build_task(staged_names: Sequence[str]) -> str:
        return build_image_observation_task()

    def build_review_task(staged_names: Sequence[str], inventory_text: str) -> str:
        return build_image_observation_review_task(inventory_text=inventory_text)

    return run_image_observation_via_codex(
        image_paths=[Path(semantic_image_path)],
        output_dir=output_dir,
        build_task=build_task,
        build_review_task=build_review_task,
        product="reconstruction",
        manifest_schema=OBSERVATION_MANIFEST_SCHEMA,
        scope="reconstruction_image_observation",
        agent_context_schema="reconstruction_image_observation_agent_context",
        agent_context="clean_image_observation_only",
        extra_manifest={
            "authority": "external_first_reading_not_evidence",
            "semantic_provenance_role": SEMANTIC_PROVENANCE_ROLE,
        },
        extra_context={
            "authority": "external_first_reading_not_evidence",
            "semantic_image": f"input_1_{Path(semantic_image_path).name}",
            "semantic_provenance_role": SEMANTIC_PROVENANCE_ROLE,
            "source_mesh": False,
            "axial": False,
            "sdk": False,
        },
        model=model,
        reasoning_effort=reasoning_effort,
        executable=executable,
        environ=environ,
        process_runner=process_runner,
    )


def run_image_observation_via_provider(
    *,
    semantic_image_path: Path,
    output_dir: Path,
    config: Any,
    cancellation_event: Event | None = None,
    client: Any | None = None,
) -> ImageObservationArtifacts:
    """Run the Reconstruction observation Agent through the shared NVIDIA Provider transport.

    Only the semantic image travels in the user message. Task wording, schema, review merge and
    publishing are shared with the Codex arm; only the transport differs.
    """

    return _run_image_observation_via_provider(
        image_paths=[Path(semantic_image_path)],
        output_dir=output_dir,
        build_task=lambda: build_image_observation_task(agent_name=""),
        build_review_task=lambda inventory_text: build_image_observation_review_task(
            inventory_text=inventory_text, agent_name=""
        ),
        manifest_schema=OBSERVATION_MANIFEST_SCHEMA,
        scope="reconstruction_image_observation",
        agent_context="clean_image_observation_only",
        config=config,
        selection_source="provider_config_for_reconstruction_observation",
        extra_manifest={
            "authority": "external_first_reading_not_evidence",
            "semantic_provenance_role": SEMANTIC_PROVENANCE_ROLE,
        },
        cancellation_event=cancellation_event,
        client=client,
    )


__all__ = [
    "OBSERVATION_AGENT_RESULT_SCHEMA",
    "OBSERVATION_MANIFEST_SCHEMA",
    "ImageObservationArtifacts",
    "ImageObservationError",
    "ObservationProcessResult",
    "build_image_observation_review_task",
    "build_image_observation_task",
    "build_observation_codex_command",
    "parse_observation_result",
    "render_inventory_for_prompt",
    "resolve_observation_runtime",
    "run_image_observation",
    "run_image_observation_via_provider",
]
