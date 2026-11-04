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

"""Generation adapter for the shared image-observation stage.

When a confirmed Generation run has an inspiration image, one fresh-context vision call reads that
image and returns one schema-validated semantic inventory of the depicted primary subject. The
engine — schema, deterministic validation, prompt rendering, publishing and the Codex transport —
lives in :mod:`usdcraft.pipeline.observation`; this module owns only the Generation task wording (an
inspiration reading, not an evidence reading), the single-image input, the scope strings, and the
NVIDIA Provider transport. The inventory is an auditable external first reading for the Generation
Author, never binding: the request text wins for presence and the Author is free to deviate where
the request or category coherence requires. A failed or blocked observation degrades the run to
an Author-only reading with a warning; text-only runs skip the stage entirely.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any, Mapping

from usdcraft.pipeline.observation import (
    ImageObservationArtifacts,
    ImageObservationError,
    ObservationProcessRunner,
    ObservationTaskProfile,
    _default_process_runner,
    build_observation_review_task,
    build_observation_task,
    render_inventory_for_prompt,
    run_image_observation_via_codex,
    run_image_observation_via_provider,
)

GENERATION_OBSERVATION_MANIFEST_SCHEMA = "generation_image_observation"


_GENERATION_OBSERVATION_PROFILE = ObservationTaskProfile(
    task_preamble="""You are the one clean-context Image-Observation Agent for a Generation run.

Your only job is to produce one auditable semantic inventory of the primary subject depicted in
the attached inspiration image: its visible Parts, the bounded completion of parts the subject
needs but the image cannot show, and, for each Part, the qualitative geometry cues the image
shows. You are not the later Generation Author: you do not write model code, judge any candidate,
or decide what gets modeled. A later Author combines your inventory with the user's request
text; the text wins for presence and the Author may deviate from the image, so your reading is a
first reading, never binding.""",
    review_preamble="""You are the one clean-context Image-Observation Reviewer for a Generation run.

A first Observation Agent produced the semantic inventory at the end of this task from the same
attached inspiration image. Your only job is to review it against that image and return the full
corrected inventory.""",
    evidence_bullets=(
        "- Attached image 1 is the single inspiration image. You do not receive the request text.",
    ),
    review_evidence_bullets=(
        "- Attached image 1 is the single inspiration image. You do not receive the request text.",
    ),
    prohibitions_paragraph=(
        "Do not inspect any SDK, repository, authoring task, prior attempt, candidate result,\n"
        "evaluator feedback, network source, or other filesystem path. Do not delegate or start\n"
        "another agent."
    ),
    views="the image",
    views_show="the image shows",
    views_plainly_show="the image plainly shows",
    dimension_authority_clause=(
        "the later Author owns dimensional estimates and scale assumptions,\n"
        "  revises them during candidate inspection, and distinguishes image cues from hypotheses.\n"
        "  This inventory supplies qualitative cues rather than competing numeric estimates."
    ),
    completion_trailing_sentence="",
    review_category_knowledge_bullet=(
        "- introduce Parts from category knowledge, except the bounded-completion entries allowed\n"
        "  above."
    ),
    blocked_subject="the image",
)


def build_generation_image_observation_task() -> str:
    """Build the complete task for the Generation observation-only clean-context subagent."""

    return build_observation_task(_GENERATION_OBSERVATION_PROFILE)


def build_generation_image_observation_review_task(*, inventory_text: str) -> str:
    """Build the add-and-sharpen review task for the Generation observation reviewer."""

    return build_observation_review_task(
        _GENERATION_OBSERVATION_PROFILE, inventory_text=inventory_text
    )


def run_generation_image_observation(
    *,
    image_path: Path,
    output_dir: Path,
    model: str | None = None,
    reasoning_effort: str | None = None,
    executable: str | None = None,
    environ: Mapping[str, str] | None = None,
    process_runner: ObservationProcessRunner = _default_process_runner,
) -> ImageObservationArtifacts:
    """Run the Generation Codex-arm observation Agent through the shared Codex engine."""

    return run_image_observation_via_codex(
        image_paths=[Path(image_path)],
        output_dir=output_dir,
        build_task=lambda _staged_names: build_generation_image_observation_task(),
        build_review_task=lambda _staged_names, inventory_text: (
            build_generation_image_observation_review_task(inventory_text=inventory_text)
        ),
        product="generation",
        manifest_schema=GENERATION_OBSERVATION_MANIFEST_SCHEMA,
        scope="generation_image_observation",
        agent_context_schema="generation_image_observation_agent_context",
        agent_context="clean_image_observation_only",
        extra_manifest={"arm": "codex"},
        extra_context={
            "inspiration_image": f"input_1_{Path(image_path).name}",
            "request_text": False,
            "sdk": False,
        },
        model=model,
        reasoning_effort=reasoning_effort,
        executable=executable,
        environ=environ,
        process_runner=process_runner,
    )


def run_generation_image_observation_via_provider(
    *,
    image_path: Path,
    output_dir: Path,
    config: Any,
    cancellation_event: Event | None = None,
    client: Any | None = None,
) -> ImageObservationArtifacts:
    """Run the Generation Provider-arm observation call through the shared Provider transport.

    One user message (task text plus the inspiration image) with a single strict function tool
    whose parameters are the shared observation schema; one add-and-sharpen review call follows.
    Raises ``ImageObservationError`` on failure so the caller degrades to the Author-only reading.
    """

    return run_image_observation_via_provider(
        image_paths=[Path(image_path)],
        output_dir=output_dir,
        build_task=build_generation_image_observation_task,
        build_review_task=lambda inventory_text: build_generation_image_observation_review_task(
            inventory_text=inventory_text
        ),
        manifest_schema=GENERATION_OBSERVATION_MANIFEST_SCHEMA,
        scope="generation_image_observation",
        agent_context="clean_image_observation_only",
        config=config,
        selection_source="provider_config_for_generation_observation",
        cancellation_event=cancellation_event,
        client=client,
    )


__all__ = [
    "GENERATION_OBSERVATION_MANIFEST_SCHEMA",
    "ImageObservationArtifacts",
    "ImageObservationError",
    "build_generation_image_observation_review_task",
    "build_generation_image_observation_task",
    "render_inventory_for_prompt",
    "run_generation_image_observation",
    "run_generation_image_observation_via_provider",
]
