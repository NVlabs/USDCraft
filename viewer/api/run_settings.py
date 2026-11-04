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

"""Read-only run-setting projections shared by the Generation and Reconstruction stores.

Every value is derived from what a promoted record already stores (``record.json``, the active
revision's provenance runtime and, for Reconstruction, ``reconstruction.json``); nothing here
writes or re-runs.
"""

from __future__ import annotations

from typing import Any, Literal

ProviderApiStyle = Literal["responses", "chat-completions"]
MeshCleanupStatus = Literal["run", "skipped"]
GuidanceStatus = Literal["present", "absent", "unrecorded"]
RequestPromptKind = Literal["default", "custom"]

# The Host's default Reconstruction request (`usdcraft/reconstruction/default_prompt.py`); the
# Viewer keeps its own copy so it does not import the Host, and a test pins the two together.
DEFAULT_RECONSTRUCTION_PROMPT = (
    "Reconstruct the primary subject object shown in the semantic image and the source mesh as a "
    "complete, functionally usable, simulator-ready 3D asset."
)

IMAGE_OBSERVATION_MODES = frozenset({"external_inventory", "author_only", "disabled", "no_image"})


def _string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def provider_api_style(
    agent_harness: str | None, runtime: dict[str, Any], runtime_source: str | None
) -> ProviderApiStyle | None:
    """The Provider wire protocol; ``None`` for the Codex arm, which has no API style.

    The Provider runtime report carries ``api_style`` only when it differs from the default
    ``responses``; the runtime source (``nvidia-chat-completions-api``) names it as well.
    """

    if agent_harness != "provider":
        return None
    declared = runtime.get("api_style")
    if declared in ("responses", "chat-completions"):
        return declared
    if runtime_source is not None and "chat-completions" in runtime_source:
        return "chat-completions"
    return "responses"


def image_observation_mode(runtime: dict[str, Any]) -> str | None:
    """How the image reading reached the Author (`external_inventory`, `author_only`, ...)."""

    mode = runtime.get("image_observation")
    return mode if isinstance(mode, str) and mode in IMAGE_OBSERVATION_MODES else None


def recorded_max_turns(agent_harness: str | None, runtime: dict[str, Any]) -> int | None:
    """The enforced Provider turn budget; the Codex arm has none."""

    if agent_harness != "provider":
        return None
    harness = runtime.get("harness")
    value = harness.get("max_turns") if isinstance(harness, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def mesh_cleanup_status(runtime: dict[str, Any]) -> MeshCleanupStatus | None:
    """Whether the Subject-Isolation Agent ran, read from the recorded preprocessing stage."""

    stages = runtime.get("stage_timings")
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("stage") != "subject_isolation":
            continue
        detail = _string(stage.get("detail")) or ""
        if detail == "agent":
            return "run"
        if "skipped" in detail:
            return "skipped"
        return None
    return None


def request_prompt_kind(request_text: str | None) -> RequestPromptKind | None:
    """``default`` when the stored request is the Host's default Reconstruction request."""

    text = _string(request_text)
    if text is None:
        return None
    return "default" if text == DEFAULT_RECONSTRUCTION_PROMPT else "custom"


def recorded_guidance(reconstruction: dict[str, Any] | None) -> tuple[GuidanceStatus, str | None]:
    """The user guidance recorded in ``reconstruction.json`` ``input.guidance``.

    ``null`` means the run had no guidance; a record without the field (promoted before the Host
    recorded it) is ``unrecorded``.
    """

    payload = reconstruction.get("input") if isinstance(reconstruction, dict) else None
    if not isinstance(payload, dict) or "guidance" not in payload:
        return "unrecorded", None
    text = _string(payload["guidance"])
    return ("present", text) if text is not None else ("absent", None)


__all__ = [
    "DEFAULT_RECONSTRUCTION_PROMPT",
    "GuidanceStatus",
    "MeshCleanupStatus",
    "ProviderApiStyle",
    "RequestPromptKind",
    "image_observation_mode",
    "mesh_cleanup_status",
    "provider_api_style",
    "recorded_guidance",
    "recorded_max_turns",
    "request_prompt_kind",
]
