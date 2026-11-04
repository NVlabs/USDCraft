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

from typing import Mapping

from usdcraft.generation.contract import GenerationRequest
from usdcraft.generation.workspace import GenerationWorkspace
from usdcraft.pipeline.codex import (
    CLEAN_CONTEXT_DISABLED_CODEX_FEATURES,
    ProcessRunner,
    _default_process_runner,
    run_codex_task,
)
from usdcraft.pipeline.contracts import CodexRunResult


def run_codex(
    request: GenerationRequest,
    workspace: GenerationWorkspace,
    *,
    prompt: str,
    executable: str | None = None,
    env_overrides: Mapping[str, str] | None = None,
    process_runner: ProcessRunner = _default_process_runner,
) -> CodexRunResult:
    """Run one direct, ephemeral Codex process in the Generation workspace.

    Like the Reconstruction Codex Author, the Generation Author ignores the user's personal Codex
    configuration and runs with the clean-context feature set: one Author, no sub-agents, plugins,
    apps, browser or hooks. It has no time or turn limit.
    """

    attached_images = (
        (workspace.input_image_path,) if workspace.input_image_path is not None else ()
    )
    from usdcraft.generation.render_service import CandidateRenderService

    with CandidateRenderService(workspace.root):
        return run_codex_task(
            request,
            workspace,
            prompt=prompt,
            attached_images=attached_images,
            executable=executable,
            env_overrides=env_overrides,
            process_runner=process_runner,
            ignore_user_config=True,
            disabled_features=CLEAN_CONTEXT_DISABLED_CODEX_FEATURES,
            workspace_environment_variable="USDCRAFT_GENERATION_WORKSPACE",
            unavailable_message=(
                "Codex generation requires the `codex` executable; install/login to Codex CLI "
                "or set USDCRAFT_CODEX_BIN."
            ),
        )
