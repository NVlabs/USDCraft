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

from pathlib import Path
from typing import Mapping, Sequence, TypedDict

from usdcraft.pipeline.codex import (
    CLEAN_CONTEXT_DISABLED_CODEX_FEATURES,
    ProcessRunner,
    run_codex_task,
)
from usdcraft.pipeline.contracts import CodexRunResult
from usdcraft.reconstruction.contract import ReconstructionRequest
from usdcraft.reconstruction.workspace import ReconstructionAuthorWorkspace

# The shared clean-context set, also used by the Generation Codex Author.
RECONSTRUCTION_DISABLED_CODEX_FEATURES: tuple[str, ...] = CLEAN_CONTEXT_DISABLED_CODEX_FEATURES


class _CodexTaskKwargs(TypedDict):
    """``run_codex_task`` keyword arguments shared by both Author invocation paths."""

    prompt: str
    attached_images: Sequence[Path]
    executable: str | None
    env_overrides: Mapping[str, str] | None
    ignore_user_config: bool
    disabled_features: Sequence[str]
    workspace_environment_variable: str
    unavailable_message: str


def run_reconstruction_author(
    request: ReconstructionRequest,
    workspace: ReconstructionAuthorWorkspace,
    *,
    prompt: str,
    executable: str | None = None,
    env_overrides: Mapping[str, str] | None = None,
    process_runner: ProcessRunner | None = None,
    source_usd_path: Path | None = None,
) -> CodexRunResult:
    """Run one fresh Author with only the semantic image initially attached."""

    if workspace.repo_root != request.repo_root:
        raise ValueError("Reconstruction Author workspace and request use different repositories.")
    if workspace.attached_images != (workspace.semantic_image_path,):
        raise ValueError("The Reconstruction Author must receive exactly the semantic image.")
    common: _CodexTaskKwargs = {
        "prompt": prompt,
        "attached_images": workspace.attached_images,
        "executable": executable,
        "env_overrides": env_overrides,
        "ignore_user_config": True,
        "disabled_features": RECONSTRUCTION_DISABLED_CODEX_FEATURES,
        "workspace_environment_variable": "USDCRAFT_RECONSTRUCTION_WORKSPACE",
        "unavailable_message": (
            "Codex Reconstruction requires the `codex` executable; install/login to Codex CLI "
            "or set USDCRAFT_CODEX_BIN."
        ),
    }
    from usdcraft.reconstruction.render_service import CandidateRenderService

    with CandidateRenderService(workspace.root):
        if process_runner is None:
            return run_codex_task(request, workspace, **common)
        return run_codex_task(
            request,
            workspace,
            process_runner=process_runner,
            **common,
        )


__all__ = [
    "RECONSTRUCTION_DISABLED_CODEX_FEATURES",
    "run_reconstruction_author",
]
