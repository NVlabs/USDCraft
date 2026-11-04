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

"""Generation bindings for the shared Host render service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from usdcraft.render.service import CandidateRenderService as SharedCandidateRenderService
from usdcraft.render.service import RenderServiceConfig
from usdcraft.render.service import request_candidate_render as request_render

CONFIG = RenderServiceConfig(
    workspace_environment_variable="USDCRAFT_GENERATION_WORKSPACE",
    label="Generation",
    execution="generation_host",
    require_build_directory=True,
)


class CandidateRenderService(SharedCandidateRenderService):
    def __init__(self, root: Path) -> None:
        from usdcraft.generation.candidate_render import render_candidate

        super().__init__(root, config=CONFIG, render_candidate=render_candidate)


def request_candidate_render(
    asset_dir: Path,
    *,
    out_dir: Path,
    azimuth: float,
    elevation: float,
    joint_positions: dict[str, float],
) -> dict[str, Any]:
    return request_render(
        asset_dir,
        out_dir=out_dir,
        azimuth=azimuth,
        elevation=elevation,
        joint_positions=joint_positions,
        config=CONFIG,
    )
