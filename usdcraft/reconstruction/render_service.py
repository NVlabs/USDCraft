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

"""Reconstruction bindings for the shared Host render service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from usdcraft.render.service import CandidateRenderService as SharedCandidateRenderService
from usdcraft.render.service import RenderServiceConfig
from usdcraft.render.service import request_candidate_render as request_render

CONFIG = RenderServiceConfig(
    workspace_environment_variable="USDCRAFT_RECONSTRUCTION_WORKSPACE",
    label="Reconstruction",
    execution="reconstruction_host",
    require_build_directory=False,
)


class CandidateRenderService(SharedCandidateRenderService):
    def __init__(self, root: Path) -> None:
        from usdcraft.reconstruction.author_tools import render_candidate, render_source

        def render(asset_dir: Path, *, joint_positions: dict, **kwargs: Any) -> dict:
            if asset_dir.resolve() == root.resolve() / "inputs" / "source_render":
                if joint_positions:
                    raise ValueError("Frozen source rendering has no candidate joint poses")
                return render_source(asset_dir, **kwargs)
            if asset_dir.resolve() != root.resolve() / "build":
                raise ValueError("Render input must be the staged source or current candidate")
            return render_candidate(asset_dir, joint_positions=joint_positions, **kwargs)

        super().__init__(root, config=CONFIG, render_candidate=render)


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


def request_source_render(
    asset_dir: Path, *, out_dir: Path, azimuth: float, elevation: float
) -> dict[str, Any]:
    root = Path(os.environ[CONFIG.workspace_environment_variable]).resolve()
    if asset_dir.resolve() != root / "inputs" / "source_render":
        raise ValueError("Source render input must be this run's staged source carrier")
    return request_candidate_render(
        asset_dir, out_dir=out_dir, azimuth=azimuth, elevation=elevation, joint_positions={}
    )
