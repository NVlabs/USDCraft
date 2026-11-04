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

"""Generation's candidate feedback; shared by its Codex and Provider adapters.

These images describe compiled output, never target evidence or a promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping
from xml.etree import ElementTree


def render_candidate(
    asset_dir: Path,
    *,
    out_dir: Path,
    azimuth: float = 45,
    elevation: float = 30,
    joint_positions: Mapping[str, float] | None = None,
    renderer: Callable | None = None,
) -> dict[str, Any]:
    from usdcraft.render.browser import RenderRequest, render_generated_model_views

    positions = dict(joint_positions or {})
    if not all(math.isfinite(v) for v in [azimuth, elevation, *positions.values()]):
        raise ValueError("Camera and joint positions must be finite")
    if not -90 <= elevation <= 90:
        raise ValueError("elevation must be within [-90, 90] degrees")
    urdf = asset_dir / "model.urdf"
    names = {j.attrib["name"] for j in ElementTree.parse(urdf).getroot().findall("joint")}
    if set(positions) - names:
        raise ValueError(f"Unknown joints: {sorted(set(positions) - names)}")
    result = (renderer or render_generated_model_views)(
        RenderRequest(
            asset_dir=asset_dir,
            out_dir=out_dir,
            views=(("author", float(azimuth), float(elevation)),),
            size=768,
            joint_positions=positions,
        )
    )
    payload = {
        "ok": True,
        "images": [str(path.resolve()) for path in result.image_paths],
        "backend": result.backend,
        "fallback_reason": getattr(result, "fallback_reason", None),
        "capabilities": result.capabilities,
        "azimuth": azimuth,
        "elevation": elevation,
        "joint_positions": positions,
        "model_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "scope": "compiled visual geometry; posed kinematically, not a physics simulation",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feedback.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_render_cli(args: argparse.Namespace) -> int:
    """Use the run's Host when launched inside Codex, or render a standalone compiled build."""
    try:
        positions = {}
        for item in args.joint:
            name, value = item.split("=", 1)
            if not name or name in positions:
                raise ValueError("Joint names must be nonempty and unique")
            positions[name] = float(value)
        render: Callable[..., dict[str, Any]] = render_candidate
        if os.environ.get("USDCRAFT_GENERATION_WORKSPACE"):
            from usdcraft.generation.render_service import request_candidate_render

            render = request_candidate_render
        result = render(
            Path(args.asset_dir).expanduser().resolve(),
            out_dir=Path(args.output).expanduser().resolve(),
            azimuth=args.azimuth,
            elevation=args.elevation,
            joint_positions=positions,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
