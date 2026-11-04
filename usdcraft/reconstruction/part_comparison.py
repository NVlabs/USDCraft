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

"""Author-selected region to named candidate-part comparison, independent of feature detection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from usdcraft.reconstruction._face_grid import FACE_FEATURE_FACES, sample_surface_points
from usdcraft.reconstruction.author_tools import _bounds
from usdcraft.reconstruction.axial import _scene_components
from usdcraft.reconstruction.diagnostic_binding import diagnostic_binding
from usdcraft.reconstruction.local_surface_audit import local_surface_audit
from usdcraft.reconstruction.source import SourceInput, load_normalized_source
from usdcraft.render.offscreen import load_urdf_visuals

MAX_TRIANGLES = 500_000


def _sample(components) -> np.ndarray:
    if sum(len(c.faces) for c in components) > MAX_TRIANGLES:
        raise ValueError("Part comparison exceeds 500000 triangles; select simpler geometry")
    return sample_surface_points(components, target=40_000, seed=7)


def compare_part(
    source: Path,
    asset_dir: Path,
    *,
    out_dir: Path,
    part: str,
    face: str,
    bounds: Sequence[float],
    element: str | None = None,
) -> dict[str, Any]:
    """Compare a q=0 link/visual against a manually chosen source region. No automatic labels."""
    region = _bounds(bounds)
    if region is None:
        raise ValueError("Select source bounds for this part")
    axes = {name: (axis, sign, u, v) for name, axis, sign, u, v in FACE_FEATURE_FACES}
    if face not in axes:
        raise ValueError("face must be one of -X,+X,-Y,+Y,-Z,+Z")
    _, _, u, v = axes[face]
    stage = load_normalized_source(SourceInput(source))
    src = _sample(_scene_components(stage))
    src = src[np.all((src >= region[:3]) & (src <= region[3:]), axis=1)]
    batches = [
        b
        for b in load_urdf_visuals(asset_dir)
        if b.link == part and (element is None or b.element == element)
    ]
    if not batches:
        raise ValueError(f"No candidate visual matches part={part!r}, element={element!r}")
    cand = _sample(batches)
    entry = SimpleNamespace(
        face=face,
        center_uv=((region[:3] + region[3:]) / 2)[[u, v]],
        size_uv=(region[3:] - region[:3])[[u, v]],
    )
    audit = local_surface_audit(src, cand, source_features=[entry])
    sheet = Image.new("RGB", (640, 680), "white")
    draw = ImageDraw.Draw(sheet)
    lo, hi = region[:3][[u, v]], region[3:][[u, v]]
    if len(cand):
        lo = np.minimum(lo, cand[:, [u, v]].min(axis=0))
        hi = np.maximum(hi, cand[:, [u, v]].max(axis=0))
    scale = 540 / max(float((hi - lo).max()), 1e-9)
    for points, color in [(src, (30, 100, 200)), (cand, (210, 110, 30))]:
        for point in points[np.linspace(0, len(points) - 1, min(len(points), 4096), dtype=int)]:
            x, y = (point[[u, v]] - lo) * scale
            draw.ellipse((40 + x, 600 - y, 42 + x, 602 - y), fill=color)
    draw.text((15, 12), f"{part} / {element or 'all visuals'}; q=0; face {face}", fill="black")
    draw.text(
        (15, 635),
        f"Blue=selected source; orange=candidate; {'XYZ'[u]}/{'XYZ'[v]} axes",
        fill="black",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    image = out_dir / "part_comparison.png"
    sheet.save(image)
    result = {
        "ok": True,
        "part": part,
        "element": element,
        "face": face,
        "source_bounds_m": region.tolist(),
        "source_sample_count": len(src),
        "candidate_bounds_m": [cand.min(axis=0).tolist(), cand.max(axis=0).tolist()]
        if len(cand)
        else None,
        "binding": diagnostic_binding(source, asset_dir / "isaac/model.usdc"),
        "local_surface": audit["regions"][0],
        "images": [str(image.resolve())],
        "interpretation": "Author-selected correspondence, not validated segmentation. Positive depth offset is outward; centred profile residual separates shape from translation on common columns. Inspect projection, selection, curvature and q=0 pose; empty/sparse source is unassessed, never permission to omit a visible part.",
    }
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
