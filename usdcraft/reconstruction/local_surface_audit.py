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

"""Bounded local first-surface readings at source anchors, shared by both Recon arms.

Regions are geometric evidence, not Part segmentation or automatic semantic findings. Unknown
surface stays unassessed. The Author owns correspondence, pose interpretation and any repair.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from usdcraft.reconstruction._face_grid import FACE_FEATURE_FACES

MAX_REGIONS = 32
MAX_SIDE_CELLS = 64
MIN_COMMON_COLUMNS = 8
MIN_PITCH_M = 0.001
_FACE_AXES = {name: (axis, sign, u, v) for name, axis, sign, u, v in FACE_FEATURE_FACES}


def _surface_grid(
    points: np.ndarray,
    *,
    axes: tuple[int, int, int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    pitch: float,
    shape: tuple[int, int],
) -> np.ndarray:
    axis, sign, u, v = axes
    grid = np.full(shape, -np.inf)
    if not len(points):
        return grid
    uv = points[:, [u, v]]
    selected = np.all((uv >= lower) & (uv < upper), axis=1)
    if not selected.any():
        return grid
    cells = np.floor((uv[selected] - lower) / pitch).astype(np.int64)
    cells = np.minimum(cells, np.asarray(shape) - 1)
    np.maximum.at(grid, (cells[:, 0], cells[:, 1]), points[selected, axis] * sign)
    return grid


def local_surface_audit(
    source_points: np.ndarray,
    candidate_points: np.ndarray,
    *,
    source_features: Sequence[Any] = (),
    source_openings: Sequence[Any] = (),
) -> dict[str, Any]:
    """Compare bounded face regions without resampling meshes or changing global diagnostics.

    The pitch is bounded by region extent and available source sample density. Depth residual
    includes placement; subtracting its median isolates profile variation on common columns.
    Neither statistic establishes Part identity, source occupancy or a semantic shape verdict.
    """

    anchors = [("feature", index, entry) for index, entry in enumerate(source_features)] + [
        ("opening", index, entry)
        for index, entry in enumerate(source_openings)
        if not getattr(entry, "open_face", False)
    ]
    rows: list[dict[str, Any]] = []
    for kind, index, entry in anchors[:MAX_REGIONS]:
        centre = np.asarray(entry.center_uv, dtype=float)
        size = np.asarray(entry.size_uv, dtype=float)
        row: dict[str, Any] = {
            "anchor": f"{kind}:{index}",
            "face": entry.face,
            "center_uv_m": centre.tolist(),
            "size_uv_m": size.tolist(),
            "status": "unassessed",
        }
        rows.append(row)
        if (
            entry.face not in _FACE_AXES
            or not np.isfinite(centre).all()
            or not np.isfinite(size).all()
            or np.any(size <= 0)
        ):
            row["reason"] = "invalid_region"
            continue
        if not len(source_points):
            row["reason"] = "source_surface_unavailable"
            continue
        axes = _FACE_AXES[entry.face]
        _, _, u, v = axes
        lo, hi = centre - size / 2, centre + size / 2
        uv = source_points[:, [u, v]]
        count = int(np.count_nonzero(np.all((uv >= lo) & (uv <= hi), axis=1)))
        # A density estimate controls sparse first-surface fall-through; at most 64 cells/side,
        # including a one-cell margin. Raw samples behind the face are not free-space evidence.
        pitch = max(
            MIN_PITCH_M,
            float(size.max()) / (MAX_SIDE_CELLS - 2),
            float(np.sqrt(float(np.prod(size)) * 4.0 / max(count, 1))),
        )
        lower, upper = lo - pitch, hi + pitch
        shape = tuple(
            int(x)
            for x in np.minimum(MAX_SIDE_CELLS, np.maximum(1, np.ceil((upper - lower) / pitch)))
        )
        kwargs: dict[str, Any] = dict(axes=axes, lower=lower, upper=upper, pitch=pitch, shape=shape)
        src = _surface_grid(source_points, **kwargs)
        cand = _surface_grid(candidate_points, **kwargs)
        source_has, candidate_has = np.isfinite(src), np.isfinite(cand)
        common = source_has & candidate_has
        ns, nc, nb = (int(np.count_nonzero(mask)) for mask in (source_has, candidate_has, common))
        row.update(
            {
                "pitch_m": round(pitch, 6),
                "source_columns": ns,
                "candidate_columns": nc,
                "common_columns": nb,
                "source_column_coverage": round(nb / ns, 4) if ns else None,
                "candidate_only_columns": int(np.count_nonzero(candidate_has & ~source_has)),
                "missing_candidate_columns": int(np.count_nonzero(source_has & ~candidate_has)),
            }
        )
        if ns < MIN_COMMON_COLUMNS:
            row["reason"] = "insufficient_source_surface"
            continue
        if nb < MIN_COMMON_COLUMNS or nb / ns < 0.25:
            row["reason"] = "insufficient_common_surface"
            continue
        delta = cand[common] - src[common]
        offset = float(np.median(delta))
        row.update(
            {
                "status": "measured",
                "depth_offset_mm": round(offset * 1000, 2),
                "surface_delta_p90_mm": round(float(np.percentile(np.abs(delta), 90)) * 1000, 2),
                "centred_profile_p90_mm": round(
                    float(np.percentile(np.abs(delta - offset), 90)) * 1000, 2
                ),
            }
        )
    return {
        "scope": "source_anchored_first_surface_regions_not_part_segmentation",
        "regions": rows,
        "omitted_regions": max(0, len(anchors) - MAX_REGIONS),
        "note": (
            "Readings only: map anchors to the semantic evidence table. Overlapping regions, "
            "occlusion, scan noise and pose differences can explain residuals. Missing source "
            "evidence is unassessed, not a missing-Part finding. Unknown interiors are not scored."
        ),
    }


def format_local_surfaces(
    audit: dict[str, Any], *, source_bounds: Sequence[float] | None = None
) -> list[str]:
    rows = audit.get("regions", [])
    if not rows:
        return []
    lines = [
        "local_surfaces (source-anchored visible patches, not Part segmentation or DEFECTS; "
        "map face/anchor to a visible part; no source evidence means unassessed):"
    ]
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("status") != "measured",
            -float(row.get("centred_profile_p90_mm", 0)),
            -abs(float(row.get("depth_offset_mm", 0))),
        ),
    )
    for row in ranked[:6]:
        prefix = f"  {row['anchor']} {row['face']} c={row['center_uv_m']} size={row['size_uv_m']}"
        if row["status"] != "measured":
            lines.append(f"{prefix}: unassessed ({row['reason']})")
        else:
            lines.append(
                f"{prefix}: coverage={row['source_column_coverage']:.0%}, "
                f"depth_offset={row['depth_offset_mm']} mm, "
                f"surface_p90={row['surface_delta_p90_mm']} mm, "
                f"centred_profile_p90={row['centred_profile_p90_mm']} mm, "
                f"pitch={row['pitch_m'] * 1000:.2f} mm"
            )
        if source_bounds is not None and row["face"] in _FACE_AXES:
            _, _, u, v = _FACE_AXES[row["face"]]
            bounds = list(source_bounds)
            for index, axis in enumerate((u, v)):
                half_size = row["size_uv_m"][index] / 2
                bounds[axis] = row["center_uv_m"][index] - half_size
                bounds[axis + 3] = row["center_uv_m"][index] + half_size
            if all(math.isfinite(x) for x in bounds) and all(
                bounds[i] < bounds[i + 3] for i in range(3)
            ):
                # Round outwards so the displayed ROI retains boundary vertices.
                values = [math.floor(x * 1e6) / 1e6 for x in bounds[:3]] + [
                    math.ceil(x * 1e6) / 1e6 for x in bounds[3:]
                ]
                lines.append(
                    f"    ROI --face={row['face']} --bounds " + " ".join(f"{x:.6f}" for x in values)
                )
    if len(rows) > 6:
        lines.append(
            f"  {len(rows) - 6} more patches in the full diagnostic JSON; shown by profile residual, not semantic importance."
        )
    if audit.get("omitted_regions"):
        lines.append(f"  {audit['omitted_regions']} further regions omitted by the bounded audit.")
    lines.append(
        "  Large depth_offset: check placement. Large centred_profile: inspect curvature, thickness or pose. High coverage alone does not establish correct shape. ROI bounds span source depth, not a segmented part; use their six values for measure_mesh or compare_part after selecting the candidate part. Occlusion and scan noise remain possible explanations."
    )
    return lines
