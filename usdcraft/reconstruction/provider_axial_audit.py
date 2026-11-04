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

"""Provider-arm ``check_axial`` audit: mesh-rasterized face depth, envelope and openings.

The shared re-encode reports recall on the coarse stack grid. Recall cannot see a candidate that
pushes a whole face outward, and the stack pitch (several centimetres on typical objects) hides
millimetre errors and clips anything the candidate builds outside the source grid. This module
therefore compares the two meshes directly: both are sampled and binned into
per-face first-surface depth maps on one fine grid over the union of their bounds, and every
column is classified — on the source plane, shifted plane, structural difference, candidate-only
(excess beyond the source silhouette) or source-only. The same readings drive one numbered
discrepancies list the Author answers item by item; recall stays a reference line. FACE_OPENINGS
placements are matched from the AXIAL text as before.

Authoring feedback in the same sense as the shared diagnostic: never a Host gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from usdcraft.reconstruction.axial import _scene_components
from usdcraft.reconstruction.diagnostics import parse_axial_stack
from usdcraft.reconstruction.face_features import (
    FaceFeature,
    detect_face_features,
    parse_face_features,
)
from usdcraft.reconstruction.face_openings import detect_face_openings, parse_face_openings
from usdcraft.reconstruction.interior_islands import InteriorIslandGroup, parse_interior_islands
from usdcraft.reconstruction.local_surface_audit import format_local_surfaces, local_surface_audit
from usdcraft.reconstruction.source import SourceInput, load_normalized_source

_MICRO_LINE = re.compile(
    r"^s(?P<s0>\d+):(?P<s1>\d+) mz=(?P<z0>\d+):(?P<z1>\d+) xy=(?P<islands>.*)$"
)
_ISLAND_TOKEN = re.compile(
    r"^(?P<x0>\d+):(?P<x1>\d+),(?P<y0>\d+):(?P<y1>\d+)(?:\((?P<extra>[^)]*)\))?$"
)


@dataclass(frozen=True)
class MicroIsland:
    x0: int
    x1: int
    y0: int
    y1: int
    contour: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MicroSlab:
    stack_z0: int
    stack_z1: int
    micro_z0: int
    micro_z1: int
    islands: tuple[MicroIsland, ...]


def parse_micro_slabs(text: str) -> tuple[MicroSlab, ...]:
    """Parse ``MICRO_SLICE_BOUNDS_Z192`` slab lines into islands (bbox plus optional contour)."""

    slabs: list[MicroSlab] = []
    active = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[MICRO_SLICE_BOUNDS_Z192"):
            active = True
            continue
        if active and line.startswith("["):
            break
        if not active:
            continue
        match = _MICRO_LINE.match(line)
        if match is None:
            continue
        islands: list[MicroIsland] = []
        for token in match.group("islands").split("|"):
            island = _ISLAND_TOKEN.match(token.strip())
            if island is None:
                continue
            contour: tuple[tuple[int, int], ...] = ()
            extra = island.group("extra") or ""
            for field in extra.split(";"):
                if field.startswith("p="):
                    points = []
                    for pair in field[2:].split():
                        x_text, _, y_text = pair.partition(",")
                        if x_text.isdigit() and y_text.isdigit():
                            points.append((int(x_text), int(y_text)))
                    contour = tuple(points)
            islands.append(
                MicroIsland(
                    x0=int(island.group("x0")),
                    x1=int(island.group("x1")),
                    y0=int(island.group("y0")),
                    y1=int(island.group("y1")),
                    contour=contour,
                )
            )
        slabs.append(
            MicroSlab(
                stack_z0=int(match.group("s0")),
                stack_z1=int(match.group("s1")),
                micro_z0=int(match.group("z0")),
                micro_z1=int(match.group("z1")),
                islands=tuple(islands),
            )
        )
    return tuple(slabs)


_FACES: tuple[tuple[str, int, int], ...] = (
    ("-Y front", 1, -1),
    ("+Y back", 1, 1),
    ("-X left", 0, -1),
    ("+X right", 0, 1),
    ("+Z top", 2, 1),
    ("-Z bottom", 2, -1),
)

# Depth-map audit: both meshes are sampled and binned into per-face first-surface depth maps on
# one fine grid spanning the union of both bounds, so nothing the candidate builds outside the
# source envelope is clipped away. Columns are classified by the signed depth difference.
DEPTH_SAMPLES = 600_000
DEPTH_PITCH_MIN_M = 0.003
DEPTH_SAMPLES_PER_COLUMN = 8.0
PLANE_TOLERANCE_M = 0.005
PLANE_SHIFT_MAX_M = 0.030
PLANE_SHIFT_DEFECT_MIN_M = 0.008
# A face-plane defect needs a source face that is itself a plane: at least this share of the
# compared columns must have their source depth within the plane tolerance of the face's median
# depth. A curved body, a leg-and-rail structure or a scanned underside has no plane to be off.
PLANE_SOURCE_PLANARITY_MIN = 0.50
EXCESS_DEFECT_SHARE = 0.05
# Candidate-only columns count as excess only beyond this many columns from the nearest source
# column: the one- or two-column fringe every smooth member leaves around a lumpy source is not a
# body standing outside the source.
EXCESS_FRINGE_COLUMNS = 2


def _depth_pitch(stack_pitch: float, extent: np.ndarray, sample_count: int) -> float:
    """Column pitch: fine, but coarse enough that a column holds ~8 samples of a box-like surface
    (an empty first-surface column would otherwise fall through to a deeper surface and read as a
    structural difference), never above half a stack pitch."""

    bbox_area = 2.0 * float(extent[0] * extent[1] + extent[1] * extent[2] + extent[0] * extent[2])
    dense = float(np.sqrt(DEPTH_SAMPLES_PER_COLUMN * bbox_area / max(sample_count, 1)))
    ceiling = max(DEPTH_PITCH_MIN_M, stack_pitch / 2.0)
    return float(min(ceiling, max(DEPTH_PITCH_MIN_M, dense)))


def _depth_map(
    points: np.ndarray,
    *,
    axis: int,
    sign: int,
    u_axis: int,
    v_axis: int,
    origin: np.ndarray,
    shape: tuple[int, int],
    pitch: float,
) -> np.ndarray:
    """Outward coordinate of the first surface per (u, v) column seen from ``sign`` along ``axis``;
    ``-inf`` where the column holds no surface."""

    depth = np.full(shape[0] * shape[1], -np.inf)
    if len(points) == 0:
        return depth.reshape(shape)
    iu = np.clip(
        np.floor((points[:, u_axis] - origin[u_axis]) / pitch).astype(np.int64), 0, shape[0] - 1
    )
    iv = np.clip(
        np.floor((points[:, v_axis] - origin[v_axis]) / pitch).astype(np.int64), 0, shape[1] - 1
    )
    np.maximum.at(depth, iu * shape[1] + iv, points[:, axis] * sign)
    return depth.reshape(shape)


def _dilate2d(mask: np.ndarray, cells: int) -> np.ndarray:
    """Square dilation of a 2-D boolean mask by ``cells``."""

    result = mask.copy()
    for _ in range(cells):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(result)
        for du in (0, 1, 2):
            for dv in (0, 1, 2):
                grown |= padded[du : du + result.shape[0], dv : dv + result.shape[1]]
        result = grown
    return result


def _region(mask: np.ndarray, origin_uv: tuple[float, float], pitch: float) -> list[float] | None:
    if not mask.any():
        return None
    cells = np.argwhere(mask)
    lo = cells.min(axis=0)
    hi = cells.max(axis=0) + 1
    return [
        round(origin_uv[0] + lo[0] * pitch, 3),
        round(origin_uv[0] + hi[0] * pitch, 3),
        round(origin_uv[1] + lo[1] * pitch, 3),
        round(origin_uv[1] + hi[1] * pitch, 3),
    ]


def face_depth_audit(
    source_points: np.ndarray,
    candidate_points: np.ndarray,
    *,
    stack_pitch: float,
    source_openings: Sequence[Any] = (),
) -> dict[str, Any]:
    """Per axis face, the candidate's first surface against the source's, column by column.

    Both point clouds are binned into first-surface depth maps on one fine grid over the union of
    their bounds. Per column the signed difference (candidate minus source, positive = candidate
    nearer to the viewer, i.e. outward) sorts the column into on-plane (|Δ| within
    ``PLANE_TOLERANCE_M``), shifted plane (up to ``PLANE_SHIFT_MAX_M``) or structural difference
    (a cavity, an articulated part, a missing or extra member). Columns inside a source
    ``FACE_OPENINGS`` outline on that face are reported separately and never enter the face
    verdict. Columns where only the candidate has surface are the candidate standing beyond the
    source silhouette (excess); columns where only the source has surface are missing coverage.
    """

    if len(source_points) == 0 or len(candidate_points) == 0:
        return {
            "pitch_m": None,
            "faces": [],
            "defects": [],
            "excess": {"columns": 0, "area_m2": 0.0},
        }
    extent = np.maximum(source_points.max(axis=0), candidate_points.max(axis=0)) - np.minimum(
        source_points.min(axis=0), candidate_points.min(axis=0)
    )
    pitch = _depth_pitch(stack_pitch, extent, min(len(source_points), len(candidate_points)))
    lo = np.minimum(source_points.min(axis=0), candidate_points.min(axis=0)) - pitch
    hi = np.maximum(source_points.max(axis=0), candidate_points.max(axis=0)) + pitch
    dims = np.maximum(1, np.ceil((hi - lo) / pitch).astype(np.int64))
    faces: list[dict[str, Any]] = []
    all_abs: list[np.ndarray] = []
    excess_columns = 0
    candidate_columns = 0
    for name, axis, sign in _FACES:
        face_key = name.split()[0]
        _axis, _sign, u_axis, v_axis = _FACE_AXES[face_key]
        shape = (int(dims[u_axis]), int(dims[v_axis]))
        kwargs = dict(
            axis=axis, sign=sign, u_axis=u_axis, v_axis=v_axis, origin=lo, shape=shape, pitch=pitch
        )
        src = _depth_map(source_points, **kwargs)
        cand = _depth_map(candidate_points, **kwargs)
        s_has = np.isfinite(src)
        c_has = np.isfinite(cand)
        opening = np.zeros(shape, dtype=bool)
        for entry in source_openings:
            if getattr(entry, "face", None) != face_key or getattr(entry, "open_face", False):
                continue
            cu, cv = entry.center_uv
            su, sv = entry.size_uv
            u0 = int(np.floor((cu - su / 2 - lo[u_axis]) / pitch))
            u1 = int(np.ceil((cu + su / 2 - lo[u_axis]) / pitch))
            v0 = int(np.floor((cv - sv / 2 - lo[v_axis]) / pitch))
            v1 = int(np.ceil((cv + sv / 2 - lo[v_axis]) / pitch))
            opening[max(u0, 0) : max(u1, 0), max(v0, 0) : max(v1, 0)] = True
        both = s_has & c_has & ~opening
        compared = int(np.count_nonzero(both))
        cand_only = c_has & ~_dilate2d(s_has, EXCESS_FRINGE_COLUMNS)
        src_only = s_has & ~c_has & ~opening
        n_c = int(np.count_nonzero(c_has))
        candidate_columns += n_c
        excess_columns += int(np.count_nonzero(cand_only))
        origin_uv = (float(lo[u_axis]), float(lo[v_axis]))
        row: dict[str, Any] = {
            "face": name,
            "columns_compared": compared,
            "opening_columns": int(np.count_nonzero(opening & (s_has | c_has))),
            "missing_columns": int(np.count_nonzero(src_only)),
            "excess_columns": int(np.count_nonzero(cand_only)),
            "excess_share_of_candidate": round(int(np.count_nonzero(cand_only)) / n_c, 3)
            if n_c
            else 0.0,
            "excess_area_m2": round(int(np.count_nonzero(cand_only)) * pitch * pitch, 5),
            "excess_region_uv_m": _region(cand_only, origin_uv, pitch),
            "defects": [],
        }
        if compared:
            delta = cand[both] - src[both]
            all_abs.append(np.abs(delta))
            on_plane = np.abs(delta) <= PLANE_TOLERANCE_M
            shifted = (~on_plane) & (np.abs(delta) <= PLANE_SHIFT_MAX_M)
            structural = np.abs(delta) > PLANE_SHIFT_MAX_M
            structural_mask = np.zeros(shape, dtype=bool)
            structural_mask[both] = structural
            source_depths = src[both]
            planarity = float(
                np.mean(np.abs(source_depths - np.median(source_depths)) <= PLANE_TOLERANCE_M)
            )
            # Face plane offset: over the columns where the source sits on its dominant plane,
            # the median candidate-minus-source depth. Judged only on the source's own plane
            # columns, an open door, a knob or a chair beside a table cannot drag it; a face
            # without a source plane (planarity below the minimum) gets a reading, no defect.
            plane_columns = np.abs(source_depths - np.median(source_depths)) <= PLANE_TOLERANCE_M
            plane_offset = float(np.median(delta[plane_columns])) if plane_columns.any() else 0.0
            row.update(
                {
                    "source_planarity": round(planarity, 3),
                    "plane_offset_m": round(plane_offset, 4),
                    "on_plane_share": round(float(on_plane.mean()), 3),
                    "shifted_share": round(float(shifted.mean()), 3),
                    "shifted_median_m": (
                        round(float(np.median(delta[shifted])), 4) if shifted.any() else 0.0
                    ),
                    "structural_share": round(float(structural.mean()), 3),
                    "structural_region_uv_m": _region(structural_mask, origin_uv, pitch),
                    "abs_delta_p50_m": round(float(np.percentile(np.abs(delta), 50)), 4),
                    "abs_delta_p90_m": round(float(np.percentile(np.abs(delta), 90)), 4),
                }
            )
            if (
                planarity >= PLANE_SOURCE_PLANARITY_MIN
                and abs(plane_offset) >= PLANE_SHIFT_DEFECT_MIN_M
            ):
                direction = "outward" if plane_offset > 0 else "inward"
                row["defects"].append(
                    f"{name.strip()} face built {abs(plane_offset) * 1000:.0f} mm {direction} of "
                    f"the source plane (median over the {planarity:.0%} of columns where the "
                    "source is planar)"
                )
        row["excess_axes"] = ("xyz"[u_axis], "xyz"[v_axis])
        faces.append(row)
    pooled = np.concatenate(all_abs) if all_abs else np.zeros(0)
    # One body standing outside the source shows up on every face whose view contains that
    # direction, so excess is one consolidated defect naming the face with the largest area.
    defects: list[str] = [text for row in faces for text in row["defects"]]
    share = excess_columns / candidate_columns if candidate_columns else 0.0
    if excess_columns and share >= EXCESS_DEFECT_SHARE:
        worst = max(faces, key=lambda row: row["excess_area_m2"])
        r = worst["excess_region_uv_m"]
        u_name, v_name = worst["excess_axes"]
        defects.append(
            f"candidate surface beyond the source silhouette: {excess_columns * pitch * pitch * 1e4:.0f} "
            f"cm² ({share:.0%} of the candidate's columns); largest seen from {worst['face'].strip()} "
            f"at {u_name}[{r[0]:+.3f}..{r[1]:+.3f}] {v_name}[{r[2]:+.3f}..{r[3]:+.3f}]"
        )
    return {
        "pitch_m": pitch,
        "plane_tolerance_m": PLANE_TOLERANCE_M,
        "plane_shift_max_m": PLANE_SHIFT_MAX_M,
        "faces": faces,
        "defects": defects,
        "abs_delta_p50_m": round(float(np.percentile(pooled, 50)), 4) if len(pooled) else None,
        "abs_delta_p90_m": round(float(np.percentile(pooled, 90)), 4) if len(pooled) else None,
        "excess": {
            "columns": excess_columns,
            "candidate_columns": candidate_columns,
            "share_of_candidate_columns": (
                round(excess_columns / candidate_columns, 4) if candidate_columns else 0.0
            ),
            "area_m2": round(excess_columns * pitch * pitch, 5),
        },
    }


def envelope_comparison(
    source_points: np.ndarray, candidate_points: np.ndarray, *, stack_pitch: float
) -> dict[str, Any]:
    """Per-axis bounds of both meshes (no grid, no clipping): a reading, never a defect.

    A bbox extreme is carried by whatever reaches furthest — an open door, a chair beside a table,
    a knob — so a reach difference alone says nothing about the body; the per-face position
    reading in ``face_depth_audit`` (median first surface on both sides) judges the faces.
    """

    axes: list[dict[str, Any]] = []
    for axis, name in enumerate("XYZ"):
        s_lo, s_hi = float(source_points[:, axis].min()), float(source_points[:, axis].max())
        c_lo, c_hi = float(candidate_points[:, axis].min()), float(candidate_points[:, axis].max())
        axes.append(
            {
                "axis": name,
                "source_m": [round(s_lo, 3), round(s_hi, 3)],
                "candidate_m": [round(c_lo, 3), round(c_hi, 3)],
                "delta_low_m": round(c_lo - s_lo, 3),
                "delta_high_m": round(c_hi - s_hi, 3),
            }
        )
    return {"axes": axes, "defects": []}


def audit_candidate_against_axial(
    axial_path: Path,
    candidate_usd_path: Path,
    *,
    source_usd_path: Path | None = None,
) -> dict[str, Any]:
    """Compute the Provider-arm readings from the AXIAL text, the source mesh and the candidate.

    The face depth and envelope readings need the Host-side normalized source mesh; without it
    only the FACE_OPENINGS placements are computed.
    """

    from usdcraft.reconstruction._face_grid import sample_surface_points

    text = Path(axial_path).read_text(encoding="utf-8")
    parsed = parse_axial_stack(text)
    stack_pitch = float(parsed.frame.pitch)
    scene = load_normalized_source(SourceInput(Path(candidate_usd_path)))
    candidate_components = _scene_components(scene)
    candidate_points = sample_surface_points(candidate_components, target=DEPTH_SAMPLES, seed=2)
    source_openings = parse_face_openings(text)
    interior = interior_fitting_readings(
        parse_interior_islands(text), candidate_points, stack_pitch=stack_pitch
    )
    face_depth: dict[str, Any] | None = None
    envelope: dict[str, Any] | None = None
    source_points = np.empty((0, 3), dtype=float)
    if source_usd_path is not None and Path(source_usd_path).is_file():
        source_scene = load_normalized_source(SourceInput(Path(source_usd_path)))
        source_points = sample_surface_points(
            _scene_components(source_scene), target=DEPTH_SAMPLES, seed=1
        )
        face_depth = face_depth_audit(
            source_points,
            candidate_points,
            stack_pitch=stack_pitch,
            source_openings=source_openings,
        )
        envelope = envelope_comparison(source_points, candidate_points, stack_pitch=stack_pitch)
    openings = opening_placement(
        source_openings, detect_face_openings(candidate_components).openings, stack_pitch
    )
    protrusions = feature_placement(
        parse_face_features(text),
        detect_face_features(candidate_components).features,
        stack_pitch=stack_pitch,
    )
    defects: list[str] = []
    if envelope is not None:
        defects.extend(envelope["defects"])
    if face_depth is not None:
        defects.extend(face_depth.get("defects", []))
    for row in openings.get("candidate_openings", []):
        if (
            "offset_mm" in row
            and not row.get("source_open_face")
            and not str(row.get("reading", "")).startswith("opening matches")
        ):
            defects.append(
                f"{row['face']} {row['kind']} at {row['candidate_center_uv_m']} is "
                f"{row['offset_mm']} mm off its source opening or off its plan size "
                f"(ratios {row['size_ratio_uv']})"
            )
    return {
        "envelope": envelope,
        "face_depth": face_depth,
        "opening_placement": openings,
        "feature_placement": protrusions,
        "interior_fittings": interior,
        "local_surfaces": local_surface_audit(
            source_points,
            candidate_points,
            source_features=parse_face_features(text),
            source_openings=source_openings,
        ),
        "defects": defects,
        "stack_pitch_m": stack_pitch,
    }


# Compact protrusions of both meshes, matched one-to-one per face by centre distance. Readings
# only: the image decides which protrusions are controls, and a source protrusion the candidate
# lacks may be a fused, eroded or texture-only feature.
FEATURE_MATCH_RADIUS_M = 0.060
FEATURE_COMPACT_MAX_M = 0.120
FEATURE_SIZE_RATIO_MIN = 0.5
FEATURE_SIZE_RATIO_MAX = 2.0
FEATURE_TOLERANCE_MIN_M = 0.015
UNBUILT_PROTRUSIONS_LISTED = 6
CANDIDATE_ONLY_PROTRUSIONS_LISTED = 3


def _feature_row(feature: FaceFeature) -> dict[str, Any]:
    return {
        "face": feature.face,
        "center_uv_m": [
            round(float(feature.center_uv[0]), 4),
            round(float(feature.center_uv[1]), 4),
        ],
        "size_mm": [
            round(float(feature.size_uv[0]) * 1000),
            round(float(feature.size_uv[1]) * 1000),
        ],
        "height_mm": round(float(feature.height_median_m) * 1000),
    }


def feature_placement(
    source: Sequence[FaceFeature],
    candidate: Sequence[FaceFeature],
    *,
    stack_pitch: float,
) -> dict[str, Any]:
    """Match the candidate's compact FACE_FEATURES to the source's, one-to-one per face."""

    def compact(feature: FaceFeature) -> bool:
        return max(float(feature.size_uv[0]), float(feature.size_uv[1])) <= FEATURE_COMPACT_MAX_M

    src = [f for f in source if compact(f)]
    cand = [f for f in candidate if compact(f)]
    pairs: list[tuple[float, int, int]] = []
    for i, s in enumerate(src):
        for j, c in enumerate(cand):
            if s.face != c.face:
                continue
            distance = float(
                np.hypot(s.center_uv[0] - c.center_uv[0], s.center_uv[1] - c.center_uv[1])
            )
            if distance <= FEATURE_MATCH_RADIUS_M:
                pairs.append((distance, i, j))
    pairs.sort()
    tolerance = max(FEATURE_TOLERANCE_MIN_M, 2.0 * float(stack_pitch))
    used_source: set[int] = set()
    used_candidate: set[int] = set()
    matched: list[dict[str, Any]] = []
    for distance, i, j in pairs:
        if i in used_source or j in used_candidate:
            continue
        s, c = src[i], cand[j]
        ratios = [float(c.size_uv[k]) / max(float(s.size_uv[k]), 1e-6) for k in range(2)]
        if not all(FEATURE_SIZE_RATIO_MIN <= r <= FEATURE_SIZE_RATIO_MAX for r in ratios):
            continue
        used_source.add(i)
        used_candidate.add(j)
        offset_mm = round(distance * 1000)
        matched.append(
            {
                **_feature_row(s),
                "candidate_center_uv_m": [
                    round(float(c.center_uv[0]), 4),
                    round(float(c.center_uv[1]), 4),
                ],
                "candidate_size_mm": [
                    round(float(c.size_uv[0]) * 1000),
                    round(float(c.size_uv[1]) * 1000),
                ],
                "candidate_height_mm": round(float(c.height_median_m) * 1000),
                "offset_mm": offset_mm,
                "size_ratio_uv": [round(r, 2) for r in ratios],
                "reading": (
                    "placed on its source protrusion"
                    if offset_mm <= tolerance * 1000
                    else f"{offset_mm} mm off its source protrusion"
                ),
            }
        )
    unbuilt = [_feature_row(s) for i, s in enumerate(src) if i not in used_source]
    unbuilt.sort(key=lambda row: (row["face"], -row["height_mm"]))
    candidate_only = [_feature_row(c) for j, c in enumerate(cand) if j not in used_candidate]
    candidate_only.sort(key=lambda row: (row["face"], -row["height_mm"]))
    return {
        "tolerance_mm": round(tolerance * 1000),
        "matched": matched,
        "unbuilt_source": unbuilt,
        "candidate_only": candidate_only,
    }


# A source INTERIOR_ISLANDS group counts as modelled when at least this many candidate surface
# samples fall inside its footprint (grown by half a stack pitch) within a height window of the
# source band grown by a few stack pitches; the median height of those samples against the band
# is the placement reading. Readings only: the Author decides from the image what the fitting is.
INTERIOR_MODELLED_MIN_POINTS = 25
INTERIOR_HEIGHT_SEARCH_PITCHES = 3.0


def interior_fitting_readings(
    groups: Sequence[InteriorIslandGroup],
    candidate_points: np.ndarray,
    *,
    stack_pitch: float,
) -> list[dict[str, Any]]:
    """Per source interior group: how much candidate surface stands where the source fitting is,
    and how far its median height sits from the source band."""

    rows: list[dict[str, Any]] = []
    grow = 0.5 * float(stack_pitch)
    reach = INTERIOR_HEIGHT_SEARCH_PITCHES * float(stack_pitch)
    points = np.asarray(candidate_points, dtype=float)
    for group in groups:
        count = 0
        candidate_z: float | None = None
        offset_mm: int | None = None
        if len(points):
            inside = (
                (points[:, 0] >= group.x_m[0] - grow)
                & (points[:, 0] <= group.x_m[1] + grow)
                & (points[:, 1] >= group.y_m[0] - grow)
                & (points[:, 1] <= group.y_m[1] + grow)
                & (points[:, 2] >= group.z_m[0] - reach)
                & (points[:, 2] <= group.z_m[1] + reach)
            )
            count = int(np.count_nonzero(inside))
            if count >= INTERIOR_MODELLED_MIN_POINTS:
                candidate_z = float(np.median(points[inside, 2]))
                if candidate_z > group.z_m[1]:
                    offset_mm = round((candidate_z - group.z_m[1]) * 1000)
                elif candidate_z < group.z_m[0]:
                    offset_mm = round((candidate_z - group.z_m[0]) * 1000)
                else:
                    offset_mm = 0
        rows.append(
            {
                **group.to_dict(),
                "candidate_surface_points": count,
                "modelled": count >= INTERIOR_MODELLED_MIN_POINTS,
                "candidate_z_m": None if candidate_z is None else round(candidate_z, 4),
                "z_offset_mm": offset_mm,
            }
        )
    return rows


OPENING_MATCH_RADIUS_M = 0.060


def opening_placement(
    source_openings: Sequence[Any], candidate_openings: Sequence[Any], stack_pitch: float
) -> dict[str, Any]:
    """Match the candidate's face openings/recesses to the source's by face and centre.

    Reports the centre offset and plan-size ratios per matched pair, the source openings no
    candidate opening covers (unmodelled), and the candidate openings the source does not show
    (unsupported). Depth is not compared: the source's visible depth is a lower bound.
    """

    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    for cand in candidate_openings:
        best, best_dist = None, None
        for index, src in enumerate(source_openings):
            if src.face != cand.face or src.kind != cand.kind:
                continue
            dist = float(
                np.hypot(cand.center_uv[0] - src.center_uv[0], cand.center_uv[1] - src.center_uv[1])
            )
            if best_dist is None or dist < best_dist:
                best, best_dist = index, dist
        row: dict[str, Any] = {
            "face": cand.face,
            "kind": cand.kind,
            "candidate_center_uv_m": [round(float(v), 4) for v in cand.center_uv],
            "candidate_size_uv_m": [round(float(v), 4) for v in cand.size_uv],
        }
        if best is not None and best_dist is not None and best_dist <= OPENING_MATCH_RADIUS_M:
            src = source_openings[best]
            used.add(best)
            row.update(
                {
                    "source_center_uv_m": [round(float(v), 4) for v in src.center_uv],
                    "source_open_face": bool(getattr(src, "open_face", False)),
                    "offset_mm": round(best_dist * 1000),
                    "size_ratio_uv": [
                        round(float(cand.size_uv[i]) / float(src.size_uv[i]), 2)
                        if src.size_uv[i] > 0
                        else None
                        for i in range(2)
                    ],
                    "reading": (
                        "opening matches the source in place and plan size"
                        if best_dist <= stack_pitch
                        and all(
                            0.75 <= r <= 1.33
                            for r in (cand.size_uv[i] / src.size_uv[i] for i in range(2))
                            if r
                        )
                        else "opening centre or plan size differs from the source"
                    ),
                }
            )
        else:
            row["reading"] = "no source opening of this kind within the match radius on this face"
        rows.append(row)
    unmodelled = [
        {
            "face": src.face,
            "kind": src.kind,
            "source_center_uv_m": [round(src.center_uv[0], 4), round(src.center_uv[1], 4)],
            "size_mm": [round(src.size_uv[0] * 1000), round(src.size_uv[1] * 1000)],
            "visible_depth_mm": None if src.depth_p50_m is None else round(src.depth_p50_m * 1000),
        }
        for index, src in enumerate(source_openings)
        if index not in used and src.kind == "opening" and not src.open_face
    ]
    return {"candidate_openings": rows, "unmodelled_source_openings": unmodelled}


_FACE_AXES = {
    # face: (depth axis, outward sign, u axis, v axis)
    "-Y": (1, -1, 0, 2),
    "+Y": (1, +1, 0, 2),
    "-X": (0, -1, 1, 2),
    "+X": (0, +1, 1, 2),
    "+Z": (2, +1, 0, 1),
    "-Z": (2, -1, 0, 1),
}


def format_provider_audit(audit: dict[str, Any]) -> str:
    """Render the audit: the numbered discrepancies list first, then the compact readings behind it."""

    lines: list[str] = []
    defects = list(audit.get("defects") or [])
    if defects:
        lines.append(
            f"DISCREPANCIES ({len(defects)}) — resolve each D item using the source and image; "
            "repair supported errors or explain retained differences:"
        )
        lines.extend(f"  D{index} {text}" for index, text in enumerate(defects, start=1))
    else:
        lines.append(
            "DISCREPANCIES (0): no threshold finding; still inspect visible parts and local shape."
        )
    envelope = audit.get("envelope")
    if envelope and envelope.get("axes"):
        lines.append(
            "envelope (mesh bounds, metres): "
            + " | ".join(
                f"{a['axis']} source[{a['source_m'][0]:+.3f}..{a['source_m'][1]:+.3f}] "
                f"candidate[{a['candidate_m'][0]:+.3f}..{a['candidate_m'][1]:+.3f}]"
                for a in envelope["axes"]
            )
        )
    depth = audit.get("face_depth")
    if depth and depth.get("faces") and depth.get("pitch_m"):
        lines.append(
            f"face depth ({depth['pitch_m'] * 1000:.0f} mm columns; positive offset = candidate outward; "
            "plane_offset is median on source-planar columns. Check face assignment/pose before "
            "moving a wall. Opening regions excluded; missing columns are not proof of empty space):"
        )
        for row in depth["faces"]:
            excess_text = (
                f"excess={row['excess_columns']} cols ({row['excess_share_of_candidate']:.0%}, "
                f"{row['excess_area_m2'] * 1e4:.0f} cm²)"
            )
            if row.get("excess_region_uv_m"):
                r = row["excess_region_uv_m"]
                excess_text += f" at [{r[0]:+.3f}..{r[1]:+.3f}]x[{r[2]:+.3f}..{r[3]:+.3f}]"
            if row["columns_compared"]:
                lines.append(
                    f"  {row['face']:10s} cols={row['columns_compared']} "
                    f"plane_offset={row['plane_offset_m'] * 1000:+.0f}mm "
                    f"source_planar={row['source_planarity']:.0%} "
                    f"on_plane={row['on_plane_share']:.0%} shifted={row['shifted_share']:.0%} "
                    f"(median {row['shifted_median_m'] * 1000:+.0f} mm) "
                    f"structural={row['structural_share']:.0%} {excess_text} "
                    f"openings={row['opening_columns']} missing={row['missing_columns']}"
                )
            else:
                lines.append(
                    f"  {row['face']:10s} no shared columns; {excess_text} "
                    f"openings={row['opening_columns']} missing={row['missing_columns']}"
                )
        excess = depth.get("excess") or {}
        p50 = depth.get("abs_delta_p50_m")
        p90 = depth.get("abs_delta_p90_m")
        lines.append(
            "  overall |Δ| "
            + (f"p50={p50 * 1000:.1f} mm p90={p90 * 1000:.1f} mm; " if p50 is not None else "")
            + f"candidate beyond the source silhouette: {excess.get('columns', 0)} cols "
            f"({excess.get('share_of_candidate_columns', 0.0):.1%}, {excess.get('area_m2', 0.0) * 1e4:.0f} cm²)"
        )
    openings = audit.get("opening_placement") or {}
    if openings.get("candidate_openings") or openings.get("unmodelled_source_openings"):
        lines.append(
            "openings (candidate face openings/recesses vs source FACE_OPENINGS; plan size only, "
            "depth-threshold regions, not guaranteed full semantic rims; visible depth is a lower bound. "
            "Check rim boundaries before resizing or recentering a tray):"
        )
        for row in openings.get("candidate_openings", []):
            if "offset_mm" in row:
                lines.append(
                    f"  {row['face']} {row['kind']} at {row['candidate_center_uv_m']} "
                    f"size={row['candidate_size_uv_m']} offset={row['offset_mm']}mm "
                    f"size_ratio={row['size_ratio_uv']} -> {row['reading']}"
                )
            else:
                lines.append(
                    f"  {row['face']} {row['kind']} at {row['candidate_center_uv_m']} "
                    f"size={row['candidate_size_uv_m']} -> {row['reading']}"
                )
        for item in openings.get("unmodelled_source_openings", [])[:3]:
            lines.append(
                f"  unmodelled source opening {item['face']} c={item['source_center_uv_m']} "
                f"size={item['size_mm'][0]}x{item['size_mm'][1]}mm visible_depth>={item['visible_depth_mm']}mm"
            )
    protrusions = audit.get("feature_placement") or {}
    if (
        protrusions.get("matched")
        or protrusions.get("unbuilt_source")
        or protrusions.get("candidate_only")
    ):
        lines.append(
            "protrusions (automatic face matches; use image identity to select controls. "
            "Undetected/flush details may still be required; inspect size/profile, not just centre):"
        )
        for row in protrusions.get("matched", []):
            lines.append(
                f"  {row['face']} source c={row['center_uv_m']} size={row['size_mm'][0]}x"
                f"{row['size_mm'][1]}mm h={row['height_mm']}mm -> candidate c="
                f"{row['candidate_center_uv_m']} size={row['candidate_size_mm'][0]}x"
                f"{row['candidate_size_mm'][1]}mm h={row['candidate_height_mm']}mm "
                f"offset={row['offset_mm']}mm size_ratio={row['size_ratio_uv']} -> {row['reading']}"
            )
        for row in protrusions.get("unbuilt_source", [])[:UNBUILT_PROTRUSIONS_LISTED]:
            lines.append(
                f"  unbuilt source protrusion {row['face']} c={row['center_uv_m']} "
                f"size={row['size_mm'][0]}x{row['size_mm'][1]}mm h={row['height_mm']}mm"
            )
        more = len(protrusions.get("unbuilt_source", [])) - UNBUILT_PROTRUSIONS_LISTED
        if more > 0:
            lines.append(f"  ... {more} more unbuilt source protrusions")
        for row in protrusions.get("candidate_only", [])[:CANDIDATE_ONLY_PROTRUSIONS_LISTED]:
            lines.append(
                f"  candidate-only protrusion {row['face']} c={row['center_uv_m']} "
                f"size={row['size_mm'][0]}x{row['size_mm'][1]}mm h={row['height_mm']}mm"
            )
    source_axes = {row["axis"]: row["source_m"] for row in (envelope or {}).get("axes", [])}
    # The displayed envelope is rounded to millimetres; leave one stack pitch around its
    # sampled depth so a follow-up vertex query does not trim the outer face at that rounding.
    margin = max(float(audit.get("stack_pitch_m") or 0.0), 0.001)
    source_bounds = (
        [
            source_axes[axis][end] + (-margin if end == 0 else margin)
            for end in (0, 1)
            for axis in "XYZ"
        ]
        if all(axis in source_axes for axis in "XYZ")
        else None
    )
    lines.extend(
        format_local_surfaces(audit.get("local_surfaces") or {}, source_bounds=source_bounds)
    )
    fittings = audit.get("interior_fittings") or []
    if fittings:
        lines.append(
            "interior fittings (source INTERIOR_ISLANDS groups vs candidate surface inside each "
            "group's footprint and height band; a reading, not a defect — the image says what the "
            "fitting is):"
        )
        for row in fittings:
            if not row.get("modelled"):
                verdict = "no candidate surface there"
            elif not row.get("z_offset_mm"):
                verdict = f"modelled at height ({row['candidate_surface_points']} candidate points)"
            else:
                offset = int(row["z_offset_mm"])
                verdict = (
                    f"modelled {abs(offset)} mm {'above' if offset > 0 else 'below'} the source "
                    f"band ({row['candidate_surface_points']} candidate points)"
                )
            lines.append(
                f"  {row['id']} z[{row['z_m'][0]:+.3f},{row['z_m'][1]:+.3f}] "
                f"size={row['size_mm'][0]}x{row['size_mm'][1]}mm kind={row['kind']} -> {verdict}"
            )
    return "\n".join(lines)


__all__ = [
    "MicroIsland",
    "MicroSlab",
    "audit_candidate_against_axial",
    "envelope_comparison",
    "face_depth_audit",
    "feature_placement",
    "format_provider_audit",
    "interior_fitting_readings",
    "opening_placement",
    "parse_micro_slabs",
]
