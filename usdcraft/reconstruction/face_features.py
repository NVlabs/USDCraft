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

"""FACE_FEATURES: compact protrusions on the six axis-aligned faces of the normalized source.

A scanned or generated mesh keeps its knobs, buttons, handles and levers as compact bumps on an
otherwise smooth face. This detector turns each face into a first-surface depth map, fits a local
base that follows a curved shell (median window), and keeps the compact residual blobs. Three
separation levels: L1 on the whole face; L2 with a smaller window inside a large, band-shaped or
silhouette-touching L1 region (a control standing on a skirt, plinth or corner bulge); L3 keeps
only the cells above half the peak inside a band that still fuses a control with ribs or a step.
Steps and tiers are long bands and are rejected by size; openings and cliffs are excluded by their
depth jump. Thresholds are physical (metres) and assume the source is in true scale.

Typical detections are appliance knobs and levers, knob rows, buttons, drawer pulls and cabinet
handles; stepped furniture and appliance door tiers produce no protrusions. Known limits: thin
paddles on ribbed shells, rounded corners of concave basins, wire and organic geometry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from usdcraft.reconstruction._face_grid import (
    FACE_AXIS_LABELS,
    FACE_FEATURE_FACES,
    MIN_PITCH_M,
    SAMPLE_MAX,
    SAMPLE_MIN,
    SAMPLES_PER_CELL,
    SAMPLING_GAP_MAX_M2,
    extent_and_pitch,
    rasterize_face,
    sample_surface_points,
    section_byte_count,
    silhouette_margin_cells,
)

HEIGHT_MIN_M = 0.003
HEIGHT_MIN_PITCHES = 1.5
# A blob whose peak clears twice the median floor is a protrusion on its own account. Below that,
# in the band down to 1.2x the floor, a blob is kept only as a button-like cap: flat-topped (median
# at least three quarters of its peak — not a ramp fragment of a step, a dimple or a rounded
# bulge), on an object whose median floor is still near the 3 mm physical minimum (on a 2 m body the
# floor is already 7 mm and the band holds door seams and panel steps, not controls), and standing
# in a row — another blob of like size within 150 mm on the same face, as buttons and keys do.
# Many appliance buttons protrude only 4-6 mm and fall in that band; the three tests keep them
# while rejecting the seam and step fragments the band would otherwise admit.
HEIGHT_PEAK_FACTOR = 2.0
HEIGHT_PLATEAU_PEAK_FACTOR = 1.2
HEIGHT_PLATEAU_MIN_RATIO = 0.75
HEIGHT_PLATEAU_MAX_HEIGHT_MIN_M = 0.006
PLATEAU_SIBLING_RADIUS_M = 0.150
PLATEAU_SIBLING_SIZE_TOLERANCE = 0.35
HEIGHT_CAP_M = 0.080
SIZE_MAX_M = 0.120
SIZE_MIN_M = 0.008
MIN_CELLS = 6
FILL_MIN = 0.35
BASE_WINDOW_M = 0.060
SMALL_WINDOW_M = 0.030
# L0: a second, wider base for protrusions larger than the 60 mm window (drawer pulls, oven
# knobs, door rails), which the 60 mm median otherwise treats as face and leaves as corner
# fragments. Compact in at least one direction so panels and drawer fronts stay "band".
# The base window must be wider than the protrusion it is meant to lift (a 220 mm oven dial needs
# a window whose area is more than twice the dial's), hence 400 mm.
LARGE_WINDOW_M = 0.400
LARGE_SIZE_MAX_M = 0.400
LARGE_MIN_SIDE_MAX_M = 0.250
# The 400 mm median is computed on a grid coarsened to about this many window cells so its cost
# stays flat across object sizes (a 2 mm-pitch face would otherwise need a 201-cell window).
LARGE_WINDOW_CELLS_TARGET = 41
LARGE_HEIGHT_CAP_M = 0.150
# A door rail or pull often runs close to the face's edge; L0 accepts an entry with at least this
# share of its cells inside the silhouette margin instead of requiring all of them.
LARGE_INTERIOR_MIN_SHARE = 0.5
# An L0 entry must be something the 60 mm levels cannot report whole: longer than their size cap
# or containing at least this many of their entries as fragments. At most this many per face,
# largest first, so the section stays inside the core byte budget.
LARGE_MIN_FRAGMENTS = 2
LARGE_PER_FACE_MAX = 12
# An L0 entry must stand clearly above the wide base over most of its footprint; a control's
# surrounding face patch (median a few millimetres) is not a protrusion.
LARGE_HEIGHT_MED_MIN_M = 0.008
EDGE_JUMP_M = 0.060
DEDUPE_M = 0.006


@dataclass(frozen=True)
class FaceFeature:
    face: str
    level: int
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    center_uv: tuple[float, float]
    size_uv: tuple[float, float]
    height_max_m: float
    height_median_m: float
    cells: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "face": self.face,
            "level": self.level,
            "u_range_m": [round(self.u_range[0], 4), round(self.u_range[1], 4)],
            "v_range_m": [round(self.v_range[0], 4), round(self.v_range[1], 4)],
            "center_uv_m": [round(self.center_uv[0], 4), round(self.center_uv[1], 4)],
            "size_uv_m": [round(self.size_uv[0], 4), round(self.size_uv[1], 4)],
            "height_max_m": round(self.height_max_m, 4),
            "height_median_m": round(self.height_median_m, 4),
            "cells": self.cells,
        }


@dataclass(frozen=True)
class FaceFeatureSummary:
    features: tuple[FaceFeature, ...]
    pitch_m: float
    sample_count: int
    rejected: dict[str, int]
    section_bytes: int = 0


def _sample_target(extent: np.ndarray, pitch: float) -> int:
    cells = 2.0 * (extent[0] * extent[1] + extent[0] * extent[2] + extent[1] * extent[2])
    cells /= pitch * pitch
    return int(min(SAMPLE_MAX, max(SAMPLE_MIN, SAMPLES_PER_CELL * cells)))


def detect_face_features(components: Sequence[Any], *, seed: int = 0) -> FaceFeatureSummary:
    """Detect compact protrusions on all six faces of the component set (metres, source frame)."""

    grid = extent_and_pitch(components)
    if grid is None:
        return FaceFeatureSummary((), MIN_PITCH_M, 0, {})
    extent, pitch = grid
    points = sample_surface_points(components, target=_sample_target(extent, pitch), seed=seed)
    features: list[FaceFeature] = []
    rejected: dict[str, int] = {}
    for name, axis, sign, u_axis, v_axis in FACE_FEATURE_FACES:
        hits, rej = _detect_face(points, name, axis, sign, u_axis, v_axis, pitch)
        features.extend(hits)
        for key, value in rej.items():
            rejected[key] = rejected.get(key, 0) + value
    summary = FaceFeatureSummary(tuple(features), pitch, len(points), rejected)
    section_bytes = section_byte_count(format_face_features(summary))
    return FaceFeatureSummary(tuple(features), pitch, len(points), rejected, section_bytes)


def _detect_face(
    points: np.ndarray, name: str, axis: int, sign: int, u_axis: int, v_axis: int, pitch: float
) -> tuple[list[FaceFeature], dict[str, int]]:
    raster = rasterize_face(points, axis, sign, u_axis, v_axis, pitch)
    if raster is None:
        return [], {}
    depth, valid, u0, v0 = raster
    nu, nv = depth.shape
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = depth[tuple(nearest)]
    window = max(5, int(round(BASE_WINDOW_M / pitch)) | 1)
    base = ndimage.median_filter(filled, size=window)
    resid = np.where(valid, depth - base, 0.0)
    # openings and cliffs: depth jumps far beyond any control height
    cliff = valid & (np.abs(resid) > EDGE_JUMP_M)
    # silhouette = the face's outer footprint; sampling gaps inside it are filled first, and
    # rounded-shell corners bulge in the residual for about a centimetre inside the outline
    holes = ndimage.binary_fill_holes(valid) & ~valid
    labels, count = ndimage.label(holes)
    small = np.zeros_like(valid)
    if count:
        sizes = ndimage.sum(holes, labels, range(1, count + 1))
        for index, size in enumerate(sizes, start=1):
            if size * pitch * pitch <= SAMPLING_GAP_MAX_M2:
                small |= labels == index
    margin = silhouette_margin_cells(pitch)
    interior = ndimage.binary_erosion(valid | small, iterations=margin)
    valid = valid & ~cliff
    height_min = max(HEIGHT_MIN_M, HEIGHT_MIN_PITCHES * pitch)
    hits: list[FaceFeature] = []
    band_hits: list[int] = []
    rejected: dict[str, int] = {}
    structure = np.ones((3, 3))

    def evaluate(cells: np.ndarray, resid_map: np.ndarray, level: int) -> tuple[str | None, Any]:
        count = int(cells.sum())
        if count < MIN_CELLS:
            return "tiny", None
        ys, xs = np.where(cells)
        su = (ys.max() - ys.min() + 1) * pitch
        sv = (xs.max() - xs.min() + 1) * pitch
        large = level == 0
        size_max = LARGE_SIZE_MAX_M if large else SIZE_MAX_M
        if max(su, sv) > size_max or (large and min(su, sv) > LARGE_MIN_SIDE_MAX_M):
            return "band", (ys, xs)
        if min(su, sv) < SIZE_MIN_M:
            return "sliver", None
        if count / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) < FILL_MIN:
            return "sparse", None
        interior_share = float(interior[cells].mean())
        if interior_share < (LARGE_INTERIOR_MIN_SHARE if large else 1.0):
            return "silhouette", (ys, xs)
        h_max = float(resid_map[cells].max())
        h_med = float(np.median(resid_map[cells]))
        if h_med < height_min or h_max < HEIGHT_PLATEAU_PEAK_FACTOR * height_min:
            return "low", None
        band = h_max < HEIGHT_PEAK_FACTOR * height_min
        if band and (
            large
            or height_min > HEIGHT_PLATEAU_MAX_HEIGHT_MIN_M
            or h_med < HEIGHT_PLATEAU_MIN_RATIO * h_max
        ):
            return "low", None
        height_cap = LARGE_HEIGHT_CAP_M if large else HEIGHT_CAP_M
        if h_med > height_cap or h_max > height_cap:
            return "tall", None
        if band:
            band_hits.append(len(hits))
        hits.append(
            FaceFeature(
                face=name,
                level=level,
                u_range=(ys.min() * pitch + u0, (ys.max() + 1) * pitch + u0),
                v_range=(xs.min() * pitch + v0, (xs.max() + 1) * pitch + v0),
                center_uv=(float(ys.mean()) * pitch + u0, float(xs.mean()) * pitch + v0),
                size_uv=(su, sv),
                height_max_m=h_max,
                height_median_m=h_med,
                cells=count,
            )
        )
        return None, None

    def reject(prefix: str, reason: str) -> None:
        key = prefix + reason
        rejected[key] = rejected.get(key, 0) + 1

    small_window = max(3, int(round(SMALL_WINDOW_M / pitch)) | 1)
    labels, count = ndimage.label((resid > height_min) & valid, structure=structure)
    for index in range(1, count + 1):
        cells = labels == index
        reason, coords = evaluate(cells, resid, 1)
        if reason is None:
            continue
        reject("L1_", reason)
        if coords is None:
            continue
        ys, xs = coords
        y0, y1 = max(0, ys.min() - small_window), min(nu, ys.max() + small_window + 1)
        x0, x1 = max(0, xs.min() - small_window), min(nv, xs.max() + small_window + 1)
        sub = filled[y0:y1, x0:x1]
        local_base = ndimage.median_filter(sub, size=small_window)
        local = np.zeros_like(resid)
        local[y0:y1, x0:x1] = np.where(valid[y0:y1, x0:x1], sub - local_base, 0.0)
        region = np.zeros_like(cells)
        region[y0:y1, x0:x1] = True
        labels2, count2 = ndimage.label((local > height_min) & region & valid, structure=structure)
        for index2 in range(1, count2 + 1):
            cells2 = labels2 == index2
            reason2, _ = evaluate(cells2, local, 2)
            if reason2 is None:
                continue
            reject("L2_", reason2)
            if reason2 != "band":
                continue
            peak = float(local[cells2].max())
            labels3, count3 = ndimage.label(
                cells2 & (local > max(height_min, 0.5 * peak)), structure=structure
            )
            for index3 in range(1, count3 + 1):
                reason3, _ = evaluate(labels3 == index3, local, 3)
                if reason3:
                    reject("L3_", reason3)
    # A band blob (peak under twice the floor) stands only in a row: another blob of like size,
    # distinct from it, within the sibling radius. A lone 4 mm bump is a rivet, a seam or noise.
    band_features = [hits[index] for index in band_hits]
    if band_features:

        def _has_sibling(feature: FaceFeature) -> bool:
            for other in hits:
                if other is feature:
                    continue
                du = other.center_uv[0] - feature.center_uv[0]
                dv = other.center_uv[1] - feature.center_uv[1]
                distance = float(np.hypot(du, dv))
                if distance > PLATEAU_SIBLING_RADIUS_M or distance < max(feature.size_uv):
                    continue
                if all(
                    abs(other.size_uv[k] - feature.size_uv[k])
                    <= PLATEAU_SIBLING_SIZE_TOLERANCE * max(feature.size_uv[k], other.size_uv[k])
                    for k in range(2)
                ):
                    return True
            return False

        lone = [feature for feature in band_features if not _has_sibling(feature)]
        for feature in lone:
            hits.remove(feature)
            reject(f"L{feature.level}_", "lone")
    # L0: the same face against a 150 mm base, keeping only protrusions the 60 mm levels did not
    # already report as a whole. A large pull that the 60 mm base split into corner fragments
    # appears here as one entry; the fragments stay listed and are read as its corners.
    large_base = _coarse_median_base(filled, LARGE_WINDOW_M / pitch)
    large_resid = np.where(valid, depth - large_base, 0.0)
    small_hits = list(hits)
    large_hits: list[FaceFeature] = []

    def consider_large(cells0: np.ndarray) -> None:
        before = len(hits)
        reason0, _ = evaluate(cells0, large_resid, 0)
        if reason0 == "band":
            # A rail or pull fused with the slightly proud door or panel it sits on: keep the
            # cells above half the peak, which separates the tall member from the low panel.
            peak = float(large_resid[cells0].max())
            labels_split, count_split = ndimage.label(
                cells0 & (large_resid > max(height_min, 0.5 * peak)), structure=structure
            )
            if count_split >= 1 and int((labels_split > 0).sum()) < int(cells0.sum()):
                for index_split in range(1, count_split + 1):
                    consider_large(labels_split == index_split)
                return
        if reason0 is not None:
            reject("L0_", reason0)
            return
        hit0 = hits.pop(before)
        if hit0.height_median_m < LARGE_HEIGHT_MED_MIN_M:
            reject("L0_", "low")
            return
        contained = [
            other
            for other in small_hits
            if hit0.u_range[0] <= other.center_uv[0] <= hit0.u_range[1]
            and hit0.v_range[0] <= other.center_uv[1] <= hit0.v_range[1]
        ]
        whole_already = any(
            abs(other.center_uv[0] - hit0.center_uv[0]) < DEDUPE_M
            and abs(other.center_uv[1] - hit0.center_uv[1]) < DEDUPE_M
            and 0.8 <= other.size_uv[0] / max(hit0.size_uv[0], 1e-9) <= 1.25
            and 0.8 <= other.size_uv[1] / max(hit0.size_uv[1], 1e-9) <= 1.25
            for other in contained
        )
        if whole_already:
            reject("L0_", "duplicate")
        elif max(hit0.size_uv) > SIZE_MAX_M or len(contained) >= LARGE_MIN_FRAGMENTS:
            large_hits.append(hit0)
        else:
            reject("L0_", "small")

    labels0, count0 = ndimage.label((large_resid > height_min) & valid, structure=structure)
    for index0 in range(1, count0 + 1):
        consider_large(labels0 == index0)
    large_hits.sort(key=lambda h: -(h.size_uv[0] * h.size_uv[1]))
    for dropped in large_hits[LARGE_PER_FACE_MAX:]:
        reject("L0_", "capped")
    hits.extend(large_hits[:LARGE_PER_FACE_MAX])
    unique: list[FaceFeature] = []
    for hit in hits:
        if any(
            abs(other.center_uv[0] - hit.center_uv[0]) < DEDUPE_M
            and abs(other.center_uv[1] - hit.center_uv[1]) < DEDUPE_M
            and other.level == hit.level
            for other in unique
        ):
            continue
        unique.append(hit)
    unique.sort(key=lambda h: (h.center_uv[1], h.center_uv[0]))
    return unique, rejected


def _coarse_median_base(filled: np.ndarray, window_cells: float) -> np.ndarray:
    """Median base over a wide window, computed on a coarsened grid and expanded back."""

    factor = max(1, int(round(window_cells / LARGE_WINDOW_CELLS_TARGET)))
    if factor == 1:
        window = max(5, int(round(window_cells)) | 1)
        return ndimage.median_filter(filled, size=window)
    coarse = filled[::factor, ::factor]
    window = max(5, int(round(window_cells / factor)) | 1)
    base_coarse = ndimage.median_filter(coarse, size=window)
    base = np.repeat(np.repeat(base_coarse, factor, axis=0), factor, axis=1)
    return base[: filled.shape[0], : filled.shape[1]]


def format_face_features(summary: FaceFeatureSummary) -> tuple[str, ...]:
    """Serialize the section; empty when nothing was detected (absence proves nothing)."""

    if not summary.features:
        return ()
    lines = [
        (
            f"[FACE_FEATURES faces=-Y,+Y,-X,+X,+Z,-Z pitch={summary.pitch_m:.4g} "
            f"shown={len(summary.features)} levels=L0_large_base,L1_face,L2_local_base,L3_half_peak]"
        ),
        (
            "semantics: compact protrusions above a locally fitted face base (knobs, buttons, "
            "handles, feet); anonymous, nonsemantic; absence never proves a face flat. L0 = a "
            "protrusion wider than the 60mm face base (a pull, a large knob, a door rail) read "
            "against a 150mm base; L1-L3 entries lying inside an L0 footprint are its corners or "
            "steps, not separate controls"
        ),
        (
            "coords: world metres, same q=0 frame as every section; c(u,v)=center on the face, "
            "size=u x v extent in mm, h=max/median height above the local base in mm"
        ),
    ]
    for feature in summary.features:
        axes = FACE_AXIS_LABELS[feature.face]
        lines.append(
            f"{feature.face} c({axes})=({feature.center_uv[0]:+.3f},{feature.center_uv[1]:+.3f}) "
            f"size={feature.size_uv[0] * 1000:.0f}x{feature.size_uv[1] * 1000:.0f}mm "
            f"h={feature.height_max_m * 1000:.0f}/{feature.height_median_m * 1000:.0f}mm "
            f"L{feature.level}"
        )
    return tuple(lines)


_FEATURE_LINE = re.compile(
    r"^(?P<face>[-+][XYZ]) c\([xyz],[xyz]\)=\((?P<cu>[+-][0-9.]+),(?P<cv>[+-][0-9.]+)\) "
    r"size=(?P<su>[0-9]+)x(?P<sv>[0-9]+)mm h=(?P<hmax>[0-9]+)/(?P<hmed>[0-9]+)mm L(?P<level>[0-3])$"
)


def parse_face_features(text: str) -> tuple[FaceFeature, ...]:
    """Read serialized FACE_FEATURES lines back into features (ranges derived from center/size)."""

    features: list[FaceFeature] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("[FACE_FEATURES"):
            inside = True
            continue
        if inside and line.startswith("["):
            break
        if not inside:
            continue
        match = _FEATURE_LINE.match(line.strip())
        if match is None:
            continue
        cu, cv = float(match["cu"]), float(match["cv"])
        su, sv = int(match["su"]) / 1000.0, int(match["sv"]) / 1000.0
        features.append(
            FaceFeature(
                face=match["face"],
                level=int(match["level"]),
                u_range=(cu - su / 2, cu + su / 2),
                v_range=(cv - sv / 2, cv + sv / 2),
                center_uv=(cu, cv),
                size_uv=(su, sv),
                height_max_m=int(match["hmax"]) / 1000.0,
                height_median_m=int(match["hmed"]) / 1000.0,
                cells=0,
            )
        )
    return tuple(features)


__all__ = [
    "FaceFeature",
    "FaceFeatureSummary",
    "detect_face_features",
    "format_face_features",
    "parse_face_features",
]
