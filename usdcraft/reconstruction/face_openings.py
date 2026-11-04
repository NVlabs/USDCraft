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

"""FACE_OPENINGS: enclosed recesses and openings on the six axis-aligned faces of the source.

The complement of ``FACE_FEATURES``: where a face steps *in* instead of out — a bread slot, a
basin, a cup recess, an open box top, a door left open in the scan. Each face is a first-surface
depth map; the reference plane is the face's own rim when the rim is one continuous plane
(``rim``), the silhouette outline when the face plane itself is missing (``outline``: an open box
seen from above), otherwise the depth most columns share (``mode``). Two nested levels: a
``recess`` steps in at least 6 mm, an ``opening`` at least 15 mm or has no surface at all; a recess
carrying openings is a recessed panel with N openings. The visible depth is a lower bound — the
scanner sees a cavity only down to the first occluding surface — and the task treats it so.

Typical detections are toaster slots, sink basins, open box tops, drawer-box trays and cup
recesses; plain drawer fronts, closed appliance bodies and pans stay silent. Known limits: shallow
burner recesses on a flat cooktop, grid and wire objects (capped per face).
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

RECESS_MIN_M = 0.006
OPENING_MIN_M = 0.015
MIN_DIM_M = 0.015
MIN_AREA_M2 = 4e-4
RIM_BAND_M = 0.030
RIM_TOLERANCE_M = 0.010
RIM_CONTINUITY = 0.75
OUTLINE_GAP_FRACTION = 0.20
OPEN_FACE_FRACTION = 0.60
# A face is not "open" while one solid patch of it lies on the reference plane: a range hood's
# fascia in front of a receding chimney holds the controls and is a face. The patch must cover this
# share of the face and fill its own box (an open box top leaves only a thin rim ring on the plane).
OPEN_FACE_PLANE_MIN_SHARE = 0.10
OPEN_FACE_PLANE_FILL_MIN = 0.50
MAX_PER_FACE = 6


@dataclass(frozen=True)
class FaceOpening:
    face: str
    kind: str  # "recess" | "opening"
    center_uv: tuple[float, float]
    size_uv: tuple[float, float]
    depth_p50_m: float | None
    depth_max_m: float | None
    through_fraction: float
    open_face: bool
    contains: int
    reference: str  # rim | outline | mode

    def to_dict(self) -> dict[str, Any]:
        return {
            "face": self.face,
            "kind": self.kind,
            "center_uv_m": [round(self.center_uv[0], 4), round(self.center_uv[1], 4)],
            "size_uv_m": [round(self.size_uv[0], 4), round(self.size_uv[1], 4)],
            "visible_depth_p50_m": None if self.depth_p50_m is None else round(self.depth_p50_m, 4),
            "visible_depth_max_m": None if self.depth_max_m is None else round(self.depth_max_m, 4),
            "depth_is_lower_bound": True,
            "through_fraction": round(self.through_fraction, 3),
            "open_face": self.open_face,
            "contains_openings": self.contains,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class FaceOpeningSummary:
    openings: tuple[FaceOpening, ...]
    pitch_m: float
    section_bytes: int = 0


def _sample_target(extent: np.ndarray, pitch: float) -> int:
    cells = 2.0 * (extent[0] * extent[1] + extent[0] * extent[2] + extent[1] * extent[2])
    return int(min(SAMPLE_MAX, max(SAMPLE_MIN, SAMPLES_PER_CELL * cells / (pitch * pitch))))


def detect_face_openings(components: Sequence[Any], *, seed: int = 3) -> FaceOpeningSummary:
    """Detect enclosed recesses and openings on all six faces (metres, source frame)."""

    grid = extent_and_pitch(components)
    if grid is None:
        return FaceOpeningSummary((), MIN_PITCH_M)
    extent, pitch = grid
    points = sample_surface_points(components, target=_sample_target(extent, pitch), seed=seed)
    found: list[FaceOpening] = []
    for name, axis, sign, u_axis, v_axis in FACE_FEATURE_FACES:
        found.extend(_detect_face(points, name, axis, sign, u_axis, v_axis, pitch, extent))
    summary = FaceOpeningSummary(tuple(found), pitch)
    return FaceOpeningSummary(
        tuple(found), pitch, section_byte_count(format_face_openings(summary))
    )


def _detect_face(
    points: np.ndarray,
    name: str,
    axis: int,
    sign: int,
    u_axis: int,
    v_axis: int,
    pitch: float,
    extent: np.ndarray,
) -> list[FaceOpening]:
    raster = rasterize_face(points, axis, sign, u_axis, v_axis, pitch)
    if raster is None:
        return []
    depth, valid, u0, v0 = raster
    footprint = ndimage.binary_fill_holes(valid)
    holes = footprint & ~valid
    labels, count = ndimage.label(holes)
    through = np.zeros_like(valid)
    if count:
        sizes = ndimage.sum(holes, labels, range(1, count + 1))
        for index, size in enumerate(sizes, start=1):
            if size * pitch * pitch > SAMPLING_GAP_MAX_M2:
                through |= labels == index
    margin = silhouette_margin_cells(pitch)
    interior = ndimage.binary_erosion(footprint, iterations=margin)
    band = max(2, int(round(RIM_BAND_M / pitch)))
    rim = footprint & ~ndimage.binary_erosion(footprint, iterations=band) & valid
    values = depth[valid]
    rim_values = depth[rim] if rim.any() else values
    rim_wall = float(np.percentile(rim_values, 90))
    rim_ok = float(np.mean(np.abs(rim_values - rim_wall) < RIM_TOLERANCE_M))
    bins = max(10, int((values.max() - values.min()) / pitch) + 1)
    hist, edges = np.histogram(values, bins=bins)
    mode = float(edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2)
    p95 = float(np.percentile(values, 95))
    if rim_ok >= RIM_CONTINUITY:
        wall, reference = rim_wall, "rim"
    elif p95 - mode > OUTLINE_GAP_FRACTION * float(extent[axis]):
        wall, reference = p95, "outline"
    else:
        wall, reference = max(mode, float(np.percentile(values, 60))), "mode"
    recess = np.where(valid, wall - depth, 0.0)
    face_area = float(footprint.sum()) * pitch * pitch
    structure = np.ones((3, 3))
    face_may_be_open = True
    plane_mask = interior & valid & (np.abs(depth - wall) < RIM_TOLERANCE_M)
    if plane_mask.any():
        plane_labels, plane_count = ndimage.label(plane_mask, structure=structure)
        plane_sizes = ndimage.sum(plane_mask, plane_labels, range(1, plane_count + 1))
        patch = plane_labels == (int(np.argmax(plane_sizes)) + 1)
        pys, pxs = np.where(patch)
        patch_box = (pys.max() - pys.min() + 1) * (pxs.max() - pxs.min() + 1)
        patch_area = float(patch.sum()) * pitch * pitch
        if (
            patch_area >= OPEN_FACE_PLANE_MIN_SHARE * face_area
            and float(patch.sum()) / patch_box >= OPEN_FACE_PLANE_FILL_MIN
        ):
            face_may_be_open = False

    def regions(mask: np.ndarray, kind: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        lab, n = ndimage.label(mask, structure=structure)
        for index in range(1, n + 1):
            cells = lab == index
            area = float(cells.sum()) * pitch * pitch
            if area < MIN_AREA_M2:
                continue
            ys, xs = np.where(cells)
            su = (ys.max() - ys.min() + 1) * pitch
            sv = (xs.max() - xs.min() + 1) * pitch
            if min(su, sv) < MIN_DIM_M:
                continue
            open_face = face_may_be_open and area > OPEN_FACE_FRACTION * face_area
            if open_face and kind == "recess":
                continue
            if not open_face and not interior[cells].all():
                continue
            rv = recess[cells & valid]
            out.append(
                {
                    "cells": cells,
                    "kind": kind,
                    "center": (float(ys.mean()) * pitch + u0, float(xs.mean()) * pitch + v0),
                    "size": (su, sv),
                    "area": area,
                    "p50": float(np.percentile(rv, 50)) if len(rv) else None,
                    "max": float(rv.max()) if len(rv) else None,
                    "through": float(through[cells].mean()),
                    "open_face": open_face,
                }
            )
        return out

    deep = regions(((recess > OPENING_MIN_M) & valid) | through, "opening")
    shallow = regions(((recess > RECESS_MIN_M) & valid) | through, "recess")
    kept: list[dict[str, Any]] = []
    for item in shallow:
        inside = [d for d in deep if (d["cells"] & item["cells"]).sum() > 0.9 * d["cells"].sum()]
        if inside and sum(d["cells"].sum() for d in inside) > 0.85 * item["cells"].sum():
            continue  # the same opening seen at the shallow level
        item["contains"] = len(inside)
        kept.append(item)
    for item in deep:
        item["contains"] = 0
        kept.append(item)
    kept.sort(key=lambda item: -item["area"])
    return [
        FaceOpening(
            face=name,
            kind=item["kind"],
            center_uv=item["center"],
            size_uv=item["size"],
            depth_p50_m=item["p50"],
            depth_max_m=item["max"],
            through_fraction=item["through"],
            open_face=item["open_face"],
            contains=item["contains"],
            reference=reference,
        )
        for item in kept[:MAX_PER_FACE]
    ]


def format_face_openings(summary: FaceOpeningSummary) -> tuple[str, ...]:
    """Serialize the section; empty when nothing was detected (absence proves nothing)."""

    if not summary.openings:
        return ()
    lines = [
        (
            f"[FACE_OPENINGS faces=-Y,+Y,-X,+X,+Z,-Z pitch={summary.pitch_m:.4g} "
            f"shown={len(summary.openings)} levels=recess>=6mm,opening>=15mm_or_through]"
        ),
        (
            "semantics: enclosed recesses and openings on each axis face (slots, basins, cup wells, "
            "open tops, doors left open); anonymous, nonsemantic; a recess with in=N carries N "
            "openings; open_face means the whole face is open"
        ),
        (
            "coords: world metres, same q=0 frame; c(u,v)=centre on the face, size=plan extent in mm, "
            "depth>=p50 visible depth in mm is a LOWER BOUND (the scan sees a cavity only to its "
            "first occluding surface; the cavity's real depth follows the normal-use audit), "
            "max=deepest visible, through=share of columns with no surface at all"
        ),
    ]
    for item in summary.openings:
        axes = FACE_AXIS_LABELS[item.face]
        p50 = "n/a" if item.depth_p50_m is None else f"{item.depth_p50_m * 1000:.0f}"
        mx = "n/a" if item.depth_max_m is None else f"{item.depth_max_m * 1000:.0f}"
        flags = " open_face" if item.open_face else ""
        contains = f" in={item.contains}" if item.contains else ""
        lines.append(
            f"{item.face} c({axes})=({item.center_uv[0]:+.3f},{item.center_uv[1]:+.3f}) "
            f"size={item.size_uv[0] * 1000:.0f}x{item.size_uv[1] * 1000:.0f}mm depth>={p50}mm "
            f"max={mx}mm through={item.through_fraction:.0%} kind={item.kind}"
            f"{contains}{flags} ref={item.reference}"
        )
    return tuple(lines)


_OPENING_LINE = re.compile(
    r"^(?P<face>[-+][XYZ]) c\([xyz],[xyz]\)=\((?P<cu>[+-][0-9.]+),(?P<cv>[+-][0-9.]+)\) "
    r"size=(?P<su>[0-9]+)x(?P<sv>[0-9]+)mm depth>=(?P<p50>[0-9]+|n/a)mm max=(?P<max>[0-9]+|n/a)mm "
    r"through=(?P<through>[0-9]+)% kind=(?P<kind>recess|opening)(?: in=(?P<contains>\d+))?"
    r"(?P<open> open_face)? ref=(?P<ref>rim|outline|mode)$"
)


def parse_face_openings(text: str) -> tuple[FaceOpening, ...]:
    """Read serialized FACE_OPENINGS lines back into openings."""

    out: list[FaceOpening] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("[FACE_OPENINGS"):
            inside = True
            continue
        if inside and line.startswith("["):
            break
        if not inside:
            continue
        match = _OPENING_LINE.match(line.strip())
        if match is None:
            continue
        p50 = None if match["p50"] == "n/a" else int(match["p50"]) / 1000.0
        mx = None if match["max"] == "n/a" else int(match["max"]) / 1000.0
        out.append(
            FaceOpening(
                face=match["face"],
                kind=match["kind"],
                center_uv=(float(match["cu"]), float(match["cv"])),
                size_uv=(int(match["su"]) / 1000.0, int(match["sv"]) / 1000.0),
                depth_p50_m=p50,
                depth_max_m=mx,
                through_fraction=int(match["through"]) / 100.0,
                open_face=bool(match["open"]),
                contains=int(match["contains"] or 0),
                reference=match["ref"],
            )
        )
    return tuple(out)


__all__ = [
    "FaceOpening",
    "FaceOpeningSummary",
    "detect_face_openings",
    "format_face_openings",
    "parse_face_openings",
]
