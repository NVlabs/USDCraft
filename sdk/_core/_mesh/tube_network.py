# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

"""Welded tube networks built from shared-node wire paths."""

from __future__ import annotations

import logging
from math import atan2, sqrt
from typing import Iterable, List, Optional, Sequence, Tuple

from ._sweep_paths import _dedupe_path_points
from .booleans import (
    _boolean_union_many,
    _is_expected_boolean_fallback_error,
)
from .common import (
    _EPS,
    Vec3,
    _v_lerp,
    _v_norm,
    _v_normalize,
    _v_sub,
)
from .primitives import (
    CylinderGeometry,
    MeshGeometry,
    SphereGeometry,
)
from .sweeps import (
    _build_wire_centerline,
    _preprocess_wire_polyline_points,
    wire_from_points,
)

logger = logging.getLogger(__name__)


def _rpy_from_z_axis(direction: Vec3) -> Vec3:
    d = _v_normalize(direction)
    yaw = atan2(d[1], d[0])
    pitch = atan2(sqrt(d[0] * d[0] + d[1] * d[1]), d[2])
    return (0.0, pitch, yaw)


def _vec3_quant_key(point: Vec3, *, tol: float) -> Tuple[int, int, int]:
    scale = 1.0 / max(float(tol), _EPS)
    return (
        int(round(point[0] * scale)),
        int(round(point[1] * scale)),
        int(round(point[2] * scale)),
    )


def _coerce_tube_network_paths(paths: Iterable[Iterable[Vec3]]) -> List[List[Vec3]]:
    out: List[List[Vec3]] = []
    for path in paths:
        pts = [(float(x), float(y), float(z)) for (x, y, z) in path]
        if len(pts) < 2:
            raise ValueError(
                "tube_network_from_paths requires each path to contain at least two points"
            )
        out.append(pts)
    if not out:
        raise ValueError("tube_network_from_paths requires at least one path")
    return out


def _is_closed_network_path(path: Sequence[Vec3], *, tol: float) -> bool:
    return len(path) >= 3 and _v_norm(_v_sub(path[0], path[-1])) <= max(float(tol), _EPS)


def _collect_shared_network_nodes(paths: Sequence[Sequence[Vec3]], *, tol: float) -> List[Vec3]:
    counts: dict[Tuple[int, int, int], int] = {}
    canonical: dict[Tuple[int, int, int], Vec3] = {}
    for path in paths:
        seen: set[Tuple[int, int, int]] = set()
        for point in path:
            key = _vec3_quant_key(point, tol=tol)
            canonical.setdefault(key, (float(point[0]), float(point[1]), float(point[2])))
            seen.add(key)
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    return [canonical[key] for key, count in counts.items() if count > 1]


def _segment_cylinder_geometry(
    start: Vec3,
    end: Vec3,
    *,
    radius: float,
    radial_segments: int,
) -> Optional[MeshGeometry]:
    segment = _v_sub(end, start)
    length = _v_norm(segment)
    if length <= _EPS:
        return None
    mid = _v_lerp(start, end, 0.5)
    _roll, pitch, yaw = _rpy_from_z_axis(segment)
    return (
        CylinderGeometry(
            radius=float(radius),
            height=length,
            radial_segments=max(6, int(radial_segments)),
            closed=True,
        )
        .rotate_y(pitch)
        .rotate_z(yaw)
        .translate(*mid)
    )


def _tube_network_visual_mesh_from_paths(
    path_list: Sequence[Sequence[Vec3]],
    *,
    radius: float,
    radial_segments: int,
    cap_ends: bool,
    corner_mode: str,
    corner_radius: float,
    corner_segments: int,
    up_hint: Tuple[float, float, float],
    min_segment_length: float,
    shared_node_radius: float,
) -> MeshGeometry:
    out = MeshGeometry()
    for path in path_list:
        closed_path = _is_closed_network_path(path, tol=min_segment_length)
        out.merge(
            wire_from_points(
                path,
                radius=radius,
                radial_segments=radial_segments,
                closed_path=closed_path,
                cap_ends=cap_ends,
                corner_mode=corner_mode,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
                up_hint=up_hint,
                min_segment_length=min_segment_length,
            )
        )

    if shared_node_radius > 0.0:
        width_segments = max(12, int(radial_segments))
        height_segments = max(8, int(radial_segments) // 2)
        for point in _collect_shared_network_nodes(path_list, tol=min_segment_length):
            out.merge(
                SphereGeometry(
                    radius=shared_node_radius,
                    width_segments=width_segments,
                    height_segments=height_segments,
                ).translate(*point)
            )
    return out


def tube_network_from_paths(
    paths: Iterable[Iterable[Vec3]],
    *,
    radius: float,
    radial_segments: int = 16,
    cap_ends: bool = True,
    corner_mode: str = "fillet",
    corner_radius: float = 0.0,
    corner_segments: int = 8,
    up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    min_segment_length: float = 1e-6,
    shared_node_radius: Optional[float] = None,
) -> MeshGeometry:
    """
    Build a tubular frame/network from multiple centerline paths.

    This helper aims to return a watertight solid by boolean-unioning closed
    segment cylinders and optional node spheres. If the boolean backend is not
    available, it falls back to the older merge-based visual mesh construction.
    """
    radius = float(radius)
    node_radius = radius if shared_node_radius is None else float(shared_node_radius)
    tol = max(float(min_segment_length), _EPS)
    path_list = _coerce_tube_network_paths(paths)

    if radius <= 0.0:
        raise ValueError("radius must be > 0")
    if node_radius < 0.0:
        raise ValueError("shared_node_radius must be >= 0")

    solids: List[MeshGeometry] = []
    for path in path_list:
        closed_path = _is_closed_network_path(path, tol=tol)
        preprocessed = _preprocess_wire_polyline_points(
            path,
            min_segment_length=min_segment_length,
            closed_path=closed_path,
        )
        centerline = _build_wire_centerline(
            preprocessed,
            closed_path=closed_path,
            corner_mode=corner_mode,
            corner_radius=corner_radius,
            corner_segments=corner_segments,
        )
        centerline = _dedupe_path_points(centerline)
        for i in range(len(centerline) - 1):
            segment_geom = _segment_cylinder_geometry(
                centerline[i],
                centerline[i + 1],
                radius=radius,
                radial_segments=radial_segments,
            )
            if segment_geom is not None:
                solids.append(segment_geom)

    if node_radius > 0.0 and cap_ends:
        width_segments = max(12, int(radial_segments))
        height_segments = max(8, int(radial_segments) // 2)
        node_keys: set[Tuple[int, int, int]] = set()
        for path in path_list:
            for point in path:
                node_keys.add(_vec3_quant_key(point, tol=tol))
        canonical = {
            _vec3_quant_key(point, tol=tol): (float(point[0]), float(point[1]), float(point[2]))
            for path in path_list
            for point in path
        }
        for key in sorted(node_keys):
            solids.append(
                SphereGeometry(
                    radius=node_radius,
                    width_segments=width_segments,
                    height_segments=height_segments,
                ).translate(*canonical[key])
            )

    if not solids:
        raise ValueError("tube_network_from_paths produced no solid segments")

    try:
        out = _boolean_union_many(solids)
    except Exception as exc:
        if not _is_expected_boolean_fallback_error(exc):
            raise
        logger.warning(
            "tube_network_from_paths falling back to visual mesh after boolean union failure: %s: %s",
            type(exc).__name__,
            exc,
        )
        out = _tube_network_visual_mesh_from_paths(
            path_list,
            radius=radius,
            radial_segments=radial_segments,
            cap_ends=cap_ends,
            corner_mode=corner_mode,
            corner_radius=corner_radius,
            corner_segments=corner_segments,
            up_hint=up_hint,
            min_segment_length=min_segment_length,
            shared_node_radius=node_radius if cap_ends else 0.0,
        )
    return out
