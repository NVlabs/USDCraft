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

"""Sweep path preparation: dedupe, subdivision, section insertion, tangents and frames."""

from __future__ import annotations

from math import atan2, ceil, cos, isfinite, pi, sin
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from .common import (
    _EPS,
    Vec3,
    _clamp,
    _v_add,
    _v_cross,
    _v_dot,
    _v_lerp,
    _v_norm,
    _v_normalize,
    _v_scale,
    _v_sub,
)

_MAX_SWEEP_GENERATED_VERTICES = 250_000


def _dedupe_path_points(path: List[Vec3]) -> List[Vec3]:
    if not path:
        return []
    out = [path[0]]
    for p in path[1:]:
        if _v_norm(_v_sub(p, out[-1])) > _EPS:
            out.append(p)
    return out


def _compute_path_tangents(path: Sequence[Vec3], *, closed: bool) -> List[Vec3]:
    n = len(path)
    if n < 2:
        raise ValueError("Pipe path requires at least two distinct points")
    if closed and n < 3:
        raise ValueError("A closed pipe path requires at least three distinct points")

    tangents: List[Vec3] = []
    for i in range(n):
        if not closed and i == 0:
            d = _v_sub(path[1], path[0])
        elif not closed and i == n - 1:
            d = _v_sub(path[-1], path[-2])
        else:
            previous = path[(i - 1) % n]
            following = path[(i + 1) % n]
            incoming = _v_normalize(_v_sub(path[i], previous))
            outgoing = _v_normalize(_v_sub(following, path[i]))
            d = _v_add(incoming, outgoing)
            if _v_norm(d) <= _EPS:
                raise ValueError("Pipe path contains a 180-degree reversal with no unique tangent")
        tangents.append(_v_normalize(d))
    return tangents


def _initial_frame(tangent: Vec3, up_hint: Vec3) -> Tuple[Vec3, Vec3]:
    up = _v_normalize(up_hint)
    if abs(_v_dot(up, tangent)) > 0.95:
        up = (0.0, 1.0, 0.0) if abs(tangent[1]) < 0.95 else (1.0, 0.0, 0.0)
    n = _v_cross(up, tangent)
    if _v_norm(n) <= _EPS:
        n = _v_cross((0.0, 0.0, 1.0), tangent)
    n = _v_normalize(n)
    b = _v_normalize(_v_cross(tangent, n))
    return (n, b)


def _rotate_path_frame_vector(value: Vec3, axis: Vec3, angle: float) -> Vec3:
    cosine = cos(angle)
    sine = sin(angle)
    return _v_add(
        _v_add(
            _v_scale(value, cosine),
            _v_scale(_v_cross(axis, value), sine),
        ),
        _v_scale(axis, _v_dot(axis, value) * (1.0 - cosine)),
    )


def _transport_path_frame(
    normal: Vec3,
    binormal: Vec3,
    previous_tangent: Vec3,
    tangent: Vec3,
) -> Tuple[Vec3, Vec3]:
    axis = _v_cross(previous_tangent, tangent)
    sine = _v_norm(axis)
    cosine = _clamp(_v_dot(previous_tangent, tangent), -1.0, 1.0)
    if sine > _EPS:
        next_normal = _rotate_path_frame_vector(
            normal,
            _v_scale(axis, 1.0 / sine),
            atan2(sine, cosine),
        )
    elif cosine < 0.0:
        next_normal = _rotate_path_frame_vector(normal, binormal, pi)
    else:
        next_normal = normal

    next_normal = _v_sub(next_normal, _v_scale(tangent, _v_dot(next_normal, tangent)))
    if _v_norm(next_normal) <= _EPS:
        next_normal = _v_cross(binormal, tangent)
    next_normal = _v_normalize(next_normal)
    return next_normal, _v_normalize(_v_cross(tangent, next_normal))


def _signed_path_frame_angle(start: Vec3, end: Vec3, axis: Vec3) -> float:
    return atan2(_v_dot(axis, _v_cross(start, end)), _v_dot(start, end))


def _transport_path_frame_between_rings(
    normal: Vec3,
    binormal: Vec3,
    start_tangent: Vec3,
    end_tangent: Vec3,
    start_point: Vec3,
    end_point: Vec3,
) -> Tuple[Vec3, Vec3]:
    """Transport through the physical segment direction between two rings.

    Explicitly visiting the segment tangent makes the result invariant to
    inserting collinear section or subdivision rings along that segment.
    """

    segment_tangent = _v_normalize(_v_sub(end_point, start_point))
    segment_normal, segment_binormal = _transport_path_frame(
        normal,
        binormal,
        start_tangent,
        segment_tangent,
    )
    return _transport_path_frame(
        segment_normal,
        segment_binormal,
        segment_tangent,
        end_tangent,
    )


def _compute_path_frames(
    path: Sequence[Vec3],
    tangents: Sequence[Vec3],
    *,
    closed: bool,
    up_hint: Vec3,
) -> Tuple[List[Vec3], List[Vec3]]:
    normal, binormal = _initial_frame(tangents[0], up_hint)
    normals = [normal]
    binormals = [binormal]
    for index in range(1, len(path)):
        normal, binormal = _transport_path_frame_between_rings(
            normals[-1],
            binormals[-1],
            tangents[index - 1],
            tangents[index],
            path[index - 1],
            path[index],
        )
        normals.append(normal)
        binormals.append(binormal)

    if not closed:
        return normals, binormals

    closing_normal, _closing_binormal = _transport_path_frame_between_rings(
        normals[-1],
        binormals[-1],
        tangents[-1],
        tangents[0],
        path[-1],
        path[0],
    )
    residual = _signed_path_frame_angle(closing_normal, normals[0], tangents[0])
    fractions, _total = _sweep_path_fractions(path, closed=True)
    for index in range(1, len(path)):
        correction = residual * fractions[index]
        normals[index] = _v_normalize(
            _rotate_path_frame_vector(normals[index], tangents[index], correction)
        )
        binormals[index] = _v_normalize(_v_cross(tangents[index], normals[index]))
    return normals, binormals


def _sweep_path_fractions(
    path: Sequence[Vec3],
    *,
    closed: bool = False,
) -> Tuple[List[float], float]:
    cumulative = [0.0]
    for start, end in zip(path, path[1:]):
        cumulative.append(cumulative[-1] + _v_norm(_v_sub(end, start)))
    total = cumulative[-1]
    if closed:
        total += _v_norm(_v_sub(path[0], path[-1]))
    if not isfinite(total) or total <= _EPS:
        raise ValueError("Sweep path length must be finite and positive")
    return ([distance / total for distance in cumulative], total)


def _insert_sweep_section_points(
    path: Sequence[Vec3],
    positions: Iterable[float],
    *,
    closed: bool = False,
    profile_point_count: int,
    extra_vertices: int = 0,
) -> List[Vec3]:
    """Linearly merge sorted controls into path rings without dropping bends."""

    # Reject the input-ring lower bound before allocating fractions, sorted
    # controls, or any inserted path-ring storage.
    _validate_sweep_vertex_budget(
        len(path),
        profile_point_count,
        extra_vertices=extra_vertices,
    )
    fractions, _total = _sweep_path_fractions(path, closed=closed)
    ordered_controls = sorted(float(position) for position in positions)
    controls: List[float] = []
    for position in ordered_controls:
        if closed and position >= 1.0 - _EPS:
            continue
        if not controls or abs(position - controls[-1]) > _EPS:
            controls.append(position)

    def merged_events() -> Iterator[Tuple[Optional[int], float, int]]:
        control_index = 0
        for path_index, fraction in enumerate(fractions):
            while control_index < len(controls) and controls[control_index] < fraction - _EPS:
                yield None, controls[control_index], path_index
                control_index += 1

            # Original rings are always retained, even if two path fractions
            # happen to be closer than the section-position tolerance.
            yield path_index, fraction, path_index
            while control_index < len(controls) and controls[control_index] <= fraction + _EPS:
                control_index += 1

        while control_index < len(controls):
            yield None, controls[control_index], len(path)
            control_index += 1

    merged_ring_count = sum(1 for _event in merged_events())
    _validate_sweep_vertex_budget(
        merged_ring_count,
        profile_point_count,
        extra_vertices=extra_vertices,
    )

    result: List[Vec3] = []
    for source_index, position, following_index in merged_events():
        if source_index is not None:
            result.append(path[source_index])
            continue
        if following_index < len(path):
            previous_index = following_index - 1
            start_fraction = fractions[previous_index]
            end_fraction = fractions[following_index]
            amount = (position - start_fraction) / (end_fraction - start_fraction)
            result.append(
                _v_lerp(path[previous_index], path[following_index], _clamp(amount, 0.0, 1.0))
            )
            continue
        if not closed:
            result.append(path[-1])
            continue
        span = 1.0 - fractions[-1]
        amount = 0.0 if span <= _EPS else (position - fractions[-1]) / span
        result.append(_v_lerp(path[-1], path[0], _clamp(amount, 0.0, 1.0)))
    return result


def _validate_sweep_vertex_budget(
    ring_count: int,
    profile_point_count: int,
    *,
    extra_vertices: int = 0,
) -> None:
    total = int(ring_count) * int(profile_point_count) + int(extra_vertices)
    if total > _MAX_SWEEP_GENERATED_VERTICES:
        raise ValueError(
            "Sweep would generate "
            f"{total:,} vertices, exceeding the {_MAX_SWEEP_GENERATED_VERTICES:,}-vertex "
            "limit; increase max_segment_length, reduce path samples, or simplify profiles"
        )


def _subdivide_sweep_path(
    path: Sequence[Vec3],
    *,
    max_segment_length: Optional[float],
    closed: bool = False,
    profile_point_count: int,
    extra_vertices: int = 0,
) -> List[Vec3]:
    """Subdivide without moving existing bends or inserted section controls."""

    if max_segment_length is None:
        _validate_sweep_vertex_budget(
            len(path),
            profile_point_count,
            extra_vertices=extra_vertices,
        )
        return list(path)
    maximum = float(max_segment_length)
    if not isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_segment_length must be finite and positive")

    segment_count = len(path) if closed else len(path) - 1
    divisions_per_segment: List[int] = []
    predicted_ring_count = 0
    for index in range(segment_count):
        start = path[index]
        end = path[(index + 1) % len(path)]
        segment_length = _v_norm(_v_sub(end, start))
        ratio = segment_length / maximum
        if not isfinite(ratio):
            raise ValueError(
                "max_segment_length is too small for the path and would exceed the "
                f"{_MAX_SWEEP_GENERATED_VERTICES:,}-vertex sweep limit"
            )
        divisions = max(1, int(ceil(ratio)))
        divisions_per_segment.append(divisions)
        predicted_ring_count += divisions

    if not closed:
        predicted_ring_count += 1
    _validate_sweep_vertex_budget(
        predicted_ring_count,
        profile_point_count,
        extra_vertices=extra_vertices,
    )

    result: List[Vec3] = []
    for index, divisions in enumerate(divisions_per_segment):
        start = path[index]
        end = path[(index + 1) % len(path)]
        result.extend(_v_lerp(start, end, sample / divisions) for sample in range(divisions))
    if not closed:
        result.append(path[-1])
    return result
