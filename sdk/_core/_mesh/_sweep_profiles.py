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

"""Sweep section profiles: corner alignment, resampling and interpolation along the path."""

from __future__ import annotations

import sys
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .common import (
    _EPS,
    Vec2,
    _clamp,
    _polygon_area,
)

_MAX_SWEEP_ALIGNMENT_POINTS = 262_144
_MAX_SWEEP_ALIGNMENT_SAMPLES = 262_144


@dataclass(frozen=True)
class _SweepSectionProfile:
    position: float
    points: Tuple[Vec2, ...]
    scale: Tuple[float, float]
    rotation: float
    offset: Vec2
    interpolation: str
    tension: float


def _validate_sweep_alignment_budget(control_count: int, profile_point_count: int) -> None:
    total = int(control_count) * int(profile_point_count)
    if total > _MAX_SWEEP_ALIGNMENT_POINTS:
        raise ValueError(
            "Sweep profile alignment would require "
            f"{total:,} control-profile points, exceeding the "
            f"{_MAX_SWEEP_ALIGNMENT_POINTS:,}-point alignment limit; reduce sections or "
            "simplify profiles"
        )


def _resample_sweep_profile(
    points: Sequence[Vec2],
    count: int,
    *,
    closed: bool,
) -> List[Vec2]:
    """Resample a profile by perimeter/arc length while preserving its orientation."""

    if len(points) == count:
        return list(points)
    edge_count = len(points) if closed else len(points) - 1
    cumulative = [0.0]
    for index in range(edge_count):
        start = points[index]
        end = points[(index + 1) % len(points)]
        cumulative.append(
            cumulative[-1] + sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        )
    total = cumulative[-1]
    if not isfinite(total) or total <= _EPS:
        raise ValueError("Sweep profile perimeter must be finite and positive")

    if closed:
        targets = [total * index / count for index in range(count)]
    else:
        targets = [total * index / (count - 1) for index in range(count)]

    result: List[Vec2] = []
    edge_index = 0
    for target in targets:
        while edge_index + 1 < len(cumulative) - 1 and cumulative[edge_index + 1] < target:
            edge_index += 1
        span = cumulative[edge_index + 1] - cumulative[edge_index]
        amount = 0.0 if span <= _EPS else (target - cumulative[edge_index]) / span
        start = points[edge_index]
        end = points[(edge_index + 1) % len(points)]
        result.append(
            (
                start[0] + (end[0] - start[0]) * amount,
                start[1] + (end[1] - start[1]) * amount,
            )
        )
    return result


def _sweep_profile_knots(
    points: Sequence[Vec2],
    *,
    closed: bool,
) -> Tuple[List[float], float]:
    edge_count = len(points) if closed else len(points) - 1
    cumulative = [0.0]
    for index in range(edge_count):
        start = points[index]
        end = points[(index + 1) % len(points)]
        cumulative.append(
            cumulative[-1] + sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        )
    total = cumulative[-1]
    if not isfinite(total) or total <= _EPS:
        raise ValueError("Sweep profile perimeter must be finite and positive")
    values = cumulative[:-1] if closed else cumulative
    return ([distance / total for distance in values], total)


def _sample_sweep_profile_at_knots(
    points: Sequence[Vec2],
    targets: Sequence[float],
    *,
    closed: bool,
) -> List[Vec2]:
    knots, _total = _sweep_profile_knots(points, closed=closed)
    extended_knots = [*knots, 1.0] if closed else list(knots)
    extended_points = [*points, points[0]] if closed else list(points)
    result: List[Vec2] = []
    edge_index = 0
    for target in targets:
        while (
            edge_index + 1 < len(extended_knots) - 1
            and extended_knots[edge_index + 1] < target - _EPS
        ):
            edge_index += 1
        if abs(target - extended_knots[edge_index]) <= _EPS:
            result.append(extended_points[edge_index])
            continue
        if abs(target - extended_knots[edge_index + 1]) <= _EPS:
            result.append(extended_points[edge_index + 1])
            continue
        span = extended_knots[edge_index + 1] - extended_knots[edge_index]
        amount = (target - extended_knots[edge_index]) / span
        start = extended_points[edge_index]
        end = extended_points[edge_index + 1]
        result.append(
            (
                start[0] + (end[0] - start[0]) * amount,
                start[1] + (end[1] - start[1]) * amount,
            )
        )
    return result


def _closed_sweep_profile_alignment_phase(
    reference: Sequence[Vec2],
    candidate: Sequence[Vec2],
) -> float:
    sample_count = min(
        _MAX_SWEEP_ALIGNMENT_SAMPLES,
        max(32, 4 * max(len(reference), len(candidate))),
    )
    reference_samples = np.asarray(
        _resample_sweep_profile(reference, sample_count, closed=True),
        dtype=np.float64,
    )
    candidate_samples = np.asarray(
        _resample_sweep_profile(candidate, sample_count, closed=True),
        dtype=np.float64,
    )
    correlation = np.zeros(sample_count, dtype=np.float64)
    for axis in range(2):
        correlation += np.fft.irfft(
            np.conj(np.fft.rfft(reference_samples[:, axis]))
            * np.fft.rfft(candidate_samples[:, axis]),
            n=sample_count,
        )
    return float(int(np.argmax(correlation))) / float(sample_count)


def _exact_closed_sweep_profile_alignment_phase(
    reference: Sequence[Vec2],
    candidate: Sequence[Vec2],
) -> Optional[float]:
    """Return the exact cyclic phase when two authored vertex sequences match.

    A KMP search keeps this linear in profile size, including profiles with
    repeated coordinates.  Detecting this before sampled correlation prevents
    an approximate phase from inserting a needless seam point into an already
    identical cyclic profile.
    """

    if len(reference) != len(candidate) or not reference:
        return None

    prefix = [0] * len(reference)
    matched = 0
    for index in range(1, len(reference)):
        while matched and reference[index] != reference[matched]:
            matched = prefix[matched - 1]
        if reference[index] == reference[matched]:
            matched += 1
            prefix[index] = matched

    matched = 0
    for index in range(2 * len(candidate) - 1):
        value = candidate[index % len(candidate)]
        while matched and value != reference[matched]:
            matched = prefix[matched - 1]
        if value == reference[matched]:
            matched += 1
            if matched == len(reference):
                start = index - len(reference) + 1
                knots, _total = _sweep_profile_knots(candidate, closed=True)
                return knots[start]
    return None


def _best_closed_sweep_profile_shift(
    reference: Sequence[Vec2],
    candidate: Sequence[Vec2],
) -> int:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Discrete closed-profile alignment requires equal non-empty profiles")
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array -= np.mean(reference_array, axis=0)
    candidate_array -= np.mean(candidate_array, axis=0)
    correlation = np.zeros(len(candidate), dtype=np.float64)
    for axis in range(2):
        correlation += np.fft.irfft(
            np.conj(np.fft.rfft(reference_array[:, axis])) * np.fft.rfft(candidate_array[:, axis]),
            n=len(candidate),
        )
    return int(np.argmax(correlation))


def _ordered_sweep_profiles_close(
    first: Sequence[Vec2],
    second: Sequence[Vec2],
    *,
    relative_tolerance: float = 1e-9,
) -> bool:
    if len(first) != len(second) or not first:
        return False
    all_points = [*first, *second]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    extent = max(max(x_values) - min(x_values), max(y_values) - min(y_values))
    coordinate_scale = max(abs(component) for point in all_points for component in point)
    tolerance = max(
        extent * relative_tolerance,
        coordinate_scale * 64.0 * sys.float_info.epsilon,
        sys.float_info.min,
    )
    tolerance_squared = tolerance * tolerance
    return all(
        (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 <= tolerance_squared
        for left, right in zip(first, second)
    )


def _rotate_closed_sweep_profile_to_phase(
    points: Sequence[Vec2],
    phase: float,
) -> List[Vec2]:
    knots, _total = _sweep_profile_knots(points, closed=True)
    for index, knot in enumerate(knots):
        if abs(phase - knot) <= _EPS:
            return [*points[index:], *points[:index]]

    seam = _sample_sweep_profile_at_knots(points, [phase], closed=True)[0]
    shifted_corners = sorted(
        (((knot - phase) % 1.0, point) for knot, point in zip(knots, points)),
        key=lambda item: item[0],
    )
    return [seam, *(point for shifted, point in shifted_corners if shifted > _EPS)]


def _align_sweep_profile_corners(
    reference: Sequence[Vec2],
    candidate: Sequence[Vec2],
    *,
    closed: bool,
) -> List[Vec2]:
    if closed:
        phase = _exact_closed_sweep_profile_alignment_phase(reference, candidate)
        if phase is not None:
            return _rotate_closed_sweep_profile_to_phase(candidate, phase)
        if len(reference) == len(candidate):
            shift = _best_closed_sweep_profile_shift(reference, candidate)
            shifted = [*candidate[shift:], *candidate[:shift]]
            if _ordered_sweep_profiles_close(reference, shifted):
                return shifted
        return _rotate_closed_sweep_profile_to_phase(
            candidate,
            _closed_sweep_profile_alignment_phase(reference, candidate),
        )

    count = max(len(reference), len(candidate))
    reference_samples = _resample_sweep_profile(reference, count, closed=False)
    candidate_samples = _resample_sweep_profile(candidate, count, closed=False)
    reversed_candidate = list(reversed(candidate))
    reversed_samples = _resample_sweep_profile(reversed_candidate, count, closed=False)
    if _sweep_profile_distance(reference_samples, reversed_samples) < _sweep_profile_distance(
        reference_samples,
        candidate_samples,
    ):
        return reversed_candidate
    return list(candidate)


def _merged_sweep_profile_knots(
    profiles: Sequence[Sequence[Vec2]],
    *,
    closed: bool,
) -> List[float]:
    values: List[float] = []
    for profile in profiles:
        knots, _total = _sweep_profile_knots(profile, closed=closed)
        values.extend(knots)
    values.sort()
    merged: List[float] = []
    for value in values:
        if not merged or abs(value - merged[-1]) > _EPS:
            merged.append(value)
    if not closed and (not merged or merged[-1] < 1.0 - _EPS):
        merged.append(1.0)
    return merged


def _sweep_profile_distance(first: Sequence[Vec2], second: Sequence[Vec2]) -> float:
    return sum(
        (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 for left, right in zip(first, second)
    )


def _transform_sweep_profile(
    points: Sequence[Vec2],
    *,
    scale: Tuple[float, float],
    rotation: float,
    offset: Vec2,
) -> List[Vec2]:
    scale_x, scale_y = scale
    offset_x, offset_y = offset
    cosine = cos(rotation)
    sine = sin(rotation)
    return [
        (
            x * scale_x * cosine - y * scale_y * sine + offset_x,
            x * scale_x * sine + y * scale_y * cosine + offset_y,
        )
        for x, y in points
    ]


def _sweep_profile_derivative(
    profiles: Sequence[_SweepSectionProfile],
    index: int,
    *,
    path_closed: bool,
    tension: float,
) -> List[Vec2]:
    previous_index, following_index, previous_position, following_position = (
        _sweep_derivative_neighbors(profiles, index, path_closed=path_closed)
    )
    previous = profiles[previous_index]
    following = profiles[following_index]
    span = following_position - previous_position
    if span <= _EPS:
        return [(0.0, 0.0) for _point in previous.points]
    factor = (1.0 - tension) / span
    return [
        (
            (end[0] - start[0]) * factor,
            (end[1] - start[1]) * factor,
        )
        for start, end in zip(previous.points, following.points)
    ]


def _sweep_derivative_neighbors(
    profiles: Sequence[_SweepSectionProfile],
    index: int,
    *,
    path_closed: bool,
) -> Tuple[int, int, float, float]:
    last_index = len(profiles) - 1
    if path_closed and index == 0:
        previous_index = last_index - 1
        following_index = 1
        previous_position = profiles[previous_index].position - 1.0
        following_position = profiles[following_index].position
    elif path_closed and index == last_index:
        previous_index = last_index - 1
        following_index = 1
        previous_position = profiles[previous_index].position
        following_position = profiles[following_index].position + 1.0
    elif index == 0:
        previous_index = 0
        following_index = 1
        previous_position = profiles[previous_index].position
        following_position = profiles[following_index].position
    elif index == last_index:
        previous_index = last_index - 1
        following_index = last_index
        previous_position = profiles[previous_index].position
        following_position = profiles[following_index].position
    else:
        previous_index = index - 1
        following_index = index + 1
        previous_position = profiles[previous_index].position
        following_position = profiles[following_index].position
    return previous_index, following_index, previous_position, following_position


def _sweep_scalar_derivative(
    profiles: Sequence[_SweepSectionProfile],
    value_at: Callable[[_SweepSectionProfile], float],
    index: int,
    *,
    path_closed: bool,
    tension: float,
    cyclic_delta: float = 0.0,
) -> float:
    previous_index, following_index, previous_position, following_position = (
        _sweep_derivative_neighbors(profiles, index, path_closed=path_closed)
    )
    previous_value = value_at(profiles[previous_index])
    following_value = value_at(profiles[following_index])
    if path_closed and index == 0:
        previous_value -= cyclic_delta
    elif path_closed and index == len(profiles) - 1:
        following_value += cyclic_delta
    span = following_position - previous_position
    if span <= _EPS:
        return 0.0
    return (following_value - previous_value) * (1.0 - tension) / span


def _cubic_hermite_value(
    first: float,
    second: float,
    first_derivative: float,
    second_derivative: float,
    *,
    span: float,
    amount: float,
) -> float:
    amount2 = amount * amount
    amount3 = amount2 * amount
    return (
        (2.0 * amount3 - 3.0 * amount2 + 1.0) * first
        + (amount3 - 2.0 * amount2 + amount) * span * first_derivative
        + (-2.0 * amount3 + 3.0 * amount2) * second
        + (amount3 - amount2) * span * second_derivative
    )


def _profile_scale_component(axis: int) -> Callable[[_SweepSectionProfile], float]:
    return lambda profile: profile.scale[axis]


def _profile_offset_component(axis: int) -> Callable[[_SweepSectionProfile], float]:
    return lambda profile: profile.offset[axis]


def _catmull_rom_sweep_scalar(
    profiles: Sequence[_SweepSectionProfile],
    value_at: Callable[[_SweepSectionProfile], float],
    start_index: int,
    amount: float,
    *,
    path_closed: bool,
    cyclic_delta: float = 0.0,
) -> float:
    start = profiles[start_index]
    end = profiles[start_index + 1]
    return _cubic_hermite_value(
        value_at(profiles[start_index]),
        value_at(profiles[start_index + 1]),
        _sweep_scalar_derivative(
            profiles,
            value_at,
            start_index,
            path_closed=path_closed,
            tension=start.tension,
            cyclic_delta=cyclic_delta,
        ),
        _sweep_scalar_derivative(
            profiles,
            value_at,
            start_index + 1,
            path_closed=path_closed,
            tension=start.tension,
            cyclic_delta=cyclic_delta,
        ),
        span=end.position - start.position,
        amount=amount,
    )


def _catmull_rom_sweep_profile(
    profiles: Sequence[_SweepSectionProfile],
    start_index: int,
    amount: float,
    *,
    path_closed: bool,
) -> List[Vec2]:
    start = profiles[start_index]
    end = profiles[start_index + 1]
    start_derivative = _sweep_profile_derivative(
        profiles,
        start_index,
        path_closed=path_closed,
        tension=start.tension,
    )
    end_derivative = _sweep_profile_derivative(
        profiles,
        start_index + 1,
        path_closed=path_closed,
        tension=start.tension,
    )
    amount2 = amount * amount
    amount3 = amount2 * amount
    h00 = 2.0 * amount3 - 3.0 * amount2 + 1.0
    h10 = amount3 - 2.0 * amount2 + amount
    h01 = -2.0 * amount3 + 3.0 * amount2
    h11 = amount3 - amount2
    span = end.position - start.position
    return [
        (
            h00 * first[0]
            + h10 * span * first_derivative[0]
            + h01 * second[0]
            + h11 * span * second_derivative[0],
            h00 * first[1]
            + h10 * span * first_derivative[1]
            + h01 * second[1]
            + h11 * span * second_derivative[1],
        )
        for first, second, first_derivative, second_derivative in zip(
            start.points,
            end.points,
            start_derivative,
            end_derivative,
        )
    ]


def _sweep_profile_at_position(
    profiles: Sequence[_SweepSectionProfile],
    position: float,
    *,
    closed: bool,
    path_closed: bool,
) -> List[Vec2]:
    raw_points = list(profiles[-1].points)
    scale: Tuple[float, ...] = profiles[-1].scale
    rotation = profiles[-1].rotation
    offset: Tuple[float, ...] = profiles[-1].offset
    end_index = bisect_left(
        profiles,
        position - _EPS,
        key=lambda profile: profile.position,
    )
    index = max(0, min(len(profiles) - 2, end_index - 1))
    start = profiles[index]
    end = profiles[index + 1]
    span = end.position - start.position
    amount = 0.0 if span <= _EPS else (position - start.position) / span
    amount = _clamp(amount, 0.0, 1.0)
    if start.interpolation == "catmull_rom":
        raw_points = _catmull_rom_sweep_profile(
            profiles,
            index,
            amount,
            path_closed=path_closed,
        )
        scale = tuple(
            _catmull_rom_sweep_scalar(
                profiles,
                _profile_scale_component(axis),
                index,
                amount,
                path_closed=path_closed,
            )
            for axis in range(2)
        )
        rotation_delta = profiles[-1].rotation - profiles[0].rotation
        rotation = _catmull_rom_sweep_scalar(
            profiles,
            lambda profile: profile.rotation,
            index,
            amount,
            path_closed=path_closed,
            cyclic_delta=(rotation_delta if path_closed else 0.0),
        )
        offset = tuple(
            _catmull_rom_sweep_scalar(
                profiles,
                _profile_offset_component(axis),
                index,
                amount,
                path_closed=path_closed,
            )
            for axis in range(2)
        )
    else:
        raw_points = [
            (
                first[0] + (second[0] - first[0]) * amount,
                first[1] + (second[1] - first[1]) * amount,
            )
            for first, second in zip(start.points, end.points)
        ]
        scale = (
            start.scale[0] + (end.scale[0] - start.scale[0]) * amount,
            start.scale[1] + (end.scale[1] - start.scale[1]) * amount,
        )
        rotation = start.rotation + (end.rotation - start.rotation) * amount
        offset = (
            start.offset[0] + (end.offset[0] - start.offset[0]) * amount,
            start.offset[1] + (end.offset[1] - start.offset[1]) * amount,
        )

    if not all(isfinite(component) and component > 0.0 for component in scale):
        raise ValueError(
            "Interpolated sweep scale became non-finite or non-positive; "
            "increase Catmull-Rom tension or adjust section scales"
        )
    if not isfinite(rotation) or not all(isfinite(component) for component in offset):
        raise ValueError("Interpolated sweep rotation and offset must remain finite")
    result = _transform_sweep_profile(
        raw_points,
        scale=(float(scale[0]), float(scale[1])),
        rotation=rotation,
        offset=(float(offset[0]), float(offset[1])),
    )
    if closed:
        area = _polygon_area(result)
        if not isfinite(area) or area <= 0.0:
            raise ValueError(
                "Interpolated sweep profile collapsed or reversed; "
                "adjust section profiles or tension"
            )
    return result
