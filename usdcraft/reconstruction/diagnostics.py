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

"""Authoring-only AXIAL-to-rebuild geometry diagnostics.

The diagnostic in this module consumes exactly one serialized ``AXIAL`` Agent Core and
one candidate USD stage.  It re-encodes the candidate on the grids declared by the Agent Core and
compares only those two inputs.  Source geometry, normalization artifacts, renders, semantic
images, and hidden evaluator output are deliberately outside the interface.

Both source and candidate are rasterized as sampled surfaces; closed candidate primitives are
never filled for comparison.  The result is repair feedback, not a semantic score, canonical-record
validator, or Host promotion gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from usdcraft.reconstruction.axial import (
    CELL_GLYPHS,
    MICRO_LONGEST_AXIS_CELLS,
    GridFrame,
    _build_micro_evidence,
    _quantize_volume,
    _scene_components,
    _stack_layout,
)
from usdcraft.reconstruction.source import SourceInput, load_normalized_source

SCHEMA = "axial_reencode_diagnostic"

_MICRO_GRID_PATTERN = re.compile(
    r"^micro_grid192: dims_xyz=(?P<dimensions>\S+) "
    r"pitch=(?P<pitch>\S+) grid_min_xyz=(?P<minimum>\S+)$"
)
_STACK_HEADER_PATTERN = re.compile(r"^\[SLICE_STACK(?P<resolution>\d+) (?P<fields>.+)\]$")
_RUN_PATTERN = re.compile(r"^run (?P<glyph>\S+): slices=(?P<start>\d+):(?P<end>\d+)$")
_PATTERN_HEADER = re.compile(r"^\[PATTERN (?P<glyph>\S+)\]$")
_VECTOR_LENGTH = 3
_GLYPH_TO_BITS = {glyph: bits for bits, glyph in CELL_GLYPHS.items()}


@dataclass(frozen=True)
class ParsedAxialStack:
    """The micro frame and selected positive-evidence stack decoded from Agent text."""

    micro_frame: GridFrame
    frame: GridFrame
    volume: np.ndarray
    stack_axis: int
    row_axis: int
    col_axis: int
    row_direction: str
    col_direction: str
    sequence_direction: str


def _integer_vector(value: str, *, length: int, field: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if len(pieces) != length:
        raise ValueError(f"{field} must contain {length} comma-separated integers")
    try:
        result = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"{field} must contain integers") from exc
    return result


def _float_vector(value: str, *, length: int, field: str) -> tuple[float, ...]:
    pieces = value.split(",")
    if len(pieces) != length:
        raise ValueError(f"{field} must contain {length} comma-separated numbers")
    try:
        result = tuple(float(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"{field} must contain numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite numbers")
    return result


def _fields(value: str, *, field: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in value.split():
        if token.count("=") != 1:
            raise ValueError(f"malformed {field} token: {token!r}")
        key, item = token.split("=", 1)
        if not key or not item or key in result:
            raise ValueError(f"malformed {field} field: {token!r}")
        result[key] = item
    return result


def _frame(
    *,
    dimensions: tuple[int, ...],
    pitch: float,
    minimum: tuple[float, ...],
    field: str,
) -> GridFrame:
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"{field} dimensions must be positive")
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError(f"{field} pitch must be finite and positive")
    dimensions_array = np.asarray(dimensions, dtype=np.int64)
    minimum_array = np.asarray(minimum, dtype=np.float64)
    return GridFrame(
        pitch=pitch,
        dimensions=dimensions_array,
        bounds_min=minimum_array,
        bounds_max=minimum_array + dimensions_array.astype(np.float64) * pitch,
    )


def _parse_micro_frame(lines: list[str]) -> GridFrame:
    matches = [match for line in lines if (match := _MICRO_GRID_PATTERN.fullmatch(line))]
    if len(matches) != 1:
        raise ValueError("AXIAL input must contain exactly one complete micro_grid192 line")
    match = matches[0]
    dimensions = _integer_vector(
        match.group("dimensions"), length=_VECTOR_LENGTH, field="micro_grid192 dims_xyz"
    )
    if max(dimensions) != MICRO_LONGEST_AXIS_CELLS:
        raise ValueError("micro_grid192 longest dimension must be exactly 192 cells")
    pitch = _float_vector(match.group("pitch"), length=1, field="micro_grid192 pitch")[0]
    minimum = _float_vector(
        match.group("minimum"), length=_VECTOR_LENGTH, field="micro_grid192 grid_min_xyz"
    )
    return _frame(dimensions=dimensions, pitch=pitch, minimum=minimum, field="micro_grid192")


def _parse_stack_header(lines: list[str]) -> tuple[int, GridFrame, int, int, int, str, str, str]:
    matches = [match for line in lines if (match := _STACK_HEADER_PATTERN.fullmatch(line))]
    if len(matches) != 1:
        raise ValueError("AXIAL input must contain exactly one selected SLICE_STACK section")
    match = matches[0]
    resolution = int(match.group("resolution"))
    fields = _fields(match.group("fields"), field="SLICE_STACK header")
    required = {
        "axis",
        "dims_xyz",
        "pitch",
        "grid_min_xyz",
        "slice_order",
        "rows",
        "cols",
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise ValueError("SLICE_STACK header is missing fields: " + ", ".join(missing))
    axis_name = fields["axis"]
    if axis_name not in "XYZ" or len(axis_name) != 1:
        raise ValueError(f"SLICE_STACK axis must be X, Y, or Z; got {axis_name!r}")
    stack_axis = "XYZ".index(axis_name)
    dimensions = _integer_vector(fields["dims_xyz"], length=3, field="SLICE_STACK dims_xyz")
    if max(dimensions) != resolution:
        raise ValueError(
            "SLICE_STACK header resolution does not match its longest dimension "
            f"({resolution} != {max(dimensions)})"
        )
    pitch = _float_vector(fields["pitch"], length=1, field="SLICE_STACK pitch")[0]
    minimum = _float_vector(fields["grid_min_xyz"], length=3, field="SLICE_STACK grid_min_xyz")
    frame = _frame(dimensions=dimensions, pitch=pitch, minimum=minimum, field="SLICE_STACK")
    row_axis, col_axis, row_direction, col_direction, sequence_direction = _stack_layout(stack_axis)
    declared_directions = (fields["rows"], fields["cols"], fields["slice_order"])
    expected_directions = (row_direction, col_direction, sequence_direction)
    if declared_directions != expected_directions:
        raise ValueError(
            "SLICE_STACK directions do not match its axis "
            f"(got {declared_directions}, expected {expected_directions})"
        )
    return (
        resolution,
        frame,
        stack_axis,
        row_axis,
        col_axis,
        row_direction,
        col_direction,
        sequence_direction,
    )


def _decode_pattern(
    raw_lines: list[str],
    *,
    row_count: int,
    col_count: int,
    glyph: str,
) -> np.ndarray:
    if len(raw_lines) != row_count:
        raise ValueError(f"PATTERN {glyph} must contain {row_count} rows; got {len(raw_lines)}")
    displayed = np.zeros((row_count, col_count), dtype=np.uint8)
    for row_index, line in enumerate(raw_lines):
        if len(line) != col_count:
            raise ValueError(
                f"PATTERN {glyph} row {row_index} must contain {col_count} cells; got {len(line)}"
            )
        try:
            displayed[row_index] = [_GLYPH_TO_BITS[cell] for cell in line]
        except KeyError as exc:
            raise ValueError(
                f"PATTERN {glyph} contains unsupported cell glyph {exc.args[0]!r}"
            ) from exc
    return np.flipud(displayed)


def _parse_patterns(lines: list[str], *, row_count: int, col_count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    index = 0
    while index < len(lines):
        match = _PATTERN_HEADER.fullmatch(lines[index])
        if match is None:
            index += 1
            continue
        glyph = match.group("glyph")
        if glyph in result:
            raise ValueError(f"duplicate PATTERN {glyph}")
        start = index + 1
        end = start + row_count
        if end > len(lines):
            raise ValueError(f"PATTERN {glyph} is truncated")
        result[glyph] = _decode_pattern(
            lines[start:end], row_count=row_count, col_count=col_count, glyph=glyph
        )
        index = end
    if not result:
        raise ValueError("SLICE_STACK contains no PATTERN sections")
    return result


def _parse_runs(lines: list[str], *, slice_count: int) -> tuple[str, ...]:
    sequence: list[str | None] = [None] * slice_count
    found = False
    for line in lines:
        match = _RUN_PATTERN.fullmatch(line)
        if match is None:
            continue
        found = True
        glyph = match.group("glyph")
        start = int(match.group("start"))
        end = int(match.group("end"))
        if not 0 <= start < end <= slice_count:
            raise ValueError(f"SLICE_STACK run is out of bounds: {line}")
        if any(sequence[index] is not None for index in range(start, end)):
            raise ValueError(f"SLICE_STACK runs overlap: {line}")
        sequence[start:end] = [glyph] * (end - start)
    if not found or any(glyph is None for glyph in sequence):
        raise ValueError("SLICE_STACK runs must cover every slice exactly once")
    return tuple(glyph for glyph in sequence if glyph is not None)


def parse_axial_stack(text: str) -> ParsedAxialStack:
    """Decode one complete selected evidence stack."""

    lines = text.splitlines()
    if not lines or lines[0] != "AXIAL":
        raise ValueError("re-encode diagnostic requires AXIAL text")
    if lines.count("profile: agent_core_surface") != 1:
        raise ValueError("AXIAL requires profile: agent_core_surface")
    micro_frame = _parse_micro_frame(lines)
    (
        _resolution,
        stack_frame,
        stack_axis,
        row_axis,
        col_axis,
        row_direction,
        col_direction,
        sequence_direction,
    ) = _parse_stack_header(lines)
    dimensions = tuple(int(value) for value in stack_frame.dimensions)
    patterns = _parse_patterns(
        lines,
        row_count=dimensions[row_axis],
        col_count=dimensions[col_axis],
    )
    sequence = _parse_runs(lines, slice_count=dimensions[stack_axis])
    unknown_patterns = sorted(set(sequence) - patterns.keys())
    if unknown_patterns:
        raise ValueError(
            "SLICE_STACK runs reference missing patterns: " + ", ".join(unknown_patterns)
        )

    volume = np.zeros(dimensions, dtype=np.uint8)
    remaining_axes = [axis for axis in range(3) if axis != stack_axis]
    displayed_axes = [row_axis, col_axis]
    transpose_axes = tuple(displayed_axes.index(axis) for axis in remaining_axes)
    for stack_index, glyph in enumerate(sequence):
        native_plane = np.transpose(patterns[glyph], transpose_axes)
        selector: list[int | slice] = [slice(None), slice(None), slice(None)]
        selector[stack_axis] = stack_index
        volume[tuple(selector)] = native_plane

    return ParsedAxialStack(
        micro_frame=micro_frame,
        frame=stack_frame,
        volume=volume,
        stack_axis=stack_axis,
        row_axis=row_axis,
        col_axis=col_axis,
        row_direction=row_direction,
        col_direction=col_direction,
        sequence_direction=sequence_direction,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _axis_occupied_span_m(mask: np.ndarray, frame: GridFrame, axis: int) -> list[float] | None:
    occupied = np.flatnonzero(np.any(mask, axis=tuple(a for a in range(3) if a != axis)))
    if occupied.size == 0:
        return None
    return [
        float(frame.bounds_min[axis] + occupied[0] * frame.pitch),
        float(frame.bounds_min[axis] + (occupied[-1] + 1) * frame.pitch),
    ]


def _envelope_comparison(
    source_positive: np.ndarray,
    candidate_positive: np.ndarray,
    frame: GridFrame,
) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for axis in range(3):
        input_span = _axis_occupied_span_m(source_positive, frame, axis)
        candidate_span = _axis_occupied_span_m(candidate_positive, frame, axis)
        entry: dict[str, Any] = {
            "input_span_m": input_span,
            "candidate_span_m": candidate_span,
        }
        if input_span is not None and candidate_span is not None:
            entry["candidate_short_low_m"] = max(0.0, candidate_span[0] - input_span[0])
            entry["candidate_short_high_m"] = max(0.0, input_span[1] - candidate_span[1])
        axes["XYZ"[axis]] = entry
    return {"pitch_m": frame.pitch, "axes": axes}


def _evidence_metrics(source: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    source_positive = source != 0
    candidate_positive = candidate != 0
    matched = source_positive & candidate_positive
    missing = source_positive & ~candidate_positive
    extra = candidate_positive & ~source_positive
    source_count = int(np.count_nonzero(source_positive))
    candidate_count = int(np.count_nonzero(candidate_positive))
    matched_count = int(np.count_nonzero(matched))
    missing_count = int(np.count_nonzero(missing))
    extra_count = int(np.count_nonzero(extra))
    channels: dict[str, dict[str, Any]] = {}
    for name, bit in (("sampled_source_surface", 2),):
        selected = (source & bit) != 0
        selected_count = int(np.count_nonzero(selected))
        channel_matched = int(np.count_nonzero(selected & candidate_positive))
        channels[name] = {
            "input_cell_count": selected_count,
            "matched_cell_count": channel_matched,
            "missing_cell_count": selected_count - channel_matched,
            "recall": _ratio(channel_matched, selected_count),
        }
    return {
        "input_cell_count": source_count,
        "candidate_cell_count": candidate_count,
        "matched_cell_count": matched_count,
        "missing_cell_count": missing_count,
        "recall": _ratio(matched_count, source_count),
        "complete": missing_count == 0,
        "channels": channels,
        "candidate_only_evidence": {
            "cell_count": extra_count,
            "fraction_of_candidate": _ratio(extra_count, candidate_count),
            "severity": "soft_warning" if extra_count else "none",
            "reason": (
                "Input '.' cells are unknown. Candidate-only sampled surfaces may be valid inner "
                "walls, image-supported parts, or necessary connections and never fail this check."
            ),
        },
        "evidence_mode": "surface_only",
    }


def _axis_shape(
    mask: np.ndarray,
    *,
    frame: GridFrame,
    plane_axes: tuple[int, ...],
    coordinate_axis: int,
) -> dict[str, float | int] | None:
    occupied = np.argwhere(mask)
    if len(occupied) == 0:
        return None
    local_axis = plane_axes.index(coordinate_axis)
    indices = occupied[:, local_axis]
    minimum_index = int(indices.min())
    maximum_index = int(indices.max())
    centers = frame.bounds_min[coordinate_axis] + (indices.astype(np.float64) + 0.5) * frame.pitch
    return {
        "cell_count": int(len(occupied)),
        "min_m": float(frame.bounds_min[coordinate_axis] + minimum_index * frame.pitch),
        "max_m": float(frame.bounds_min[coordinate_axis] + (maximum_index + 1) * frame.pitch),
        "centroid_m": float(centers.mean()),
    }


def _difference(
    source_shape: dict[str, float | int] | None,
    candidate_shape: dict[str, float | int] | None,
    *,
    pitch: float,
) -> dict[str, float] | None:
    if source_shape is None or candidate_shape is None:
        return None
    result: dict[str, float] = {}
    for name in ("min_m", "max_m", "centroid_m"):
        error = float(candidate_shape[name]) - float(source_shape[name])
        stem = name.removesuffix("_m")
        result[f"{stem}_error_m"] = error
        result[f"{stem}_error_cells"] = error / pitch
    return result


def _axis_semantics(axis: int) -> dict[str, str]:
    if axis == 0:
        return {"min": "-X", "max": "+X"}
    if axis == 1:
        return {"min": "-Y", "max": "+Y"}
    return {"min": "bottom(-Z)", "max": "top(+Z)"}


def _profile_hotspot_score(profile: dict[str, Any]) -> tuple[float, float]:
    recall = profile["positive_recall"]
    recall_gap = 1.0 if recall is None else 1.0 - float(recall)
    maximum_error = max(
        (
            abs(float(value))
            for axis_delta in profile["coordinate_deltas"].values()
            if axis_delta is not None
            for key, value in axis_delta.items()
            if key.endswith("_error_cells")
        ),
        default=0.0,
    )
    return recall_gap, maximum_error


def _slice_axis_metrics(
    source_positive: np.ndarray,
    candidate_positive: np.ndarray,
    *,
    frame: GridFrame,
    slice_axis: int,
) -> dict[str, Any]:
    plane_axes = tuple(axis for axis in range(3) if axis != slice_axis)
    slice_count = int(frame.dimensions[slice_axis])
    profiles: list[dict[str, Any]] = []
    missing_slices: list[int] = []
    candidate_only_slices: list[int] = []
    for slice_index in range(slice_count):
        source_mask = np.take(source_positive, slice_index, axis=slice_axis)
        candidate_mask = np.take(candidate_positive, slice_index, axis=slice_axis)
        if not np.any(source_mask) and not np.any(candidate_mask):
            continue
        if np.any(source_mask) and not np.any(candidate_mask):
            missing_slices.append(slice_index)
        if not np.any(source_mask) and np.any(candidate_mask):
            candidate_only_slices.append(slice_index)
        input_count = int(np.count_nonzero(source_mask))
        matched_count = int(np.count_nonzero(source_mask & candidate_mask))
        missing_mask = source_mask & ~candidate_mask
        missing_plane_bounds_m: dict[str, list[float]] | None = None
        if np.any(missing_mask):
            missing_cells = np.argwhere(missing_mask)
            missing_plane_bounds_m = {}
            for position, coordinate_axis in enumerate(plane_axes):
                low_index = int(missing_cells[:, position].min())
                high_index = int(missing_cells[:, position].max())
                missing_plane_bounds_m["XYZ"[coordinate_axis]] = [
                    float(frame.bounds_min[coordinate_axis] + low_index * frame.pitch),
                    float(frame.bounds_min[coordinate_axis] + (high_index + 1) * frame.pitch),
                ]
        input_shapes: dict[str, Any] = {}
        candidate_shapes: dict[str, Any] = {}
        deltas: dict[str, Any] = {}
        for coordinate_axis in plane_axes:
            axis_name = "XYZ"[coordinate_axis]
            input_shape = _axis_shape(
                source_mask,
                frame=frame,
                plane_axes=plane_axes,
                coordinate_axis=coordinate_axis,
            )
            candidate_shape = _axis_shape(
                candidate_mask,
                frame=frame,
                plane_axes=plane_axes,
                coordinate_axis=coordinate_axis,
            )
            input_shapes[axis_name] = input_shape
            candidate_shapes[axis_name] = candidate_shape
            deltas[axis_name] = _difference(input_shape, candidate_shape, pitch=frame.pitch)
        profiles.append(
            {
                "slice_index": slice_index,
                "slice_range_m": [
                    float(frame.bounds_min[slice_axis] + slice_index * frame.pitch),
                    float(frame.bounds_min[slice_axis] + (slice_index + 1) * frame.pitch),
                ],
                "positive_recall": _ratio(matched_count, input_count),
                "input_evidence_cell_count": input_count,
                "missing_plane_bounds_m": missing_plane_bounds_m,
                "extra_cell_count": int(np.count_nonzero(candidate_mask & ~source_mask)),
                "input": input_shapes,
                "candidate": candidate_shapes,
                "coordinate_deltas": deltas,
            }
        )

    coordinate_summaries: dict[str, Any] = {}
    for coordinate_axis in plane_axes:
        axis_name = "XYZ"[coordinate_axis]
        comparable = [
            profile["coordinate_deltas"][axis_name]
            for profile in profiles
            if profile["coordinate_deltas"][axis_name] is not None
        ]
        summary: dict[str, Any] = {"semantics": _axis_semantics(coordinate_axis)}
        for stem in ("min", "max", "centroid"):
            for unit in ("m", "cells"):
                key = f"{stem}_error_{unit}"
                summary[f"max_abs_{key}"] = max(
                    (abs(float(item[key])) for item in comparable), default=0.0
                )
        coordinate_summaries[axis_name] = summary

    mismatches = [
        profile
        for profile in profiles
        if profile["input_evidence_cell_count"] > 0
        and _profile_hotspot_score(profile) != (0.0, 0.0)
    ]
    hotspots = sorted(mismatches, key=_profile_hotspot_score, reverse=True)[:8]
    return {
        "slice_axis": "XYZ"[slice_axis],
        "slice_count": slice_count,
        "occupied_profile_count": len(profiles),
        "missing_input_slices": missing_slices,
        "candidate_only_slices": candidate_only_slices,
        "coordinate_summaries": coordinate_summaries,
        "hotspots": hotspots,
    }


def _slice_metrics(source: np.ndarray, candidate: np.ndarray, frame: GridFrame) -> dict[str, Any]:
    source_positive = source != 0
    candidate_positive = candidate != 0
    return {
        "pitch_m": frame.pitch,
        "axis_semantics": {
            "X": _axis_semantics(0),
            "Y": _axis_semantics(1),
            "Z": _axis_semantics(2),
        },
        "by_slice_axis": {
            "XYZ"[axis]: _slice_axis_metrics(
                source_positive,
                candidate_positive,
                frame=frame,
                slice_axis=axis,
            )
            for axis in range(3)
        },
    }


def check_axial_reencode(axial_path: Path, rebuild_usd: Path) -> dict[str, Any]:
    """Re-encode a candidate USD and compare it with serialized positive AXIAL evidence only."""

    resolved_axial = Path(axial_path).expanduser().resolve()
    if not resolved_axial.is_file():
        raise FileNotFoundError(resolved_axial)
    parsed = parse_axial_stack(resolved_axial.read_text(encoding="utf-8"))

    resolved_rebuild = Path(rebuild_usd).expanduser().resolve()
    if not resolved_rebuild.is_file():
        raise FileNotFoundError(resolved_rebuild)
    candidate_scene = load_normalized_source(SourceInput(resolved_rebuild))
    candidate_micro, classifications = _build_micro_evidence(
        _scene_components(candidate_scene), parsed.micro_frame
    )
    candidate_stack = _quantize_volume(candidate_micro, parsed.micro_frame, parsed.frame)

    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "host_gate": False,
        "evidence_mode": "surface_only",
        "inputs": {
            "axial": str(resolved_axial),
            "rebuild_usd": str(resolved_rebuild),
        },
        "grid": {
            "dimensions_xyz": [int(value) for value in parsed.frame.dimensions],
            "pitch_m": parsed.frame.pitch,
            "bounds_min_xyz_m": [float(value) for value in parsed.frame.bounds_min],
            "bounds_max_xyz_m": [float(value) for value in parsed.frame.bounds_max],
        },
        "candidate_evidence": {
            "component_count": len(classifications),
            "sampled_surface_component_count": sum(
                item.evidence_kind == "surface_hit_band" for item in classifications
            ),
            "interior_semantics": "unknown",
        },
        "evidence_comparison": _evidence_metrics(parsed.volume, candidate_stack),
        "envelope_comparison": _envelope_comparison(
            parsed.volume != 0,
            candidate_stack != 0,
            parsed.frame,
        ),
        "per_slice_profiles": _slice_metrics(parsed.volume, candidate_stack, parsed.frame),
        "limitations": [
            "Compares only the serialized lossy AXIAL stack with a candidate re-encoding; "
            "it never reads source geometry, ground truth, a render, or evaluator output.",
            "A missing sampled source-surface cell is objective serialized-"
            "evidence mismatch feedback, but this diagnostic sees no semantic image or declared "
            "mesh renders and cannot classify or approve a localized source-artifact exception.",
            "Ordinary missing support identifies a focused repair opportunity. "
            "The Author performs at most one AXIAL-driven repair pass, records the final residual, "
            "and does not treat nonzero recall as a completion or promotion blocker.",
            "Do not reintroduce a documented bounded artifact merely to maximize sampled-surface "
            "recall.",
            "Candidate-only surface evidence is a soft ambiguity because input '.' cells are "
            "unknown; inner walls, image-supported additions and connections may be valid.",
            "Closed candidate primitives are rasterized by boundary surface under the current "
            "contract; their SDK solid/collision semantics do not establish source interior.",
            "This is a static normalized-q0 Reconstruction authoring diagnostic, not a Host "
            "promotion gate, semantic approval, or simulator-runtime claim.",
        ],
    }


__all__ = [
    "ParsedAxialStack",
    "SCHEMA",
    "check_axial_reencode",
    "parse_axial_stack",
]
