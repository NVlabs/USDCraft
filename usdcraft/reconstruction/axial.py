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

"""Compact surface-evidence geometry observation for Reconstruction.

AXIAL records only sampled source surfaces.  Every reconstruction mesh has surface-only
authority: watertightness, consistent winding, ray parity, and a nonzero oriented surface integral
remain topology diagnostics and never establish material fill.  Every nonsurface location,
including a closed surface's bounded interior, is unknown rather than solid or free space.

The module intentionally consumes the small structural interface exposed by ``source.py`` instead
of depending on a particular source-file format.  This keeps normalization (axis, unit, transform,
triangulation, and component splitting) outside the observation contract.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence, cast

import numpy as np

from usdcraft.reconstruction.face_features import (
    FaceFeatureSummary,
    detect_face_features,
    format_face_features,
)
from usdcraft.reconstruction.face_openings import (
    FaceOpeningSummary,
    detect_face_openings,
    format_face_openings,
)
from usdcraft.reconstruction.interior_islands import (
    InteriorIslandSummary,
    detect_interior_islands,
    format_interior_islands,
)

if TYPE_CHECKING:
    from usdcraft.reconstruction.source import NormalizedSourceScene


MICRO_LONGEST_AXIS_CELLS = 192
# Observations use horizontal XY planes only.
STACK_AXIS = 2
STACK_RESOLUTION_CANDIDATES = (32, 24, 16)
STACK_BLOCK_CELLS = 4
# The Core's stack resolution is the finest of the candidates whose stack-bound core — header,
# MICRO islands, BODY_WALLS and the raster stack with its patterns — fits this ceiling. The
# face-derived sections (FACE_FEATURES, FACE_OPENINGS, MEMBER_TRACKS192, INTERIOR_ISLANDS) each
# carry their own caps and ride above it, so an extra button entry or a dropped opening never
# changes the stack a source selects.
# The ceiling is a runaway fuse, not a truncation trigger: presence evidence (island bboxes) is
# always serialized completely, so no candidate fitting means the scene is anomalous and the run
# fails loudly instead of silently dropping parts.
AGENT_CORE_BYTE_BUDGET = 32 * 1024
# The face-derived sections may add up to this much on top of the core ceiling (dense wire objects
# reach about 17 KB of tracks and features); the complete text is fused at the sum.
AGENT_CORE_SECTION_ALLOWANCE_BYTES = 24 * 1024
AGENT_CORE_TEXT_MAX_BYTES = AGENT_CORE_BYTE_BUDGET + AGENT_CORE_SECTION_ALLOWANCE_BYTES
# Keep candidate eligibility from becoming a smaller hidden limit than the complete serialized Core.
STACK_PALETTE_GLYPH_BUDGET = AGENT_CORE_BYTE_BUDGET
# Governs only the optional contour payload; island bboxes are presence evidence and are always
# serialized completely regardless of this budget.
MICRO_SLICE_BOUNDS_BYTE_BUDGET = 3072
# Provisional contour policy: emit a simplified sampled-support outline only where it materially
# differs from the island bbox, so rectangular islands keep their compact bbox-only byte layout.
MICRO_CONTOUR_MATERIALITY = 0.10
MICRO_CONTOUR_MAX_VERTICES = 12
MICRO_CONTOUR_MIN_BBOX_CELLS = 24
MICRO_CONTOUR_BASE_EPSILON_CELLS = 0.75
# Member tracks: per-micro-slab islands linked across Z into slender-member centerlines.
# Distance linking (bbox IoU is zero exactly for diagonal thin members), joint splitting at
# sharp centerline bends, and a slender-only selection cap; all provisional empirically derived values.
TRACK_MIN_SLABS = 8
TRACK_AREA_RATIO_MAX = 4.0
TRACK_LINK_SLOPE_CELLS = 2.5
TRACK_JOINT_WINDOW = 8
TRACK_JOINT_ANGLE_DEG = 40.0
TRACK_MAX_MEAN_WIDTH_CELLS = 64
TRACK_MAX_SHOWN = 16
PATTERN_GLYPHS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SURFACE_BIT = np.uint8(2)
CELL_GLYPHS = {0: ".", int(SURFACE_BIT): "~"}


class _NormalizedComponentLike(Protocol):
    @property
    def component_id(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def vertices(self) -> object: ...

    @property
    def faces(self) -> object: ...

    @property
    def topology(self) -> object: ...


class _NormalizedSceneLike(Protocol):
    @property
    def components(self) -> Sequence[_NormalizedComponentLike]: ...


@dataclass(frozen=True)
class GridFrame:
    pitch: float
    dimensions: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray


@dataclass(frozen=True)
class ViewSpec:
    name: str
    depth_axis: int
    depth_sign: int
    depth_direction: str
    row_axis: int
    col_axis: int
    row_direction: str
    col_direction: str


@dataclass(frozen=True)
class ComponentClassification:
    component_id: str
    source_kind: str
    evidence_kind: str
    reason: str
    vertex_count: int
    face_count: int
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    provenance_available: bool


@dataclass(frozen=True)
class SurfaceIslandBounds:
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    cell_count: int = 0
    contour: tuple[tuple[int, int], ...] = ()

    @property
    def bbox_cells(self) -> int:
        return (self.x_end - self.x_start) * (self.y_end - self.y_start)


@dataclass(frozen=True)
class MicroSliceBoundsRun:
    stack_z_start: int
    stack_z_end: int
    micro_z_start: int
    micro_z_end: int
    islands: tuple[SurfaceIslandBounds, ...]


@dataclass(frozen=True)
class MicroSliceBoundsSummary:
    stack_longest_axis_cells: int
    stack_slice_count: int
    encoded_slice_count: int
    island_count: int
    runs: tuple[MicroSliceBoundsRun, ...]
    section_bytes: int
    candidate_bytes: int
    selection: str = "complete"
    contoured_island_count: int = 0


@dataclass(frozen=True)
class BodyWallRow:
    """Per stack slab: robust wall position and bbox extreme on each XY side, in metres."""

    z0: float
    z1: float
    x_wall: tuple[float, float]
    x_extreme: tuple[float, float]
    y_wall: tuple[float, float]
    y_extreme: tuple[float, float]
    # share of the object's whole XY footprint that has sampled surface inside this slab
    cover: float = 1.0
    # True when the slab's outline is round or strongly rounded: the per-column extremes then
    # vary continuously and their median is not a wall (for a circle it sits at 0.866 R), so the
    # wall values are reported at the extremes and the row says `section=round`.
    round_section: bool = False


@dataclass(frozen=True)
class BodySupport:
    """How the sampled surface meets the ground: flat scanned bottom, or a contact patch."""

    contact_cover: float
    body_full_width_from_z: float
    verdict: str


@dataclass(frozen=True)
class BodyWallSummary:
    rows: tuple[BodyWallRow, ...]
    section_bytes: int
    support: BodySupport | None = None


SUPPORT_FLAT_COVER = 0.75
# A slab outline is round when, on both axes, its column widths follow an elliptical profile:
# mean/max width below this share (a circle gives pi/4 = 0.785, a box 1.0, a filleted box about
# 0.97) and the median deviation from the fitted ellipse's width profile below this share of the
# maximum width. Legs, wings and irregular outlines fail the profile test and keep their walls.
ROUND_SECTION_WIDTH_RATIO = 0.90
ROUND_SECTION_PROFILE_DEVIATION = 0.08
ROUND_SECTION_MIN_COLUMNS = 8
SUPPORT_CONTACT_COVER = 0.35
SUPPORT_FULL_WIDTH_RATIO = 0.90
# A slab wall is the median of its column extremes — unless an open door, lid or drawer drags it:
# the slab's extremes are spread (interquartile range above WALL_SPREAD_CELLS micro cells) with a
# long outward tail (at least WALL_TAIL_CELLS cells and WALL_TAIL_SHARE of the slab's own span
# along that axis beyond the face plane — a door or lid, not a handle, tray or step), the median sits outward of the plane the whole face ends on (the median
# of the column extremes over every column of that face, which must read as a plateau: at least
# WALL_FACE_DOMINANT of those columns within one cell, and WALL_FACE_PLATEAU times denser per cell
# than the columns spread outward of it — a real face with open parts in front, not a curved or
# many-member surface), and the slab still shows some columns (WALL_FACE_SHARE) on that plane.
# Then its wall is the face plane. Every other slab keeps the plain median.
WALL_SPREAD_CELLS = 3
WALL_TAIL_CELLS = 10
WALL_TAIL_SHARE = 0.35
WALL_FACE_DOMINANT = 0.08
WALL_FACE_PLATEAU = 3.0
WALL_FACE_SHARE = 0.02


@dataclass(frozen=True)
class MemberTrack:
    z_start: int
    z_end: int
    start_uv: tuple[float, float]
    end_uv: tuple[float, float]
    via_uv: tuple[tuple[float, float], ...]
    mean_width_cells: float
    slab_count: int


@dataclass(frozen=True)
class MemberTrackSummary:
    tracks: tuple[MemberTrack, ...]
    detected_track_count: int
    section_bytes: int


@dataclass(frozen=True)
class SliceStack:
    longest_axis_cells: int
    frame: GridFrame
    volume: np.ndarray
    stack_axis: int
    row_axis: int
    col_axis: int
    row_direction: str
    col_direction: str
    sequence_direction: str
    sequence: str
    patterns: tuple[tuple[str, tuple[str, ...]], ...]
    palette_grid_glyphs: int
    evidence_cell_count: int
    surface_cell_count: int
    candidate_costs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AxialObservation:
    source_bounds_min: np.ndarray
    source_bounds_max: np.ndarray
    semantic_front_resolved: bool
    micro_frame: GridFrame
    micro_volume: np.ndarray
    classifications: tuple[ComponentClassification, ...]
    micro_slice_bounds: MicroSliceBoundsSummary
    body_walls: BodyWallSummary
    face_features: FaceFeatureSummary
    face_openings: FaceOpeningSummary
    interior_islands: InteriorIslandSummary
    member_tracks: MemberTrackSummary
    stack: SliceStack
    agent_core: str
    agent_core_byte_count: int
    selection_core_byte_budget: int
    selection_reason: str


VIEW_SPECS = (
    ViewSpec(
        name="FRONT",
        depth_axis=1,
        depth_sign=1,
        depth_direction="-Y->+Y",
        row_axis=2,
        col_axis=0,
        row_direction="+Z->-Z",
        col_direction="-X->+X",
    ),
    ViewSpec(
        name="RIGHT",
        depth_axis=0,
        depth_sign=-1,
        depth_direction="+X->-X",
        row_axis=2,
        col_axis=1,
        row_direction="+Z->-Z",
        col_direction="-Y->+Y",
    ),
    ViewSpec(
        name="TOP",
        depth_axis=2,
        depth_sign=-1,
        depth_direction="+Z->-Z",
        row_axis=1,
        col_axis=0,
        row_direction="+Y->-Y",
        col_direction="-X->+X",
    ),
)


@dataclass(frozen=True)
class _ComponentArrays:
    component_id: str
    kind: str
    vertices: np.ndarray
    faces: np.ndarray
    topology: object
    provenance_available: bool
    geometry_authority: str
    interior_semantics: str
    solid_authority: str


def _format_vector(values: Sequence[float] | np.ndarray) -> str:
    return ",".join(f"{float(value):.9g}" for value in values)


def _component_kind(value: object) -> str:
    raw = getattr(value, "value", value)
    text = str(raw).strip().lower()
    if text.endswith(".mesh"):
        text = "mesh"
    elif text.endswith(".points") or text.endswith(".point_cloud"):
        text = "points"
    if text not in {"mesh", "points"}:
        raise ValueError(f"unsupported normalized component kind: {raw!r}")
    return text


def _as_component_arrays(component: _NormalizedComponentLike, index: int) -> _ComponentArrays:
    component_id = str(getattr(component, "component_id", f"component_{index:04d}"))
    kind = _component_kind(getattr(component, "kind", "mesh"))
    vertices = np.asarray(getattr(component, "vertices"), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"{component_id}: vertices must be a non-empty Nx3 array")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{component_id}: vertices contain a non-finite value")

    raw_faces = getattr(component, "faces", ())
    if raw_faces is None:
        raw_faces = ()
    faces = np.asarray(raw_faces, dtype=np.int64)
    if faces.size == 0:
        faces = np.empty((0, 3), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{component_id}: normalized mesh faces must be an Nx3 array")
    if kind == "points" and len(faces):
        raise ValueError(f"{component_id}: point component must not contain faces")
    if len(faces) and (np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError(f"{component_id}: face vertex index is out of range")
    raw_provenance = getattr(component, "provenance", None)
    provenance_available = raw_provenance is not None and bool(str(raw_provenance).strip())
    geometry_authority = str(getattr(component, "geometry_authority", "surface_only"))
    interior_semantics = str(getattr(component, "interior_semantics", "unknown"))
    solid_authority = str(getattr(component, "solid_authority", "none"))
    if (
        geometry_authority != "surface_only"
        or interior_semantics != "unknown"
        or solid_authority != "none"
    ):
        raise ValueError(
            f"{component_id}: current Reconstruction inputs must declare surface_only geometry, "
            "unknown interior, and no solid authority"
        )
    return _ComponentArrays(
        component_id=component_id,
        kind=kind,
        vertices=vertices,
        faces=faces,
        topology=getattr(component, "topology", None),
        provenance_available=provenance_available,
        geometry_authority=geometry_authority,
        interior_semantics=interior_semantics,
        solid_authority=solid_authority,
    )


def _scene_components(scene: _NormalizedSceneLike) -> tuple[_ComponentArrays, ...]:
    raw_components = tuple(getattr(scene, "components", ()))
    if not raw_components:
        raise ValueError("normalized source scene contains no components")
    components = tuple(
        _as_component_arrays(component, index) for index, component in enumerate(raw_components)
    )
    ids = [component.component_id for component in components]
    if len(ids) != len(set(ids)):
        raise ValueError("normalized component_id values must be unique")
    return components


def _validate_normalized_scene_metadata(scene: object) -> None:
    up_axis = getattr(scene, "up_axis", getattr(scene, "stage_up_axis", None))
    if up_axis is not None and str(getattr(up_axis, "value", up_axis)).upper() != "Z":
        raise ValueError("AXIAL requires a normalized Z-up source scene")
    meters_per_unit = getattr(scene, "meters_per_unit", None)
    if meters_per_unit is not None and not math.isclose(
        float(meters_per_unit), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("AXIAL requires normalized metre coordinates")


def _semantic_front_resolved(scene: object) -> bool:
    """Return whether normalization received an explicit semantic-front axis.

    A resolved source front is always rotated onto canonical ``-Y`` by ``source.py``.  Without
    that declaration, the fixed FRONT view remains only a geometric projection label and must not
    be promoted into semantic orientation.
    """

    frame = getattr(scene, "frame", None)
    return frame is not None and getattr(frame, "front_axis", None) is not None


def build_grid_frame(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    *,
    longest_axis_cells: int,
    block_cells: int,
) -> GridFrame:
    if longest_axis_cells <= 0 or block_cells <= 0:
        raise ValueError("grid sizes must be positive")
    if longest_axis_cells % block_cells:
        raise ValueError("longest_axis_cells must be divisible by block_cells")
    extent = np.asarray(bounds_max, dtype=np.float64) - np.asarray(bounds_min, dtype=np.float64)
    maximum = float(extent.max())
    if not np.isfinite(extent).all() or maximum <= 0.0:
        raise ValueError("source bounds must be finite with a positive extent")
    pitch = maximum / float(longest_axis_cells)
    block_extent = pitch * block_cells
    dimensions = np.maximum(
        block_cells,
        block_cells * np.ceil(extent / block_extent - 1e-12).astype(np.int64),
    )
    center = (np.asarray(bounds_min) + np.asarray(bounds_max)) / 2.0
    padded_extent = dimensions.astype(np.float64) * pitch
    return GridFrame(
        pitch=pitch,
        dimensions=dimensions,
        bounds_min=center - padded_extent / 2.0,
        bounds_max=center + padded_extent / 2.0,
    )


def _merge_close_values(values: Sequence[float], epsilon: float) -> tuple[float, ...]:
    merged: list[float] = []
    for value in sorted(float(item) for item in values):
        if merged and value - merged[-1] <= epsilon:
            merged[-1] = (merged[-1] + value) / 2.0
        else:
            merged.append(value)
    return tuple(merged)


def _cast_mesh_hits(
    component: _ComponentArrays,
    frame: GridFrame,
    spec: ViewSpec,
) -> dict[tuple[int, int], tuple[float, ...]]:
    hits: dict[tuple[int, int], list[float]] = defaultdict(list)
    row_count = int(frame.dimensions[spec.row_axis])
    col_count = int(frame.dimensions[spec.col_axis])
    projected_area_epsilon = max(frame.pitch * frame.pitch * 1e-12, 1e-18)
    barycentric_epsilon = 1e-10

    # Projection, denominators and candidate cell ranges are precomputed for every triangle in
    # one vectorized pass; the loop below then touches only triangles that can hit a ray. The
    # per-cell arithmetic is unchanged from the original per-triangle formulation.
    triangles_all = component.vertices[component.faces]
    if len(triangles_all) == 0:
        return {}
    projected_all = triangles_all[:, :, [spec.col_axis, spec.row_axis]]
    denominators = (projected_all[:, 1, 1] - projected_all[:, 2, 1]) * (
        projected_all[:, 0, 0] - projected_all[:, 2, 0]
    ) + (projected_all[:, 2, 0] - projected_all[:, 1, 0]) * (
        projected_all[:, 0, 1] - projected_all[:, 2, 1]
    )
    col_origin = float(frame.bounds_min[spec.col_axis])
    row_origin = float(frame.bounds_min[spec.row_axis])
    first_cols = np.maximum(
        0,
        np.ceil(
            (projected_all[:, :, 0].min(axis=1) - col_origin) / frame.pitch - 0.5 - 1e-12
        ).astype(np.int64),
    )
    last_cols = np.minimum(
        col_count - 1,
        np.floor(
            (projected_all[:, :, 0].max(axis=1) - col_origin) / frame.pitch - 0.5 + 1e-12
        ).astype(np.int64),
    )
    first_rows = np.maximum(
        0,
        np.ceil(
            (projected_all[:, :, 1].min(axis=1) - row_origin) / frame.pitch - 0.5 - 1e-12
        ).astype(np.int64),
    )
    last_rows = np.minimum(
        row_count - 1,
        np.floor(
            (projected_all[:, :, 1].max(axis=1) - row_origin) / frame.pitch - 0.5 + 1e-12
        ).astype(np.int64),
    )
    candidates = (
        (np.abs(denominators) > projected_area_epsilon)
        & (first_cols <= last_cols)
        & (first_rows <= last_rows)
    )
    signed_depths_all = spec.depth_sign * triangles_all[:, :, spec.depth_axis]

    for index in np.flatnonzero(candidates):
        projected = projected_all[index]
        denominator = denominators[index]
        cols_to_sample = np.arange(first_cols[index], last_cols[index] + 1, dtype=np.int64)
        rows_to_sample = np.arange(first_rows[index], last_rows[index] + 1, dtype=np.int64)
        cols, rows = np.meshgrid(cols_to_sample, rows_to_sample)
        x = frame.bounds_min[spec.col_axis] + (cols + 0.5) * frame.pitch
        y = frame.bounds_min[spec.row_axis] + (rows + 0.5) * frame.pitch
        x0, y0 = projected[0]
        x1, y1 = projected[1]
        x2, y2 = projected[2]
        weight0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
        weight1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
        weight2 = 1.0 - weight0 - weight1
        inside = (
            (weight0 >= -barycentric_epsilon)
            & (weight1 >= -barycentric_epsilon)
            & (weight2 >= -barycentric_epsilon)
        )
        if not inside.any():
            continue
        signed_depths = signed_depths_all[index]
        depths = (
            weight0 * signed_depths[0] + weight1 * signed_depths[1] + weight2 * signed_depths[2]
        )
        for row, col, depth in zip(rows[inside], cols[inside], depths[inside]):
            hits[(int(row), int(col))].append(float(depth))

    extent = float(np.ptp(component.vertices, axis=0).max(initial=0.0))
    hit_epsilon = max(extent * 1e-9, 1e-12)
    return {ray: _merge_close_values(depths, hit_epsilon) for ray, depths in hits.items()}


def _world_cell_indices(points: np.ndarray, frame: GridFrame) -> np.ndarray:
    indices = np.floor((points - frame.bounds_min) / frame.pitch).astype(np.int64)
    return np.clip(indices, 0, frame.dimensions - 1)


def _mark_surface_samples(volume: np.ndarray, frame: GridFrame, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    indices = _world_cell_indices(np.asarray(points, dtype=np.float64), frame)
    volume[tuple(indices.T)] |= SURFACE_BIT


def _mark_surface_hits(
    volume: np.ndarray,
    frame: GridFrame,
    spec: ViewSpec,
    hits: Mapping[tuple[int, int], Sequence[float]],
) -> None:
    for (row, col), depths in hits.items():
        for signed_depth in depths:
            index = [0, 0, 0]
            index[spec.row_axis] = row
            index[spec.col_axis] = col
            world_depth = spec.depth_sign * float(signed_depth)
            index[spec.depth_axis] = int(
                np.clip(
                    math.floor((world_depth - frame.bounds_min[spec.depth_axis]) / frame.pitch),
                    0,
                    int(frame.dimensions[spec.depth_axis]) - 1,
                )
            )
            volume[tuple(index)] |= SURFACE_BIT


def _component_bounds(
    component: _ComponentArrays,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # Vertices are validated as a non-empty Nx3 array, so each bound has exactly three values.
    return (
        cast(
            "tuple[float, float, float]",
            tuple(float(value) for value in component.vertices.min(axis=0)),
        ),
        cast(
            "tuple[float, float, float]",
            tuple(float(value) for value in component.vertices.max(axis=0)),
        ),
    )


def _build_micro_evidence(
    components: Sequence[_ComponentArrays],
    frame: GridFrame,
) -> tuple[np.ndarray, tuple[ComponentClassification, ...]]:
    volume = np.zeros(tuple(int(value) for value in frame.dimensions), dtype=np.uint8)
    classifications: list[ComponentClassification] = []
    for component in components:
        bounds_min, bounds_max = _component_bounds(component)
        if component.kind == "points":
            _mark_surface_samples(volume, frame, component.vertices)
            classifications.append(
                ComponentClassification(
                    component_id=component.component_id,
                    source_kind="points",
                    evidence_kind="surface_hit_band",
                    reason="point_samples",
                    vertex_count=len(component.vertices),
                    face_count=0,
                    bounds_min_xyz=bounds_min,
                    bounds_max_xyz=bounds_max,
                    provenance_available=component.provenance_available,
                )
            )
            continue

        for spec in VIEW_SPECS:
            _mark_surface_hits(volume, frame, spec, _cast_mesh_hits(component, frame, spec))
        triangles = component.vertices[component.faces]
        sample_groups = [component.vertices]
        if len(triangles):
            sample_groups.append(triangles.mean(axis=1))
        _mark_surface_samples(volume, frame, np.concatenate(sample_groups, axis=0))
        classifications.append(
            ComponentClassification(
                component_id=component.component_id,
                source_kind="mesh",
                evidence_kind="surface_hit_band",
                reason="reconstruction_mesh_surface_only_interior_unknown",
                vertex_count=len(component.vertices),
                face_count=len(component.faces),
                bounds_min_xyz=bounds_min,
                bounds_max_xyz=bounds_max,
                provenance_available=component.provenance_available,
            )
        )
    classifications.sort(key=lambda item: item.component_id)
    return volume, tuple(classifications)


def _glyph_lines(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(CELL_GLYPHS[int(value)] for value in row) for row in grid)


def _island_boundary_ring(cells: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Trace one island's outer boundary on the cell-corner lattice, interior kept on the left.

    Corner-touching lobes of an 8-connected island stay on a single closed walk because the trace
    always takes the sharpest available right turn through a pinch vertex.
    """

    cell_set = frozenset(cells)
    outgoing: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for x, y in cells:
        if (x, y - 1) not in cell_set:
            outgoing[(x, y)].add((x + 1, y))
        if (x + 1, y) not in cell_set:
            outgoing[(x + 1, y)].add((x + 1, y + 1))
        if (x, y + 1) not in cell_set:
            outgoing[(x + 1, y + 1)].add((x, y + 1))
        if (x - 1, y) not in cell_set:
            outgoing[(x, y + 1)].add((x, y))
    start = min(outgoing)
    ring: list[tuple[int, int]] = [start]
    current = start
    # The lexicographically smallest boundary corner always has exactly one outgoing edge, +X.
    heading = (1, 0)
    while True:
        turn_preference = (
            (heading[1], -heading[0]),
            heading,
            (-heading[1], heading[0]),
            (-heading[0], -heading[1]),
        )
        step = next(
            direction
            for direction in turn_preference
            if (current[0] + direction[0], current[1] + direction[1]) in outgoing[current]
        )
        following = (current[0] + step[0], current[1] + step[1])
        outgoing[current].discard(following)
        if following == start:
            break
        ring.append(following)
        current = following
        heading = step
    collapsed: list[tuple[int, int]] = []
    count = len(ring)
    for index in range(count):
        previous_point = ring[index - 1]
        point = ring[index]
        next_point = ring[(index + 1) % count]
        cross = (point[0] - previous_point[0]) * (next_point[1] - point[1]) - (
            point[1] - previous_point[1]
        ) * (next_point[0] - point[0])
        if cross != 0:
            collapsed.append(point)
    if not collapsed:
        return ()
    pivot = collapsed.index(min(collapsed))
    return tuple(collapsed[pivot:] + collapsed[:pivot])


def _perpendicular_distance(
    point: tuple[int, int],
    segment_start: tuple[int, int],
    segment_end: tuple[int, int],
) -> float:
    delta_x = segment_end[0] - segment_start[0]
    delta_y = segment_end[1] - segment_start[1]
    length = math.hypot(delta_x, delta_y)
    if length == 0.0:
        return math.hypot(point[0] - segment_start[0], point[1] - segment_start[1])
    return (
        abs(delta_x * (segment_start[1] - point[1]) - (segment_start[0] - point[0]) * delta_y)
        / length
    )


def _simplify_closed_ring(
    points: Sequence[tuple[int, int]],
    *,
    max_vertices: int,
    epsilon: float,
) -> list[tuple[int, int]]:
    """Vertex-count-capped Ramer-Douglas-Peucker via deterministic farthest-point insertion.

    ``points`` is a closed polyline whose first and last entries repeat one anchor vertex.  The
    result keeps the most salient corners first, so a thin diagonal island still yields a usable
    sliver polygon instead of collapsing at a fixed tolerance.
    """

    if len(points) < 3:
        return list(points[:-1])
    keep = [0, len(points) - 1]
    while len(keep) - 1 < max_vertices:
        best_index = None
        best_distance = epsilon
        for position in range(len(keep) - 1):
            first, last = keep[position], keep[position + 1]
            for index in range(first + 1, last):
                distance = _perpendicular_distance(points[index], points[first], points[last])
                if distance > best_distance + 1e-12:
                    best_index = index
                    best_distance = distance
        if best_index is None:
            break
        keep.insert(next(i for i, kept in enumerate(keep) if kept > best_index), best_index)
    return [points[index] for index in keep[:-1]]


def _polygon_area(points: Sequence[tuple[int, int]]) -> float:
    area = 0.0
    count = len(points)
    for index, (x_start, y_start) in enumerate(points):
        x_end, y_end = points[(index + 1) % count]
        area += x_start * y_end - x_end * y_start
    return abs(area) / 2.0


def _island_contour(
    cells: Sequence[tuple[int, int]],
    *,
    bbox_cells: int,
) -> tuple[tuple[int, int], ...]:
    """Return a simplified material outer contour for one island, or () when the bbox suffices."""

    if bbox_cells < MICRO_CONTOUR_MIN_BBOX_CELLS:
        return ()
    ring = _island_boundary_ring(cells)
    if len(ring) < 3:
        return ()
    simplified = _simplify_closed_ring(
        (*ring, ring[0]),
        max_vertices=MICRO_CONTOUR_MAX_VERTICES,
        epsilon=MICRO_CONTOUR_BASE_EPSILON_CELLS,
    )
    if len(simplified) < 3:
        return ()
    if bbox_cells - _polygon_area(simplified) < MICRO_CONTOUR_MATERIALITY * bbox_cells:
        return ()
    return tuple(simplified)


def _surface_island_bounds(projection: np.ndarray) -> tuple[SurfaceIslandBounds, ...]:
    """Return deterministic 8-connected XY bounds for one sampled-surface projection."""

    selected = np.asarray(projection, dtype=bool)
    if selected.ndim != 2:
        raise ValueError("surface-island projection must be a 2-D array")
    unvisited = selected.copy()
    islands: list[SurfaceIslandBounds] = []
    row_count, col_count = (int(value) for value in selected.shape)
    for seed_array in np.argwhere(unvisited):
        seed = tuple(int(value) for value in seed_array)
        if not unvisited[seed]:
            continue
        pending = deque((seed,))
        unvisited[seed] = False
        cells: list[tuple[int, int]] = []
        while pending:
            x_index, y_index = pending.popleft()
            cells.append((x_index, y_index))
            for x_delta in (-1, 0, 1):
                for y_delta in (-1, 0, 1):
                    if x_delta == 0 and y_delta == 0:
                        continue
                    candidate = (x_index + x_delta, y_index + y_delta)
                    if (
                        0 <= candidate[0] < row_count
                        and 0 <= candidate[1] < col_count
                        and unvisited[candidate]
                    ):
                        unvisited[candidate] = False
                        pending.append(candidate)
        indices = np.asarray(cells, dtype=np.int64)
        x_start = int(indices[:, 0].min())
        x_end = int(indices[:, 0].max()) + 1
        y_start = int(indices[:, 1].min())
        y_end = int(indices[:, 1].max()) + 1
        bbox_cells = (x_end - x_start) * (y_end - y_start)
        islands.append(
            SurfaceIslandBounds(
                x_start=x_start,
                x_end=x_end,
                y_start=y_start,
                y_end=y_end,
                cell_count=len(cells),
                contour=_island_contour(cells, bbox_cells=bbox_cells),
            )
        )
    return tuple(
        sorted(
            islands,
            key=lambda item: (
                item.x_start,
                item.x_end,
                item.y_start,
                item.y_end,
                item.cell_count,
                item.contour,
            ),
        )
    )


def _micro_slice_bounds_runs(
    micro_volume: np.ndarray,
    micro_frame: GridFrame,
    stack: SliceStack,
) -> tuple[tuple[MicroSliceBoundsRun, ...], int, int]:
    """Bind high-resolution XY surface islands to the selected fixed-Z stack slices."""

    if micro_volume.ndim != 3 or stack.stack_axis != STACK_AXIS:
        raise ValueError("micro slice bounds require a 3-D volume and a fixed-Z stack")
    surface_indices = np.argwhere((micro_volume & SURFACE_BIT) != 0)
    if len(surface_indices) == 0:
        return (), 0, 0
    centers = micro_frame.bounds_min + (surface_indices + 0.5) * micro_frame.pitch
    stack_indices = _world_cell_indices(centers, stack.frame)
    runs: list[MicroSliceBoundsRun] = []
    encoded_slice_count = 0
    island_count = 0
    for stack_z in range(int(stack.frame.dimensions[STACK_AXIS])):
        selected = surface_indices[stack_indices[:, STACK_AXIS] == stack_z]
        if len(selected) == 0:
            continue
        projection = np.zeros(
            tuple(int(value) for value in micro_frame.dimensions[:2]),
            dtype=bool,
        )
        projection[selected[:, 0], selected[:, 1]] = True
        islands = _surface_island_bounds(projection)
        if not islands:
            continue
        encoded_slice_count += 1
        island_count += len(islands)
        micro_z_start = int(selected[:, STACK_AXIS].min())
        micro_z_end = int(selected[:, STACK_AXIS].max()) + 1
        runs.append(
            MicroSliceBoundsRun(
                stack_z_start=stack_z,
                stack_z_end=stack_z + 1,
                micro_z_start=micro_z_start,
                micro_z_end=micro_z_end,
                islands=islands,
            )
        )
    # Runs stay per-slab here; each budget tier merges its own emitted form so that dropping
    # contours restores the bbox-equality merge exactly.
    return tuple(runs), encoded_slice_count, island_count


def _merge_micro_runs(
    runs: Sequence[MicroSliceBoundsRun],
) -> tuple[MicroSliceBoundsRun, ...]:
    """Merge adjacent slabs whose serialized island tokens and omitted counts are identical."""

    merged: list[MicroSliceBoundsRun] = []
    for run in runs:
        if (
            merged
            and merged[-1].stack_z_end == run.stack_z_start
            and tuple(_island_token(island) for island in merged[-1].islands)
            == tuple(_island_token(island) for island in run.islands)
        ):
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                stack_z_end=run.stack_z_end,
                micro_z_start=min(previous.micro_z_start, run.micro_z_start),
                micro_z_end=max(previous.micro_z_end, run.micro_z_end),
            )
            continue
        merged.append(run)
    return tuple(merged)


def _fill_ratio_text(cell_count: int, bbox_cells: int) -> str:
    text = f"{cell_count / bbox_cells:.2f}"
    return text[1:] if text.startswith("0") else text


def _island_token(island: SurfaceIslandBounds) -> str:
    token = f"{island.x_start}:{island.x_end},{island.y_start}:{island.y_end}"
    if island.contour:
        fill = _fill_ratio_text(island.cell_count, island.bbox_cells)
        vertices = " ".join(f"{x},{y}" for x, y in island.contour)
        token += f"(f={fill};p={vertices})"
    return token


def _format_micro_slice_bounds(
    *,
    stack_longest_axis_cells: int,
    stack_slice_count: int,
    runs: Sequence[MicroSliceBoundsRun],
    selection: str,
) -> tuple[str, ...]:
    contoured = sum(1 for run in runs for island in run.islands if island.contour)
    # A contour-free complete or boxes-only section keeps the compact bbox-only byte layout so
    # borderline dense cases do not lose budget headroom to unused header tokens.
    if contoured == 0 and selection in {"complete", "boxes_only"}:
        header = (
            f"[MICRO_SLICE_BOUNDS_Z192 parent=SLICE_STACK{stack_longest_axis_cells} "
            f"axis=Z stack_slices={stack_slice_count} connectivity=8 ranges=half_open]"
        )
    else:
        header = (
            f"[MICRO_SLICE_BOUNDS_Z192 parent=SLICE_STACK{stack_longest_axis_cells} "
            f"axis=Z stack_slices={stack_slice_count} connectivity=8 ranges=half_open "
            f"contours=corner_lattice_rdp selection={selection}]"
        )
    lines = [
        header,
        (
            "semantics: per-Z-slice sampled-surface island XY AABBs at micro-grid resolution; "
            "islands=anonymous_nonsemantic_not_Parts; box_interiors=UNKNOWN"
        ),
    ]
    if contoured:
        lines.append(
            "contours: p=simplified_outer_boundary_of_sampled_support on cell-corner lattice "
            "(bbox x0:x1 spans corners x0..x1); may cut corners or bridge unsampled cells by "
            "~1 micro cell; bbox=exact_extent_authority; f=sampled_cells/bbox_cells; "
            "holes/interior=UNKNOWN; missing_p!=rectangle_proof"
        )
    lines.append(
        "projection: XY connectivity may merge surfaces across one selected Z slab; "
        "bounds may bridge unsampled cells; omitted_slice=UNKNOWN_not_free_space"
    )
    for run in runs:
        encoded = "|".join(_island_token(island) for island in run.islands)
        lines.append(
            f"s{run.stack_z_start}:{run.stack_z_end} "
            f"mz={run.micro_z_start}:{run.micro_z_end} xy={encoded}"
        )
    return tuple(lines)


def _strip_contours(
    runs: Sequence[MicroSliceBoundsRun],
    *,
    keep: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[MicroSliceBoundsRun, ...]:
    return tuple(
        replace(
            run,
            islands=tuple(
                island if (run_index, island_index) in keep else replace(island, contour=())
                for island_index, island in enumerate(run.islands)
            ),
        )
        for run_index, run in enumerate(runs)
    )


def _build_micro_slice_bounds_summary(
    micro_volume: np.ndarray,
    micro_frame: GridFrame,
    stack: SliceStack,
) -> MicroSliceBoundsSummary:
    runs, encoded_slice_count, island_count = _micro_slice_bounds_runs(
        micro_volume,
        micro_frame,
        stack,
    )
    stack_slice_count = int(stack.frame.dimensions[STACK_AXIS])
    budget = MICRO_SLICE_BOUNDS_BYTE_BUDGET

    def _serialized_bytes(
        emitted_runs: Sequence[MicroSliceBoundsRun],
        selection: str,
    ) -> int:
        lines = _format_micro_slice_bounds(
            stack_longest_axis_cells=stack.longest_axis_cells,
            stack_slice_count=stack_slice_count,
            runs=emitted_runs,
            selection=selection,
        )
        return len(("\n".join(lines) + "\n").encode("utf-8"))

    def _summary(
        emitted_runs: tuple[MicroSliceBoundsRun, ...],
        selection: str,
        section_bytes: int,
        candidate_bytes: int,
    ) -> MicroSliceBoundsSummary:
        return MicroSliceBoundsSummary(
            stack_longest_axis_cells=stack.longest_axis_cells,
            stack_slice_count=stack_slice_count,
            encoded_slice_count=encoded_slice_count,
            island_count=island_count,
            runs=emitted_runs,
            section_bytes=section_bytes,
            candidate_bytes=candidate_bytes,
            selection=selection,
            contoured_island_count=sum(
                1 for run in emitted_runs for island in run.islands if island.contour
            ),
        )

    # The budget governs only the contour payload: it is measured as the increment over the
    # boxes-only baseline, so a dense scene whose presence boxes alone exceed the number still
    # receives its full contour allowance on top.
    boxes_baseline = _serialized_bytes(_merge_micro_runs(_strip_contours(runs)), "boxes_only")
    merged_full = _merge_micro_runs(runs)
    candidate_bytes = _serialized_bytes(merged_full, "complete")
    if candidate_bytes - boxes_baseline <= budget:
        return _summary(merged_full, "complete", candidate_bytes, candidate_bytes)

    # Keep every bbox and degrade contours first, largest-bbox islands keeping theirs longest.
    material = sorted(
        (
            (run_index, island_index)
            for run_index, run in enumerate(runs)
            for island_index, island in enumerate(run.islands)
            if island.contour
        ),
        key=lambda key: (-runs[key[0]].islands[key[1]].bbox_cells, key[0], key[1]),
    )
    kept: frozenset[tuple[int, int]] = frozenset()
    for key in material:
        trial = kept | {key}
        trial_runs = _merge_micro_runs(_strip_contours(runs, keep=trial))
        if _serialized_bytes(trial_runs, "contours_partial") - boxes_baseline <= budget:
            kept = trial
    if kept:
        partial_runs = _merge_micro_runs(_strip_contours(runs, keep=kept))
        return _summary(
            partial_runs,
            "contours_partial",
            _serialized_bytes(partial_runs, "contours_partial"),
            candidate_bytes,
        )

    # Island bboxes are presence evidence: they are always serialized completely, whatever the
    # contour budget. Nothing below this line may drop an island.
    boxes_runs = _merge_micro_runs(_strip_contours(runs))
    boxes_bytes = _serialized_bytes(boxes_runs, "boxes_only")
    return _summary(boxes_runs, "boxes_only", boxes_bytes, candidate_bytes)


def _quantize_volume(
    micro_volume: np.ndarray,
    micro_frame: GridFrame,
    target_frame: GridFrame,
) -> np.ndarray:
    target = np.zeros(tuple(int(value) for value in target_frame.dimensions), dtype=np.uint8)
    source_indices = np.argwhere(micro_volume != 0)
    if len(source_indices) == 0:
        return target
    centers = micro_frame.bounds_min + (source_indices + 0.5) * micro_frame.pitch
    target_indices = _world_cell_indices(centers, target_frame)
    np.bitwise_or.at(target, tuple(target_indices.T), micro_volume[tuple(source_indices.T)])
    return target


def _stack_layout(axis: int) -> tuple[int, int, str, str, str]:
    if axis == 0:
        return 2, 1, "+Z->-Z", "-Y->+Y", "-X->+X"
    if axis == 1:
        return 2, 0, "+Z->-Z", "-X->+X", "-Y->+Y"
    if axis == 2:
        return 1, 0, "+Y->-Y", "-X->+X", "-Z->+Z"
    raise ValueError(f"invalid stack axis: {axis}")


def _slice_pattern(
    volume: np.ndarray,
    *,
    stack_axis: int,
    stack_index: int,
    row_axis: int,
    col_axis: int,
) -> tuple[str, ...]:
    plane = np.take(volume, stack_index, axis=stack_axis)
    remaining_axes = [axis for axis in range(3) if axis != stack_axis]
    displayed = np.transpose(
        plane,
        (remaining_axes.index(row_axis), remaining_axes.index(col_axis)),
    )
    return _glyph_lines(np.flipud(displayed))


def _best_stack(
    volume: np.ndarray,
    frame: GridFrame,
    *,
    longest_axis_cells: int,
) -> SliceStack | None:
    stack_axis = STACK_AXIS
    row_axis, col_axis, row_direction, col_direction, sequence_direction = _stack_layout(stack_axis)
    pattern_to_glyph: dict[tuple[str, ...], str] = {}
    patterns: list[tuple[str, tuple[str, ...]]] = []
    sequence: list[str] = []
    for stack_index in range(volume.shape[stack_axis]):
        pattern = _slice_pattern(
            volume,
            stack_axis=stack_axis,
            stack_index=stack_index,
            row_axis=row_axis,
            col_axis=col_axis,
        )
        glyph = pattern_to_glyph.get(pattern)
        if glyph is None:
            if len(patterns) >= len(PATTERN_GLYPHS):
                return None
            glyph = PATTERN_GLYPHS[len(patterns)]
            pattern_to_glyph[pattern] = glyph
            patterns.append((glyph, pattern))
        sequence.append(glyph)
    plane_cells = int(volume.shape[row_axis] * volume.shape[col_axis])
    return SliceStack(
        longest_axis_cells=longest_axis_cells,
        frame=frame,
        volume=volume,
        stack_axis=stack_axis,
        row_axis=row_axis,
        col_axis=col_axis,
        row_direction=row_direction,
        col_direction=col_direction,
        sequence_direction=sequence_direction,
        sequence="".join(sequence),
        patterns=tuple(patterns),
        palette_grid_glyphs=len(patterns) * plane_cells,
        evidence_cell_count=int(np.count_nonzero(volume)),
        surface_cell_count=int(np.count_nonzero(volume & SURFACE_BIT)),
        candidate_costs=(),
    )


def _sequence_runs(sequence: str) -> tuple[tuple[str, int, int], ...]:
    if not sequence:
        return ()
    result: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, len(sequence) + 1):
        if index == len(sequence) or sequence[index] != sequence[start]:
            result.append((sequence[start], start, index))
            start = index
    return tuple(result)


def _classification_counts(
    classifications: Sequence[ComponentClassification],
) -> tuple[int, int]:
    surface_meshes = sum(
        item.source_kind == "mesh" and item.evidence_kind == "surface_hit_band"
        for item in classifications
    )
    point_sets = sum(item.source_kind == "points" for item in classifications)
    return surface_meshes, point_sets


def _track_slab_islands(volume: np.ndarray) -> list[list[dict]]:
    """Per-micro-Z-slab 8-connected islands with centroid, bbox width and area."""

    from scipy import ndimage

    eight = np.ones((3, 3), dtype=int)
    per_slab: list[list[dict]] = []
    for z in range(volume.shape[2]):
        plane = volume[:, :, z]
        islands: list[dict] = []
        if plane.any():
            labels, count = ndimage.label(plane, structure=eight)
            for label_id in range(1, count + 1):
                xs, ys = np.where(labels == label_id)
                islands.append(
                    {
                        "centroid": (float(xs.mean()), float(ys.mean())),
                        "width": float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1),
                        "area": len(xs),
                    }
                )
        per_slab.append(islands)
    return per_slab


def _link_track_islands(per_slab: list[list[dict]]) -> list[list[tuple[int, dict]]]:
    """Greedy nearest-centroid linking across adjacent slabs; deterministic tie order."""

    finished: list[list[tuple[int, dict]]] = []
    active: list[list[tuple[int, dict]]] = []
    for z, islands in enumerate(per_slab):
        candidates = []
        for track_index, track in enumerate(active):
            previous = track[-1][1]
            for island_index, island in enumerate(islands):
                du = previous["centroid"][0] - island["centroid"][0]
                dv = previous["centroid"][1] - island["centroid"][1]
                distance = (du * du + dv * dv) ** 0.5
                gate = max(max(previous["area"], island["area"]) ** 0.5, TRACK_LINK_SLOPE_CELLS)
                ratio = max(previous["area"], island["area"]) / max(
                    1, min(previous["area"], island["area"])
                )
                if distance <= gate and ratio <= TRACK_AREA_RATIO_MAX:
                    candidates.append((distance, track_index, island_index))
        candidates.sort()
        used_tracks: set[int] = set()
        used_islands: set[int] = set()
        next_active: list[list[tuple[int, dict]]] = []
        for _, track_index, island_index in candidates:
            if track_index in used_tracks or island_index in used_islands:
                continue
            used_tracks.add(track_index)
            used_islands.add(island_index)
            active[track_index].append((z, islands[island_index]))
            next_active.append(active[track_index])
        for track_index, track in enumerate(active):
            if track_index not in used_tracks:
                finished.append(track)
        for island_index, island in enumerate(islands):
            if island_index not in used_islands:
                next_active.append([(z, island)])
        active = next_active
    finished.extend(active)
    return finished


def _split_track_at_joints(
    track: list[tuple[int, dict]],
) -> list[list[tuple[int, dict]]]:
    """Split one track at sharp centerline bends so an articulation never fuses two members."""

    if len(track) < 2 * TRACK_JOINT_WINDOW + TRACK_MIN_SLABS:
        return [track]
    points = np.asarray([(isl["centroid"][0], isl["centroid"][1], float(z)) for z, isl in track])
    cuts: list[int] = []
    index = TRACK_JOINT_WINDOW
    while index < len(track) - TRACK_JOINT_WINDOW:
        before = points[index] - points[index - TRACK_JOINT_WINDOW]
        after = points[index + TRACK_JOINT_WINDOW] - points[index]
        norms = np.linalg.norm(before) * np.linalg.norm(after)
        if norms > 0:
            cosine = float(np.dot(before, after)) / float(norms)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if angle > TRACK_JOINT_ANGLE_DEG:
                cuts.append(index)
                index += TRACK_MIN_SLABS
                continue
        index += 1
    if not cuts:
        return [track]
    segments: list[list[tuple[int, dict]]] = []
    start = 0
    for cut in cuts:
        if cut - start >= TRACK_MIN_SLABS:
            segments.append(track[start:cut])
        start = cut
    if len(track) - start >= TRACK_MIN_SLABS:
        segments.append(track[start:])
    return segments or [track]


def _build_member_track_summary(
    micro_volume: np.ndarray, micro_frame: GridFrame
) -> MemberTrackSummary:
    surface = (micro_volume & SURFACE_BIT) != 0
    raw = [
        track
        for track in _link_track_islands(_track_slab_islands(surface))
        if len(track) >= TRACK_MIN_SLABS
    ]
    segments = [
        segment
        for track in raw
        for segment in _split_track_at_joints(track)
        if len(segment) >= TRACK_MIN_SLABS
    ]
    slender = [
        segment
        for segment in segments
        if np.mean([isl["width"] for _, isl in segment]) <= TRACK_MAX_MEAN_WIDTH_CELLS
    ]
    detected = len(slender)
    # The stack already shows broad bodies; tracks exist for slender members, longest first.
    slender.sort(key=lambda seg: (-len(seg), seg[0][0], seg[0][1]["centroid"]))
    shown = slender[:TRACK_MAX_SHOWN]

    def uv(entry: tuple[int, dict]) -> tuple[float, float]:
        return entry[1]["centroid"]

    tracks: list[MemberTrack] = []
    for segment in shown:
        via: tuple[tuple[float, float], ...]
        if len(segment) >= 24:
            via = (uv(segment[len(segment) // 3]), uv(segment[2 * len(segment) // 3]))
        else:
            via = (uv(segment[len(segment) // 2]),)
        tracks.append(
            MemberTrack(
                z_start=segment[0][0],
                z_end=segment[-1][0] + 1,
                start_uv=uv(segment[0]),
                end_uv=uv(segment[-1]),
                via_uv=via,
                mean_width_cells=float(np.mean([isl["width"] for _, isl in segment])),
                slab_count=len(segment),
            )
        )
    tracks.sort(key=lambda t: (t.z_start, t.z_end, t.start_uv))
    summary = MemberTrackSummary(
        tracks=tuple(tracks), detected_track_count=detected, section_bytes=0
    )
    lines = _format_member_tracks(summary, micro_frame)
    byte_count = len(("\n".join(lines) + "\n").encode("utf-8")) if lines else 0
    return replace(summary, section_bytes=byte_count)


def _format_member_tracks(summary: MemberTrackSummary, frame: GridFrame) -> tuple[str, ...]:
    if not summary.tracks:
        return ()

    def world(uv: tuple[float, float]) -> tuple[float, float]:
        return (
            float(frame.bounds_min[0] + (uv[0] + 0.5) * frame.pitch),
            float(frame.bounds_min[1] + (uv[1] + 0.5) * frame.pitch),
        )

    lines = [
        (
            f"[MEMBER_TRACKS192 shown={len(summary.tracks)} "
            f"detected={summary.detected_track_count} min_slabs={TRACK_MIN_SLABS} "
            f"joint_split_deg={TRACK_JOINT_ANGLE_DEG:.0f}]"
        ),
        (
            "semantics: Host-computed deterministic cross-slice linking HYPOTHESIS for slender "
            "members — candidate sweep centerlines with mean cross-section width; coordinates in "
            "world metres, width in mm; NOT source-proven continuity, NOT Parts or joints; adopt "
            "a track only where renders/semantics cohere; absence_of_track!=absence_of_member"
        ),
    ]
    for track in summary.tracks:
        z0 = float(frame.bounds_min[2] + (track.z_start + 0.5) * frame.pitch)
        z1 = float(frame.bounds_min[2] + (track.z_end - 0.5) * frame.pitch)
        sx, sy = world(track.start_uv)
        ex, ey = world(track.end_uv)
        vias = " ".join(f"via({vx:+.3f},{vy:+.3f})" for vx, vy in (world(v) for v in track.via_uv))
        lines.append(
            f"t z[{z0:+.3f},{z1:+.3f}] c({sx:+.3f},{sy:+.3f})->({ex:+.3f},{ey:+.3f}) {vias} "
            f"w~{track.mean_width_cells * frame.pitch * 1000:.0f}mm slabs={track.slab_count}"
        )
    return tuple(lines)


def _build_body_walls(
    micro_volume: np.ndarray, micro_frame: GridFrame, runs: Sequence[MicroSliceBoundsRun]
) -> BodyWallSummary:
    """Robust wall positions per stack slab.

    A bbox extreme is carried by whatever cell sticks out furthest — a scan burr, a rim, a knob.
    The wall is where the face actually is: on each side, the median over the slab's columns of
    the outermost sampled-surface cell, made face-consistent against open parts — a slab whose
    extremes are spread by an open door, lid or drawer and whose median therefore lies outward of
    the whole face's plane, while some of its columns still end on that plane, takes the face
    plane (an open appliance door can cover most of a knob band's columns and drag the median
    centimetres in front of the fascia plane the knobs sit on). Both are serialized so the
    Author builds the body to the wall and reads the extreme as "something protrudes here".
    """

    surface = (micro_volume & SURFACE_BIT) != 0
    origin = np.asarray(micro_frame.bounds_min, dtype=float)
    pitch = float(micro_frame.pitch)
    footprint = int(surface.any(axis=2).sum()) or 1
    face_planes = _face_planes(surface)
    rows: list[BodyWallRow] = []
    for run in runs:
        slab = surface[:, :, run.micro_z_start : run.micro_z_end].any(axis=2)
        if not slab.any():
            continue
        xs, ys = np.nonzero(slab)
        cover = float(slab.sum()) / footprint
        sides: dict[str, tuple[float, float]] = {}
        width_ratios: list[float] = []
        for axis_name, along, across in (("x", xs, ys), ("y", ys, xs)):
            lo_vals: dict[int, int] = {}
            hi_vals: dict[int, int] = {}
            for a, c in zip(along.tolist(), across.tolist()):
                if a < lo_vals.get(c, 1 << 30):
                    lo_vals[c] = a
                if a > hi_vals.get(c, -1):
                    hi_vals[c] = a
            lo = np.array(list(lo_vals.values()), dtype=float)
            hi = np.array(list(hi_vals.values()), dtype=float)
            columns = np.array(list(lo_vals.keys()), dtype=float)
            width_ratios.append(_elliptical_profile(columns, hi - lo + 1.0))
            axis = 0 if axis_name == "x" else 1
            plane_lo, plane_hi = face_planes[axis]
            span = float(hi.max() - lo.min()) + 1.0
            sides[axis_name + "_wall"] = (
                origin[axis] + _face_consistent_wall(lo, plane_lo, -1, span) * pitch,
                origin[axis] + (_face_consistent_wall(hi, plane_hi, +1, span) + 1.0) * pitch,
            )
            sides[axis_name + "_extreme"] = (
                origin[axis] + float(lo.min()) * pitch,
                origin[axis] + (float(hi.max()) + 1.0) * pitch,
            )
        # A round or strongly rounded outline has no straight wall on either axis: its column
        # widths shrink continuously towards the ends, so the median extreme under-reads the
        # diameter (0.866 R for a circle). Report the extremes as the walls and flag the slab.
        round_section = all(width_ratios)
        if round_section:
            sides["x_wall"] = sides["x_extreme"]
            sides["y_wall"] = sides["y_extreme"]
        rows.append(
            BodyWallRow(
                z0=origin[2] + run.micro_z_start * pitch,
                z1=origin[2] + run.micro_z_end * pitch,
                x_wall=sides["x_wall"],
                x_extreme=sides["x_extreme"],
                y_wall=sides["y_wall"],
                y_extreme=sides["y_extreme"],
                cover=cover,
                round_section=round_section,
            )
        )
    support = _body_support(rows)
    summary = BodyWallSummary(tuple(rows), 0, support)
    lines = _format_body_walls(summary) if rows else ()
    return BodyWallSummary(
        tuple(rows), sum(len(line.encode("utf-8")) + 1 for line in lines), support
    )


def _face_planes(surface: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Per XY axis, the (low, high) face plane in micro cells: the median over every column of
    the whole sampled surface of its outermost cell on that side, or NaN when fewer than
    ``WALL_FACE_DOMINANT`` of those columns end within one cell of it or that plane is not a
    plateau against the columns spread outward of it (no face there to hold a slab to)."""

    planes: list[tuple[float, float]] = []
    for axis in (0, 1):
        occupied = surface.any(axis=axis)
        if not occupied.any():
            planes.append((float("nan"), float("nan")))
            continue
        first = np.argmax(surface, axis=axis)[occupied].astype(float)
        last = (surface.shape[axis] - 1 - np.argmax(np.flip(surface, axis=axis), axis=axis))[
            occupied
        ].astype(float)
        sides = []
        for values, outward in ((first, -1), (last, +1)):
            plane = float(np.median(values))
            on_plane = int(np.count_nonzero(np.abs(values - plane) <= 1.0))
            tail = values[outward * (values - plane) > 1.0]
            tail_bins = len(np.unique(np.rint(tail))) if len(tail) else 0
            plateau = on_plane / 3.0
            tail_density = len(tail) / tail_bins if tail_bins else 0.0
            dominant = on_plane >= WALL_FACE_DOMINANT * len(values)
            spike = tail_bins == 0 or plateau >= WALL_FACE_PLATEAU * tail_density
            sides.append(plane if dominant and spike else float("nan"))
        planes.append((sides[0], sides[1]))
    return planes[0], planes[1]


def _face_consistent_wall(
    values: np.ndarray, face_plane: float, outward: int, span: float
) -> float:
    """Median of a slab's column extremes, or the whole-face plane when an open part spreads the
    extremes with a long outward tail and drags the median outward of it while some columns
    still end on the plane."""

    median = float(np.median(values))
    if not np.isfinite(face_plane) or len(values) < 3:
        return median
    q25, q75 = np.percentile(values, (25.0, 75.0))
    if q75 - q25 <= WALL_SPREAD_CELLS:
        return median
    if outward * (median - face_plane) <= 2.0:
        return median
    tail = outward * (float(values.max() if outward > 0 else values.min()) - face_plane)
    if tail < WALL_TAIL_CELLS or tail < WALL_TAIL_SHARE * span:
        return median
    on_plane = int(np.count_nonzero(np.abs(values - face_plane) <= 1.0))
    if on_plane < max(3, WALL_FACE_SHARE * len(values)):
        return median
    return face_plane


def _elliptical_profile(columns: np.ndarray, widths: np.ndarray) -> bool:
    """True when the per-column widths follow an ellipse: a round section on this axis."""

    if len(columns) < ROUND_SECTION_MIN_COLUMNS or widths.max() <= 0:
        return False
    if widths.mean() / widths.max() >= ROUND_SECTION_WIDTH_RATIO:
        return False
    c0 = (columns.min() + columns.max()) / 2.0
    half_span = (columns.max() - columns.min() + 1.0) / 2.0
    normalized = np.clip((columns - c0) / half_span, -1.0, 1.0)
    predicted = widths.max() * np.sqrt(np.maximum(0.0, 1.0 - normalized * normalized))
    deviation = float(np.median(np.abs(widths - predicted))) / float(widths.max())
    return deviation < ROUND_SECTION_PROFILE_DEVIATION


def _body_support(rows: Sequence[BodyWallRow]) -> BodySupport | None:
    """Describe lowest-slab coverage and first broad section, not support identity/height."""

    if not rows:
        return None
    ordered = sorted(rows, key=lambda r: r.z0)
    contact = ordered[0].cover
    areas = [(r.x_wall[1] - r.x_wall[0]) * (r.y_wall[1] - r.y_wall[0]) for r in ordered]
    full = max(areas) if areas else 0.0
    body_from = ordered[0].z0
    for row, area in zip(ordered, areas):
        if full > 0 and area >= SUPPORT_FULL_WIDTH_RATIO * full:
            body_from = row.z0
            break
    if contact >= SUPPORT_FLAT_COVER:
        verdict = "flat_scanned_bottom"
        body_from = ordered[0].z0
    elif contact < SUPPORT_CONTACT_COVER:
        verdict = "contact_patch_or_unscanned_underside"
    else:
        verdict = "partial_bottom"
    return BodySupport(contact_cover=contact, body_full_width_from_z=body_from, verdict=verdict)


def support_reading(line: str) -> str:
    """Keep support statistics, including archived readings, separate from structural inference."""

    return (
        line.split(";", 1)[0]
        + "; coverage describes sampled surface only; body_full_width_from_z is the first broad "
        "section, not a measured body underside or foot height. The grounded source anchors the "
        "contact plane. Sparse coverage cannot distinguish rounded shell, missing underside or "
        "feet; unresolved underside construction remains an Author hypothesis."
    )


def _format_body_walls(summary: BodyWallSummary) -> tuple[str, ...]:
    lines = [
        f"[BODY_WALLS_Z slabs={len(summary.rows)} metres wall=median_of_column_extremes extreme=bbox]",
        (
            "semantics: per stack slab, where each XY face of the sampled surface actually is "
            "(wall) versus how far the furthest cell reaches (extreme); cover = share of the "
            "object's whole XY footprint that has surface in this slab. Build the body to the "
            "wall values; a gap between wall and extreme is a rim, burr, knob or other part "
            "standing proud of that face, never body width. A slab marked section=round has a "
            "round or strongly rounded outline with no straight wall: its wall values equal its "
            "extremes and name the diameter or the rounded body's full extent."
        ),
    ]
    if summary.support is not None:
        s = summary.support
        lines.append(
            support_reading(
                f"support: contact_cover={s.contact_cover:.0%} "
                f"body_full_width_from_z={s.body_full_width_from_z:+.3f} verdict={s.verdict}"
            )
        )
    for row in summary.rows:
        lines.append(
            f"z[{row.z0:+.3f},{row.z1:+.3f}] "
            f"x_wall[{row.x_wall[0]:+.3f},{row.x_wall[1]:+.3f}] "
            f"x_ext[{row.x_extreme[0]:+.3f},{row.x_extreme[1]:+.3f}] "
            f"y_wall[{row.y_wall[0]:+.3f},{row.y_wall[1]:+.3f}] "
            f"y_ext[{row.y_extreme[0]:+.3f},{row.y_extreme[1]:+.3f}] "
            f"cover={row.cover:.0%}" + (" section=round" if row.round_section else "")
        )
    return tuple(lines)


def _format_core(
    *,
    source_bounds_min: np.ndarray,
    source_bounds_max: np.ndarray,
    semantic_front_resolved: bool,
    micro_frame: GridFrame,
    classifications: Sequence[ComponentClassification],
    micro_slice_bounds: MicroSliceBoundsSummary,
    body_walls: BodyWallSummary,
    face_features: FaceFeatureSummary,
    face_openings: FaceOpeningSummary,
    interior_islands: InteriorIslandSummary,
    member_tracks: MemberTrackSummary,
    stack: SliceStack,
) -> str:
    extent = source_bounds_max - source_bounds_min
    surface_meshes, point_sets = _classification_counts(classifications)
    profile = "agent_core_surface"
    lines = [
        "AXIAL",
        f"profile: {profile}",
        "pose: normalized_q0",
        "frame: recon_world Z_up meters; isotropic; no PCA/per-view scaling",
        f"semantic_front: {'-Y' if semantic_front_resolved else 'unresolved'}",
        f"bounds_min_xyz: {_format_vector(source_bounds_min)}",
        f"bounds_max_xyz: {_format_vector(source_bounds_max)}",
        f"extent_xyz: {_format_vector(extent)}",
        "cells: .=unknown ~=sampled_source_surface",
        (
            "interior: UNKNOWN; closed/watertight surface topology does not imply solid material, "
            "fill, wall thickness, cavity contents, or free space"
        ),
        "absence: no_sampled_surface=UNKNOWN; never inside/outside/free space",
        (
            f"sources: components={len(classifications)} mesh_surfaces={surface_meshes} "
            f"point_sets={point_sets}"
        ),
        (
            f"micro_grid192: dims_xyz={_format_vector(micro_frame.dimensions)} "
            f"pitch={micro_frame.pitch:.9g} "
            f"grid_min_xyz={_format_vector(micro_frame.bounds_min)}"
        ),
        (
            "stack_policy: fixed_axis=Z planes=XY candidates=32,24,16 "
            f"selected={stack.longest_axis_cells}"
        ),
        (
            "negative_space: no trusted clear/inside/outside evidence; "
            "void_topology=unknown global_free_space=false"
        ),
        "surface_rule: bands=sampled boundary support only; not occupancy/thickness/interior",
    ]
    reason_counts = Counter(
        item.reason for item in classifications if item.evidence_kind == "surface_hit_band"
    )
    if reason_counts:
        lines.append(
            "surface_sources: "
            + ",".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
        )

    lines.extend(
        [
            "",
            *_format_micro_slice_bounds(
                stack_longest_axis_cells=micro_slice_bounds.stack_longest_axis_cells,
                stack_slice_count=micro_slice_bounds.stack_slice_count,
                runs=micro_slice_bounds.runs,
                selection=micro_slice_bounds.selection,
            ),
        ]
    )

    if body_walls.rows:
        lines.extend(["", *_format_body_walls(body_walls)])

    if face_features.features:
        lines.extend(["", *format_face_features(face_features)])

    if face_openings.openings:
        lines.extend(["", *format_face_openings(face_openings)])

    if interior_islands.groups:
        lines.extend(["", *format_interior_islands(interior_islands)])

    if member_tracks.tracks:
        lines.extend(["", *_format_member_tracks(member_tracks, micro_frame)])

    axis_name = "XYZ"[stack.stack_axis]
    lines.extend(
        [
            "",
            (
                f"[SLICE_STACK{stack.longest_axis_cells} axis={axis_name} "
                f"dims_xyz={_format_vector(stack.frame.dimensions)} "
                f"pitch={stack.frame.pitch:.9g} "
                f"grid_min_xyz={_format_vector(stack.frame.bounds_min)} "
                f"slice_order={stack.sequence_direction} rows={stack.row_direction} "
                f"cols={stack.col_direction}]"
            ),
            (
                f"stack_stats: evidence_cells={stack.evidence_cell_count} "
                f"sampled_surface_cells={stack.surface_cell_count} "
                f"patterns={len(stack.patterns)} "
                f"palette_grid_glyphs={stack.palette_grid_glyphs}"
            ),
            "slice_runs: half-open indices in slice_order",
        ]
    )
    for glyph, start, end in _sequence_runs(stack.sequence):
        lines.append(f"run {glyph}: slices={start}:{end}")
    for glyph, pattern in stack.patterns:
        lines.extend(["", f"[PATTERN {glyph}]", *pattern])
    return "\n".join(lines) + "\n"


def _select_complete_core(
    *,
    source_bounds_min: np.ndarray,
    source_bounds_max: np.ndarray,
    semantic_front_resolved: bool,
    micro_frame: GridFrame,
    micro_volume: np.ndarray,
    classifications: tuple[ComponentClassification, ...],
    stack_resolution_candidates: Sequence[int],
    face_features: FaceFeatureSummary,
    face_openings: FaceOpeningSummary,
    member_tracks: MemberTrackSummary,
    core_byte_budget: int,
    palette_glyph_budget: int,
) -> tuple[AxialObservation | None, tuple[tuple[int, int], ...]]:
    """Select the first candidate whose stack-bound core fits ``core_byte_budget``.

    The stack-bound core is the serialized Core minus the face-derived sections (FACE_FEATURES,
    FACE_OPENINGS, MEMBER_TRACKS192, INTERIOR_ISLANDS); those ride above the budget under their own
    caps, so nothing they add or lose changes the stack resolution. No candidate-owned section is
    ever truncated.
    """

    candidate_costs: list[tuple[int, int]] = []
    for resolution in stack_resolution_candidates:
        frame = build_grid_frame(
            source_bounds_min,
            source_bounds_max,
            longest_axis_cells=resolution,
            block_cells=STACK_BLOCK_CELLS,
        )
        volume = _quantize_volume(micro_volume, micro_frame, frame)
        stack = _best_stack(volume, frame, longest_axis_cells=resolution)
        if stack is None:
            candidate_costs.append((resolution, -1))
            continue
        candidate_costs.append((resolution, stack.palette_grid_glyphs))
        stack = replace(stack, candidate_costs=tuple(candidate_costs))
        if stack.palette_grid_glyphs > palette_glyph_budget:
            continue
        micro_slice_bounds = _build_micro_slice_bounds_summary(
            micro_volume,
            micro_frame,
            stack,
        )
        body_walls = _build_body_walls(micro_volume, micro_frame, micro_slice_bounds.runs)
        interior_islands = detect_interior_islands(micro_slice_bounds.runs, micro_frame)
        text = _format_core(
            source_bounds_min=source_bounds_min,
            source_bounds_max=source_bounds_max,
            semantic_front_resolved=semantic_front_resolved,
            micro_frame=micro_frame,
            classifications=classifications,
            micro_slice_bounds=micro_slice_bounds,
            body_walls=body_walls,
            face_features=face_features,
            face_openings=face_openings,
            interior_islands=interior_islands,
            member_tracks=member_tracks,
            stack=stack,
        )
        byte_count = len(text.encode("utf-8"))
        core_bytes = byte_count - (
            face_features.section_bytes
            + face_openings.section_bytes
            + interior_islands.section_bytes
            + member_tracks.section_bytes
        )
        if core_bytes <= core_byte_budget:
            return (
                AxialObservation(
                    source_bounds_min=source_bounds_min,
                    source_bounds_max=source_bounds_max,
                    semantic_front_resolved=semantic_front_resolved,
                    micro_frame=micro_frame,
                    micro_volume=micro_volume,
                    classifications=classifications,
                    micro_slice_bounds=micro_slice_bounds,
                    body_walls=body_walls,
                    face_features=face_features,
                    face_openings=face_openings,
                    interior_islands=interior_islands,
                    member_tracks=member_tracks,
                    stack=stack,
                    agent_core=text,
                    agent_core_byte_count=byte_count,
                    selection_core_byte_budget=core_byte_budget,
                    selection_reason="unclassified",
                ),
                tuple(candidate_costs),
            )
    return None, tuple(candidate_costs)


def build_axial_observation(
    scene: NormalizedSourceScene | _NormalizedSceneLike,
    *,
    core_byte_budget: int = AGENT_CORE_BYTE_BUDGET,
    palette_glyph_budget: int = STACK_PALETTE_GLYPH_BUDGET,
) -> AxialObservation:
    """Build one deterministic AXIAL surface observation from a normalized source scene."""

    if core_byte_budget <= 0 or palette_glyph_budget <= 0:
        raise ValueError("AXIAL budgets must be positive")
    if core_byte_budget > AGENT_CORE_BYTE_BUDGET:
        raise ValueError("AXIAL observation budget cannot exceed the 32 KiB hard maximum")
    if palette_glyph_budget > AGENT_CORE_BYTE_BUDGET:
        raise ValueError("AXIAL palette budget cannot exceed the 32 KiB hard maximum")
    _validate_normalized_scene_metadata(scene)
    semantic_front_resolved = _semantic_front_resolved(scene)
    components = _scene_components(scene)
    all_vertices = np.concatenate([component.vertices for component in components], axis=0)
    source_bounds_min = all_vertices.min(axis=0)
    source_bounds_max = all_vertices.max(axis=0)
    micro_frame = build_grid_frame(
        source_bounds_min,
        source_bounds_max,
        longest_axis_cells=MICRO_LONGEST_AXIS_CELLS,
        block_cells=16,
    )
    micro_volume, classifications = _build_micro_evidence(components, micro_frame)
    if not np.any(micro_volume):
        raise ValueError("normalized source produced no sampled AXIAL evidence")
    face_features = detect_face_features(components)
    face_openings = detect_face_openings(components)
    member_tracks = _build_member_track_summary(micro_volume, micro_frame)

    selected, candidate_costs = _select_complete_core(
        source_bounds_min=source_bounds_min,
        source_bounds_max=source_bounds_max,
        semantic_front_resolved=semantic_front_resolved,
        micro_frame=micro_frame,
        micro_volume=micro_volume,
        classifications=classifications,
        stack_resolution_candidates=STACK_RESOLUTION_CANDIDATES,
        face_features=face_features,
        face_openings=face_openings,
        member_tracks=member_tracks,
        core_byte_budget=core_byte_budget,
        palette_glyph_budget=palette_glyph_budget,
    )
    if selected is None:
        raise ValueError(
            "no AXIAL slice stack keeps the stack-bound core within the Agent Core ceiling; "
            f"limit={core_byte_budget} candidates={candidate_costs}"
        )
    if selected.agent_core_byte_count > AGENT_CORE_TEXT_MAX_BYTES:
        raise ValueError(
            "AXIAL face-derived sections exceed their allowance above the core ceiling; "
            f"text={selected.agent_core_byte_count} limit={AGENT_CORE_TEXT_MAX_BYTES}"
        )
    finest = selected.stack.longest_axis_cells == STACK_RESOLUTION_CANDIDATES[0]
    return replace(
        selected,
        selection_reason="finest_within_core_ceiling" if finest else "coarser_for_core_ceiling",
    )


def format_axial(observation: AxialObservation) -> str:
    """Return the already budget-checked deterministic Agent Core."""

    return observation.agent_core


def _linear_runs(mask: np.ndarray) -> list[list[int]]:
    occupied = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
    if len(occupied) == 0:
        return []
    runs: list[list[int]] = []
    start = int(occupied[0])
    previous = start
    for raw_value in occupied[1:]:
        value = int(raw_value)
        if value != previous + 1:
            runs.append([start, previous - start + 1])
            start = value
        previous = value
    runs.append([start, previous - start + 1])
    return runs


def format_voxel_artifact(observation: AxialObservation) -> dict[str, object]:
    """Return the selected sampled-surface cells without inventing an interior channel."""

    stack = observation.stack
    volume = stack.volume
    return {
        "schema": "axial_voxel",
        "pose": "normalized_q0",
        "coordinate_frame": "recon_world",
        "evidence_semantics": "sampled_reconstruction_surface_interior_unknown",
        "interior_semantics": "unknown",
        "solid_authority": "none",
        "source_bounds_min_xyz": [float(value) for value in observation.source_bounds_min],
        "source_bounds_max_xyz": [float(value) for value in observation.source_bounds_max],
        "grid": {
            "longest_axis_cells": stack.longest_axis_cells,
            "dimensions_xyz": [int(value) for value in stack.frame.dimensions],
            "pitch": stack.frame.pitch,
            "bounds_min_xyz": [float(value) for value in stack.frame.bounds_min],
            "bounds_max_xyz": [float(value) for value in stack.frame.bounds_max],
            "dense_cell_count": int(volume.size),
            "evidence_cell_count": stack.evidence_cell_count,
            "sampled_surface_cell_count": stack.surface_cell_count,
        },
        "encoding": {
            "kind": "sampled_surface_linear_runs",
            "axis_order": "XYZ",
            "fastest_axis": "Z",
            "linear_index": "((x * ny) + y) * nz + z",
            "runs": _linear_runs((volume & SURFACE_BIT) != 0),
        },
        "limitations": [
            "Cells are sampled source-surface support, not occupied material or wall thickness.",
            "Closed or watertight topology does not make the bounded interior observed or solid.",
            "No-evidence cells are unknown and never constitute a global free-space proof.",
            "This selected stack is a lossy projection of the 192-cell micro evidence.",
        ],
    }
