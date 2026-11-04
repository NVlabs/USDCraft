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

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, isfinite, pi, sin, tan
from numbers import Real
from typing import Iterable, List, Optional, Sequence, Tuple, cast

from ._sweep_paths import (
    _compute_path_frames,
    _compute_path_tangents,
    _dedupe_path_points,
    _insert_sweep_section_points,
    _subdivide_sweep_path,
    _sweep_path_fractions,
    _validate_sweep_vertex_budget,
)
from ._sweep_profiles import (
    _align_sweep_profile_corners,
    _merged_sweep_profile_knots,
    _ordered_sweep_profiles_close,
    _sample_sweep_profile_at_knots,
    _sweep_profile_at_position,
    _SweepSectionProfile,
    _transform_sweep_profile,
    _validate_sweep_alignment_budget,
)
from .booleans import (
    boolean_difference,
)
from .common import (
    _EPS,
    Vec2,
    Vec3,
    _clamp,
    _ensure_ccw,
    _profile_points_2d,
    _v_add,
    _v_cross,
    _v_dot,
    _v_norm,
    _v_normalize,
    _v_scale,
    _v_sub,
    sample_arc_3d,
    sample_catmull_rom_spline_3d,
    sample_cubic_bezier_spline_3d,
)
from .primitives import (
    BoxGeometry,
    LoftGeometry,
    MeshGeometry,
    RoundedBoxGeometry,
)


def _normalize_section_interpolation(value: str) -> str:
    interpolation = str(value).strip().lower().replace("-", "_")
    if interpolation not in {"linear", "catmull_rom"}:
        raise ValueError("section interpolation must be 'linear' or 'catmull_rom'")
    return interpolation


def _normalize_section_tension(value: float) -> float:
    tension = float(value)
    if not isfinite(tension) or not 0.0 <= tension <= 1.0:
        raise ValueError("section tension must be finite and between 0 and 1")
    return tension


@dataclass(frozen=True)
class SweepSection:
    """Control one profile at a normalized arc-length position on a sweep path.

    ``profile`` replaces the sweep's base profile at this control. ``scale``,
    ``rotation`` and ``offset`` are then applied in the local 2D profile plane.
    ``interpolation`` and ``tension`` override how this control transitions to
    the following control.
    """

    position: float
    profile: Optional[Sequence[Vec2]] = None
    scale: float | Sequence[float] = 1.0
    rotation: float = 0.0
    offset: Sequence[float] = (0.0, 0.0)
    interpolation: Optional[str] = None
    tension: Optional[float] = None

    def __post_init__(self) -> None:
        position = float(self.position)
        if not isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("sweep section position must be finite and between 0 and 1")
        object.__setattr__(self, "position", position)

        if isinstance(self.scale, Real):
            scale = (float(self.scale), float(self.scale))
        else:
            scale_values = cast(Sequence[float], self.scale)
            if len(scale_values) != 2:
                raise ValueError("sweep section scale must be a scalar or 2-sequence")
            scale = (float(scale_values[0]), float(scale_values[1]))
        if not all(isfinite(component) and component > 0.0 for component in scale):
            raise ValueError("sweep section scale values must be finite and positive")
        object.__setattr__(self, "scale", scale)

        rotation = float(self.rotation)
        if not isfinite(rotation):
            raise ValueError("sweep section rotation must be finite")
        object.__setattr__(self, "rotation", rotation)

        if len(self.offset) != 2:
            raise ValueError("sweep section offset must contain two values")
        offset = (float(self.offset[0]), float(self.offset[1]))
        if not all(isfinite(component) for component in offset):
            raise ValueError("sweep section offset values must be finite")
        object.__setattr__(self, "offset", offset)

        if self.interpolation is not None:
            object.__setattr__(
                self,
                "interpolation",
                _normalize_section_interpolation(self.interpolation),
            )
        if self.tension is not None:
            object.__setattr__(self, "tension", _normalize_section_tension(self.tension))

        if self.profile is not None:
            profile = tuple(_profile_points_2d(self.profile))
            if not all(isfinite(component) for point in profile for component in point):
                raise ValueError("sweep section profile values must be finite")
            object.__setattr__(self, "profile", profile)


class SweepGeometry(MeshGeometry):
    """Sweep a 2D profile along a 3D path, keeping the profile plane parallel (no tangent frame).

    ``sections`` vary profile, scale, rotation and offset by polyline arc length; use
    ``PipeGeometry`` for tangent-following sweeps.
    """

    def __init__(
        self,
        profile: Iterable[Tuple[float, float]],
        path: Iterable[Tuple[float, float, float]],
        *,
        cap: bool = False,
        closed: bool = True,
        sections: Sequence[SweepSection] = (),
        section_interpolation: str = "linear",
        section_tension: float = 0.0,
        max_segment_length: Optional[float] = None,
    ):
        super().__init__()
        points = _ensure_ccw(_profile_points_2d(profile))
        if not closed:
            cap = False

        path_points = [(float(x), float(y), float(z)) for (x, y, z) in path]
        if len(path_points) < 2:
            raise ValueError("Sweep requires at least two path points")

        section_values = tuple(sections)
        interpolation = _normalize_section_interpolation(section_interpolation)
        tension = _normalize_section_tension(section_tension)
        if not section_values and max_segment_length is None:
            _validate_sweep_vertex_budget(len(path_points), len(points))
            profiles = [[(x + px, y + py, pz) for (x, y) in points] for (px, py, pz) in path_points]
            self.merge(LoftGeometry(profiles, cap=cap, closed=closed))
            return

        path_points = _dedupe_path_points(path_points)
        if len(path_points) < 2:
            raise ValueError("Sweep requires at least two distinct path points")
        section_profiles = _prepare_sweep_section_profiles(
            points,
            section_values,
            closed=closed,
            path_closed=False,
            interpolation=interpolation,
            tension=tension,
        )
        path_points = _insert_sweep_section_points(
            path_points,
            (section.position for section in section_profiles),
            closed=False,
            profile_point_count=len(section_profiles[0].points),
        )
        path_points = _subdivide_sweep_path(
            path_points,
            max_segment_length=max_segment_length,
            closed=False,
            profile_point_count=len(section_profiles[0].points),
        )
        path_fractions, _total = _sweep_path_fractions(path_points, closed=False)
        profiles = []
        for (px, py, pz), position in zip(path_points, path_fractions):
            profile_points = _sweep_profile_at_position(
                section_profiles,
                position,
                closed=closed,
                path_closed=False,
            )
            profiles.append([(x + px, y + py, pz) for (x, y) in profile_points])
        self.merge(LoftGeometry(profiles, cap=cap, closed=closed))


def _section_scale_pair(scale: float | Sequence[float]) -> tuple[float, float]:
    """Read a ``SweepSection.scale`` that ``__post_init__`` already normalized to a pair."""
    pair = cast(Sequence[float], scale)
    return (float(pair[0]), float(pair[1]))


def _prepare_sweep_section_profiles(
    base_profile: Sequence[Vec2],
    sections: Sequence[SweepSection],
    *,
    closed: bool,
    path_closed: bool,
    interpolation: str,
    tension: float,
) -> List[_SweepSectionProfile]:
    controls = list(sections)
    if any(not isinstance(section, SweepSection) for section in controls):
        raise TypeError("sections must contain SweepSection values")
    controls.sort(key=lambda section: section.position)
    if any(
        abs(controls[index].position - controls[index - 1].position) <= _EPS
        for index in range(1, len(controls))
    ):
        raise ValueError("sweep section positions must be unique")

    if not controls or controls[0].position > _EPS:
        controls.insert(0, SweepSection(0.0))
    has_explicit_end = controls[-1].position >= 1.0 - _EPS
    if not has_explicit_end:
        controls.append(SweepSection(1.0))

    raw_profiles: List[List[Vec2]] = []
    for control in controls:
        raw = list(base_profile) if control.profile is None else list(control.profile)
        prepared = _ensure_ccw(raw) if closed else raw
        # The common knot vector can never be smaller than any one source
        # profile, so this is a safe lower-bound rejection before alignment or
        # union workspace is allocated.
        _validate_sweep_alignment_budget(len(controls), len(prepared))
        raw_profiles.append(prepared)

    aligned_raw_profiles: List[List[Vec2]] = []
    for index, raw_profile in enumerate(raw_profiles):
        aligned = list(raw_profile)
        if aligned_raw_profiles:
            reference = (
                aligned_raw_profiles[0]
                if path_closed and controls[index].position >= 1.0 - _EPS
                else aligned_raw_profiles[-1]
            )
            aligned = _align_sweep_profile_corners(reference, aligned, closed=closed)
        aligned_raw_profiles.append(aligned)

    common_knots = _merged_sweep_profile_knots(aligned_raw_profiles, closed=closed)
    _validate_sweep_alignment_budget(len(controls), len(common_knots))
    aligned_profiles = [
        _sample_sweep_profile_at_knots(profile, common_knots, closed=closed)
        for profile in aligned_raw_profiles
    ]

    profiles = [
        _SweepSectionProfile(
            position=control.position,
            points=tuple(points),
            scale=_section_scale_pair(control.scale),
            rotation=control.rotation,
            offset=(float(control.offset[0]), float(control.offset[1])),
            interpolation=(control.interpolation or interpolation),
            tension=(tension if control.tension is None else control.tension),
        )
        for control, points in zip(controls, aligned_profiles)
    ]

    if path_closed and not has_explicit_end:
        first = profiles[0]
        final = profiles[-1]
        profiles[-1] = _SweepSectionProfile(
            position=1.0,
            points=first.points,
            scale=first.scale,
            rotation=first.rotation,
            offset=first.offset,
            interpolation=final.interpolation,
            tension=final.tension,
        )
    if path_closed:
        final = profiles[-1]
        first_transformed = _transform_sweep_profile(
            profiles[0].points,
            scale=profiles[0].scale,
            rotation=profiles[0].rotation,
            offset=profiles[0].offset,
        )
        final_transformed = _transform_sweep_profile(
            final.points,
            scale=final.scale,
            rotation=final.rotation,
            offset=final.offset,
        )
        if not _ordered_sweep_profiles_close(first_transformed, final_transformed):
            raise ValueError(
                "A closed sweep path requires the same ordered transformed profile at "
                "positions 0 and 1; net cyclic seam permutation/twist is unsupported"
            )
    return profiles


@dataclass
class _WireCorner:
    incoming_end: Vec3
    outgoing_start: Vec3
    bridge: List[Vec3]


def _append_distinct_path_point(points: List[Vec3], point: Vec3, *, tol: float = _EPS) -> None:
    p = (float(point[0]), float(point[1]), float(point[2]))
    if not points or _v_norm(_v_sub(points[-1], p)) > tol:
        points.append(p)


def _preprocess_wire_polyline_points(
    points: Iterable[Vec3],
    *,
    min_segment_length: float,
    closed_path: bool,
) -> List[Vec3]:
    raw = [(float(x), float(y), float(z)) for (x, y, z) in points]
    if len(raw) < 2:
        raise ValueError("WirePolylineGeometry requires at least two input points")

    tol = max(float(min_segment_length), _EPS)
    out: List[Vec3] = [raw[0]]
    for point in raw[1:]:
        if _v_norm(_v_sub(point, out[-1])) >= tol:
            out.append(point)

    if len(out) < 2:
        raise ValueError("Too few distinct points after dedupe")

    if closed_path:
        if _v_norm(_v_sub(out[0], out[-1])) <= tol:
            out[-1] = out[0]
        else:
            out.append(out[0])
        # Closed loops need at least three distinct vertices.
        if len(out) < 4:
            raise ValueError("Closed path requires at least three distinct points")

    return out


def _wire_corner_from_three_points(
    prev_point: Vec3,
    corner_point: Vec3,
    next_point: Vec3,
    *,
    corner_mode: str,
    corner_radius: float,
    corner_segments: int,
) -> _WireCorner:
    incoming = _v_sub(corner_point, prev_point)
    outgoing = _v_sub(next_point, corner_point)
    len_in = _v_norm(incoming)
    len_out = _v_norm(outgoing)

    if len_in <= _EPS or len_out <= _EPS:
        if corner_mode == "fillet" and corner_radius > 0.0:
            raise ValueError("Impossible fillet: corner has zero-length segment")
        return _WireCorner(corner_point, corner_point, [corner_point])

    u = _v_scale(incoming, 1.0 / len_in)
    v = _v_scale(outgoing, 1.0 / len_out)
    theta = acos(_clamp(_v_dot(_v_scale(u, -1.0), v), -1.0, 1.0))

    # Straight-through and U-turn corners fall back to miter behavior.
    if theta <= 1e-6 or abs(pi - theta) <= 1e-6:
        return _WireCorner(corner_point, corner_point, [corner_point])
    if corner_mode == "miter" or corner_radius <= 0.0:
        return _WireCorner(corner_point, corner_point, [corner_point])

    tan_half = tan(theta * 0.5)
    if abs(tan_half) <= _EPS:
        return _WireCorner(corner_point, corner_point, [corner_point])

    target_trim = corner_radius * tan_half
    if not isfinite(target_trim):
        target_trim = float("inf")
    trim = min(target_trim, 0.49 * min(len_in, len_out))
    if trim <= _EPS:
        return _WireCorner(corner_point, corner_point, [corner_point])

    a = _v_sub(corner_point, _v_scale(u, trim))
    b = _v_add(corner_point, _v_scale(v, trim))

    if corner_mode == "bevel":
        return _WireCorner(a, b, [a, b])

    radius_eff = trim / tan_half
    if radius_eff <= _EPS:
        return _WireCorner(corner_point, corner_point, [corner_point])

    bisector = _v_add(_v_scale(u, -1.0), v)
    if _v_norm(bisector) <= _EPS:
        return _WireCorner(corner_point, corner_point, [corner_point])
    bisector = _v_normalize(bisector)

    sin_half = sin(theta * 0.5)
    if abs(sin_half) <= _EPS:
        return _WireCorner(corner_point, corner_point, [corner_point])
    center_dist = radius_eff / sin_half
    center = _v_add(corner_point, _v_scale(bisector, center_dist))

    axis = _v_cross(u, v)
    if _v_norm(axis) <= _EPS:
        return _WireCorner(corner_point, corner_point, [corner_point])
    axis = _v_normalize(axis)

    ra = _v_sub(a, center)
    rb = _v_sub(b, center)
    if _v_norm(ra) <= _EPS or _v_norm(rb) <= _EPS:
        return _WireCorner(corner_point, corner_point, [corner_point])
    ra_u = _v_normalize(ra)
    rb_u = _v_normalize(rb)

    arc_angle = acos(_clamp(_v_dot(ra_u, rb_u), -1.0, 1.0))
    if _v_dot(axis, _v_cross(ra_u, rb_u)) < 0.0:
        arc_angle = -arc_angle
    if abs(arc_angle) <= _EPS:
        return _WireCorner(corner_point, corner_point, [corner_point])

    arc_points = sample_arc_3d(
        start_point=a,
        center=center,
        normal=axis,
        angle=arc_angle,
        segments=max(2, int(corner_segments)),
    )
    arc_points[0] = a
    arc_points[-1] = b
    return _WireCorner(a, b, arc_points)


def _build_wire_centerline(
    points: List[Vec3],
    *,
    closed_path: bool,
    corner_mode: str,
    corner_radius: float,
    corner_segments: int,
) -> List[Vec3]:
    if not closed_path:
        if len(points) < 2:
            raise ValueError("Wire path requires at least two points")
        if len(points) == 2:
            return list(points)

        out: List[Vec3] = []
        _append_distinct_path_point(out, points[0])
        for i in range(1, len(points) - 1):
            corner = _wire_corner_from_three_points(
                points[i - 1],
                points[i],
                points[i + 1],
                corner_mode=corner_mode,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
            )
            _append_distinct_path_point(out, corner.incoming_end)
            for point in corner.bridge[1:]:
                _append_distinct_path_point(out, point)
        _append_distinct_path_point(out, points[-1])
        return out

    ring = points[:-1]
    if len(ring) < 3:
        raise ValueError("Closed path requires at least three distinct points")

    corners: List[_WireCorner] = []
    n = len(ring)
    for i in range(n):
        corners.append(
            _wire_corner_from_three_points(
                ring[(i - 1) % n],
                ring[i],
                ring[(i + 1) % n],
                corner_mode=corner_mode,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
            )
        )

    out = []
    _append_distinct_path_point(out, corners[0].outgoing_start)
    for i in range(n):
        j = (i + 1) % n
        corner = corners[j]
        _append_distinct_path_point(out, corner.incoming_end)
        for point in corner.bridge[1:]:
            _append_distinct_path_point(out, point)
    if _v_norm(_v_sub(out[0], out[-1])) > _EPS:
        out.append(out[0])
    else:
        out[-1] = out[0]
    return out


def _wire_circle_profile(radius: float, radial_segments: int) -> List[Vec2]:
    return [
        (
            float(radius) * cos(2.0 * pi * i / radial_segments),
            float(radius) * sin(2.0 * pi * i / radial_segments),
        )
        for i in range(radial_segments)
    ]


class PipeGeometry(MeshGeometry):
    """
    Sweep a 2D profile along a 3D path using transported local frames.

    Unlike `SweepGeometry`, profile orientation follows the path tangent,
    which is important for wire-like and tube-like parts.
    """

    def __init__(
        self,
        profile: Iterable[Tuple[float, float]],
        path: Iterable[Tuple[float, float, float]],
        *,
        cap: bool = False,
        closed: bool = True,
        path_closed: bool = False,
        up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        sections: Sequence[SweepSection] = (),
        section_interpolation: str = "linear",
        section_tension: float = 0.0,
        max_segment_length: Optional[float] = None,
    ):
        super().__init__()
        raw_points = _profile_points_2d(profile)
        points = _ensure_ccw(raw_points) if closed else raw_points
        interpolation = _normalize_section_interpolation(section_interpolation)
        tension = _normalize_section_tension(section_tension)
        if not closed:
            cap = False
        path_points = [(float(x), float(y), float(z)) for (x, y, z) in path]
        path_points = _dedupe_path_points(path_points)
        if len(path_points) < 2:
            raise ValueError("Pipe requires at least two distinct path points")
        if path_closed:
            if _v_norm(_v_sub(path_points[0], path_points[-1])) <= _EPS:
                path_points.pop()
            if len(path_points) < 3:
                raise ValueError("A closed pipe path requires at least three distinct points")
            cap = False

        section_profiles = _prepare_sweep_section_profiles(
            points,
            tuple(sections),
            closed=closed,
            path_closed=path_closed,
            interpolation=interpolation,
            tension=tension,
        )
        path_points = _insert_sweep_section_points(
            path_points,
            (section.position for section in section_profiles),
            closed=path_closed,
            profile_point_count=len(section_profiles[0].points),
            extra_vertices=(2 if cap and closed else 0),
        )
        path_points = _subdivide_sweep_path(
            path_points,
            max_segment_length=max_segment_length,
            closed=path_closed,
            profile_point_count=len(section_profiles[0].points),
            extra_vertices=(2 if cap and closed else 0),
        )

        tangents = _compute_path_tangents(path_points, closed=path_closed)
        normals, binormals = _compute_path_frames(
            path_points,
            tangents,
            closed=path_closed,
            up_hint=(float(up_hint[0]), float(up_hint[1]), float(up_hint[2])),
        )

        path_fractions, _total = _sweep_path_fractions(
            path_points,
            closed=path_closed,
        )
        profiles: List[List[Vec3]] = []
        for i, (p, position) in enumerate(zip(path_points, path_fractions)):
            n = normals[i]
            binormal = binormals[i]
            profile_points = _sweep_profile_at_position(
                section_profiles,
                position,
                closed=closed,
                path_closed=path_closed,
            )
            ring: List[Vec3] = []
            for x, y in profile_points:
                ring.append(
                    (
                        p[0] + x * n[0] + y * binormal[0],
                        p[1] + x * n[1] + y * binormal[1],
                        p[2] + x * n[2] + y * binormal[2],
                    )
                )
            profiles.append(ring)
        ring_count = len(section_profiles[0].points)
        if ring_count < 2:
            raise ValueError("Pipe profile must contain at least 2 points")

        ring_indices: List[List[int]] = []
        for ring in profiles:
            idxs: List[int] = []
            for x, y, z in ring:
                idxs.append(self.add_vertex(x, y, z))
            ring_indices.append(idxs)

        seg_count = ring_count if closed else (ring_count - 1)
        path_segment_count = len(ring_indices) if path_closed else len(ring_indices) - 1
        for i in range(path_segment_count):
            r0 = ring_indices[i]
            r1 = ring_indices[(i + 1) % len(ring_indices)]
            for j in range(seg_count):
                j2 = (j + 1) % ring_count
                a = r0[j]
                b = r0[j2]
                c = r1[j2]
                d = r1[j]
                self.add_face(a, b, c)
                self.add_face(a, c, d)

        if cap and closed and ring_count >= 3:
            first = profiles[0]
            last = profiles[-1]
            c0 = (
                sum(v[0] for v in first) / ring_count,
                sum(v[1] for v in first) / ring_count,
                sum(v[2] for v in first) / ring_count,
            )
            c1 = (
                sum(v[0] for v in last) / ring_count,
                sum(v[1] for v in last) / ring_count,
                sum(v[2] for v in last) / ring_count,
            )
            i0 = self.add_vertex(*c0)
            i1 = self.add_vertex(*c1)
            r0 = ring_indices[0]
            r1 = ring_indices[-1]
            for j in range(ring_count):
                j2 = (j + 1) % ring_count
                self.add_face(i0, r0[j2], r0[j])
                self.add_face(i1, r1[j], r1[j2])


class ArcPipeGeometry(MeshGeometry):
    """
    Advanced compatibility helper: profile swept over a circular 3D arc.

    Prefer `sample_arc_3d(...)` + `PipeGeometry(...)` or the higher-level wire
    helpers for new code.
    """

    def __init__(
        self,
        profile: Iterable[Tuple[float, float]],
        *,
        start_point: Vec3,
        center: Vec3,
        normal: Vec3,
        angle: float,
        arc_segments: int = 20,
        cap: bool = False,
        closed: bool = True,
        up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        sections: Sequence[SweepSection] = (),
        section_interpolation: str = "linear",
        section_tension: float = 0.0,
        max_segment_length: Optional[float] = None,
    ):
        super().__init__()
        path = sample_arc_3d(
            start_point=start_point,
            center=center,
            normal=normal,
            angle=angle,
            segments=arc_segments,
        )
        self.merge(
            PipeGeometry(
                profile,
                path,
                cap=cap,
                closed=closed,
                up_hint=up_hint,
                sections=sections,
                section_interpolation=section_interpolation,
                section_tension=section_tension,
                max_segment_length=max_segment_length,
            )
        )


class WirePolylineGeometry(MeshGeometry):
    """
    Compatibility class-form wire primitive built from centerline points.

    Prefer `wire_from_points(...)` for new code. This class remains available
    when the explicit class form is useful.
    """

    def __init__(
        self,
        points: Iterable[Tuple[float, float, float]],
        *,
        radius: float,
        radial_segments: int = 16,
        closed_path: bool = False,
        cap_ends: bool = False,
        corner_mode: str = "fillet",
        corner_radius: float = 0.0,
        corner_segments: int = 8,
        up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        min_segment_length: float = 1e-6,
    ):
        super().__init__()
        radius = float(radius)
        radial_segments = int(radial_segments)
        closed_path = bool(closed_path)
        corner_mode = str(corner_mode).strip().lower()
        corner_radius = float(corner_radius)
        corner_segments = int(corner_segments)
        up_hint_v = (float(up_hint[0]), float(up_hint[1]), float(up_hint[2]))
        min_segment_length = float(min_segment_length)

        if radius <= 0.0:
            raise ValueError("radius must be > 0")
        if radial_segments < 6:
            raise ValueError("radial_segments must be >= 6")
        if corner_mode not in {"fillet", "miter", "bevel"}:
            raise ValueError("corner_mode must be one of: fillet, miter, bevel")
        if corner_radius < 0.0:
            raise ValueError("corner_radius must be >= 0")
        if corner_mode == "fillet" and corner_radius > 0.0 and corner_segments < 2:
            raise ValueError("corner_segments must be >= 2 when fillets are active")
        if _v_norm(up_hint_v) <= _EPS:
            raise ValueError("up_hint must be non-zero")
        if min_segment_length <= 0.0:
            raise ValueError("min_segment_length must be > 0")

        preprocessed = _preprocess_wire_polyline_points(
            points,
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
        if len(centerline) < 2:
            raise ValueError("Too few distinct points after corner preprocessing")

        profile = _wire_circle_profile(radius, radial_segments)
        self.merge(
            PipeGeometry(
                profile,
                centerline,
                cap=bool(cap_ends and not closed_path),
                closed=True,
                path_closed=closed_path,
                up_hint=up_hint_v,
            )
        )


def wire_from_points(
    points: Iterable[Tuple[float, float, float]],
    *,
    radius: float,
    radial_segments: int = 16,
    closed_path: bool = False,
    cap_ends: bool = False,
    corner_mode: str = "fillet",
    corner_radius: float = 0.0,
    corner_segments: int = 8,
    up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    min_segment_length: float = 1e-6,
) -> MeshGeometry:
    """Build one tube/wire mesh from a centerline path and radius."""
    return WirePolylineGeometry(
        points=points,
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


def _sample_supported_spline_path(
    points: Iterable[Vec3],
    *,
    spline: str,
    samples_per_segment: int,
    closed_spline: bool,
    alpha: float,
    min_segment_length: float,
) -> List[Vec3]:
    spline_key = str(spline or "catmull_rom").strip().lower().replace("-", "_")
    if spline_key in {"catmullrom", "catmull_rom"}:
        sampled = sample_catmull_rom_spline_3d(
            list(points),
            samples_per_segment=int(samples_per_segment),
            closed=bool(closed_spline),
            alpha=float(alpha),
        )
    elif spline_key in {"bezier", "cubic_bezier", "cubicbezier"}:
        sampled = sample_cubic_bezier_spline_3d(
            list(points),
            samples_per_segment=int(samples_per_segment),
        )
        if bool(closed_spline) and _v_norm(_v_sub(sampled[0], sampled[-1])) > max(
            float(min_segment_length), _EPS
        ):
            raise ValueError(
                "closed_spline=True with spline='bezier' requires the sampled "
                "Bezier chain to end where it starts"
            )
    else:
        raise ValueError("spline must be one of: 'catmull_rom', 'bezier'")

    return _preprocess_wire_polyline_points(
        sampled,
        min_segment_length=float(min_segment_length),
        closed_path=bool(closed_spline),
    )


def tube_from_spline_points(
    points: Iterable[Vec3],
    *,
    radius: float,
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    radial_segments: int = 16,
    cap_ends: bool = True,
    up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    min_segment_length: float = 1e-6,
) -> MeshGeometry:
    """
    Fit a spline through points, then build a circular tube along that path.

    This is the default "click points, fit a nice handle/tube" helper.
    Supported spline families: ``catmull_rom`` and chained cubic ``bezier``.
    """
    centerline = _sample_supported_spline_path(
        points,
        spline=spline,
        samples_per_segment=samples_per_segment,
        closed_spline=closed_spline,
        alpha=alpha,
        min_segment_length=min_segment_length,
    )
    return wire_from_points(
        centerline,
        radius=float(radius),
        radial_segments=int(radial_segments),
        closed_path=bool(closed_spline),
        cap_ends=bool(cap_ends),
        corner_mode="miter",
        up_hint=up_hint,
        min_segment_length=min_segment_length,
    )


def sweep_profile_along_spline(
    points: Iterable[Vec3],
    *,
    profile: Iterable[Vec2],
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    cap_profile: bool = True,
    up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    min_segment_length: float = 1e-6,
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: Optional[float] = None,
) -> MeshGeometry:
    """
    Fit a spline through points, then sweep a 2D profile along the sampled path.

    The profile is treated as a closed outline. Use `tube_from_spline_points(...)`
    for the common circular-tube case. Supported spline families:
    ``catmull_rom`` and chained cubic ``bezier``.
    """
    centerline = _sample_supported_spline_path(
        points,
        spline=spline,
        samples_per_segment=samples_per_segment,
        closed_spline=closed_spline,
        alpha=alpha,
        min_segment_length=min_segment_length,
    )
    return PipeGeometry(
        profile=profile,
        path=centerline,
        cap=bool(cap_profile and not closed_spline),
        closed=True,
        path_closed=bool(closed_spline),
        up_hint=up_hint,
        sections=sections,
        section_interpolation=section_interpolation,
        section_tension=section_tension,
        max_segment_length=max_segment_length,
    )


def beam_between(
    start: Vec3,
    end: Vec3,
    *,
    width: float,
    height: Optional[float] = None,
    cap_ends: bool = True,
    up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> MeshGeometry:
    """
    Straight rectangular-section beam spanning ``start`` -> ``end``.

    The endpoints are baked into world coordinates, so you place the beam without
    computing any rotation yourself. Use this for the angled and diagonal members
    that otherwise force hand-derived rpy math — ladder rails, A-frame legs, roof
    rafters, trusses, railings. For a round strut use
    ``tube_from_spline_points([start, end], radius=...)`` instead.

    ``width`` and ``height`` are the cross-section sizes (``height`` defaults to a
    square section). ``up_hint`` rolls the section about the beam axis; change it if
    a flat member comes out edge-on.
    """
    half_w = float(width) / 2.0
    half_h = (float(width) if height is None else float(height)) / 2.0
    profile: List[Vec2] = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]
    return PipeGeometry(
        profile=profile,
        path=[
            (float(start[0]), float(start[1]), float(start[2])),
            (float(end[0]), float(end[1]), float(end[2])),
        ],
        cap=bool(cap_ends),
        closed=True,
        up_hint=up_hint,
    )


def carve_pocket(
    host: MeshGeometry,
    *,
    center: Vec3,
    size: Vec3,
    corner_radius: float = 0.0,
    round_axis: str = "z",
) -> MeshGeometry:
    """Carve a rectangular pocket (a recess / inset) into ``host`` by boolean
    subtraction, returning the cut solid.

    This is the right tool when the appearance read marks a part as
    ``recessed``/``inset`` (a recessed door handle, an inset panel, a grip
    channel) — model it as material REMOVED, never as a protruding box stuck on
    the surface. ``center`` and ``size`` are in the host's world coordinates; make
    the pocket overshoot the surface along its depth axis (e.g. size a touch larger
    and center slightly outside) so the cut is clean. ``corner_radius`` rounds the
    pocket's mouth via :class:`RoundedBoxGeometry`. ``host`` must be a manifold
    solid.
    """
    if corner_radius and corner_radius > 0.0:
        cutter: MeshGeometry = RoundedBoxGeometry(size, corner_radius, round_axis=round_axis)
    else:
        cutter = BoxGeometry(size)
    cutter.translate(float(center[0]), float(center[1]), float(center[2]))
    return boolean_difference(host, cutter)
