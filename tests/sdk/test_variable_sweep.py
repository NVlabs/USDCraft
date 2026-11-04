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

from __future__ import annotations

from collections import Counter
from math import cos, dist, pi, sin, sqrt

import numpy as np
import pytest

import sdk._core._mesh._sweep_paths as sweep_paths
import sdk._core._mesh._sweep_profiles as sweep_profiles
from sdk._core._mesh.booleans import _manifold_from_geometry
from sdk._core._mesh.health import mesh_health
from sdk._core._mesh.sweeps import (
    PipeGeometry,
    SweepGeometry,
    SweepSection,
    sweep_profile_along_spline,
)

_SQUARE = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
_SMALL_SQUARE = [(-0.03, -0.02), (0.03, -0.02), (0.03, 0.02), (-0.03, 0.02)]
_SMALL_EQUAL_SQUARE = [(-0.03, -0.03), (0.03, -0.03), (0.03, 0.03), (-0.03, 0.03)]


def _ring_span_y(geometry: PipeGeometry, ring_index: int, ring_count: int = 4) -> float:
    ring = geometry.vertices[ring_index * ring_count : (ring_index + 1) * ring_count]
    return max(vertex[1] for vertex in ring) - min(vertex[1] for vertex in ring)


def _ring_area_xy(ring: list[tuple[float, float, float]]) -> float:
    return 0.5 * abs(
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(ring, [*ring[1:], ring[0]])
        )
    )


def _undirected_edge_counts(geometry: PipeGeometry) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for face in geometry.faces:
        for start, end in zip(face, (face[1], face[2], face[0])):
            counts[tuple(sorted((start, end)))] += 1
    return counts


def _ring_at_center(
    geometry: PipeGeometry,
    center: tuple[float, float, float],
    *,
    ring_count: int,
) -> list[tuple[float, float, float]]:
    for start in range(0, len(geometry.vertices), ring_count):
        ring = geometry.vertices[start : start + ring_count]
        if len(ring) != ring_count:
            continue
        actual = tuple(sum(vertex[axis] for vertex in ring) / ring_count for axis in range(3))
        if dist(actual, center) <= 1e-8:
            return ring
    raise AssertionError(f"No ring centered at {center!r}")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"position": -0.01}, "position"),
        ({"position": 0.5, "scale": (1.0, 0.0)}, "scale"),
        ({"position": 0.5, "rotation": float("inf")}, "rotation"),
        ({"position": 0.5, "offset": (0.0, float("nan"))}, "offset"),
        ({"position": 0.5, "interpolation": "cubic"}, "interpolation"),
        ({"position": 0.5, "tension": 1.01}, "tension"),
    ],
)
def test_sweep_section_rejects_invalid_controls(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SweepSection(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("scale", (np.float32(0.5), np.int64(2)))
def test_sweep_section_accepts_real_numpy_scalar_scale(scale: object) -> None:
    section = SweepSection(0.5, scale=scale)  # type: ignore[arg-type]

    assert section.scale == pytest.approx((float(scale), float(scale)))


def test_pipe_variable_section_uses_normalized_arc_length_and_inserts_exact_ring() -> None:
    geometry = PipeGeometry(
        _SQUARE,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
        cap=True,
        sections=[SweepSection(0.5, scale=2.0)],
    )

    # The middle input vertex lies at 1/3 of total arc length. A new ring must
    # therefore be inserted at x=1.5 for the section at normalized position 0.5.
    assert len(geometry.vertices) == 4 * 4 + 2
    inserted_ring = geometry.vertices[8:12]
    assert all(vertex[0] == pytest.approx(1.5) for vertex in inserted_ring)
    assert max(vertex[1] for vertex in inserted_ring) == pytest.approx(1.0)
    assert min(vertex[1] for vertex in inserted_ring) == pytest.approx(-1.0)
    assert max(vertex[2] for vertex in inserted_ring) == pytest.approx(1.0)
    assert min(vertex[2] for vertex in inserted_ring) == pytest.approx(-1.0)


def test_pipe_section_transform_is_applied_in_the_transported_profile_frame() -> None:
    geometry = PipeGeometry(
        _SQUARE,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        sections=[
            SweepSection(
                0.5,
                scale=(2.0, 1.0),
                rotation=pi / 2.0,
                offset=(0.25, -0.1),
            )
        ],
    )

    # For a +X path with +Z up, local profile X maps to world +Y and local
    # profile Y maps to world +Z. The first square corner transforms from
    # (-0.5, -0.5) -> (0.75, -1.1) in that local plane.
    assert geometry.vertices[4] == pytest.approx((0.5, 0.75, -1.1))


def test_rotation_interpolates_as_rigid_transform_at_midpoint() -> None:
    profile = [(-0.5, -0.25), (0.5, -0.25), (0.5, 0.25), (-0.5, 0.25)]
    geometry = SweepGeometry(
        profile,
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)],
        sections=[SweepSection(0.0), SweepSection(1.0, rotation=pi / 2.0)],
    )

    middle = geometry.vertices[4:8]
    assert _ring_area_xy(middle) == pytest.approx(0.5)
    assert dist(middle[0], (0.0, 0.0, 0.5)) == pytest.approx(sqrt(0.5**2 + 0.25**2))
    assert middle[0] == pytest.approx(
        (
            -0.5 * cos(pi / 4.0) + 0.25 * sin(pi / 4.0),
            -0.5 * sin(pi / 4.0) - 0.25 * cos(pi / 4.0),
            0.5,
        )
    )


def test_half_turn_rotation_is_sampling_invariant_and_does_not_collapse() -> None:
    profile = [(-0.5, -0.25), (0.5, -0.25), (0.5, 0.25), (-0.5, 0.25)]
    path = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)]
    sections = [SweepSection(0.0), SweepSection(1.0, rotation=pi)]
    coarse = SweepGeometry(profile, path, sections=sections)
    dense = SweepGeometry(profile, path, sections=sections, max_segment_length=0.1)

    coarse_middle = _ring_at_center(coarse, (0.0, 0.0, 0.5), ring_count=4)
    dense_middle = _ring_at_center(dense, (0.0, 0.0, 0.5), ring_count=4)
    assert _ring_area_xy(coarse_middle) == pytest.approx(0.5)
    for coarse_vertex, dense_vertex in zip(coarse_middle, dense_middle):
        assert coarse_vertex == pytest.approx(dense_vertex)


def test_full_turn_rotation_preserves_explicit_winding() -> None:
    profile = [(-0.5, -0.2), (0.4, -0.3), (0.6, 0.25), (-0.25, 0.45)]
    geometry = SweepGeometry(
        profile,
        [(0.0, 0.0, value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)],
        sections=[SweepSection(0.0), SweepSection(1.0, rotation=2.0 * pi)],
    )

    quarter = geometry.vertices[4:8]
    half = geometry.vertices[8:12]
    assert quarter[0] == pytest.approx((0.2, -0.5, 0.25))
    assert half[0] == pytest.approx((0.5, 0.2, 0.5))
    assert quarter[0][:2] != pytest.approx(geometry.vertices[0][:2])
    assert half[0][:2] != pytest.approx(geometry.vertices[0][:2])


def test_variable_sweep_accepts_small_positive_profile_area() -> None:
    geometry = PipeGeometry(
        _SQUARE,
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
        sections=[SweepSection(0.0, scale=1e-8), SweepSection(1.0, scale=1e-8)],
    )

    assert len(geometry.vertices) == 12


def test_pipe_variable_sections_resample_profiles_and_keep_capped_mesh_watertight() -> None:
    octagon = [
        (1.0, 0.0),
        (0.7, 0.7),
        (0.0, 1.0),
        (-0.7, 0.7),
        (-1.0, 0.0),
        (-0.7, -0.7),
        (0.0, -1.0),
        (0.7, -0.7),
    ]
    geometry = PipeGeometry(
        _SQUARE,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        cap=True,
        sections=[SweepSection(1.0, profile=octagon, rotation=pi / 8.0)],
    )

    # Both rings use the larger profile's point count, so ring correspondence is
    # explicit and every closed-mesh edge has exactly two incident triangles.
    assert len(geometry.vertices) == 8 * 2 + 2
    assert set(_undirected_edge_counts(geometry).values()) == {2}


def test_profile_knot_union_preserves_triangle_and_octagon_control_corners() -> None:
    octagon = [(cos(2.0 * pi * index / 8), sin(2.0 * pi * index / 8)) for index in range(8)]
    triangle = [(-1.0, -0.5), (1.0, -0.5), (0.0, 1.0)]
    geometry = PipeGeometry(
        octagon,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        sections=[SweepSection(1.0, profile=triangle)],
    )

    ring_count = len(geometry.vertices) // 2
    assert ring_count == 11
    start_profile = [(vertex[1], vertex[2]) for vertex in geometry.vertices[:ring_count]]
    end_profile = [(vertex[1], vertex[2]) for vertex in geometry.vertices[-ring_count:]]
    for corner in octagon:
        assert any(actual == pytest.approx(corner) for actual in start_profile)
    for corner in triangle:
        assert any(actual == pytest.approx(corner) for actual in end_profile)


def test_closed_profile_alignment_accepts_exact_asymmetric_cyclic_endpoint() -> None:
    profile = [(0.0, 0.0), (3.0, 0.0), (2.7, 1.0), (1.0, 2.0), (-0.2, 0.7)]
    shifted = [*profile[2:], *profile[:2]]
    geometry = PipeGeometry(
        profile,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.1), (1.0, 1.0, 0.0), (0.0, 1.0, -0.1)],
        path_closed=True,
        sections=[SweepSection(0.0), SweepSection(1.0, profile=shifted)],
    )

    assert len(geometry.vertices) == len(profile) * 4
    assert set(_undirected_edge_counts(geometry).values()) == {2}


def test_closed_profile_alignment_accepts_noisy_cyclic_relabel_without_extra_seam() -> None:
    point_count = 257
    profile = [
        (0.02 * cos(2.0 * pi * index / point_count), 0.02 * sin(2.0 * pi * index / point_count))
        for index in range(point_count)
    ]
    shifted = [*profile[73:], *profile[:73]]
    noisy_shifted = [
        (x + (1e-12 if index % 2 else -1e-12), y + (1e-12 if index % 3 else -1e-12))
        for index, (x, y) in enumerate(shifted)
    ]
    geometry = PipeGeometry(
        profile,
        [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.0, 0.1, 0.0)],
        path_closed=True,
        sections=[
            SweepSection(0.0),
            SweepSection(1.0, profile=noisy_shifted),
        ],
    )

    assert len(geometry.vertices) == point_count * 4


def test_catmull_rom_section_interpolation_and_tension_change_intermediate_rings() -> None:
    path = [(position, 0.0, 0.0) for position in (0.0, 0.125, 0.25, 0.375, 0.5, 1.0)]
    loose = PipeGeometry(
        _SQUARE,
        path,
        sections=[SweepSection(0.5, scale=2.0)],
        section_interpolation="catmull_rom",
        section_tension=0.0,
    )
    tight = PipeGeometry(
        _SQUARE,
        path,
        sections=[SweepSection(0.5, scale=2.0)],
        section_interpolation="catmull_rom",
        section_tension=1.0,
    )

    assert _ring_span_y(loose, 0) == pytest.approx(1.0)
    assert _ring_span_y(loose, 4) == pytest.approx(2.0)
    assert _ring_span_y(loose, 1) > _ring_span_y(tight, 1)


def test_catmull_rom_rejects_non_positive_scale_overshoot() -> None:
    with pytest.raises(ValueError, match="scale.*non-positive"):
        SweepGeometry(
            _SQUARE,
            [(0.0, 0.0, value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)],
            sections=[
                SweepSection(0.0, scale=10.0),
                SweepSection(0.25, scale=0.1),
                SweepSection(0.75, scale=0.1),
                SweepSection(1.0, scale=10.0),
            ],
            section_interpolation="catmull_rom",
        )


def test_max_segment_length_represents_smooth_sections_on_a_sparse_path() -> None:
    sparse = PipeGeometry(
        _SQUARE,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        sections=[SweepSection(0.5, scale=2.0)],
        section_interpolation="catmull_rom",
        max_segment_length=0.125,
    )

    # The section at 0.5 first splits the path, then each half is subdivided.
    # Nine rings expose the Catmull-Rom scale curve instead of connecting only
    # the three control rings with linearly triangulated side walls.
    assert len(sparse.vertices) == 9 * 4
    assert _ring_span_y(sparse, 2) == pytest.approx(1.625)
    assert _ring_span_y(sparse, 2) > 1.5


@pytest.mark.parametrize(
    "maximum",
    [0.0, -1.0, float("nan"), float("inf"), 5e-324],
)
def test_max_segment_length_must_be_finite_and_positive(maximum: float) -> None:
    with pytest.raises(ValueError, match="max_segment_length"):
        PipeGeometry(
            _SQUARE,
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            max_segment_length=maximum,
        )


def test_max_segment_length_rejects_vertex_budget_before_allocation() -> None:
    limit = sweep_paths._MAX_SWEEP_GENERATED_VERTICES
    with pytest.raises(ValueError, match=rf"{limit:,}-vertex limit"):
        PipeGeometry(
            _SQUARE,
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            max_segment_length=1e-9,
        )


def test_section_sampling_does_not_scan_or_materialize_all_controls() -> None:
    control_count = 10_000
    controls = [
        sweep_profiles._SweepSectionProfile(
            position=index / (control_count - 1),
            points=tuple(_SMALL_SQUARE),
            scale=(1.0, 1.0),
            rotation=0.1 * index,
            offset=(0.0, 0.0),
            interpolation="catmull_rom",
            tension=0.0,
        )
        for index in range(control_count)
    ]

    class CountingControls:
        def __init__(self, values: list[object]):
            self.values = values
            self.reads = 0

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int):
            self.reads += 1
            return self.values[index]

    counted = CountingControls(controls)
    result = sweep_profiles._sweep_profile_at_position(
        counted,  # type: ignore[arg-type]
        0.87654321,
        closed=True,
        path_closed=False,
    )

    assert len(result) == len(_SMALL_SQUARE)
    assert counted.reads < 100


def test_fixed_sweep_legacy_path_checks_vertex_budget_before_profile_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sweep_paths, "_MAX_SWEEP_GENERATED_VERTICES", 7)

    with pytest.raises(ValueError, match="7-vertex limit"):
        SweepGeometry(
            _SQUARE,
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )


def test_profile_alignment_rejects_budget_lower_bound_before_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sweep_profiles, "_MAX_SWEEP_ALIGNMENT_POINTS", 15)
    octagon = [(cos(2.0 * pi * index / 8), sin(2.0 * pi * index / 8)) for index in range(8)]

    with pytest.raises(ValueError, match="15-point alignment limit"):
        PipeGeometry(
            _SQUARE,
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            sections=[SweepSection(0.5, profile=octagon)],
        )


def test_section_insertion_uses_linear_merge_for_large_path_and_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FractionSpy(float):
        operations = 0

        def __sub__(self, other: object) -> float:
            type(self).operations += 1
            return float(self) - float(other)  # type: ignore[arg-type]

        def __rsub__(self, other: object) -> float:
            type(self).operations += 1
            return float(other) - float(self)  # type: ignore[arg-type]

    path_count = 8_000
    path = [(float(index), 0.0, 0.0) for index in range(path_count)]
    fractions = [FractionSpy(index / (path_count - 1)) for index in range(path_count)]
    controls = [(index + 0.5) / (path_count - 1) for index in range(path_count - 1)]
    monkeypatch.setattr(
        sweep_paths,
        "_sweep_path_fractions",
        lambda *_args, **_kwargs: (fractions, float(path_count - 1)),
    )

    result = sweep_paths._insert_sweep_section_points(
        path,
        controls,
        profile_point_count=1,
    )

    assert len(result) == path_count + len(controls)
    assert FractionSpy.operations < 8 * (path_count + len(controls))


def test_neutral_sections_and_subdivision_preserve_original_bend_rings() -> None:
    path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.5)]
    baseline = PipeGeometry(_SMALL_SQUARE, path)
    neutral = PipeGeometry(_SMALL_SQUARE, path, sections=[SweepSection(0.32)])
    subdivided = PipeGeometry(_SMALL_SQUARE, path, max_segment_length=0.2)

    expected = _ring_at_center(baseline, path[1], ring_count=4)
    for geometry in (neutral, subdivided):
        actual = _ring_at_center(geometry, path[1], ring_count=4)
        for actual_vertex, expected_vertex in zip(actual, expected):
            assert actual_vertex == pytest.approx(expected_vertex)


def test_closed_neutral_sections_and_subdivision_preserve_original_rings() -> None:
    path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.2), (1.0, 1.0, 0.7), (0.0, 1.0, -0.3)]
    baseline = PipeGeometry(_SMALL_SQUARE, path, path_closed=True)
    neutral = PipeGeometry(
        _SMALL_SQUARE,
        path,
        path_closed=True,
        sections=[SweepSection(0.17)],
    )
    subdivided = PipeGeometry(
        _SMALL_SQUARE,
        path,
        path_closed=True,
        max_segment_length=0.2,
    )

    for center in path:
        expected = _ring_at_center(baseline, center, ring_count=4)
        for geometry in (neutral, subdivided):
            actual = _ring_at_center(geometry, center, ring_count=4)
            for actual_vertex, expected_vertex in zip(actual, expected):
                assert actual_vertex == pytest.approx(expected_vertex)


def test_closed_frame_distributes_residual_twist_by_arc_length() -> None:
    path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.2), (1.0, 1.0, 0.7), (0.0, 1.0, -0.3)]
    tangents = sweep_paths._compute_path_tangents(path, closed=True)
    raw_normal, raw_binormal = sweep_paths._initial_frame(tangents[0], (0.0, 0.0, 1.0))
    raw_normals = [raw_normal]
    raw_binormals = [raw_binormal]
    for index in range(1, len(path)):
        raw_normal, raw_binormal = sweep_paths._transport_path_frame_between_rings(
            raw_normals[-1],
            raw_binormals[-1],
            tangents[index - 1],
            tangents[index],
            path[index - 1],
            path[index],
        )
        raw_normals.append(raw_normal)
        raw_binormals.append(raw_binormal)

    raw_closing_normal, _raw_closing_binormal = sweep_paths._transport_path_frame_between_rings(
        raw_normals[-1],
        raw_binormals[-1],
        tangents[-1],
        tangents[0],
        path[-1],
        path[0],
    )
    residual = sweep_paths._signed_path_frame_angle(
        raw_closing_normal,
        raw_normals[0],
        tangents[0],
    )
    fractions, _total = sweep_paths._sweep_path_fractions(path, closed=True)
    normals, binormals = sweep_paths._compute_path_frames(
        path,
        tangents,
        closed=True,
        up_hint=(0.0, 0.0, 1.0),
    )

    assert abs(residual) > 0.1
    for index, fraction in enumerate(fractions):
        correction = sweep_paths._signed_path_frame_angle(
            raw_normals[index],
            normals[index],
            tangents[index],
        )
        assert correction == pytest.approx(residual * fraction)

    closing_normal, _closing_binormal = sweep_paths._transport_path_frame_between_rings(
        normals[-1],
        binormals[-1],
        tangents[-1],
        tangents[0],
        path[-1],
        path[0],
    )
    closing_residual = sweep_paths._signed_path_frame_angle(
        closing_normal,
        normals[0],
        tangents[0],
    )
    assert closing_residual == pytest.approx(residual * (1.0 - fractions[-1]))


def test_closed_pipe_requires_matching_seam_sections_and_copies_implicit_end() -> None:
    path = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.2),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]

    with pytest.raises(ValueError, match="same ordered transformed profile"):
        PipeGeometry(
            _SQUARE,
            path,
            path_closed=True,
            sections=[SweepSection(0.0), SweepSection(1.0, scale=2.0)],
        )

    geometry = PipeGeometry(
        _SMALL_SQUARE,
        path,
        path_closed=True,
        sections=[SweepSection(0.5, scale=(1.5, 0.75), offset=(0.1, 0.0))],
    )
    report = mesh_health(geometry)
    assert report.watertight
    assert "duplicate_vertices" not in report.issue_codes
    assert "boundary_edges" not in report.issue_codes
    _manifold_from_geometry(geometry, name="closed_variable_pipe")

    with pytest.raises(ValueError, match="net cyclic seam permutation/twist"):
        PipeGeometry(
            _SMALL_EQUAL_SQUARE,
            path,
            path_closed=True,
            sections=[SweepSection(0.0), SweepSection(1.0, rotation=pi / 2.0)],
        )


def test_closed_pipe_supports_internal_twist_that_returns_to_seam_orientation() -> None:
    path = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.2),
        (0.0, 1.0, 0.0),
    ]
    geometry = PipeGeometry(
        _SMALL_SQUARE,
        path,
        path_closed=True,
        sections=[
            SweepSection(0.0),
            SweepSection(0.5, rotation=pi / 2.0),
            SweepSection(1.0),
        ],
        max_segment_length=0.2,
    )

    assert mesh_health(geometry).watertight


def test_closed_pipe_supports_explicit_full_turn_without_erasing_winding() -> None:
    path = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.2),
        (0.0, 1.0, 0.0),
    ]
    geometry = PipeGeometry(
        _SMALL_SQUARE,
        path,
        path_closed=True,
        sections=[SweepSection(0.0), SweepSection(1.0, rotation=2.0 * pi)],
        max_segment_length=0.2,
    )
    untwisted = PipeGeometry(
        _SMALL_SQUARE,
        path,
        path_closed=True,
        max_segment_length=0.2,
    )

    assert mesh_health(geometry).watertight
    assert any(
        actual != pytest.approx(baseline)
        for actual, baseline in zip(geometry.vertices[4:-4], untwisted.vertices[4:-4])
    )


def test_closed_spline_builds_watertight_manifold_variable_sweep() -> None:
    geometry = sweep_profile_along_spline(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.1), (1.0, 1.0, 0.0), (0.0, 1.0, -0.1)],
        profile=_SMALL_SQUARE,
        closed_spline=True,
        samples_per_segment=6,
        sections=[SweepSection(0.5, scale=(1.2, 0.8))],
        max_segment_length=0.2,
    )

    assert mesh_health(geometry).watertight
    _manifold_from_geometry(geometry, name="closed_variable_spline")


def test_variable_sections_flow_through_fixed_axis_and_spline_sweep_apis() -> None:
    fixed = SweepGeometry(
        _SQUARE,
        [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)],
        sections=[SweepSection(0.25, scale=(2.0, 1.0), rotation=pi / 2.0)],
    )
    assert any(vertex[2] == pytest.approx(0.5) for vertex in fixed.vertices)

    spline = sweep_profile_along_spline(
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
        profile=_SQUARE,
        samples_per_segment=2,
        sections=[SweepSection(0.5, scale=2.0)],
        max_segment_length=0.1,
    )
    center_ring_index = next(
        index
        for index in range(len(spline.vertices) // 4)
        if all(
            vertex[0] == pytest.approx(0.5)
            for vertex in spline.vertices[index * 4 : (index + 1) * 4]
        )
    )
    assert _ring_span_y(spline, center_ring_index) == pytest.approx(2.0)
