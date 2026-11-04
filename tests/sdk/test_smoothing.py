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

import pytest

from sdk import (
    BoxGeometry,
    SphereGeometry,
    boolean_difference,
    displace_geometry,
    simplify_geometry,
    smooth_geometry,
    subdivide_geometry,
    taubin_smooth_geometry,
)


def _bounds(geometry) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xs = [v[0] for v in geometry.vertices]
    ys = [v[1] for v in geometry.vertices]
    zs = [v[2] for v in geometry.vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def test_subdivide_quadruples_faces_and_keeps_shape() -> None:
    box = BoxGeometry((0.1, 0.2, 0.3))
    subdivided = subdivide_geometry(box, levels=2)

    assert len(subdivided.faces) == len(box.faces) * 16
    assert _bounds(subdivided) == _bounds(box)


def test_subdivide_rejects_bad_levels() -> None:
    with pytest.raises(ValueError, match="levels"):
        subdivide_geometry(BoxGeometry((0.1, 0.1, 0.1)), levels=0)


def test_subdivide_does_not_mutate_input() -> None:
    box = BoxGeometry((0.1, 0.1, 0.1))
    face_count = len(box.faces)
    subdivide_geometry(box, levels=1)

    assert len(box.faces) == face_count


def test_smooth_geometry_curves_a_box() -> None:
    box = BoxGeometry((0.1, 0.1, 0.1))
    smoothed = smooth_geometry(box, min_sharp_angle=95.0, subdivisions=3)

    assert len(smoothed.faces) > len(box.faces)
    # Smoothing interpolates curved patches: some vertices leave the original
    # box surface (manifold's smooth_out bulges edges outward).
    off_surface = [
        v for v in smoothed.vertices if abs(max(abs(v[0]), abs(v[1]), abs(v[2])) - 0.05) > 1e-4
    ]
    assert off_surface


def test_smooth_geometry_keeps_sharp_edges_when_requested() -> None:
    box = BoxGeometry((0.1, 0.1, 0.1))
    smoothed = smooth_geometry(box, min_sharp_angle=45.0, subdivisions=2)

    # 90-degree box edges stay sharp, so the bounding box is unchanged.
    mins, maxs = _bounds(smoothed)
    expected_mins, expected_maxs = _bounds(box)
    assert mins == pytest.approx(expected_mins)
    assert maxs == pytest.approx(expected_maxs)


def test_smooth_geometry_accepts_boolean_results() -> None:
    outer = BoxGeometry((0.1, 0.1, 0.1))
    inner = BoxGeometry((0.05, 0.05, 0.2))
    shell = boolean_difference(outer, inner)

    smoothed = smooth_geometry(shell, min_sharp_angle=95.0, subdivisions=2)
    assert len(smoothed.faces) > 0


def test_taubin_smooth_preserves_topology_and_resists_shrink() -> None:
    sphere = SphereGeometry(0.05)
    smoothed = taubin_smooth_geometry(sphere, iterations=20)

    assert len(smoothed.vertices) == len(sphere.vertices)
    assert len(smoothed.faces) == len(sphere.faces)
    (_, _, min_z), (_, _, max_z) = _bounds(smoothed)
    assert (max_z - min_z) > 0.08  # far less than 20% shrink on a 0.1 diameter


def test_taubin_smooth_validates_factors() -> None:
    sphere = SphereGeometry(0.05)
    with pytest.raises(ValueError, match="lambda_factor"):
        taubin_smooth_geometry(sphere, lambda_factor=0.0)
    with pytest.raises(ValueError, match="mu_factor"):
        taubin_smooth_geometry(sphere, mu_factor=0.1)


def test_displace_is_deterministic_and_seed_sensitive() -> None:
    base = subdivide_geometry(BoxGeometry((0.1, 0.1, 0.1)), levels=2)

    first = displace_geometry(base, amplitude=0.005, scale=0.03, seed=3)
    second = displace_geometry(base, amplitude=0.005, scale=0.03, seed=3)
    other_seed = displace_geometry(base, amplitude=0.005, scale=0.03, seed=4)

    assert first.vertices == second.vertices
    assert first.vertices != other_seed.vertices


def test_displace_respects_amplitude_bound() -> None:
    base = subdivide_geometry(BoxGeometry((0.1, 0.1, 0.1)), levels=2)
    displaced = displace_geometry(base, amplitude=0.002, scale=0.03, seed=1)

    max_offset = max(
        abs(a - b)
        for original, moved in zip(base.vertices, displaced.vertices)
        for a, b in zip(original, moved)
    )
    assert 0.0 < max_offset <= 0.002 + 1e-12


def test_displace_validates_parameters() -> None:
    base = BoxGeometry((0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="scale"):
        displace_geometry(base, amplitude=0.001, scale=0.0)
    with pytest.raises(ValueError, match="octaves"):
        displace_geometry(base, amplitude=0.001, octaves=0)


def test_simplify_collapses_redundant_triangles() -> None:
    dense = subdivide_geometry(BoxGeometry((0.1, 0.1, 0.1)), levels=2)
    simplified = simplify_geometry(dense, tolerance=0.0001)

    assert 0 < len(simplified.faces) < len(dense.faces)


def test_simplify_validates_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        simplify_geometry(BoxGeometry((0.1, 0.1, 0.1)), tolerance=0.0)


def test_single_input_mesh_rebuilds_preserve_normal_crease_policy() -> None:
    base = SphereGeometry(0.05, width_segments=12, height_segments=8).set_normal_crease_deg(35.0)

    results = (
        subdivide_geometry(base, levels=1),
        smooth_geometry(base, subdivisions=1),
        taubin_smooth_geometry(base, iterations=1),
        displace_geometry(base, amplitude=0.001, scale=0.03, seed=1),
        simplify_geometry(base, tolerance=0.0001),
    )

    assert all(result.normal_crease_deg == 35.0 for result in results)
