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

from math import pi

import pytest

from sdk import (
    BoxGeometry,
    boolean_union,
    mesh_from_sdf,
    sdf_box,
    sdf_capsule,
    sdf_cylinder,
    sdf_ellipsoid,
    sdf_sphere,
    sdf_torus,
)


def _bounds(geometry):
    xs = [v[0] for v in geometry.vertices]
    ys = [v[1] for v in geometry.vertices]
    zs = [v[2] for v in geometry.vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def test_sphere_meshes_to_expected_size() -> None:
    geometry = mesh_from_sdf(sdf_sphere(0.05), edge_length=0.004)

    assert len(geometry.faces) > 100
    mins, maxs = _bounds(geometry)
    assert maxs[0] - mins[0] == pytest.approx(0.1, abs=0.005)
    assert maxs[2] - mins[2] == pytest.approx(0.1, abs=0.005)


def test_primitive_distances_have_correct_signs() -> None:
    cases = [
        (sdf_sphere(0.05), (0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        (sdf_box((0.1, 0.1, 0.1)), (0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        (sdf_cylinder(0.04, 0.1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.1)),
        (sdf_capsule(0.1, 0.02), (0.0, 0.0, 0.0), (0.05, 0.0, 0.0)),
        (sdf_torus(0.05, 0.01), (0.05, 0.0, 0.0), (0.0, 0.0, 0.0)),
        (sdf_ellipsoid((0.05, 0.04, 0.03)), (0.0, 0.0, 0.0), (0.06, 0.0, 0.0)),
    ]
    for field, inside, outside in cases:
        assert field(*inside) < 0.0
        assert field(*outside) > 0.0


def test_smooth_union_bridges_the_seam() -> None:
    a = sdf_sphere(0.03)
    b = sdf_sphere(0.03).translate(0.05, 0.0, 0.0)
    hard = a.union(b)
    soft = a.smooth_union(b, 0.02)

    # At the seam midpoint the smooth blend is fuller (more negative).
    seam = (0.025, 0.0, 0.025)
    assert soft(*seam) < hard(*seam)

    geometry = mesh_from_sdf(soft, edge_length=0.003)
    assert len(geometry.faces) > 100


def test_difference_and_shell() -> None:
    hollow = sdf_sphere(0.05).shell(0.006)
    assert hollow(0.0, 0.0, 0.0) > 0.0  # core removed
    assert hollow(0.05, 0.0, 0.0) < 0.0  # wall is solid

    cut = sdf_box((0.1, 0.1, 0.1)).difference(sdf_sphere(0.04))
    assert cut(0.0, 0.0, 0.0) > 0.0
    assert cut(0.049, 0.049, 0.049) < 0.0


def test_transforms_move_field_and_bounds() -> None:
    moved = sdf_sphere(0.02).translate(0.1, 0.0, 0.0)
    assert moved(0.1, 0.0, 0.0) < 0.0
    assert moved(0.0, 0.0, 0.0) > 0.0
    assert moved.bounds[0][0] == pytest.approx(0.08)

    scaled = sdf_sphere(0.02).scale(2.0)
    assert scaled(0.039, 0.0, 0.0) < 0.0
    assert scaled(0.041, 0.0, 0.0) > 0.0

    rotated = sdf_cylinder(0.01, 0.1).rotate_y(pi / 2.0)
    assert rotated(0.04, 0.0, 0.0) < 0.0  # axis now along X
    assert rotated(0.0, 0.0, 0.04) > 0.0


def test_sdf_mesh_interoperates_with_booleans() -> None:
    geometry = mesh_from_sdf(sdf_sphere(0.03), edge_length=0.004)
    merged = boolean_union(geometry, BoxGeometry((0.02, 0.02, 0.02)))

    assert len(merged.faces) > 0


def test_mesh_from_sdf_rejects_empty_field() -> None:
    nothing = sdf_sphere(0.01).intersection(sdf_sphere(0.01).translate(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="no solid"):
        mesh_from_sdf(nothing, edge_length=0.005)


def test_parameter_validation() -> None:
    with pytest.raises(ValueError, match="radius"):
        sdf_sphere(0.0)
    with pytest.raises(ValueError, match="radius"):
        sdf_box((0.1, 0.1, 0.1), radius=0.06)
    with pytest.raises(ValueError, match="k"):
        sdf_sphere(0.01).smooth_union(sdf_sphere(0.01), 0.0)
    with pytest.raises(ValueError, match="edge_length"):
        mesh_from_sdf(sdf_sphere(0.01), edge_length=0.0)
