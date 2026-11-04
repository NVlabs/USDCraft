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

import numpy as np
import pytest
import trimesh

import sdk._core._mesh.smooth_mesh as smooth_mesh_module
from sdk import BoxGeometry, MeshGeometry, SphereGeometry, ValidationError
from sdk._core._mesh.health import mesh_health
from sdk._core._mesh.smooth_mesh import (
    _BLEND_PROFILES,
    SnapRefused,
    _smooth_max,
    smooth_difference,
    snap_to,
    weld,
)


def _bounds(geometry: MeshGeometry) -> np.ndarray:
    return np.ptp(np.asarray(geometry.vertices, dtype=np.float64).reshape((-1, 3)), axis=0)


def _trimesh(geometry: MeshGeometry) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=geometry.vertices,
        faces=geometry.faces,
        process=False,
        validate=False,
    )


@pytest.mark.parametrize("profile", sorted(_BLEND_PROFILES))
def test_smooth_max_profiles_preserve_union_envelope(profile: str) -> None:
    first = np.linspace(-1.0, 1.0, 101)
    second = np.linspace(1.0, -1.0, 101)

    result = _smooth_max(first, second, radius=1.0, profile=profile)

    assert np.all(result >= np.maximum(first, second) - 1e-12)


def test_snap_to_closes_a_small_axis_aligned_gap_with_overlap() -> None:
    anchor = BoxGeometry((0.04, 0.04, 0.04))
    piece = BoxGeometry((0.02, 0.02, 0.02)).translate(0.04, 0.0, 0.0)

    returned = snap_to(anchor, piece, overlap=0.002, max_move=0.02, axis="x")

    assert returned is piece
    xs = [vertex[0] for vertex in piece.vertices]
    assert min(xs) == pytest.approx(0.018, abs=1e-9)
    assert max(xs) == pytest.approx(0.038, abs=1e-9)


def test_snap_to_leaves_an_already_overlapping_solid_unchanged() -> None:
    anchor = BoxGeometry((0.04, 0.04, 0.04))
    piece = BoxGeometry((0.02, 0.02, 0.02)).translate(0.025, 0.0, 0.0)
    before = list(piece.vertices)

    assert snap_to(anchor, piece, overlap=0.004, max_move=0.02) is piece
    assert piece.vertices == before

    with pytest.raises(ValueError, match="axis must be"):
        snap_to(anchor, piece, axis="diagonal")


def test_snap_to_refuses_excessive_or_misaligned_axis_motion() -> None:
    anchor = BoxGeometry((0.04, 0.04, 0.04))
    far_piece = BoxGeometry((0.02, 0.02, 0.02)).translate(0.08, 0.0, 0.0)
    with pytest.raises(SnapRefused, match="over max_move"):
        snap_to(anchor, far_piece, max_move=0.01)

    diagonal_piece = BoxGeometry((0.02, 0.02, 0.02)).translate(0.04, 0.04, 0.0)
    with pytest.raises(SnapRefused, match="not aligned"):
        snap_to(anchor, diagonal_piece, max_move=0.1, axis="x")


def test_weld_resamples_box_and_sphere_as_one_healthy_solid() -> None:
    body = BoxGeometry((0.06, 0.06, 0.06)).set_normal_crease_deg(35.0)
    boss = (
        SphereGeometry(0.03, width_segments=24, height_segments=12)
        .translate(0.045, 0.0, 0.0)
        .set_normal_crease_deg(35.0)
    )

    result = weld(body, boss, radius=0.008, tolerance=0.004)

    report = mesh_health(result)
    assert report.healthy
    assert report.watertight
    assert report.component_count == 1
    mesh = _trimesh(result)
    assert mesh.is_watertight
    assert mesh.body_count == 1
    assert len(result.faces) > len(body.faces) + len(boss.faces)
    assert result.normal_crease_deg == 35.0


def test_weld_canonicalizes_equivalent_default_normal_policies() -> None:
    first = BoxGeometry((0.04, 0.04, 0.04)).set_normal_crease_deg(60.0)
    second = BoxGeometry((0.04, 0.04, 0.04)).translate(0.035, 0.0, 0.0)

    forward = weld(first, second, radius=0.006, tolerance=0.003)
    reverse = weld(second, first, radius=0.006, tolerance=0.003)

    assert forward.normal_crease_deg is None
    assert reverse.normal_crease_deg is None


def test_weld_refuses_generated_python_mesh_above_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smooth_mesh_module, "_MAX_GENERATED_VERTICES", 8)
    first = BoxGeometry((0.04, 0.04, 0.04))
    second = BoxGeometry((0.04, 0.04, 0.04)).translate(0.035, 0.0, 0.0)

    with pytest.raises(ValueError, match="vertex/500,000-triangle safety budget"):
        weld(first, second, radius=0.006, tolerance=0.003)


def test_weld_rejects_disconnected_and_open_inputs() -> None:
    left = BoxGeometry((0.03, 0.03, 0.03)).translate(-0.05, 0.0, 0.0)
    right = BoxGeometry((0.03, 0.03, 0.03)).translate(0.05, 0.0, 0.0)
    with pytest.raises(ValueError, match="inputs are separated"):
        weld(left, right, radius=0.006, tolerance=0.003)

    open_triangle = MeshGeometry(
        vertices=[(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0)],
        faces=[(0, 1, 2)],
    )
    with pytest.raises(ValidationError, match="watertight"):
        weld(BoxGeometry((0.03, 0.03, 0.03)), open_triangle)


def test_weld_only_bridges_a_small_gap_when_explicitly_allowed() -> None:
    left = BoxGeometry((0.03, 0.03, 0.03)).translate(-0.0165, 0.0, 0.0)
    right = BoxGeometry((0.03, 0.03, 0.03)).translate(0.0165, 0.0, 0.0)

    with pytest.raises(ValueError, match="separated by 3.00 mm"):
        weld(left, right, radius=0.008, tolerance=0.002)

    bridged = weld(left, right, radius=0.008, tolerance=0.002, max_gap=0.0031)
    report = mesh_health(bridged)
    assert report.healthy
    assert report.watertight
    assert report.component_count == 1


def test_smooth_difference_makes_a_healthy_rounded_recess() -> None:
    body = BoxGeometry((0.08, 0.08, 0.08)).use_hard_normals()
    cutter = SphereGeometry(0.025, width_segments=24, height_segments=12).translate(0.035, 0.0, 0.0)

    result = smooth_difference(body, cutter, radius=0.008, tolerance=0.004)

    report = mesh_health(result)
    assert report.healthy
    assert report.watertight
    assert report.component_count == 1
    assert _trimesh(result).volume < 0.08**3
    assert _bounds(result) == pytest.approx((0.08, 0.08, 0.08), abs=0.006)
    assert result.normal_crease_deg == 0.0


def test_smooth_difference_refuses_a_gap_beyond_the_blend_radius() -> None:
    body = BoxGeometry((0.04, 0.04, 0.04))
    external = SphereGeometry(0.005, width_segments=12, height_segments=6).translate(
        0.035, 0.0, 0.0
    )

    with pytest.raises(ValueError, match="max_gap must not exceed radius"):
        smooth_difference(body, external, radius=0.002, tolerance=0.002, max_gap=0.02)


@pytest.mark.slow
def test_smooth_difference_handles_a_low_resolution_cutter_across_a_shallow_gap() -> None:
    body = BoxGeometry((0.04, 0.04, 0.04))
    cutter = SphereGeometry(0.005, width_segments=16, height_segments=8).translate(0.029, 0.0, 0.0)

    result = smooth_difference(
        body,
        cutter,
        radius=0.005,
        tolerance=0.001,
        max_gap=0.005,
    )

    report = mesh_health(result)
    assert report.healthy
    assert report.watertight
    assert report.component_count == 1


def test_smooth_difference_preserves_a_valid_enclosed_cavity() -> None:
    body = BoxGeometry((0.08, 0.08, 0.08))
    cutter = SphereGeometry(0.015, width_segments=20, height_segments=10)

    result = smooth_difference(body, cutter, radius=0.006, tolerance=0.004)

    report = mesh_health(result)
    assert report.healthy
    assert report.watertight
    assert report.component_count == 1
    assert report.surface_component_count == 2  # outer boundary plus inward cavity boundary
    surface_shells = _trimesh(result).split(only_watertight=True)
    assert sum(shell.volume > 0.0 for shell in surface_shells) == 1
    assert sum(shell.volume < 0.0 for shell in surface_shells) == 1


@pytest.mark.slow
def test_weld_accepts_a_single_solid_with_an_enclosed_cavity() -> None:
    hollow_body = smooth_difference(
        BoxGeometry((0.08, 0.08, 0.08)),
        SphereGeometry(0.015, width_segments=20, height_segments=10),
        radius=0.006,
        tolerance=0.004,
    )
    boss = SphereGeometry(0.02, width_segments=20, height_segments=10).translate(0.045, 0.0, 0.0)

    result = weld(hollow_body, boss, radius=0.006, tolerance=0.004)

    report = mesh_health(result)
    assert report.healthy
    assert report.watertight
    assert report.component_count == 1
    assert report.surface_component_count == 2


def test_weld_trim_removes_an_internal_stub_without_splitting_the_solid() -> None:
    left = BoxGeometry((0.05, 0.05, 0.05)).translate(-0.01, 0.0, 0.0)
    right = BoxGeometry((0.05, 0.05, 0.05)).translate(0.01, 0.0, 0.0)
    cavity = SphereGeometry(0.01, width_segments=16, height_segments=8)

    result = weld(left, right, radius=0.006, tolerance=0.003, trim=cavity)

    report = mesh_health(result)
    assert report.healthy
    assert report.watertight
    assert report.component_count == 1
    assert report.surface_component_count == 2


def test_weld_refuses_a_field_grid_above_the_resource_limit() -> None:
    left = BoxGeometry((1.0, 1.0, 1.0)).translate(-0.1, 0.0, 0.0)
    right = BoxGeometry((1.0, 1.0, 1.0)).translate(0.1, 0.0, 0.0)

    with pytest.raises(ValueError, match="field samples.*safety limit"):
        weld(left, right, radius=0.01, tolerance=0.00001)
