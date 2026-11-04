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

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh

from usdcraft.reconstruction import axial
from usdcraft.reconstruction.axial import (
    AGENT_CORE_BYTE_BUDGET,
    STACK_AXIS,
    STACK_RESOLUTION_CANDIDATES,
    SURFACE_BIT,
    AxialObservation,
    build_axial_observation,
    format_voxel_artifact,
)
from usdcraft.reconstruction.diagnostics import parse_axial_stack
from usdcraft.reconstruction.source import (
    MeshTopology,
    NormalizedSourceComponent,
    NormalizedSourceScene,
    ResolvedSourceFrame,
)


def _topology(
    classification: str,
    *,
    vertex_count: int,
    triangle_count: int,
    closed: bool,
) -> MeshTopology:
    boundary_edges = 0 if closed else max(1, triangle_count)
    return MeshTopology(
        classification=classification,  # type: ignore[arg-type]
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        exact_welded_vertex_count=vertex_count,
        boundary_edge_count=boundary_edges,
        non_manifold_edge_count=0,
        inconsistent_winding_edge_count=0,
        degenerate_triangle_count=0,
        duplicate_triangle_count=0,
        oriented_surface_integral_m3=1.0 if closed else 0.0,
        surface_closed=closed,
    )


def _mesh_component(
    component_id: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    closed: bool,
) -> NormalizedSourceComponent:
    return NormalizedSourceComponent(
        component_id=component_id,
        kind="mesh",
        provenance=f"test:{component_id}",
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        normals=None,
        colors_rgba=None,
        topology=_topology(
            "closed_orientable_surface" if closed else "open_surface",
            vertex_count=len(vertices),
            triangle_count=len(faces),
            closed=closed,
        ),
    )


def _point_component(component_id: str, points: np.ndarray) -> NormalizedSourceComponent:
    return NormalizedSourceComponent(
        component_id=component_id,
        kind="points",
        provenance=f"test:{component_id}",
        vertices=np.asarray(points, dtype=np.float64),
        faces=None,
        normals=None,
        colors_rgba=None,
        topology=_topology(
            "point_samples",
            vertex_count=len(points),
            triangle_count=0,
            closed=False,
        ),
    )


def _scene(*components: NormalizedSourceComponent) -> NormalizedSourceScene:
    vertices = np.concatenate([component.vertices for component in components], axis=0)
    return NormalizedSourceScene(
        source_path=Path("/tmp/normalized-test.usdc"),
        source_format="usd",
        frame=ResolvedSourceFrame(
            up_axis="+Z",
            up_axis_source="explicit",
            meters_per_unit=1.0,
            meters_per_unit_source="explicit",
            front_axis="-Y",
            front_axis_source="explicit",
            source_to_normalized=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        components=tuple(components),
        bounds_min_xyz=vertices.min(axis=0),
        bounds_max_xyz=vertices.max(axis=0),
        source_dependencies=(),
        warnings=(),
    )


def _box(component_id: str = "box") -> NormalizedSourceComponent:
    box = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
    return _mesh_component(
        component_id,
        np.asarray(box.vertices),
        np.asarray(box.faces),
        closed=True,
    )


def _plane(axis: int, component_id: str = "plane") -> NormalizedSourceComponent:
    vertices = np.asarray(
        [
            (-0.45, -0.35, 0.0),
            (0.45, -0.35, 0.0),
            (0.45, 0.35, 0.0),
            (-0.45, 0.35, 0.0),
        ],
        dtype=np.float64,
    )
    if axis == 0:
        vertices = vertices[:, [2, 0, 1]]
    elif axis == 1:
        vertices = vertices[:, [0, 2, 1]]
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    return _mesh_component(component_id, vertices, faces, closed=False)


def _decode_runs(runs: list[list[int]], size: int) -> np.ndarray:
    decoded = np.zeros(size, dtype=bool)
    for start, length in runs:
        decoded[start : start + length] = True
    return decoded


@pytest.fixture(scope="module")
def box_observation() -> AxialObservation:
    """One default observation of the plain box, shared by the tests that only read it.

    The observation is a frozen dataclass; its arrays are made read-only so a test that tried
    to mutate the shared value would fail instead of leaking state into later tests.
    """
    observation = build_axial_observation(_scene(_box()))
    for array in (observation.micro_volume, observation.stack.volume):
        array.flags.writeable = False
    return observation


def test_default_observation_rejects_budgets_above_the_32_kib_hard_maximum() -> None:
    scene = _scene(_box())

    with pytest.raises(ValueError, match="cannot exceed the 32 KiB hard maximum"):
        build_axial_observation(scene, core_byte_budget=AGENT_CORE_BYTE_BUDGET + 1)
    with pytest.raises(ValueError, match="palette budget cannot exceed"):
        build_axial_observation(scene, palette_glyph_budget=AGENT_CORE_BYTE_BUDGET + 1)


def test_closed_mesh_is_surface_only_and_center_cell_remains_unknown(
    box_observation: AxialObservation,
) -> None:
    observation = box_observation

    assert [item.evidence_kind for item in observation.classifications] == ["surface_hit_band"]
    assert observation.classifications[0].reason == (
        "reconstruction_mesh_surface_only_interior_unknown"
    )
    assert not np.any(observation.micro_volume & ~SURFACE_BIT)
    assert np.any(observation.micro_volume & SURFACE_BIT)
    center = tuple(int(value // 2) for value in observation.micro_frame.dimensions)
    assert observation.micro_volume[center] == 0
    assert observation.agent_core.startswith("AXIAL\n")
    assert "cells: .=unknown ~=sampled_source_surface" in observation.agent_core
    assert "interior: UNKNOWN" in observation.agent_core
    assert observation.semantic_front_resolved is True
    assert "semantic_front: -Y" in observation.agent_core
    assert observation.agent_core_byte_count <= AGENT_CORE_BYTE_BUDGET
    assert observation.stack.stack_axis == STACK_AXIS
    assert "stack_policy: fixed_axis=Z planes=XY candidates=32,24,16" in observation.agent_core
    assert f"[SLICE_STACK{observation.stack.longest_axis_cells} axis=Z " in observation.agent_core
    assert "slice_order=-Z->+Z rows=+Y->-Y cols=-X->+X]" in observation.agent_core
    assert "[MICRO_SLICE_BOUNDS_Z192" in observation.agent_core
    assert "MICRO_ENVELOPE_Z192" not in observation.agent_core


@pytest.mark.slow
def test_unresolved_semantic_front_is_not_inferred_from_front_projection() -> None:
    scene = _scene(_box())
    unresolved = replace(
        scene,
        frame=replace(scene.frame, front_axis=None, front_axis_source="unresolved"),
    )

    observation = build_axial_observation(unresolved)

    assert observation.semantic_front_resolved is False
    assert "semantic_front: unresolved" in observation.agent_core


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_each_axial_open_plane_survives_as_surface_hit_bands(axis: int) -> None:
    observation = build_axial_observation(_scene(_plane(axis)))

    assert observation.classifications[0].evidence_kind == "surface_hit_band"
    assert observation.classifications[0].reason == (
        "reconstruction_mesh_surface_only_interior_unknown"
    )
    assert not np.any(observation.micro_volume & ~SURFACE_BIT)
    assert np.any(observation.micro_volume & SURFACE_BIT)
    assert "~=sampled_source_surface" in observation.agent_core
    assert "global_free_space=false" in observation.agent_core
    assert "[VOID" not in observation.agent_core
    assert observation.stack.stack_axis == STACK_AXIS
    assert observation.stack.row_axis == 1
    assert observation.stack.col_axis == 0
    assert f"[SLICE_STACK{observation.stack.longest_axis_cells} axis=Z " in observation.agent_core


@pytest.mark.slow
def test_closed_and_open_meshes_share_one_surface_only_channel() -> None:
    observation = build_axial_observation(_scene(_box(), _plane(2)))

    assert {item.evidence_kind for item in observation.classifications} == {"surface_hit_band"}
    assert not np.any(observation.micro_volume & ~SURFACE_BIT)
    assert np.any(observation.micro_volume & SURFACE_BIT)
    assert "global_free_space=false" in observation.agent_core


@pytest.mark.slow
def test_source_topology_can_forbid_solid_inference_for_closed_looking_faces() -> None:
    box = _box()
    assert box.faces is not None
    declared_open = replace(
        box,
        topology=_topology(
            "open_surface",
            vertex_count=len(box.vertices),
            triangle_count=len(box.faces),
            closed=False,
        ),
    )

    observation = build_axial_observation(_scene(declared_open))

    assert observation.classifications[0].evidence_kind == "surface_hit_band"
    assert observation.classifications[0].reason == (
        "reconstruction_mesh_surface_only_interior_unknown"
    )
    assert not np.any(observation.micro_volume & ~SURFACE_BIT)


def test_mesh_cannot_claim_solid_authority_from_watertightness_or_a_forged_flag() -> None:
    forged = replace(_box(), solid_authority="watertight_mesh")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="surface_only geometry, unknown interior"):
        build_axial_observation(_scene(forged))


@pytest.mark.slow
def test_topology_and_odd_ray_parity_do_not_change_surface_only_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = axial._cast_mesh_hits

    def odd_front(component, frame, spec):  # type: ignore[no-untyped-def]
        hits = original(component, frame, spec)
        if spec.name == "FRONT":
            hits = dict(hits)
            hits[(0, 0)] = (0.0,)
        return hits

    monkeypatch.setattr(axial, "_cast_mesh_hits", odd_front)
    observation = build_axial_observation(_scene(_box()))

    classification = observation.classifications[0]
    assert classification.evidence_kind == "surface_hit_band"
    assert classification.reason == "reconstruction_mesh_surface_only_interior_unknown"
    assert not np.any(observation.micro_volume & ~SURFACE_BIT)
    assert np.any(observation.micro_volume & SURFACE_BIT)


def test_point_samples_are_exact_surface_evidence_and_order_independent() -> None:
    points = np.asarray(
        (
            (-0.5, -0.2, 0.0),
            (0.5, -0.2, 0.0),
            (0.5, 0.2, 0.8),
            (-0.5, 0.2, 0.8),
        )
    )
    first = build_axial_observation(_scene(_point_component("scan", points)))
    second = build_axial_observation(_scene(_point_component("scan", points[::-1])))

    assert first.classifications[0].reason == "point_samples"
    assert not np.any(first.micro_volume & ~SURFACE_BIT)
    assert np.count_nonzero(first.micro_volume & SURFACE_BIT) == len(points)
    assert first.agent_core == second.agent_core
    assert np.array_equal(first.stack.volume, second.stack.volume)


@pytest.mark.slow
def test_complete_core_budget_can_fall_back_to_a_lower_stack_resolution(
    box_observation: AxialObservation,
) -> None:
    scene = _scene(_box())
    detailed = box_observation
    assert detailed.stack.longest_axis_cells == 32

    constrained = build_axial_observation(
        scene,
        core_byte_budget=detailed.agent_core_byte_count - 1,
    )

    assert constrained.agent_core_byte_count < detailed.agent_core_byte_count
    assert constrained.stack.longest_axis_cells < detailed.stack.longest_axis_cells
    assert constrained.micro_slice_bounds.stack_longest_axis_cells == (
        constrained.stack.longest_axis_cells
    )
    assert detailed.micro_slice_bounds.stack_longest_axis_cells == detailed.stack.longest_axis_cells
    assert constrained.micro_slice_bounds != detailed.micro_slice_bounds
    assert constrained.agent_core_byte_count <= detailed.agent_core_byte_count - 1


def test_the_finest_stack_whose_core_fits_the_ceiling_is_selected(
    box_observation: AxialObservation,
) -> None:
    observation = box_observation

    assert observation.stack.longest_axis_cells == 32
    assert observation.selection_core_byte_budget == AGENT_CORE_BYTE_BUDGET
    assert observation.selection_reason == "finest_within_core_ceiling"


@pytest.mark.slow
def test_face_derived_sections_ride_above_the_core_ceiling_and_never_move_the_stack(
    monkeypatch: pytest.MonkeyPatch,
    box_observation: AxialObservation,
) -> None:
    from usdcraft.reconstruction.face_openings import FaceOpening, FaceOpeningSummary

    scene = _scene(_box())
    plain = box_observation
    # a core that just fits the ceiling at stack 32
    tight_budget = plain.agent_core_byte_count + 8
    tight = build_axial_observation(scene, core_byte_budget=tight_budget)
    assert tight.stack.longest_axis_cells == 32

    def many_openings(components, *, seed=3):  # type: ignore[no-untyped-def]
        openings = tuple(
            FaceOpening(
                "+Z", "opening", (0.01 * i, 0.0), (0.02, 0.03), 0.04, 0.05, 0.0, False, 0, "rim"
            )
            for i in range(120)
        )
        from usdcraft.reconstruction.face_openings import format_face_openings

        summary = FaceOpeningSummary(openings, 0.002)
        lines = format_face_openings(summary)
        return FaceOpeningSummary(
            openings, 0.002, sum(len(line.encode("utf-8")) + 1 for line in lines)
        )

    monkeypatch.setattr(axial, "detect_face_openings", many_openings)
    padded = build_axial_observation(scene, core_byte_budget=tight_budget)

    # ten kilobytes of FACE_OPENINGS did not change the stack, and the total text carries them
    assert padded.stack.longest_axis_cells == 32
    assert padded.agent_core_byte_count - tight.agent_core_byte_count > 8000
    assert padded.face_openings.section_bytes > 8000
    assert "[FACE_OPENINGS" in padded.agent_core

    # the sections still have a fuse of their own above the core ceiling
    def far_too_many_openings(components, *, seed=3):  # type: ignore[no-untyped-def]
        summary = many_openings(components, seed=seed)
        openings = summary.openings * 8
        return FaceOpeningSummary(openings, 0.002, summary.section_bytes * 8)

    monkeypatch.setattr(axial, "detect_face_openings", far_too_many_openings)
    with pytest.raises(ValueError, match="exceed their allowance above the core ceiling"):
        build_axial_observation(scene)


def test_micro_slice_bounds_use_world_mapped_z_slices_and_8_connected_islands() -> None:
    micro_volume = np.zeros((8, 8, 8), dtype=np.uint8)
    for micro_z in (0, 1):
        micro_volume[1, 1, micro_z] = SURFACE_BIT
        micro_volume[2, 2, micro_z] = SURFACE_BIT
        micro_volume[6, 6, micro_z] = SURFACE_BIT
    micro_volume[4, 4:6, 3] = SURFACE_BIT
    micro_frame = axial.GridFrame(
        pitch=1.0,
        dimensions=np.asarray(micro_volume.shape, dtype=np.int64),
        bounds_min=np.asarray((10.0, 20.0, 30.0)),
        bounds_max=np.asarray((18.0, 28.0, 38.0)),
    )
    # The shifted stack origin deliberately maps micro Z cells 0 and 1 to different selected
    # stack slices. Index-ratio mapping would put both in slice 0 and fail this test.
    stack_frame = axial.GridFrame(
        pitch=2.0,
        dimensions=np.asarray((4, 4, 4), dtype=np.int64),
        bounds_min=np.asarray((9.0, 19.0, 29.0)),
        bounds_max=np.asarray((17.0, 27.0, 37.0)),
    )
    stack = axial._best_stack(
        np.zeros((4, 4, 4), dtype=np.uint8),
        stack_frame,
        longest_axis_cells=4,
    )
    assert stack is not None

    summary = axial._build_micro_slice_bounds_summary(
        micro_volume,
        micro_frame,
        stack,
    )

    assert summary.stack_slice_count == 4
    assert summary.encoded_slice_count == 3
    assert summary.island_count == 5
    assert [
        (
            run.stack_z_start,
            run.stack_z_end,
            run.micro_z_start,
            run.micro_z_end,
            tuple(
                (island.x_start, island.x_end, island.y_start, island.y_end)
                for island in run.islands
            ),
        )
        for run in summary.runs
    ] == [
        (0, 2, 0, 2, ((1, 3, 1, 3), (6, 7, 6, 7))),
        (2, 3, 3, 4, ((4, 5, 4, 6),)),
    ]
    assert summary.selection == "complete"
    assert summary.contoured_island_count == 0


def test_micro_island_contours_emit_only_where_the_bbox_is_not_faithful() -> None:
    micro_volume = np.zeros((32, 32, 4), dtype=np.uint8)
    # Filled rectangle: its exact outline equals the bbox, so no contour may be emitted.
    micro_volume[2:10, 2:10, 0] = SURFACE_BIT
    # Filled right triangle: half the bbox area is empty, so a material contour must appear.
    for x in range(16, 32):
        micro_volume[x, 2 : 2 + (x - 15), 1] = SURFACE_BIT
    micro_frame = axial.GridFrame(
        pitch=1.0,
        dimensions=np.asarray(micro_volume.shape, dtype=np.int64),
        bounds_min=np.asarray((0.0, 0.0, 0.0)),
        bounds_max=np.asarray((32.0, 32.0, 4.0)),
    )
    stack_frame = axial.GridFrame(
        pitch=1.0,
        dimensions=np.asarray((4, 4, 4), dtype=np.int64),
        bounds_min=np.asarray((0.0, 0.0, 0.0)),
        bounds_max=np.asarray((4.0, 4.0, 4.0)),
    )
    stack = axial._best_stack(
        np.zeros((4, 4, 4), dtype=np.uint8), stack_frame, longest_axis_cells=4
    )
    assert stack is not None
    summary = axial._build_micro_slice_bounds_summary(micro_volume, micro_frame, stack)

    assert summary.selection == "complete"
    assert summary.contoured_island_count == 1
    rectangle = summary.runs[0].islands[0]
    triangle = summary.runs[1].islands[0]
    assert rectangle.contour == ()
    assert len(triangle.contour) >= 3
    for x, y in triangle.contour:
        assert triangle.x_start <= x <= triangle.x_end
        assert triangle.y_start <= y <= triangle.y_end
    assert axial._polygon_area(triangle.contour) < triangle.bbox_cells * (
        1.0 - axial.MICRO_CONTOUR_MATERIALITY
    )


def test_micro_island_contour_is_vertex_bounded() -> None:
    micro_volume = np.zeros((24, 24, 4), dtype=np.uint8)
    for x in range(2, 22):
        micro_volume[x, 2 : 2 + max(1, (x * 7) % 19), 0] = SURFACE_BIT
    micro_frame = axial.GridFrame(
        pitch=1.0,
        dimensions=np.asarray(micro_volume.shape, dtype=np.int64),
        bounds_min=np.asarray((0.0, 0.0, 0.0)),
        bounds_max=np.asarray((24.0, 24.0, 4.0)),
    )
    stack_frame = axial.GridFrame(
        pitch=1.0,
        dimensions=np.asarray((4, 4, 4), dtype=np.int64),
        bounds_min=np.asarray((0.0, 0.0, 0.0)),
        bounds_max=np.asarray((4.0, 4.0, 4.0)),
    )
    stack = axial._best_stack(
        np.zeros((4, 4, 4), dtype=np.uint8), stack_frame, longest_axis_cells=4
    )
    assert stack is not None
    summary = axial._build_micro_slice_bounds_summary(micro_volume, micro_frame, stack)
    island = summary.runs[0].islands[0]
    assert island.contour
    assert len(island.contour) <= axial.MICRO_CONTOUR_MAX_VERTICES


@pytest.mark.slow
def test_observation_level_contours_round_trip_through_the_diagnostic_parser() -> None:
    # A vertical diagonal wall: every horizontal slab supports a thin staircase band whose
    # bbox wildly overstates the sampled support, so a material contour must be emitted.
    wall_vertices = np.asarray(
        [
            (-0.5, -0.4, 0.0),
            (0.5, 0.4, 0.0),
            (0.5, 0.4, 0.3),
            (-0.5, -0.4, 0.3),
        ],
        dtype=np.float64,
    )
    wall = _mesh_component(
        "wall",
        wall_vertices,
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        closed=False,
    )
    observation = build_axial_observation(_scene(wall))
    assert observation.micro_slice_bounds.contoured_island_count >= 1
    assert "(f=." in observation.agent_core
    assert ";p=" in observation.agent_core
    parsed = parse_axial_stack(observation.agent_core)
    assert np.array_equal(parsed.volume, observation.stack.volume)


@pytest.mark.slow
def test_micro_slice_bounds_section_is_never_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(axial, "MICRO_SLICE_BOUNDS_BYTE_BUDGET", 0)

    observation = build_axial_observation(_scene(_box()))
    summary = observation.micro_slice_bounds

    assert summary.selection in ("complete", "boxes_only")
    assert summary.runs != ()
    assert summary.section_bytes > 0
    assert summary.island_count > 0
    assert sum(len(run.islands) for run in summary.runs) > 0
    assert "[MICRO_SLICE_BOUNDS_Z192 parent=" in observation.agent_core
    assert "omitted_due_budget" not in observation.agent_core
    parsed = parse_axial_stack(observation.agent_core)
    assert np.array_equal(parsed.volume, observation.stack.volume)


@pytest.mark.slow
def test_complete_core_fails_after_trying_every_stack_resolution() -> None:
    with pytest.raises(
        ValueError, match="no AXIAL slice stack keeps the stack-bound core"
    ) as caught:
        build_axial_observation(_scene(_box()), core_byte_budget=1)

    # The error lists the cost of every candidate resolution it tried.
    for resolution in STACK_RESOLUTION_CANDIDATES:
        assert f"({resolution}, " in str(caught.value)


def _rod(component_id: str, start, end, radius: float = 0.02):
    """A thin cylinder from start to end as a mesh component."""
    import trimesh.creation

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    cylinder = trimesh.creation.cylinder(
        radius=radius, segment=np.vstack([start, end]), sections=12
    )
    return _mesh_component(
        component_id, np.asarray(cylinder.vertices), np.asarray(cylinder.faces), closed=True
    )


@pytest.mark.slow
def test_member_track_follows_a_diagonal_rod() -> None:
    scene = _scene(
        _rod("rod", (-0.30, -0.10, -0.50), (0.30, 0.10, 0.50)),
        _box("anchor"),
    )
    observation = build_axial_observation(scene)
    summary = observation.member_tracks
    assert summary.tracks, observation.agent_core
    track = max(summary.tracks, key=lambda t: t.slab_count)
    pitch = observation.micro_frame.pitch
    sx = observation.micro_frame.bounds_min[0] + (track.start_uv[0] + 0.5) * pitch
    ex = observation.micro_frame.bounds_min[0] + (track.end_uv[0] + 0.5) * pitch
    # The centerline drifts monotonically across the rod's x span.
    assert ex - sx > 0.3
    assert "[MEMBER_TRACKS192" in observation.agent_core
    assert "t z[" in observation.agent_core
    parsed = parse_axial_stack(observation.agent_core)
    assert np.array_equal(parsed.volume, observation.stack.volume)


@pytest.mark.slow
def test_member_track_splits_at_a_joint() -> None:
    scene = _scene(
        _rod("lower", (0.0, 0.0, -0.60), (0.0, 0.0, 0.0)),
        _rod("upper", (0.0, 0.0, 0.0), (0.55, 0.0, 0.35)),
        _box("anchor"),
    )
    observation = build_axial_observation(scene)
    tracks = observation.member_tracks.tracks
    # The vertical segment and the slanted segment come out as separate tracks.
    assert len(tracks) >= 2, observation.agent_core
    spans = sorted((t.z_start, t.z_end) for t in tracks)
    assert spans[0][1] <= spans[-1][0] + observation.member_tracks.tracks[0].slab_count


@pytest.mark.slow
def test_parallel_rods_never_cross_link() -> None:
    scene = _scene(
        _rod("left", (-0.30, 0.0, -0.50), (-0.30, 0.0, 0.50)),
        _rod("right", (0.30, 0.0, -0.50), (0.30, 0.0, 0.50)),
    )
    observation = build_axial_observation(scene)
    tracks = observation.member_tracks.tracks
    assert len(tracks) == 2, observation.agent_core
    for track in tracks:
        # Each track stays on its own rod: start and end share the same x side.
        assert (track.start_uv[0] - 96) * (track.end_uv[0] - 96) > 0


@pytest.mark.slow
def test_member_tracks_cap_keeps_the_longest() -> None:
    rods = [
        _rod(f"rod_{i}", (-0.9 + 0.09 * i, 0.0, -0.5), (-0.9 + 0.09 * i, 0.0, 0.5))
        for i in range(20)
    ]
    observation = build_axial_observation(_scene(*rods))
    summary = observation.member_tracks
    assert summary.detected_track_count >= 20
    assert len(summary.tracks) == axial.TRACK_MAX_SHOWN


def test_missing_component_provenance_does_not_block_mixed_point_evidence() -> None:
    first = replace(
        _point_component("first", np.asarray(((-0.5, 0.0, 0.0), (0.0, 0.0, 0.5)))),
        provenance="",
    )
    second = replace(
        _point_component("second", np.asarray(((0.0, 0.0, 0.5), (0.5, 0.0, 1.0)))),
        provenance="",
    )

    observation = build_axial_observation(_scene(first, second))

    assert "~=sampled_source_surface" in observation.agent_core


@pytest.mark.slow
def test_voxel_surface_runs_round_trip_without_an_interior_channel() -> None:
    observation = build_axial_observation(_scene(_box(), _plane(2)))
    payload = format_voxel_artifact(observation)

    assert payload["schema"] == "axial_voxel"
    assert payload["interior_semantics"] == "unknown"
    assert payload["solid_authority"] == "none"
    encoding = payload["encoding"]
    assert isinstance(encoding, dict)
    size = observation.stack.volume.size
    decoded_surface = _decode_runs(encoding["runs"], size).reshape(observation.stack.volume.shape)

    assert np.array_equal(decoded_surface, (observation.stack.volume & SURFACE_BIT) != 0)
    assert not np.any(observation.stack.volume & ~SURFACE_BIT)


def test_agent_core_round_trips_through_the_diagnostic_parser(
    box_observation: AxialObservation,
) -> None:
    observation = box_observation

    assert observation.agent_core.startswith("AXIAL\n")
    parsed = parse_axial_stack(observation.agent_core)

    # The serialized text carries finitely many digits; the parser reconstructs the declared
    # values, so frame scalars round-trip only to that printed precision.
    assert np.array_equal(parsed.frame.dimensions, observation.stack.frame.dimensions)
    assert parsed.frame.pitch == pytest.approx(observation.stack.frame.pitch, rel=1e-9)
    assert np.allclose(parsed.frame.bounds_min, observation.stack.frame.bounds_min, atol=1e-9)
    assert np.array_equal(parsed.micro_frame.dimensions, observation.micro_frame.dimensions)
    assert parsed.micro_frame.pitch == pytest.approx(observation.micro_frame.pitch, rel=1e-9)
    assert np.array_equal(parsed.volume != 0, observation.stack.volume != 0)
    assert not np.any(parsed.volume & ~SURFACE_BIT)


@pytest.mark.slow
def test_observation_is_byte_deterministic_for_identical_scenes(
    box_observation: AxialObservation,
) -> None:
    # The shared observation and a fresh build are two independent runs over identical scenes.
    first = box_observation
    second = build_axial_observation(_scene(_box()))

    assert first.agent_core == second.agent_core
    assert first.agent_core_byte_count == second.agent_core_byte_count
    assert format_voxel_artifact(first) == format_voxel_artifact(second)


@pytest.mark.slow
def test_face_features_detect_a_fused_front_knob_in_world_metres() -> None:
    # A 4 cm knob standing 3 cm proud of the front face: within the detector's compact range
    # (the local base window is 6 cm, so a control must be narrower than that to separate).
    body = trimesh.creation.box(extents=(1.0, 0.4, 0.8))
    knob = trimesh.creation.box(extents=(0.04, 0.03, 0.04))
    knob_vertices = np.asarray(knob.vertices) + np.asarray((0.3, -0.215, 0.2))
    merged_vertices = np.concatenate([np.asarray(body.vertices), knob_vertices])
    merged_faces = np.concatenate(
        [np.asarray(body.faces), np.asarray(knob.faces) + len(body.vertices)]
    )
    fused = _mesh_component("fused_body", merged_vertices, merged_faces, closed=False)
    observation = build_axial_observation(_scene(fused))

    front = [f for f in observation.face_features.features if f.face == "-Y"]
    assert front, observation.face_features
    knob_feature = max(front, key=lambda f: f.height_max_m)
    assert abs(knob_feature.center_uv[0] - 0.3) < 0.01
    assert abs(knob_feature.center_uv[1] - 0.2) < 0.01
    assert (
        abs(knob_feature.size_uv[0] - 0.04) < 0.012 and abs(knob_feature.size_uv[1] - 0.04) < 0.012
    )
    assert abs(knob_feature.height_max_m - 0.03) < 0.008
    assert "[FACE_FEATURES" in observation.agent_core
    assert "-Y c(x,z)=(+0.30" in observation.agent_core
    for absent in ("[FACE_RELIEF192", "[MASK24", "[ANON_THIN_SURFACE_EXTENTS192", "mask_grid24"):
        assert absent not in observation.agent_core, absent
    parsed = parse_axial_stack(observation.agent_core)
    assert np.array_equal(parsed.volume, observation.stack.volume)


@pytest.mark.slow
def test_face_features_are_silent_for_a_plain_box_and_a_stepped_tier() -> None:
    box = trimesh.creation.box(extents=(1.0, 0.4, 0.8))
    plain = build_axial_observation(
        _scene(_mesh_component("box", np.asarray(box.vertices), np.asarray(box.faces), closed=True))
    )
    assert "[FACE_FEATURES" not in plain.agent_core
    # a lower tier wider than the upper body is a step, not a compact control
    lower = trimesh.creation.box(extents=(1.0, 0.4, 0.3))
    upper = trimesh.creation.box(extents=(0.8, 0.3, 0.5))
    upper_vertices = np.asarray(upper.vertices) + np.asarray((0.0, 0.0, 0.4))
    vertices = np.concatenate([np.asarray(lower.vertices), upper_vertices])
    faces = np.concatenate([np.asarray(lower.faces), np.asarray(upper.faces) + len(lower.vertices)])
    stepped = build_axial_observation(
        _scene(_mesh_component("stepped", vertices, faces, closed=False))
    )
    assert "[FACE_FEATURES" not in stepped.agent_core


def test_body_walls_follow_the_face_while_the_extreme_follows_a_burr() -> None:
    body = trimesh.creation.box(extents=(0.14, 0.24, 0.16))
    # a 6 mm burr on the +X face at mid height: the extreme moves, the wall does not
    burr = trimesh.creation.box(extents=(0.006, 0.006, 0.006))
    burr_vertices = np.asarray(burr.vertices) + np.asarray((0.073, 0.0, 0.0))
    vertices = np.concatenate([np.asarray(body.vertices), burr_vertices])
    faces = np.concatenate([np.asarray(body.faces), np.asarray(burr.faces) + len(body.vertices)])
    observation = build_axial_observation(
        _scene(_mesh_component("burred", vertices, faces, closed=False))
    )
    rows = observation.body_walls.rows
    assert rows and "[BODY_WALLS_Z" in observation.agent_core
    mid = [row for row in rows if row.z0 <= 0.0 <= row.z1]
    assert mid
    row = mid[0]
    assert abs(row.x_wall[1] - 0.07) < 0.003
    assert row.x_extreme[1] > row.x_wall[1] + 0.004
    assert abs(row.x_wall[0] + 0.07) < 0.003 and abs(row.x_extreme[0] - row.x_wall[0]) < 0.003
    parsed = parse_axial_stack(observation.agent_core)
    assert np.array_equal(parsed.volume, observation.stack.volume)


@pytest.mark.slow
def test_body_walls_ignore_an_open_door_covering_most_columns() -> None:
    # a box body whose -Y face carries a door hinged at -X and swung open 45 degrees: at the
    # door's height it covers 60% of the front columns, so the plain median of column extremes
    # would land inside the door; the wall must stay on the body face the whole front ends on
    body = trimesh.creation.box(extents=(0.40, 0.30, 0.20))
    door = trimesh.creation.box(extents=(0.28, 0.006, 0.12))
    angle = math.radians(45.0)
    rotation = np.asarray(
        [
            [math.cos(angle), math.sin(angle), 0.0],
            [-math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    door_vertices = (np.asarray(door.vertices) + np.asarray((0.14, 0.0, 0.0))) @ rotation.T
    door_vertices = door_vertices + np.asarray((-0.20, -0.15, 0.0))
    vertices = np.concatenate([np.asarray(body.vertices), door_vertices])
    faces = np.concatenate([np.asarray(body.faces), np.asarray(door.faces) + len(body.vertices)])
    observation = build_axial_observation(
        _scene(_mesh_component("open_door", vertices, faces, closed=False))
    )
    rows = observation.body_walls.rows
    mid = [row for row in rows if row.z0 <= 0.0 <= row.z1]
    assert mid
    row = mid[0]
    assert abs(row.y_wall[0] + 0.15) < 0.006, row.y_wall  # within one micro cell
    assert row.y_extreme[0] < -0.25


@pytest.mark.slow
def test_body_support_reads_feet_as_a_contact_patch_and_a_flat_bottom_as_flat() -> None:
    body = trimesh.creation.box(extents=(0.20, 0.30, 0.16))
    body_vertices = np.asarray(body.vertices) + np.asarray((0.0, 0.0, 0.10))  # underside z=0.02
    parts = [body_vertices]
    faces = [np.asarray(body.faces)]
    offset = len(body_vertices)
    for sx in (-1, 1):
        for sy in (-1, 1):
            foot = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
            parts.append(np.asarray(foot.vertices) + np.asarray((sx * 0.08, sy * 0.13, 0.01)))
            faces.append(np.asarray(foot.faces) + offset)
            offset += len(foot.vertices)
    observation = build_axial_observation(
        _scene(
            _mesh_component("on_feet", np.concatenate(parts), np.concatenate(faces), closed=False)
        )
    )
    support = observation.body_walls.support
    assert support is not None
    assert support.verdict == "contact_patch_or_unscanned_underside"
    assert support.contact_cover < 0.2
    assert abs(support.body_full_width_from_z - 0.02) < 0.012
    assert "support: contact_cover=" in observation.agent_core
    assert "cover=" in observation.agent_core

    flat = trimesh.creation.box(extents=(0.20, 0.30, 0.16))
    flat_observation = build_axial_observation(
        _scene(
            _mesh_component("flat", np.asarray(flat.vertices), np.asarray(flat.faces), closed=True)
        )
    )
    assert flat_observation.body_walls.support.verdict == "flat_scanned_bottom"


def test_face_openings_section_and_brief_carry_a_top_slot() -> None:
    from usdcraft.reconstruction.axial_brief import render_axial_brief

    body = trimesh.creation.box(extents=(0.14, 0.25, 0.16))
    body.apply_translation((0.0, 0.0, 0.08))
    slot = trimesh.creation.box(extents=(0.035, 0.13, 0.12))
    slot.apply_translation((0.0, 0.0, 0.101))
    shape = body.difference(slot)
    observation = build_axial_observation(
        _scene(
            _mesh_component(
                "slotted", np.asarray(shape.vertices), np.asarray(shape.faces), closed=False
            )
        )
    )
    assert "[FACE_OPENINGS" in observation.agent_core
    assert any(o.face == "+Z" and o.kind == "opening" for o in observation.face_openings.openings)
    brief = render_axial_brief(observation.agent_core)
    assert "[FACE_OPENINGS" in brief and "depth>=" in brief
    parsed = parse_axial_stack(observation.agent_core)
    assert np.array_equal(parsed.volume, observation.stack.volume)


@pytest.mark.slow
def test_interior_islands_section_and_brief_carry_two_rack_levels_inside_an_open_box(
    box_observation: AxialObservation,
) -> None:
    from usdcraft.reconstruction.axial_brief import render_axial_brief

    outer = trimesh.creation.box(extents=(0.4, 0.3, 0.3))
    inner = trimesh.creation.box(extents=(0.38, 0.28, 0.3))
    inner.apply_translation((0.0, 0.0, 0.011))
    shell = outer.difference(inner)
    rods = []
    for level in (-0.05, 0.05):
        for index in range(5):
            rod = trimesh.creation.box(extents=(0.006, 0.24, 0.006))
            rod.apply_translation((-0.12 + 0.06 * index, 0.0, level))
            rods.append(rod)
    rack = trimesh.util.concatenate(rods)
    observation = build_axial_observation(
        _scene(
            _mesh_component(
                "shell", np.asarray(shell.vertices), np.asarray(shell.faces), closed=False
            ),
            _mesh_component(
                "rack", np.asarray(rack.vertices), np.asarray(rack.faces), closed=False
            ),
        )
    )
    groups = observation.interior_islands.groups
    assert "[INTERIOR_ISLANDS" in observation.agent_core
    assert len(groups) == 2 and all(group.kind == "grid" for group in groups)
    assert groups[0].members == 5 and groups[1].same_as == 1
    assert groups[1].dz_m is not None and abs(groups[1].dz_m - 0.1) < 0.02
    assert abs(groups[0].size_m[0] - 0.246) < 0.01 and abs(groups[0].size_m[1] - 0.24) < 0.01
    brief = render_axial_brief(observation.agent_core)
    assert "[INTERIOR_ISLANDS" in brief and "same_as=g1" in brief
    parsed = parse_axial_stack(observation.agent_core)
    assert np.array_equal(parsed.volume, observation.stack.volume)
    # a closed box has no interior island and pays no bytes for the section
    closed = box_observation
    assert closed.interior_islands.groups == () and "[INTERIOR_ISLANDS" not in closed.agent_core
