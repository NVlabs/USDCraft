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

from dataclasses import dataclass

import numpy as np

from usdcraft.reconstruction.interior_islands import (
    INTERIOR_MAX_GROUPS,
    detect_interior_islands,
    format_interior_islands,
    parse_interior_islands,
)
from usdcraft.reconstruction.provider_axial_audit import (
    format_provider_audit,
    interior_fitting_readings,
)


@dataclass(frozen=True)
class _Island:
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    cell_count: int

    @property
    def bbox_cells(self) -> int:
        return (self.x_end - self.x_start) * (self.y_end - self.y_start)


@dataclass(frozen=True)
class _Run:
    stack_z_start: int
    stack_z_end: int
    micro_z_start: int
    micro_z_end: int
    islands: tuple[_Island, ...]


@dataclass(frozen=True)
class _Frame:
    pitch: float
    dimensions: np.ndarray
    bounds_min: np.ndarray


PITCH = 0.005
FRAME = _Frame(PITCH, np.array([120, 100, 60]), np.array([-0.3, -0.25, 0.0]))
BODY = _Island(0, 120, 0, 100, 800)  # the shell ring: full footprint, sparse fill


def _rack_slab(stack_z: int, micro_z: int, *, tines: int = 12) -> _Run:
    """A slab holding the shell ring, two frame rails and a row of tines seen end-on."""

    rails = (_Island(20, 100, 20, 22, 160), _Island(20, 100, 78, 80, 160))
    row = tuple(_Island(24 + 6 * i, 26 + 6 * i, 40, 42, 4) for i in range(tines))
    return _Run(stack_z, stack_z + 1, micro_z, micro_z + 4, (BODY, *rails, *row))


def _plate_slab(stack_z: int, micro_z: int) -> _Run:
    return _Run(stack_z, stack_z + 1, micro_z, micro_z + 4, (BODY, _Island(20, 100, 20, 80, 4600)))


def _empty_slab(stack_z: int, micro_z: int) -> _Run:
    return _Run(stack_z, stack_z + 1, micro_z, micro_z + 4, (BODY,))


def test_two_rack_levels_group_their_plate_rails_and_tines_and_mark_the_repeat() -> None:
    runs = (
        _plate_slab(2, 8),
        _rack_slab(3, 12),
        _rack_slab(4, 16),
        _empty_slab(5, 20),
        _empty_slab(6, 24),
        _plate_slab(7, 28),
        _rack_slab(8, 32),
        _rack_slab(9, 36),
    )
    summary = detect_interior_islands(runs, FRAME)
    assert summary.detected == 2 and len(summary.groups) == 2
    lower, upper = summary.groups
    assert lower.kind == "grid" and lower.kinds == ("plate", "grid")
    assert lower.members == 12 and lower.member_classes[0][0] == 12
    assert lower.z_m == (0.0 + 8 * PITCH, 0.0 + 20 * PITCH)
    # the rails set the footprint: cells 20..100 x 20..80 in metres from the frame origin
    assert np.allclose(lower.x_m, (-0.3 + 20 * PITCH, -0.3 + 100 * PITCH))
    assert np.allclose(lower.y_m, (-0.25 + 20 * PITCH, -0.25 + 80 * PITCH))
    assert upper.same_as == 1 and upper.dz_m is not None and abs(upper.dz_m - 20 * PITCH) < 1e-9
    lines = format_interior_islands(summary)
    assert lines[0].startswith(
        "[INTERIOR_ISLANDS parent=MICRO_SLICE_BOUNDS_Z192 shown=2 detected=2"
    )
    assert "HYPOTHESIS" in lines[1] and "NOT Parts" in lines[1] and "UNKNOWN" in lines[1]
    assert "kind=grid n=12 members=12@10x10mm" in lines[3]
    assert "same_as=g1 dz=+0.100" in lines[4]
    assert summary.section_bytes == sum(len(line.encode("utf-8")) + 1 for line in lines)


def test_parse_round_trips_the_serialized_groups() -> None:
    runs = (_plate_slab(2, 8), _rack_slab(3, 12), _plate_slab(7, 28), _rack_slab(8, 32))
    summary = detect_interior_islands(runs, FRAME)
    text = "\n".join(format_interior_islands(summary)) + "\n[NEXT]\nignored"
    back = parse_interior_islands(text)
    assert [(g.index, g.kind, g.kinds, g.members, g.same_as) for g in back] == [
        (g.index, g.kind, g.kinds, g.members, g.same_as) for g in summary.groups
    ]
    assert all(abs(a.z_m[0] - b.z_m[0]) < 1e-3 for a, b in zip(back, summary.groups, strict=True))
    assert back[0].member_classes == ((12, 0.01, 0.01),)


def test_wall_attached_islands_specks_and_sparse_slabs_are_not_fittings() -> None:
    touching = _Island(0, 60, 20, 80, 3000)  # shares the body's x_start: attached to a wall
    speck = _Island(50, 52, 50, 52, 4)
    inner_wall = _Island(3, 117, 3, 97, 700)  # the shell's inner face, 15 mm inside the outer
    runs = (
        _Run(1, 2, 4, 8, (BODY, touching, speck)),
        _Run(2, 3, 8, 12, (BODY, inner_wall)),
        # a slab of legs only: the largest island is far smaller than the footprint
        _Run(0, 1, 0, 4, (_Island(0, 10, 0, 10, 100), _Island(3, 5, 3, 5, 4))),
    )
    summary = detect_interior_islands(runs, FRAME)
    assert summary.groups == () and summary.detected == 0
    assert format_interior_islands(summary) == ()


def test_one_free_bar_spanning_the_shell_is_reported_but_two_plates_side_by_side_are_not_repeats() -> (
    None
):
    bar = _Island(30, 90, 50, 51, 60)  # 300 mm long, 5 mm wide, 60% of the body's width
    plates = (_Island(10, 55, 20, 80, 2600), _Island(65, 110, 20, 80, 2600))
    runs = (_Run(2, 3, 8, 12, (BODY, bar)), _Run(5, 6, 20, 24, (BODY, *plates)))
    summary = detect_interior_islands(runs, FRAME)
    kinds = [g.kind for g in summary.groups]
    assert kinds == ["bar", "plate", "plate"]
    assert all(g.same_as is None for g in summary.groups)


def test_group_cap_keeps_the_largest_and_records_detected() -> None:
    # one plate every other slab: never adjacent, so each is its own group
    runs = tuple(
        _Run(
            2 * z,
            2 * z + 1,
            8 * z,
            8 * z + 4,
            (BODY, _Island(10 + z, 40 + z + 2 * z, 10, 60 + z, 1500 + 100 * z)),
        )
        for z in range(1, 13)
    )
    summary = detect_interior_islands(runs, FRAME)
    assert summary.detected == 12 and len(summary.groups) == INTERIOR_MAX_GROUPS
    assert f"shown={INTERIOR_MAX_GROUPS} detected=12" in format_interior_islands(summary)[0]
    assert [g.index for g in summary.groups] == list(range(1, INTERIOR_MAX_GROUPS + 1))


def test_interior_fitting_readings_count_candidate_surface_in_the_group_box() -> None:
    runs = (_plate_slab(2, 8), _rack_slab(3, 12))
    groups = detect_interior_islands(runs, FRAME).groups
    group = groups[0]
    inside = np.column_stack(
        [
            np.linspace(group.x_m[0], group.x_m[1], 40),
            np.linspace(group.y_m[0], group.y_m[1], 40),
            np.full(40, sum(group.z_m) / 2),
        ]
    )
    far = inside + np.array([0.0, 0.0, 0.5])
    high = inside + np.array([0.0, 0.0, 0.04])  # 40 mm above the band, inside the search window
    modelled = interior_fitting_readings(groups, inside, stack_pitch=0.02)
    missing = interior_fitting_readings(groups, far, stack_pitch=0.02)
    displaced = interior_fitting_readings(groups, high, stack_pitch=0.02)
    assert modelled[0]["modelled"] and modelled[0]["candidate_surface_points"] == 40
    assert modelled[0]["z_offset_mm"] == 0
    assert not missing[0]["modelled"] and missing[0]["candidate_surface_points"] == 0
    assert missing[0]["z_offset_mm"] is None
    assert displaced[0]["modelled"]
    band_top = group.z_m[1]
    assert displaced[0]["z_offset_mm"] == round((sum(group.z_m) / 2 + 0.04 - band_top) * 1000)
    text = format_provider_audit(
        {"defects": [], "interior_fittings": missing + displaced + modelled}
    )
    assert "interior fittings" in text and "g1 z[" in text and "no candidate surface there" in text
    assert "above the source band" in text and "modelled at height" in text
    assert "DISCREPANCIES (0)" in text
