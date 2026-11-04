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

from sdk import (
    AFrameGeometry,
    BeltSeatGeometry,
    CasterForkGeometry,
    CasterGeometry,
    FlexibleHoseGeometry,
    GableRoofGeometry,
    HangingChainGeometry,
    LadderGeometry,
    MoldedSeatGeometry,
    PostBeamBracketGeometry,
    SlideChuteGeometry,
)


@pytest.mark.parametrize(
    "geometry",
    [
        FlexibleHoseGeometry([(0, 0, 0), (0.1, 0, 0.2), (0.2, 0.1, 0.1)], radius=0.01),
        CasterGeometry(wheel_radius=0.08, wheel_width=0.035),
        CasterForkGeometry(wheel_radius=0.08, wheel_width=0.035, trail_offset=0.03),
        MoldedSeatGeometry(
            width=0.46,
            depth=0.30,
            back_height=0.34,
            thickness=0.025,
            seat_sag=0.02,
            back_recline=0.04,
        ),
        PostBeamBracketGeometry(
            member_width=0.10,
            member_height=0.14,
            plate_length=0.22,
            plate_thickness=0.008,
            bolt_radius=0.006,
        ),
        GableRoofGeometry(
            width=0.9,
            depth=1.1,
            rise=0.28,
            panel_thickness=0.035,
            overhang=0.08,
        ),
        BeltSeatGeometry(width=0.4, depth=0.18, thickness=0.02, sag=0.025),
        SlideChuteGeometry(
            width=0.4,
            length=0.9,
            drop=0.5,
            thickness=0.025,
            rail_radius=0.018,
        ),
        HangingChainGeometry(length=0.7, link_count=8, link_radius=0.025, wire_radius=0.004),
        LadderGeometry(
            width=0.4,
            rise=0.9,
            run=0.35,
            rail_section=0.04,
            rung_count=5,
        ),
        AFrameGeometry(
            height=1.1,
            footprint_width=0.7,
            leg_section=0.06,
            spreader_height=0.3,
            diagonal_brace=True,
        ),
    ],
)
def test_high_frequency_assembly_geometry_is_finite_and_nonempty(geometry) -> None:
    vertices = np.asarray(geometry.vertices, dtype=float)
    assert len(vertices) > 8
    assert geometry.faces
    assert np.isfinite(vertices).all()
    assert np.all(vertices.max(axis=0) > vertices.min(axis=0))


def test_hose_rejects_invalid_route() -> None:
    with pytest.raises(ValueError, match="at least two"):
        FlexibleHoseGeometry([(0, 0, 0)], radius=0.01)


def test_ladder_and_a_frame_use_documented_local_anchors() -> None:
    ladder = np.asarray(
        LadderGeometry(
            width=0.4,
            rise=0.9,
            run=0.35,
            rail_section=0.04,
            rung_count=5,
        ).vertices
    )
    frame = np.asarray(
        AFrameGeometry(
            height=1.1,
            footprint_width=0.7,
            leg_section=0.06,
            spreader_height=0.3,
        ).vertices
    )
    assert ladder[:, 2].min() < 0.03
    assert ladder[:, 2].max() > 0.87
    assert frame[:, 2].min() < 0.03
    assert frame[:, 2].max() > 1.07


def test_molded_seat_and_gable_roof_preserve_documented_frames() -> None:
    seat = np.asarray(
        MoldedSeatGeometry(
            width=0.46,
            depth=0.30,
            back_height=0.34,
            thickness=0.025,
            seat_sag=0.02,
            back_recline=0.04,
        ).vertices
    )
    roof = np.asarray(
        GableRoofGeometry(
            width=0.9,
            depth=1.1,
            rise=0.28,
            panel_thickness=0.035,
            overhang=0.08,
        ).vertices
    )

    assert seat[:, 0].min() < -0.22
    assert seat[:, 0].max() > 0.22
    assert seat[:, 2].max() > 0.32
    assert roof[:, 0].min() < -0.50
    assert roof[:, 0].max() > 0.50
    assert roof[:, 2].max() > 0.28
