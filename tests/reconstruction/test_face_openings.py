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
import trimesh

from usdcraft.reconstruction.face_openings import (
    FaceOpening,
    detect_face_openings,
    format_face_openings,
    parse_face_openings,
)
from usdcraft.reconstruction.provider_axial_audit import format_provider_audit, opening_placement


class _Component:
    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.vertices = np.asarray(mesh.vertices, dtype=float)
        self.faces = np.asarray(mesh.faces, dtype=int)


def _slotted_body() -> trimesh.Trimesh:
    body = trimesh.creation.box(extents=(0.14, 0.25, 0.16))
    body.apply_translation((0.0, 0.0, 0.08))
    slot = trimesh.creation.box(extents=(0.035, 0.13, 0.12))
    slot.apply_translation((0.0, 0.0, 0.16 - 0.06 + 0.001))
    return body.difference(slot)


def test_face_openings_find_a_top_slot_with_plan_size_and_lower_bound_depth() -> None:
    summary = detect_face_openings([_Component(_slotted_body())])
    tops = [o for o in summary.openings if o.face == "+Z" and o.kind == "opening"]
    assert tops, summary
    slot = max(tops, key=lambda o: o.size_uv[0] * o.size_uv[1])
    assert abs(slot.center_uv[0]) < 0.006 and abs(slot.center_uv[1]) < 0.006
    assert abs(slot.size_uv[0] - 0.035) < 0.008 and abs(slot.size_uv[1] - 0.13) < 0.01
    assert slot.depth_p50_m is not None and slot.depth_p50_m > 0.10
    assert slot.reference == "rim" and not slot.open_face
    text = "\n".join(format_face_openings(summary))
    assert "[FACE_OPENINGS" in text and "LOWER BOUND" in text
    back = parse_face_openings(text + "\n[NEXT]")
    assert [(o.face, o.kind) for o in back] == [(o.face, o.kind) for o in summary.openings]


def test_face_openings_read_an_open_box_top_and_stay_silent_on_a_closed_box() -> None:
    closed = trimesh.creation.box(extents=(0.4, 0.3, 0.2))
    assert detect_face_openings([_Component(closed)]).openings == ()
    # open-top box: four walls and a floor, no lid
    outer = trimesh.creation.box(extents=(0.4, 0.3, 0.2))
    inner = trimesh.creation.box(extents=(0.38, 0.28, 0.2))
    inner.apply_translation((0.0, 0.0, 0.011))
    box = outer.difference(inner)
    summary = detect_face_openings([_Component(box)])
    tops = [o for o in summary.openings if o.face == "+Z"]
    assert tops and tops[0].open_face and tops[0].kind == "opening"
    assert tops[0].depth_p50_m is not None and tops[0].depth_p50_m > 0.15
    assert not [o for o in summary.openings if o.face in ("-Y", "+Y", "-X", "+X")]


def test_a_fascia_in_front_of_a_receding_chimney_is_a_face_not_an_open_face() -> None:
    # a range hood: a shallow canopy whose front plane holds the controls, and a tall chimney
    # standing well behind it that covers most of the -Y columns
    canopy = trimesh.creation.box(extents=(0.6, 0.15, 0.15))
    chimney = trimesh.creation.box(extents=(0.6, 0.3, 0.75))
    chimney.apply_translation((0.0, 0.225, 0.45))
    hood = trimesh.util.concatenate([canopy, chimney])
    front = [o for o in detect_face_openings([_Component(hood)]).openings if o.face == "-Y"]
    assert not [o for o in front if o.open_face], front
    # while a genuinely open box top keeps its open_face reading
    outer = trimesh.creation.box(extents=(0.4, 0.3, 0.2))
    inner = trimesh.creation.box(extents=(0.38, 0.28, 0.2))
    inner.apply_translation((0.0, 0.0, 0.011))
    tops = [
        o
        for o in detect_face_openings([_Component(outer.difference(inner))]).openings
        if o.face == "+Z"
    ]
    assert tops and tops[0].open_face


def _opening(face, cu, cv, su, sv, kind="opening") -> FaceOpening:
    return FaceOpening(face, kind, (cu, cv), (su, sv), 0.04, 0.05, 0.0, False, 0, "rim")


def test_opening_placement_matches_by_face_and_reports_unmodelled_source_openings() -> None:
    source = [
        _opening("+Z", -0.017, 0.025, 0.032, 0.124),
        _opening("+Z", 0.025, 0.023, 0.03, 0.126),
    ]
    candidate = [_opening("+Z", -0.02, 0.02, 0.044, 0.15)]
    result = opening_placement(source, candidate, stack_pitch=0.0105)
    rows = result["candidate_openings"]
    assert len(rows) == 1 and rows[0]["offset_mm"] == 6
    assert rows[0]["size_ratio_uv"] == [1.38, 1.21]
    assert "differs" in rows[0]["reading"]
    assert len(result["unmodelled_source_openings"]) == 1
    assert result["unmodelled_source_openings"][0]["size_mm"] == [30, 126]


def test_opening_placement_rows_render_as_plain_numbers_for_numpy_inputs() -> None:
    # detected openings carry numpy scalars; the check_axial text must not leak ``np.float64(...)``
    source = [_opening("+Z", -0.017, 0.025, 0.032, 0.124)]
    candidate = [
        FaceOpening(
            "+Z",
            "opening",
            (np.float64(-0.018), np.float64(0.0245)),
            (np.float64(0.03), np.float64(0.122)),
            0.04,
            0.05,
            0.0,
            False,
            0,
            "rim",
        )
    ]
    result = opening_placement(source, candidate, stack_pitch=0.0105)
    row = result["candidate_openings"][0]
    assert all(type(v) is float for v in row["candidate_size_uv_m"] + row["size_ratio_uv"])
    text = format_provider_audit(
        {
            "exterior_excess": {
                "tolerance_stack_cells": 1,
                "excess_cells": 0,
                "candidate_surface_cells": 1,
                "fraction_of_candidate": 0.0,
                "slices_with_excess": 0,
                "interior_candidate_only_cells": 0,
                "worst_slices": [],
            },
            "face_depth": [],
            "opening_placement": result,
            "stack_pitch_m": 0.0105,
        }
    )
    assert "np.float64" not in text
    assert "size=[0.03, 0.122] offset=1mm size_ratio=[0.94, 0.98]" in text
