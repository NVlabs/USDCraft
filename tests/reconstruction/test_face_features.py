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

from usdcraft.reconstruction.face_features import (
    HEIGHT_MIN_M,
    HEIGHT_PEAK_FACTOR,
    HEIGHT_PLATEAU_PEAK_FACTOR,
    detect_face_features,
    format_face_features,
    parse_face_features,
)
from usdcraft.reconstruction.provider_axial_audit import feature_placement, format_provider_audit


class _Component:
    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.vertices = np.asarray(mesh.vertices, dtype=float)
        self.faces = np.asarray(mesh.faces, dtype=int)


def _fascia_with_buttons(height: float, *, xs=(-0.08, -0.04, 0.0, 0.04, 0.08)) -> trimesh.Trimesh:
    """A 300 x 200 x 250 mm body with a row of 24 mm caps standing `height` proud of its -Y face."""

    body = trimesh.creation.box(extents=(0.3, 0.2, 0.25))
    parts = [body]
    for x in xs:
        cap = trimesh.creation.box(extents=(0.024, height + 0.004, 0.024))
        cap.apply_translation((x, -0.1 - height / 2 + 0.002, -0.03))
        parts.append(cap)
    return trimesh.util.concatenate(parts)


def test_five_millimetre_buttons_survive_the_peak_gate() -> None:
    summary = detect_face_features([_Component(_fascia_with_buttons(0.005))])
    assert summary.pitch_m == 0.002  # 300 mm body: the 2 mm floor applies, height_min is 3 mm
    front = [f for f in summary.features if f.face == "-Y"]
    assert len(front) == 5, (summary.features, summary.rejected)
    assert all(0.004 <= f.height_median_m <= 0.0065 for f in front)
    assert all(0.020 <= min(f.size_uv) and max(f.size_uv) <= 0.030 for f in front)
    assert sorted(round(f.center_uv[0], 2) for f in front) == [-0.08, -0.04, 0.0, 0.04, 0.08]
    # the 2x peak floor alone (6 mm) would drop every one of these 5 mm caps; they pass as plateaus
    assert HEIGHT_PLATEAU_PEAK_FACTOR * HEIGHT_MIN_M < 0.005 < HEIGHT_PEAK_FACTOR * HEIGHT_MIN_M
    text = "\n".join(format_face_features(summary))
    back = parse_face_features(text + "\n[NEXT]")
    assert len([f for f in back if f.face == "-Y"]) == 5


def test_two_millimetre_relief_is_still_below_the_median_floor() -> None:
    summary = detect_face_features([_Component(_fascia_with_buttons(0.002))])
    assert not [f for f in summary.features if f.face == "-Y"]


def test_feature_placement_matches_one_to_one_and_lists_both_remainders() -> None:
    source = detect_face_features([_Component(_fascia_with_buttons(0.005))]).features
    # the candidate builds three of the five caps, one of them 20 mm off, plus a stray knob
    candidate_mesh = _fascia_with_buttons(0.006, xs=(-0.08, -0.02, 0.08))
    knob = trimesh.creation.cylinder(radius=0.015, height=0.012)
    knob.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    knob.apply_translation((0.11, -0.104, 0.08))
    candidate = detect_face_features(
        [_Component(trimesh.util.concatenate([candidate_mesh, knob]))]
    ).features
    result = feature_placement(source, candidate, stack_pitch=0.0125)
    assert result["tolerance_mm"] == 25
    matched = [row for row in result["matched"] if row["face"] == "-Y"]
    assert len(matched) == 3
    offsets = sorted(row["offset_mm"] for row in matched)
    assert offsets[0] <= 3 and offsets[1] <= 3 and 18 <= offsets[2] <= 22
    # no source cap is claimed twice and the two unbuilt caps are the ones at x=0.0 and x=0.04
    sources = {tuple(row["center_uv_m"]) for row in matched}
    assert len(sources) == 3
    unbuilt_x = sorted(
        row["center_uv_m"][0] for row in result["unbuilt_source"] if row["face"] == "-Y"
    )
    assert [round(x, 2) for x in unbuilt_x] == [0.0, 0.04]
    extra = [row for row in result["candidate_only"] if row["face"] == "-Y"]
    assert len(extra) == 1 and abs(extra[0]["center_uv_m"][0] - 0.11) < 0.005
    text = format_provider_audit({"defects": [], "feature_placement": result})
    assert "protrusions (" in text and "unbuilt source protrusion -Y" in text
    assert "candidate-only protrusion -Y" in text and "np.float64" not in text
    assert "DISCREPANCIES (0)" in text


def test_feature_placement_rejects_a_size_mismatch_as_a_match() -> None:
    source = detect_face_features(
        [_Component(_fascia_with_buttons(0.005, xs=(-0.03, 0.03)))]
    ).features
    assert len([f for f in source if f.face == "-Y"]) == 2
    # a 54 mm cap where the source has a 24 mm one: same place, 2.25x the size, not a match
    big = trimesh.creation.box(extents=(0.3, 0.2, 0.25))
    cap = trimesh.creation.box(extents=(0.054, 0.012, 0.054))
    cap.apply_translation((-0.03, -0.104, -0.03))
    candidate = detect_face_features([_Component(trimesh.util.concatenate([big, cap]))]).features
    assert len([f for f in candidate if f.face == "-Y"]) == 1
    result = feature_placement(source, candidate, stack_pitch=0.01)
    assert result["matched"] == []
    assert len(result["unbuilt_source"]) == 2 and len(result["candidate_only"]) == 1


def test_a_lone_band_blob_is_not_a_button_but_a_row_is() -> None:
    lone = detect_face_features([_Component(_fascia_with_buttons(0.005, xs=(0.0,)))])
    assert not [f for f in lone.features if f.face == "-Y"]
    assert lone.rejected.get("L1_lone", 0) >= 1
    pair = detect_face_features([_Component(_fascia_with_buttons(0.005, xs=(-0.03, 0.03)))])
    assert len([f for f in pair.features if f.face == "-Y"]) == 2
    # a tall lone cap (peak above twice the floor) needs no row
    tall = detect_face_features([_Component(_fascia_with_buttons(0.010, xs=(0.0,)))])
    assert len([f for f in tall.features if f.face == "-Y"]) == 1
