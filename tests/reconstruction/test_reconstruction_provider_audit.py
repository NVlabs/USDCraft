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

"""Provider-arm AXIAL audit: mesh-rasterized face depth, envelope, excess and openings."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from usdcraft.reconstruction.face_features import FaceFeature, parse_face_features
from usdcraft.reconstruction.provider_axial_audit import (
    envelope_comparison,
    face_depth_audit,
    format_provider_audit,
    parse_micro_slabs,
)


def _box_points(lo: tuple[float, float, float], hi: tuple[float, float, float], n: int = 400_000):
    """Deterministic surface samples of an axis-aligned box shell."""

    rng = np.random.default_rng(7)
    lo_a, hi_a = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    faces = []
    for axis in range(3):
        for value in (lo_a[axis], hi_a[axis]):
            pts = lo_a + rng.random((n // 6, 3)) * (hi_a - lo_a)
            pts[:, axis] = value
            faces.append(pts)
    return np.vstack(faces)


def test_face_depth_audit_is_clean_for_an_identical_shell_and_flags_a_pushed_front() -> None:
    source = _box_points((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8))
    same = face_depth_audit(source, source.copy(), stack_pitch=0.02)
    assert same["excess"]["columns"] == 0
    assert all(not row["defects"] for row in same["faces"])
    front = next(row for row in same["faces"] if row["face"] == "-Y front")
    assert front["on_plane_share"] > 0.95 and front["shifted_share"] < 0.05

    pushed = _box_points((-0.5, -0.52, 0.0), (0.5, 0.5, 0.8))  # front face 20 mm outward
    result = face_depth_audit(source, pushed, stack_pitch=0.03)
    front = next(row for row in result["faces"] if row["face"] == "-Y front")
    assert front["shifted_share"] > 0.9 and front["source_planarity"] > 0.9
    assert 0.015 <= front["plane_offset_m"] <= 0.025  # positive = candidate outward
    assert any(
        text.startswith("-Y front face built") and "outward" in text for text in front["defects"]
    )
    # The other faces only see the 20 mm strip as a few new columns, never as a plane shift.
    back = next(row for row in result["faces"] if row["face"] == "+Y back")
    assert back["shifted_share"] < 0.05 and not back["defects"]
    text = format_provider_audit({"face_depth": result, "defects": front["defects"]})
    assert text.startswith("DISCREPANCIES (1)") and "D1 -Y front face built" in text
    assert "overall |Δ|" in text


def test_face_depth_audit_reports_candidate_surface_beyond_the_source_silhouette() -> None:
    source = _box_points((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8))
    wing = _box_points((0.5, -0.5, 0.0), (0.7, 0.5, 0.8))  # a slab standing beyond +X
    candidate = np.vstack([source, wing])
    result = face_depth_audit(source, candidate, stack_pitch=0.02)
    top = next(row for row in result["faces"] if row["face"] == "+Z top")
    assert top["excess_columns"] > 0 and top["excess_share_of_candidate"] > 0.1
    assert top["excess_region_uv_m"][0] >= 0.48  # the excess starts at the source's +X face
    assert result["excess"]["columns"] > 0 and result["excess"]["area_m2"] > 0
    # One body outside the source is one consolidated defect, not one per face that sees it.
    excess_defects = [t for t in result["defects"] if t.startswith("candidate surface beyond")]
    assert len(excess_defects) == 1 and "largest seen from" in excess_defects[0]


def test_face_depth_audit_excludes_source_opening_columns_from_the_face_verdict() -> None:
    source = _box_points((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8))
    opening = SimpleNamespace(
        face="-Y", kind="opening", open_face=False, center_uv=(0.0, 0.4), size_uv=(0.4, 0.4)
    )
    plain = face_depth_audit(source, source.copy(), stack_pitch=0.02)
    masked = face_depth_audit(source, source.copy(), stack_pitch=0.02, source_openings=[opening])
    front_plain = next(row for row in plain["faces"] if row["face"] == "-Y front")
    front_masked = next(row for row in masked["faces"] if row["face"] == "-Y front")
    assert front_masked["opening_columns"] > 0
    assert front_masked["columns_compared"] < front_plain["columns_compared"]
    other = next(row for row in masked["faces"] if row["face"] == "+X right")
    assert other["opening_columns"] == 0


def test_face_position_flags_a_face_moved_beyond_one_stack_pitch_but_bounds_never_do() -> None:
    source = _box_points((-0.5, -0.5, 0.0), (0.5, 0.5, 0.8))
    candidate = _box_points((-0.5, -0.5, 0.0), (0.5, 0.5, 0.9))  # top face 100 mm higher
    result = face_depth_audit(source, candidate, stack_pitch=0.03)
    top = next(row for row in result["faces"] if row["face"] == "+Z top")
    assert 0.09 <= top["plane_offset_m"] <= 0.11
    assert any(text.startswith("+Z top face built 100 mm outward") for text in top["defects"])
    front = next(row for row in result["faces"] if row["face"] == "-Y front")
    assert abs(front["plane_offset_m"]) < 0.01 and not front["defects"]
    # Bounds are a reading only: a reach difference is never a defect by itself.
    envelope = envelope_comparison(source, candidate, stack_pitch=0.03)
    assert [axis["axis"] for axis in envelope["axes"]] == ["X", "Y", "Z"]
    assert envelope["axes"][2]["delta_high_m"] == 0.1 and envelope["defects"] == []


def test_parsers_read_micro_slabs_and_face_feature_lines() -> None:
    text = "\n".join(
        (
            "AXIAL",
            "[MICRO_SLICE_BOUNDS_Z192 parent=SLICE_STACK24 axis=Z stack_slices=16]",
            "semantics: islands",
            "s0:4 mz=0:24 xy=10:50,20:60(f=0.5;p=10,20 50,20 50,60 10,60)|70:80,20:30",
            "[FACE_FEATURES faces=-Y pitch=0.002 shown=1 levels=L1_face]",
            "semantics: compact protrusions",
            "coords: world metres",
            "-Y c(x,z)=(+0.293,+0.076) size=82x74mm h=30/22mm L1",
            "[SLICE_STACK24 axis=Z]",
        )
    )
    slabs = parse_micro_slabs(text)
    assert len(slabs) == 1 and len(slabs[0].islands) == 2
    assert slabs[0].islands[0].contour == ((10, 20), (50, 20), (50, 60), (10, 60))
    features = parse_face_features(text)
    assert len(features) == 1 and isinstance(features[0], FaceFeature)
    assert features[0].face == "-Y" and abs(features[0].height_max_m - 0.030) < 1e-9
