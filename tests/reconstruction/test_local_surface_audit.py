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

from types import SimpleNamespace

import numpy as np

from usdcraft.reconstruction.local_surface_audit import (
    MAX_REGIONS,
    MAX_SIDE_CELLS,
    format_local_surfaces,
    local_surface_audit,
)


def _patch():
    u, v = np.meshgrid(np.linspace(-0.02, 0.02, 81), np.linspace(-0.02, 0.02, 81))
    return np.column_stack((u.ravel(), np.zeros(u.size), v.ravel()))


def _feature():
    return SimpleNamespace(face="-Y", center_uv=(0.0, 0.0), size_uv=(0.04, 0.04))


def test_local_profile_separates_offset_from_shape() -> None:
    source = _patch()
    shifted = source.copy()
    shifted[:, 1] -= 0.006
    row = local_surface_audit(source, shifted, source_features=[_feature()])["regions"][0]
    assert row["status"] == "measured"
    assert row["depth_offset_mm"] == 6
    assert row["surface_delta_p90_mm"] == 6
    assert row["centred_profile_p90_mm"] == 0

    curved = source.copy()
    curved[:, 1] = 0.008 * (1 - (curved[:, 0] / 0.02) ** 2)
    row = local_surface_audit(curved, source, source_features=[_feature()])["regions"][0]
    assert row["status"] == "measured"
    assert row["centred_profile_p90_mm"] > 2
    assert row["source_column_coverage"] == 1


def test_local_missing_source_is_unassessed_and_not_a_defect() -> None:
    audit = local_surface_audit(np.empty((0, 3)), _patch(), source_features=[_feature()])
    assert audit["regions"][0]["reason"] == "source_surface_unavailable"
    assert audit["regions"][0]["status"] == "unassessed"
    assert "defects" not in audit
    assert "unassessed" in "\n".join(format_local_surfaces(audit))


def test_local_missing_candidate_records_coverage_without_inventing_free_space() -> None:
    audit = local_surface_audit(_patch(), np.empty((0, 3)), source_features=[_feature()])
    row = audit["regions"][0]
    assert row["source_column_coverage"] == 0
    assert row["missing_candidate_columns"] > 0
    assert row["reason"] == "insufficient_common_surface"
    assert "centred_profile_p90_mm" not in row


def test_local_region_excludes_unrelated_geometry_and_bounds_work() -> None:
    source = _patch()
    other = source + np.array([10, -30, 20])
    plain = local_surface_audit(source, source, source_features=[_feature()])
    combined = local_surface_audit(
        np.vstack([source, other]),
        np.vstack([source, other * 2]),
        source_features=[_feature()],
    )
    assert plain == combined
    audit = local_surface_audit(source, source, source_features=[_feature()] * (MAX_REGIONS + 3))
    assert len(audit["regions"]) == MAX_REGIONS
    assert audit["omitted_regions"] == 3
    assert all(row["source_columns"] <= MAX_SIDE_CELLS**2 for row in audit["regions"])


def test_local_opening_does_not_score_unknown_backside_or_an_open_face() -> None:
    opening = SimpleNamespace(**vars(_feature()), open_face=False)
    open_face = SimpleNamespace(**vars(_feature()), open_face=True)
    source = _patch()
    # First surface facing -Y is y=0; an unseen surface behind it does not change the reading.
    candidate = np.vstack([source, source + np.array([0, 0.05, 0])])
    audit = local_surface_audit(source, candidate, source_openings=[opening, open_face])
    assert len(audit["regions"]) == 1
    assert audit["regions"][0]["anchor"] == "opening:0"
    assert audit["regions"][0]["surface_delta_p90_mm"] == 0
