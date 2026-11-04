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
    BarHandleGeometry,
    BottleGeometry,
    BoxGeometry,
    BridgeHandleGeometry,
    CabinetCarcassGeometry,
    CapGeometry,
    JarGeometry,
    PanelGeometry,
    ShelvingUnitGeometry,
    TaperedLegGeometry,
    boolean_union,
)


def _bounds(geometry):
    xs = [v[0] for v in geometry.vertices]
    ys = [v[1] for v in geometry.vertices]
    zs = [v[2] for v in geometry.vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


# --- containers ---------------------------------------------------------------


def test_bottle_dimensions_and_manifold_validity() -> None:
    bottle = BottleGeometry(height=0.24, body_radius=0.035)
    mins, maxs = _bounds(bottle)

    assert mins[2] == pytest.approx(0.0)
    assert maxs[2] == pytest.approx(0.24)
    # Body radius holds within spline overshoot tolerance.
    assert maxs[0] == pytest.approx(0.035, abs=0.004)
    # Valid manifold solid: booleans work on it.
    merged = boolean_union(bottle, BoxGeometry((0.01, 0.01, 0.01)).translate(0, 0, 0.005))
    assert len(merged.faces) > 0


def test_bottle_rejects_bad_proportions() -> None:
    with pytest.raises(ValueError, match="neck_radius"):
        BottleGeometry(body_radius=0.03, neck_radius=0.03)
    with pytest.raises(ValueError, match="smaller than height"):
        BottleGeometry(height=0.05, neck_height=0.03, shoulder_height=0.03)


def test_jar_and_cap_fit_together() -> None:
    jar = JarGeometry(height=0.12, radius=0.04, mouth_radius=0.032)
    cap = CapGeometry(radius=0.034, height=0.014)

    jar_mins, jar_maxs = _bounds(jar)
    cap_mins, cap_maxs = _bounds(cap)
    assert jar_maxs[2] == pytest.approx(0.12)
    assert cap_maxs[0] == pytest.approx(0.034, abs=1e-6)
    assert cap_maxs[0] > 0.032  # lid covers the mouth
    assert jar_maxs[0] == pytest.approx(0.04, abs=1e-6)


def test_jar_rejects_wide_mouth() -> None:
    with pytest.raises(ValueError, match="mouth_radius"):
        JarGeometry(radius=0.03, mouth_radius=0.03)


# --- shelving ------------------------------------------------------------------


def test_shelving_levels_and_footprint() -> None:
    unit = ShelvingUnitGeometry(0.9, 0.4, 1.8, shelf_count=4, back_panel=True, kick_height=0.1)
    mins, maxs = _bounds(unit)

    assert len(unit.shelf_levels) == 4
    assert unit.shelf_levels[0] == pytest.approx(0.1)
    assert maxs[0] - mins[0] == pytest.approx(0.9)
    assert mins[2] == pytest.approx(0.0)
    assert all(
        unit.shelf_levels[i] < unit.shelf_levels[i + 1] for i in range(len(unit.shelf_levels) - 1)
    )


def test_shelving_explicit_heights_and_validation() -> None:
    unit = ShelvingUnitGeometry(shelf_heights=[0.2, 0.8, 1.4])
    assert unit.shelf_levels == [0.2, 0.8, 1.4]

    with pytest.raises(ValueError, match="within the unit height"):
        ShelvingUnitGeometry(shelf_heights=[2.5])
    with pytest.raises(ValueError, match="post_size"):
        ShelvingUnitGeometry(width=0.08, depth=0.08, post_size=0.05)


def test_shelf_lip_rises_above_top_board() -> None:
    plain = ShelvingUnitGeometry(shelf_count=2)
    lipped = ShelvingUnitGeometry(shelf_count=2, shelf_lip=0.03)

    assert _bounds(lipped)[1][2] > _bounds(plain)[1][2]


# --- furniture -----------------------------------------------------------------


def test_panel_dimensions_and_edge_rounding() -> None:
    panel = PanelGeometry(0.6, 0.4, 0.018, edge_radius=0.004)
    mins, maxs = _bounds(panel)

    assert maxs[0] - mins[0] == pytest.approx(0.6, abs=1e-6)
    assert maxs[1] - mins[1] == pytest.approx(0.4, abs=1e-6)
    assert maxs[2] - mins[2] == pytest.approx(0.018, abs=1e-6)
    # Rounded corner: no vertex exactly at the sharp corner.
    corner_hits = [
        v for v in panel.vertices if abs(abs(v[0]) - 0.3) < 1e-9 and abs(abs(v[1]) - 0.2) < 1e-9
    ]
    assert not corner_hits

    with pytest.raises(ValueError, match="edge_radius"):
        PanelGeometry(0.1, 0.1, 0.01, edge_radius=0.06)


def test_tapered_leg_profile() -> None:
    leg = TaperedLegGeometry(height=0.4, top_radius=0.02, bottom_radius=0.012)
    mins, maxs = _bounds(leg)

    assert mins[2] == pytest.approx(0.0)
    assert maxs[2] == pytest.approx(0.4)
    assert maxs[0] <= 0.022 + 1e-6

    with pytest.raises(ValueError, match="foot_height"):
        TaperedLegGeometry(height=0.1, foot_height=0.2)


def test_bar_handle_styles() -> None:
    bar = BarHandleGeometry(length=0.16, style="bar")
    arch = BarHandleGeometry(length=0.16, style="arch")

    for handle in (bar, arch):
        mins, maxs = _bounds(handle)
        assert maxs[0] - mins[0] == pytest.approx(0.16, abs=0.012)
        assert maxs[2] > 0.02  # stands off the mounting face

    with pytest.raises(ValueError, match="style"):
        BarHandleGeometry(style="round")
    with pytest.raises(ValueError, match="standoff"):
        BarHandleGeometry(radius=0.02, standoff=0.03)


def test_bridge_handle_is_symmetric_and_has_coplanar_mounts() -> None:
    handle = BridgeHandleGeometry(
        mount_span=0.24,
        rise=0.10,
        section_width=0.024,
        section_depth=0.034,
        shoulder_radius=0.035,
        crown_height=0.012,
        corner_radius=0.004,
    )
    mins, maxs = _bounds(handle)
    points = {(round(x, 8), round(y, 8), round(z, 8)) for x, y, z in handle.vertices}

    assert mins[0] == pytest.approx(-maxs[0], abs=1e-8)
    assert mins[1] == pytest.approx(-maxs[1], abs=1e-8)
    assert mins[2] == pytest.approx(0.0, abs=1e-8)
    assert maxs[2] > 0.112
    assert all((-x, y, z) in points for x, y, z in points)

    with pytest.raises(ValueError, match="half the mount_span"):
        BridgeHandleGeometry(mount_span=0.1, shoulder_radius=0.06)


def test_cabinet_carcass_interior_metadata() -> None:
    carcass = CabinetCarcassGeometry(
        0.6, 0.45, 0.75, panel_thickness=0.018, shelf_count=2, kick_height=0.08
    )
    mins, maxs = _bounds(carcass)

    assert maxs[0] - mins[0] == pytest.approx(0.6)
    assert maxs[2] == pytest.approx(0.75)
    assert carcass.interior_width == pytest.approx(0.6 - 0.036)
    assert len(carcass.shelf_levels) == 2
    assert 0.08 < carcass.shelf_levels[0] < carcass.shelf_levels[1] < 0.75

    with pytest.raises(ValueError, match="kick_height"):
        CabinetCarcassGeometry(height=0.1, kick_height=0.09, panel_thickness=0.018)
