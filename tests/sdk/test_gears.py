# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

import importlib

import pytest

import sdk

cq = pytest.importorskip("cadquery")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("gear_ctor", "kwargs", "expected_type_name"),
    [
        ("SpurGear", {"module": 0.5, "teeth_number": 12, "width": 3.0, "bore_d": 2.0}, "Compound"),
        ("RingGear", {"module": 0.5, "teeth_number": 24, "width": 3.0, "rim_width": 1.5}, "Solid"),
        (
            "HerringbonePlanetaryGearset",
            {
                "module": 0.5,
                "sun_teeth_number": 8,
                "planet_teeth_number": 6,
                "width": 3.0,
                "rim_width": 1.0,
                "n_planets": 2,
                "helix_angle": 15.0,
            },
            "Compound",
        ),
        (
            "BevelGearPair",
            {
                "module": 0.5,
                "gear_teeth": 12,
                "pinion_teeth": 8,
                "face_width": 2.0,
                "axis_angle": 90.0,
            },
            "Compound",
        ),
        ("RackGear", {"module": 0.5, "length": 10.0, "width": 3.0, "height": 2.0}, "Solid"),
        (
            "Worm",
            {"module": 0.5, "lead_angle": 20.0, "n_threads": 1, "length": 8.0},
            "Solid",
        ),
        (
            "CrossedGearPair",
            {
                "module": 0.5,
                "gear1_teeth_number": 12,
                "gear2_teeth_number": 12,
                "gear1_width": 2.0,
                "gear2_width": 2.0,
                "shaft_angle": 90.0,
                "gear1_helix_angle": 30.0,
            },
            "Compound",
        ),
        (
            "HyperbolicGearPair",
            {"module": 0.5, "gear1_teeth_number": 12, "width": 2.0, "shaft_angle": 40.0},
            "Compound",
        ),
    ],
)
def test_representative_vendored_gears_build(
    gear_ctor: str,
    kwargs: dict[str, float | int],
    expected_type_name: str,
) -> None:
    gear_cls = getattr(sdk, gear_ctor)
    gear = gear_cls(**kwargs)
    body = gear.build()

    assert isinstance(body, cq.Shape)
    assert type(body).__name__ == expected_type_name


@pytest.mark.slow
def test_workplane_gear_plugin_and_top_level_helpers() -> None:
    spur = sdk.SpurGear(module=0.5, teeth_number=12, width=3.0, bore_d=2.0)

    wp = cq.Workplane("XY").gear(spur)
    assert isinstance(wp, cq.Workplane)
    assert len(wp.vals()) == 1

    helper_wp = sdk.gear(cq.Workplane("XY"), spur)
    assert isinstance(helper_wp, cq.Workplane)
    assert len(helper_wp.vals()) == 1

    union_wp = cq.Workplane("XY").circle(4.0).extrude(1.0).addGear(spur)
    assert isinstance(union_wp, cq.Workplane)
    assert union_wp.solids().vals()


def test_sdk_gears_module_reexports_classes() -> None:
    from sdk._extensions.cadquery.gears import RingGear, SpurGear

    current_sdk = importlib.import_module("sdk")

    assert SpurGear.__name__ == current_sdk.SpurGear.__name__
    assert RingGear.__name__ == current_sdk.RingGear.__name__
