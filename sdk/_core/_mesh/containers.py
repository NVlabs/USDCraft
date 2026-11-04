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

from typing import List, Tuple

from .common import sample_catmull_rom_spline_2d
from .primitives import LatheGeometry

Vec2 = Tuple[float, float]


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


class BottleGeometry(LatheGeometry):
    """
    Parametric bottle: cylindrical body, smooth shoulder, neck, and mouth lip.
    Z-up with the base at z=0. Authored as a solid (good for both visuals and
    collisions); pair with a CapGeometry for a closed product.
    """

    def __init__(
        self,
        height: float = 0.24,
        body_radius: float = 0.035,
        *,
        neck_radius: float = 0.013,
        neck_height: float = 0.025,
        shoulder_height: float = 0.05,
        base_fillet: float = 0.004,
        lip_bead: float = 0.0015,
        waist: float = 0.0,
        segments: int = 48,
    ):
        height = _positive(height, name="height")
        body_radius = _positive(body_radius, name="body_radius")
        neck_radius = _positive(neck_radius, name="neck_radius")
        neck_height = _positive(neck_height, name="neck_height")
        shoulder_height = _positive(shoulder_height, name="shoulder_height")
        base_fillet = max(0.0, float(base_fillet))
        lip_bead = max(0.0, float(lip_bead))
        waist = max(0.0, float(waist))
        if neck_radius >= body_radius:
            raise ValueError("neck_radius must be smaller than body_radius")
        if neck_height + shoulder_height >= height:
            raise ValueError("neck_height + shoulder_height must be smaller than height")
        if base_fillet >= body_radius:
            raise ValueError("base_fillet must be smaller than body_radius")

        body_top = height - neck_height - shoulder_height
        waist_z = base_fillet + (body_top - base_fillet) * 0.45

        # Smooth body + shoulder profile through key points.
        spline_points: List[Vec2] = [
            (body_radius, base_fillet),
            (body_radius - waist, waist_z),
            (body_radius, body_top),
            (
                (body_radius + neck_radius) / 2.0,
                body_top + shoulder_height * 0.62,
            ),
            (neck_radius, body_top + shoulder_height),
        ]
        shoulder = sample_catmull_rom_spline_2d(spline_points, samples_per_segment=8)

        profile: List[Vec2] = [(0.0, 0.0)]
        if base_fillet > 0.0:
            profile.append((body_radius - base_fillet, 0.0))
        profile.extend(shoulder)
        neck_top = height - lip_bead * 2.0 if lip_bead > 0.0 else height
        profile.append((neck_radius, neck_top))
        if lip_bead > 0.0:
            lip = sample_catmull_rom_spline_2d(
                [
                    (neck_radius, neck_top),
                    (neck_radius + lip_bead, height - lip_bead),
                    (neck_radius, height),
                ],
                samples_per_segment=4,
            )
            profile.extend(lip[1:])
        profile.append((0.0, height))

        super().__init__(profile, segments=segments, closed=True)


class JarGeometry(LatheGeometry):
    """
    Wide-mouth jar: straight body, short rounded shoulder, wide mouth ring.
    Z-up with the base at z=0. Solid by default; the mouth ring slightly
    overhangs so a CapGeometry seats naturally on top.
    """

    def __init__(
        self,
        height: float = 0.12,
        radius: float = 0.04,
        *,
        mouth_radius: float = 0.032,
        shoulder_height: float = 0.018,
        base_fillet: float = 0.003,
        mouth_ring_height: float = 0.01,
        segments: int = 48,
    ):
        height = _positive(height, name="height")
        radius = _positive(radius, name="radius")
        mouth_radius = _positive(mouth_radius, name="mouth_radius")
        shoulder_height = _positive(shoulder_height, name="shoulder_height")
        base_fillet = max(0.0, float(base_fillet))
        mouth_ring_height = _positive(mouth_ring_height, name="mouth_ring_height")
        if mouth_radius >= radius:
            raise ValueError("mouth_radius must be smaller than radius")
        if shoulder_height + mouth_ring_height >= height:
            raise ValueError("shoulder_height + mouth_ring_height must be smaller than height")
        if base_fillet >= radius:
            raise ValueError("base_fillet must be smaller than radius")

        body_top = height - shoulder_height - mouth_ring_height
        shoulder = sample_catmull_rom_spline_2d(
            [
                (radius, body_top),
                ((radius + mouth_radius) / 2.0, body_top + shoulder_height * 0.7),
                (mouth_radius, body_top + shoulder_height),
            ],
            samples_per_segment=6,
        )

        profile: List[Vec2] = [(0.0, 0.0)]
        if base_fillet > 0.0:
            profile.append((radius - base_fillet, 0.0))
            profile.append((radius, base_fillet))
        else:
            profile.append((radius, 0.0))
        profile.append((radius, body_top))
        profile.extend(shoulder[1:])
        profile.append((mouth_radius, height))
        profile.append((0.0, height))

        super().__init__(profile, segments=segments, closed=True)


class CapGeometry(LatheGeometry):
    """
    Screw-cap / lid blank with a rounded top edge and an optional skirt taper.
    Z-up with the open side at z=0; translate onto the bottle or jar mouth.
    """

    def __init__(
        self,
        radius: float = 0.015,
        height: float = 0.014,
        *,
        top_fillet: float = 0.003,
        skirt_taper: float = 0.0,
        segments: int = 48,
    ):
        radius = _positive(radius, name="radius")
        height = _positive(height, name="height")
        top_fillet = max(0.0, float(top_fillet))
        skirt_taper = max(0.0, float(skirt_taper))
        if top_fillet >= min(radius, height):
            raise ValueError("top_fillet must be smaller than radius and height")
        if skirt_taper >= radius:
            raise ValueError("skirt_taper must be smaller than radius")

        profile: List[Vec2] = [(0.0, 0.0), (radius - skirt_taper, 0.0), (radius, height * 0.25)]
        if top_fillet > 0.0:
            rounded = sample_catmull_rom_spline_2d(
                [
                    (radius, height - top_fillet),
                    (radius - top_fillet * 0.3, height - top_fillet * 0.3),
                    (radius - top_fillet, height),
                ],
                samples_per_segment=5,
            )
            profile.append((radius, height - top_fillet))
            profile.extend(rounded[1:])
        else:
            profile.append((radius, height))
        profile.append((0.0, height))

        super().__init__(profile, segments=segments, closed=True)
