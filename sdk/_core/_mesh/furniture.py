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

from math import cos, pi, sin
from typing import List, Tuple

from .common import rounded_rect_profile, sample_catmull_rom_spline_2d
from .primitives import (
    BoxGeometry,
    ExtrudeGeometry,
    LatheGeometry,
    MeshGeometry,
    _adopt_mesh_geometry,
)
from .sweeps import PipeGeometry, tube_from_spline_points

Vec2 = Tuple[float, float]


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


class PanelGeometry(MeshGeometry):
    """
    Furniture board: a panel of ``width`` x ``height`` face dimensions and
    ``thickness``, with rounded edges along the thickness axis (real boards
    never have zero-radius edges). Lies in the XY plane, centered at the
    origin, thickness along Z — rotate/translate into place.
    """

    def __init__(
        self,
        width: float,
        height: float,
        thickness: float,
        *,
        edge_radius: float = 0.002,
        corner_segments: int = 4,
    ):
        super().__init__()
        width = _positive(width, name="width")
        height = _positive(height, name="height")
        thickness = _positive(thickness, name="thickness")
        edge_radius = float(edge_radius)
        if edge_radius < 0.0:
            raise ValueError("edge_radius must be >= 0")
        if edge_radius * 2.0 >= min(width, height):
            raise ValueError("edge_radius too large for the panel face")

        if edge_radius > 0.0:
            profile = rounded_rect_profile(
                width, height, edge_radius, corner_segments=corner_segments
            )
        else:
            half_w, half_h = width / 2.0, height / 2.0
            profile = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        board = ExtrudeGeometry.centered(profile, thickness)
        _adopt_mesh_geometry(self, board)


class TaperedLegGeometry(LatheGeometry):
    """
    Round furniture leg tapering from ``top_radius`` to ``bottom_radius``,
    with an optional foot pad. Z-up with the floor at z=0; the part above
    z=height attaches to the furniture body.
    """

    def __init__(
        self,
        height: float = 0.4,
        top_radius: float = 0.022,
        bottom_radius: float = 0.012,
        *,
        curve: float = 0.0,
        foot_height: float = 0.0,
        segments: int = 32,
    ):
        height = _positive(height, name="height")
        top_radius = _positive(top_radius, name="top_radius")
        bottom_radius = _positive(bottom_radius, name="bottom_radius")
        curve = float(curve)
        foot_height = max(0.0, float(foot_height))
        if foot_height >= height:
            raise ValueError("foot_height must be smaller than height")

        leg_base = foot_height
        mid_radius = (top_radius + bottom_radius) / 2.0 + curve
        if mid_radius <= 0.0:
            raise ValueError("curve pulls the leg radius below zero")
        side = sample_catmull_rom_spline_2d(
            [
                (bottom_radius, leg_base),
                (mid_radius, leg_base + (height - leg_base) * 0.5),
                (top_radius, height),
            ],
            samples_per_segment=8,
        )

        profile: List[Vec2] = [(0.0, 0.0)]
        if foot_height > 0.0:
            profile.append((bottom_radius * 1.15, 0.0))
            profile.append((bottom_radius * 1.15, foot_height))
        profile.extend(side)
        profile.append((0.0, height))

        super().__init__(profile, segments=segments, closed=True)


class BarHandleGeometry(MeshGeometry):
    """
    Cabinet/drawer bar handle: a grip bar on two standoff posts, or a smooth
    arch ("bow") handle. The grip runs along X, posts point down -Z onto the
    mounting face at z=0. Centered on the X axis.
    """

    def __init__(
        self,
        length: float = 0.16,
        *,
        radius: float = 0.005,
        standoff: float = 0.03,
        style: str = "bar",
        radial_segments: int = 16,
    ):
        super().__init__()
        length = _positive(length, name="length")
        radius = _positive(radius, name="radius")
        standoff = _positive(standoff, name="standoff")
        if style not in ("bar", "arch"):
            raise ValueError("style must be 'bar' or 'arch'")
        if radius * 2.0 >= standoff:
            raise ValueError("standoff must exceed the handle diameter")
        half = length / 2.0
        post_inset = min(radius * 2.0, half * 0.5)

        if style == "bar":
            grip = tube_from_spline_points(
                [(-half, 0.0, standoff), (half, 0.0, standoff)],
                radius=radius,
                radial_segments=radial_segments,
                samples_per_segment=1,
            )
            self.merge(grip)
            for px in (-(half - post_inset), half - post_inset):
                post = tube_from_spline_points(
                    [(px, 0.0, 0.0), (px, 0.0, standoff)],
                    radius=radius * 0.85,
                    radial_segments=radial_segments,
                    samples_per_segment=1,
                )
                self.merge(post)
        else:
            arch = tube_from_spline_points(
                [
                    (-half, 0.0, 0.0),
                    (-half * 0.82, 0.0, standoff * 0.82),
                    (0.0, 0.0, standoff),
                    (half * 0.82, 0.0, standoff * 0.82),
                    (half, 0.0, 0.0),
                ],
                radius=radius,
                radial_segments=radial_segments,
                samples_per_segment=8,
            )
            self.merge(arch)


class BridgeHandleGeometry(MeshGeometry):
    """Symmetric bridge pull with vertical feet and circular shoulders.

    Unlike a free-form spline, this family has an exact bilateral symmetry plane,
    two coplanar mounting contacts, an optional symmetric crown, and bounded shoulder radii.
    The centerline lies in XZ, the mounting plane is z=0, and ``mount_span`` is the
    center-to-center distance between the feet. ``section_width`` is thickness in
    the bend plane; ``section_depth`` is thickness along Y.
    """

    def __init__(
        self,
        mount_span: float = 0.16,
        rise: float = 0.06,
        *,
        section_width: float = 0.012,
        section_depth: float = 0.018,
        shoulder_radius: float = 0.025,
        crown_height: float = 0.0,
        corner_radius: float = 0.003,
        shoulder_segments: int = 8,
        top_segments: int = 12,
        corner_segments: int = 4,
    ):
        super().__init__()
        mount_span = _positive(mount_span, name="mount_span")
        rise = _positive(rise, name="rise")
        section_width = _positive(section_width, name="section_width")
        section_depth = _positive(section_depth, name="section_depth")
        shoulder_radius = _positive(shoulder_radius, name="shoulder_radius")
        crown_height = float(crown_height)
        corner_radius = float(corner_radius)
        shoulder_segments = int(shoulder_segments)
        top_segments = int(top_segments)
        corner_segments = int(corner_segments)
        if shoulder_radius >= mount_span / 2.0:
            raise ValueError("shoulder_radius must be smaller than half the mount_span")
        if shoulder_radius >= rise:
            raise ValueError("shoulder_radius must be smaller than rise")
        if section_width >= mount_span:
            raise ValueError("section_width must be smaller than mount_span")
        if crown_height < 0.0 or crown_height >= rise:
            raise ValueError("crown_height must be non-negative and smaller than rise")
        if corner_radius < 0.0 or corner_radius * 2.0 >= min(section_width, section_depth):
            raise ValueError("corner_radius must be non-negative and fit inside the section")
        if shoulder_segments < 2:
            raise ValueError("shoulder_segments must be >= 2")
        if top_segments < 1:
            raise ValueError("top_segments must be >= 1")
        if corner_segments < 1:
            raise ValueError("corner_segments must be >= 1")

        half = mount_span / 2.0
        shoulder_z = rise - shoulder_radius
        path = [(-half, 0.0, 0.0), (-half, 0.0, shoulder_z)]

        left_cx = -half + shoulder_radius
        for i in range(1, shoulder_segments + 1):
            angle = pi - (pi / 2.0) * i / shoulder_segments
            path.append(
                (
                    left_cx + shoulder_radius * cos(angle),
                    0.0,
                    shoulder_z + shoulder_radius * sin(angle),
                )
            )

        right_cx = half - shoulder_radius
        for i in range(1, top_segments + 1):
            t = i / top_segments
            path.append(
                (
                    left_cx + (right_cx - left_cx) * t,
                    0.0,
                    rise + crown_height * sin(pi * t) ** 2,
                )
            )
        for i in range(1, shoulder_segments + 1):
            angle = pi / 2.0 - (pi / 2.0) * i / shoulder_segments
            path.append(
                (
                    right_cx + shoulder_radius * cos(angle),
                    0.0,
                    shoulder_z + shoulder_radius * sin(angle),
                )
            )
        path.append((half, 0.0, 0.0))

        profile = rounded_rect_profile(
            section_width,
            section_depth,
            corner_radius,
            corner_segments=corner_segments,
        )
        self.merge(PipeGeometry(profile, path, cap=True, up_hint=(0.0, 1.0, 0.0)))

        self.mount_span = mount_span
        self.rise = rise
        self.section_width = section_width
        self.section_depth = section_depth
        self.shoulder_radius = shoulder_radius
        self.crown_height = crown_height


class CabinetCarcassGeometry(MeshGeometry):
    """
    Open-front panel-furniture carcass: two sides, top, bottom, back, and
    optional fixed shelves. The standard starting point for cabinets,
    nightstands, sideboards, and fridges — author doors and drawers as
    separate parts hinged onto this body.

    Z-up with the floor at z=0, centered on X and Y; the open front faces -Y.
    """

    def __init__(
        self,
        width: float = 0.6,
        depth: float = 0.45,
        height: float = 0.75,
        *,
        panel_thickness: float = 0.018,
        back_panel: bool = True,
        back_thickness: float = 0.006,
        shelf_count: int = 0,
        kick_height: float = 0.0,
    ):
        super().__init__()
        width = _positive(width, name="width")
        depth = _positive(depth, name="depth")
        height = _positive(height, name="height")
        panel_thickness = _positive(panel_thickness, name="panel_thickness")
        back_thickness = _positive(back_thickness, name="back_thickness")
        shelf_count = int(shelf_count)
        if shelf_count < 0:
            raise ValueError("shelf_count must be >= 0")
        kick_height = max(0.0, float(kick_height))
        if panel_thickness * 2.0 >= min(width, height):
            raise ValueError("panel_thickness too large for the carcass")
        if kick_height + panel_thickness * 2.0 >= height:
            raise ValueError("kick_height leaves no interior height")

        half_w = width / 2.0
        body_depth = depth - (back_thickness if back_panel else 0.0)
        body_y = -back_thickness / 2.0 if back_panel else 0.0
        inner_w = width - panel_thickness * 2.0
        bottom_z = kick_height

        def add_box(size, center) -> None:
            self.merge(BoxGeometry(size).translate(*center))

        # Sides.
        for sx in (-(half_w - panel_thickness / 2.0), half_w - panel_thickness / 2.0):
            add_box(
                (panel_thickness, body_depth, height - bottom_z),
                (sx, body_y, bottom_z + (height - bottom_z) / 2.0),
            )
        # Bottom and top.
        add_box(
            (inner_w, body_depth, panel_thickness),
            (0.0, body_y, bottom_z + panel_thickness / 2.0),
        )
        add_box(
            (inner_w, body_depth, panel_thickness), (0.0, body_y, height - panel_thickness / 2.0)
        )
        # Back.
        if back_panel:
            add_box(
                (width, back_thickness, height - bottom_z),
                (0.0, depth / 2.0 - back_thickness / 2.0, bottom_z + (height - bottom_z) / 2.0),
            )
        # Kick plinth.
        if kick_height > 0.0:
            add_box(
                (inner_w, body_depth - panel_thickness, kick_height),
                (0.0, body_y + panel_thickness / 2.0, kick_height / 2.0),
            )
        # Fixed shelves, evenly spaced in the interior.
        interior_bottom = bottom_z + panel_thickness
        interior_top = height - panel_thickness
        shelf_levels: List[float] = []
        for i in range(shelf_count):
            level = interior_bottom + (interior_top - interior_bottom) * (i + 1) / (shelf_count + 1)
            shelf_levels.append(level)
            add_box(
                (inner_w, body_depth - panel_thickness, panel_thickness),
                (0.0, body_y + panel_thickness / 2.0, level),
            )

        self.shelf_levels = shelf_levels
        self.interior_width = inner_w
        self.interior_depth = body_depth - panel_thickness
