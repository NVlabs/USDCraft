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

from typing import List, Optional, Sequence

from .primitives import BoxGeometry, MeshGeometry


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


class ShelvingUnitGeometry(MeshGeometry):
    """
    Parametric shelving / rack unit: four posts, evenly spaced (or explicit)
    shelf boards, optional back panel and kick plate. Covers supermarket
    gondolas, storage racks, and open bookcases.

    Z-up with the floor at z=0, centered on X (width) and Y (depth). The
    whole unit is one rigid mesh — model doors or other moving parts as
    separate parts.
    """

    def __init__(
        self,
        width: float = 0.9,
        depth: float = 0.4,
        height: float = 1.8,
        *,
        shelf_count: int = 4,
        shelf_heights: Optional[Sequence[float]] = None,
        shelf_thickness: float = 0.028,
        post_size: float = 0.045,
        back_panel: bool = False,
        back_thickness: float = 0.012,
        kick_height: float = 0.0,
        shelf_lip: float = 0.0,
        lip_thickness: float = 0.006,
        top_shelf: bool = True,
    ):
        super().__init__()
        width = _positive(width, name="width")
        depth = _positive(depth, name="depth")
        height = _positive(height, name="height")
        shelf_thickness = _positive(shelf_thickness, name="shelf_thickness")
        post_size = _positive(post_size, name="post_size")
        back_thickness = _positive(back_thickness, name="back_thickness")
        lip_thickness = _positive(lip_thickness, name="lip_thickness")
        kick_height = max(0.0, float(kick_height))
        shelf_lip = max(0.0, float(shelf_lip))
        if post_size * 2.0 >= min(width, depth):
            raise ValueError("post_size is too large for the unit footprint")

        if shelf_heights is not None:
            levels = sorted(float(z) for z in shelf_heights)
            if not levels:
                raise ValueError("shelf_heights must not be empty")
            if levels[0] < 0.0 or levels[-1] > height - shelf_thickness:
                raise ValueError("shelf_heights must lie within the unit height")
        else:
            shelf_count = int(shelf_count)
            if shelf_count < 1:
                raise ValueError("shelf_count must be >= 1")
            bottom = max(kick_height, shelf_thickness * 0.5)
            top = height - shelf_thickness
            if shelf_count == 1:
                levels = [bottom]
            else:
                # top_shelf=True puts the last board flush with the unit top.
                step = (top - bottom) / (shelf_count - 1 if top_shelf else shelf_count)
                levels = [bottom + step * i for i in range(shelf_count)]

        half_w = width / 2.0
        half_d = depth / 2.0

        def add_box(size, center) -> None:
            self.merge(BoxGeometry(size).translate(*center))

        # Posts at the four corners.
        post_centers = [
            (half_w - post_size / 2.0, half_d - post_size / 2.0),
            (-(half_w - post_size / 2.0), half_d - post_size / 2.0),
            (half_w - post_size / 2.0, -(half_d - post_size / 2.0)),
            (-(half_w - post_size / 2.0), -(half_d - post_size / 2.0)),
        ]
        for px, py in post_centers:
            add_box((post_size, post_size, height), (px, py, height / 2.0))

        # Shelf boards (span between posts).
        shelf_width = width - post_size * 0.5
        shelf_depth = depth - (back_thickness if back_panel else 0.0)
        shelf_y = -back_thickness / 2.0 if back_panel else 0.0
        for level in levels:
            add_box(
                (shelf_width, shelf_depth, shelf_thickness),
                (0.0, shelf_y, level + shelf_thickness / 2.0),
            )
            if shelf_lip > 0.0:
                add_box(
                    (shelf_width, lip_thickness, shelf_lip),
                    (
                        0.0,
                        shelf_y - shelf_depth / 2.0 + lip_thickness / 2.0,
                        level + shelf_thickness + shelf_lip / 2.0,
                    ),
                )

        if back_panel:
            add_box(
                (width, back_thickness, height),
                (0.0, half_d - back_thickness / 2.0, height / 2.0),
            )

        if kick_height > 0.0:
            add_box(
                (width - post_size, depth - post_size, kick_height),
                (0.0, 0.0, kick_height / 2.0),
            )

        self.shelf_levels: List[float] = [float(level) for level in levels]
