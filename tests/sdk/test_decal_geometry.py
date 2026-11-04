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

import math

import pytest

from sdk import DecalGeometry, TextureCanvas


def _canvas() -> TextureCanvas:
    c = TextureCanvas(32, 32, background=(1.0, 0.0, 0.0, 1.0))
    return c


def test_flat_decal_has_quad_uvs_and_texture() -> None:
    tex = _canvas()
    decal = DecalGeometry(tex, width=0.2, height=0.3)

    assert len(decal.vertices) == 4
    assert len(decal.faces) == 2
    assert decal.uvs is not None and len(decal.uvs) == 4
    # UVs span the full 0..1 image rectangle.
    us = sorted({round(u, 3) for u, _ in decal.uvs})
    vs = sorted({round(v, 3) for _, v in decal.uvs})
    assert us == [0.0, 1.0]
    assert vs == [0.0, 1.0]
    assert decal.texture is tex


def test_flat_decal_dimensions_and_proud_offset() -> None:
    decal = DecalGeometry(_canvas(), width=0.2, height=0.3, proud_offset=0.001)
    xs = [v[0] for v in decal.vertices]
    ys = [v[1] for v in decal.vertices]
    zs = [v[2] for v in decal.vertices]

    assert max(xs) - min(xs) == pytest.approx(0.2)
    assert max(ys) - min(ys) == pytest.approx(0.3)
    assert all(z == pytest.approx(0.001) for z in zs)  # pushed proud along +Z


def test_curved_decal_wraps_onto_cylinder() -> None:
    radius = 0.05
    width = 0.1
    decal = DecalGeometry(_canvas(), width=width, height=0.2, curvature_radius=radius, segments=12)

    assert len(decal.vertices) == (12 + 1) * 2
    assert decal.uvs is not None and len(decal.uvs) == len(decal.vertices)
    # Wrap is about the Z axis (matching the SDK Cylinder): every vertex sits on the
    # cylinder of radius ~ curvature_radius (+proud) in the XY plane.
    for x, y, _z in decal.vertices:
        assert math.hypot(x, y) == pytest.approx(radius, abs=0.001)
    # Height runs along Z.
    zs = [z for _x, _y, z in decal.vertices]
    assert max(zs) - min(zs) == pytest.approx(0.2)
    # Front face points radially outward (-Y at the center).
    assert min(y for _x, y, _z in decal.vertices) == pytest.approx(-(radius), abs=0.001)
    # Arc length across the width matches the requested width.
    assert decal.uvs[0] == (0.0, 1.0)
    assert decal.uvs[-1] == (1.0, 0.0)


def test_copy_preserves_uvs_and_texture() -> None:
    tex = _canvas()
    decal = DecalGeometry(tex, width=0.2, height=0.2)
    clone = decal.copy()

    assert clone.uvs == decal.uvs
    assert clone.texture is tex
    clone.uvs[0] = (0.5, 0.5)
    assert decal.uvs[0] != (0.5, 0.5)  # deep-copied


def test_merge_with_untextured_mesh_clears_uvs() -> None:
    from sdk import BoxGeometry

    decal = DecalGeometry(_canvas(), width=0.2, height=0.2)
    decal.merge(BoxGeometry((0.1, 0.1, 0.1)))

    assert decal.uvs is None
    assert decal.texture is None


def test_validation() -> None:
    with pytest.raises(ValueError, match="width/height"):
        DecalGeometry(_canvas(), width=0.0, height=0.2)
    with pytest.raises(ValueError, match="curvature_radius"):
        DecalGeometry(_canvas(), width=0.2, height=0.2, curvature_radius=-1.0)
    with pytest.raises(ValueError, match="finite"):
        DecalGeometry(_canvas(), width=float("nan"), height=0.2)
    with pytest.raises(ValueError, match="proud_offset"):
        DecalGeometry(_canvas(), width=0.2, height=0.2, proud_offset=float("inf"))
