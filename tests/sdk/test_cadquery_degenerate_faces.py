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

from sdk._extensions.cadquery.cadquery import _tessellate_cadquery_shape


class _Shape:
    def __init__(self, vertices, triangles):
        self._vertices = vertices
        self._triangles = triangles

    def tessellate(self, tolerance, angular_tolerance=None):
        return self._vertices, self._triangles


def test_tessellation_drops_degenerate_triangles_and_keeps_real_ones() -> None:
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),  # coincides with vertex 1 under a different index
        (2.0, 0.0, 0.0),  # collinear with 0 and 1
    ]
    triangles = [
        (0, 1, 2),  # real
        (0, 1, 1),  # repeated index
        (0, 1, 3),  # distinct indices, coincident coordinates
        (0, 1, 4),  # collinear, zero area
    ]
    kept_vertices, kept = _tessellate_cadquery_shape(_Shape(vertices, triangles))
    assert kept == [(0, 1, 2)]
    assert len(kept_vertices) == 5


def test_tessellation_with_only_degenerate_faces_is_an_error() -> None:
    with pytest.raises(ValueError, match="only degenerate faces"):
        _tessellate_cadquery_shape(_Shape([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [(0, 1, 1)]))
