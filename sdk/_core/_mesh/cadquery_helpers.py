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

import functools
import hashlib
import os
import sys
import tempfile
from contextlib import suppress
from math import cos, pi, sin
from pathlib import Path
from typing import Any, Callable, List, Sequence, Union, cast

import numpy as np

from .common import (
    Face,
    Vec2,
    Vec3,
    _set_manifold_provenance,
    rounded_rect_profile,
    superellipse_profile,
)
from .primitives import MeshGeometry

# Deterministic parametric CadQuery builds (OCC booleans + tessellation) cost seconds per shape
# and are re-paid by every check/compile process of one authoring loop. The memo below caches the
# finished mesh by class, geometry-code mtime and constructor arguments: a process dict first,
# then a small on-disk .npz so repeated tool invocations reuse it. Only pure-parametric classes
# (no file inputs, no randomness) may opt in via ``_memoize_parametric_mesh``. The disk layer
# always lives under ``~/.cache/usdcraft/parametric-mesh``.
_PARAMETRIC_MESH_MEMO: dict[str, tuple[np.ndarray, np.ndarray, float | None]] = {}
_CODE_STAMP_CACHE: dict[str, int] = {}


def _parametric_mesh_cache_dir() -> Path:
    return Path.home() / ".cache" / "usdcraft" / "parametric-mesh"


def _code_stamp(cls: type) -> int:
    module_name = cls.__module__
    cached = _CODE_STAMP_CACHE.get(module_name)
    if cached is None:
        module_file = getattr(sys.modules.get(module_name), "__file__", None)
        cached = Path(module_file).stat().st_mtime_ns if module_file else 0
        _CODE_STAMP_CACHE[module_name] = cached
    return cached


def _parametric_memo_load(digest: str) -> tuple[np.ndarray, np.ndarray, float | None] | None:
    cached = _PARAMETRIC_MESH_MEMO.get(digest)
    if cached is not None:
        return cached
    path = _parametric_mesh_cache_dir() / f"{digest}.npz"
    try:
        with np.load(path) as payload:
            vertices = np.asarray(payload["vertices"], dtype=np.float64)
            faces = np.asarray(payload["faces"], dtype=np.int64)
            crease_raw = float(payload["normal_crease_deg"])
        entry = (vertices, faces, None if np.isnan(crease_raw) else crease_raw)
    except Exception:
        return None
    _PARAMETRIC_MESH_MEMO[digest] = entry
    return entry


def _parametric_memo_store(digest: str, geometry: "MeshGeometry") -> None:
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    faces = np.asarray(geometry.faces, dtype=np.int64)
    crease = geometry.normal_crease_deg
    _PARAMETRIC_MESH_MEMO[digest] = (vertices, faces, crease)
    cache_dir = _parametric_mesh_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz.tmp", delete=False) as handle:
            np.savez(
                handle,
                vertices=vertices,
                faces=faces,
                normal_crease_deg=np.float64("nan" if crease is None else crease),
            )
            staged = Path(handle.name)
        os.replace(staged, cache_dir / f"{digest}.npz")
    except Exception:
        return


def _memoize_parametric_mesh(init: Callable) -> Callable:
    """Cache one pure-parametric CadQuery ``__init__`` by its constructor arguments."""

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        try:
            key_text = "|".join(
                (
                    type(self).__qualname__,
                    str(_code_stamp(type(self))),
                    repr(args),
                    repr(sorted(kwargs.items())),
                )
            )
        except Exception:
            init(self, *args, **kwargs)
            return
        digest = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
        cached = _parametric_memo_load(digest)
        if cached is not None:
            vertices, faces, crease = cached
            MeshGeometry.__init__(self)
            self.vertices = [tuple(vertex) for vertex in vertices.tolist()]
            self.faces = [tuple(face) for face in faces.tolist()]
            self.normal_crease_deg = crease
            try:
                from .booleans import _manifold_from_geometry

                _set_manifold_provenance(
                    self,
                    _manifold_from_geometry(self, name="cadquery_tessellation"),
                )
            except ValueError:
                pass
            return
        init(self, *args, **kwargs)
        if self.faces and self.uvs is None and self.texture is None:
            _parametric_memo_store(digest, self)

    return wrapper


def _cq_polyline_wire(cq_module, points: Sequence[Vec2], plane: str = "XY"):
    return cq_module.Workplane(plane).polyline(points).close()


def _sample_ellipse_profile(width: float, height: float, *, segments: int = 64) -> list[Vec2]:
    rx = float(width) * 0.5
    ry = float(height) * 0.5
    return [
        (rx * cos(2.0 * pi * index / float(segments)), ry * sin(2.0 * pi * index / float(segments)))
        for index in range(segments)
    ]


def _sample_rect_profile(width: float, height: float) -> list[Vec2]:
    hw = float(width) * 0.5
    hh = float(height) * 0.5
    return [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]


def _shape_profile_points_2d(
    shape: str,
    size: Sequence[float],
    *,
    corner_radius: float = 0.0,
    segments: int = 64,
) -> list[Vec2]:
    width = float(size[0])
    height = float(size[1])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("shape size values must be positive")
    if shape == "rect":
        return _sample_rect_profile(width, height)
    if shape == "rounded_rect":
        return rounded_rect_profile(
            width, height, max(0.0, corner_radius), corner_segments=max(4, segments // 16)
        )
    if shape == "circle":
        diameter = min(width, height)
        return _sample_ellipse_profile(diameter, diameter, segments=segments)
    if shape == "ellipse":
        return _sample_ellipse_profile(width, height, segments=segments)
    if shape == "superellipse":
        return superellipse_profile(width, height, exponent=2.8, segments=segments)
    raise ValueError(f"Unsupported shape {shape!r}")


def _shape_size_from_wall(
    inner_size: Sequence[float],
    wall: Union[float, Sequence[float]],
) -> tuple[float, float]:
    inner_w = float(inner_size[0])
    inner_h = float(inner_size[1])
    if isinstance(wall, (int, float)):
        wall_left = wall_right = wall_bottom = wall_top = float(wall)
    else:
        if len(wall) != 4:
            raise ValueError("wall must be a float or a 4-sequence")
        wall_left, wall_right, wall_bottom, wall_top = (float(value) for value in wall)
    if min(wall_left, wall_right, wall_bottom, wall_top) < 0.0:
        raise ValueError("wall values must be non-negative")
    return (
        inner_w + wall_left + wall_right,
        inner_h + wall_bottom + wall_top,
    )


def _cq_ring_solid(
    cq_module,
    outer_points: Sequence[Vec2],
    inner_points: Sequence[Vec2],
    depth: float,
    *,
    center: bool = True,
):
    shape = _cq_polyline_wire(cq_module, outer_points).extrude(float(depth) * 0.5, both=center)
    inner_cut = _cq_polyline_wire(cq_module, inner_points).extrude(
        float(depth) + max(0.002, float(depth) * 0.5),
        both=center,
    )
    return shape.cut(inner_cut)


def _cq_annulus_x(cq_module, outer_radius: float, inner_radius: float, width: float):
    return (
        cq_module.Workplane("YZ")
        .circle(float(outer_radius))
        .circle(max(float(inner_radius), 1.0e-4))
        .extrude(float(width) * 0.5, both=True)
    )


def _centered_pattern_positions(count: int, spacing: float) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    origin = -0.5 * float(spacing) * float(count - 1)
    return [origin + float(index) * float(spacing) for index in range(count)]


def _rounded_slot_profile(length: float, width: float, *, segments: int = 10) -> list[Vec2]:
    radius = width * 0.5
    straight = max(length - width, 0.0)
    half = straight * 0.5
    points: list[Vec2] = []
    for index in range(segments + 1):
        theta = -pi * 0.5 + pi * (index / float(segments))
        points.append((half + radius * cos(theta), radius * sin(theta)))
    for index in range(segments + 1):
        theta = pi * 0.5 + pi * (index / float(segments))
        points.append((-half + radius * cos(theta), radius * sin(theta)))
    return points


def _loft_between_radii_z(
    cq_module,
    radii_and_offsets: Sequence[tuple[float, float]],
):
    wp = None
    previous_offset = 0.0
    for radius, offset in radii_and_offsets:
        if wp is None:
            wp = cq_module.Workplane("XY").workplane(offset=offset).circle(radius)
        else:
            wp = wp.workplane(offset=offset - previous_offset).circle(radius)
        previous_offset = offset
    if wp is None:
        raise ValueError("No loft sections were provided")
    return wp.loft(combine=True, ruled=False)


def _cut_with_pattern(
    shape,
    cutters: Sequence[object],
):
    for index, cutter in enumerate(cutters, start=1):
        # Inspect the Boolean result before OCC's optional same-domain cleanup:
        # an invalid filleted/scalloped cut can otherwise allocate tens of GB.
        result = shape.cut(cutter, clean=False)
        if not result.val().isValid():
            raise ValueError(
                f"CadQuery pattern cut {index} produced an invalid solid before cleanup. "
                "Adjust the base profile, fillet radius or cutter depth/spacing."
            )
        shape = result.clean()
    return shape


def _mesh_geometry_from_cadquery_model(
    model: Any,
    *,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
) -> "MeshGeometry":
    import trimesh

    shape = model.val() if hasattr(model, "val") else model
    if hasattr(shape, "fix"):
        with suppress(Exception):
            shape = shape.fix()

    try:
        vertices_raw, triangles_raw = shape.tessellate(
            float(tolerance),
            float(angular_tolerance),
        )
    except TypeError:
        vertices_raw, triangles_raw = shape.tessellate(float(tolerance))

    vertices = [
        (float(vertex.x), float(vertex.y), float(vertex.z))
        if hasattr(vertex, "x")
        else (float(vertex[0]), float(vertex[1]), float(vertex[2]))
        for vertex in vertices_raw
    ]
    faces = [(int(face[0]), int(face[1]), int(face[2])) for face in triangles_raw]
    if not vertices or not faces:
        raise ValueError("CadQuery tessellation produced an empty mesh")

    # CadQuery tessellation can duplicate shared vertices per face. Normalize the
    # triangle soup so downstream manifold checks and OBJ export see one closed mesh.
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("CadQuery tessellation normalization produced an empty mesh")

    geometry = MeshGeometry(
        vertices=cast(
            List[Vec3], [tuple(float(coord) for coord in vertex) for vertex in mesh.vertices]
        ),
        faces=cast(List[Face], [tuple(int(index) for index in face) for face in mesh.faces]),
    )
    try:
        from .booleans import _manifold_from_geometry

        _set_manifold_provenance(
            geometry,
            _manifold_from_geometry(geometry, name="cadquery_tessellation"),
        )
    except ValueError:
        pass
    return geometry


__all__ = [name for name in globals() if not name.startswith("__")]
