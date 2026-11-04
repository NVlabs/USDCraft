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

from dataclasses import dataclass, field
from math import cos, isfinite, pi, sin
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from ..errors import ValidationError
from ..types import Box, Cylinder, Sphere, _coerce_normal_crease_deg
from .common import (
    _EPS,
    _OBJ_COORD_DECIMALS,
    Face,
    LatheCapMode,
    Mat4,
    Vec2,
    Vec3,
    _clear_manifold_provenance,
    _clear_primitive_provenance,
    _copy_manifold_provenance,
    _copy_primitive_provenance,
    _ensure_ccw,
    _lathe_shell_profile,
    _point_in_polygon,
    _points_match_2d,
    _polygon_area,
    _polygon_centroid,
    _polyline_points_2d,
    _prepend_primitive_transform,
    _profile_points_2d,
    _profile_points_3d,
    _set_primitive_provenance,
    _triangulate_polygon,
    _triangulate_polygon_with_holes,
    _v_normalize,
    rounded_rect_profile,
)

if TYPE_CHECKING:
    from .health import MeshHealthReport


@dataclass
class MeshGeometry:
    vertices: List[Vec3] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)
    # Optional per-vertex texture coordinates (aligned to ``vertices``) and the
    # TextureCanvas they sample. Only textured decals set these; plain geometry
    # leaves them None. Transforms preserve UVs; merge with a non-textured mesh
    # clears them (alignment cannot be guaranteed across a mixed merge).
    uvs: Optional[List[Vec2]] = None
    texture: Optional[object] = field(default=None, compare=False, repr=False)
    # Visual-only face-normal smoothing policy. ``None`` preserves the SDK's
    # established 60-degree exporter default; 0 is fully hard and 180 fully
    # smooth across connected face corners.
    normal_crease_deg: Optional[float] = None

    def __post_init__(self) -> None:
        self.vertices = _validated_vertices(self.vertices)
        self.faces = _validated_faces(self.faces, vertex_count=len(self.vertices))
        self.uvs = _validated_uvs(self.uvs, vertex_count=len(self.vertices))
        self.normal_crease_deg = _coerce_normal_crease_deg(
            self.normal_crease_deg,
            name="mesh_geometry.normal_crease_deg",
        )

    def validate(self) -> "MeshGeometry":
        """Validate the current mutable arrays without changing their topology.

        Construction validates immediately, while this explicit boundary also
        catches invalid values introduced by direct list mutation before mesh
        analysis, managed materialization, or export.
        """

        vertices = _validated_vertices(self.vertices)
        faces = _validated_faces(self.faces, vertex_count=len(vertices))
        uvs = _validated_uvs(self.uvs, vertex_count=len(vertices))
        normal_crease_deg = _coerce_normal_crease_deg(
            self.normal_crease_deg,
            name="mesh_geometry.normal_crease_deg",
        )
        # Direct list mutation is supported as a low-level escape hatch. Apply
        # validated numeric normalization atomically so accepted string/numpy
        # scalar inputs cannot fail later in OBJ formatting or native geometry.
        self.vertices = vertices
        self.faces = faces
        self.uvs = uvs
        self.normal_crease_deg = normal_crease_deg
        return self

    def health(self, *, tolerance: float = 1e-9) -> "MeshHealthReport":
        """Return a structured, non-mutating mesh-health diagnostic report."""

        from .health import mesh_health

        return mesh_health(self, tolerance=tolerance)

    def set_normal_crease_deg(self, degrees: float | None) -> "MeshGeometry":
        """Set the visual normal crease in degrees; ``None`` restores the default."""

        self.normal_crease_deg = _coerce_normal_crease_deg(
            degrees,
            name="mesh_geometry.normal_crease_deg",
        )
        return self

    def use_hard_normals(self) -> "MeshGeometry":
        return self.set_normal_crease_deg(0.0)

    def use_smooth_normals(self) -> "MeshGeometry":
        return self.set_normal_crease_deg(180.0)

    def copy(self) -> "MeshGeometry":
        copied = MeshGeometry(
            vertices=cast(List[Vec3], [tuple(vertex) for vertex in self.vertices]),
            faces=cast(List[Face], [tuple(face) for face in self.faces]),
            uvs=cast(List[Vec2], [tuple(uv) for uv in self.uvs]) if self.uvs is not None else None,
            texture=self.texture,
            normal_crease_deg=self.normal_crease_deg,
        )
        _copy_primitive_provenance(self, copied)
        _copy_manifold_provenance(self, copied)
        return copied

    def clone(self) -> "MeshGeometry":
        return self.copy()

    def add_vertex(self, x: float, y: float, z: float) -> int:
        if self.uvs is not None:
            raise ValidationError(
                "Cannot add an untextured vertex while mesh_geometry.uvs is present"
            )
        vertex = _validated_vertex((x, y, z), name="vertex")
        _clear_manifold_provenance(self)
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    def add_face(self, a: int, b: int, c: int) -> None:
        face = _validated_face((a, b, c), vertex_count=len(self.vertices), name="face")
        _clear_manifold_provenance(self)
        self.faces.append(face)

    def merge(self, other: "MeshGeometry") -> "MeshGeometry":
        if not isinstance(other, MeshGeometry):
            raise TypeError("other must be MeshGeometry")
        self.validate()
        # Validate the source into local normalized values. ``merge`` mutates
        # only its receiver; callers may safely reuse ``other`` and retain
        # references to its public arrays.
        other_vertices = _validated_vertices(other.vertices)
        other_faces = _validated_faces(other.faces, vertex_count=len(other_vertices))
        other_uvs = _validated_uvs(other.uvs, vertex_count=len(other_vertices))
        other_normal_crease_deg = _coerce_normal_crease_deg(
            other.normal_crease_deg,
            name="mesh_geometry.normal_crease_deg",
        )
        if not self.vertices and not self.faces:
            self.vertices = cast(List[Vec3], [tuple(vertex) for vertex in other_vertices])
            self.faces = cast(List[Face], [tuple(face) for face in other_faces])
            self.uvs = (
                cast(List[Vec2], [tuple(uv) for uv in other_uvs]) if other_uvs is not None else None
            )
            self.texture = other.texture
            self.normal_crease_deg = other_normal_crease_deg
            _copy_primitive_provenance(other, self)
            _copy_manifold_provenance(other, self)
            return self
        from .normals import DEFAULT_NORMAL_CREASE_DEGREES

        self_crease = (
            DEFAULT_NORMAL_CREASE_DEGREES
            if self.normal_crease_deg is None
            else self.normal_crease_deg
        )
        other_crease = (
            DEFAULT_NORMAL_CREASE_DEGREES
            if other_normal_crease_deg is None
            else other_normal_crease_deg
        )
        if self_crease != other_crease:
            # One materialized OBJ/managed Mesh can carry only one shading
            # policy. A heterogeneous merge therefore falls back explicitly
            # to the SDK default instead of silently applying either input's
            # policy to every face.
            self.normal_crease_deg = None
        elif self_crease == DEFAULT_NORMAL_CREASE_DEGREES:
            # ``None`` and explicit 60 degrees are the same public policy.
            # Canonicalize their merged result so input order cannot change
            # serialized metadata or equality-visible state.
            self.normal_crease_deg = None
        offset = len(self.vertices)
        if self.uvs is not None and other_uvs is not None:
            self.uvs = list(self.uvs) + cast(List[Vec2], [tuple(uv) for uv in other_uvs])
        else:
            # Mixed/absent UVs cannot stay aligned across a merge; drop them.
            self.uvs = None
            self.texture = None
        self.vertices.extend(other_vertices)
        self.faces.extend((a + offset, b + offset, c + offset) for (a, b, c) in other_faces)
        _clear_primitive_provenance(self)
        _clear_manifold_provenance(self)
        return self

    def translate(self, dx: float, dy: float, dz: float) -> "MeshGeometry":
        self.vertices = [(x + dx, y + dy, z + dz) for (x, y, z) in self.vertices]
        transform: Mat4 = (
            (1.0, 0.0, 0.0, float(dx)),
            (0.0, 1.0, 0.0, float(dy)),
            (0.0, 0.0, 1.0, float(dz)),
            (0.0, 0.0, 0.0, 1.0),
        )
        _prepend_primitive_transform(self, transform)
        _clear_manifold_provenance(self)
        return self

    def scale(
        self, sx: float, sy: Optional[float] = None, sz: Optional[float] = None
    ) -> "MeshGeometry":
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        self.vertices = [(x * sx, y * sy, z * sz) for (x, y, z) in self.vertices]
        _prepend_primitive_transform(
            self,
            (
                (float(sx), 0.0, 0.0, 0.0),
                (0.0, float(sy), 0.0, 0.0),
                (0.0, 0.0, float(sz), 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        _clear_manifold_provenance(self)
        return self

    def rotate(
        self,
        axis: Sequence[float],
        angle: float,
        *,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "MeshGeometry":
        kx, ky, kz = _v_normalize((float(axis[0]), float(axis[1]), float(axis[2])))
        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
        c = cos(angle)
        s = sin(angle)
        t = 1.0 - c

        r00 = t * kx * kx + c
        r01 = t * kx * ky - s * kz
        r02 = t * kx * kz + s * ky
        r10 = t * ky * kx + s * kz
        r11 = t * ky * ky + c
        r12 = t * ky * kz - s * kx
        r20 = t * kz * kx - s * ky
        r21 = t * kz * ky + s * kx
        r22 = t * kz * kz + c

        tx = ox - (r00 * ox + r01 * oy + r02 * oz)
        ty = oy - (r10 * ox + r11 * oy + r12 * oz)
        tz = oz - (r20 * ox + r21 * oy + r22 * oz)
        transform: Mat4 = (
            (float(r00), float(r01), float(r02), float(tx)),
            (float(r10), float(r11), float(r12), float(ty)),
            (float(r20), float(r21), float(r22), float(tz)),
            (0.0, 0.0, 0.0, 1.0),
        )

        self.vertices = [
            (
                r00 * x + r01 * y + r02 * z + tx,
                r10 * x + r11 * y + r12 * z + ty,
                r20 * x + r21 * y + r22 * z + tz,
            )
            for (x, y, z) in self.vertices
        ]
        _prepend_primitive_transform(self, transform)
        _clear_manifold_provenance(self)
        return self

    def rotate_x(self, angle: float) -> "MeshGeometry":
        return self.rotate((1.0, 0.0, 0.0), angle)

    def rotate_y(self, angle: float) -> "MeshGeometry":
        return self.rotate((0.0, 1.0, 0.0), angle)

    def rotate_z(self, angle: float) -> "MeshGeometry":
        return self.rotate((0.0, 0.0, 1.0), angle)

    def to_obj(self) -> str:
        self.validate()
        coordinate_decimals = _obj_coordinate_decimals(self.vertices, self.faces)
        from .normals import DEFAULT_NORMAL_CREASE_DEGREES

        effective_crease = (
            DEFAULT_NORMAL_CREASE_DEGREES
            if self.normal_crease_deg is None
            else self.normal_crease_deg
        )
        lines = [
            "o mesh",
            f"# usdcraft_normal_crease_deg {effective_crease:.12g}",
        ]
        for x, y, z in self.vertices:
            lines.append(
                "v "
                + " ".join(
                    _format_obj_component(value, decimals=coordinate_decimals)
                    for value in (x, y, z)
                )
            )
        if self.uvs is not None:
            for u, v in self.uvs:
                lines.append(
                    "vt "
                    + " ".join(
                        _format_obj_component(value, decimals=coordinate_decimals)
                        for value in (u, v)
                    )
                )
        for a, b, c in self.faces:
            if self.uvs is None:
                lines.append(f"f {a + 1} {b + 1} {c + 1}")
            else:
                lines.append(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}")
        lines.append("")
        return "\n".join(lines)

    def save_obj(self, path: Union[str, Path]) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_obj())


def _validated_vertex(value: Sequence[float], *, name: str) -> Vec3:
    coordinates = _validated_sequence(value, length=3, name=name, item_label="coordinates")
    try:
        vertex = (float(coordinates[0]), float(coordinates[1]), float(coordinates[2]))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} coordinates must be numeric") from exc
    if not all(isfinite(component) for component in vertex):
        raise ValidationError(f"{name} coordinates must be finite")
    return vertex


def _validated_vertices(values: Iterable[Sequence[float]]) -> List[Vec3]:
    if isinstance(values, (str, bytes)):
        raise ValidationError("mesh_geometry.vertices must be an iterable of 3D coordinates")
    try:
        vertex_values = list(values)
    except TypeError as exc:
        raise ValidationError(
            "mesh_geometry.vertices must be an iterable of 3D coordinates"
        ) from exc
    # Bulk fast path: authored meshes routinely carry hundreds of thousands of vertices, and the
    # per-vertex loop below costs seconds per authoring check on such inputs. Anything that does
    # not form a clean finite (N, 3) float array falls back to the exact per-item diagnostics.
    if vertex_values:
        try:
            array = np.asarray(vertex_values, dtype=np.float64)
        except (TypeError, ValueError):
            array = None
        if (
            array is not None
            and array.ndim == 2
            and array.shape[1] == 3
            and np.isfinite(array).all()
        ):
            return [(row[0], row[1], row[2]) for row in array.tolist()]
    return [
        _validated_vertex(value, name=f"mesh_geometry.vertices[{index}]")
        for index, value in enumerate(vertex_values)
    ]


def _validated_face(
    value: Sequence[int],
    *,
    vertex_count: int,
    name: str,
) -> Face:
    face_values = _validated_sequence(
        value,
        length=3,
        name=name,
        item_label="vertex indices",
    )
    indices: list[int] = []
    for corner, raw_index in enumerate(face_values):
        if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
            raise ValidationError(f"{name}[{corner}] must be an integer vertex index")
        index = int(raw_index)
        if index < 0 or index >= vertex_count:
            raise ValidationError(f"{name}[{corner}] index {index} is outside [0, {vertex_count})")
        indices.append(index)
    if len(set(indices)) != 3:
        raise ValidationError(f"{name} must reference 3 distinct vertices")
    return (indices[0], indices[1], indices[2])


def _validated_faces(values: Iterable[Sequence[int]], *, vertex_count: int) -> List[Face]:
    if isinstance(values, (str, bytes)):
        raise ValidationError("mesh_geometry.faces must be an iterable of triangle indices")
    try:
        face_values = list(values)
    except TypeError as exc:
        raise ValidationError(
            "mesh_geometry.faces must be an iterable of triangle indices"
        ) from exc
    # Bulk fast path mirroring _validated_vertices: only clean integer (N, 3) arrays with in-range
    # distinct indices qualify; every other input keeps the precise per-face error reporting.
    if face_values:
        try:
            array = np.asarray(face_values)
        except (TypeError, ValueError):
            array = None
        if (
            array is not None
            and array.ndim == 2
            and array.shape[1] == 3
            and array.dtype.kind in "iu"
            and array.size
            and int(array.min()) >= 0
            and int(array.max()) < vertex_count
            and not np.any(
                (array[:, 0] == array[:, 1])
                | (array[:, 1] == array[:, 2])
                | (array[:, 0] == array[:, 2])
            )
        ):
            return [(row[0], row[1], row[2]) for row in array.tolist()]
    return [
        _validated_face(
            value,
            vertex_count=vertex_count,
            name=f"mesh_geometry.faces[{index}]",
        )
        for index, value in enumerate(face_values)
    ]


def _validated_uvs(
    values: Optional[Iterable[Sequence[float]]],
    *,
    vertex_count: int,
) -> Optional[List[Vec2]]:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValidationError("mesh_geometry.uvs must be an iterable of 2D coordinates")
    try:
        uv_values = list(values)
    except TypeError as exc:
        raise ValidationError("mesh_geometry.uvs must be an iterable of 2D coordinates") from exc
    normalized: List[Vec2] = []
    for index, value in enumerate(uv_values):
        uv_coordinates = _validated_sequence(
            value,
            length=2,
            name=f"mesh_geometry.uvs[{index}]",
            item_label="coordinates",
        )
        try:
            uv = (float(uv_coordinates[0]), float(uv_coordinates[1]))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"mesh_geometry.uvs[{index}] coordinates must be numeric"
            ) from exc
        if not all(isfinite(component) for component in uv):
            raise ValidationError(f"mesh_geometry.uvs[{index}] coordinates must be finite")
        normalized.append(uv)
    if len(normalized) != vertex_count:
        raise ValidationError(
            "mesh_geometry.uvs must contain exactly one coordinate per vertex "
            f"({len(normalized)} != {vertex_count})"
        )
    return normalized


def _validated_sequence(
    value: Any,
    *,
    length: int,
    name: str,
    item_label: str,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise ValidationError(f"{name} must contain exactly {length} {item_label}")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValidationError(f"{name} must contain exactly {length} {item_label}") from exc
    if len(items) != length:
        raise ValidationError(f"{name} must contain exactly {length} {item_label}")
    return items


def _format_obj_component(value: float, *, decimals: int | None) -> str:
    if decimals is None:
        # 17 significant digits round-trip every finite IEEE-754 double.
        return format(value, ".17g")
    return f"{value:.{decimals}f}"


def _obj_faces_survive_precision(
    vertices: Sequence[Vec3],
    faces: Sequence[Face],
    *,
    decimals: int | None,
) -> tuple[bool, int | None]:
    quantized = [
        tuple(float(_format_obj_component(component, decimals=decimals)) for component in vertex)
        for vertex in vertices
    ]
    for face_index, (a, b, c) in enumerate(faces):
        first = quantized[a]
        second = quantized[b]
        third = quantized[c]
        ab = (
            second[0] - first[0],
            second[1] - first[1],
            second[2] - first[2],
        )
        ac = (
            third[0] - first[0],
            third[1] - first[1],
            third[2] - first[2],
        )
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        if cross == (0.0, 0.0, 0.0):
            return False, face_index
    return True, None


def _obj_coordinate_decimals(vertices: Sequence[Vec3], faces: Sequence[Face]) -> int | None:
    """Choose the smallest deterministic OBJ precision that preserves every face."""

    for decimals in range(_OBJ_COORD_DECIMALS, 16):
        survives, _face_index = _obj_faces_survive_precision(
            vertices,
            faces,
            decimals=decimals,
        )
        if survives:
            return decimals
    survives, face_index = _obj_faces_survive_precision(vertices, faces, decimals=None)
    if survives:
        return None
    raise ValidationError(
        f"mesh_geometry.faces[{face_index}] is coordinate-degenerate before OBJ serialization"
    )


def _coerce_positive_radii3(
    value: Union[float, Sequence[float]],
    *,
    name: str,
) -> Vec3:
    if isinstance(value, (int, float)):
        radius = float(value)
        if not isfinite(radius) or radius <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return (radius, radius, radius)
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive float or a 3-sequence") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must be a positive float or a 3-sequence")
    rx = float(values[0])
    ry = float(values[1])
    rz = float(values[2])
    if not all(isfinite(component) and component > 0.0 for component in (rx, ry, rz)):
        raise ValueError(f"{name} values must be finite and positive")
    return (rx, ry, rz)


def _coerce_positive_size3(value: Sequence[float], *, name: str) -> Vec3:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain three finite positive values") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain three finite positive values")
    result = (float(values[0]), float(values[1]), float(values[2]))
    if not all(isfinite(component) and component > 0.0 for component in result):
        raise ValueError(f"{name} values must be finite and positive")
    return result


def _coerce_positive_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class BoxGeometry(MeshGeometry):
    """Axis-aligned box mesh of full extents ``size``, centered at the origin."""

    def __init__(self, size: Sequence[float]):
        super().__init__()
        x, y, z = _coerce_positive_size3(size, name="size")
        primitive = Box((x, y, z))
        _set_primitive_provenance(self, primitive)
        hx, hy, hz = x / 2.0, y / 2.0, z / 2.0
        verts = [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]
        self.vertices = verts
        faces = [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 5, 4),
            (0, 1, 5),
            (1, 6, 5),
            (1, 2, 6),
            (2, 7, 6),
            (2, 3, 7),
            (3, 4, 7),
            (3, 0, 4),
        ]
        self.faces = faces
        self.validate()


class CylinderGeometry(MeshGeometry):
    """Cylinder mesh along local Z, centered at the origin; ``closed`` adds the end caps."""

    def __init__(
        self,
        radius: float,
        height: float,
        *,
        radial_segments: int = 24,
        closed: bool = True,
    ):
        super().__init__()
        radius = _coerce_positive_scalar(radius, name="radius")
        height = _coerce_positive_scalar(height, name="height")
        radial_segments = max(3, int(radial_segments))
        primitive = Cylinder(radius=radius, length=height)
        _set_primitive_provenance(self, primitive)
        half = height / 2.0

        top_indices = []
        bottom_indices = []
        for i in range(radial_segments):
            theta = 2 * pi * i / radial_segments
            x = radius * cos(theta)
            y = radius * sin(theta)
            top_indices.append(self.add_vertex(x, y, half))
            bottom_indices.append(self.add_vertex(x, y, -half))

        for i in range(radial_segments):
            i2 = (i + 1) % radial_segments
            a = top_indices[i]
            b = bottom_indices[i]
            c = bottom_indices[i2]
            d = top_indices[i2]
            self.add_face(a, b, c)
            self.add_face(a, c, d)

        if closed:
            top_center = self.add_vertex(0.0, 0.0, half)
            bottom_center = self.add_vertex(0.0, 0.0, -half)
            for i in range(radial_segments):
                i2 = (i + 1) % radial_segments
                self.add_face(top_center, top_indices[i], top_indices[i2])
                self.add_face(bottom_center, bottom_indices[i2], bottom_indices[i])


class ConeGeometry(MeshGeometry):
    """Cone mesh along local Z, centered: base ring at ``z=-height/2``, apex at ``z=+height/2``."""

    def __init__(
        self,
        radius: float,
        height: float,
        *,
        radial_segments: int = 24,
        closed: bool = True,
    ):
        super().__init__()
        radius = _coerce_positive_scalar(radius, name="radius")
        height = _coerce_positive_scalar(height, name="height")
        radial_segments = max(3, int(radial_segments))
        half = height / 2.0
        apex = self.add_vertex(0.0, 0.0, half)
        ring_indices = []
        for i in range(radial_segments):
            theta = 2 * pi * i / radial_segments
            x = radius * cos(theta)
            y = radius * sin(theta)
            ring_indices.append(self.add_vertex(x, y, -half))

        for i in range(radial_segments):
            i2 = (i + 1) % radial_segments
            self.add_face(apex, ring_indices[i], ring_indices[i2])

        if closed:
            base_center = self.add_vertex(0.0, 0.0, -half)
            for i in range(radial_segments):
                i2 = (i + 1) % radial_segments
                self.add_face(base_center, ring_indices[i2], ring_indices[i])


class SphereGeometry(MeshGeometry):
    """UV-sphere mesh of ``radius`` centered at the origin, poles on local Z."""

    def __init__(
        self,
        radius: float,
        *,
        width_segments: int = 24,
        height_segments: int = 16,
    ):
        super().__init__()
        radius = _coerce_positive_scalar(radius, name="radius")
        width_segments = max(3, int(width_segments))
        height_segments = max(2, int(height_segments))
        primitive = Sphere(radius=radius)
        _set_primitive_provenance(self, primitive)

        north = self.add_vertex(0.0, 0.0, radius)
        rings: List[List[int]] = []
        for iy in range(1, height_segments):
            v = iy / height_segments
            phi = pi * v
            ring: List[int] = []
            for ix in range(width_segments):
                u = ix / width_segments
                theta = 2 * pi * u
                x = radius * sin(phi) * cos(theta)
                y = radius * sin(phi) * sin(theta)
                z = radius * cos(phi)
                ring.append(self.add_vertex(x, y, z))
            rings.append(ring)
        south = self.add_vertex(0.0, 0.0, -radius)

        first_ring = rings[0]
        for ix in range(width_segments):
            ix2 = (ix + 1) % width_segments
            self.add_face(north, first_ring[ix], first_ring[ix2])

        for upper, lower in zip(rings, rings[1:]):
            for ix in range(width_segments):
                ix2 = (ix + 1) % width_segments
                self.add_face(upper[ix], lower[ix], upper[ix2])
                self.add_face(lower[ix], lower[ix2], upper[ix2])

        last_ring = rings[-1]
        for ix in range(width_segments):
            ix2 = (ix + 1) % width_segments
            self.add_face(last_ring[ix], south, last_ring[ix2])


class DomeGeometry(MeshGeometry):
    """Upper half-ellipsoid mesh with its base on ``z=0``; ``closed`` adds the base disk.

    ``radius`` is a scalar or per-axis ``(rx, ry, rz)``.
    """

    def __init__(
        self,
        radius: Union[float, Sequence[float]],
        *,
        radial_segments: int = 24,
        height_segments: int = 12,
        closed: bool = True,
    ):
        super().__init__()
        rx, ry, rz = _coerce_positive_radii3(radius, name="radius")
        radial_segments = max(3, int(radial_segments))
        height_segments = max(1, int(height_segments))

        north = self.add_vertex(0.0, 0.0, rz)
        rings: List[List[int]] = []
        for iy in range(1, height_segments + 1):
            v = iy / height_segments
            phi = 0.5 * pi * v
            sin_phi = sin(phi)
            cos_phi = cos(phi)
            z = rz * cos_phi
            ring: List[int] = []
            for ix in range(radial_segments):
                u = ix / radial_segments
                theta = 2.0 * pi * u
                x = rx * sin_phi * cos(theta)
                y = ry * sin_phi * sin(theta)
                ring.append(self.add_vertex(x, y, z))
            rings.append(ring)

        first_ring = rings[0]
        for ix in range(radial_segments):
            ix2 = (ix + 1) % radial_segments
            self.add_face(north, first_ring[ix], first_ring[ix2])

        for upper, lower in zip(rings, rings[1:]):
            for ix in range(radial_segments):
                ix2 = (ix + 1) % radial_segments
                self.add_face(upper[ix], lower[ix], upper[ix2])
                self.add_face(lower[ix], lower[ix2], upper[ix2])

        if closed:
            base_center = self.add_vertex(0.0, 0.0, 0.0)
            equator = rings[-1]
            for ix in range(radial_segments):
                ix2 = (ix + 1) % radial_segments
                self.add_face(base_center, equator[ix2], equator[ix])


class CapsuleGeometry(MeshGeometry):
    """Capsule mesh along local Z, centered at the origin.

    ``length`` is the cylindrical mid-section between the hemispherical caps; ``0`` gives a sphere.
    """

    def __init__(
        self,
        radius: float,
        length: float,
        *,
        radial_segments: int = 24,
        height_segments: int = 8,
    ):
        super().__init__()
        radius = _coerce_positive_scalar(radius, name="radius")
        length = float(length)
        radial_segments = max(3, int(radial_segments))
        height_segments = max(2, int(height_segments))

        if not isfinite(length) or length < 0.0:
            raise ValueError("length must be finite and non-negative")

        if length <= _EPS:
            self.merge(
                SphereGeometry(
                    radius,
                    width_segments=radial_segments,
                    height_segments=max(4, 2 * height_segments),
                )
            )
            return

        half = length * 0.5
        rows: List[Tuple[float, float]] = []

        for iy in range(height_segments + 1):
            v = iy / height_segments
            phi = 0.5 * pi * v
            rows.append((radius * sin(phi), half + radius * cos(phi)))

        for iy in range(height_segments + 1):
            v = iy / height_segments
            phi = 0.5 * pi + 0.5 * pi * v
            rows.append((radius * sin(phi), -half + radius * cos(phi)))

        north = self.add_vertex(0.0, 0.0, rows[0][1])
        rings: List[List[int]] = []
        for ring_radius, z in rows[1:-1]:
            ring: List[int] = []
            for ix in range(radial_segments):
                u = ix / radial_segments
                theta = 2.0 * pi * u
                x = ring_radius * cos(theta)
                y = ring_radius * sin(theta)
                ring.append(self.add_vertex(x, y, z))
            rings.append(ring)
        south = self.add_vertex(0.0, 0.0, rows[-1][1])

        first_ring = rings[0]
        for ix in range(radial_segments):
            ix2 = (ix + 1) % radial_segments
            self.add_face(north, first_ring[ix], first_ring[ix2])

        for upper, lower in zip(rings, rings[1:]):
            for ix in range(radial_segments):
                ix2 = (ix + 1) % radial_segments
                self.add_face(upper[ix], lower[ix], upper[ix2])
                self.add_face(lower[ix], lower[ix2], upper[ix2])

        last_ring = rings[-1]
        for ix in range(radial_segments):
            ix2 = (ix + 1) % radial_segments
            self.add_face(last_ring[ix], south, last_ring[ix2])


class TorusGeometry(MeshGeometry):
    """Torus mesh around local Z, centered at the origin.

    ``radius`` is the ring (center-line) radius and ``tube`` the tube radius; ``0 < tube < radius``.
    """

    def __init__(
        self,
        radius: float,
        tube: float,
        *,
        radial_segments: int = 16,
        tubular_segments: int = 32,
    ):
        super().__init__()
        radius = _coerce_positive_scalar(radius, name="radius")
        tube = _coerce_positive_scalar(tube, name="tube")
        if tube >= radius:
            raise ValueError("tube must be smaller than radius for a non-self-intersecting torus")
        radial_segments = max(3, int(radial_segments))
        tubular_segments = max(3, int(tubular_segments))

        for j in range(radial_segments):
            v = 2 * pi * j / radial_segments
            cv = cos(v)
            sv = sin(v)
            for i in range(tubular_segments):
                u = 2 * pi * i / tubular_segments
                cu = cos(u)
                su = sin(u)
                x = (radius + tube * cv) * cu
                y = (radius + tube * cv) * su
                z = tube * sv
                self.add_vertex(x, y, z)

        for j in range(radial_segments):
            j2 = (j + 1) % radial_segments
            for i in range(tubular_segments):
                i2 = (i + 1) % tubular_segments
                a = j * tubular_segments + i
                b = j2 * tubular_segments + i
                c = j2 * tubular_segments + i2
                d = j * tubular_segments + i2
                self.add_face(a, d, b)
                self.add_face(b, d, c)


class LatheGeometry(MeshGeometry):
    """Surface of revolution about local Z from a ``(radius, z)`` profile.

    ``closed`` treats the profile as a closed loop; otherwise it is an open polyline.
    """

    @classmethod
    def from_shell_profiles(
        cls,
        outer_profile: Iterable[Tuple[float, float]],
        inner_profile: Iterable[Tuple[float, float]],
        *,
        segments: int = 32,
        start_cap: LatheCapMode = "flat",
        end_cap: LatheCapMode = "flat",
        lip_samples: int = 6,
    ) -> "LatheGeometry":
        profile = _lathe_shell_profile(
            outer_profile,
            inner_profile,
            start_cap=start_cap,
            end_cap=end_cap,
            lip_samples=lip_samples,
        )
        return cls(profile, segments=segments, closed=True)

    def __init__(
        self,
        profile: Iterable[Tuple[float, float]],
        *,
        segments: int = 32,
        closed: bool = True,
    ):
        super().__init__()
        segments = max(3, int(segments))
        points = _profile_points_2d(profile) if closed else _polyline_points_2d(profile)
        if closed:
            points = _ensure_ccw(points)

        normalized_points: List[Vec2] = []
        for r, z in points:
            if r < -_EPS:
                raise ValueError("Lathe profile radii must be non-negative")
            normalized_points.append((0.0 if abs(r) <= _EPS else float(r), float(z)))
        points = normalized_points

        ring_indices: List[List[int]] = []
        for r, z in points:
            if r <= _EPS:
                axis_vertex = self.add_vertex(0.0, 0.0, z)
                ring_indices.append([axis_vertex] * segments)
                continue

            ring: List[int] = []
            for i in range(segments):
                theta = 2 * pi * i / segments
                ring.append(self.add_vertex(r * cos(theta), r * sin(theta), z))
            ring_indices.append(ring)

        def add_face_if_non_degenerate(a: int, b: int, c: int) -> None:
            if a == b or b == c or c == a:
                return
            self.add_face(a, b, c)

        edge_count = len(points) if closed else len(points) - 1
        for j in range(edge_count):
            j2 = (j + 1) % len(points)
            r0 = points[j][0]
            r1 = points[j2][0]
            if r0 <= _EPS and r1 <= _EPS:
                continue

            ring0 = ring_indices[j]
            ring1 = ring_indices[j2]
            for i in range(segments):
                i2 = (i + 1) % segments
                a = ring0[i]
                b = ring0[i2]
                c = ring1[i2]
                d = ring1[i]
                add_face_if_non_degenerate(a, b, d)
                add_face_if_non_degenerate(b, c, d)


class LoftGeometry(MeshGeometry):
    """Low-level mesh loft through 3D point loops of equal point count; ``cap`` closes the ends."""

    def __init__(
        self,
        profiles: Iterable[Iterable[Tuple[float, float, float]]],
        *,
        cap: bool = True,
        closed: bool = True,
    ):
        super().__init__()
        profile_list = [_profile_points_3d(profile) for profile in profiles]
        if len(profile_list) < 2:
            raise ValueError("Loft requires at least two profiles")

        ring_count = len(profile_list[0])
        if closed and ring_count < 3:
            raise ValueError("Closed loft profiles must have at least 3 points")

        for profile in profile_list:
            if len(profile) != ring_count:
                raise ValueError("Loft profiles must have the same point count")

        profile_xy = [(x, y) for (x, y, _z) in profile_list[0]]
        base_area = _polygon_area(profile_xy)
        if abs(base_area) <= 1e-9:
            raise ValueError(
                "Loft profile area must be non-zero in XY projection; "
                "profiles should usually be closed XY loops at constant z"
            )

        if base_area < 0:
            profile_list = [list(reversed(profile)) for profile in profile_list]
            base_area = -base_area

        for idx, profile in enumerate(profile_list):
            area = _polygon_area([(x, y) for (x, y, _z) in profile])
            if abs(area) <= 1e-9:
                raise ValueError(
                    "Loft profile area must be non-zero in XY projection; "
                    "profiles should usually be closed XY loops at constant z"
                )
            if area * base_area < 0:
                profile_list[idx] = list(reversed(profile))

        for profile in profile_list:
            for x, y, z in profile:
                self.add_vertex(x, y, z)

        segment_count = ring_count if closed else ring_count - 1
        for i in range(len(profile_list) - 1):
            offset0 = i * ring_count
            offset1 = (i + 1) * ring_count
            for j in range(segment_count):
                j2 = (j + 1) % ring_count
                a = offset0 + j
                b = offset0 + j2
                c = offset1 + j2
                d = offset1 + j
                self.add_face(a, b, c)
                self.add_face(a, c, d)

        if cap and closed:
            first_profile = profile_list[0]
            last_profile = profile_list[-1]
            first_xy = [(x, y) for (x, y, _z) in first_profile]
            last_xy = [(x, y) for (x, y, _z) in last_profile]

            z_spread = max(z for (_x, _y, z) in first_profile) - min(
                z for (_x, _y, z) in first_profile
            )
            if z_spread > 1e-6:
                raise ValueError("Loft caps require planar profiles")

            z_spread = max(z for (_x, _y, z) in last_profile) - min(
                z for (_x, _y, z) in last_profile
            )
            if z_spread > 1e-6:
                raise ValueError("Loft caps require planar profiles")

            triangles = _triangulate_polygon(first_xy)
            for a, b, c in triangles:
                self.add_face(c, b, a)

            offset_last = (len(profile_list) - 1) * ring_count
            triangles = _triangulate_polygon(last_xy)
            for a, b, c in triangles:
                self.add_face(offset_last + a, offset_last + b, offset_last + c)


class ExtrudeGeometry(MeshGeometry):
    """Extrusion of a closed XY profile along local Z by ``height``.

    ``center=True`` spans ``z`` in ``[-height/2, height/2]``; ``center=False`` spans
    ``[0, height]``.
    """

    @classmethod
    def centered(
        cls,
        profile: Iterable[Tuple[float, float]],
        height: float,
        *,
        cap: bool = True,
        closed: bool = True,
    ) -> "ExtrudeGeometry":
        return cls(profile, height, cap=cap, center=True, closed=closed)

    @classmethod
    def from_z0(
        cls,
        profile: Iterable[Tuple[float, float]],
        height: float,
        *,
        cap: bool = True,
        closed: bool = True,
    ) -> "ExtrudeGeometry":
        return cls(profile, height, cap=cap, center=False, closed=closed)

    def __init__(
        self,
        profile: Iterable[Tuple[float, float]],
        height: float,
        *,
        cap: bool = True,
        center: bool = True,
        closed: bool = True,
    ):
        super().__init__()
        height = float(height)
        if height <= 0:
            raise ValueError("Extrude height must be positive")

        points = _ensure_ccw(_profile_points_2d(profile))
        if not closed:
            cap = False

        z0 = -height / 2.0 if center else 0.0
        z1 = z0 + height

        profiles = [
            [(x, y, z0) for (x, y) in points],
            [(x, y, z1) for (x, y) in points],
        ]
        self.merge(LoftGeometry(profiles, cap=cap, closed=closed))


class RoundedBoxGeometry(MeshGeometry):
    """A box of ``size`` with the four edges parallel to ``round_axis`` filleted to
    ``radius``.

    A cheap antidote to blocky, toy-looking output — use it for handles, bars,
    casings, rails, panels, and anything that should read as a real moulded/
    machined part rather than a hard cuboid. Only the edges parallel to
    ``round_axis`` are rounded (the usual case); for an all-edges-rounded solid,
    ``smooth_geometry`` a plain :class:`BoxGeometry` instead.
    """

    def __init__(
        self,
        size: Sequence[float],
        radius: float,
        *,
        round_axis: str = "z",
        corner_segments: int = 6,
        cap: bool = True,
    ):
        super().__init__()
        size_values = _validated_sequence(
            size,
            length=3,
            name="size",
            item_label="dimensions",
        )
        sx, sy, sz = (float(size_values[0]), float(size_values[1]), float(size_values[2]))
        if not all(isfinite(value) and value > 0.0 for value in (sx, sy, sz)):
            raise ValueError("size dimensions must be finite and positive")
        radius = float(radius)
        if not isfinite(radius) or radius < 0.0:
            raise ValueError("radius must be finite and non-negative")
        axis = str(round_axis).lower()
        # (profile width, profile height, extrude length) before reorientation; the
        # profile is built in XY and extruded along Z, then rotated so the extrude
        # axis lands on round_axis with the remaining dimensions preserved.
        if axis == "z":
            pw, ph, length = sx, sy, sz
        elif axis == "x":
            pw, ph, length = sz, sy, sx
        elif axis == "y":
            pw, ph, length = sx, sz, sy
        else:
            raise ValueError("round_axis must be 'x', 'y', or 'z'")
        r = min(radius, 0.5 * min(pw, ph))
        profile = _ensure_ccw(rounded_rect_profile(pw, ph, r, corner_segments=corner_segments))
        z0, z1 = -length / 2.0, length / 2.0
        rings = [
            [(x, y, z0) for (x, y) in profile],
            [(x, y, z1) for (x, y) in profile],
        ]
        self.merge(LoftGeometry(rings, cap=cap, closed=True))
        if axis == "x":
            self.rotate_y(pi / 2.0)
        elif axis == "y":
            self.rotate_x(-pi / 2.0)


class ExtrudeWithHolesGeometry(MeshGeometry):
    """
    Extrude a 2D outer profile with one or more through-holes.

    Notes
    -----
    - This is a lightweight CAD-like helper for panel/filter parts.
    - For capped, closed solids, cap triangulation uses the shared manifold-backed
      polygon triangulation path when available.
    """

    def __init__(
        self,
        outer_profile: Iterable[Tuple[float, float]],
        hole_profiles: Iterable[Iterable[Tuple[float, float]]],
        height: float,
        *,
        cap: bool = True,
        center: bool = True,
        closed: bool = True,
    ):
        super().__init__()
        height = float(height)
        if height <= 0:
            raise ValueError("Extrude height must be positive")

        outer = _ensure_ccw(_profile_points_2d(outer_profile))
        holes = [_ensure_ccw(_profile_points_2d(h)) for h in hole_profiles]
        if not closed:
            cap = False

        for hole in holes:
            centroid = _polygon_centroid(hole)
            if not _point_in_polygon(centroid, outer):
                raise ValueError("Hole profile must lie inside outer profile")

        if not holes:
            self.merge(
                ExtrudeGeometry(
                    outer,
                    height,
                    cap=cap,
                    center=center,
                    closed=closed,
                )
            )
            return

        z0 = -height / 2.0 if center else 0.0
        z1 = z0 + height

        # Exterior side walls.
        outer_loft = LoftGeometry(
            [[(x, y, z0) for (x, y) in outer], [(x, y, z1) for (x, y) in outer]],
            cap=False,
            closed=closed,
        )
        outer_offset = len(self.vertices)
        self.merge(outer_loft)
        bottom_cap_indices = [outer_offset + i for i in range(len(outer))]
        top_cap_indices = [outer_offset + len(outer) + i for i in range(len(outer))]

        # Interior side walls for each hole.
        holes_cw: List[List[Vec2]] = []
        for hole in holes:
            hole_cw = list(reversed(hole))
            hole_offset = len(self.vertices)
            # ``LoftGeometry`` intentionally normalizes every closed profile to
            # CCW.  That is correct for an exterior wall, but a hole boundary
            # must remain CW so its side-wall normals point into the void and
            # its vertex order stays aligned with the cap triangulator's flat
            # outer+holes ring.  Build this small loft directly to preserve the
            # domain-boundary winding.
            for z in (z0, z1):
                for x, y in hole_cw:
                    self.add_vertex(x, y, z)

            segment_count = len(hole_cw) if closed else len(hole_cw) - 1
            top_offset = hole_offset + len(hole_cw)
            for i in range(segment_count):
                i2 = (i + 1) % len(hole_cw)
                a = hole_offset + i
                b = hole_offset + i2
                c = top_offset + i2
                d = top_offset + i
                self.add_face(a, b, c)
                self.add_face(a, c, d)

            bottom_cap_indices.extend(hole_offset + i for i in range(len(hole_cw)))
            top_cap_indices.extend(top_offset + i for i in range(len(hole_cw)))
            holes_cw.append(hole_cw)

        if not (cap and closed):
            return

        cap_ring, cap_triangles = _triangulate_polygon_with_holes(outer, holes)
        expected_cap_ring = list(outer)
        for hole_cw in holes_cw:
            expected_cap_ring.extend(hole_cw)

        reuse_sidewall_rims = len(cap_ring) == len(expected_cap_ring) and all(
            _points_match_2d(actual, expected)
            for actual, expected in zip(cap_ring, expected_cap_ring)
        )
        if reuse_sidewall_rims:
            bottom_idx = bottom_cap_indices
            top_idx = top_cap_indices
        else:
            bottom_idx = [self.add_vertex(x, y, z0) for (x, y) in cap_ring]
            top_idx = [self.add_vertex(x, y, z1) for (x, y) in cap_ring]

        for a, b, c in cap_triangles:
            # Bottom points downward, top points upward.
            self.add_face(bottom_idx[c], bottom_idx[b], bottom_idx[a])
            self.add_face(top_idx[a], top_idx[b], top_idx[c])


def _mesh_bounds(geometry: "MeshGeometry") -> tuple[Vec3, Vec3]:
    if not geometry.vertices:
        raise ValueError("MeshGeometry has no vertices")
    xs = [vertex[0] for vertex in geometry.vertices]
    ys = [vertex[1] for vertex in geometry.vertices]
    zs = [vertex[2] for vertex in geometry.vertices]
    return (
        (min(xs), min(ys), min(zs)),
        (max(xs), max(ys), max(zs)),
    )


def _normalize_pitch_2d(
    pitch: Union[float, Sequence[float]],
    *,
    name: str = "pitch",
) -> tuple[float, float]:
    if isinstance(pitch, Sequence) and not isinstance(pitch, (str, bytes)):
        values = list(pitch)
        if len(values) != 2:
            raise ValueError(f"{name} must be a positive scalar or a 2-tuple")
        pitch_x = float(values[0])
        pitch_y = float(values[1])
    else:
        pitch_x = float(pitch)
        pitch_y = float(pitch)
    if pitch_x <= 0.0 or pitch_y <= 0.0:
        raise ValueError(f"{name} values must be positive")
    return (pitch_x, pitch_y)


def _centered_axis_positions(limit: float, pitch: float) -> list[float]:
    if limit < -1e-9:
        return []
    if limit <= 1e-9:
        return [0.0]
    count = int((2.0 * limit) / pitch) + 1
    if count <= 0:
        return []
    return [((index - (count - 1) * 0.5) * pitch) for index in range(count)]


def _mesh_geometry_shifted_to_z0(geometry: "MeshGeometry") -> "MeshGeometry":
    shifted = geometry.copy()
    mins, _maxs = _mesh_bounds(shifted)
    shifted.translate(0.0, 0.0, -mins[2])
    return shifted


def _mesh_geometry_shifted_to_axis0(geometry: "MeshGeometry", axis: int) -> "MeshGeometry":
    shifted = geometry.copy()
    mins, _maxs = _mesh_bounds(shifted)
    delta = [0.0, 0.0, 0.0]
    delta[int(axis)] = -mins[int(axis)]
    shifted.translate(delta[0], delta[1], delta[2])
    return shifted


def _adopt_mesh_geometry(target: "MeshGeometry", geometry: "MeshGeometry") -> None:
    target.vertices = cast(List[Vec3], [tuple(vertex) for vertex in geometry.vertices])
    target.faces = cast(List[Face], [tuple(face) for face in geometry.faces])
    target.uvs = (
        cast(List[Vec2], [tuple(uv) for uv in geometry.uvs]) if geometry.uvs is not None else None
    )
    target.texture = geometry.texture
    target.normal_crease_deg = geometry.normal_crease_deg
    _copy_manifold_provenance(geometry, target)
