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

from math import cos, sin, sqrt
from typing import Callable, Tuple, cast

import manifold3d as _m3d

from .common import _set_manifold_provenance
from .primitives import MeshGeometry

Vec3 = Tuple[float, float, float]
_SdfFn = Callable[[float, float, float], float]
_Bounds = Tuple[Vec3, Vec3]


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _smin(a: float, b: float, k: float) -> float:
    """Polynomial smooth minimum (Inigo Quilez); k is the blend radius."""
    h = _clamp01(0.5 + 0.5 * (b - a) / k)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _bounds_union(a: _Bounds, b: _Bounds) -> _Bounds:
    return (
        (min(a[0][0], b[0][0]), min(a[0][1], b[0][1]), min(a[0][2], b[0][2])),
        (max(a[1][0], b[1][0]), max(a[1][1], b[1][1]), max(a[1][2], b[1][2])),
    )


def _bounds_expand(bounds: _Bounds, margin: float) -> _Bounds:
    (x0, y0, z0), (x1, y1, z1) = bounds
    return ((x0 - margin, y0 - margin, z0 - margin), (x1 + margin, y1 + margin, z1 + margin))


class Sdf:
    """
    A signed-distance field: negative inside, positive outside, zero on the
    surface. Combine fields with union/difference/intersection (or their
    smooth_* variants for organic blends), then mesh with ``mesh_from_sdf``.

    Each field carries a conservative bounding box so meshing does not need
    explicit bounds.
    """

    def __init__(self, fn: _SdfFn, bounds: _Bounds):
        self._fn = fn
        self.bounds = bounds

    def __call__(self, x: float, y: float, z: float) -> float:
        return self._fn(float(x), float(y), float(z))

    # --- combinations -----------------------------------------------------

    def union(self, other: "Sdf") -> "Sdf":
        fa, fb = self._fn, other._fn
        return Sdf(
            lambda x, y, z: min(fa(x, y, z), fb(x, y, z)), _bounds_union(self.bounds, other.bounds)
        )

    def intersection(self, other: "Sdf") -> "Sdf":
        fa, fb = self._fn, other._fn
        return Sdf(lambda x, y, z: max(fa(x, y, z), fb(x, y, z)), self.bounds)

    def difference(self, other: "Sdf") -> "Sdf":
        fa, fb = self._fn, other._fn
        return Sdf(lambda x, y, z: max(fa(x, y, z), -fb(x, y, z)), self.bounds)

    def smooth_union(self, other: "Sdf", k: float) -> "Sdf":
        k = _positive(k, name="k")
        fa, fb = self._fn, other._fn
        bounds = _bounds_expand(_bounds_union(self.bounds, other.bounds), k)
        return Sdf(lambda x, y, z: _smin(fa(x, y, z), fb(x, y, z), k), bounds)

    def smooth_intersection(self, other: "Sdf", k: float) -> "Sdf":
        k = _positive(k, name="k")
        fa, fb = self._fn, other._fn
        return Sdf(lambda x, y, z: -_smin(-fa(x, y, z), -fb(x, y, z), k), self.bounds)

    def smooth_difference(self, other: "Sdf", k: float) -> "Sdf":
        k = _positive(k, name="k")
        fa, fb = self._fn, other._fn
        return Sdf(lambda x, y, z: -_smin(-fa(x, y, z), fb(x, y, z), k), self.bounds)

    # --- transforms -------------------------------------------------------

    def translate(self, dx: float, dy: float, dz: float) -> "Sdf":
        dx, dy, dz = float(dx), float(dy), float(dz)
        fn = self._fn
        (x0, y0, z0), (x1, y1, z1) = self.bounds
        return Sdf(
            lambda x, y, z: fn(x - dx, y - dy, z - dz),
            ((x0 + dx, y0 + dy, z0 + dz), (x1 + dx, y1 + dy, z1 + dz)),
        )

    def scale(self, factor: float) -> "Sdf":
        """Uniform scale (non-uniform scaling does not preserve distances)."""
        factor = _positive(factor, name="factor")
        fn = self._fn
        inv = 1.0 / factor
        (x0, y0, z0), (x1, y1, z1) = self.bounds
        return Sdf(
            lambda x, y, z: fn(x * inv, y * inv, z * inv) * factor,
            ((x0 * factor, y0 * factor, z0 * factor), (x1 * factor, y1 * factor, z1 * factor)),
        )

    def rotate_x(self, angle: float) -> "Sdf":
        return self._rotate(angle, axis=0)

    def rotate_y(self, angle: float) -> "Sdf":
        return self._rotate(angle, axis=1)

    def rotate_z(self, angle: float) -> "Sdf":
        return self._rotate(angle, axis=2)

    def _rotate(self, angle: float, *, axis: int) -> "Sdf":
        angle = float(angle)
        c, s = cos(angle), sin(angle)
        fn = self._fn

        if axis == 0:
            rotated: _SdfFn = lambda x, y, z: fn(x, c * y + s * z, -s * y + c * z)  # noqa: E731
        elif axis == 1:
            rotated = lambda x, y, z: fn(c * x - s * z, y, s * x + c * z)  # noqa: E731
        else:
            rotated = lambda x, y, z: fn(c * x + s * y, -s * x + c * y, z)  # noqa: E731

        corners = [
            (x, y, z)
            for x in (self.bounds[0][0], self.bounds[1][0])
            for y in (self.bounds[0][1], self.bounds[1][1])
            for z in (self.bounds[0][2], self.bounds[1][2])
        ]
        if axis == 0:
            moved = [(x, c * y - s * z, s * y + c * z) for x, y, z in corners]
        elif axis == 1:
            moved = [(c * x + s * z, y, -s * x + c * z) for x, y, z in corners]
        else:
            moved = [(c * x - s * y, s * x + c * y, z) for x, y, z in corners]
        mins = tuple(min(p[i] for p in moved) for i in range(3))
        maxs = tuple(max(p[i] for p in moved) for i in range(3))
        return Sdf(rotated, cast(_Bounds, (mins, maxs)))

    # --- modifiers ----------------------------------------------------------

    def round(self, radius: float) -> "Sdf":
        """Offset the surface outward, rounding all edges by ``radius``."""
        radius = _positive(radius, name="radius")
        fn = self._fn
        return Sdf(lambda x, y, z: fn(x, y, z) - radius, _bounds_expand(self.bounds, radius))

    def shell(self, thickness: float) -> "Sdf":
        """Hollow the solid into a shell of the given wall thickness."""
        thickness = _positive(thickness, name="thickness")
        fn = self._fn
        half = thickness / 2.0
        return Sdf(
            lambda x, y, z: abs(fn(x, y, z)) - half,
            _bounds_expand(self.bounds, half),
        )


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


# --- primitives -------------------------------------------------------------


def sdf_sphere(radius: float) -> Sdf:
    """Sphere of ``radius`` centered at the origin."""
    radius = _positive(radius, name="radius")
    return Sdf(
        lambda x, y, z: sqrt(x * x + y * y + z * z) - radius,
        ((-radius, -radius, -radius), (radius, radius, radius)),
    )


def sdf_box(size: Tuple[float, float, float], *, radius: float = 0.0) -> Sdf:
    """Axis-aligned box of full extents ``size``, optionally edge-rounded."""
    hx = _positive(size[0], name="size[0]") / 2.0
    hy = _positive(size[1], name="size[1]") / 2.0
    hz = _positive(size[2], name="size[2]") / 2.0
    radius = float(radius)
    if radius < 0.0:
        raise ValueError("radius must be >= 0")
    if radius >= min(hx, hy, hz):
        raise ValueError("radius must be smaller than half the smallest extent")
    ix, iy, iz = hx - radius, hy - radius, hz - radius

    def fn(x: float, y: float, z: float) -> float:
        qx = abs(x) - ix
        qy = abs(y) - iy
        qz = abs(z) - iz
        ox = qx if qx > 0.0 else 0.0
        oy = qy if qy > 0.0 else 0.0
        oz = qz if qz > 0.0 else 0.0
        outside = sqrt(ox * ox + oy * oy + oz * oz)
        inside = min(max(qx, max(qy, qz)), 0.0)
        return outside + inside - radius

    return Sdf(fn, ((-hx, -hy, -hz), (hx, hy, hz)))


def sdf_cylinder(radius: float, height: float) -> Sdf:
    """Z-aligned cylinder centered at the origin."""
    radius = _positive(radius, name="radius")
    half = _positive(height, name="height") / 2.0

    def fn(x: float, y: float, z: float) -> float:
        dr = sqrt(x * x + y * y) - radius
        dz = abs(z) - half
        ox = dr if dr > 0.0 else 0.0
        oz = dz if dz > 0.0 else 0.0
        outside = sqrt(ox * ox + oz * oz)
        inside = min(max(dr, dz), 0.0)
        return outside + inside

    return Sdf(fn, ((-radius, -radius, -half), (radius, radius, half)))


def sdf_capsule(height: float, radius: float) -> Sdf:
    """Z-aligned capsule: a segment of the given height with rounded caps."""
    radius = _positive(radius, name="radius")
    half = _positive(height, name="height") / 2.0

    def fn(x: float, y: float, z: float) -> float:
        cz = min(max(z, -half), half)
        dz = z - cz
        return sqrt(x * x + y * y + dz * dz) - radius

    extent = half + radius
    return Sdf(fn, ((-radius, -radius, -extent), (radius, radius, extent)))


def sdf_torus(radius: float, tube: float) -> Sdf:
    """Torus in the XY plane: ring radius ``radius``, tube radius ``tube``."""
    radius = _positive(radius, name="radius")
    tube = _positive(tube, name="tube")

    def fn(x: float, y: float, z: float) -> float:
        q = sqrt(x * x + y * y) - radius
        return sqrt(q * q + z * z) - tube

    extent = radius + tube
    return Sdf(fn, ((-extent, -extent, -tube), (extent, extent, tube)))


def sdf_ellipsoid(radii: Tuple[float, float, float]) -> Sdf:
    """Approximate ellipsoid SDF (exact on the surface, approximate away from it)."""
    rx = _positive(radii[0], name="radii[0]")
    ry = _positive(radii[1], name="radii[1]")
    rz = _positive(radii[2], name="radii[2]")

    def fn(x: float, y: float, z: float) -> float:
        k0 = sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
        k1 = sqrt((x / rx**2) ** 2 + (y / ry**2) ** 2 + (z / rz**2) ** 2)
        if k1 <= 1e-12:
            return -min(rx, ry, rz)
        return k0 * (k0 - 1.0) / k1

    return Sdf(fn, ((-rx, -ry, -rz), (rx, ry, rz)))


# --- meshing ------------------------------------------------------------------


def mesh_from_sdf(
    sdf: Sdf,
    *,
    edge_length: float,
    bounds: _Bounds | None = None,
    margin: float | None = None,
) -> MeshGeometry:
    """
    Mesh the zero level set of ``sdf`` into a watertight MeshGeometry.

    ``edge_length`` sets surface resolution in meters (smaller = finer; aim
    for roughly 1/20 of the object size). Bounds default to the field's own
    bounding box plus a margin of one ``edge_length``.
    """
    edge_length = _positive(edge_length, name="edge_length")
    if bounds is None:
        bounds = _bounds_expand(sdf.bounds, margin if margin is not None else edge_length)
    (x0, y0, z0), (x1, y1, z1) = bounds
    if not (x1 > x0 and y1 > y0 and z1 > z0):
        raise ValueError("bounds must span a positive volume")

    fn = sdf._fn
    # manifold3d expects positive-inside; Sdf uses the conventional
    # negative-inside, so flip the sign at the boundary.
    manifold = _m3d.Manifold.level_set(
        lambda x, y, z: -fn(x, y, z),
        [x0, y0, z0, x1, y1, z1],
        float(edge_length),
    )
    if manifold.status() != _m3d.Error.NoError or manifold.is_empty():
        raise ValueError(
            "SDF produced no solid within bounds; check the field sign convention "
            "(negative inside) and the meshing bounds"
        )

    out_mesh = manifold.to_mesh()
    geometry = MeshGeometry(
        vertices=[(float(v[0]), float(v[1]), float(v[2])) for v in out_mesh.vert_properties],
        faces=[(int(a), int(b), int(c)) for a, b, c in out_mesh.tri_verts],
    )
    _set_manifold_provenance(geometry, manifold)
    return geometry
