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

from typing import Tuple

import numpy as np

from .booleans import _manifold_from_geometry
from .common import (
    _clear_primitive_provenance,
    _set_manifold_provenance,
)
from .primitives import MeshGeometry


def _geometry_from_manifold_mesh(
    manifold,
    *,
    normal_crease_deg: float | None,
) -> MeshGeometry:
    """Convert a manifold back to MeshGeometry keeping the raw triangulation.

    Unlike booleans._geometry_from_manifold, this does NOT regroup coplanar
    faces — curved/smoothed surfaces have no coplanar structure to recover and
    must keep their dense triangulation as-is.
    """
    if manifold is None or manifold.is_empty():
        return MeshGeometry(normal_crease_deg=normal_crease_deg)
    out_mesh = manifold.to_mesh()
    vp = np.asarray(out_mesh.vert_properties, dtype=np.float64)
    tri = np.asarray(out_mesh.tri_verts, dtype=np.int64)
    if vp.size == 0 or tri.size == 0:
        return MeshGeometry(normal_crease_deg=normal_crease_deg)
    geometry = MeshGeometry(
        vertices=[(float(x), float(y), float(z)) for x, y, z in vp[:, :3]],
        faces=[(int(a), int(b), int(c)) for a, b, c in tri],
        normal_crease_deg=normal_crease_deg,
    )
    _set_manifold_provenance(geometry, manifold)
    return geometry


def _vertices_faces_arrays(geometry: MeshGeometry) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(geometry, MeshGeometry):
        raise TypeError("geometry must be MeshGeometry")
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    faces = np.asarray(geometry.faces, dtype=np.int64)
    return vertices, faces


def _geometry_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    normal_crease_deg: float | None,
) -> MeshGeometry:
    geometry = MeshGeometry(
        vertices=[(float(x), float(y), float(z)) for x, y, z in vertices],
        faces=[(int(a), int(b), int(c)) for a, b, c in faces],
        normal_crease_deg=normal_crease_deg,
    )
    _clear_primitive_provenance(geometry)
    return geometry


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > 1e-12
    normals[nonzero] /= lengths[nonzero][:, None]
    normals[~nonzero] = (0.0, 0.0, 1.0)
    return normals


def subdivide_geometry(geometry: MeshGeometry, *, levels: int = 1) -> MeshGeometry:
    """
    Midpoint-subdivide each triangle into 4. Topology-only: the shape is
    unchanged, but the denser mesh gives smoothing/displacement room to work.
    """
    levels = int(levels)
    if levels < 1:
        raise ValueError("levels must be >= 1")
    vertices, faces = _vertices_faces_arrays(geometry)
    if vertices.size == 0 or faces.size == 0:
        return geometry.copy()

    for _ in range(levels):
        face_count = faces.shape[0]
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges_sorted = np.sort(edges, axis=1)
        unique_edges, inverse = np.unique(edges_sorted, axis=0, return_inverse=True)
        midpoints = vertices[unique_edges].mean(axis=1)
        midpoint_index = vertices.shape[0] + inverse
        m01 = midpoint_index[:face_count]
        m12 = midpoint_index[face_count : 2 * face_count]
        m20 = midpoint_index[2 * face_count :]
        vertices = np.concatenate([vertices, midpoints])
        faces = np.concatenate(
            [
                np.stack([faces[:, 0], m01, m20], axis=1),
                np.stack([faces[:, 1], m12, m01], axis=1),
                np.stack([faces[:, 2], m20, m12], axis=1),
                np.stack([m01, m12, m20], axis=1),
            ]
        )

    return _geometry_from_arrays(
        vertices,
        faces,
        normal_crease_deg=geometry.normal_crease_deg,
    )


def smooth_geometry(
    geometry: MeshGeometry,
    *,
    min_sharp_angle: float = 60.0,
    min_smoothness: float = 0.0,
    subdivisions: int = 3,
) -> MeshGeometry:
    """
    Round off a faceted solid into a smooth surface.

    Edges whose dihedral angle exceeds ``min_sharp_angle`` (degrees) are kept
    sharp; everything else is interpolated with smooth tangents and refined by
    ``subdivisions`` segments per edge. The input must be a valid manifold
    solid (same requirement as the boolean operations).
    """
    subdivisions = int(subdivisions)
    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")
    min_sharp_angle = float(min_sharp_angle)
    if not 0.0 <= min_sharp_angle <= 180.0:
        raise ValueError("min_sharp_angle must be between 0 and 180 degrees")
    min_smoothness = float(min_smoothness)
    if not 0.0 <= min_smoothness <= 1.0:
        raise ValueError("min_smoothness must be between 0 and 1")

    manifold = _manifold_from_geometry(geometry, name="geometry")
    smoothed = manifold.smooth_out(min_sharp_angle, min_smoothness).refine(subdivisions)
    return _geometry_from_manifold_mesh(
        smoothed,
        normal_crease_deg=geometry.normal_crease_deg,
    )


def taubin_smooth_geometry(
    geometry: MeshGeometry,
    *,
    iterations: int = 10,
    lambda_factor: float = 0.5,
    mu_factor: float = -0.53,
) -> MeshGeometry:
    """
    Taubin lambda/mu smoothing: relaxes the mesh towards an organic surface
    without the shrinkage of plain Laplacian smoothing. Works on any mesh
    (does not require a manifold solid) and preserves triangle count —
    subdivide first if the mesh is coarse.
    """
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    lambda_factor = float(lambda_factor)
    mu_factor = float(mu_factor)
    if not 0.0 < lambda_factor <= 1.0:
        raise ValueError("lambda_factor must be in (0, 1]")
    if not -1.0 <= mu_factor < 0.0:
        raise ValueError("mu_factor must be in [-1, 0)")

    vertices, faces = _vertices_faces_arrays(geometry)
    if vertices.size == 0 or faces.size == 0:
        return geometry.copy()

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    counts = np.zeros(vertices.shape[0], dtype=np.float64)
    np.add.at(counts, edges[:, 0], 1.0)
    np.add.at(counts, edges[:, 1], 1.0)
    connected = counts > 0

    for _ in range(iterations):
        for factor in (lambda_factor, mu_factor):
            neighbor_sum = np.zeros_like(vertices)
            np.add.at(neighbor_sum, edges[:, 0], vertices[edges[:, 1]])
            np.add.at(neighbor_sum, edges[:, 1], vertices[edges[:, 0]])
            average = vertices.copy()
            average[connected] = neighbor_sum[connected] / counts[connected][:, None]
            vertices = vertices + factor * (average - vertices)

    return _geometry_from_arrays(
        vertices,
        faces,
        normal_crease_deg=geometry.normal_crease_deg,
    )


def _hash_lattice(coords: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic integer-lattice hash mapped to [0, 1)."""
    h = coords.astype(np.uint64)
    seed_term = np.uint64(((seed & 0xFFFFFFFF) * 0x27D4EB2F165667C5) % (1 << 64))
    with np.errstate(over="ignore"):
        value = (
            h[..., 0] * np.uint64(0x9E3779B185EBCA87)
            + h[..., 1] * np.uint64(0xC2B2AE3D27D4EB4F)
            + h[..., 2] * np.uint64(0x165667B19E3779F9)
            + seed_term
        )
        value ^= value >> np.uint64(33)
        value *= np.uint64(0xFF51AFD7ED558CCD)
        value ^= value >> np.uint64(33)
    return (value >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _value_noise(points: np.ndarray, seed: int) -> np.ndarray:
    """Trilinearly interpolated value noise in [-1, 1] over a unit lattice."""
    base = np.floor(points)
    frac = points - base
    # Smoothstep fade for C1 continuity across lattice cells.
    fade = frac * frac * (3.0 - 2.0 * frac)
    base_int = base.astype(np.int64)

    result = np.zeros(points.shape[0], dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner = base_int + np.array([dx, dy, dz], dtype=np.int64)
                weight = (
                    (fade[:, 0] if dx else 1.0 - fade[:, 0])
                    * (fade[:, 1] if dy else 1.0 - fade[:, 1])
                    * (fade[:, 2] if dz else 1.0 - fade[:, 2])
                )
                result += weight * _hash_lattice(corner, seed)
    return result * 2.0 - 1.0


def displace_geometry(
    geometry: MeshGeometry,
    *,
    amplitude: float,
    scale: float = 0.05,
    seed: int = 0,
    octaves: int = 1,
) -> MeshGeometry:
    """
    Displace vertices along their normals with deterministic procedural noise.

    ``amplitude`` is the peak offset and ``scale`` the approximate feature
    size, both in mesh units (meters). Same inputs + same seed always produce
    the same result. Subdivide coarse meshes first or the noise has no
    vertices to act on.
    """
    amplitude = float(amplitude)
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("scale must be > 0")
    octaves = int(octaves)
    if octaves < 1:
        raise ValueError("octaves must be >= 1")

    vertices, faces = _vertices_faces_arrays(geometry)
    if vertices.size == 0 or faces.size == 0:
        return geometry.copy()

    normals = _vertex_normals(vertices, faces)
    noise = np.zeros(vertices.shape[0], dtype=np.float64)
    octave_amplitude = 1.0
    octave_scale = scale
    total_weight = 0.0
    for octave in range(octaves):
        noise += octave_amplitude * _value_noise(vertices / octave_scale, seed + octave)
        total_weight += octave_amplitude
        octave_amplitude *= 0.5
        octave_scale *= 0.5
    noise /= total_weight

    displaced = vertices + normals * (noise * amplitude)[:, None]
    return _geometry_from_arrays(
        displaced,
        faces,
        normal_crease_deg=geometry.normal_crease_deg,
    )


def simplify_geometry(geometry: MeshGeometry, *, tolerance: float) -> MeshGeometry:
    """
    Reduce triangle count while keeping the surface within ``tolerance``
    (mesh units) of the original. Useful for collision meshes and for taming
    dense smoothed/displaced outputs. The input must be a valid manifold solid.
    """
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")
    manifold = _manifold_from_geometry(geometry, name="geometry")
    simplified = manifold.simplify(tolerance)
    return _geometry_from_manifold_mesh(
        simplified,
        normal_crease_deg=geometry.normal_crease_deg,
    )
