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

import numpy as np

DEFAULT_NORMAL_CREASE_DEGREES = 60.0
_MAX_NORMAL_FAN_FACES = 4096
_MAX_NORMAL_PAIR_BLOCK_ELEMENTS = 1_000_000


def _corner_fans(
    candidates: list[tuple[int, int]],
    faces: np.ndarray,
    *,
    vertex_index: int,
) -> list[list[int]]:
    """Return edge-connected candidate positions around one vertex."""

    edge_owners: dict[int, list[int]] = {}
    for candidate_index, (face_index, _corner) in enumerate(candidates):
        for other_vertex in faces[face_index]:
            other = int(other_vertex)
            if other != vertex_index:
                edge_owners.setdefault(other, []).append(candidate_index)

    adjacency: list[set[int]] = [set() for _candidate in candidates]
    for owners in edge_owners.values():
        # Every face sharing this edge belongs to the same fan. A star has the
        # same connected components as a clique and keeps pathological
        # non-manifold input linear instead of allocating O(k^2) links.
        if len(owners) > 1:
            root = owners[0]
            for owner in owners[1:]:
                adjacency[root].add(owner)
                adjacency[owner].add(root)

    unseen = set(range(len(candidates)))
    fans: list[list[int]] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        fan = [start]
        while stack:
            current = stack.pop()
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
            fan.extend(neighbors)
        fans.append(fan)
    return fans


def crease_aware_face_varying_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    crease_angle_degrees: float = DEFAULT_NORMAL_CREASE_DEGREES,
) -> np.ndarray:
    """Return one corner-angle-weighted normal per triangle corner.

    The crease is a shading policy: it groups incident face normals whose
    angular separation is at most ``crease_angle_degrees`` without changing
    positions, topology, collision, or mass properties.
    """

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh vertices must have shape (n, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh faces must be triangles")
    crease = float(crease_angle_degrees)
    if not math.isfinite(crease) or crease < 0.0 or crease > 180.0:
        raise ValueError("crease_angle_degrees must be finite and between 0 and 180")
    if len(faces) == 0:
        return np.empty((0, 3), dtype=float)

    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    if np.any(lengths <= 1e-12) or not np.all(np.isfinite(lengths)):
        raise ValueError("mesh contains degenerate triangles")
    face_normals = crosses / lengths[:, None]

    # Corner-angle weighting is independent of the triangulation diagonal and
    # gives cylinders/spheres stable smooth normals even with unequal faces.
    corner_angles = np.empty((len(faces), 3), dtype=float)
    for corner in range(3):
        edge_a = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        edge_b = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        edge_a_length = np.linalg.norm(edge_a, axis=1)
        edge_b_length = np.linalg.norm(edge_b, axis=1)
        denominator = edge_a_length * edge_b_length
        if np.any(denominator <= 1e-12):
            raise ValueError("mesh contains degenerate triangle corners")
        cosine = np.einsum("ij,ij->i", edge_a, edge_b) / denominator
        corner_angles[:, corner] = np.arccos(np.clip(cosine, -1.0, 1.0))

    incident_corners: list[list[tuple[int, int]]] = [[] for _ in range(len(vertices))]
    for face_index, face in enumerate(faces):
        for corner, vertex_index in enumerate(face):
            incident_corners[int(vertex_index)].append((face_index, corner))

    cosine_threshold = math.cos(math.radians(crease))
    corner_normals = np.empty((len(faces), 3, 3), dtype=float)
    for vertex_index, candidates in enumerate(incident_corners):
        if not candidates:
            continue
        # Bound work before constructing fan adjacency. A pathological
        # non-manifold edge can make every candidate an owner of the same
        # edge, so building the adjacency itself would otherwise be O(k^2)
        # before the per-fan guard below had a chance to run.
        if len(candidates) > _MAX_NORMAL_FAN_FACES:
            raise ValueError(
                f"mesh vertex {vertex_index} has {len(candidates):,} incident faces, above the "
                f"{_MAX_NORMAL_FAN_FACES:,}-face normal smoothing limit"
            )
        for fan in _corner_fans(candidates, faces, vertex_index=vertex_index):
            if len(fan) > _MAX_NORMAL_FAN_FACES:
                raise ValueError(
                    f"mesh vertex {vertex_index} has {len(fan):,} edge-connected incident "
                    f"faces, above the {_MAX_NORMAL_FAN_FACES:,}-face normal smoothing limit"
                )
            adjacent_faces = np.fromiter(
                (candidates[index][0] for index in fan),
                dtype=np.int64,
                count=len(fan),
            )
            adjacent_corners = np.fromiter(
                (candidates[index][1] for index in fan),
                dtype=np.int64,
                count=len(fan),
            )
            normals = face_normals[adjacent_faces]
            weights = corner_angles[adjacent_faces, adjacent_corners]
            weighted_sources = normals * weights[:, None]
            block_size = max(
                1,
                min(
                    len(fan),
                    _MAX_NORMAL_PAIR_BLOCK_ELEMENTS // max(1, len(fan)),
                ),
            )
            for start in range(0, len(fan), block_size):
                stop = min(start + block_size, len(fan))
                in_smoothing_group = normals[start:stop] @ normals.T >= cosine_threshold - 1e-12
                weighted_normals = in_smoothing_group @ weighted_sources
                normal_lengths = np.linalg.norm(weighted_normals, axis=1)
                valid = (normal_lengths > 1e-12) & np.isfinite(normal_lengths)
                weighted_normals[valid] /= normal_lengths[valid, None]
                weighted_normals[~valid] = normals[start:stop][~valid]
                corner_normals[
                    adjacent_faces[start:stop],
                    adjacent_corners[start:stop],
                ] = weighted_normals

    return corner_normals.reshape(-1, 3)


__all__ = ["DEFAULT_NORMAL_CREASE_DEGREES", "crease_aware_face_varying_normals"]
