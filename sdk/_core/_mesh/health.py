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
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from ..errors import ValidationError
from ..types import _coerce_normal_crease_deg
from .common import Vec3
from .primitives import MeshGeometry, _validated_faces, _validated_uvs, _validated_vertices

MeshHealthSeverity = Literal["warning", "error"]
MeshBounds = tuple[Vec3, Vec3]


@dataclass(frozen=True, slots=True)
class MeshHealthFinding:
    """One structured, localizable mesh-quality or topology finding."""

    issue: str
    severity: MeshHealthSeverity
    count: int
    message: str
    sample_indices: tuple[int, ...] = ()
    bounds: MeshBounds | None = None


@dataclass(frozen=True, slots=True)
class MeshHealthReport:
    """Non-mutating diagnostics for a triangle ``MeshGeometry``.

    ``healthy`` means no structurally unsafe error findings. ``clean`` is
    stricter and means no findings at all. Open surfaces, disconnected visual
    islands, slivers, and inward orientation remain visible diagnostics rather
    than becoming a global compilation or promotion gate.
    """

    vertex_count: int
    triangle_count: int
    # Number of positive-volume solids. Enclosed inward cavity shells do not
    # inflate this count.
    component_count: int
    # Raw edge-connected triangle shells, including cavity boundaries.
    surface_component_count: int
    watertight: bool
    winding_consistent: bool
    signed_volume: float
    findings: tuple[MeshHealthFinding, ...] = ()

    @property
    def healthy(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(finding.issue for finding in self.findings)


@dataclass(frozen=True, slots=True)
class _ComponentShell:
    minimum: np.ndarray
    maximum: np.ndarray
    triangles: np.ndarray
    sample_points: np.ndarray


def mesh_health(
    geometry: "MeshGeometry",
    *,
    tolerance: float = 1e-9,
) -> MeshHealthReport:
    """Inspect triangle quality, connectivity, orientation, and closure.

    Gross schema errors (non-finite coordinates, non-triangular faces,
    non-integer/out-of-range indices, or misaligned UVs) are rejected by
    ``MeshGeometry.validate`` before topology analysis. Findings here are
    authoring diagnostics and are never consulted by the global Host gate.
    """

    if not isinstance(geometry, MeshGeometry):
        raise TypeError("geometry must be MeshGeometry")
    normalized_tolerance = float(tolerance)
    if not math.isfinite(normalized_tolerance) or normalized_tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    # Validate into local normalized arrays so this diagnostic remains truly
    # non-mutating even after low-level direct list edits.
    validated_vertices = _validated_vertices(geometry.vertices)
    validated_faces = _validated_faces(
        geometry.faces,
        vertex_count=len(validated_vertices),
    )
    _validated_uvs(geometry.uvs, vertex_count=len(validated_vertices))
    _coerce_normal_crease_deg(
        geometry.normal_crease_deg,
        name="mesh_geometry.normal_crease_deg",
    )

    vertices = np.asarray(validated_vertices, dtype=np.float64).reshape((-1, 3))
    faces = np.asarray(validated_faces, dtype=np.int64).reshape((-1, 3))
    findings: list[MeshHealthFinding] = []

    if len(vertices) == 0 or len(faces) == 0:
        findings.append(
            MeshHealthFinding(
                issue="empty_mesh",
                severity="error",
                count=1,
                message="Mesh must contain at least one vertex and one triangle",
            )
        )
        return MeshHealthReport(
            vertex_count=len(vertices),
            triangle_count=len(faces),
            component_count=0,
            surface_component_count=0,
            watertight=False,
            winding_consistent=False,
            signed_volume=0.0,
            findings=tuple(findings),
        )

    mesh_span = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    cross_tolerance = normalized_tolerance * max(mesh_span, normalized_tolerance)
    triangles = vertices[faces]
    cross_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    doubled_areas = np.linalg.norm(cross_vectors, axis=1)
    degenerate_faces = np.flatnonzero(doubled_areas <= cross_tolerance)
    if len(degenerate_faces):
        findings.append(
            _face_finding(
                "degenerate_faces",
                "error",
                degenerate_faces,
                faces=faces,
                vertices=vertices,
                message="Triangles with near-zero area are numerically unsafe",
            )
        )

    edge_lengths_sq = (
        np.sum((triangles[:, 1] - triangles[:, 0]) ** 2, axis=1)
        + np.sum((triangles[:, 2] - triangles[:, 1]) ** 2, axis=1)
        + np.sum((triangles[:, 0] - triangles[:, 2]) ** 2, axis=1)
    )
    triangle_quality = np.zeros(len(faces), dtype=np.float64)
    nonzero_edges = edge_lengths_sq > cross_tolerance * cross_tolerance
    triangle_quality[nonzero_edges] = (
        2.0 * math.sqrt(3.0) * doubled_areas[nonzero_edges] / edge_lengths_sq[nonzero_edges]
    )
    sliver_faces = np.flatnonzero((doubled_areas > cross_tolerance) & (triangle_quality < 1e-5))
    if len(sliver_faces):
        findings.append(
            _face_finding(
                "sliver_faces",
                "warning",
                sliver_faces,
                faces=faces,
                vertices=vertices,
                message="Extremely thin triangles can destabilize rendering and geometry queries",
            )
        )

    canonical_faces = np.sort(faces, axis=1)
    _, inverse_faces, face_occurrences = np.unique(
        canonical_faces,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    duplicate_face_indices = np.flatnonzero(face_occurrences[inverse_faces] > 1)
    if len(duplicate_face_indices):
        findings.append(
            _face_finding(
                "duplicate_faces",
                "error",
                duplicate_face_indices,
                faces=faces,
                vertices=vertices,
                message="Duplicate triangles make surface ownership ambiguous",
            )
        )

    duplicate_vertex_indices = _duplicate_vertex_indices(
        vertices,
        tolerance=normalized_tolerance,
    )
    if duplicate_vertex_indices:
        findings.append(
            _vertex_finding(
                "duplicate_vertices",
                "warning",
                duplicate_vertex_indices,
                vertices=vertices,
                message="Coincident vertices should be welded when they represent one surface",
            )
        )

    used_vertices = set(int(index) for index in faces.reshape(-1))
    unused_vertices = [index for index in range(len(vertices)) if index not in used_vertices]
    if unused_vertices:
        findings.append(
            _vertex_finding(
                "unused_vertices",
                "warning",
                unused_vertices,
                vertices=vertices,
                message="Vertices not referenced by a triangle add stale mesh data",
            )
        )

    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for face_index, (a, b, c) in enumerate(faces):
        for start, end in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (start, end) if start < end else (end, start)
            orientation = 1 if (start, end) == key else -1
            edge_occurrences[key].append((face_index, orientation))

    boundary_edges = [edge for edge, owners in edge_occurrences.items() if len(owners) == 1]
    nonmanifold_edges = [edge for edge, owners in edge_occurrences.items() if len(owners) > 2]
    inconsistent_edges = [
        edge
        for edge, owners in edge_occurrences.items()
        if len(owners) == 2 and owners[0][1] == owners[1][1]
    ]
    if boundary_edges:
        findings.append(
            _edge_finding(
                "boundary_edges",
                "warning",
                boundary_edges,
                vertices=vertices,
                message="Edges owned by one triangle form open surface boundaries",
            )
        )
    if nonmanifold_edges:
        findings.append(
            _edge_finding(
                "nonmanifold_edges",
                "error",
                nonmanifold_edges,
                vertices=vertices,
                message="Edges owned by more than two triangles are non-manifold",
            )
        )
    if inconsistent_edges:
        findings.append(
            _edge_finding(
                "inconsistent_winding",
                "error",
                inconsistent_edges,
                vertices=vertices,
                message="Adjacent triangles traverse their shared edge in the same direction",
            )
        )

    nonmanifold_vertices = _nonmanifold_vertex_indices(
        faces,
        edge_occurrences,
        vertex_count=len(vertices),
    )
    if nonmanifold_vertices:
        findings.append(
            _vertex_finding(
                "nonmanifold_vertices",
                "error",
                nonmanifold_vertices,
                vertices=vertices,
                message=("Incident triangles form multiple edge-disconnected fans at one vertex"),
            )
        )

    surface_components = _face_components(len(faces), edge_occurrences)
    surface_component_count = len(surface_components)

    signed_volume = _signed_triangle_volume(triangles)
    watertight = not boundary_edges and not nonmanifold_edges
    winding_consistent = not inconsistent_edges and not nonmanifold_edges
    # Convert the linear tolerance to a scale-aware volume tolerance without
    # mixing meter and square-meter quantities. This keeps valid micron-scale
    # solids classifiable while remaining proportional for ordinary assets.
    volume_tolerance = normalized_tolerance * max(
        mesh_span * mesh_span,
        normalized_tolerance * normalized_tolerance,
    )
    component_volumes = [
        _signed_triangle_volume(triangles[list(component)]) for component in surface_components
    ]
    # Classify closure in O(F + E). Scanning every edge once per component is
    # quadratic for meshes made from many disconnected shells.
    face_component_indices = np.empty(len(faces), dtype=np.int64)
    for component_index, component in enumerate(surface_components):
        face_component_indices[list(component)] = component_index
    component_closed = [True] * surface_component_count
    for owners in edge_occurrences.values():
        component_index = int(face_component_indices[owners[0][0]])
        if len(owners) != 2 or owners[0][1] == owners[1][1]:
            component_closed[component_index] = False
    component_count = sum(
        closed and volume > volume_tolerance
        for closed, volume in zip(component_closed, component_volumes, strict=True)
    )
    if component_count > 1:
        findings.append(
            MeshHealthFinding(
                issue="multiple_components",
                severity="warning",
                count=component_count,
                message="One MeshGeometry contains multiple positive-volume solids",
                bounds=_bounds_for_vertices(vertices, range(len(vertices))),
            )
        )

    oriented_component_indices = [
        index
        for index, (closed, volume) in enumerate(
            zip(component_closed, component_volumes, strict=True)
        )
        if closed and abs(volume) > volume_tolerance
    ]
    oriented_shells = {
        index: _build_component_shell(
            surface_components[index],
            vertices=vertices,
            faces=faces,
        )
        for index in oriented_component_indices
    }
    containment_depths: dict[int, int] = {}
    for inner_index in oriented_component_indices:
        containment_depths[inner_index] = sum(
            _component_contains_component(
                oriented_shells[outer_index],
                oriented_shells[inner_index],
                tolerance=normalized_tolerance,
            )
            for outer_index in oriented_component_indices
            if outer_index != inner_index
        )

    # A valid boundary representation alternates orientation with containment:
    # positive outer solid, negative cavity, positive material island, and so on.
    invalid_negative_indices = [
        index
        for index, depth in containment_depths.items()
        if component_volumes[index] < 0.0 and depth % 2 == 0
    ]
    if invalid_negative_indices:
        negative_faces = sorted(
            face_index
            for index in invalid_negative_indices
            for face_index in surface_components[index]
        )
        findings.append(
            _face_finding(
                "inward_orientation",
                "error",
                np.asarray(negative_faces, dtype=np.int64),
                faces=faces,
                vertices=vertices,
                message=(
                    "Negative-volume shell occurs at an even containment depth and cannot "
                    "represent a valid cavity boundary"
                ),
            )
        )

    invalid_positive_indices = [
        index
        for index, depth in containment_depths.items()
        if component_volumes[index] > 0.0 and depth % 2 == 1
    ]
    if invalid_positive_indices:
        positive_faces = sorted(
            face_index
            for index in invalid_positive_indices
            for face_index in surface_components[index]
        )
        findings.append(
            _face_finding(
                "outward_cavity_orientation",
                "error",
                np.asarray(positive_faces, dtype=np.int64),
                faces=faces,
                vertices=vertices,
                message=(
                    "Positive-volume shell occurs at an odd containment depth where an inward "
                    "cavity boundary is required"
                ),
            )
        )

    return MeshHealthReport(
        vertex_count=len(vertices),
        triangle_count=len(faces),
        component_count=component_count,
        surface_component_count=surface_component_count,
        watertight=watertight,
        winding_consistent=winding_consistent,
        signed_volume=signed_volume,
        findings=tuple(findings),
    )


def require_mesh_health(
    geometry: "MeshGeometry",
    *,
    name: str = "geometry",
    require_watertight: bool = False,
    require_single_solid: bool = False,
    tolerance: float = 1e-9,
) -> MeshHealthReport:
    """Require operation-safe mesh health without creating a global SDK gate."""

    report = mesh_health(geometry, tolerance=tolerance)
    blockers = [finding.issue for finding in report.findings if finding.severity == "error"]
    if require_watertight and not report.watertight:
        blockers.append("not_watertight")
    if require_single_solid and report.component_count != 1:
        blockers.append(f"component_count={report.component_count}")
    if blockers:
        unique_blockers = tuple(dict.fromkeys(blockers))
        raise ValidationError(
            f"{name} failed mesh health requirements: {', '.join(unique_blockers)}"
        )
    return report


def _duplicate_vertex_indices(vertices: np.ndarray, *, tolerance: float) -> list[int]:
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    duplicate_indices: list[int] = []
    for index, vertex in enumerate(vertices):
        cell = tuple(int(round(float(value) / tolerance)) for value in vertex)
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates.extend(buckets.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()))
        if any(
            float(np.linalg.norm(vertex - vertices[candidate])) <= tolerance
            for candidate in candidates
        ):
            duplicate_indices.append(index)
        buckets[cell].append(index)
    return duplicate_indices


def _signed_triangle_volume(triangles: np.ndarray) -> float:
    """Compute oriented volume after a local translation for numeric stability."""

    if len(triangles) == 0:
        return 0.0
    reference = np.mean(triangles.reshape((-1, 3)), axis=0)
    shifted = triangles - reference
    return float(
        np.einsum(
            "ij,ij->i",
            shifted[:, 0],
            np.cross(shifted[:, 1], shifted[:, 2]),
        ).sum()
        / 6.0
    )


def _face_components(
    face_count: int,
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[frozenset[int]]:
    if face_count == 0:
        return []
    adjacency: list[set[int]] = [set() for _ in range(face_count)]
    for owners in edge_occurrences.values():
        owner_faces = [face_index for face_index, _orientation in owners]
        if len(owner_faces) > 1:
            root = owner_faces[0]
            for face_index in owner_faces[1:]:
                adjacency[root].add(face_index)
                adjacency[face_index].add(root)
    unseen = set(range(face_count))
    components: list[frozenset[int]] = []
    while unseen:
        start = unseen.pop()
        queue: deque[int] = deque([start])
        component = {start}
        while queue:
            current = queue.popleft()
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            queue.extend(neighbors)
            component.update(neighbors)
        components.append(frozenset(component))
    return components


def _nonmanifold_vertex_indices(
    faces: np.ndarray,
    edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]],
    *,
    vertex_count: int,
) -> list[int]:
    incident_faces: list[list[int]] = [[] for _index in range(vertex_count)]
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            incident_faces[int(vertex_index)].append(face_index)

    incident_edge_owners: list[list[list[int]]] = [[] for _index in range(vertex_count)]
    for (start, end), owners in edge_occurrences.items():
        owner_faces = [face_index for face_index, _orientation in owners]
        incident_edge_owners[start].append(owner_faces)
        incident_edge_owners[end].append(owner_faces)

    nonmanifold: list[int] = []
    for vertex_index, vertex_faces in enumerate(incident_faces):
        if len(vertex_faces) <= 1:
            continue
        face_set = set(vertex_faces)
        adjacency: dict[int, set[int]] = {face_index: set() for face_index in vertex_faces}
        for owner_faces in incident_edge_owners[vertex_index]:
            connected_owners = [face_index for face_index in owner_faces if face_index in face_set]
            if len(connected_owners) > 1:
                root = connected_owners[0]
                for face_index in connected_owners[1:]:
                    adjacency[root].add(face_index)
                    adjacency[face_index].add(root)

        unseen = set(vertex_faces)
        fan_count = 0
        while unseen and fan_count < 2:
            fan_count += 1
            stack = [unseen.pop()]
            while stack:
                neighbors = adjacency[stack.pop()] & unseen
                unseen.difference_update(neighbors)
                stack.extend(neighbors)
        if fan_count > 1:
            nonmanifold.append(vertex_index)
    return nonmanifold


def _component_contains_component(
    outer: _ComponentShell,
    inner: _ComponentShell,
    *,
    tolerance: float,
) -> bool:
    if np.any(inner.minimum <= outer.minimum + tolerance) or np.any(
        inner.maximum >= outer.maximum - tolerance
    ):
        return False

    # AABB containment is necessary but not sufficient for concave solids.
    # Test deterministic samples from the inner boundary against the actual
    # outer shell. Extremal vertices are always included so a separate or
    # partially protruding inverted shell cannot masquerade as a cavity.
    return all(
        _point_inside_triangle_shell(
            point,
            outer.triangles,
            tolerance=tolerance,
        )
        for point in inner.sample_points
    )


def _build_component_shell(
    component_faces: frozenset[int],
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> _ComponentShell:
    face_indices = np.fromiter(sorted(component_faces), dtype=np.int64)
    component_triangles = vertices[faces[face_indices]]
    vertex_indices = np.unique(faces[face_indices].reshape(-1))
    component_vertices = vertices[vertex_indices]
    minimum = component_vertices.min(axis=0)
    maximum = component_vertices.max(axis=0)
    if len(vertex_indices) <= 64:
        sampled_indices = vertex_indices
    else:
        positions = np.linspace(0, len(vertex_indices) - 1, num=58, dtype=np.int64)
        extrema = np.concatenate(
            (
                vertex_indices[np.argmin(component_vertices, axis=0)],
                vertex_indices[np.argmax(component_vertices, axis=0)],
            )
        )
        sampled_indices = np.unique(np.concatenate((vertex_indices[positions], extrema)))
    return _ComponentShell(
        minimum=minimum,
        maximum=maximum,
        triangles=component_triangles,
        sample_points=vertices[sampled_indices],
    )


def _point_inside_triangle_shell(
    point: np.ndarray,
    triangles: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    directions = (
        np.asarray((1.0, 0.371390676, 0.529822128), dtype=np.float64),
        np.asarray((0.21764288, 1.0, 0.68313005), dtype=np.float64),
        np.asarray((0.49153915, 0.32769277, 1.0), dtype=np.float64),
    )
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    inside_votes = 0
    epsilon = max(float(tolerance), 1e-12)
    for raw_direction in directions:
        direction = raw_direction / np.linalg.norm(raw_direction)
        cross = np.cross(np.broadcast_to(direction, edge_b.shape), edge_b)
        determinant = np.einsum("ij,ij->i", edge_a, cross)
        valid = np.abs(determinant) > epsilon
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        offset = point - triangles[:, 0]
        barycentric_u = inverse * np.einsum("ij,ij->i", offset, cross)
        qvec = np.cross(offset, edge_a)
        barycentric_v = inverse * (qvec @ direction)
        distance = inverse * np.einsum("ij,ij->i", edge_b, qvec)
        hits = distance[
            valid
            & (barycentric_u >= -epsilon)
            & (barycentric_v >= -epsilon)
            & (barycentric_u + barycentric_v <= 1.0 + epsilon)
            & (distance > epsilon)
        ]
        if len(hits):
            ordered = np.sort(hits)
            distinct_hit_count = 1 + int(np.count_nonzero(np.diff(ordered) > epsilon))
        else:
            distinct_hit_count = 0
        inside_votes += distinct_hit_count % 2
    return inside_votes >= 2


def _bounds_for_vertices(vertices: np.ndarray, indices: Iterable[int]) -> MeshBounds | None:
    index_list = list(dict.fromkeys(int(index) for index in indices))
    if not index_list:
        return None
    selected = vertices[index_list]
    minimum = selected.min(axis=0)
    maximum = selected.max(axis=0)
    return (
        (float(minimum[0]), float(minimum[1]), float(minimum[2])),
        (float(maximum[0]), float(maximum[1]), float(maximum[2])),
    )


def _face_finding(
    issue: str,
    severity: MeshHealthSeverity,
    face_indices: np.ndarray,
    *,
    faces: np.ndarray,
    vertices: np.ndarray,
    message: str,
) -> MeshHealthFinding:
    indices = [int(index) for index in face_indices]
    vertex_indices = faces[indices].reshape(-1) if indices else ()
    return MeshHealthFinding(
        issue=issue,
        severity=severity,
        count=len(indices),
        message=message,
        sample_indices=tuple(indices[:16]),
        bounds=_bounds_for_vertices(vertices, vertex_indices),
    )


def _edge_finding(
    issue: str,
    severity: MeshHealthSeverity,
    edges: list[tuple[int, int]],
    *,
    vertices: np.ndarray,
    message: str,
) -> MeshHealthFinding:
    sample_edges = edges[:8]
    sample_indices = tuple(index for edge in sample_edges for index in edge)
    return MeshHealthFinding(
        issue=issue,
        severity=severity,
        count=len(edges),
        message=message,
        sample_indices=sample_indices,
        bounds=_bounds_for_vertices(vertices, sample_indices),
    )


def _vertex_finding(
    issue: str,
    severity: MeshHealthSeverity,
    vertex_indices: list[int],
    *,
    vertices: np.ndarray,
    message: str,
) -> MeshHealthFinding:
    return MeshHealthFinding(
        issue=issue,
        severity=severity,
        count=len(vertex_indices),
        message=message,
        sample_indices=tuple(vertex_indices[:16]),
        bounds=_bounds_for_vertices(vertices, vertex_indices),
    )


__all__ = [
    "MeshHealthFinding",
    "MeshHealthReport",
    "mesh_health",
    "require_mesh_health",
]
