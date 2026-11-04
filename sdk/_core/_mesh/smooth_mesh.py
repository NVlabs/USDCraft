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
from collections.abc import Callable, Sequence

import manifold3d as _m3d
import numpy as np
import trimesh

from ..errors import ValidationError
from .booleans import _manifold_from_geometry
from .common import _set_manifold_provenance
from .health import require_mesh_health
from .normals import DEFAULT_NORMAL_CREASE_DEGREES
from .primitives import MeshGeometry

_AXES = {"x": 0, "y": 1, "z": 2}
# -0.25 is the most negative fourth-order profile coefficient that preserves
# smooth_max(a, b) >= max(a, b) throughout the blend band. Going lower can
# erode a claimed union or add material during a claimed difference.
_BLEND_PROFILES = {"tight": -0.25, "round": 0.0, "soft": 1.0}
_FIELD_CHUNK = 200_000
_MAX_GRID_POINTS = 2_000_000
_MAX_GENERATED_VERTICES = 250_000
_MAX_GENERATED_TRIANGLES = 500_000


class SnapRefused(ValueError):
    """Raised when a bounded or constrained snap cannot be satisfied safely."""


def _require_geometry(value: object, *, name: str) -> MeshGeometry:
    if not isinstance(value, MeshGeometry):
        raise TypeError(f"{name} must be MeshGeometry")
    return value


def _merged_normal_crease(geometries: Sequence[MeshGeometry]) -> float | None:
    effective = [
        DEFAULT_NORMAL_CREASE_DEGREES
        if geometry.normal_crease_deg is None
        else geometry.normal_crease_deg
        for geometry in geometries
    ]
    if not effective or any(value != effective[0] for value in effective[1:]):
        return None
    # Canonicalize the established default so commutative weld inputs cannot
    # produce observably different metadata merely because one spelling was
    # explicit ``60`` and another used ``None``.
    return None if effective[0] == DEFAULT_NORMAL_CREASE_DEGREES else effective[0]


def _as_trimesh(geometry: MeshGeometry) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(geometry.vertices, dtype=np.float64),
        faces=np.asarray(geometry.faces, dtype=np.int64),
        process=False,
        validate=False,
    )


def _geometry_from_manifold_mesh(
    manifold: _m3d.Manifold,
    *,
    enforce_generated_budget: bool = False,
) -> MeshGeometry:
    output = manifold.to_mesh()
    vertices = np.asarray(output.vert_properties, dtype=np.float64)
    faces = np.asarray(output.tri_verts, dtype=np.int64)
    if enforce_generated_budget and (
        len(vertices) > _MAX_GENERATED_VERTICES or len(faces) > _MAX_GENERATED_TRIANGLES
    ):
        raise ValueError(
            "smooth mesh output would create "
            f"{len(vertices):,} vertices and {len(faces):,} triangles, above the "
            f"{_MAX_GENERATED_VERTICES:,}-vertex/{_MAX_GENERATED_TRIANGLES:,}-triangle "
            "safety budget; increase tolerance or simplify the inputs"
        )
    geometry = MeshGeometry(
        vertices=[(float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in vertices],
        faces=[(int(face[0]), int(face[1]), int(face[2])) for face in faces],
    )
    _set_manifold_provenance(geometry, manifold)
    return geometry


def _positive_solid_count(manifold: _m3d.Manifold) -> int:
    bounds = manifold.bounding_box()
    extents = np.asarray(bounds[3:6], dtype=np.float64) - np.asarray(bounds[0:3], dtype=np.float64)
    scale = float(np.linalg.norm(extents))
    linear_tolerance = 1e-9
    volume_tolerance = linear_tolerance * max(
        scale * scale,
        linear_tolerance * linear_tolerance,
    )
    return sum(part.volume() > volume_tolerance for part in manifold.decompose())


def _canonical_solid(
    value: object,
    *,
    name: str,
) -> tuple[MeshGeometry, trimesh.Trimesh, _m3d.Manifold]:
    geometry = _require_geometry(value, name=name)
    require_mesh_health(
        geometry,
        name=name,
        require_watertight=True,
        require_single_solid=True,
    )
    try:
        manifold = _manifold_from_geometry(geometry, name=name)
    except (ValueError, RuntimeError) as exc:
        raise ValidationError(f"{name} is not a valid closed manifold solid: {exc}") from exc
    if manifold.is_empty():
        raise ValidationError(f"{name} must be a non-empty closed solid")

    # manifold3d gives the distance field a consistently oriented, welded source
    # without silently repairing an input which failed the public health contract.
    canonical = _geometry_from_manifold_mesh(manifold)
    require_mesh_health(
        canonical,
        name=f"canonical {name}",
        require_watertight=True,
        require_single_solid=True,
    )
    return canonical, _as_trimesh(canonical), manifold


def _nearest_gap(
    anchor: trimesh.Trimesh,
    piece: trimesh.Trimesh,
) -> tuple[float, np.ndarray]:
    """Return the sampled surface gap and a unit direction from piece to anchor."""
    anchor_query = trimesh.proximity.ProximityQuery(anchor)
    piece_query = trimesh.proximity.ProximityQuery(piece)

    points_on_anchor, distances_from_piece, _ = anchor_query.on_surface(piece.vertices)
    piece_index = int(np.argmin(distances_from_piece))
    point_on_anchor = points_on_anchor[piece_index]
    point_on_piece = piece.vertices[piece_index]
    distance = float(distances_from_piece[piece_index])

    points_on_piece, distances_from_anchor, _ = piece_query.on_surface(anchor.vertices)
    anchor_index = int(np.argmin(distances_from_anchor))
    if float(distances_from_anchor[anchor_index]) < distance:
        point_on_anchor = anchor.vertices[anchor_index]
        point_on_piece = points_on_piece[anchor_index]
        distance = float(distances_from_anchor[anchor_index])

    direction = np.asarray(point_on_anchor, dtype=np.float64) - np.asarray(
        point_on_piece, dtype=np.float64
    )
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        # Touching surfaces have no closest-point direction. Their center-to-center
        # direction is the least surprising direction in which to add overlap.
        direction = np.asarray(anchor.centroid, dtype=np.float64) - np.asarray(
            piece.centroid, dtype=np.float64
        )
        length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return distance, np.zeros(3, dtype=np.float64)
    return distance, direction / length


def _solids_contact_or_overlap(
    anchor: MeshGeometry,
    piece: MeshGeometry,
    *,
    sampled_gap: float,
) -> bool:
    coordinate_scale = max(
        (
            abs(float(value))
            for vertices in (anchor.vertices, piece.vertices)
            for vertex in vertices
            for value in vertex
        ),
        default=1.0,
    )
    scale = max(1.0, coordinate_scale)
    if sampled_gap <= scale * 1e-9:
        return True
    try:
        anchor_solid = _manifold_from_geometry(anchor, name="anchor")
        piece_solid = _manifold_from_geometry(piece, name="piece")
    except (TypeError, ValueError, RuntimeError):
        # snap_to is also useful for open visual meshes. For those, only the
        # sampled surface-distance result is available and no solid claim is made.
        return False
    return _positive_solid_count(anchor_solid + piece_solid) == 1


def snap_to(
    anchor: MeshGeometry,
    piece: MeshGeometry,
    *,
    overlap: float = 0.004,
    max_move: float = 0.02,
    axis: str | None = None,
) -> MeshGeometry:
    """Translate ``piece`` to close a small surface gap to ``anchor``.

    The operation mutates and returns ``piece``. It is only a geometry placement
    helper: it does not create a joint, constraint, collider, or physics weld.
    ``overlap`` moves the piece slightly through the contact surface so a later
    boolean or :func:`weld` has a real intersection. ``max_move`` is a refusal
    threshold which prevents a misplaced part from being silently teleported.

    Already touching or intersecting solids are returned unchanged. Pass ``axis``
    as ``"x"``, ``"y"``, or ``"z"`` to constrain translation to one world axis.
    Axis-constrained snapping is accepted only when the sampled nearest-surface
    direction is substantially aligned with that axis; otherwise an Euclidean gap
    is not a sound estimate of axial clearance and ``SnapRefused`` is raised.
    Nearest points are sampled in both directions, so very sparse meshes can require
    a larger modeling tolerance than finely tessellated meshes.
    """
    anchor = _require_geometry(anchor, name="anchor")
    piece = _require_geometry(piece, name="piece")
    require_mesh_health(anchor, name="anchor")
    require_mesh_health(piece, name="piece")
    if not anchor.vertices or not anchor.faces:
        raise ValueError("anchor must be a non-empty triangle mesh")
    if not piece.vertices or not piece.faces:
        raise ValueError("piece must be a non-empty triangle mesh")

    overlap = float(overlap)
    if not math.isfinite(overlap) or overlap < 0.0:
        raise ValueError("overlap must be finite and non-negative")
    max_move = float(max_move)
    if not math.isfinite(max_move) or max_move <= 0.0:
        raise ValueError("max_move must be finite and positive")
    axis_name: str | None = None
    if axis is not None:
        axis_name = str(axis).strip().lower()
        if axis_name not in _AXES:
            raise ValueError("axis must be one of 'x', 'y', or 'z'")

    anchor_mesh = _as_trimesh(anchor)
    piece_mesh = _as_trimesh(piece)
    gap, direction = _nearest_gap(anchor_mesh, piece_mesh)
    if _solids_contact_or_overlap(anchor, piece, sampled_gap=gap):
        return piece

    move_distance = gap + overlap
    if axis_name is not None:
        component = _AXES[axis_name]
        sign = float(np.sign(direction[component]))
        if sign == 0.0:
            center_delta = float(anchor_mesh.centroid[component] - piece_mesh.centroid[component])
            sign = float(np.sign(center_delta))
        if sign == 0.0:
            raise SnapRefused(
                f"cannot infer whether to move piece along {axis_name!r}; "
                "place it off the anchor center or snap without an axis constraint"
            )
        alignment = abs(float(direction[component]))
        if alignment < 0.95:
            raise SnapRefused(
                f"nearest separation is not aligned with axis {axis_name!r} "
                f"(alignment={alignment:.3f}); place the piece explicitly or snap "
                "without an axis constraint"
            )
        direction = np.zeros(3, dtype=np.float64)
        direction[component] = sign
        move_distance = gap / alignment + overlap

    movement = move_distance * direction
    magnitude = float(np.linalg.norm(movement))
    if magnitude > max_move:
        raise SnapRefused(
            f"closing the gap needs {magnitude * 1000.0:.1f} mm, over max_move "
            f"{max_move * 1000.0:.1f} mm; fix the placement or shape instead of snapping"
        )
    return piece.translate(float(movement[0]), float(movement[1]), float(movement[2]))


def _smooth_max(
    first: np.ndarray,
    second: np.ndarray,
    *,
    radius: float,
    profile: str,
) -> np.ndarray:
    amount = np.clip(0.5 + 0.5 * (first - second) / radius, 0.0, 1.0)
    blend = amount * (1.0 - amount)
    return (
        second * (1.0 - amount)
        + first * amount
        + radius * blend
        + _BLEND_PROFILES[profile] * 4.0 * radius * blend * blend
    )


def _surface_controls(
    *,
    radius: float,
    tolerance: float | None,
    profile: str,
) -> tuple[float, float, str]:
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    tolerance = radius * 0.25 if tolerance is None else float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    profile = str(profile).strip().lower().replace("-", "_")
    if profile not in _BLEND_PROFILES:
        raise ValueError("profile must be 'tight', 'round', or 'soft'")
    return radius, tolerance, profile


def _grid_shape(extents: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    dimensions: list[int] = []
    for extent in extents:
        ratio = float(extent) / tolerance
        if not math.isfinite(ratio):
            return (_MAX_GRID_POINTS + 1,) * 3
        dimensions.append(max(2, math.ceil(ratio) + 1))
    return dimensions[0], dimensions[1], dimensions[2]


def _grid_point_count(extents: np.ndarray, tolerance: float) -> int:
    return math.prod(_grid_shape(extents, tolerance))


def _minimum_tolerance(base_extents: np.ndarray, *, radius: float) -> float:
    def count(candidate: float) -> int:
        margin = radius + 2.371 * candidate
        return _grid_point_count(base_extents + 2.0 * margin, candidate)

    low = 0.0
    high = max(float(np.max(base_extents)), radius)
    while count(high) > _MAX_GRID_POINTS:
        high *= 2.0
    for _ in range(64):
        middle = (low + high) * 0.5
        if middle <= 0.0 or count(middle) > _MAX_GRID_POINTS:
            low = middle
        else:
            high = middle
    return high


def _grid_points(
    start: int,
    stop: int,
    dimensions: tuple[int, int, int],
    lower: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    flat = np.arange(start, stop, dtype=np.int64)
    yz = dimensions[1] * dimensions[2]
    x = flat // yz
    remainder = flat - x * yz
    y = remainder // dimensions[2]
    z = remainder - y * dimensions[2]
    return lower + np.column_stack((x, y, z)) * spacing


def _sample_mesh_field(
    query: trimesh.proximity.ProximityQuery,
    mesh_bounds: np.ndarray,
    points: np.ndarray,
    *,
    band: float,
    far: float,
) -> np.ndarray:
    values = np.full(len(points), -far, dtype=np.float64)
    near = np.all(
        (points >= mesh_bounds[0] - band) & (points <= mesh_bounds[1] + band),
        axis=1,
    )
    if not np.any(near):
        return values
    signed = np.asarray(query.signed_distance(points[near]), dtype=np.float64)
    if not np.all(np.isfinite(signed)):
        raise ValueError("signed-distance evaluation produced non-finite values")
    # trimesh uses positive-inside, matching the blend field convention here.
    values[near] = signed
    return values


def _trilinear_field(
    samples: np.ndarray,
    *,
    lower: np.ndarray,
    spacing: np.ndarray,
) -> Callable[[float, float, float], float]:
    nx, ny, nz = samples.shape
    lx, ly, lz = (float(value) for value in lower)
    ix_scale, iy_scale, iz_scale = (1.0 / float(value) for value in spacing)

    def coordinate(value: float, origin: float, scale: float, size: int) -> tuple[int, float]:
        position = (float(value) - origin) * scale
        if position <= 0.0:
            return 0, 0.0
        if position >= size - 1:
            return size - 2, 1.0
        index = int(position)
        return index, position - index

    def field(x: float, y: float, z: float) -> float:
        ix, tx = coordinate(x, lx, ix_scale, nx)
        iy, ty = coordinate(y, ly, iy_scale, ny)
        iz, tz = coordinate(z, lz, iz_scale, nz)

        c00 = samples[ix, iy, iz] * (1.0 - tz) + samples[ix, iy, iz + 1] * tz
        c01 = samples[ix, iy + 1, iz] * (1.0 - tz) + samples[ix, iy + 1, iz + 1] * tz
        c10 = samples[ix + 1, iy, iz] * (1.0 - tz) + samples[ix + 1, iy, iz + 1] * tz
        c11 = samples[ix + 1, iy + 1, iz] * (1.0 - tz) + samples[ix + 1, iy + 1, iz + 1] * tz
        c0 = c00 * (1.0 - ty) + c01 * ty
        c1 = c10 * (1.0 - ty) + c11 * ty
        return float(c0 * (1.0 - tx) + c1 * tx)

    return field


def _connection_gap(
    first_mesh: trimesh.Trimesh,
    first_manifold: _m3d.Manifold,
    second_mesh: trimesh.Trimesh,
    second_manifold: _m3d.Manifold,
) -> float:
    if _positive_solid_count(first_manifold + second_manifold) == 1:
        return 0.0
    return _nearest_gap(first_mesh, second_mesh)[0]


def _require_connected_inputs(
    meshes: Sequence[trimesh.Trimesh],
    manifolds: Sequence[_m3d.Manifold],
    *,
    max_gap: float,
    operation: str,
) -> None:
    gaps = [
        (
            first,
            second,
            _connection_gap(meshes[first], manifolds[first], meshes[second], manifolds[second]),
        )
        for first in range(len(meshes))
        for second in range(first + 1, len(meshes))
    ]
    connected = {0}
    while True:
        additions = {
            second if first in connected else first
            for first, second, gap in gaps
            if gap <= max_gap and ((first in connected) != (second in connected))
        }
        if not additions:
            break
        connected.update(additions)
    if len(connected) == len(meshes):
        return
    nearest = min(
        gap for first, second, gap in gaps if (first in connected) != (second in connected)
    )
    raise ValueError(
        f"{operation} inputs are separated by {nearest * 1000.0:.2f} mm; "
        "overlap them, call snap_to, or increase max_gap"
    )


def _extract_level_set(
    field: np.ndarray,
    *,
    dimensions: tuple[int, int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    spacing: np.ndarray,
    tolerance: float,
) -> MeshGeometry:
    samples = field.reshape(dimensions)
    manifold = _m3d.Manifold.level_set(
        _trilinear_field(samples, lower=lower, spacing=spacing),
        [*lower, *upper],
        float(tolerance),
    )
    if manifold.status() != _m3d.Error.NoError or manifold.is_empty():
        raise ValueError(
            f"smooth mesh level set produced an invalid or empty solid (status={manifold.status()})"
        )
    simplified = manifold.simplify(float(np.min(spacing)) * 0.05)
    if simplified.status() == _m3d.Error.NoError and not simplified.is_empty():
        manifold = simplified
    # level_set can leave exactly collinear triangles on lattice-aligned planar
    # faces even after tolerance simplification. A zero-tolerance manifold pass
    # removes only those redundant facets and preserves the sampled surface.
    cleaned = manifold.simplify(0.0)
    if cleaned.status() == _m3d.Error.NoError and not cleaned.is_empty():
        manifold = cleaned
    # The boolean conversion helper intentionally re-triangulates large coplanar
    # regions. A sampled level set contains thousands of small, near-coplanar
    # facets, where that optimization can incorrectly merge distinct planes. Keep
    # manifold3d's already valid topology verbatim for this generated surface.
    return _geometry_from_manifold_mesh(manifold, enforce_generated_budget=True)


def _smooth_operation(
    meshes: Sequence[trimesh.Trimesh],
    *,
    radius: float,
    tolerance: float,
    profile: str,
    operation: str,
    trim: trimesh.Trimesh | None = None,
) -> MeshGeometry:
    if operation == "union":
        minimum = np.min([mesh.bounds[0] for mesh in meshes], axis=0)
        maximum = np.max([mesh.bounds[1] for mesh in meshes], axis=0)
    else:
        minimum = np.asarray(meshes[0].bounds[0], dtype=np.float64)
        maximum = np.asarray(meshes[0].bounds[1], dtype=np.float64)

    # Offset the lattice so authored planar faces do not all land on zero-valued
    # samples, which can otherwise create zero-volume tetrahedra at corners.
    base_extents = maximum - minimum
    margin = radius + 2.371 * tolerance
    if not math.isfinite(margin):
        raise ValueError("radius and tolerance produce a non-finite field margin")
    lower = minimum - margin
    upper = maximum + margin
    extents = upper - lower
    if not (
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(np.isfinite(extents))
        and np.all(extents > 0.0)
    ):
        raise ValueError("smooth mesh operation requires finite, positive sampling bounds")
    dimensions = _grid_shape(extents, tolerance)
    point_count = math.prod(dimensions)
    if point_count > _MAX_GRID_POINTS:
        minimum_tolerance = _minimum_tolerance(base_extents, radius=radius)
        raise ValueError(
            f"tolerance creates {point_count:,} field samples, above the "
            f"{_MAX_GRID_POINTS:,} safety limit; use tolerance >= "
            f"{minimum_tolerance:.6g} m for this size"
        )
    spacing = extents / (np.asarray(dimensions, dtype=np.float64) - 1.0)
    far = max(1.0, float(np.linalg.norm(extents)) * 10.0)
    band = radius + 2.0 * tolerance
    field = np.full(point_count, -far, dtype=np.float64)

    for mesh_index, mesh in enumerate(meshes):
        query = trimesh.proximity.ProximityQuery(mesh)
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        for start in range(0, point_count, _FIELD_CHUNK):
            stop = min(start + _FIELD_CHUNK, point_count)
            points = _grid_points(start, stop, dimensions, lower, spacing)
            values = _sample_mesh_field(query, bounds, points, band=band, far=far)
            if mesh_index == 0:
                field[start:stop] = values
            elif operation == "difference":
                field[start:stop] = -_smooth_max(
                    -field[start:stop], values, radius=radius, profile=profile
                )
            else:
                field[start:stop] = _smooth_max(
                    field[start:stop], values, radius=radius, profile=profile
                )

    if trim is not None:
        query = trimesh.proximity.ProximityQuery(trim)
        bounds = np.asarray(trim.bounds, dtype=np.float64)
        for start in range(0, point_count, _FIELD_CHUNK):
            stop = min(start + _FIELD_CHUNK, point_count)
            points = _grid_points(start, stop, dimensions, lower, spacing)
            trim_values = _sample_mesh_field(query, bounds, points, band=band, far=far)
            field[start:stop] = np.minimum(field[start:stop], -trim_values)

    return _extract_level_set(
        field,
        dimensions=dimensions,
        lower=lower,
        upper=upper,
        spacing=spacing,
        tolerance=tolerance,
    )


def weld(
    *geometries: MeshGeometry,
    radius: float = 0.006,
    tolerance: float | None = None,
    profile: str = "round",
    max_gap: float = 0.0,
    trim: MeshGeometry | None = None,
) -> MeshGeometry:
    """Fuse two or more closed mesh solids through a smooth generated transition.

    This is a geometry-only field operation, not an articulation joint or a
    physical welding model. It resamples the complete input surfaces, so use exact
    CAD booleans for fits, bores, controlled wall thickness, or dimension-critical
    features. UVs, textures, material partitions, and analytic primitive provenance
    are intentionally not preserved on the generated mesh.

    ``radius`` controls blend reach, ``tolerance`` controls field and triangle
    resolution, and ``profile`` is ``"tight"``, ``"round"``, or ``"soft"``.
    Inputs must form a connected graph by overlap, unless ``max_gap`` explicitly
    permits a small generated bridge. ``trim`` can remove an internal protrusion
    after fusing into a hollow body.
    """
    if len(geometries) < 2:
        raise ValueError("weld needs at least two geometries")
    radius, tolerance, profile = _surface_controls(
        radius=radius,
        tolerance=tolerance,
        profile=profile,
    )
    max_gap = float(max_gap)
    if not math.isfinite(max_gap) or max_gap < 0.0:
        raise ValueError("max_gap must be finite and non-negative")
    if max_gap > radius:
        raise ValueError("max_gap must not exceed radius; a farther field cannot form a weld")

    canonical = [
        _canonical_solid(geometry, name=f"geometries[{index}]")
        for index, geometry in enumerate(geometries)
    ]
    meshes = [item[1] for item in canonical]
    manifolds = [item[2] for item in canonical]
    _require_connected_inputs(
        meshes,
        manifolds,
        max_gap=max_gap,
        operation="weld",
    )

    trim_mesh = None
    if trim is not None:
        trim_mesh = _canonical_solid(trim, name="trim")[1]
    result = _smooth_operation(
        meshes,
        radius=radius,
        tolerance=tolerance,
        profile=profile,
        operation="union",
        trim=trim_mesh,
    )
    result.normal_crease_deg = _merged_normal_crease(tuple(geometries))
    require_mesh_health(
        result,
        name="weld result",
        require_watertight=True,
        require_single_solid=True,
    )
    return result


def smooth_difference(
    geometry: MeshGeometry,
    *cutters: MeshGeometry,
    radius: float = 0.006,
    tolerance: float | None = None,
    profile: str = "round",
    max_gap: float = 0.0,
) -> MeshGeometry:
    """Subtract closed cutter meshes with a smooth generated transition.

    This is a geometry-only field operation. It does not define contact, collision,
    or joint semantics, and it resamples the complete base surface. Use an exact
    boolean for dimension-critical holes or mating features. Every input must be a
    healthy, watertight, single solid. Cutters must reach the base directly or via
    another cutter unless ``max_gap`` explicitly permits a shallow generated recess.
    """
    geometry = _require_geometry(geometry, name="geometry")
    if not cutters:
        raise ValueError("smooth_difference needs at least one cutter")
    radius, tolerance, profile = _surface_controls(
        radius=radius,
        tolerance=tolerance,
        profile=profile,
    )
    max_gap = float(max_gap)
    if not math.isfinite(max_gap) or max_gap < 0.0:
        raise ValueError("max_gap must be finite and non-negative")
    if max_gap > radius:
        raise ValueError(
            "max_gap must not exceed radius; a farther cutter cannot affect the smooth difference"
        )

    inputs = (geometry, *cutters)
    canonical = [
        _canonical_solid(item, name="geometry" if index == 0 else f"cutters[{index - 1}]")
        for index, item in enumerate(inputs)
    ]
    meshes = [item[1] for item in canonical]
    manifolds = [item[2] for item in canonical]
    _require_connected_inputs(
        meshes,
        manifolds,
        max_gap=max_gap,
        operation="smooth_difference",
    )
    result = _smooth_operation(
        meshes,
        radius=radius,
        tolerance=tolerance,
        profile=profile,
        operation="difference",
    )
    result.normal_crease_deg = geometry.normal_crease_deg
    require_mesh_health(
        result,
        name="smooth_difference result",
        require_watertight=True,
        require_single_solid=True,
    )
    return result


__all__ = ["SnapRefused", "smooth_difference", "snap_to", "weld"]
