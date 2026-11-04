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

"""Normalize arbitrary Reconstruction geometry into one metric, Z-up source scene.

This module is intentionally Reconstruction-specific.  It resolves only mechanically declared
coordinate metadata, bakes scene transforms, and classifies topology without repairing geometry or
inferring semantic orientation.  The normalized scene is a Host/Viewer artifact; it is not an Agent
observation and must not be copied into a Reconstruction Agent workspace.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias, cast

import numpy as np
import trimesh
from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdUtils

from usdcraft.pipeline.hashing import sha256_file as _sha256

SignedAxis: TypeAlias = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
FrameValueSource: TypeAlias = Literal[
    "explicit",
    "embedded_metadata",
    "format_contract",
    "unresolved",
    "estimated_from_semantic_image",
]
SourceFormat: TypeAlias = Literal["usd", "gltf", "obj", "stl", "ply", "off", "xyz"]
ComponentKind: TypeAlias = Literal["mesh", "points"]
GeometryAuthority: TypeAlias = Literal["surface_only"]
InteriorSemantics: TypeAlias = Literal["unknown"]
SolidAuthority: TypeAlias = Literal["none"]
TopologyClass: TypeAlias = Literal[
    "closed_orientable_surface", "open_surface", "non_manifold_surface", "point_samples"
]

_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
_TRIMESH_FORMATS: dict[str, SourceFormat] = {
    ".glb": "gltf",
    ".gltf": "gltf",
    ".obj": "obj",
    ".stl": "stl",
    ".ply": "ply",
    ".off": "off",
    ".xyz": "xyz",
}
_CANONICAL_UP: SignedAxis = "+Z"
_CANONICAL_FRONT: SignedAxis = "-Y"


class SourceNormalizationError(ValueError):
    """The source cannot be normalized without changing or inventing evidence."""


class SourceMetadataError(SourceNormalizationError):
    """Required frame metadata is absent, contradictory, or invalid."""


class UnsupportedSourceGeometryError(SourceNormalizationError):
    """A visible source geometry type has no Phase 1 lossless adapter."""


@dataclass(frozen=True, slots=True)
class SourceInput:
    """One source path plus explicit frame overrides supplied by the user.

    Axis tokens are signed cardinal directions in the file's source-world frame.  ``front_axis``
    is optional because a mesh or point cloud generally carries no reliable semantic front.
    """

    path: Path
    up_axis: str | None = None
    meters_per_unit: float | None = None
    front_axis: str | None = None
    # How an explicit ``meters_per_unit`` was obtained; ``None`` means the user declared it. The
    # Provider arm passes ``estimated_from_semantic_image`` when it set the scale from a category
    # estimate of a unit-normalized mesh, so every manifest records that provenance.
    meters_per_unit_source: FrameValueSource | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())


@dataclass(frozen=True, slots=True)
class ResolvedSourceFrame:
    up_axis: SignedAxis
    up_axis_source: FrameValueSource
    meters_per_unit: float
    meters_per_unit_source: FrameValueSource
    front_axis: SignedAxis | None
    front_axis_source: FrameValueSource
    source_to_normalized: tuple[tuple[float, float, float, float], ...]

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.source_to_normalized, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SourceInspection:
    path: Path
    source_format: SourceFormat
    embedded_up_axis: SignedAxis | None
    embedded_meters_per_unit: float | None
    resolved_frame: ResolvedSourceFrame | None
    required_inputs: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.resolved_frame is not None and not self.required_inputs


@dataclass(frozen=True, slots=True)
class SourceDependency:
    path: Path
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class MeshTopology:
    classification: TopologyClass
    vertex_count: int
    triangle_count: int
    exact_welded_vertex_count: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    inconsistent_winding_edge_count: int
    degenerate_triangle_count: int
    duplicate_triangle_count: int
    oriented_surface_integral_m3: float
    surface_closed: bool


@dataclass(frozen=True, slots=True)
class NormalizedSourceComponent:
    component_id: str
    kind: ComponentKind
    provenance: str
    vertices: np.ndarray
    faces: np.ndarray | None
    normals: np.ndarray | None
    colors_rgba: np.ndarray | None
    topology: MeshTopology
    geometry_authority: GeometryAuthority = "surface_only"
    interior_semantics: InteriorSemantics = "unknown"
    solid_authority: SolidAuthority = "none"
    # Preview-only same-source appearance.  AXIAL and normalized.usdc intentionally ignore
    # this object; orientation carriers may preserve a source TextureVisuals/PBR material so
    # declared geometry renders do not discard embedded GLB appearance.
    render_visual: Any | None = None

    @property
    def bounds_min_xyz(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    @property
    def bounds_max_xyz(self) -> np.ndarray:
        return self.vertices.max(axis=0)


@dataclass(frozen=True, slots=True)
class NormalizedSourceScene:
    source_path: Path
    source_format: SourceFormat
    frame: ResolvedSourceFrame
    components: tuple[NormalizedSourceComponent, ...]
    bounds_min_xyz: np.ndarray
    bounds_max_xyz: np.ndarray
    source_dependencies: tuple[SourceDependency, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NormalizedSourceArtifacts:
    root: Path
    normalized_usdc: Path
    source_manifest: Path
    normalized_usdc_sha256: str
    source_manifest_sha256: str


def _source_format(path: Path) -> SourceFormat:
    suffix = path.suffix.lower()
    if suffix in _USD_SUFFIXES:
        return "usd"
    source_format = _TRIMESH_FORMATS.get(suffix)
    if source_format is None:
        supported = sorted(_USD_SUFFIXES | frozenset(_TRIMESH_FORMATS))
        raise SourceNormalizationError(
            f"unsupported source format {suffix or '<none>'!r}; expected one of {supported}"
        )
    return source_format


def _signed_axis(value: str, *, label: str) -> SignedAxis:
    token = value.strip().upper()
    if len(token) == 1:
        token = f"+{token}"
    if token not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
        raise SourceMetadataError(f"{label} must be one of +X, -X, +Y, -Y, +Z, -Z; got {value!r}")
    return cast(SignedAxis, token)


def _axis_vector(axis: SignedAxis) -> np.ndarray:
    result = np.zeros(3, dtype=np.float64)
    result["XYZ".index(axis[1])] = 1.0 if axis[0] == "+" else -1.0
    return result


def _rotation_to_canonical(up_axis: SignedAxis, front_axis: SignedAxis | None) -> np.ndarray:
    source_up = _axis_vector(up_axis)
    target_up = _axis_vector(_CANONICAL_UP)
    if front_axis is not None:
        source_front = _axis_vector(front_axis)
        if not math.isclose(
            float(np.dot(source_up, source_front)), 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise SourceMetadataError("front_axis must be orthogonal to up_axis")
        target_front = _axis_vector(_CANONICAL_FRONT)
        source_right = np.cross(source_up, source_front)
        target_right = np.cross(target_up, target_front)
        source_basis = np.column_stack((source_right, source_front, source_up))
        target_basis = np.column_stack((target_right, target_front, target_up))
        rotation = target_basis @ source_basis.T
    else:
        dot = float(np.dot(source_up, target_up))
        if dot > 1.0 - 1e-12:
            rotation = np.eye(3, dtype=np.float64)
        elif dot < -1.0 + 1e-12:
            # The antiparallel shortest rotation has no unique axis.  +X is a fixed,
            # mechanically reproducible choice and introduces no geometry-dependent yaw.
            rotation = np.diag((1.0, -1.0, -1.0))
        else:
            cross = np.cross(source_up, target_up)
            skew = np.asarray(
                (
                    (0.0, -cross[2], cross[1]),
                    (cross[2], 0.0, -cross[0]),
                    (-cross[1], cross[0], 0.0),
                ),
                dtype=np.float64,
            )
            rotation = np.eye(3) + skew + (skew @ skew) * (1.0 / (1.0 + dot))
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12):
        raise SourceMetadataError("resolved source axes did not produce an orthonormal rotation")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SourceMetadataError("resolved source axes would reflect the source frame")
    return rotation


def _resolved_frame(
    source: SourceInput,
    *,
    source_format: SourceFormat,
    embedded_up_axis: SignedAxis | None,
    embedded_meters_per_unit: float | None,
) -> tuple[ResolvedSourceFrame | None, tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    required: list[str] = []

    explicit_up = (
        _signed_axis(source.up_axis, label="up_axis") if source.up_axis is not None else None
    )
    explicit_front = (
        _signed_axis(source.front_axis, label="front_axis")
        if source.front_axis is not None
        else None
    )
    explicit_meters = None
    if source.meters_per_unit is not None:
        explicit_meters = float(source.meters_per_unit)
        if not math.isfinite(explicit_meters) or explicit_meters <= 0.0:
            raise SourceMetadataError("meters_per_unit must be finite and greater than zero")

    contract_up: SignedAxis | None = "+Y" if source_format == "gltf" else None
    contract_meters: float | None = 1.0 if source_format == "gltf" else None

    if explicit_up is not None:
        up_axis = explicit_up
        up_source: FrameValueSource = "explicit"
        inherited_up = embedded_up_axis or contract_up
        if inherited_up is not None and inherited_up != explicit_up:
            warnings.append(f"explicit up_axis {explicit_up} overrides declared {inherited_up}")
    elif embedded_up_axis is not None:
        up_axis = embedded_up_axis
        up_source = "embedded_metadata"
    elif contract_up is not None:
        up_axis = contract_up
        up_source = "format_contract"
    else:
        up_axis = None
        up_source = "unresolved"
        required.append("up_axis")

    if explicit_meters is not None:
        meters_per_unit = explicit_meters
        meters_source: FrameValueSource = source.meters_per_unit_source or "explicit"
        inherited_meters = embedded_meters_per_unit or contract_meters
        if inherited_meters is not None and not math.isclose(
            inherited_meters, explicit_meters, rel_tol=0.0, abs_tol=0.0
        ):
            warnings.append(
                "explicit meters_per_unit "
                f"{explicit_meters:.17g} overrides declared {inherited_meters:.17g}"
            )
    elif embedded_meters_per_unit is not None:
        meters_per_unit = embedded_meters_per_unit
        meters_source = "embedded_metadata"
    elif contract_meters is not None:
        meters_per_unit = contract_meters
        meters_source = "format_contract"
    else:
        meters_per_unit = None
        meters_source = "unresolved"
        required.append("meters_per_unit")

    if required:
        return None, tuple(required), tuple(warnings)
    assert up_axis is not None and meters_per_unit is not None
    rotation = _rotation_to_canonical(up_axis, explicit_front)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation * meters_per_unit
    return (
        ResolvedSourceFrame(
            up_axis=up_axis,
            up_axis_source=up_source,
            meters_per_unit=meters_per_unit,
            meters_per_unit_source=meters_source,
            front_axis=explicit_front,
            front_axis_source="explicit" if explicit_front is not None else "unresolved",
            source_to_normalized=cast(
                "tuple[tuple[float, float, float, float], ...]",
                tuple(tuple(float(value) for value in row) for row in transform),
            ),
        ),
        (),
        tuple(warnings),
    )


def inspect_source(source: SourceInput) -> SourceInspection:
    """Resolve the source format and trustworthy frame metadata without writing artifacts."""

    path = source.path
    if not path.is_file():
        raise FileNotFoundError(path)
    source_format = _source_format(path)
    embedded_up: SignedAxis | None = None
    embedded_meters: float | None = None
    if source_format == "usd":
        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
        if stage is None:
            raise SourceNormalizationError(f"failed to open USD stage: {path}")
        if stage.HasAuthoredMetadata(UsdGeom.Tokens.upAxis):
            embedded_up = _signed_axis(str(UsdGeom.GetStageUpAxis(stage)), label="USD upAxis")
        if stage.HasAuthoredMetadata(UsdGeom.Tokens.metersPerUnit):
            embedded_meters = float(UsdGeom.GetStageMetersPerUnit(stage))
            if not math.isfinite(embedded_meters) or embedded_meters <= 0.0:
                raise SourceMetadataError(
                    "authored USD metersPerUnit must be finite and greater than zero"
                )
    frame, required, warnings = _resolved_frame(
        source,
        source_format=source_format,
        embedded_up_axis=embedded_up,
        embedded_meters_per_unit=embedded_meters,
    )
    return SourceInspection(
        path=path,
        source_format=source_format,
        embedded_up_axis=embedded_up,
        embedded_meters_per_unit=embedded_meters,
        resolved_frame=frame,
        required_inputs=required,
        warnings=warnings,
    )


def _package_member_path(raw: str) -> Path | None:
    """Map a USDZ package-relative path (`pkg.usdz[member]`) to the package file.

    A USDZ package is one self-contained file: its member layers and textures are dependencies of
    the closure only through the package itself, so the package file is the dependency and the
    member must exist inside the archive. Returns ``None`` for an ordinary filesystem path.
    """

    if not Ar.IsPackageRelativePath(raw):
        return None
    outer, member = Ar.SplitPackageRelativePathOuter(raw)
    package = Path(outer).resolve()
    if not package.is_file():
        raise SourceNormalizationError(f"USD dependency is missing: {package}")
    try:
        with zipfile.ZipFile(package) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceNormalizationError(
            f"USD package is not a readable USDZ archive: {package}"
        ) from exc
    if member not in names:
        raise SourceNormalizationError(f"USD dependency is missing: {package}[{member}]")
    return package


def _usd_dependency_paths(path: Path) -> tuple[Path, ...]:
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
    if unresolved:
        raise SourceNormalizationError(
            "USD source has unresolved dependencies: " + ", ".join(map(str, unresolved))
        )
    dependencies: set[Path] = {path.resolve()}
    for layer in layers:
        raw_layer = str(layer.realPath or layer.identifier)
        package = _package_member_path(raw_layer)
        if package is not None:
            dependencies.add(package)
            continue
        candidate = Path(raw_layer).resolve()
        if candidate.is_file():
            dependencies.add(candidate)
    for asset in assets:
        raw = getattr(asset, "resolvedPath", None) or getattr(asset, "path", None) or str(asset)
        package = _package_member_path(str(raw))
        if package is not None:
            dependencies.add(package)
            continue
        candidate = Path(str(raw)).resolve()
        if not candidate.is_file():
            raise SourceNormalizationError(f"USD dependency is missing: {candidate}")
        dependencies.add(candidate)
    return tuple(sorted(dependencies, key=lambda item: item.as_posix()))


def _gltf_dependency_paths(path: Path) -> tuple[Path, ...]:
    if path.suffix.lower() != ".gltf":
        return (path.resolve(),)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceNormalizationError(f"failed to parse glTF JSON: {exc}") from exc
    dependencies: set[Path] = {path.resolve()}
    for group in ("buffers", "images"):
        values = payload.get(group, [])
        if not isinstance(values, list):
            raise SourceNormalizationError(f"glTF {group} must be an array")
        for item in values:
            if not isinstance(item, dict) or not isinstance(item.get("uri"), str):
                continue
            uri = item["uri"]
            if uri.startswith("data:"):
                continue
            if "://" in uri:
                raise SourceNormalizationError("remote glTF dependencies are not supported")
            candidate = (path.parent / uri).resolve()
            if not candidate.is_file():
                raise SourceNormalizationError(f"glTF dependency is missing: {candidate}")
            dependencies.add(candidate)
    return tuple(sorted(dependencies, key=lambda item: item.as_posix()))


def _local_text_dependency(base: Path, value: str, *, label: str) -> Path:
    if "://" in value or value.startswith("data:"):
        raise SourceNormalizationError(f"remote {label} dependencies are not supported")
    candidate = (base / value).resolve()
    if not candidate.is_file():
        raise SourceNormalizationError(f"{label} dependency is missing: {candidate}")
    return candidate


def _text_directive_tokens(path: Path, *, label: str) -> tuple[tuple[str, ...], ...]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise SourceNormalizationError(f"failed to parse {label}: {exc}") from exc
    directives: list[tuple[str, ...]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            tokens = tuple(shlex.split(line, comments=True, posix=True))
        except ValueError as exc:
            raise SourceNormalizationError(
                f"failed to parse {label} line {line_number}: {exc}"
            ) from exc
        if tokens:
            directives.append(tokens)
    return tuple(directives)


_MTL_TEXTURE_DIRECTIVES = frozenset(
    {
        "bump",
        "decal",
        "disp",
        "norm",
        "refl",
        "map_aat",
        "map_bump",
        "map_d",
        "map_ka",
        "map_kd",
        "map_ke",
        "map_ks",
        "map_ns",
        "map_pm",
        "map_pr",
        "map_ps",
    }
)
_MTL_OPTION_ARITY: dict[str, tuple[int, int]] = {
    "-blendu": (1, 1),
    "-blendv": (1, 1),
    "-boost": (1, 1),
    "-bm": (1, 1),
    "-cc": (1, 1),
    "-clamp": (1, 1),
    "-imfchan": (1, 1),
    "-mm": (2, 2),
    "-o": (1, 3),
    "-s": (1, 3),
    "-t": (1, 3),
    "-texres": (1, 1),
    "-type": (1, 1),
}


def _mtl_texture_reference(tokens: tuple[str, ...], *, label: str) -> str:
    values = list(tokens[1:])
    index = 0
    while index < len(values) and values[index].startswith("-"):
        option = values[index].lower()
        arity = _MTL_OPTION_ARITY.get(option)
        if arity is None:
            raise SourceNormalizationError(f"unsupported {label} texture option: {values[index]}")
        index += 1
        minimum, maximum = arity
        if len(values) - index < minimum:
            raise SourceNormalizationError(f"incomplete {label} texture option: {option}")
        index += minimum
        consumed = minimum
        while index < len(values) and consumed < maximum:
            try:
                float(values[index])
            except ValueError:
                break
            index += 1
            consumed += 1
    reference = " ".join(values[index:]).strip()
    if not reference:
        raise SourceNormalizationError(f"{label} texture directive has no file path")
    return reference


def _obj_dependency_paths(path: Path) -> tuple[Path, ...]:
    dependencies: set[Path] = {path.resolve()}
    material_paths: set[Path] = set()
    for tokens in _text_directive_tokens(path, label="OBJ"):
        if tokens[0].lower() != "mtllib":
            continue
        if len(tokens) == 1:
            raise SourceNormalizationError("OBJ mtllib directive has no file path")
        for reference in tokens[1:]:
            material_path = _local_text_dependency(
                path.parent,
                reference,
                label="OBJ material library",
            )
            dependencies.add(material_path)
            material_paths.add(material_path)
    for material_path in sorted(material_paths, key=lambda item: item.as_posix()):
        for tokens in _text_directive_tokens(material_path, label="MTL"):
            if tokens[0].lower() not in _MTL_TEXTURE_DIRECTIVES:
                continue
            reference = _mtl_texture_reference(tokens, label="MTL")
            dependencies.add(
                _local_text_dependency(
                    material_path.parent,
                    reference,
                    label="MTL texture",
                )
            )
    return tuple(sorted(dependencies, key=lambda item: item.as_posix()))


def _source_dependencies(path: Path, source_format: SourceFormat) -> tuple[SourceDependency, ...]:
    if source_format == "usd":
        paths = _usd_dependency_paths(path)
    elif source_format == "gltf":
        paths = _gltf_dependency_paths(path)
    elif source_format == "obj":
        paths = _obj_dependency_paths(path)
    else:
        paths = (path.resolve(),)
    return tuple(
        SourceDependency(candidate, _sha256(candidate), candidate.stat().st_size)
        for candidate in paths
    )


def _require_vertices(values: Any, *, label: str) -> np.ndarray:
    vertices = np.asarray(values, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise SourceNormalizationError(f"{label} must contain a non-empty Nx3 point array")
    if not np.isfinite(vertices).all():
        raise SourceNormalizationError(f"{label} contains a non-finite point")
    return vertices


def _require_affine(transform: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise SourceNormalizationError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-12):
        raise SourceNormalizationError(f"{label} must be affine")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-15:
        raise SourceNormalizationError(f"{label} has a singular linear transform")
    return matrix


def _apply_affine(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    result = vertices @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(result).all():
        raise SourceNormalizationError("transformed source geometry contains a non-finite point")
    return result


def _transform_normals(
    normals: np.ndarray | None,
    *,
    node_transform: np.ndarray,
    frame: ResolvedSourceFrame,
) -> np.ndarray | None:
    if normals is None:
        return None
    values = np.asarray(normals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all():
        return None
    normal_matrix = np.linalg.inv(node_transform[:3, :3]).T
    rotation = frame.matrix[:3, :3] / frame.meters_per_unit
    result = values @ normal_matrix.T @ rotation.T
    lengths = np.linalg.norm(result, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1e-15):
        return None
    return result / lengths[:, None]


def _normalized_node_geometry(
    vertices: Any,
    *,
    node_transform: Any,
    frame: ResolvedSourceFrame,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    source_vertices = _require_vertices(vertices, label=label)
    node_matrix = _require_affine(node_transform, label=f"{label} transform")
    combined = frame.matrix @ node_matrix
    return _apply_affine(source_vertices, combined), node_matrix


def _gf_matrix(transform: Gf.Matrix4d) -> np.ndarray:
    origin = np.asarray(tuple(transform.Transform(Gf.Vec3d(0.0))), dtype=np.float64)
    columns = []
    for axis in (Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(0.0, 1.0, 0.0), Gf.Vec3d(0.0, 0.0, 1.0)):
        columns.append(np.asarray(tuple(transform.Transform(axis)), dtype=np.float64) - origin)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack(columns)
    matrix[:3, 3] = origin
    return matrix


def _triangulate_polygon(
    face: Sequence[int], vertices: np.ndarray, *, label: str
) -> list[list[int]]:
    if len(face) == 3:
        return [[int(value) for value in face]]
    if len(face) < 3:
        raise SourceNormalizationError(f"{label} contains a face with fewer than three corners")
    polygon = vertices[np.asarray(face, dtype=np.int64)]
    # Newell's method remains stable for either winding and selects the best projection plane.
    normal = np.zeros(3, dtype=np.float64)
    for index, point in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        normal += np.asarray(
            (
                (point[1] - following[1]) * (point[2] + following[2]),
                (point[2] - following[2]) * (point[0] + following[0]),
                (point[0] - following[0]) * (point[1] + following[1]),
            )
        )
    normal_length = float(np.linalg.norm(normal))
    extent = float(np.ptp(polygon, axis=0).max())
    if not math.isfinite(normal_length) or normal_length <= max(extent * extent * 1e-14, 1e-30):
        raise SourceNormalizationError(f"{label} contains a degenerate polygon face")
    unit_normal = normal / normal_length
    distances = np.abs((polygon - polygon[0]) @ unit_normal)
    if float(distances.max()) > max(extent * 1e-9, 1e-12):
        raise SourceNormalizationError(
            f"{label} contains a non-planar polygon whose triangulation is ambiguous"
        )
    drop_axis = int(np.argmax(np.abs(unit_normal)))
    projected = np.delete(polygon, drop_axis, axis=1)
    try:
        import manifold3d

        local_faces = np.asarray(manifold3d.triangulate([projected]), dtype=np.int64)
    except Exception as exc:
        raise SourceNormalizationError(f"failed to triangulate {label}: {exc}") from exc
    if local_faces.shape != (len(face) - 2, 3):
        raise SourceNormalizationError(f"failed to triangulate {label} without changing vertices")
    return [[int(face[int(local)]) for local in triangle] for triangle in local_faces]


def _triangulate_usd_mesh(
    mesh: UsdGeom.Mesh,
    vertices: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    counts_raw = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    indices_raw = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if counts_raw is None or indices_raw is None:
        raise SourceNormalizationError(f"{label} is missing face topology")
    counts = [int(value) for value in counts_raw]
    indices = [int(value) for value in indices_raw]
    if any(value < 0 or value >= len(vertices) for value in indices):
        raise SourceNormalizationError(f"{label} has a face vertex index outside its point array")
    holes_raw = mesh.GetHoleIndicesAttr().Get(Usd.TimeCode.Default())
    holes = {int(value) for value in holes_raw or ()}
    triangles: list[list[int]] = []
    cursor = 0
    for face_index, count in enumerate(counts):
        face = indices[cursor : cursor + count]
        if len(face) != count:
            raise SourceNormalizationError(f"{label} has inconsistent topology array lengths")
        cursor += count
        if face_index in holes:
            continue
        triangles.extend(_triangulate_polygon(face, vertices, label=f"{label} face {face_index}"))
    if cursor != len(indices):
        raise SourceNormalizationError(f"{label} has inconsistent topology array lengths")
    if not triangles:
        raise SourceNormalizationError(f"{label} contains no visible triangle faces")
    return np.asarray(triangles, dtype=np.int64)


def _face_components(faces: np.ndarray, welded_faces: np.ndarray) -> tuple[np.ndarray, ...]:
    # Faces sharing one welded edge belong to one component. The partition is computed as sparse
    # connected components instead of Python set flooding; the result is identical: components
    # ordered by their smallest face index, faces ascending inside each component.
    face_count = len(faces)
    if face_count == 0:
        return ()
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    undirected = np.sort(
        np.stack(
            (welded_faces.reshape(-1), welded_faces[:, [1, 2, 0]].reshape(-1)),
            axis=1,
        ),
        axis=1,
    )
    face_ids = np.repeat(np.arange(face_count, dtype=np.int64), 3)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    sorted_edges = undirected[order]
    sorted_faces = face_ids[order]
    new_group = np.empty(len(sorted_edges), dtype=bool)
    new_group[0] = True
    new_group[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    group_ids = np.cumsum(new_group) - 1
    # Link every face in one edge group to the group's first face.
    group_first = np.empty(group_ids[-1] + 1, dtype=np.int64)
    first_positions = np.flatnonzero(new_group)
    group_first[:] = sorted_faces[first_positions]
    rows = group_first[group_ids]
    links = rows != sorted_faces
    graph = sparse.coo_matrix(
        (np.ones(int(links.sum()), dtype=np.int8), (rows[links], sorted_faces[links])),
        shape=(face_count, face_count),
    )
    _, labels = connected_components(graph, directed=False)
    components: list[np.ndarray] = []
    for label in np.unique(labels[np.argsort(labels, kind="stable")]):
        members = np.flatnonzero(labels == label)
        components.append(np.asarray(members, dtype=np.int64))
    components.sort(key=lambda item: int(item[0]))
    return tuple(components)


def _mesh_topology(vertices: np.ndarray, faces: np.ndarray) -> MeshTopology:
    # Vectorized edge/face accounting: the per-face Python loop this replaces dominated source
    # normalization on dense meshes. Every counter keeps the exact semantics of that loop.
    unique_vertices, welded_inverse = np.unique(vertices, axis=0, return_inverse=True)
    welded_faces = welded_inverse[faces]
    extent = float(np.ptp(vertices, axis=0).max())
    area_epsilon = max(extent * extent * 1e-14, 1e-30)
    triangles_all = vertices[faces]
    cross_all = np.cross(
        triangles_all[:, 1] - triangles_all[:, 0],
        triangles_all[:, 2] - triangles_all[:, 0],
    )
    area_twice_all = np.linalg.norm(cross_all, axis=1)
    distinct = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 0] != welded_faces[:, 2])
    )
    usable_mask = distinct & np.isfinite(area_twice_all) & (area_twice_all > area_epsilon)
    usable_count = int(np.count_nonzero(usable_mask))
    degenerate = int(len(faces) - usable_count)
    duplicate = 0
    boundary = 0
    non_manifold = 0
    inconsistent = 0
    if usable_count:
        usable_welded = welded_faces[usable_mask]
        _, key_counts = np.unique(np.sort(usable_welded, axis=1), axis=0, return_counts=True)
        duplicate = int((key_counts - 1)[key_counts > 1].sum())
        directed = np.stack(
            (usable_welded.reshape(-1), usable_welded[:, [1, 2, 0]].reshape(-1)),
            axis=1,
        )
        directed_unique, directed_counts = np.unique(directed, axis=0, return_counts=True)
        _, undirected_inverse = np.unique(
            np.sort(directed_unique, axis=1), axis=0, return_inverse=True
        )
        undirected_total = int(undirected_inverse.max()) + 1
        totals = np.zeros(undirected_total, dtype=np.int64)
        np.add.at(totals, undirected_inverse, directed_counts)
        boundary = int(np.count_nonzero(totals == 1))
        non_manifold = int(np.count_nonzero(totals > 2))
        # An undirected edge with exactly two incidences is inconsistently wound only when both
        # faces traverse it in the same direction, i.e. one directed form appears twice.
        repeated_direction = np.zeros(undirected_total, dtype=np.int64)
        np.add.at(repeated_direction, undirected_inverse[directed_counts == 2], 1)
        inconsistent = int(np.count_nonzero((totals == 2) & (repeated_direction > 0)))
    signed_volume = 0.0
    if usable_count:
        triangles = triangles_all[usable_mask]
        signed_volume = float(
            np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
            / 6.0
        )
    volume_epsilon = max(extent**3 * 1e-12, 1e-45)
    eligible = bool(
        usable_count
        and degenerate == 0
        and duplicate == 0
        and boundary == 0
        and non_manifold == 0
        and inconsistent == 0
        and abs(signed_volume) > volume_epsilon
    )
    if eligible:
        classification: TopologyClass = "closed_orientable_surface"
    elif non_manifold:
        classification = "non_manifold_surface"
    else:
        classification = "open_surface"
    return MeshTopology(
        classification=classification,
        vertex_count=len(vertices),
        triangle_count=len(faces),
        exact_welded_vertex_count=len(vertices) - len(unique_vertices),
        boundary_edge_count=boundary,
        non_manifold_edge_count=non_manifold,
        inconsistent_winding_edge_count=inconsistent,
        degenerate_triangle_count=degenerate,
        duplicate_triangle_count=duplicate,
        oriented_surface_integral_m3=signed_volume,
        surface_closed=eligible,
    )


def _split_mesh_components(
    *,
    provenance: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
    colors: np.ndarray | None,
    render_visual: Any | None = None,
) -> list[NormalizedSourceComponent]:
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise SourceNormalizationError(f"{provenance} must contain a non-empty Mx3 face array")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise SourceNormalizationError(f"{provenance} has a face vertex index out of range")
    _, welded_inverse = np.unique(vertices, axis=0, return_inverse=True)
    groups = _face_components(faces, welded_inverse[faces])
    result: list[NormalizedSourceComponent] = []
    for component_index, face_indices in enumerate(groups):
        selected_faces = faces[face_indices]
        used = np.unique(selected_faces.reshape(-1))
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        local_vertices = vertices[used]
        local_faces = remap[selected_faces]
        local_normals = (
            normals[used] if normals is not None and len(normals) == len(vertices) else None
        )
        local_colors = colors[used] if colors is not None and len(colors) == len(vertices) else None
        local_render_visual = None
        if render_visual is not None:
            try:
                # The source material is immutable throughout preprocessing. face_subset()
                # deep-copies its decoded texture images per connected component, multiplying
                # atlas memory by the component count. Only the per-component UVs need a copy.
                local_render_visual = trimesh.visual.TextureVisuals(
                    uv=render_visual.uv[used] if render_visual.uv is not None else None,
                    material=render_visual.material,
                    face_materials=(
                        render_visual.face_materials.copy()
                        if render_visual.face_materials is not None
                        else None
                    ),
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                local_render_visual = None
        result.append(
            NormalizedSourceComponent(
                component_id="",
                kind="mesh",
                provenance=f"{provenance}#surface-{component_index:04d}",
                vertices=local_vertices,
                faces=local_faces,
                normals=local_normals,
                colors_rgba=local_colors,
                topology=_mesh_topology(local_vertices, local_faces),
                render_visual=local_render_visual,
            )
        )
    return result


def _point_component(
    *,
    provenance: str,
    vertices: np.ndarray,
    normals: np.ndarray | None,
    colors: np.ndarray | None,
) -> NormalizedSourceComponent:
    topology = MeshTopology(
        classification="point_samples",
        vertex_count=len(vertices),
        triangle_count=0,
        exact_welded_vertex_count=len(vertices) - len(np.unique(vertices, axis=0)),
        boundary_edge_count=0,
        non_manifold_edge_count=0,
        inconsistent_winding_edge_count=0,
        degenerate_triangle_count=0,
        duplicate_triangle_count=0,
        oriented_surface_integral_m3=0.0,
        surface_closed=False,
    )
    return NormalizedSourceComponent(
        component_id="",
        kind="points",
        provenance=provenance,
        vertices=vertices,
        faces=None,
        normals=normals if normals is not None and len(normals) == len(vertices) else None,
        colors_rgba=colors if colors is not None and len(colors) == len(vertices) else None,
        topology=topology,
    )


def _usd_vertex_attribute(
    prim: Usd.Prim,
    name: str,
    *,
    expected_count: int,
    dimensions: int,
) -> np.ndarray | None:
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
    if not primvar or not primvar.HasValue():
        return None
    interpolation = str(primvar.GetInterpolation())
    if interpolation not in {
        str(UsdGeom.Tokens.constant),
        str(UsdGeom.Tokens.vertex),
        str(UsdGeom.Tokens.varying),
    }:
        return None
    values_raw = primvar.ComputeFlattened(Usd.TimeCode.Default())
    if values_raw is None:
        return None
    values = np.asarray(
        [tuple(value) if hasattr(value, "__len__") else value for value in values_raw]
    )
    if dimensions == 1:
        values = np.asarray(values, dtype=np.float64).reshape((-1, 1))
    else:
        values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != dimensions or not np.isfinite(values).all():
        return None
    if len(values) == 1:
        return np.repeat(values, expected_count, axis=0)
    return values if len(values) == expected_count else None


def _usd_colors(prim: Usd.Prim, *, expected_count: int) -> np.ndarray | None:
    rgb = _usd_vertex_attribute(prim, "displayColor", expected_count=expected_count, dimensions=3)
    if rgb is None:
        return None
    opacity = _usd_vertex_attribute(
        prim, "displayOpacity", expected_count=expected_count, dimensions=1
    )
    if opacity is None:
        opacity = np.ones((expected_count, 1), dtype=np.float64)
    return np.clip(np.column_stack((rgb, opacity)), 0.0, 1.0)


def _usd_normals(
    point_based: UsdGeom.PointBased,
    *,
    expected_count: int,
    node_transform: np.ndarray,
    frame: ResolvedSourceFrame,
) -> np.ndarray | None:
    interpolation = str(point_based.GetNormalsInterpolation())
    if interpolation not in {str(UsdGeom.Tokens.vertex), str(UsdGeom.Tokens.varying)}:
        return None
    raw = point_based.GetNormalsAttr().Get(Usd.TimeCode.Default())
    if raw is None or len(raw) != expected_count:
        return None
    return _transform_normals(
        np.asarray([tuple(value) for value in raw], dtype=np.float64),
        node_transform=node_transform,
        frame=frame,
    )


def _load_usd_components(
    path: Path, frame: ResolvedSourceFrame
) -> tuple[list[NormalizedSourceComponent], list[str]]:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise SourceNormalizationError(f"failed to open USD stage: {path}")
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    accepted_purposes = {str(UsdGeom.Tokens.default_), str(UsdGeom.Tokens.render)}
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    components: list[NormalizedSourceComponent] = []
    warnings: list[str] = []
    for prim in stage.Traverse(predicate):
        if not prim.IsA(UsdGeom.Imageable):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if str(imageable.ComputePurpose()) not in accepted_purposes:
            continue
        if prim.IsA(UsdGeom.PointInstancer):
            raise UnsupportedSourceGeometryError(
                f"visible PointInstancer {prim.GetPath()} is not expanded in Phase 1"
            )
        if not prim.IsA(UsdGeom.Gprim):
            continue
        provenance = str(prim.GetPath())
        node_transform = _require_affine(
            _gf_matrix(xforms.GetLocalToWorldTransform(prim)),
            label=f"{provenance} local-to-world transform",
        )
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            subdivision = str(mesh.GetSubdivisionSchemeAttr().Get() or "catmullClark")
            if subdivision != str(UsdGeom.Tokens.none):
                raise UnsupportedSourceGeometryError(
                    f"{provenance} uses unsupported subdivision scheme {subdivision!r}"
                )
            points_raw = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
            if points_raw is None:
                raise SourceNormalizationError(f"{provenance} is missing points")
            vertices, node_transform = _normalized_node_geometry(
                np.asarray(points_raw, dtype=np.float64),
                node_transform=node_transform,
                frame=frame,
                label=provenance,
            )
            faces = _triangulate_usd_mesh(mesh, vertices, label=provenance)
            orientation = str(mesh.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded)
            if orientation not in {
                str(UsdGeom.Tokens.rightHanded),
                str(UsdGeom.Tokens.leftHanded),
            }:
                raise SourceNormalizationError(
                    f"{provenance} has unsupported orientation token {orientation!r}"
                )
            flip_winding = orientation == str(UsdGeom.Tokens.leftHanded)
            if float(np.linalg.det(node_transform[:3, :3])) < 0.0:
                flip_winding = not flip_winding
            if flip_winding:
                faces = faces[:, (0, 2, 1)]
            normals = _usd_normals(
                mesh,
                expected_count=len(vertices),
                node_transform=node_transform,
                frame=frame,
            )
            colors = _usd_colors(prim, expected_count=len(vertices))
            components.extend(
                _split_mesh_components(
                    provenance=provenance,
                    vertices=vertices,
                    faces=faces,
                    normals=normals,
                    colors=colors,
                )
            )
        elif prim.IsA(UsdGeom.Points):
            points = UsdGeom.Points(prim)
            points_raw = points.GetPointsAttr().Get(Usd.TimeCode.Default())
            if points_raw is None:
                raise SourceNormalizationError(f"{provenance} is missing points")
            vertices, node_transform = _normalized_node_geometry(
                np.asarray(points_raw, dtype=np.float64),
                node_transform=node_transform,
                frame=frame,
                label=provenance,
            )
            normals = _usd_normals(
                points,
                expected_count=len(vertices),
                node_transform=node_transform,
                frame=frame,
            )
            components.append(
                _point_component(
                    provenance=provenance,
                    vertices=vertices,
                    normals=normals,
                    colors=_usd_colors(prim, expected_count=len(vertices)),
                )
            )
        else:
            raise UnsupportedSourceGeometryError(
                f"unsupported visible USD geometry {provenance} ({prim.GetTypeName()})"
            )
    if not components:
        raise SourceNormalizationError(
            "USD stage contains no visible default/render Mesh or Points"
        )
    return components, warnings


def _trimesh_vertex_colors(geometry: trimesh.Trimesh) -> np.ndarray | None:
    if getattr(geometry.visual, "kind", None) != "vertex":
        return None
    colors = np.asarray(cast("trimesh.visual.ColorVisuals", geometry.visual).vertex_colors)
    if colors.ndim != 2 or len(colors) != len(geometry.vertices) or colors.shape[1] not in {3, 4}:
        return None
    values = colors.astype(np.float64)
    if np.issubdtype(colors.dtype, np.integer) or float(values.max(initial=0.0)) > 1.0:
        values /= 255.0
    if values.shape[1] == 3:
        values = np.column_stack((values, np.ones(len(values))))
    return np.clip(values, 0.0, 1.0)


def _trimesh_render_visual(geometry: trimesh.Trimesh) -> Any | None:
    """Return validated preview-only native texture/material appearance when available."""

    visual = getattr(geometry, "visual", None)
    if getattr(visual, "kind", None) != "texture" or getattr(visual, "material", None) is None:
        return None
    uv = getattr(visual, "uv", None)
    if uv is not None:
        values = np.asarray(uv)
        if values.ndim != 2 or len(values) != len(geometry.vertices) or values.shape[1] < 2:
            return None
        if not np.isfinite(values[:, :2]).all():
            return None
    return visual


def _point_colors(geometry: trimesh.PointCloud) -> np.ndarray | None:
    colors = np.asarray(geometry.colors)
    if colors.ndim != 2 or len(colors) != len(geometry.vertices) or colors.shape[1] not in {3, 4}:
        return None
    values = colors.astype(np.float64)
    if np.issubdtype(colors.dtype, np.integer) or float(values.max(initial=0.0)) > 1.0:
        values /= 255.0
    if values.shape[1] == 3:
        values = np.column_stack((values, np.ones(len(values))))
    return np.clip(values, 0.0, 1.0)


def _trimesh_authored_normals(geometry: trimesh.Trimesh) -> np.ndarray | None:
    cached = getattr(getattr(geometry, "_cache", None), "cache", {}).get("vertex_normals")
    if cached is None:
        return None
    values = np.asarray(cached, dtype=np.float64)
    return values if values.shape == (len(geometry.vertices), 3) else None


def _ply_point_normals(geometry: trimesh.PointCloud) -> np.ndarray | None:
    try:
        data = geometry.metadata["_ply_raw"]["vertex"]["data"]
        values = np.column_stack([data[name] for name in ("nx", "ny", "nz")]).astype(np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    return values if values.shape == (len(geometry.vertices), 3) else None


def _load_trimesh_components(
    path: Path, frame: ResolvedSourceFrame
) -> tuple[list[NormalizedSourceComponent], list[str]]:
    try:
        loaded = trimesh.load(path, force="scene", process=False, allow_remote=False)
    except Exception as exc:
        raise SourceNormalizationError(
            f"failed to load {path.name}: {type(exc).__name__}: {exc}"
        ) from exc
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    nodes = sorted(scene.graph.nodes_geometry, key=str)
    if not nodes:
        raise SourceNormalizationError(f"{path.name} contains no mesh or point geometry")
    components: list[NormalizedSourceComponent] = []
    warnings: list[str] = []
    for node_name in nodes:
        transform_raw, geometry_name = scene.graph.get(node_name)
        node_transform = _require_affine(transform_raw, label=f"scene node {node_name!r} transform")
        geometry = scene.geometry[geometry_name]
        # geometry_name leads: trimesh appends a random disambiguation suffix to
        # node_name when two scene-graph nodes share a base name (e.g. duplicate
        # hinge/hardware nodes), so node_name is not stable across reloads of the
        # same file. component ordering below sorts by provenance, so keeping the
        # stable geometry_name first keeps source-identity hashing deterministic.
        provenance = f"geometry:{geometry_name}/node:{node_name}"
        if isinstance(geometry, trimesh.Trimesh):
            vertices, node_transform = _normalized_node_geometry(
                geometry.vertices,
                node_transform=node_transform,
                frame=frame,
                label=provenance,
            )
            faces = np.asarray(geometry.faces, dtype=np.int64)
            if faces.ndim != 2 or faces.shape[1:] != (3,):
                raise UnsupportedSourceGeometryError(
                    f"{provenance} was not triangulated by the source loader"
                )
            if float(np.linalg.det(node_transform[:3, :3])) < 0.0:
                faces = faces[:, (0, 2, 1)]
            normals = _transform_normals(
                _trimesh_authored_normals(geometry),
                node_transform=node_transform,
                frame=frame,
            )
            render_visual = (
                _trimesh_render_visual(geometry)
                if path.suffix.lower() in {".glb", ".gltf"}
                else None
            )
            components.extend(
                _split_mesh_components(
                    provenance=provenance,
                    vertices=vertices,
                    faces=faces,
                    normals=normals,
                    colors=_trimesh_vertex_colors(geometry),
                    render_visual=render_visual,
                )
            )
        elif isinstance(geometry, trimesh.PointCloud):
            vertices, node_transform = _normalized_node_geometry(
                geometry.vertices,
                node_transform=node_transform,
                frame=frame,
                label=provenance,
            )
            components.append(
                _point_component(
                    provenance=provenance,
                    vertices=vertices,
                    normals=_transform_normals(
                        _ply_point_normals(geometry),
                        node_transform=node_transform,
                        frame=frame,
                    ),
                    colors=_point_colors(geometry),
                )
            )
        else:
            raise UnsupportedSourceGeometryError(
                f"unsupported scene geometry {provenance} ({type(geometry).__name__})"
            )
    return components, warnings


def _component_sort_key(component: NormalizedSourceComponent) -> tuple[Any, ...]:
    return (
        component.provenance,
        component.kind,
        tuple(float(value) for value in component.bounds_min_xyz),
        tuple(float(value) for value in component.bounds_max_xyz),
        component.topology.triangle_count,
        component.topology.vertex_count,
    )


def load_normalized_source(source: SourceInput) -> NormalizedSourceScene:
    """Load geometry into metres and Z-up without recentering, repair, or semantic alignment."""

    inspection = inspect_source(source)
    if not inspection.ready or inspection.resolved_frame is None:
        missing = ", ".join(inspection.required_inputs)
        raise SourceMetadataError(
            f"{source.path.name} requires explicit frame metadata before normalization: {missing}"
        )
    frame = inspection.resolved_frame
    if inspection.source_format == "usd":
        components, loader_warnings = _load_usd_components(source.path, frame)
    else:
        components, loader_warnings = _load_trimesh_components(source.path, frame)
    ordered = sorted(components, key=_component_sort_key)
    identified = tuple(
        replace(component, component_id=f"c{index:06d}") for index, component in enumerate(ordered)
    )
    all_vertices = np.concatenate([component.vertices for component in identified], axis=0)
    bounds_min = all_vertices.min(axis=0)
    bounds_max = all_vertices.max(axis=0)
    if not np.isfinite(bounds_min).all() or not np.isfinite(bounds_max).all():
        raise SourceNormalizationError("normalized source bounds are non-finite")
    if float((bounds_max - bounds_min).max()) <= 0.0:
        raise SourceNormalizationError("normalized source must have a positive spatial extent")
    warnings = list(inspection.warnings)
    warnings.extend(loader_warnings)
    if frame.front_axis is None:
        warnings.append(
            "front_axis is unresolved; normalization preserves deterministic horizontal yaw"
        )
    warnings.append(
        "all reconstruction geometry is surface-only evidence; every enclosed interior remains "
        "unknown even when surface topology is closed"
    )
    return NormalizedSourceScene(
        source_path=source.path,
        source_format=inspection.source_format,
        frame=frame,
        components=identified,
        bounds_min_xyz=bounds_min,
        bounds_max_xyz=bounds_max,
        source_dependencies=_source_dependencies(source.path, inspection.source_format),
        warnings=tuple(warnings),
    )


def _author_colors(prim: Usd.Prim, colors: np.ndarray | None) -> None:
    if colors is None:
        return
    primvars = UsdGeom.PrimvarsAPI(prim)
    display_color = primvars.CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    )
    display_color.Set(
        [Gf.Vec3f(float(value[0]), float(value[1]), float(value[2])) for value in colors]
    )
    display_opacity = primvars.CreatePrimvar(
        "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
    )
    display_opacity.Set([float(value[3]) for value in colors])


def _write_normalized_usdc(scene: NormalizedSourceScene, destination: Path) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Source")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Source/Meshes")
    UsdGeom.Scope.Define(stage, "/Source/Points")
    for component in scene.components:
        points = [Gf.Vec3f(*(float(value) for value in vertex)) for vertex in component.vertices]
        if component.kind == "mesh":
            assert component.faces is not None
            mesh = UsdGeom.Mesh.Define(stage, f"/Source/Meshes/{component.component_id}")
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([3] * len(component.faces))
            mesh.CreateFaceVertexIndicesAttr([int(value) for value in component.faces.reshape(-1)])
            if component.normals is not None:
                mesh.CreateNormalsAttr(
                    [Gf.Vec3f(*(float(value) for value in normal)) for normal in component.normals]
                )
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            _author_colors(mesh.GetPrim(), component.colors_rgba)
        else:
            point_set = UsdGeom.Points.Define(stage, f"/Source/Points/{component.component_id}")
            point_set.CreatePointsAttr(points)
            point_set.CreateIdsAttr(list(range(len(component.vertices))))
            if component.normals is not None:
                point_set.CreateNormalsAttr(
                    [Gf.Vec3f(*(float(value) for value in normal)) for normal in component.normals]
                )
                point_set.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            _author_colors(point_set.GetPrim(), component.colors_rgba)
    if not stage.GetRootLayer().Export(str(destination)):
        raise SourceNormalizationError(f"failed to export normalized USD: {destination}")


def component_data_sha256(component: NormalizedSourceComponent) -> str:
    """Return the stable authored-data identity of one normalized surface component."""

    digest = hashlib.sha256()
    digest.update(np.asarray(component.vertices, dtype="<f8").tobytes(order="C"))
    if component.faces is not None:
        digest.update(np.asarray(component.faces, dtype="<i8").tobytes(order="C"))
    if component.normals is not None:
        digest.update(np.asarray(component.normals, dtype="<f8").tobytes(order="C"))
    if component.colors_rgba is not None:
        digest.update(np.asarray(component.colors_rgba, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _manifest_payload(
    scene: NormalizedSourceScene, *, normalized_usdc_sha256: str
) -> dict[str, Any]:
    source_parent = scene.source_path.parent
    dependencies = []
    for dependency in scene.source_dependencies:
        try:
            logical_path = dependency.path.relative_to(source_parent).as_posix()
        except ValueError:
            logical_path = os.path.relpath(dependency.path, source_parent)
        dependencies.append(
            {
                "path": logical_path,
                "sha256": dependency.sha256,
                "byte_count": dependency.byte_count,
            }
        )
    component_payloads = []
    for component in scene.components:
        topology = component.topology
        component_payloads.append(
            {
                "component_id": component.component_id,
                "kind": component.kind,
                "provenance": component.provenance,
                "bounds_min_xyz_m": _vector(component.bounds_min_xyz),
                "bounds_max_xyz_m": _vector(component.bounds_max_xyz),
                "data_sha256": component_data_sha256(component),
                "normals": "authored" if component.normals is not None else "absent",
                "colors": "authored" if component.colors_rgba is not None else "absent",
                "render_appearance": (
                    "native_texture_or_material"
                    if component.render_visual is not None
                    else "vertex_color"
                    if component.colors_rgba is not None
                    else "neutral_fallback"
                ),
                "topology": {
                    "classification": topology.classification,
                    "vertex_count": topology.vertex_count,
                    "triangle_count": topology.triangle_count,
                    "exact_welded_vertex_count": topology.exact_welded_vertex_count,
                    "boundary_edge_count": topology.boundary_edge_count,
                    "non_manifold_edge_count": topology.non_manifold_edge_count,
                    "inconsistent_winding_edge_count": topology.inconsistent_winding_edge_count,
                    "degenerate_triangle_count": topology.degenerate_triangle_count,
                    "duplicate_triangle_count": topology.duplicate_triangle_count,
                    "oriented_surface_integral_m3": topology.oriented_surface_integral_m3,
                    "surface_closed": topology.surface_closed,
                    "geometry_authority": component.geometry_authority,
                    "interior_semantics": component.interior_semantics,
                    "solid_authority": component.solid_authority,
                },
            }
        )
    kinds = {component.kind for component in scene.components}
    geometry_kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
    return {
        "schema": "reconstruction_normalized_source",
        "schema_version": 2,
        "source": {
            "entry_name": scene.source_path.name,
            "format": scene.source_format,
            "dependencies": dependencies,
        },
        "frame_resolution": {
            "up_axis": {
                "value": scene.frame.up_axis,
                "source": scene.frame.up_axis_source,
            },
            "meters_per_unit": {
                "value": scene.frame.meters_per_unit,
                "source": scene.frame.meters_per_unit_source,
            },
            "front_axis": {
                "value": scene.frame.front_axis,
                "source": scene.frame.front_axis_source,
            },
        },
        "target_frame": {
            "up_axis": _CANONICAL_UP,
            "front_axis": _CANONICAL_FRONT,
            "meters_per_unit": 1.0,
        },
        "source_to_normalized": {
            "convention": "row-major matrix; applied to column vectors",
            "matrix4": [list(row) for row in scene.frame.source_to_normalized],
            "recentered": False,
            "ground_snapped": False,
            "pca_aligned": False,
        },
        "geometry": {
            "kind": geometry_kind,
            "authority": "surface_only",
            "interior_semantics": "unknown",
            "solid_authority": "none",
            "provenance_semantics": "component_lineage_only_not_solid_authority",
            "bounds_min_xyz_m": _vector(scene.bounds_min_xyz),
            "bounds_max_xyz_m": _vector(scene.bounds_max_xyz),
            "extent_xyz_m": _vector(scene.bounds_max_xyz - scene.bounds_min_xyz),
            "component_count": len(scene.components),
            "mesh_component_count": sum(component.kind == "mesh" for component in scene.components),
            "point_component_count": sum(
                component.kind == "points" for component in scene.components
            ),
            "components": component_payloads,
        },
        "normalized_artifact": {
            "path": "normalized.usdc",
            "sha256": normalized_usdc_sha256,
            "up_axis": _CANONICAL_UP,
            "meters_per_unit": 1.0,
        },
        "warnings": list(scene.warnings),
    }


def write_normalized_source(
    scene: NormalizedSourceScene, output_dir: Path
) -> NormalizedSourceArtifacts:
    """Write one deterministic geometry snapshot without creating a canonical USDCraft record."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    normalized_usdc = root / "normalized.usdc"
    source_manifest = root / "source_manifest.json"
    temporary_usdc = root / f".normalized-{os.getpid()}.tmp.usdc"
    temporary_manifest = root / f".source-manifest-{os.getpid()}.tmp.json"
    try:
        _write_normalized_usdc(scene, temporary_usdc)
        opened = Usd.Stage.Open(str(temporary_usdc))
        if opened is None:
            raise SourceNormalizationError("normalized USD could not be reopened")
        if str(UsdGeom.GetStageUpAxis(opened)) != str(UsdGeom.Tokens.z):
            raise SourceNormalizationError("normalized USD did not retain Z-up metadata")
        if not math.isclose(
            float(UsdGeom.GetStageMetersPerUnit(opened)), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise SourceNormalizationError("normalized USD did not retain metre units")
        normalized_hash = _sha256(temporary_usdc)
        payload = _manifest_payload(scene, normalized_usdc_sha256=normalized_hash)
        temporary_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_hash = _sha256(temporary_manifest)
        os.replace(temporary_usdc, normalized_usdc)
        os.replace(temporary_manifest, source_manifest)
    finally:
        temporary_usdc.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return NormalizedSourceArtifacts(
        root=root,
        normalized_usdc=normalized_usdc,
        source_manifest=source_manifest,
        normalized_usdc_sha256=normalized_hash,
        source_manifest_sha256=manifest_hash,
    )


__all__ = [
    "MeshTopology",
    "NormalizedSourceArtifacts",
    "NormalizedSourceComponent",
    "NormalizedSourceScene",
    "ResolvedSourceFrame",
    "SignedAxis",
    "SourceDependency",
    "SourceInput",
    "SourceInspection",
    "SourceMetadataError",
    "SourceNormalizationError",
    "UnsupportedSourceGeometryError",
    "component_data_sha256",
    "inspect_source",
    "load_normalized_source",
    "write_normalized_source",
]
