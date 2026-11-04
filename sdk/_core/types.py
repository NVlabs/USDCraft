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

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

from .assets import AssetContext, coerce_asset_context
from .errors import ValidationError

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Vec5 = Tuple[float, float, float, float, float]
Mat4 = Tuple[Tuple[float, float, float, float], ...]


def _as_finite_float(value: float, *, name: str) -> float:
    """Coerce one serialized SDK scalar while rejecting NaN and infinity."""

    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise ValidationError(f"{name} must be finite")
    return normalized


def _coerce_normal_crease_deg(value: float | None, *, name: str) -> float | None:
    """Validate an optional mesh shading crease in the public degrees contract.

    ``None`` deliberately means "use the exporter default" so existing models
    retain the established 60-degree behavior without serializing a new value.
    Zero produces hard face normals and 180 degrees smooths every connected
    face corner.  This is a visual policy only; it does not alter topology or
    collision geometry.
    """

    if value is None:
        return None
    normalized = _as_finite_float(value, name=name)
    if normalized < 0.0 or normalized > 180.0:
        raise ValidationError(f"{name} must be between 0 and 180 degrees")
    return normalized


def _as_vec3(values: Sequence[float], *, name: str) -> Vec3:
    if len(values) != 3:
        raise ValidationError(f"{name} must have 3 elements")
    return (
        _as_finite_float(values[0], name=f"{name}[0]"),
        _as_finite_float(values[1], name=f"{name}[1]"),
        _as_finite_float(values[2], name=f"{name}[2]"),
    )


def _as_vec2(values: Sequence[float], *, name: str) -> Vec2:
    if len(values) != 2:
        raise ValidationError(f"{name} must have 2 elements")
    return (
        _as_finite_float(values[0], name=f"{name}[0]"),
        _as_finite_float(values[1], name=f"{name}[1]"),
    )


def _as_vec5(values: Sequence[float], *, name: str) -> Vec5:
    if len(values) != 5:
        raise ValidationError(f"{name} must have 5 elements")
    return (
        _as_finite_float(values[0], name=f"{name}[0]"),
        _as_finite_float(values[1], name=f"{name}[1]"),
        _as_finite_float(values[2], name=f"{name}[2]"),
        _as_finite_float(values[3], name=f"{name}[3]"),
        _as_finite_float(values[4], name=f"{name}[4]"),
    )


def _as_mat4(values: Sequence[Sequence[float]], *, name: str) -> Mat4:
    if len(values) != 4:
        raise ValidationError(f"{name} must have 4 rows")
    rows: list[tuple[float, float, float, float]] = []
    for idx, row in enumerate(values):
        if len(row) != 4:
            raise ValidationError(f"{name}[{idx}] must have 4 elements")
        rows.append(
            (
                _as_finite_float(row[0], name=f"{name}[{idx}][0]"),
                _as_finite_float(row[1], name=f"{name}[{idx}][1]"),
                _as_finite_float(row[2], name=f"{name}[{idx}][2]"),
                _as_finite_float(row[3], name=f"{name}[{idx}][3]"),
            )
        )
    return (rows[0], rows[1], rows[2], rows[3])


@dataclass(frozen=True)
class Origin:
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz", _as_vec3(self.xyz, name="origin.xyz"))
        object.__setattr__(self, "rpy", _as_vec3(self.rpy, name="origin.rpy"))


@dataclass(frozen=True)
class Box:
    size: Vec3

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", _as_vec3(self.size, name="box.size"))


@dataclass(frozen=True, init=False)
class Cylinder:
    radius: float
    length: float

    def __init__(
        self,
        radius: float,
        length: float | None = None,
        *,
        height: float | None = None,
    ) -> None:
        if length is not None and height is not None:
            raise TypeError("Cylinder() accepts only one of 'length' or alias 'height'")
        resolved_length = height if length is None else length
        if resolved_length is None:
            raise TypeError("Cylinder() missing required argument: 'length'")
        object.__setattr__(self, "radius", _as_finite_float(radius, name="cylinder.radius"))
        object.__setattr__(
            self,
            "length",
            _as_finite_float(resolved_length, name="cylinder.length"),
        )

    @property
    def height(self) -> float:
        return self.length


@dataclass(frozen=True)
class Sphere:
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _as_finite_float(self.radius, name="sphere.radius"))


PrimitiveGeometry = Union[Box, Cylinder, Sphere]


@dataclass(frozen=True)
class Mesh:
    filename: Union[str, os.PathLike[str], None] = None
    name: Optional[str] = None
    scale: Optional[Vec3] = None
    source_geometry: Optional[PrimitiveGeometry] = None
    source_transform: Optional[Mat4] = None
    materialized_path: Optional[str] = field(default=None, compare=False, repr=False)
    normal_crease_deg: Optional[float] = None

    def __post_init__(self) -> None:
        filename_value = self.filename
        filename = os.fspath(filename_value) if filename_value is not None else ""
        logical_name = None if self.name is None else str(self.name).strip()
        if not filename and not logical_name:
            raise ValidationError("mesh.filename or mesh.name is required")
        if filename:
            object.__setattr__(self, "filename", filename)
        else:
            object.__setattr__(self, "filename", None)
        if logical_name is not None:
            if not logical_name:
                raise ValidationError("mesh.name must be non-empty when provided")
            object.__setattr__(self, "name", logical_name)
        if self.scale is not None:
            object.__setattr__(self, "scale", _as_vec3(self.scale, name="mesh.scale"))
        if self.source_geometry is not None and not isinstance(
            self.source_geometry, (Box, Cylinder, Sphere)
        ):
            raise ValidationError("mesh.source_geometry must be a Box, Cylinder, Sphere, or None")
        if self.source_transform is not None:
            if self.source_geometry is None:
                raise ValidationError("mesh.source_transform requires mesh.source_geometry")
            object.__setattr__(
                self,
                "source_transform",
                _as_mat4(self.source_transform, name="mesh.source_transform"),
            )
        if self.materialized_path is not None:
            object.__setattr__(self, "materialized_path", os.fspath(self.materialized_path))
        object.__setattr__(
            self,
            "normal_crease_deg",
            _coerce_normal_crease_deg(
                self.normal_crease_deg,
                name="mesh.normal_crease_deg",
            ),
        )


Geometry = Union[Box, Cylinder, Sphere, Mesh]


def estimate_mass_from_density(
    geometry: PrimitiveGeometry,
    density: float,
    fill_fraction: float = 1.0,
) -> float:
    """Estimate primitive mass from its envelope volume and material density.

    ``fill_fraction`` is the fraction of the primitive envelope occupied by the
    material.  It is a deliberately coarse authoring aid for hollow or thin
    parts, not a mesh-volume or center-of-mass solver.
    """

    normalized_density = float(density)
    if not math.isfinite(normalized_density):
        raise ValidationError("density must be finite")
    if normalized_density <= 0.0:
        raise ValidationError("density must be positive")

    normalized_fill = float(fill_fraction)
    if not math.isfinite(normalized_fill):
        raise ValidationError("fill_fraction must be finite")
    if normalized_fill <= 0.0 or normalized_fill > 1.0:
        raise ValidationError("fill_fraction must be greater than 0 and at most 1")

    if isinstance(geometry, Box):
        dimensions = _as_vec3(geometry.size, name="box.size")
        for axis, value in zip(("x", "y", "z"), dimensions, strict=True):
            if not math.isfinite(value):
                raise ValidationError(f"box.size.{axis} must be finite")
            if value <= 0.0:
                raise ValidationError(f"box.size.{axis} must be positive")
        volume = dimensions[0] * dimensions[1] * dimensions[2]
    elif isinstance(geometry, Cylinder):
        radius = float(geometry.radius)
        length = float(geometry.length)
        for field_name, value in (("radius", radius), ("length", length)):
            if not math.isfinite(value):
                raise ValidationError(f"cylinder.{field_name} must be finite")
            if value <= 0.0:
                raise ValidationError(f"cylinder.{field_name} must be positive")
        volume = math.pi * radius * radius * length
    elif isinstance(geometry, Sphere):
        radius = float(geometry.radius)
        if not math.isfinite(radius):
            raise ValidationError("sphere.radius must be finite")
        if radius <= 0.0:
            raise ValidationError("sphere.radius must be positive")
        volume = (4.0 / 3.0) * math.pi * radius * radius * radius
    elif isinstance(geometry, Mesh):
        raise ValidationError("Cannot estimate mass from mesh geometry")
    else:
        raise ValidationError("Mass estimation supports only Box, Cylinder, and Sphere geometry")

    mass = normalized_density * volume * normalized_fill
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValidationError("estimated mass must be finite and positive")
    return mass


# Keep this portable default explicit.  UsdPreviewSurface falls back to 0.01
# when clearcoatRoughness is unauthored, while the Workbench uses 0.2.
# Normalizing it into Material makes every serializer consume the
# same value instead of inheriting renderer-specific fallbacks.
DEFAULT_CLEARCOAT_ROUGHNESS = 0.2


def _normalize_material_rgba(
    rgba: Optional[Sequence[float]],
    *,
    field_name: str = "material.rgba",
) -> Optional[Tuple[float, float, float, float]]:
    if rgba is None:
        return None
    normalized = tuple(float(v) for v in rgba)
    if len(normalized) not in (3, 4):
        raise ValidationError(f"{field_name} must have 3 or 4 values")
    if any(not math.isfinite(value) for value in normalized):
        raise ValidationError(f"{field_name} values must be finite")
    if len(normalized) == 3:
        normalized = normalized + (1.0,)
    return cast(Tuple[float, float, float, float], normalized)


def _normalize_unit_scalar(
    value: Optional[float],
    *,
    field_name: str,
) -> Optional[float]:
    if value is None:
        return None
    normalized = _as_finite_float(value, name=field_name)
    if not 0.0 <= normalized <= 1.0:
        raise ValidationError(f"{field_name} must be between 0 and 1")
    return normalized


def _normalize_emissive_rgb(
    value: Optional[Sequence[float]],
    *,
    field_name: str = "material.emissive",
) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    normalized = tuple(float(v) for v in value)
    if len(normalized) != 3:
        raise ValidationError(f"{field_name} must have 3 values")
    if any(not math.isfinite(component) for component in normalized):
        raise ValidationError(f"{field_name} values must be finite")
    if any(v < 0.0 for v in normalized):
        raise ValidationError(f"{field_name} values must be >= 0")
    return normalized


def _normalize_ior(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    normalized = _as_finite_float(value, name="material.ior")
    if normalized < 1.0:
        raise ValidationError("material.ior must be >= 1")
    return normalized


@dataclass(frozen=True)
class TextureInput:
    """One texture input in a portable UsdPreviewSurface network."""

    file: str
    uv_set: str = "st"
    wrap_s: str = "repeat"
    wrap_t: str = "repeat"
    source_color_space: str = "auto"
    channel: str = "rgb"

    def __post_init__(self) -> None:
        file = os.fspath(self.file).strip()
        if not file:
            raise ValidationError("texture_input.file is required")
        object.__setattr__(self, "file", file)
        uv_set = str(self.uv_set).strip()
        if not uv_set:
            raise ValidationError("texture_input.uv_set is required")
        object.__setattr__(self, "uv_set", uv_set)
        for field_name in ("wrap_s", "wrap_t"):
            value = str(getattr(self, field_name)).strip()
            if value not in {"repeat", "mirror", "clamp", "black"}:
                raise ValidationError(
                    f"texture_input.{field_name} must be repeat, mirror, clamp, or black"
                )
            object.__setattr__(self, field_name, value)
        color_space = str(self.source_color_space).strip()
        if color_space not in {"auto", "sRGB", "raw"}:
            raise ValidationError("texture_input.source_color_space must be auto, sRGB, or raw")
        object.__setattr__(self, "source_color_space", color_space)
        channel = str(self.channel).strip()
        if channel not in {"r", "g", "b", "a", "rgb"}:
            raise ValidationError("texture_input.channel must be r, g, b, a, or rgb")
        object.__setattr__(self, "channel", channel)


TextureInputRef = Union[TextureInput, str]


@dataclass(init=False)
class Material:
    """Visual material. PBR field names follow UsdPreviewSurface conventions.

    ``rgba``/``color`` RGB channels are display-referred sRGB values.  The
    alpha channel and scalar PBR fields are linear.  ``emissive`` is linear RGB
    so values above one can represent emission intensity.
    """

    name: str
    rgba: Optional[Tuple[float, float, float, float]] = None
    texture: Optional[str] = None
    metallic: Optional[float] = None
    roughness: Optional[float] = None
    emissive: Optional[Tuple[float, float, float]] = None
    transmission: Optional[float] = None
    ior: Optional[float] = None
    clearcoat: Optional[float] = None
    clearcoat_roughness: Optional[float] = None
    base_color_texture: Optional[TextureInput] = None
    metallic_texture: Optional[TextureInput] = None
    roughness_texture: Optional[TextureInput] = None
    normal_texture: Optional[TextureInput] = None
    occlusion_texture: Optional[TextureInput] = None
    emissive_texture: Optional[TextureInput] = None
    opacity_texture: Optional[TextureInput] = None

    def __init__(
        self,
        name: str,
        rgba: Optional[Sequence[float]] = None,
        texture: Optional[str] = None,
        *,
        color: Optional[Sequence[float]] = None,
        metallic: Optional[float] = None,
        roughness: Optional[float] = None,
        emissive: Optional[Sequence[float]] = None,
        transmission: Optional[float] = None,
        ior: Optional[float] = None,
        clearcoat: Optional[float] = None,
        clearcoat_roughness: Optional[float] = None,
        base_color_texture: Optional[TextureInputRef] = None,
        metallic_texture: Optional[TextureInputRef] = None,
        roughness_texture: Optional[TextureInputRef] = None,
        normal_texture: Optional[TextureInputRef] = None,
        occlusion_texture: Optional[TextureInputRef] = None,
        emissive_texture: Optional[TextureInputRef] = None,
        opacity_texture: Optional[TextureInputRef] = None,
    ) -> None:
        if rgba is not None and color is not None:
            raise ValidationError("Material cannot set both rgba and color")
        self.name = str(name)
        self.rgba = _normalize_material_rgba(rgba if rgba is not None else color)
        self.texture = texture
        self.metallic = metallic
        self.roughness = roughness
        # Stored raw here and normalized to an RGB triple by ``__post_init__``.
        self.emissive = cast(Optional[Tuple[float, float, float]], emissive)
        self.transmission = transmission
        self.ior = ior
        self.clearcoat = clearcoat
        self.clearcoat_roughness = clearcoat_roughness
        self.base_color_texture = _coerce_texture_input(
            base_color_texture or texture, default_color_space="sRGB"
        )
        self.metallic_texture = _coerce_texture_input(
            metallic_texture, default_color_space="raw", default_channel="b"
        )
        self.roughness_texture = _coerce_texture_input(
            roughness_texture, default_color_space="raw", default_channel="g"
        )
        self.normal_texture = _coerce_texture_input(normal_texture, default_color_space="raw")
        self.occlusion_texture = _coerce_texture_input(
            occlusion_texture, default_color_space="raw", default_channel="r"
        )
        self.emissive_texture = _coerce_texture_input(emissive_texture, default_color_space="sRGB")
        self.opacity_texture = _coerce_texture_input(
            opacity_texture, default_color_space="raw", default_channel="a"
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValidationError("material.name is required")
        self.rgba = _normalize_material_rgba(self.rgba)
        self.metallic = _normalize_unit_scalar(self.metallic, field_name="material.metallic")
        self.roughness = _normalize_unit_scalar(self.roughness, field_name="material.roughness")
        self.emissive = _normalize_emissive_rgb(self.emissive)
        self.transmission = _normalize_unit_scalar(
            self.transmission, field_name="material.transmission"
        )
        self.ior = _normalize_ior(self.ior)
        self.clearcoat = _normalize_unit_scalar(self.clearcoat, field_name="material.clearcoat")
        self.clearcoat_roughness = _normalize_unit_scalar(
            self.clearcoat_roughness, field_name="material.clearcoat_roughness"
        )
        if self.clearcoat is not None and self.clearcoat > 0.0 and self.clearcoat_roughness is None:
            self.clearcoat_roughness = DEFAULT_CLEARCOAT_ROUGHNESS

    def has_pbr(self) -> bool:
        return any(
            value is not None
            for value in (
                self.metallic,
                self.roughness,
                self.emissive,
                self.transmission,
                self.ior,
                self.clearcoat,
                self.clearcoat_roughness,
            )
        )


def _coerce_texture_input(
    value: Optional[TextureInputRef],
    *,
    default_color_space: str,
    default_channel: str = "rgb",
) -> Optional[TextureInput]:
    if value is None:
        return None
    if isinstance(value, TextureInput):
        return value
    return TextureInput(
        file=os.fspath(value),
        source_color_space=default_color_space,
        channel=default_channel,
    )


MaterialRef = Union[Material, str]


def _coerce_visual_material_ref(
    *,
    material: Optional[MaterialRef],
    color: object,
) -> Optional[MaterialRef]:
    if material is not None and color is not None:
        raise TypeError("Part.visual() accepts only one of 'material' or alias 'color'")
    if color is None:
        return material
    if isinstance(color, (Material, str)):
        return color
    if isinstance(color, Sequence) and not isinstance(color, (str, bytes)):
        return Material(name="inline_color", rgba=tuple(float(v) for v in color))
    raise TypeError(
        "Part.visual() alias 'color' must be a material name, Material, or 3/4-value color tuple"
    )


@dataclass
class Visual:
    geometry: Geometry
    origin: Origin = field(default_factory=Origin)
    material: Optional[MaterialRef] = None
    name: Optional[str] = None
    # Visual-only surfaces (decals/labels) set collidable=False so collision
    # derivation skips them — a printed label has no physical presence.
    collidable: bool = True


class BaseMode(str, Enum):
    """How an articulated asset root participates in a simulation world."""

    FLOATING = "floating"
    FIXED = "fixed"
    KINEMATIC = "kinematic"


class BodyModel(str, Enum):
    """Physical interpretation of a part's authored geometry."""

    RIGID = "rigid"
    FLEXIBLE_1D = "flexible_1d"
    SURFACE_DEFORMABLE = "surface_deformable"
    VOLUME_DEFORMABLE = "volume_deformable"

    @property
    def realization(self) -> str:
        """Current export realization for this body model."""

        return "native" if self is BodyModel.RIGID else "static_rest_pose"


def _coerce_body_model(value: Union["BodyModel", str]) -> BodyModel:
    try:
        return value if isinstance(value, BodyModel) else BodyModel(str(value))
    except ValueError as exc:
        raise ValidationError(f"Unknown body model: {value}") from exc


class CollisionApproximation(str, Enum):
    """Portable UsdPhysics mesh-collision approximation tokens."""

    NONE = "none"
    CONVEX_DECOMPOSITION = "convexDecomposition"
    CONVEX_HULL = "convexHull"
    BOUNDING_SPHERE = "boundingSphere"
    BOUNDING_CUBE = "boundingCube"
    MESH_SIMPLIFICATION = "meshSimplification"


@dataclass(frozen=True)
class PhysicsMaterial:
    """Surface properties for collision response, separate from visual PBR."""

    name: str
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0
    density: Optional[float] = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValidationError("physics_material.name is required")
        object.__setattr__(self, "name", name)
        for field_name in ("static_friction", "dynamic_friction", "restitution"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValidationError(f"physics_material.{field_name} must be finite")
            if value < 0.0:
                raise ValidationError(f"physics_material.{field_name} must be >= 0")
            object.__setattr__(self, field_name, value)
        if self.restitution > 1.0:
            raise ValidationError("physics_material.restitution must be between 0 and 1")
        if self.dynamic_friction > self.static_friction:
            raise ValidationError("physics_material.dynamic_friction must be <= static_friction")
        if self.density is not None:
            density = float(self.density)
            if not math.isfinite(density):
                raise ValidationError("physics_material.density must be finite")
            if density <= 0.0:
                raise ValidationError("physics_material.density must be positive")
            object.__setattr__(self, "density", density)


@dataclass(frozen=True)
class CollisionProperties:
    physics_material: Optional[str] = None
    approximation: CollisionApproximation | str = CollisionApproximation.CONVEX_DECOMPOSITION

    def __post_init__(self) -> None:
        if self.physics_material is not None:
            name = str(self.physics_material).strip()
            if not name:
                raise ValidationError("collision.physics_material must be non-empty")
            object.__setattr__(self, "physics_material", name)
        try:
            approximation = (
                self.approximation
                if isinstance(self.approximation, CollisionApproximation)
                else CollisionApproximation(str(self.approximation))
            )
        except ValueError as exc:
            raise ValidationError(
                f"Unknown collision approximation: {self.approximation!r}"
            ) from exc
        object.__setattr__(self, "approximation", approximation)


@dataclass(frozen=True)
class RigidBodyProperties:
    enabled: bool = True
    kinematic: bool = False
    physics_material: Optional[str] = None
    collision_approximation: CollisionApproximation | str = (
        CollisionApproximation.CONVEX_DECOMPOSITION
    )

    def __post_init__(self) -> None:
        if self.physics_material is not None:
            name = str(self.physics_material).strip()
            if not name:
                raise ValidationError("rigid_body.physics_material must be non-empty")
            object.__setattr__(self, "physics_material", name)
        try:
            approximation = (
                self.collision_approximation
                if isinstance(self.collision_approximation, CollisionApproximation)
                else CollisionApproximation(str(self.collision_approximation))
            )
        except ValueError as exc:
            raise ValidationError(
                f"Unknown collision approximation: {self.collision_approximation!r}"
            ) from exc
        object.__setattr__(self, "collision_approximation", approximation)


@dataclass
class Collision:
    geometry: Geometry
    origin: Origin = field(default_factory=Origin)
    name: Optional[str] = None
    physics: Optional[CollisionProperties] = None


@dataclass(frozen=True)
class Inertia:
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float

    def __post_init__(self) -> None:
        for field_name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            object.__setattr__(
                self,
                field_name,
                _as_finite_float(getattr(self, field_name), name=f"inertia.{field_name}"),
            )


@dataclass
class Inertial:
    mass: float
    inertia: Inertia
    origin: Origin = field(default_factory=Origin)

    def __post_init__(self) -> None:
        self.mass = _as_finite_float(self.mass, name="inertial.mass")
        if not isinstance(self.inertia, Inertia):
            raise ValidationError("inertial.inertia must be an Inertia")
        if not isinstance(self.origin, Origin):
            raise ValidationError("inertial.origin must be an Origin")

    @staticmethod
    def from_geometry(
        geometry: Geometry, mass: float, *, origin: Optional[Origin] = None
    ) -> "Inertial":
        mass = _as_finite_float(mass, name="inertial.mass")
        if mass <= 0:
            raise ValidationError("Mass must be positive")

        if isinstance(geometry, Box):
            x, y, z = _as_vec3(geometry.size, name="box.size")
            ixx = (1.0 / 12.0) * mass * (y * y + z * z)
            iyy = (1.0 / 12.0) * mass * (x * x + z * z)
            izz = (1.0 / 12.0) * mass * (x * x + y * y)
        elif isinstance(geometry, Cylinder):
            r = float(geometry.radius)
            length = float(geometry.length)
            ixx = (1.0 / 12.0) * mass * (3 * r * r + length * length)
            iyy = ixx
            izz = 0.5 * mass * r * r
        elif isinstance(geometry, Sphere):
            r = float(geometry.radius)
            ixx = (2.0 / 5.0) * mass * r * r
            iyy = ixx
            izz = ixx
        else:
            raise ValidationError("Cannot derive inertia for mesh geometry")

        inertia = Inertia(ixx=ixx, ixy=0.0, ixz=0.0, iyy=iyy, iyz=0.0, izz=izz)
        return Inertial(mass=mass, inertia=inertia, origin=origin or Origin())


class ArticulationType(str, Enum):
    """Joint kind of an ``Articulation`` (a ``str`` enum; ``FLOATING`` emits no USD joint)."""

    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    SPHERICAL = "spherical"
    FIXED = "fixed"
    FLOATING = "floating"


def _coerce_articulation_type(
    value: Union["ArticulationType", str],
) -> ArticulationType:
    try:
        return value if isinstance(value, ArticulationType) else ArticulationType(str(value))
    except ValueError as exc:
        raise ValidationError(f"Unknown articulation type: {value}") from exc


@dataclass(frozen=True)
class MotionLimits:
    effort: float = 1.0
    velocity: float = 1.0
    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self) -> None:
        for field_name in ("effort", "velocity", "lower", "upper"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValidationError(f"motion_limits.{field_name} must be finite")
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True)
class MotionProperties:
    damping: Optional[float] = None
    friction: Optional[float] = None

    def __post_init__(self) -> None:
        for field_name in ("damping", "friction"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValidationError(f"motion_properties.{field_name} must be finite")
            if normalized < 0.0:
                raise ValidationError(f"motion_properties.{field_name} must be >= 0")
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True)
class NewtonJointSolverTuning:
    """Explicit MuJoCo dry-friction solver parameters for one scalar joint.

    ``solref_friction`` maps to the two-value ``mjc:solreffriction`` vector.
    ``solimp_friction`` maps to the five-value ``mjc:solimpfriction`` vector.
    These parameters are independent of PhysX joint resistance and are never
    inferred from :meth:`PhysxJointFriction.stay_put`.
    """

    solref_friction: Optional[Vec2] = None
    solimp_friction: Optional[Vec5] = None

    def __post_init__(self) -> None:
        if self.solref_friction is None and self.solimp_friction is None:
            raise ValidationError(
                "newton_joint_solver_tuning must set solref_friction or solimp_friction"
            )

        if self.solref_friction is not None:
            solref = _as_vec2(
                self.solref_friction,
                name="newton_joint_solver_tuning.solref_friction",
            )
            standard_format = solref[0] > 0.0 and solref[1] > 0.0
            direct_format = solref[0] <= 0.0 and solref[1] <= 0.0
            if not (standard_format or direct_format):
                raise ValidationError(
                    "newton_joint_solver_tuning.solref_friction must use either two positive "
                    "standard-format values or two non-positive direct-format values"
                )
            object.__setattr__(self, "solref_friction", solref)

        if self.solimp_friction is not None:
            solimp = _as_vec5(
                self.solimp_friction,
                name="newton_joint_solver_tuning.solimp_friction",
            )
            for index in (0, 1, 3):
                if not 0.0 <= solimp[index] <= 1.0:
                    raise ValidationError(
                        f"newton_joint_solver_tuning.solimp_friction[{index}] must be between "
                        "0 and 1"
                    )
            if solimp[2] <= 0.0:
                raise ValidationError(
                    "newton_joint_solver_tuning.solimp_friction[2] width must be > 0"
                )
            if solimp[4] < 1.0:
                raise ValidationError(
                    "newton_joint_solver_tuning.solimp_friction[4] power must be >= 1"
                )
            object.__setattr__(self, "solimp_friction", solimp)


@dataclass(frozen=True)
class PassiveJointSpring:
    """Passive restoring spring for one scalar joint.

    Angular joints use N*m/rad for ``stiffness`` and radians for ``reference``.
    Prismatic joints use N/m and meters. Passive viscous damping remains the
    separate ``MotionProperties.damping`` authoring contract.
    """

    stiffness: float
    reference: float = 0.0

    def __post_init__(self) -> None:
        stiffness = _as_finite_float(
            self.stiffness,
            name="passive_joint_spring.stiffness",
        )
        if stiffness <= 0.0:
            raise ValidationError("passive_joint_spring.stiffness must be > 0")
        reference = _as_finite_float(
            self.reference,
            name="passive_joint_spring.reference",
        )
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "reference", reference)


@dataclass(frozen=True)
class PhysxJointFriction:
    """Effort-based PhysX articulation friction for one scalar joint axis.

    Angular values use the SDK's SI/radian contract: efforts are N*m and the
    viscous coefficient is N*m*s/rad. Linear values are N and N*s/m.
    """

    static_friction_effort: float
    dynamic_friction_effort: float
    viscous_friction_coefficient: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "static_friction_effort",
            "dynamic_friction_effort",
            "viscous_friction_coefficient",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValidationError(f"physx_joint_friction.{field_name} must be finite")
            if value < 0.0:
                raise ValidationError(f"physx_joint_friction.{field_name} must be >= 0")
            object.__setattr__(self, field_name, value)
        if self.dynamic_friction_effort > self.static_friction_effort:
            raise ValidationError(
                "physx_joint_friction.dynamic_friction_effort must be <= static_friction_effort"
            )
        if (
            self.static_friction_effort == 0.0
            and self.dynamic_friction_effort == 0.0
            and self.viscous_friction_coefficient == 0.0
        ):
            raise ValidationError(
                "physx_joint_friction must author at least one non-zero friction property; "
                "use PhysxJointFrictionCoefficient(0.0) for an explicitly frictionless joint"
            )

    @classmethod
    def from_breakaway_effort(
        cls,
        static_effort: float,
        *,
        dynamic_ratio: float = 0.8,
        viscous_friction_coefficient: float = 0.0,
    ) -> "PhysxJointFriction":
        """Build a simple Coulomb model from a desired break-away effort."""

        static_effort = float(static_effort)
        dynamic_ratio = float(dynamic_ratio)
        if not math.isfinite(dynamic_ratio) or not 0.0 <= dynamic_ratio <= 1.0:
            raise ValidationError("dynamic_ratio must be finite and between 0 and 1")
        return cls(
            static_friction_effort=static_effort,
            dynamic_friction_effort=static_effort * dynamic_ratio,
            viscous_friction_coefficient=viscous_friction_coefficient,
        )

    @classmethod
    def stay_put(
        cls,
        max_passive_effort: float,
        *,
        dynamic_margin: float = 1.1,
        static_margin: float = 1.1,
        viscous_friction_coefficient: float = 0.0,
    ) -> "PhysxJointFriction":
        """Build a conservative stay-put model from an estimated passive load."""

        max_passive_effort = float(max_passive_effort)
        dynamic_margin = float(dynamic_margin)
        static_margin = float(static_margin)
        for name, value in (
            ("max_passive_effort", max_passive_effort),
            ("dynamic_margin", dynamic_margin),
            ("static_margin", static_margin),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValidationError(f"{name} must be positive and finite")
        dynamic_effort = max_passive_effort * dynamic_margin
        return cls(
            static_friction_effort=dynamic_effort * static_margin,
            dynamic_friction_effort=dynamic_effort,
            viscous_friction_coefficient=viscous_friction_coefficient,
        )


@dataclass(frozen=True)
class PhysxJointFrictionCoefficient:
    """Load-proportional PhysX articulation friction coefficient."""

    coefficient: float

    def __post_init__(self) -> None:
        coefficient = float(self.coefficient)
        if not math.isfinite(coefficient):
            raise ValidationError("physx_joint_friction_coefficient.coefficient must be finite")
        if coefficient < 0.0:
            raise ValidationError("physx_joint_friction_coefficient.coefficient must be >= 0")
        object.__setattr__(self, "coefficient", coefficient)


#: PhysX articulation joint resistance accepted by ``physx_joint_resistance``: either the
#: effort-based :class:`PhysxJointFriction` or the load-proportional
#: :class:`PhysxJointFrictionCoefficient`.
PhysxJointResistance = Union[PhysxJointFriction, PhysxJointFrictionCoefficient]


@dataclass(frozen=True)
class JointDrive:
    """Portable UsdPhysics drive parameters for a scalar articulation."""

    mode: str = "force"
    target_position: Optional[float] = None
    target_velocity: Optional[float] = None
    stiffness: float = 0.0
    damping: float = 0.0
    max_force: Optional[float] = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"force", "acceleration"}:
            raise ValidationError("joint_drive.mode must be 'force' or 'acceleration'")
        object.__setattr__(self, "mode", mode)
        for field_name in ("target_position", "target_velocity", "max_force"):
            value = getattr(self, field_name)
            if value is not None:
                normalized = float(value)
                if not math.isfinite(normalized):
                    raise ValidationError(f"joint_drive.{field_name} must be finite")
                if field_name == "max_force" and normalized < 0.0:
                    raise ValidationError("joint_drive.max_force must be >= 0")
                object.__setattr__(self, field_name, normalized)
        for field_name in ("stiffness", "damping"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValidationError(f"joint_drive.{field_name} must be finite")
            if value < 0.0:
                raise ValidationError(f"joint_drive.{field_name} must be >= 0")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class Mimic:
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        joint_name = str(self.joint).strip()
        if not joint_name:
            raise ValidationError("mimic.joint is required")
        object.__setattr__(self, "joint", joint_name)
        object.__setattr__(
            self,
            "multiplier",
            _as_finite_float(self.multiplier, name="mimic.multiplier"),
        )
        object.__setattr__(
            self,
            "offset",
            _as_finite_float(self.offset, name="mimic.offset"),
        )


@dataclass
class Part:
    name: str
    visuals: List[Visual] = field(default_factory=list)
    collisions: List[Collision] = field(default_factory=list)
    inertial: Optional[Inertial] = None
    rigid_body: Optional[RigidBodyProperties] = None
    body_model: Union[BodyModel, str] = BodyModel.RIGID
    meta: Dict[str, object] = field(default_factory=dict)
    assets: Optional[AssetContext] = None

    def __post_init__(self) -> None:
        self.body_model = _coerce_body_model(self.body_model)
        self.assets = coerce_asset_context(self.assets)

    @property
    def body_realization(self) -> str:
        return cast(BodyModel, self.body_model).realization

    def visual(
        self,
        geometry: Geometry,
        *,
        origin: Optional[Origin] = None,
        material: Optional[MaterialRef] = None,
        color: object = None,
        name: Optional[str] = None,
        collidable: bool = True,
    ) -> Visual:
        visual = Visual(
            geometry=geometry,
            origin=origin or Origin(),
            material=_coerce_visual_material_ref(material=material, color=color),
            name=name,
            collidable=collidable,
        )
        self.visuals.append(visual)
        return visual

    def decal(
        self,
        canvas: object,
        *,
        width: float,
        height: float,
        origin: Optional[Origin] = None,
        curvature_radius: float = 0.0,
        name: Optional[str] = None,
        proud_offset: float = 0.0005,
    ) -> Visual:
        """Attach a surface graphic (label/logo/printed text) as a textured
        decal. The graphic is authored in code via a TextureCanvas; this is the
        correct way to reproduce printed markings instead of modeling them as
        geometry. The decal is visual-only (never a collision shape)."""
        from ._mesh.decals import build_decal_visual_mesh

        mesh = build_decal_visual_mesh(
            canvas,
            width=width,
            height=height,
            curvature_radius=curvature_radius,
            proud_offset=proud_offset,
            assets=self.assets,
        )
        return self.visual(mesh, origin=origin, name=name, collidable=False)

    def get_visual(self, name: str) -> Visual:
        key = str(name).strip()
        if not key:
            raise ValidationError("visual name is required")
        for visual in self.visuals:
            if visual.name == key:
                return visual
        raise ValidationError(f"Unknown visual on part {self.name!r}: {name!r}")


@dataclass(eq=False)
class Articulation:
    name: str
    articulation_type: Union[ArticulationType, str]
    parent: str
    child: str
    origin: Origin = field(default_factory=Origin)
    axis: Vec3 = (0.0, 0.0, 1.0)
    cone_deg: Optional[float] = field(default=None, kw_only=True)
    motion_limits: Optional[MotionLimits] = None
    motion_properties: Optional[MotionProperties] = None
    physx_joint_resistance: Optional[PhysxJointResistance] = None
    physx_armature: Optional[float] = None
    drive: Optional[JointDrive] = None
    mimic: Optional[Mimic] = None
    meta: Dict[str, object] = field(default_factory=dict)
    newton_joint_solver_tuning: Optional[NewtonJointSolverTuning] = field(
        default=None,
        kw_only=True,
    )
    passive_spring: Optional[PassiveJointSpring] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        self.articulation_type = _coerce_articulation_type(self.articulation_type)
        self.axis = _as_vec3(self.axis, name="articulation.axis")
        if self.cone_deg is not None:
            self.cone_deg = _as_finite_float(self.cone_deg, name="articulation.cone_deg")
        if self.physx_armature is not None:
            self.physx_armature = float(self.physx_armature)
            if not math.isfinite(self.physx_armature) or self.physx_armature < 0.0:
                raise ValidationError("articulation.physx_armature must be finite and >= 0")
        if self.mimic is not None and not isinstance(self.mimic, Mimic):
            if isinstance(self.mimic, str):
                self.mimic = Mimic(joint=self.mimic)
            else:
                joint = getattr(self.mimic, "joint", None)
                if joint is None:
                    raise ValidationError("articulation.mimic must be a Mimic or mimic-like object")
                self.mimic = Mimic(
                    joint=joint,
                    multiplier=getattr(self.mimic, "multiplier", 1.0),
                    offset=getattr(self.mimic, "offset", 0.0),
                )

    @property
    def joint_type(self) -> ArticulationType:
        return cast(ArticulationType, self.articulation_type)

    @joint_type.setter
    def joint_type(self, value: Union[ArticulationType, str]) -> None:
        self.articulation_type = _coerce_articulation_type(value)

    @property
    def limit(self) -> Optional[MotionLimits]:
        return self.motion_limits

    @limit.setter
    def limit(self, value: Optional[MotionLimits]) -> None:
        self.motion_limits = value

    @property
    def dynamics(self) -> Optional[MotionProperties]:
        return self.motion_properties

    @dynamics.setter
    def dynamics(self, value: Optional[MotionProperties]) -> None:
        self.motion_properties = value
