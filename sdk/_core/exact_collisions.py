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

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .assets import (
    AssetContext,
    AssetContextLike,
    coerce_asset_context,
    require_mesh_filename,
    resolve_asset_root,
)
from .errors import ValidationError
from .types import Box, Collision, Cylinder, Material, Mesh, Origin, Part, Sphere, Visual

if TYPE_CHECKING:
    from .articulated_object import ArticulatedObject


_COMPILED_MODEL_CACHE_ATTR = "_sdk_exact_collision_model_cache"
_ALLOW_EXPLICIT_COLLISIONS_ATTR = "_sdk_allow_explicit_collisions"
_CACHE_VERSION = 12


def compile_object_model_with_exact_collisions(
    object_model: "ArticulatedObject",
    *,
    asset_root: Optional[AssetContextLike] = None,
    validate: bool = True,
) -> "ArticulatedObject":
    if validate:
        object_model.validate(strict=True)
    assets = _resolve_compile_assets(object_model, asset_root=asset_root)
    cache_key = _compiled_model_cache_key(object_model, assets=assets)
    cached_models = getattr(object_model, _COMPILED_MODEL_CACHE_ATTR, None)
    if isinstance(cached_models, dict):
        cached_model = cached_models.get(cache_key)
        if cached_model is not None:
            return cached_model

    from .articulated_object import ArticulatedObject

    compiled = ArticulatedObject(
        object_model.name,
        meta=copy.deepcopy(object_model.meta),
        assets=assets,
    )
    setattr(compiled, _ALLOW_EXPLICIT_COLLISIONS_ATTR, True)
    compiled.materials = [copy.deepcopy(material) for material in object_model.materials]
    compiled.physics_materials = [
        copy.deepcopy(material) for material in object_model.physics_materials
    ]
    compiled.base_mode = object_model.base_mode
    compiled.articulations = [
        copy.deepcopy(articulation) for articulation in object_model.articulations
    ]

    for source_part in object_model.parts:
        compiled_part = Part(
            name=source_part.name,
            visuals=[copy.deepcopy(visual) for visual in source_part.visuals],
            collisions=[],
            inertial=copy.deepcopy(source_part.inertial),
            rigid_body=copy.deepcopy(source_part.rigid_body),
            body_model=source_part.body_model,
            meta=copy.deepcopy(source_part.meta),
            assets=coerce_asset_context(getattr(source_part, "assets", None)) or assets,
        )
        compiled_part.collisions = _generate_part_collisions(
            source_part,
            assets=compiled_part.assets or assets,
        )
        compiled.parts.append(compiled_part)

    if not isinstance(cached_models, dict):
        cached_models = {}
        setattr(object_model, _COMPILED_MODEL_CACHE_ATTR, cached_models)
    cached_models[cache_key] = compiled
    return compiled


def _resolve_compile_assets(
    object_model: "ArticulatedObject",
    *,
    asset_root: Optional[AssetContextLike],
) -> AssetContext:
    resolved = coerce_asset_context(asset_root)
    if resolved is not None:
        return resolved

    model_assets = coerce_asset_context(getattr(object_model, "assets", None))
    if model_assets is not None:
        return model_assets

    root = resolve_asset_root(asset_root, object_model)
    if root is not None:
        return AssetContext(root)

    return AssetContext(Path(".").resolve())


def _generate_part_collisions(
    part: Part,
    *,
    assets: AssetContext,
) -> list[Collision]:
    collisions: list[Collision] = []
    for visual in part.visuals:
        if not getattr(visual, "collidable", True):
            continue  # visual-only surfaces (decals/labels) have no collision
        owner = f"part {part.name!r}"
        if visual.name:
            owner += f" visual {visual.name!r}"
        collisions.extend(_collisions_from_visual(visual, assets=assets, owner=owner))
    return collisions


def _collisions_from_visual(
    visual: Visual,
    *,
    assets: AssetContext,
    owner: str | None = None,
) -> list[Collision]:
    geometry = visual.geometry
    if isinstance(geometry, (Box, Cylinder, Sphere)):
        return [
            Collision(
                geometry=copy.deepcopy(geometry),
                origin=copy.deepcopy(visual.origin),
                name=visual.name,
            )
        ]

    if not isinstance(geometry, Mesh):
        raise ValidationError(
            f"Unsupported visual geometry for collision derivation: {type(geometry).__name__}"
        )

    mesh_path = assets.mesh_path(
        require_mesh_filename(geometry, owner=owner), ensure_dir=False
    ).resolve()
    if mesh_path.suffix.lower() != ".obj":
        raise ValidationError(
            f"Exact collision derivation only supports OBJ meshes (got: {mesh_path.name!r})"
        )
    if not mesh_path.exists():
        raise ValidationError(f"Mesh file not found: {mesh_path}")

    return [
        Collision(
            geometry=copy.deepcopy(geometry),
            origin=copy.deepcopy(visual.origin),
            name=visual.name,
        )
    ]


def _compiled_model_cache_key(
    object_model: "ArticulatedObject",
    *,
    assets: AssetContext,
) -> str:
    payload = {
        "version": _CACHE_VERSION,
        "asset_root": assets.asset_root.as_posix(),
        "model": _serialize_object_model(object_model),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_object_model(object_model: "ArticulatedObject") -> dict[str, object]:
    return {
        "name": object_model.name,
        "meta": object_model.meta,
        "assets": _serialize_assets(getattr(object_model, "assets", None)),
        "materials": [
            _serialize_material(material) for material in getattr(object_model, "materials", [])
        ],
        "physics_materials": [
            {
                "name": material.name,
                "static_friction": material.static_friction,
                "dynamic_friction": material.dynamic_friction,
                "restitution": material.restitution,
                "density": material.density,
            }
            for material in getattr(object_model, "physics_materials", [])
        ],
        "base_mode": str(getattr(object_model, "base_mode", "floating")),
        "parts": [
            {
                "name": part.name,
                "visuals": [
                    {
                        "geometry": _serialize_geometry(visual.geometry),
                        "origin": {
                            "xyz": tuple(visual.origin.xyz),
                            "rpy": tuple(visual.origin.rpy),
                        },
                        "material": _serialize_material_ref(visual.material),
                        "name": visual.name,
                        "collidable": visual.collidable,
                    }
                    for visual in part.visuals
                ],
                "inertial": _serialize_inertial(part.inertial),
                "rigid_body": repr(getattr(part, "rigid_body", None)),
                "body_model": getattr(
                    getattr(part, "body_model", None),
                    "value",
                    str(getattr(part, "body_model", "rigid")),
                ),
                "meta": part.meta,
                "assets": _serialize_assets(getattr(part, "assets", None)),
            }
            for part in getattr(object_model, "parts", [])
        ],
        "articulations": [
            {
                "name": articulation.name,
                "type": str(articulation.articulation_type),
                "parent": articulation.parent,
                "child": articulation.child,
                "origin": {
                    "xyz": tuple(articulation.origin.xyz),
                    "rpy": tuple(articulation.origin.rpy),
                },
                "axis": tuple(articulation.axis),
                "cone_deg": getattr(articulation, "cone_deg", None),
                "motion_limits": (
                    None
                    if articulation.motion_limits is None
                    else {
                        "effort": articulation.motion_limits.effort,
                        "velocity": articulation.motion_limits.velocity,
                        "lower": articulation.motion_limits.lower,
                        "upper": articulation.motion_limits.upper,
                    }
                ),
                "motion_properties": (
                    None
                    if articulation.motion_properties is None
                    else {
                        "damping": articulation.motion_properties.damping,
                        "friction": articulation.motion_properties.friction,
                    }
                ),
                "drive": repr(getattr(articulation, "drive", None)),
                "mimic": (
                    None
                    if getattr(articulation, "mimic", None) is None
                    else {
                        "joint": articulation.mimic.joint,
                        "multiplier": articulation.mimic.multiplier,
                        "offset": articulation.mimic.offset,
                    }
                ),
                "meta": articulation.meta,
            }
            for articulation in getattr(object_model, "articulations", [])
        ],
    }


def _serialize_geometry(geometry: Box | Cylinder | Sphere | Mesh) -> dict[str, object]:
    if isinstance(geometry, Box):
        return {"type": "box", "size": tuple(geometry.size)}
    if isinstance(geometry, Cylinder):
        return {"type": "cylinder", "radius": geometry.radius, "length": geometry.length}
    if isinstance(geometry, Sphere):
        return {"type": "sphere", "radius": geometry.radius}
    return {
        "type": "mesh",
        "filename": None if geometry.filename is None else os.fspath(geometry.filename),
        "scale": None if geometry.scale is None else tuple(geometry.scale),
        "source_geometry": (
            None
            if geometry.source_geometry is None
            else _serialize_geometry(geometry.source_geometry)
        ),
        "source_transform": (
            None
            if geometry.source_transform is None
            else tuple(tuple(row) for row in geometry.source_transform)
        ),
        "normal_crease_deg": geometry.normal_crease_deg,
    }


def _serialize_material_ref(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Material):
        return _serialize_material(value)
    if hasattr(value, "name"):
        name = getattr(value, "name", None)
        if isinstance(name, str):
            return name
    return str(value)


def _serialize_material(material: Material) -> dict[str, object]:
    return {
        "name": material.name,
        "rgba": material.rgba,
        "texture": material.texture,
        "metallic": material.metallic,
        "roughness": material.roughness,
        "emissive": material.emissive,
        "transmission": material.transmission,
        "ior": material.ior,
        "clearcoat": material.clearcoat,
        "clearcoat_roughness": material.clearcoat_roughness,
        "base_color_texture": _serialize_texture_input(material.base_color_texture),
        "metallic_texture": _serialize_texture_input(material.metallic_texture),
        "roughness_texture": _serialize_texture_input(material.roughness_texture),
        "normal_texture": _serialize_texture_input(material.normal_texture),
        "occlusion_texture": _serialize_texture_input(material.occlusion_texture),
        "emissive_texture": _serialize_texture_input(material.emissive_texture),
        "opacity_texture": _serialize_texture_input(material.opacity_texture),
    }


def _serialize_texture_input(texture: object) -> object:
    if texture is None:
        return None
    return {
        "file": os.fspath(getattr(texture, "file")),
        "uv_set": getattr(texture, "uv_set"),
        "wrap_s": getattr(texture, "wrap_s"),
        "wrap_t": getattr(texture, "wrap_t"),
        "source_color_space": getattr(texture, "source_color_space"),
        "channel": getattr(texture, "channel"),
    }


def _serialize_inertial(inertial: object) -> object:
    if inertial is None:
        return None
    inertia = getattr(inertial, "inertia", None)
    origin = getattr(inertial, "origin", Origin())
    return {
        "mass": getattr(inertial, "mass", None),
        "origin": {
            "xyz": tuple(origin.xyz),
            "rpy": tuple(origin.rpy),
        },
        "inertia": None
        if inertia is None
        else {
            "ixx": inertia.ixx,
            "ixy": inertia.ixy,
            "ixz": inertia.ixz,
            "iyy": inertia.iyy,
            "iyz": inertia.iyz,
            "izz": inertia.izz,
        },
    }


def _serialize_assets(assets: Any) -> object:
    ctx = coerce_asset_context(assets)
    if ctx is None:
        return None
    return {
        "root": ctx.asset_root.as_posix(),
        "mesh_subdir": ctx.mesh_subdir,
    }


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    return repr(value)
