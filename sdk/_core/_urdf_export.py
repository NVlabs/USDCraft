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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from ._mesh.normals import DEFAULT_NORMAL_CREASE_DEGREES
from .articulated_object import ArticulatedObject
from .assets import require_mesh_filename
from .errors import ValidationError
from .exact_collisions import compile_object_model_with_exact_collisions
from .types import (
    Articulation,
    ArticulationType,
    BodyModel,
    Box,
    Collision,
    Cylinder,
    Geometry,
    Inertial,
    Material,
    Mesh,
    Origin,
    Part,
    Sphere,
    Visual,
)


def compile_object_to_urdf_xml(
    object_model: ArticulatedObject,
    *,
    pretty: bool = True,
    asset_root: str | os.PathLike[str] | Path | None = None,
    include_physical_collisions: bool = True,
    validate: bool = True,
) -> str:
    if validate:
        object_model.validate(strict=True)
    compiled_model = (
        compile_object_model_with_exact_collisions(
            object_model,
            asset_root=asset_root,
            validate=False,
        )
        if include_physical_collisions
        else object_model
    )
    root = ET.Element("robot", {"name": object_model.name})

    for material in compiled_model.materials:
        root.append(_material_element(material))

    for part in compiled_model.parts:
        root.append(_part_element(part))

    for articulation in compiled_model.articulations:
        root.append(_articulation_element(articulation))

    if pretty:
        _indent(root)

    return ET.tostring(root, encoding="unicode")


def _format_float(value: float) -> str:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"URDF numeric value must be numeric and finite: {value!r}") from exc
    if not math.isfinite(normalized):
        raise ValidationError(f"URDF numeric value must be finite: {value!r}")
    return f"{normalized:.6g}"


def _format_vec(values: Sequence[float]) -> str:
    return " ".join(_format_float(v) for v in values)


def _indent(elem: ET.Element, level: int = 0) -> None:
    indent_str = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_str
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent_str


def _maybe_origin(parent: ET.Element, origin: Origin) -> None:
    if tuple(origin.xyz) == (0.0, 0.0, 0.0) and tuple(origin.rpy) == (0.0, 0.0, 0.0):
        return
    ET.SubElement(
        parent,
        "origin",
        {"xyz": _format_vec(origin.xyz), "rpy": _format_vec(origin.rpy)},
    )


def _geometry_element(parent: ET.Element, geometry: Geometry) -> None:
    geom = ET.SubElement(parent, "geometry")
    if isinstance(geometry, Box):
        ET.SubElement(geom, "box", {"size": _format_vec(geometry.size)})
    elif isinstance(geometry, Cylinder):
        ET.SubElement(
            geom,
            "cylinder",
            {
                "radius": _format_float(geometry.radius),
                "length": _format_float(geometry.length),
            },
        )
    elif isinstance(geometry, Sphere):
        ET.SubElement(geom, "sphere", {"radius": _format_float(geometry.radius)})
    elif isinstance(geometry, Mesh):
        crease = (
            DEFAULT_NORMAL_CREASE_DEGREES
            if geometry.normal_crease_deg is None
            else geometry.normal_crease_deg
        )
        attrs = {
            "filename": require_mesh_filename(geometry),
            "usdcraft_normal_crease_deg": _format_float(crease),
        }
        if geometry.scale:
            attrs["scale"] = _format_vec(geometry.scale)
        ET.SubElement(geom, "mesh", attrs)
    else:
        raise ValidationError("Unsupported geometry type")


def _material_element(material: Material) -> ET.Element:
    elem = ET.Element("material", {"name": material.name})
    if material.rgba is not None:
        rgba = tuple(float(v) for v in material.rgba)
        if len(rgba) == 3:
            rgba = rgba + (1.0,)
        if len(rgba) != 4:
            raise ValidationError("Material rgba must have 3 or 4 values")
        ET.SubElement(elem, "color", {"rgba": _format_vec(rgba)})
    if material.texture:
        ET.SubElement(elem, "texture", {"filename": material.texture})
    _maybe_pbr_element(elem, material)
    return elem


def _maybe_pbr_element(elem: ET.Element, material: Material) -> None:
    # Vendor extension carrying UsdPreviewSurface-style PBR fields. Standard
    # URDF consumers ignore unknown elements; the viewer reads it.
    if not material.has_pbr():
        return
    attrs: dict[str, str] = {}
    if material.metallic is not None:
        attrs["metallic"] = _format_float(material.metallic)
    if material.roughness is not None:
        attrs["roughness"] = _format_float(material.roughness)
    if material.emissive is not None:
        attrs["emissive_rgb"] = _format_vec(material.emissive)
    if material.transmission is not None:
        attrs["transmission"] = _format_float(material.transmission)
    if material.ior is not None:
        attrs["ior"] = _format_float(material.ior)
    if material.clearcoat is not None:
        attrs["clearcoat"] = _format_float(material.clearcoat)
    if material.clearcoat_roughness is not None:
        attrs["clearcoat_roughness"] = _format_float(material.clearcoat_roughness)
    ET.SubElement(elem, "usdcraft_pbr", attrs)


def _visual_element(visual: Visual) -> ET.Element:
    attrs = {"name": visual.name} if visual.name else {}
    elem = ET.Element("visual", attrs)
    _maybe_origin(elem, visual.origin)
    _geometry_element(elem, visual.geometry)
    if visual.material:
        if isinstance(visual.material, Material):
            elem.append(_material_element(visual.material))
        else:
            ET.SubElement(elem, "material", {"name": str(visual.material)})
    return elem


def _collision_element(collision: Collision) -> ET.Element:
    attrs = {"name": collision.name} if collision.name else {}
    elem = ET.Element("collision", attrs)
    _maybe_origin(elem, collision.origin)
    _geometry_element(elem, collision.geometry)
    return elem


def _deformable_collision_element(
    collision: Collision,
    *,
    body_model: BodyModel,
) -> ET.Element:
    """Serialize inspectable rest-shape geometry without claiming rigid URDF physics."""

    attrs = {
        "body_model": body_model.value,
        "role": "deformable_contact",
        "realization": "rest_shape",
    }
    if collision.name:
        attrs["name"] = collision.name
    elem = ET.Element("usdcraft_collision", attrs)
    _maybe_origin(elem, collision.origin)
    _geometry_element(elem, collision.geometry)
    return elem


def _inertial_element(inertial: Inertial) -> ET.Element:
    elem = ET.Element("inertial")
    _maybe_origin(elem, inertial.origin)
    ET.SubElement(elem, "mass", {"value": _format_float(inertial.mass)})
    ET.SubElement(
        elem,
        "inertia",
        {
            "ixx": _format_float(inertial.inertia.ixx),
            "ixy": _format_float(inertial.inertia.ixy),
            "ixz": _format_float(inertial.inertia.ixz),
            "iyy": _format_float(inertial.inertia.iyy),
            "iyz": _format_float(inertial.inertia.iyz),
            "izz": _format_float(inertial.inertia.izz),
        },
    )
    return elem


def _part_element(part: Part) -> ET.Element:
    body_model = _body_model(part)
    elem = ET.Element(
        "link",
        {
            "name": part.name,
            "usdcraft_body_model": body_model.value,
            "usdcraft_body_realization": body_model.realization,
        },
    )
    if body_model == BodyModel.RIGID and part.inertial:
        elem.append(_inertial_element(part.inertial))
    for visual in part.visuals:
        elem.append(_visual_element(visual))
    if body_model == BodyModel.RIGID:
        for collision in part.collisions:
            elem.append(_collision_element(collision))
    else:
        for collision in part.collisions:
            elem.append(
                _deformable_collision_element(
                    collision,
                    body_model=body_model,
                )
            )
    return elem


def _body_model(part: Part) -> BodyModel:
    value = part.body_model
    return value if isinstance(value, BodyModel) else BodyModel(str(value))


def _normalized_axis(axis: tuple[float, float, float]) -> tuple[float, float, float]:
    x = float(axis[0])
    y = float(axis[1])
    z = float(axis[2])
    mag = (x * x + y * y + z * z) ** 0.5
    if mag <= 1e-12:
        return (x, y, z)
    return (x / mag, y / mag, z / mag)


def _articulation_element(articulation: Articulation) -> ET.Element:
    joint_type = (
        articulation.articulation_type
        if isinstance(articulation.articulation_type, ArticulationType)
        else ArticulationType(str(articulation.articulation_type))
    )
    joint_attrs = {"name": articulation.name, "type": joint_type.value}
    if joint_type == ArticulationType.SPHERICAL:
        if articulation.cone_deg is None:
            raise ValidationError(f"Spherical articulation {articulation.name!r} must set cone_deg")
        joint_attrs.update(
            {
                # URDF has no spherical joint type. Keep the document portable as a
                # standard floating joint while preserving the exact SDK semantics for
                # USDCraft-aware inventory, rendering, and verification.
                "type": ArticulationType.FLOATING.value,
                "usdcraft_type": ArticulationType.SPHERICAL.value,
                "usdcraft_axis": _format_vec(_normalized_axis(articulation.axis)),
                "usdcraft_cone_deg": _format_float(articulation.cone_deg),
            }
        )
    elem = ET.Element("joint", joint_attrs)
    _maybe_origin(elem, articulation.origin)
    ET.SubElement(elem, "parent", {"link": articulation.parent})
    ET.SubElement(elem, "child", {"link": articulation.child})

    if joint_type in {
        ArticulationType.REVOLUTE,
        ArticulationType.PRISMATIC,
        ArticulationType.CONTINUOUS,
    }:
        ET.SubElement(elem, "axis", {"xyz": _format_vec(_normalized_axis(articulation.axis))})

    if articulation.motion_limits and joint_type in {
        ArticulationType.REVOLUTE,
        ArticulationType.PRISMATIC,
        ArticulationType.CONTINUOUS,
    }:
        limit_attrs = {
            "effort": _format_float(articulation.motion_limits.effort),
            "velocity": _format_float(articulation.motion_limits.velocity),
        }
        if (
            joint_type
            in {
                ArticulationType.REVOLUTE,
                ArticulationType.PRISMATIC,
            }
            and articulation.motion_limits.lower is not None
        ):
            limit_attrs["lower"] = _format_float(articulation.motion_limits.lower)
        if (
            joint_type
            in {
                ArticulationType.REVOLUTE,
                ArticulationType.PRISMATIC,
            }
            and articulation.motion_limits.upper is not None
        ):
            limit_attrs["upper"] = _format_float(articulation.motion_limits.upper)
        ET.SubElement(elem, "limit", limit_attrs)

    if articulation.motion_properties and (
        articulation.motion_properties.damping is not None
        or articulation.motion_properties.friction is not None
    ):
        dyn_attrs = {}
        if articulation.motion_properties.damping is not None:
            dyn_attrs["damping"] = _format_float(articulation.motion_properties.damping)
        if articulation.motion_properties.friction is not None:
            dyn_attrs["friction"] = _format_float(articulation.motion_properties.friction)
        ET.SubElement(elem, "dynamics", dyn_attrs)

    if articulation.mimic is not None:
        mimic_attrs = {"joint": articulation.mimic.joint}
        multiplier = float(articulation.mimic.multiplier)
        offset = float(articulation.mimic.offset)
        formatted_multiplier = _format_float(multiplier)
        formatted_offset = _format_float(offset)
        if abs(multiplier - 1.0) > 1e-12:
            mimic_attrs["multiplier"] = formatted_multiplier
        if abs(offset) > 1e-12:
            mimic_attrs["offset"] = formatted_offset
        ET.SubElement(elem, "mimic", mimic_attrs)

    return elem
