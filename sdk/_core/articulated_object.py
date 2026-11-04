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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .assets import AssetContext, coerce_asset_context, resolve_asset_context
from .errors import ValidationError
from .types import (
    Articulation,
    ArticulationType,
    BaseMode,
    BodyModel,
    Box,
    Cylinder,
    Geometry,
    Inertial,
    JointDrive,
    Material,
    Mesh,
    Mimic,
    MotionLimits,
    MotionProperties,
    NewtonJointSolverTuning,
    Origin,
    Part,
    PassiveJointSpring,
    PhysicsMaterial,
    PhysxJointResistance,
    RigidBodyProperties,
    Sphere,
    TextureInputRef,
    Vec3,
    Visual,
    _coerce_body_model,
    _normalize_material_rgba,
)

_ALLOW_EXPLICIT_COLLISIONS_ATTR = "_sdk_allow_explicit_collisions"
_SCALAR_ARTICULATION_TYPES = {
    ArticulationType.REVOLUTE,
    ArticulationType.PRISMATIC,
    ArticulationType.CONTINUOUS,
}
_AXIS_ARTICULATION_TYPES = _SCALAR_ARTICULATION_TYPES | {ArticulationType.SPHERICAL}
_ANGULAR_ARTICULATION_TYPES = {
    ArticulationType.REVOLUTE,
    ArticulationType.CONTINUOUS,
}


def _part_name_ref(value: Union[str, Part], *, field_name: str) -> str:
    name: object
    if isinstance(value, str):
        name = value
    else:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            raise ValidationError(f"{field_name} must be a part name or Part instance")
    name = name.strip()
    if not name:
        raise ValidationError(f"{field_name} must be non-empty")
    return name


def _articulation_name_ref(value: Union[str, Articulation], *, field_name: str) -> str:
    name: object
    if isinstance(value, str):
        name = value
    else:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            raise ValidationError(
                f"{field_name} must be an articulation name or Articulation instance"
            )
    name = name.strip()
    if not name:
        raise ValidationError(f"{field_name} must be non-empty")
    return name


@dataclass
class ArticulatedObject:
    """Root object model for articulated products and assemblies."""

    name: str
    parts: List[Part] = field(default_factory=list)
    articulations: List[Articulation] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    physics_materials: List[PhysicsMaterial] = field(default_factory=list)
    base_mode: BaseMode | str = BaseMode.FLOATING
    meta: Dict[str, object] = field(default_factory=dict)
    assets: Optional[AssetContext] = None
    _part_index: Dict[str, Part] = field(default_factory=dict, init=False, repr=False)
    _articulation_index: Dict[str, Articulation] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        try:
            self.base_mode = (
                self.base_mode
                if isinstance(self.base_mode, BaseMode)
                else BaseMode(str(self.base_mode))
            )
        except ValueError as exc:
            raise ValidationError(f"Unknown base mode: {self.base_mode!r}") from exc
        self.assets = resolve_asset_context(self.assets)
        for part in self.parts:
            part.assets = coerce_asset_context(getattr(part, "assets", None)) or self.assets

    @property
    def links(self) -> List[Part]:
        return self.parts

    @links.setter
    def links(self, value: List[Part]) -> None:
        self.parts = value

    @property
    def joints(self) -> List[Articulation]:
        return self.articulations

    @joints.setter
    def joints(self, value: List[Articulation]) -> None:
        self.articulations = value

    def part(
        self,
        name: str,
        *,
        visuals: Optional[Iterable[Visual]] = None,
        inertial: Optional[Inertial] = None,
        rigid_body: Optional[RigidBodyProperties] = None,
        body_model: BodyModel | str = BodyModel.RIGID,
        meta: Optional[Dict[str, object]] = None,
    ) -> Part:
        part = Part(
            name=name,
            visuals=list(visuals or []),
            collisions=[],
            inertial=inertial,
            rigid_body=rigid_body,
            body_model=body_model,
            meta=dict(meta or {}),
            assets=self.assets,
        )
        self.parts.append(part)
        self._part_index[name] = part
        return part

    def link(
        self,
        name: str,
        *,
        visuals: Optional[Iterable[Visual]] = None,
        inertial: Optional[Inertial] = None,
        rigid_body: Optional[RigidBodyProperties] = None,
        body_model: BodyModel | str = BodyModel.RIGID,
        meta: Optional[Dict[str, object]] = None,
    ) -> Part:
        return self.part(
            name,
            visuals=visuals,
            inertial=inertial,
            rigid_body=rigid_body,
            body_model=body_model,
            meta=meta,
        )

    def articulation(
        self,
        name: str,
        articulation_type: Union[ArticulationType, str],
        parent: Union[str, Part],
        child: Union[str, Part],
        *,
        origin: Optional[Origin] = None,
        axis: Optional[Vec3] = None,
        cone_deg: Optional[float] = None,
        motion_limits: Optional[MotionLimits] = None,
        motion_properties: Optional[MotionProperties] = None,
        newton_joint_solver_tuning: Optional[NewtonJointSolverTuning] = None,
        passive_spring: Optional[PassiveJointSpring] = None,
        physx_joint_resistance: Optional[PhysxJointResistance] = None,
        physx_armature: Optional[float] = None,
        drive: Optional[JointDrive] = None,
        mimic: Optional[Mimic] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> Articulation:
        try:
            resolved_type = (
                articulation_type
                if isinstance(articulation_type, ArticulationType)
                else ArticulationType(str(articulation_type))
            )
        except ValueError as exc:
            raise ValidationError(f"Unknown articulation type: {articulation_type}") from exc

        articulation = Articulation(
            name=name,
            articulation_type=resolved_type,
            parent=_part_name_ref(parent, field_name="parent"),
            child=_part_name_ref(child, field_name="child"),
            origin=origin or Origin(),
            axis=axis or (0.0, 0.0, 1.0),
            cone_deg=cone_deg,
            motion_limits=motion_limits,
            motion_properties=motion_properties,
            newton_joint_solver_tuning=newton_joint_solver_tuning,
            passive_spring=passive_spring,
            physx_joint_resistance=physx_joint_resistance,
            physx_armature=physx_armature,
            drive=drive,
            mimic=mimic,
            meta=dict(meta or {}),
        )
        self.articulations.append(articulation)
        self._articulation_index[name] = articulation
        return articulation

    def joint(
        self,
        name: str,
        joint_type: Union[ArticulationType, str],
        parent: Union[str, Part],
        child: Union[str, Part],
        *,
        origin: Optional[Origin] = None,
        axis: Optional[Vec3] = None,
        cone_deg: Optional[float] = None,
        limit: Optional[MotionLimits] = None,
        dynamics: Optional[MotionProperties] = None,
        newton_joint_solver_tuning: Optional[NewtonJointSolverTuning] = None,
        passive_spring: Optional[PassiveJointSpring] = None,
        physx_joint_resistance: Optional[PhysxJointResistance] = None,
        physx_armature: Optional[float] = None,
        drive: Optional[JointDrive] = None,
        mimic: Optional[Mimic] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> Articulation:
        return self.articulation(
            name,
            articulation_type=joint_type,
            parent=parent,
            child=child,
            origin=origin,
            axis=axis,
            cone_deg=cone_deg,
            motion_limits=limit,
            motion_properties=dynamics,
            newton_joint_solver_tuning=newton_joint_solver_tuning,
            passive_spring=passive_spring,
            physx_joint_resistance=physx_joint_resistance,
            physx_armature=physx_armature,
            drive=drive,
            mimic=mimic,
            meta=meta,
        )

    def material(
        self,
        name: str,
        *,
        rgba: Optional[Sequence[float]] = None,
        color: Optional[Sequence[float]] = None,
        texture: Optional[str] = None,
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
    ) -> Material:
        if rgba is not None and color is not None:
            raise ValidationError("Material cannot set both rgba and color")
        material = Material(
            name=name,
            rgba=_normalize_material_rgba(
                rgba if rgba is not None else color,
                field_name="Material rgba",
            ),
            texture=texture,
            metallic=metallic,
            roughness=roughness,
            emissive=emissive,
            transmission=transmission,
            ior=ior,
            clearcoat=clearcoat,
            clearcoat_roughness=clearcoat_roughness,
            base_color_texture=base_color_texture,
            metallic_texture=metallic_texture,
            roughness_texture=roughness_texture,
            normal_texture=normal_texture,
            occlusion_texture=occlusion_texture,
            emissive_texture=emissive_texture,
            opacity_texture=opacity_texture,
        )
        self.materials.append(material)
        return material

    def physics_material(
        self,
        name: str,
        *,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.0,
        density: float | None = None,
    ) -> PhysicsMaterial:
        material = PhysicsMaterial(
            name=name,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
            density=density,
        )
        self.physics_materials.append(material)
        return material

    def _rebuild_indices_if_needed(self) -> None:
        self._part_index = {part.name: part for part in self.parts}
        self._articulation_index = {
            articulation.name: articulation for articulation in self.articulations
        }

    def get_part(self, name: Union[str, Part]) -> Part:
        self._rebuild_indices_if_needed()
        key = _part_name_ref(name, field_name="name")
        part = self._part_index.get(key)
        if part is None:
            self._part_index = {p.name: p for p in self.parts}
            part = self._part_index.get(key)
        if part is None:
            raise ValidationError(f"Unknown part: {name!r}")
        return part

    def get_link(self, name: Union[str, Part]) -> Part:
        return self.get_part(name)

    def get_articulation(self, name: Union[str, Articulation]) -> Articulation:
        self._rebuild_indices_if_needed()
        key = _articulation_name_ref(name, field_name="name")
        articulation = self._articulation_index.get(key)
        if articulation is None:
            self._articulation_index = {a.name: a for a in self.articulations}
            articulation = self._articulation_index.get(key)
        if articulation is None:
            raise ValidationError(f"Unknown articulation: {name!r}")
        return articulation

    def get_joint(self, name: Union[str, Articulation]) -> Articulation:
        return self.get_articulation(name)

    def root_parts(self) -> List[Part]:
        child_names = {articulation.child for articulation in self.articulations}
        return [part for part in self.parts if part.name not in child_names]

    def root_links(self) -> List[Part]:
        return self.root_parts()

    def validate(
        self,
        *,
        strict: bool = True,
        strict_mesh_paths: bool = False,
    ) -> None:
        if not self.parts:
            raise ValidationError("Articulated object must contain at least one part")

        part_names = [part.name for part in self.parts]
        if len(set(part_names)) != len(part_names):
            raise ValidationError("Part names must be unique")

        articulation_names = [articulation.name for articulation in self.articulations]
        if len(set(articulation_names)) != len(articulation_names):
            raise ValidationError("Articulation names must be unique")
        articulation_lookup = {
            articulation.name: articulation for articulation in self.articulations
        }

        material_names = [material.name for material in self.materials]
        if len(set(material_names)) != len(material_names):
            raise ValidationError("Material names must be unique")
        material_name_set = set(material_names)
        physics_material_names = [material.name for material in self.physics_materials]
        if len(set(physics_material_names)) != len(physics_material_names):
            raise ValidationError("Physics material names must be unique")
        physics_material_name_set = set(physics_material_names)
        for material in self.materials:
            _validate_material(material, f"material {material.name!r}")
        for physics_material in self.physics_materials:
            _validate_physics_material(
                physics_material, f"physics material {physics_material.name!r}"
            )

        part_name_set = set(part_names)
        part_lookup = {part.name: part for part in self.parts}
        for part in self.parts:
            part.body_model = _coerce_body_model(part.body_model)
        child_to_articulation: Dict[str, Articulation] = {}
        for articulation in self.articulations:
            context = f"articulation {articulation.name!r}"
            _validate_origin(articulation.origin, f"{context} origin")
            _validate_finite_vector(articulation.axis, f"{context} axis", length=3)
            if articulation.cone_deg is not None:
                _require_finite(articulation.cone_deg, f"{context} cone_deg")
            if articulation.motion_limits is not None:
                _validate_optional_finite_fields(
                    articulation.motion_limits,
                    ("effort", "velocity", "lower", "upper"),
                    f"{context} motion_limits",
                )
            if articulation.motion_properties is not None:
                _validate_optional_finite_fields(
                    articulation.motion_properties,
                    ("damping", "friction"),
                    f"{context} motion_properties",
                )
            if articulation.newton_joint_solver_tuning is not None:
                _validate_newton_joint_solver_tuning(
                    articulation.newton_joint_solver_tuning,
                    f"{context} newton_joint_solver_tuning",
                )
            if articulation.passive_spring is not None:
                _validate_passive_joint_spring(
                    articulation.passive_spring,
                    f"{context} passive_spring",
                )
            if articulation.physx_joint_resistance is not None:
                _validate_joint_resistance(
                    articulation.physx_joint_resistance,
                    f"{context} physx_joint_resistance",
                )
            if articulation.physx_armature is not None:
                _require_finite(articulation.physx_armature, f"{context} physx_armature")
            if articulation.drive is not None:
                _validate_optional_finite_fields(
                    articulation.drive,
                    (
                        "target_position",
                        "target_velocity",
                        "stiffness",
                        "damping",
                        "max_force",
                    ),
                    f"{context} drive",
                )
            if articulation.mimic is not None:
                _require_finite(
                    articulation.mimic.multiplier,
                    f"{context} mimic.multiplier",
                )
                _require_finite(articulation.mimic.offset, f"{context} mimic.offset")
            if articulation.parent not in part_name_set:
                raise ValidationError(
                    f"Articulation {articulation.name!r} references missing parent part {articulation.parent!r}"
                )
            if articulation.child not in part_name_set:
                raise ValidationError(
                    f"Articulation {articulation.name!r} references missing child part {articulation.child!r}"
                )
            if articulation.parent == articulation.child:
                raise ValidationError(
                    f"Articulation {articulation.name!r} parent and child cannot be the same"
                )
            nonrigid_part_names = [
                part_name
                for part_name in (articulation.parent, articulation.child)
                if part_lookup[part_name].body_model != BodyModel.RIGID
            ]
            if nonrigid_part_names and articulation.articulation_type != ArticulationType.FIXED:
                raise ValidationError(
                    f"Articulation {articulation.name!r} must be fixed because it involves "
                    f"non-rigid part(s): {', '.join(repr(name) for name in nonrigid_part_names)}"
                )
            for role, part_name in (("parent", articulation.parent), ("child", articulation.child)):
                referenced_part = part_lookup[part_name]
                if (
                    referenced_part.rigid_body is not None
                    and not referenced_part.rigid_body.enabled
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} {role} part {part_name!r} "
                        "cannot disable its rigid body"
                    )
            if articulation.child in child_to_articulation:
                raise ValidationError(
                    f"Part {articulation.child!r} has multiple parent articulations: "
                    f"{child_to_articulation[articulation.child].name!r} and {articulation.name!r}"
                )
            child_to_articulation[articulation.child] = articulation

            if strict and articulation.articulation_type in _AXIS_ARTICULATION_TYPES:
                if len(articulation.axis) != 3:
                    raise ValidationError(
                        f"Articulation {articulation.name!r} axis must have 3 values"
                    )
                if _norm(articulation.axis) == 0.0:
                    raise ValidationError(
                        f"Articulation {articulation.name!r} axis must be non-zero"
                    )

            if articulation.articulation_type == ArticulationType.SPHERICAL:
                if strict and articulation.cone_deg is None:
                    raise ValidationError(
                        f"Spherical articulation {articulation.name!r} must set cone_deg"
                    )
                if articulation.cone_deg is not None and (
                    not math.isfinite(articulation.cone_deg)
                    or not 0.0 < articulation.cone_deg <= 180.0
                ):
                    raise ValidationError(
                        f"Spherical articulation {articulation.name!r} cone_deg must be in (0, 180]"
                    )
            elif articulation.cone_deg is not None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} only supports cone_deg on spherical "
                    "articulations"
                )

            if strict and articulation.articulation_type in {
                ArticulationType.REVOLUTE,
                ArticulationType.PRISMATIC,
            }:
                if articulation.motion_limits is None:
                    raise ValidationError(
                        f"Articulation {articulation.name!r} must include motion limits"
                    )
                if (
                    articulation.motion_limits.lower is None
                    or articulation.motion_limits.upper is None
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} requires lower and upper limits"
                    )

            if strict and articulation.articulation_type == ArticulationType.CONTINUOUS:
                if articulation.motion_limits is None:
                    raise ValidationError(
                        f"Continuous articulation {articulation.name!r} must set "
                        "motion_limits=MotionLimits(effort=..., velocity=...) and "
                        "cannot set lower/upper limits"
                    )
                if (
                    articulation.motion_limits.lower is not None
                    or articulation.motion_limits.upper is not None
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} cannot include lower/upper limits"
                    )

            if (
                articulation.motion_limits
                and articulation.articulation_type not in _SCALAR_ARTICULATION_TYPES
            ):
                raise ValidationError(
                    f"Articulation {articulation.name!r} does not support motion limits"
                )
            if (
                articulation.motion_properties is not None
                and articulation.articulation_type not in _SCALAR_ARTICULATION_TYPES
            ):
                raise ValidationError(
                    f"Articulation {articulation.name!r} does not support motion properties"
                )
            if (
                articulation.newton_joint_solver_tuning is not None
                and articulation.articulation_type not in _SCALAR_ARTICULATION_TYPES
            ):
                raise ValidationError(
                    f"Articulation {articulation.name!r} does not support Newton joint solver "
                    "tuning"
                )
            if (
                articulation.passive_spring is not None
                and articulation.articulation_type not in _SCALAR_ARTICULATION_TYPES
            ):
                raise ValidationError(
                    f"Articulation {articulation.name!r} does not support a passive spring"
                )
            if (
                articulation.physx_joint_resistance is not None
                and articulation.articulation_type not in _SCALAR_ARTICULATION_TYPES
            ):
                raise ValidationError(
                    f"Articulation {articulation.name!r} does not support PhysX joint resistance"
                )
            if (
                articulation.physx_armature is not None
                and articulation.articulation_type not in _SCALAR_ARTICULATION_TYPES
            ):
                raise ValidationError(
                    f"Articulation {articulation.name!r} does not support PhysX armature"
                )

            if articulation.motion_limits:
                if (
                    articulation.motion_limits.effort <= 0
                    or articulation.motion_limits.velocity <= 0
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} limits must be positive"
                    )
                if (
                    articulation.motion_limits.lower is not None
                    and articulation.motion_limits.upper is not None
                    and articulation.motion_limits.lower > articulation.motion_limits.upper
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} lower limit exceeds upper limit"
                    )
                if (
                    articulation.articulation_type
                    in {ArticulationType.REVOLUTE, ArticulationType.PRISMATIC}
                    and articulation.motion_limits.lower is not None
                    and articulation.motion_limits.upper is not None
                    and not (
                        articulation.motion_limits.lower <= 0.0 <= articulation.motion_limits.upper
                    )
                ):
                    raise ValidationError(
                        f"Articulation {articulation.name!r} limits must include q=0 "
                        "as the assembled rest pose"
                    )

        for articulation in self.articulations:
            if articulation.passive_spring is not None and articulation.drive is not None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} cannot set passive_spring with drive; "
                    "one scalar joint cannot carry both passive-spring and active-drive "
                    "reference semantics"
                )
            if (
                articulation.drive is not None
                and _motion_domain(articulation.articulation_type) is None
            ):
                raise ValidationError(
                    f"Articulation {articulation.name!r} only supports drives on "
                    "revolute, continuous, or prismatic articulations"
                )
            mimic = articulation.mimic
            if mimic is None:
                continue
            if _motion_domain(articulation.articulation_type) is None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} only supports mimic on "
                    "revolute, continuous, or prismatic articulations"
                )
            source = articulation_lookup.get(mimic.joint)
            if source is None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} mimic references missing articulation "
                    f"{mimic.joint!r}"
                )
            if source.name == articulation.name:
                raise ValidationError(f"Articulation {articulation.name!r} cannot mimic itself")
            if _motion_domain(source.articulation_type) is None:
                raise ValidationError(
                    f"Articulation {articulation.name!r} mimic source {source.name!r} must be "
                    "revolute, continuous, or prismatic"
                )
            if _motion_domain(source.articulation_type) != _motion_domain(
                articulation.articulation_type
            ):
                raise ValidationError(
                    f"Articulation {articulation.name!r} mimic source {source.name!r} must have "
                    "a compatible motion domain"
                )

        _validate_mimic_cycles(articulation_lookup)

        for part in self.parts:
            if part.body_model != BodyModel.RIGID:
                if part.inertial is not None:
                    raise ValidationError(
                        f"Non-rigid part {part.name!r} cannot define inertial properties"
                    )
                if part.rigid_body is not None:
                    raise ValidationError(
                        f"Non-rigid part {part.name!r} cannot define rigid_body properties"
                    )
            if part.inertial:
                _validate_inertial(part.inertial, f"part {part.name!r} inertial")
                if part.inertial.mass <= 0:
                    raise ValidationError(f"Part {part.name!r} inertial mass must be positive")
            if (
                part.rigid_body is not None
                and part.rigid_body.physics_material is not None
                and part.rigid_body.physics_material not in physics_material_name_set
            ):
                raise ValidationError(
                    f"Part {part.name!r} references unknown physics material "
                    f"{part.rigid_body.physics_material!r}"
                )
            for visual in part.visuals:
                _validate_origin(visual.origin, f"part {part.name!r} visual origin")
                if visual.material is not None:
                    if isinstance(visual.material, str):
                        if str(visual.material) not in material_name_set:
                            raise ValidationError(
                                f"Part {part.name!r} visual references unknown material {visual.material!r}"
                            )
                    elif not isinstance(visual.material, Material):
                        raise ValidationError(
                            f"Part {part.name!r} visual material must be a Material or str name"
                        )
                    else:
                        _validate_material(
                            visual.material,
                            f"part {part.name!r} visual material {visual.material.name!r}",
                        )
                _validate_geometry(
                    visual.geometry,
                    f"part {part.name} visual",
                    strict_mesh_paths=strict_mesh_paths,
                )
            if part.collisions and not bool(getattr(self, _ALLOW_EXPLICIT_COLLISIONS_ATTR, False)):
                raise ValidationError(
                    f"Part {part.name!r} defines explicit collisions. "
                    "Source models must express geometry only through visuals and "
                    "primitive provenance instead of "
                    "authoring Part.collisions directly."
                )
            for collision in part.collisions:
                _validate_origin(collision.origin, f"part {part.name!r} collision origin")
                if (
                    collision.physics is not None
                    and collision.physics.physics_material is not None
                    and collision.physics.physics_material not in physics_material_name_set
                ):
                    raise ValidationError(
                        f"Part {part.name!r} collision references unknown physics material "
                        f"{collision.physics.physics_material!r}"
                    )
                _validate_geometry(
                    collision.geometry,
                    f"part {part.name} collision",
                    strict_mesh_paths=strict_mesh_paths,
                )
        if strict:
            self._validate_connectivity(part_name_set)

    def to_urdf(
        self,
        *,
        pretty: bool = True,
        asset_root: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Compatibility shim for older generated scripts.

        URDF export is intentionally an internal implementation detail; new code
        should let the harness compile `object_model` directly.
        """

        from ._urdf_export import compile_object_to_urdf_xml

        return compile_object_to_urdf_xml(self, pretty=pretty, asset_root=asset_root)

    def _validate_connectivity(self, part_name_set: set[str]) -> None:
        if len(part_name_set) <= 1:
            return

        parent_to_children: Dict[str, List[str]] = {name: [] for name in part_name_set}
        child_set: set[str] = set()
        for articulation in self.articulations:
            parent_to_children.setdefault(articulation.parent, []).append(articulation.child)
            child_set.add(articulation.child)

        roots = sorted(part_name_set - child_set)
        if not roots:
            raise ValidationError(
                "Articulated object has no root part (cycle or every part is a child)"
            )
        if len(roots) > 1:
            raise ValidationError(
                "Articulated object must have exactly one root part; "
                f"found multiple root parts: {roots}"
            )

        visited: set[str] = set()
        queue: List[str] = list(roots)
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(parent_to_children.get(current, []))

        if visited != part_name_set:
            missing = sorted(part_name_set - visited)
            raise ValidationError(
                "Articulated object contains unreachable parts (cycle or broken articulations): "
                f"{missing}"
            )


def _norm(values: Sequence[float]) -> float:
    return sum(float(v) ** 2 for v in values) ** 0.5


def _require_finite(value: Any, context: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{context} must be numeric and finite") from exc
    if not math.isfinite(numeric):
        raise ValidationError(f"{context} must be finite")
    return numeric


def _validate_finite_vector(values: Sequence[float], context: str, *, length: int) -> None:
    if len(values) != length:
        raise ValidationError(f"{context} must have {length} values")
    for index, value in enumerate(values):
        _require_finite(value, f"{context}[{index}]")


def _validate_origin(origin: Origin, context: str) -> None:
    if not isinstance(origin, Origin):
        raise ValidationError(f"{context} must be an Origin")
    _validate_finite_vector(origin.xyz, f"{context}.xyz", length=3)
    _validate_finite_vector(origin.rpy, f"{context}.rpy", length=3)


def _validate_optional_finite_fields(
    value: object,
    field_names: Sequence[str],
    context: str,
) -> None:
    for field_name in field_names:
        field_value = getattr(value, field_name, None)
        if field_value is not None:
            _require_finite(field_value, f"{context}.{field_name}")


def _validate_joint_resistance(value: PhysxJointResistance, context: str) -> None:
    field_names = (
        "static_friction_effort",
        "dynamic_friction_effort",
        "viscous_friction_coefficient",
        "coefficient",
    )
    found = False
    for field_name in field_names:
        if hasattr(value, field_name):
            found = True
            _require_finite(getattr(value, field_name), f"{context}.{field_name}")
    if not found:
        raise ValidationError(f"{context} has an unsupported type")


def _validate_newton_joint_solver_tuning(
    value: NewtonJointSolverTuning,
    context: str,
) -> None:
    if not isinstance(value, NewtonJointSolverTuning):
        raise ValidationError(f"{context} must be a NewtonJointSolverTuning")
    NewtonJointSolverTuning(
        solref_friction=value.solref_friction,
        solimp_friction=value.solimp_friction,
    )


def _validate_passive_joint_spring(value: PassiveJointSpring, context: str) -> None:
    if not isinstance(value, PassiveJointSpring):
        raise ValidationError(f"{context} must be a PassiveJointSpring")
    PassiveJointSpring(stiffness=value.stiffness, reference=value.reference)


def _validate_inertial(inertial: Inertial, context: str) -> None:
    if not isinstance(inertial, Inertial):
        raise ValidationError(f"{context} must be an Inertial")
    _require_finite(inertial.mass, f"{context}.mass")
    inertia = inertial.inertia
    if inertia is None:
        raise ValidationError(f"{context}.inertia is required")
    for field_name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        _require_finite(getattr(inertia, field_name, None), f"{context}.inertia.{field_name}")
    _validate_origin(inertial.origin, f"{context}.origin")


def _motion_domain(articulation_type: ArticulationType | str) -> str | None:
    if articulation_type in _ANGULAR_ARTICULATION_TYPES:
        return "angular"
    if articulation_type == ArticulationType.PRISMATIC:
        return "linear"
    return None


def _validate_mimic_cycles(articulation_lookup: Dict[str, Articulation]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle_start = path.index(name)
            cycle = path[cycle_start:] + [name]
            raise ValidationError(f"Mimic cycle detected: {' -> '.join(cycle)}")

        visiting.add(name)
        path.append(name)
        mimic = articulation_lookup[name].mimic
        if mimic is not None and mimic.joint in articulation_lookup:
            visit(mimic.joint)
        path.pop()
        visiting.remove(name)
        visited.add(name)

    for name in articulation_lookup:
        visit(name)


def _validate_geometry(
    geometry: Geometry,
    context: str,
    *,
    strict_mesh_paths: bool = False,
) -> None:
    if isinstance(geometry, Box):
        _validate_finite_vector(geometry.size, f"{context} box size", length=3)
        if any(float(value) <= 0 for value in geometry.size):
            raise ValidationError(f"{context} box size must be positive")
    elif isinstance(geometry, Cylinder):
        radius = _require_finite(geometry.radius, f"{context} cylinder radius")
        length = _require_finite(geometry.length, f"{context} cylinder length")
        if radius <= 0 or length <= 0:
            raise ValidationError(f"{context} cylinder radius/length must be positive")
    elif isinstance(geometry, Sphere):
        radius = _require_finite(geometry.radius, f"{context} sphere radius")
        if radius <= 0:
            raise ValidationError(f"{context} sphere radius must be positive")
    elif isinstance(geometry, Mesh):
        if not geometry.filename and not geometry.name:
            raise ValidationError(f"{context} mesh filename or name is required")
        if strict_mesh_paths:
            filename = str(geometry.filename or "")
            if os.path.isabs(filename):
                raise ValidationError(
                    f"{context} mesh filename must be relative (got absolute path): {filename}"
                )
            if ".." in Path(filename).parts:
                raise ValidationError(
                    f"{context} mesh filename must not contain '..' segments: {filename}"
                )
        if geometry.scale is not None:
            _validate_finite_vector(geometry.scale, f"{context} mesh scale", length=3)
            if any(float(value) <= 0 for value in geometry.scale):
                raise ValidationError(f"{context} mesh scale must be positive")
        if geometry.normal_crease_deg is not None:
            crease = _require_finite(
                geometry.normal_crease_deg,
                f"{context} mesh normal_crease_deg",
            )
            if not 0.0 <= crease <= 180.0:
                raise ValidationError(
                    f"{context} mesh normal_crease_deg must be between 0 and 180 degrees"
                )
        if geometry.source_geometry is not None:
            _validate_geometry(
                geometry.source_geometry,
                f"{context} mesh source_geometry",
                strict_mesh_paths=False,
            )
        if geometry.source_transform is not None:
            if len(geometry.source_transform) != 4:
                raise ValidationError(f"{context} mesh source_transform must have 4 rows")
            for row_index, row in enumerate(geometry.source_transform):
                _validate_finite_vector(
                    row,
                    f"{context} mesh source_transform[{row_index}]",
                    length=4,
                )
    else:
        raise ValidationError(f"{context} has unsupported geometry type")


def _validate_material(material: Material, context: str) -> None:
    if not isinstance(material.name, str) or not material.name:
        raise ValidationError(f"{context} name is required")
    if material.rgba is not None:
        if len(material.rgba) not in (3, 4):
            raise ValidationError(f"{context} rgba must have 3 or 4 values")
        _validate_finite_vector(material.rgba, f"{context} rgba", length=len(material.rgba))
    if material.emissive is not None:
        _validate_finite_vector(material.emissive, f"{context} emissive", length=3)
    _validate_optional_finite_fields(
        material,
        (
            "metallic",
            "roughness",
            "transmission",
            "ior",
            "clearcoat",
            "clearcoat_roughness",
        ),
        context,
    )


def _validate_physics_material(material: PhysicsMaterial, context: str) -> None:
    _validate_optional_finite_fields(
        material,
        ("static_friction", "dynamic_friction", "restitution", "density"),
        context,
    )
