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

import contextlib
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
from pxr import Gf, Kind, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt
from scipy.spatial.transform import Rotation

from ._mesh.normals import (
    DEFAULT_NORMAL_CREASE_DEGREES,
)
from ._mesh.normals import (
    crease_aware_face_varying_normals as _crease_aware_face_varying_normals,
)
from ._usd_backend_layers import (
    _author_asset_interface,
    _author_newton_mjc_layer,
    _author_physx_layer,
    _BackendMappingNotes,
)
from ._usd_common import (
    _ASSET_INTERFACE_NAME,
    _NEWTON_ENTRY_NAME,
    _NEWTON_MJC_LAYER_NAME,
    _PHYSICS_LAYER_NAME,
    _PHYSX_ENTRY_NAME,
    _PHYSX_LAYER_NAME,
    _joint_type,
)
from ._usd_validate import (
    _resolve_native_articulation_root_bindings,
    _validate_asset_interface_variants,
    _validate_usd_asset,
)
from .articulated_object import ArticulatedObject
from .assets import require_mesh_filename, resolve_asset_root, resolve_mesh_path
from .errors import ValidationError
from .exact_collisions import compile_object_model_with_exact_collisions
from .types import (
    Articulation,
    ArticulationType,
    BaseMode,
    BodyModel,
    Box,
    Collision,
    CollisionApproximation,
    Cylinder,
    Geometry,
    Material,
    Mesh,
    Origin,
    Part,
    PhysxJointFriction,
    PhysxJointFrictionCoefficient,
    Sphere,
    TextureInput,
    Visual,
)

_NORMAL_CREASE_ANGLE_DEGREES = DEFAULT_NORMAL_CREASE_DEGREES


def _srgb_channel_to_linear(value: float) -> float:
    """Decode one display-referred sRGB channel for a USD shader input."""

    value = float(value)
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _linear_channel_to_srgb(value: float) -> float:
    """Encode one linear channel into the public Material sRGB contract."""

    value = float(value)
    if value <= 0.0031308:
        return value * 12.92
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


@dataclass(slots=True, frozen=True)
class UsdExportResult:
    usdc_path: Path
    usdz_path: Path
    warnings: tuple[str, ...] = ()
    metrics: dict[str, object] = field(default_factory=dict)
    # Declared after the required fields so they stay optional in positional
    # construction. Exported results always populate both fields.
    newton_usdc_path: Path | None = None
    asset_structure_paths: dict[str, Path] = field(default_factory=dict)
    # Informational backend-mapping notes: facts about the PhysX/Newton layers that hold for every
    # joint of the given kind (see `_BackendMappingNotes`). They are recorded, never acted on.
    notes: tuple[str, ...] = ()


class _NameScope:
    def __init__(self) -> None:
        self._used: set[str] = set()

    def make(self, value: object, *, fallback: str) -> str:
        raw = str(value or fallback).strip()
        valid = Tf.MakeValidIdentifier(raw) or fallback
        candidate = valid
        suffix = 2
        while candidate in self._used:
            candidate = f"{valid}_{suffix}"
            suffix += 1
        self._used.add(candidate)
        return candidate


class _UsdObjectExporter:
    def __init__(
        self,
        object_model: ArticulatedObject,
        *,
        stage: Usd.Stage,
        output_dir: Path,
        asset_root: Path,
    ) -> None:
        self.model = object_model
        self.stage = stage
        self.output_dir = output_dir
        self.asset_root = asset_root
        self.warnings: list[str] = []
        self.backend_notes = _BackendMappingNotes()
        self.root_path = Sdf.Path("/Asset")
        self.links_path = self.root_path.AppendChild("Links")
        self.joints_path = self.root_path.AppendChild("Joints")
        self.looks_path = self.root_path.AppendChild("Looks")
        self.physics_materials_path = self.root_path.AppendChild("PhysicsMaterials")
        self._part_paths: dict[str, Sdf.Path] = {}
        self._parts_by_name = {part.name: part for part in self.model.parts}
        self._visual_materials: dict[str, UsdShade.Material] = {}
        self._physics_materials: dict[str, UsdShade.Material] = {}
        self._joint_paths: dict[str, Sdf.Path] = {}
        self._texture_names: set[str] = set()
        self._embedded_material_index = 0
        self._rigid_runtime_collision_shape_count = 0
        self._deformable_rest_shape_collision_count = 0

    def export(self) -> dict[str, object]:
        self._define_stage_root()
        self._define_visual_materials()
        self._define_physics_materials()
        world_transforms = self._part_world_transforms()
        self._define_parts(world_transforms)
        self._define_joints()
        self._define_fixed_base_joint_if_needed()
        native_joint_count = sum(1 for prim in self.stage.Traverse() if prim.IsA(UsdPhysics.Joint))
        # Resolve from the graph that was actually authored. FLOATING placement
        # edges and non-rigid static attachments do not become native joints, so
        # they may split the authoring tree into independent USD components.
        for binding in _resolve_native_articulation_root_bindings(self.stage):
            UsdPhysics.ArticulationRootAPI.Apply(self.stage.GetPrimAtPath(binding.api_path))
        floating_placement_count = sum(
            1
            for joint in self.model.articulations
            if _joint_type(joint) == ArticulationType.FLOATING
        )
        return {
            "part_count": len(self.model.parts),
            "joint_count": native_joint_count,
            "authoring_relation_count": len(self.model.articulations),
            "floating_placement_count": floating_placement_count,
            "body_model_counts": {
                body_model.value: sum(
                    1 for part in self.model.parts if _body_model(part) == body_model
                )
                for body_model in BodyModel
            },
            "collision_shape_counts": {
                "rigid_runtime": self._rigid_runtime_collision_shape_count,
                "deformable_rest_shape": self._deformable_rest_shape_collision_count,
                "deformable_live": 0,
            },
            "visual_material_count": len(self._visual_materials),
            "physics_material_count": len(self._physics_materials),
            "prim_count": sum(1 for _ in self.stage.Traverse()),
        }

    def _define_stage_root(self) -> None:
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1.0)
        root = UsdGeom.Xform.Define(self.stage, self.root_path)
        self.stage.SetDefaultPrim(root.GetPrim())
        Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)
        root.GetPrim().SetCustomDataByKey("usdcraft:objectName", self.model.name)
        root.GetPrim().SetCustomDataByKey(
            "usdcraft:baseMode", _base_mode(self.model.base_mode).value
        )
        UsdGeom.Scope.Define(self.stage, self.links_path)
        UsdGeom.Scope.Define(self.stage, self.joints_path)
        UsdGeom.Scope.Define(self.stage, self.looks_path)
        UsdGeom.Scope.Define(self.stage, self.physics_materials_path)

    def _define_visual_materials(self) -> None:
        names = _NameScope()
        for material in self.model.materials:
            prim_name = names.make(material.name, fallback="Material")
            authored = self._author_preview_material(
                material,
                self.looks_path.AppendChild(prim_name),
            )
            self._visual_materials[material.name] = authored

    def _define_physics_materials(self) -> None:
        names = _NameScope()
        for material in self.model.physics_materials:
            prim_name = names.make(material.name, fallback="PhysicsMaterial")
            usd_material = UsdShade.Material.Define(
                self.stage, self.physics_materials_path.AppendChild(prim_name)
            )
            api = UsdPhysics.MaterialAPI.Apply(usd_material.GetPrim())
            api.CreateStaticFrictionAttr(float(material.static_friction))
            api.CreateDynamicFrictionAttr(float(material.dynamic_friction))
            api.CreateRestitutionAttr(float(material.restitution))
            if material.density is not None:
                api.CreateDensityAttr(float(material.density))
            self._physics_materials[material.name] = usd_material

    def _author_preview_material(
        self,
        material: Material,
        path: Sdf.Path,
        *,
        texture_asset: str | None = None,
        texture_alpha_mode: str | None = None,
        texture_alpha_cutoff: float | None = None,
    ) -> UsdShade.Material:
        usd_material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path.AppendChild("PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        rgba = material.rgba or (0.7, 0.7, 0.7, 1.0)
        diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        # SDK/URDF colors are display-referred sRGB; UsdPreviewSurface scalar
        # color inputs are scene-linear.  Textures carry their own
        # sourceColorSpace metadata and are intentionally not converted here.
        diffuse.Set(Gf.Vec3f(*(_srgb_channel_to_linear(value) for value in rgba[:3])))
        opacity_value = float(rgba[3])
        if material.transmission is not None:
            opacity_value *= 1.0 - float(material.transmission)
            self.backend_notes.add("transmission_opacity", material.name)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity_value)
        if material.metallic is not None:
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(material.metallic))
        if material.roughness is not None:
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(material.roughness))
        if material.emissive is not None:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*(float(value) for value in material.emissive))
            )
        if material.ior is not None:
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(material.ior))
        if material.clearcoat is not None:
            shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(material.clearcoat))
        if material.clearcoat_roughness is not None:
            shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(
                float(material.clearcoat_roughness)
            )

        resolved_texture = texture_asset
        base_texture = material.base_color_texture
        if resolved_texture is None and base_texture is not None:
            resolved_texture = self._copy_texture(Path(base_texture.file), material.name)
        if resolved_texture is not None:
            tex = self._author_texture_node(
                path,
                "BaseColorTexture",
                base_texture or TextureInput(resolved_texture, source_color_space="sRGB"),
                resolved_texture,
            )
            diffuse.ConnectToSource(tex.ConnectableAPI(), "rgb")
            # An RGBA image is not necessarily transparent. Connecting an
            # all-opaque alpha channel still sends renderers down their
            # transparency path and can cause sorting/flicker on dense repeated
            # materials. Honor imported glTF alpha semantics, but only author
            # texture opacity when at least one pixel is genuinely transparent.
            has_transparent_pixels = self._texture_has_transparent_pixels(resolved_texture)
            if has_transparent_pixels and texture_alpha_mode != "OPAQUE":
                shader.GetInput("opacity").ConnectToSource(tex.ConnectableAPI(), "a")
                if texture_alpha_mode == "MASK":
                    shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(
                        0.5 if texture_alpha_cutoff is None else texture_alpha_cutoff
                    )

        texture_connections = (
            ("metallic", Sdf.ValueTypeNames.Float, material.metallic_texture, "MetallicTexture"),
            ("roughness", Sdf.ValueTypeNames.Float, material.roughness_texture, "RoughnessTexture"),
            ("occlusion", Sdf.ValueTypeNames.Float, material.occlusion_texture, "OcclusionTexture"),
            (
                "emissiveColor",
                Sdf.ValueTypeNames.Color3f,
                material.emissive_texture,
                "EmissiveTexture",
            ),
            ("opacity", Sdf.ValueTypeNames.Float, material.opacity_texture, "OpacityTexture"),
        )
        for input_name, value_type, texture_input, node_name in texture_connections:
            if texture_input is None:
                continue
            asset = self._copy_texture(Path(texture_input.file), f"{material.name}:{input_name}")
            if asset is None:
                continue
            texture = self._author_texture_node(path, node_name, texture_input, asset)
            shader.CreateInput(input_name, value_type).ConnectToSource(
                texture.ConnectableAPI(), texture_input.channel
            )
        if material.normal_texture is not None:
            normal_input = material.normal_texture
            asset = self._copy_texture(Path(normal_input.file), f"{material.name}:normal")
            if asset is not None:
                texture = self._author_texture_node(
                    path,
                    "NormalTexture",
                    normal_input,
                    asset,
                    normal_map=True,
                )
                shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                    texture.ConnectableAPI(), "rgb"
                )

        usd_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return usd_material

    def _author_texture_node(
        self,
        material_path: Sdf.Path,
        node_name: str,
        texture_input: TextureInput,
        asset_path: str,
        *,
        normal_map: bool = False,
    ) -> UsdShade.Shader:
        texture = UsdShade.Shader.Define(self.stage, material_path.AppendChild(node_name))
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(asset_path))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            texture_input.source_color_space
        )
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(texture_input.wrap_s)
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(texture_input.wrap_t)
        if normal_map:
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
            )
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
            )
        reader_name = Tf.MakeValidIdentifier(f"PrimvarReader_{texture_input.uv_set}")
        reader_path = material_path.AppendChild(reader_name)
        reader = UsdShade.Shader.Get(self.stage, reader_path)
        if not reader:
            reader = UsdShade.Shader.Define(self.stage, reader_path)
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            # The Sdr definition for UsdPrimvarReader_float2 declares varname
            # as a string.  Authoring it as a token opens in many viewers, but
            # fails OpenUSD's ShaderSdrCompliance validator (and therefore is
            # not acceptable for packaged USDZ assets).
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(texture_input.uv_set)
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        return texture

    def _copy_texture(self, source: Path, label: str) -> str | None:
        if not source.is_absolute():
            source = (self.asset_root / source).resolve()
        if not source.is_file():
            self.warnings.append(f"Material {label!r} texture was not found: {source}")
            return None
        textures_dir = self.output_dir / "isaac" / "textures"
        textures_dir.mkdir(parents=True, exist_ok=True)
        name = _unique_filename(source.name, self._texture_names)
        shutil.copy2(source, textures_dir / name)
        return f"textures/{name}"

    def _texture_has_transparent_pixels(self, relative_asset: str) -> bool:
        """Return True when the packaged texture contains non-opaque pixels.

        ``relative_asset`` is the usdc-relative path returned by ``_copy_texture``
        / ``_save_embedded_texture`` (e.g. ``"textures/foo.png"``); the on-disk
        file lives under ``<output_dir>/isaac/``.
        """
        try:
            from PIL import Image

            with Image.open(self.output_dir / "isaac" / relative_asset) as image:
                if "A" not in image.getbands() and "transparency" not in image.info:
                    return False
                alpha = image.convert("RGBA").getchannel("A")
                minimum_alpha, _ = alpha.getextrema()
                return int(cast(int, minimum_alpha)) < 255
        except Exception as exc:
            self.warnings.append(
                f"Could not inspect texture transparency for {relative_asset!r}; "
                f"kept scalar opacity: {exc}"
            )
            return False

    def _save_embedded_texture(self, image: Any, label: str) -> str | None:
        if image is None or not hasattr(image, "save"):
            return None
        textures_dir = self.output_dir / "isaac" / "textures"
        textures_dir.mkdir(parents=True, exist_ok=True)
        base = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label).strip("_") or "texture"
        name = _unique_filename(f"{base}.png", self._texture_names)
        try:
            image.save(textures_dir / name, format="PNG")
        except Exception as exc:
            self.warnings.append(f"Failed to extract embedded texture {label!r}: {exc}")
            return None
        return f"textures/{name}"

    def _part_world_transforms(self) -> dict[str, np.ndarray]:
        children: dict[str, list[Articulation]] = {}
        child_names: set[str] = set()
        for joint in self.model.articulations:
            children.setdefault(joint.parent, []).append(joint)
            child_names.add(joint.child)
        transforms: dict[str, np.ndarray] = {}

        def visit(name: str, parent_tf: np.ndarray) -> None:
            if name in transforms:
                return
            transforms[name] = parent_tf
            for joint in children.get(name, []):
                visit(joint.child, parent_tf @ _origin_matrix(joint.origin))

        for part in self.model.parts:
            if part.name not in child_names:
                visit(part.name, np.eye(4, dtype=float))
        for part in self.model.parts:
            transforms.setdefault(part.name, np.eye(4, dtype=float))
        return transforms

    def _define_parts(self, transforms: dict[str, np.ndarray]) -> None:
        names = _NameScope()
        rigid_articulations = [
            joint for joint in self.model.articulations if not self._involves_nonrigid(joint)
        ]
        dynamic_names = {joint.parent for joint in rigid_articulations} | {
            joint.child for joint in rigid_articulations
        }
        root_names = {part.name for part in self.model.root_parts()}
        for part in self.model.parts:
            prim_name = names.make(part.name, fallback="Part")
            part_path = self.links_path.AppendChild(prim_name)
            self._part_paths[part.name] = part_path
            xform = UsdGeom.Xform.Define(self.stage, part_path)
            _set_world_transform(xform, transforms[part.name])
            xform.GetPrim().SetCustomDataByKey("usdcraft:partName", part.name)
            body_model = _body_model(part)
            xform.GetPrim().SetCustomDataByKey("usdcraft:bodyModel", body_model.value)
            xform.GetPrim().SetCustomDataByKey("usdcraft:bodyRealization", body_model.realization)
            if body_model != BodyModel.RIGID:
                self.backend_notes.add("non_rigid_body", part.name, body_model.value)
            rigid = part.rigid_body
            should_be_rigid = body_model == BodyModel.RIGID and (
                part.name in dynamic_names or part.inertial is not None or rigid is not None
            )
            if should_be_rigid and (rigid is None or rigid.enabled):
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
                is_kinematic_root = (
                    part.name in root_names
                    and _base_mode(self.model.base_mode) == BaseMode.KINEMATIC
                )
                rigid_api.CreateKinematicEnabledAttr(
                    bool(is_kinematic_root or (rigid is not None and rigid.kinematic))
                )
                if part.inertial is not None:
                    self._author_mass(xform.GetPrim(), part)

            visuals_scope = UsdGeom.Scope.Define(self.stage, part_path.AppendChild("Visuals"))
            collisions_scope = UsdGeom.Scope.Define(self.stage, part_path.AppendChild("Collisions"))
            visual_names = _NameScope()
            for index, visual in enumerate(part.visuals):
                self._author_visual(
                    visual,
                    visuals_scope.GetPath().AppendChild(
                        visual_names.make(visual.name, fallback=f"visual_{index:03d}")
                    ),
                )
            collision_names = _NameScope()
            for index, collision in enumerate(part.collisions):
                runtime_collision = body_model == BodyModel.RIGID
                authored_shape_count = self._author_collision(
                    collision,
                    part,
                    collisions_scope.GetPath().AppendChild(
                        collision_names.make(collision.name, fallback=f"collision_{index:03d}")
                    ),
                    runtime_collision=runtime_collision,
                )
                if runtime_collision:
                    self._rigid_runtime_collision_shape_count += authored_shape_count
                else:
                    self._deformable_rest_shape_collision_count += authored_shape_count

    def _author_mass(self, prim: Usd.Prim, part: Part) -> None:
        inertial = part.inertial
        if inertial is None:
            return
        tensor = np.array(
            [
                [inertial.inertia.ixx, inertial.inertia.ixy, inertial.inertia.ixz],
                [inertial.inertia.ixy, inertial.inertia.iyy, inertial.inertia.iyz],
                [inertial.inertia.ixz, inertial.inertia.iyz, inertial.inertia.izz],
            ],
            dtype=float,
        )
        eigenvalues = np.linalg.eigvalsh(tensor)
        if not np.all(np.isfinite(eigenvalues)) or np.min(eigenvalues) <= 0.0:
            raise ValidationError(f"Part {part.name!r} inertia tensor must be positive definite")
        api = UsdPhysics.MassAPI.Apply(prim)
        # Keep the SDK/URDF inertia proxy, but let USD runtimes derive the center of mass and
        # principal inertia frame from the final authored collision shapes. Coarse generation
        # proxies are sufficient for mass estimation but can destabilize PhysX when forced into
        # the runtime mass frame.
        api.CreateMassAttr(float(inertial.mass))

    def _author_visual(self, visual: Visual, path: Sdf.Path) -> None:
        element = UsdGeom.Xform.Define(self.stage, path)
        _set_origin(element, visual.origin)
        material = self._resolve_visual_material(visual.material)
        self._author_geometry(
            visual.geometry,
            path.AppendChild("Geom"),
            material=material,
            collision=False,
        )

    def _author_collision(
        self,
        collision: Collision,
        part: Part,
        path: Sdf.Path,
        *,
        runtime_collision: bool,
    ) -> int:
        element = UsdGeom.Xform.Define(self.stage, path)
        _set_origin(element, collision.origin)
        element.MakeInvisible()
        body_model = _body_model(part)
        if not runtime_collision:
            element.GetPrim().SetCustomDataByKey("usdcraft:collisionRole", "deformable_contact")
            element.GetPrim().SetCustomDataByKey("usdcraft:collisionRealization", "rest_shape")
            element.GetPrim().SetCustomDataByKey("usdcraft:bodyModel", body_model.value)
            element.GetPrim().SetCustomDataByKey("usdcraft:runtimeCollisionEnabled", False)
        approximation = (
            collision.physics.approximation
            if collision.physics is not None
            else part.rigid_body.collision_approximation
            if part.rigid_body is not None
            else CollisionApproximation.CONVEX_DECOMPOSITION
        )
        material_name = (
            collision.physics.physics_material
            if collision.physics is not None and collision.physics.physics_material
            else part.rigid_body.physics_material
            if part.rigid_body is not None
            else None
        )
        material = self._physics_materials.get(material_name or "")
        if material_name and material is None:
            raise ValidationError(
                f"Part {part.name!r} references unknown physics material {material_name!r}"
            )
        prims = self._author_geometry(
            collision.geometry,
            path.AppendChild("Geom"),
            material=material,
            collision=True,
            author_physics_collision=runtime_collision,
            approximation=CollisionApproximation(
                str(cast(CollisionApproximation, approximation).value)
            ),
        )
        return len(prims)

    def _author_geometry(
        self,
        geometry: Geometry,
        path: Sdf.Path,
        *,
        material: UsdShade.Material | None,
        collision: bool,
        author_physics_collision: bool = False,
        approximation: CollisionApproximation = CollisionApproximation.CONVEX_DECOMPOSITION,
    ) -> list[Usd.Prim]:
        prims: list[Usd.Prim]
        if isinstance(geometry, Box):
            if collision:
                shape = UsdGeom.Cube.Define(self.stage, path)
                shape.CreateSizeAttr(1.0)
                shape.AddScaleOp().Set(Gf.Vec3f(*(float(v) for v in geometry.size)))
                prims = [shape.GetPrim()]
            else:
                prims = [
                    self._author_trimesh(
                        trimesh.creation.box(extents=geometry.size), path
                    ).GetPrim()
                ]
        elif isinstance(geometry, Cylinder):
            if collision:
                shape = UsdGeom.Cylinder.Define(self.stage, path)
                shape.CreateRadiusAttr(float(geometry.radius))
                shape.CreateHeightAttr(float(geometry.length))
                shape.CreateAxisAttr(UsdGeom.Tokens.z)
                prims = [shape.GetPrim()]
            else:
                prims = [
                    self._author_trimesh(
                        trimesh.creation.cylinder(
                            radius=geometry.radius, height=geometry.length, sections=64
                        ),
                        path,
                    ).GetPrim()
                ]
        elif isinstance(geometry, Sphere):
            if collision:
                shape = UsdGeom.Sphere.Define(self.stage, path)
                shape.CreateRadiusAttr(float(geometry.radius))
                prims = [shape.GetPrim()]
            else:
                prims = [
                    self._author_trimesh(
                        trimesh.creation.icosphere(subdivisions=3, radius=geometry.radius),
                        path,
                    ).GetPrim()
                ]
        elif isinstance(geometry, Mesh):
            prims = self._author_mesh_geometry(
                geometry,
                path,
                material=material,
                split_connected_components=collision,
                author_visual_materials=not collision,
            )
        else:
            raise ValidationError(f"Unsupported USD geometry type: {type(geometry).__name__}")

        for prim in prims:
            if collision:
                if author_physics_collision:
                    UsdPhysics.CollisionAPI.Apply(prim)
                    if prim.IsA(UsdGeom.Mesh):
                        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                        mesh_api.CreateApproximationAttr(approximation.value)
                UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
                UsdGeom.Imageable(prim).MakeInvisible()
                if material is not None:
                    _bind_material(prim, material, purpose="physics")
            elif material is not None:
                _bind_material(prim, material)
        return prims

    def _author_trimesh(self, mesh: trimesh.Trimesh, path: Sdf.Path) -> UsdGeom.Mesh:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        usd_mesh = UsdGeom.Mesh.Define(self.stage, path)
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        usd_mesh.CreatePointsAttr(
            [Gf.Vec3f(*(float(value) for value in vertex)) for vertex in vertices]
        )
        usd_mesh.CreateFaceVertexCountsAttr([int(len(face)) for face in faces])
        usd_mesh.CreateFaceVertexIndicesAttr([int(value) for value in faces.reshape(-1)])
        self._author_mesh_normals(usd_mesh, vertices, faces, label=path.name)
        return usd_mesh

    def _author_mesh_normals(
        self,
        usd_mesh: UsdGeom.Mesh,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        label: str,
        normal_crease_deg: float | None = None,
    ) -> None:
        try:
            crease = (
                _NORMAL_CREASE_ANGLE_DEGREES
                if normal_crease_deg is None
                else float(normal_crease_deg)
            )
            normals = _crease_aware_face_varying_normals(
                vertices,
                faces,
                crease_angle_degrees=crease,
            )
            usd_mesh.CreateNormalsAttr(
                [Gf.Vec3f(*(float(value) for value in normal)) for normal in normals]
            )
            usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        except Exception as exc:
            self.warnings.append(
                f"Mesh {label!r} shipped without corner normals (normal computation failed: {exc})"
            )

    def _author_mesh_geometry(
        self,
        geometry: Mesh,
        path: Sdf.Path,
        *,
        material: UsdShade.Material | None,
        split_connected_components: bool = False,
        author_visual_materials: bool = True,
    ) -> list[Usd.Prim]:
        source = resolve_mesh_path(require_mesh_filename(geometry), assets=self.asset_root)
        if not source.is_file():
            raise ValidationError(f"Mesh file not found for USD export: {source}")
        loaded = trimesh.load(source, force="scene", process=False)
        scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
        nodes = list(scene.graph.nodes_geometry)
        if not nodes:
            raise ValidationError(f"Mesh contains no geometry for USD export: {source}")
        scope = UsdGeom.Xform.Define(self.stage, path)
        prims: list[Usd.Prim] = []
        names = _NameScope()
        scale = np.array(geometry.scale or (1.0, 1.0, 1.0), dtype=float)
        for index, node_name in enumerate(nodes):
            transform, geometry_name = scene.graph.get(node_name)
            tri = scene.geometry[geometry_name].copy()
            tri.apply_transform(transform)
            # MeshCollisionAPI applies its approximation per authored Mesh prim.
            # Keep topologically disconnected islands separate so an explicit
            # convexHull can never bridge the usable gaps between components.
            components = (
                _split_collision_components(
                    tri, label=cast(str, geometry_name), warnings=self.warnings
                )
                if split_connected_components
                else [tri]
            )
            if not components:
                raise ValidationError(
                    f"Mesh contains no connected geometry for USD export: {source}"
                )
            for component_index, component in enumerate(components):
                vertices = np.asarray(component.vertices, dtype=float) * scale
                faces = np.asarray(component.faces, dtype=np.int64)
                component_name = (
                    geometry_name
                    if len(components) == 1
                    else f"{geometry_name}_component_{component_index:03d}"
                )
                mesh_path = scope.GetPath().AppendChild(
                    names.make(
                        component_name,
                        fallback=f"mesh_{index:03d}_component_{component_index:03d}",
                    )
                )
                usd_mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
                usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                usd_mesh.CreatePointsAttr(
                    [Gf.Vec3f(*(float(value) for value in vertex)) for vertex in vertices]
                )
                usd_mesh.CreateFaceVertexCountsAttr([int(len(face)) for face in faces])
                usd_mesh.CreateFaceVertexIndicesAttr([int(value) for value in faces.reshape(-1)])
                self._author_mesh_normals(
                    usd_mesh,
                    vertices,
                    faces,
                    label=mesh_path.name,
                    normal_crease_deg=geometry.normal_crease_deg,
                )
                uv = getattr(component.visual, "uv", None)
                if uv is not None and len(uv) == len(vertices):
                    primvars = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim())
                    st = primvars.CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
                    )
                    # USD uses a lower-left UV convention; glTF images use upper-left.
                    st.Set([Gf.Vec2f(float(item[0]), 1.0 - float(item[1])) for item in uv])
                if author_visual_materials:
                    bound_material = material or self._material_from_trimesh(component, source.stem)
                    if bound_material is not None:
                        _bind_material(usd_mesh.GetPrim(), bound_material)
                prims.append(usd_mesh.GetPrim())
        return prims

    def _material_from_trimesh(self, mesh: trimesh.Trimesh, label: str) -> UsdShade.Material | None:
        source_material = getattr(mesh.visual, "material", None)
        if source_material is None:
            return None
        image = getattr(source_material, "baseColorTexture", None)
        if image is None:
            image = getattr(source_material, "image", None)
        texture_asset = self._save_embedded_texture(image, label)
        color = getattr(source_material, "baseColorFactor", None)
        color_is_linear = color is not None
        if color is None:
            color = getattr(source_material, "main_color", None)
        rgba = None
        if color is not None and len(color) >= 3:
            values = [float(value) for value in color[:4]]
            if any(value > 1.0 for value in values):
                values = [value / 255.0 for value in values]
            if len(values) == 3:
                values.append(1.0)
            # glTF/PBR baseColorFactor is linear, whereas Material.rgba has an
            # sRGB public contract.  Encode it before passing through the
            # shared material authoring path so USD receives the original
            # linear factor rather than a second gamma decode.  Legacy
            # main_color values are display colors and already match Material.
            if color_is_linear:
                values[:3] = [_linear_channel_to_srgb(value) for value in values[:3]]
            rgba = tuple(values)
        metallic = _coerce_unit_scalar(getattr(source_material, "metallicFactor", None))
        roughness = _coerce_unit_scalar(getattr(source_material, "roughnessFactor", None))
        emissive_factor = getattr(source_material, "emissiveFactor", None)
        emissive: tuple[float, float, float] | None = None
        if emissive_factor is not None and len(emissive_factor) >= 3:
            emissive_values = [float(value) for value in emissive_factor[:3]]
            if any(value > 1.0 for value in emissive_values):
                emissive_values = [value / 255.0 for value in emissive_values]
            if any(value > 0.0 for value in emissive_values):
                emissive = cast(tuple[float, float, float], tuple(emissive_values))
        if texture_asset is None and rgba is None and metallic is None and roughness is None:
            return None
        self._embedded_material_index += 1
        name = f"EmbeddedMaterial_{self._embedded_material_index:03d}"
        alpha_mode_value = getattr(source_material, "alphaMode", None)
        alpha_mode = None if alpha_mode_value is None else str(alpha_mode_value).strip().upper()
        if alpha_mode not in {"OPAQUE", "MASK", "BLEND"}:
            alpha_mode = None
        alpha_cutoff = _coerce_unit_scalar(getattr(source_material, "alphaCutoff", None))
        material = Material(
            name=name,
            rgba=rgba or (1.0, 1.0, 1.0, 1.0),
            metallic=metallic,
            roughness=roughness,
            emissive=emissive,
        )
        authored = self._author_preview_material(
            material,
            self.looks_path.AppendChild(name),
            texture_asset=texture_asset,
            texture_alpha_mode=alpha_mode,
            texture_alpha_cutoff=alpha_cutoff,
        )
        return authored

    def _resolve_visual_material(
        self, reference: Material | str | None
    ) -> UsdShade.Material | None:
        if reference is None:
            return None
        if isinstance(reference, str):
            material = self._visual_materials.get(reference)
            if material is None:
                raise ValidationError(f"Unknown visual material for USD export: {reference!r}")
            return material
        registered_names = {material.name for material in self.model.materials}
        existing = self._visual_materials.get(reference.name)
        if existing is not None and reference.name in registered_names:
            return existing
        prim_name = Tf.MakeValidIdentifier(reference.name) or "InlineMaterial"
        path = self.looks_path.AppendChild(prim_name)
        if self.stage.GetPrimAtPath(path):
            suffix = 2
            while self.stage.GetPrimAtPath(self.looks_path.AppendChild(f"{prim_name}_{suffix}")):
                suffix += 1
            path = self.looks_path.AppendChild(f"{prim_name}_{suffix}")
        authored = self._author_preview_material(reference, path)
        return authored

    def _define_joints(self) -> None:
        names = _NameScope()
        for joint in self.model.articulations:
            joint_type = _joint_type(joint)
            if self._involves_nonrigid(joint):
                if joint_type != ArticulationType.FIXED:
                    raise ValidationError(
                        f"Articulation {joint.name!r} must be fixed when it involves a "
                        "non-rigid part"
                    )
                self._preserve_static_attachment(joint)
                continue
            if joint_type == ArticulationType.FLOATING:
                # FLOATING is an authoring-tree placement edge. Its absence from the USD joint
                # graph is intentional: the rigid bodies remain independently simulated.
                continue
            path = self.joints_path.AppendChild(names.make(joint.name, fallback="Joint"))
            self._joint_paths[joint.name] = path
            if joint_type == ArticulationType.FIXED:
                usd_joint: Any = UsdPhysics.FixedJoint.Define(self.stage, path)
                axis_rotation = Rotation.identity()
            elif joint_type in {ArticulationType.REVOLUTE, ArticulationType.CONTINUOUS}:
                usd_joint = UsdPhysics.RevoluteJoint.Define(self.stage, path)
                usd_joint.CreateAxisAttr(UsdPhysics.Tokens.x)
                axis_rotation = _rotation_x_to_axis(joint.axis)
                if joint_type == ArticulationType.REVOLUTE and joint.motion_limits is not None:
                    if joint.motion_limits.lower is not None:
                        usd_joint.CreateLowerLimitAttr(math.degrees(joint.motion_limits.lower))
                    if joint.motion_limits.upper is not None:
                        usd_joint.CreateUpperLimitAttr(math.degrees(joint.motion_limits.upper))
            elif joint_type == ArticulationType.PRISMATIC:
                usd_joint = UsdPhysics.PrismaticJoint.Define(self.stage, path)
                usd_joint.CreateAxisAttr(UsdPhysics.Tokens.x)
                axis_rotation = _rotation_x_to_axis(joint.axis)
                if joint.motion_limits is not None:
                    if joint.motion_limits.lower is not None:
                        usd_joint.CreateLowerLimitAttr(float(joint.motion_limits.lower))
                    if joint.motion_limits.upper is not None:
                        usd_joint.CreateUpperLimitAttr(float(joint.motion_limits.upper))
            elif joint_type == ArticulationType.SPHERICAL:
                if joint.cone_deg is None:
                    raise ValidationError(
                        f"Spherical articulation {joint.name!r} must set cone_deg"
                    )
                usd_joint = UsdPhysics.SphericalJoint.Define(self.stage, path)
                usd_joint.CreateAxisAttr(UsdPhysics.Tokens.x)
                usd_joint.CreateConeAngle0LimitAttr(float(joint.cone_deg))
                usd_joint.CreateConeAngle1LimitAttr(float(joint.cone_deg))
                axis_rotation = _rotation_x_to_axis(joint.axis)
            else:
                raise ValidationError(f"Unsupported articulation type for USD: {joint_type}")

            parent_path = self._part_paths[joint.parent]
            child_path = self._part_paths[joint.child]
            # Joint-connected bodies share an attachment frame. Keep their collision
            # disabled explicitly instead of relying on the schema fallback, which
            # avoids adjacent-link contact churn in segmented chains.
            usd_joint.CreateCollisionEnabledAttr(False)
            usd_joint.CreateBody0Rel().SetTargets([parent_path])
            usd_joint.CreateBody1Rel().SetTargets([child_path])
            origin_rotation = Rotation.from_euler("xyz", joint.origin.rpy)
            usd_joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(value) for value in joint.origin.xyz)))
            usd_joint.CreateLocalRot0Attr(_gf_quatf(origin_rotation * axis_rotation))
            usd_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            usd_joint.CreateLocalRot1Attr(_gf_quatf(axis_rotation))
            usd_joint.GetPrim().SetCustomDataByKey("usdcraft:jointName", joint.name)
            usd_joint.GetPrim().SetCustomDataByKey("usdcraft:parentPart", joint.parent)
            usd_joint.GetPrim().SetCustomDataByKey("usdcraft:childPart", joint.child)
            if joint_type == ArticulationType.SPHERICAL:
                normalized_axis = np.asarray(joint.axis, dtype=float)
                normalized_axis /= np.linalg.norm(normalized_axis)
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:articulationType", ArticulationType.SPHERICAL.value
                )
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:axis",
                    tuple(float(value) for value in normalized_axis),
                )
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:coneDeg", float(cast(float, joint.cone_deg))
                )
            if joint.motion_limits is not None:
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:effortLimit", float(joint.motion_limits.effort)
                )
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:velocityLimit", float(joint.motion_limits.velocity)
                )
            if joint.motion_properties is not None:
                motion_properties = {
                    key: float(value)
                    for key, value in {
                        "damping": joint.motion_properties.damping,
                        "friction": joint.motion_properties.friction,
                    }.items()
                    if value is not None
                }
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:motionProperties",
                    motion_properties,
                )
            resistance = joint.physx_joint_resistance
            if isinstance(resistance, PhysxJointFriction):
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:physxJointResistance",
                    {
                        "model": "effort",
                        "staticFrictionEffort": float(resistance.static_friction_effort),
                        "dynamicFrictionEffort": float(resistance.dynamic_friction_effort),
                        "viscousFrictionCoefficient": float(
                            resistance.viscous_friction_coefficient
                        ),
                    },
                )
            elif isinstance(resistance, PhysxJointFrictionCoefficient):
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:physxJointResistance",
                    {
                        "model": "load_proportional",
                        "coefficient": float(resistance.coefficient),
                    },
                )
            if joint.drive is not None:
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:jointDrive",
                    {
                        "mode": joint.drive.mode,
                        "stiffness": float(joint.drive.stiffness),
                        "damping": float(joint.drive.damping),
                    },
                )
            if joint.newton_joint_solver_tuning is not None:
                tuning = joint.newton_joint_solver_tuning
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:newtonJointSolverTuning",
                    {
                        key: Vt.DoubleArray([float(component) for component in value])
                        for key, value in {
                            "solrefFriction": tuning.solref_friction,
                            "solimpFriction": tuning.solimp_friction,
                        }.items()
                        if value is not None
                    },
                )
            if joint.passive_spring is not None:
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:passiveSpring",
                    {
                        "stiffness": float(joint.passive_spring.stiffness),
                        "reference": float(joint.passive_spring.reference),
                    },
                )
            if joint.mimic is not None:
                usd_joint.GetPrim().SetCustomDataByKey(
                    "usdcraft:mimic",
                    {
                        "joint": joint.mimic.joint,
                        "multiplier": joint.mimic.multiplier,
                        "offset": joint.mimic.offset,
                    },
                )
                self.backend_notes.add("mimic", joint.name)
            self._author_drive(usd_joint.GetPrim(), joint, joint_type)

    def _involves_nonrigid(self, joint: Articulation) -> bool:
        return any(
            _body_model(self._parts_by_name[part_name]) != BodyModel.RIGID
            for part_name in (joint.parent, joint.child)
        )

    def _preserve_static_attachment(self, joint: Articulation) -> None:
        child_prim = self.stage.GetPrimAtPath(self._part_paths[joint.child])
        child_prim.SetCustomDataByKey(
            "usdcraft:attachment",
            {
                "articulation": joint.name,
                "parent": joint.parent,
                "child": joint.child,
                "type": ArticulationType.FIXED.value,
                "realization": "static_rest_pose",
            },
        )
        self.backend_notes.add("non_rigid_attachment", joint.name)

    def _author_drive(
        self,
        prim: Usd.Prim,
        joint: Articulation,
        joint_type: ArticulationType,
    ) -> None:
        drive = joint.drive
        passive_damping = (
            joint.motion_properties.damping if joint.motion_properties is not None else None
        )
        effort_limit = joint.motion_limits.effort if joint.motion_limits is not None else None
        passive_spring = joint.passive_spring
        if (
            drive is None
            and passive_damping is None
            and effort_limit is None
            and passive_spring is None
        ):
            return
        if joint_type not in {
            ArticulationType.REVOLUTE,
            ArticulationType.CONTINUOUS,
            ArticulationType.PRISMATIC,
        }:
            raise ValidationError(f"Joint drive requires a scalar articulation: {joint.name!r}")
        angular = joint_type in {ArticulationType.REVOLUTE, ArticulationType.CONTINUOUS}
        api = UsdPhysics.DriveAPI.Apply(
            prim, UsdPhysics.Tokens.angular if angular else UsdPhysics.Tokens.linear
        )
        api.CreateTypeAttr(drive.mode if drive is not None else "force")
        stiffness = float(
            drive.stiffness
            if drive is not None
            else passive_spring.stiffness
            if passive_spring is not None
            else 0.0
        )
        damping = float(drive.damping if drive is not None else passive_damping or 0.0)
        if angular:
            # SDK angular gains are torque/radian and torque/(radian/second).
            # UsdPhysics angular drive errors are expressed in degrees, so the
            # coefficients must shrink by pi/180 to preserve the same torque.
            stiffness *= math.pi / 180.0
            damping *= math.pi / 180.0
        api.CreateStiffnessAttr(stiffness)
        api.CreateDampingAttr(damping)
        max_force = drive.max_force if drive is not None else None
        if max_force is None:
            max_force = effort_limit
        if max_force is not None:
            api.CreateMaxForceAttr(float(max_force))
        if drive is None:
            if passive_spring is not None:
                value = (
                    math.degrees(passive_spring.reference) if angular else passive_spring.reference
                )
                api.CreateTargetPositionAttr(float(value))
            return
        if drive.target_position is not None:
            value = math.degrees(drive.target_position) if angular else drive.target_position
            api.CreateTargetPositionAttr(float(value))
        if drive.target_velocity is not None:
            value = math.degrees(drive.target_velocity) if angular else drive.target_velocity
            api.CreateTargetVelocityAttr(float(value))

    def _define_fixed_base_joint_if_needed(self) -> None:
        if _base_mode(self.model.base_mode) != BaseMode.FIXED:
            return
        roots = self.model.root_parts()
        if len(roots) != 1:
            raise ValidationError("Fixed-base USD export requires exactly one root part")
        root_part = roots[0]
        if _body_model(root_part) != BodyModel.RIGID:
            root_prim = self.stage.GetPrimAtPath(self._part_paths[root_part.name])
            root_prim.SetCustomDataByKey(
                "usdcraft:attachment",
                {
                    "parent": "world",
                    "child": root_part.name,
                    "type": ArticulationType.FIXED.value,
                    "realization": "static_rest_pose",
                },
            )
            self.warnings.append(
                f"Fixed base for non-rigid root part {root_part.name!r} was preserved as "
                "usdcraft attachment customData; no native physics joint was authored"
            )
            return
        path = self.joints_path.AppendChild("__root_fixed")
        joint = UsdPhysics.FixedJoint.Define(self.stage, path)
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateBody1Rel().SetTargets([self._part_paths[root_part.name]])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.GetPrim().SetCustomDataByKey("usdcraft:jointName", "__root_fixed")
        joint.GetPrim().SetCustomDataByKey("usdcraft:parentPart", "world")
        joint.GetPrim().SetCustomDataByKey("usdcraft:childPart", root_part.name)


def export_object_to_usd(
    object_model: ArticulatedObject,
    output_dir: str | os.PathLike[str] | Path,
    *,
    asset_root: str | os.PathLike[str] | Path | None = None,
    include_physical_collisions: bool = True,
    validate: bool = True,
) -> UsdExportResult:
    """Export reusable PhysX and Newton assets from one Asset Structure 3.0 stack.

    ``isaac/model.usdc`` remains the backward-compatible, flattened PhysX
    entrypoint. ``isaac/model_newton.usdc`` is an independently openable
    Newton/MuJoCo entrypoint. Their authoring sources are the engine-neutral
    ``physics.usdc`` layer plus thin ``physx.usda`` and ``mujoco.usda``
    overlays. ``asset.usda`` exposes the same choices through a Physics
    variant set for inspection and future tuning.
    """

    if validate:
        object_model.validate(strict=True)
    resolved_root = resolve_asset_root(asset_root, object_model)
    if resolved_root is None:
        resolved_root = Path(".").resolve()
    compiled_model = (
        compile_object_model_with_exact_collisions(
            object_model,
            asset_root=resolved_root,
            validate=False,
        )
        if include_physical_collisions
        else object_model
    )
    destination = Path(output_dir).resolve()
    isaac_dir = destination / "isaac"
    isaac_dir.mkdir(parents=True, exist_ok=True)
    physics_path = isaac_dir / _PHYSICS_LAYER_NAME
    physx_layer_path = isaac_dir / _PHYSX_LAYER_NAME
    newton_mjc_layer_path = isaac_dir / _NEWTON_MJC_LAYER_NAME
    interface_path = isaac_dir / _ASSET_INTERFACE_NAME
    usdc_path = isaac_dir / _PHYSX_ENTRY_NAME
    newton_usdc_path = isaac_dir / _NEWTON_ENTRY_NAME
    usdz_path = destination / "model.usdz"
    for stale in (
        physics_path,
        physx_layer_path,
        newton_mjc_layer_path,
        interface_path,
        usdc_path,
        newton_usdc_path,
        usdz_path,
    ):
        if stale.exists():
            stale.unlink()

    stage = Usd.Stage.CreateNew(physics_path.as_posix())
    exporter = _UsdObjectExporter(
        compiled_model,
        stage=stage,
        output_dir=destination,
        asset_root=Path(resolved_root),
    )
    metrics = exporter.export()
    metrics["collision_geometry_derivation_enabled"] = bool(include_physical_collisions)
    stage.GetRootLayer().Save()
    _validate_usd_asset(physics_path, backend="generic")

    physx_warnings = _author_physx_layer(
        compiled_model,
        physx_layer_path,
        physics_path,
        exporter._joint_paths,
    )
    newton_warnings = _author_newton_mjc_layer(
        compiled_model,
        newton_mjc_layer_path,
        physics_path,
        exporter._joint_paths,
        notes=exporter.backend_notes,
    )
    _author_asset_interface(
        interface_path,
        physics_path=physics_path,
        physx_path=physx_layer_path,
        newton_mjc_path=newton_mjc_layer_path,
    )

    # Distribution entrypoints are deliberately flattened. This keeps the
    # direct-file contract for PhysX and prevents a copied Newton entry from
    # becoming invalid when its sidecars are omitted. The modular source layers
    # remain alongside them for Asset Structure 3.0 inspection and tuning.
    _flatten_layer_stack(physx_layer_path, usdc_path)
    _flatten_layer_stack(newton_mjc_layer_path, newton_usdc_path)
    physx_checks = _validate_usd_asset(usdc_path, backend="physx")
    newton_checks = _validate_usd_asset(newton_usdc_path, backend="newton")
    _validate_usd_asset(interface_path, backend="physx")
    _validate_asset_interface_variants(interface_path)

    with _filtered_native_stderr(_USDZ_PACKAGING_NOISE):
        packaged = UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(usdc_path.as_posix()), usdz_path.as_posix()
        )
    if not packaged:
        raise RuntimeError(f"Failed to create USDZ package: {usdz_path}")
    _validate_usd_asset(usdz_path, backend="physx")

    metrics["asset_structure"] = {
        "version": "3.0",
        "profile": "minimal_shared_physics",
        "entrypoints_flattened": True,
        "interface_usd": "isaac/asset.usda",
        "layers": {
            "physics": "isaac/physics.usdc",
            "physx": "isaac/physx.usda",
            "newton_mjc": "isaac/mujoco.usda",
        },
    }
    metrics["backend_validation"] = {
        "physx": {
            "status": "success",
            "checks": list(physx_checks),
            "runtime_status": "not_run",
        },
        "newton": {
            "status": "success",
            "checks": list(newton_checks),
            "runtime_status": "not_run",
        },
    }
    return UsdExportResult(
        usdc_path=usdc_path,
        usdz_path=usdz_path,
        warnings=tuple(exporter.warnings + physx_warnings + newton_warnings),
        metrics=metrics,
        newton_usdc_path=newton_usdc_path,
        notes=tuple(exporter.backend_notes.render()),
        asset_structure_paths={
            "interface": interface_path,
            "physics": physics_path,
            "physx": physx_layer_path,
            "newton_mjc": newton_mjc_layer_path,
        },
    )


# ``UsdUtils.CreateNewUsdzPackage`` relocates texture dependencies into numbered package
# directories (``0/decal_x.png``) and, while rewriting the references, OpenUSD prints one
# ``_EnqueueDependency ... Failed to resolve reference @0/...@`` warning per texture to the
# process stderr. The package is complete and validated afterwards, so that native noise is
# filtered out of stderr; every other native line is re-emitted unchanged.
_USDZ_PACKAGING_NOISE = re.compile(r"_EnqueueDependency|Failed to resolve reference @\d+/")


@contextlib.contextmanager
def _filtered_native_stderr(noise: re.Pattern[str]):
    """Capture fd 2 for the block and re-emit only lines that do not match ``noise``."""

    try:
        sys.stderr.flush()
        saved_fd = os.dup(2)
    except (OSError, ValueError):
        yield
        return
    with tempfile.TemporaryFile(mode="w+b") as capture:
        os.dup2(capture.fileno(), 2)
        try:
            yield
        finally:
            try:
                sys.stderr.flush()
            except (OSError, ValueError):
                pass
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            capture.seek(0)
            text = capture.read().decode("utf-8", errors="replace")
            kept = [line for line in text.splitlines() if line and not noise.search(line)]
            if kept:
                os.write(2, ("\n".join(kept) + "\n").encode("utf-8"))


# CAD tessellation (cadquery/OCC fillets and chamfers) leaves zero-area sliver triangles that are
# not stitched to their neighbours. Splitting a collision mesh into connected components turns
# each sliver into its own single-triangle collider, which the Newton backend rejects as planar.
# Slivers carry no collision volume, so they are dropped before authoring; anything larger stays
# and is reported by the backend validators.
_COLLIDER_SLIVER_MAX_EXTENT_M = 0.005
_COLLIDER_SLIVER_MAX_AREA_M2 = 1e-5
_COLLIDER_PLANAR_RANK_TOL = 1e-9


def _is_collider_sliver(component: trimesh.Trimesh) -> bool:
    """True for a component too small and flat to bound any collision volume."""

    vertices = np.asarray(component.vertices, dtype=float)
    if len(vertices) == 0:
        return True
    extent = float(np.max(np.ptp(vertices, axis=0))) if len(vertices) > 1 else 0.0
    area = float(component.area) if len(component.faces) else 0.0
    if len(vertices) < 4:
        return extent <= _COLLIDER_SLIVER_MAX_EXTENT_M or area <= _COLLIDER_SLIVER_MAX_AREA_M2
    centered = vertices - vertices.mean(axis=0)
    planar = np.linalg.matrix_rank(centered, tol=_COLLIDER_PLANAR_RANK_TOL) < 3
    return planar and (
        extent <= _COLLIDER_SLIVER_MAX_EXTENT_M or area <= _COLLIDER_SLIVER_MAX_AREA_M2
    )


def _split_collision_components(
    tri: trimesh.Trimesh, *, label: str, warnings: list[str]
) -> list[trimesh.Trimesh]:
    """Split a collision mesh into connected components after stitching shared vertices,
    dropping tessellation slivers that cannot bound volume."""

    stitched = tri.copy()
    stitched.merge_vertices()
    components = list(stitched.split(only_watertight=False))
    if not components:
        return []
    kept = [component for component in components if not _is_collider_sliver(component)]
    dropped = len(components) - len(kept)
    if dropped and kept:
        warnings.append(
            f"Collision mesh {label!r}: dropped {dropped} zero-volume sliver component(s) left "
            "by tessellation; they carry no collision volume."
        )
        return kept
    return components


def _flatten_layer_stack(source: Path, destination: Path) -> None:
    stage = Usd.Stage.Open(source.as_posix())
    if stage is None:
        raise ValidationError(f"USD layer stack cannot be opened: {source}")
    # All Asset Structure layers intentionally live beside isaac/textures.
    # Preserve their authored relative paths instead of allowing flattening to
    # rewrite them to machine-local absolute paths.
    flattened = UsdUtils.FlattenLayerStack(stage, lambda _layer, asset_path: asset_path)
    if not flattened.Export(destination.as_posix()):
        raise RuntimeError(f"Failed to flatten USD layer stack to {destination}")


def _bind_material(
    prim: Usd.Prim,
    material: UsdShade.Material,
    *,
    purpose: str | None = None,
) -> None:
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    if purpose is None:
        binding.Bind(material)
    else:
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, purpose)


def _set_origin(xform: UsdGeom.Xform, origin: Origin) -> None:
    if any(abs(float(value)) > 1e-15 for value in origin.xyz):
        xform.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in origin.xyz)))
    if any(abs(float(value)) > 1e-15 for value in origin.rpy):
        xform.AddOrientOp().Set(_gf_quatf(Rotation.from_euler("xyz", origin.rpy)))


def _set_world_transform(xform: UsdGeom.Xform, matrix: np.ndarray) -> None:
    translation = matrix[:3, 3]
    rotation = Rotation.from_matrix(matrix[:3, :3])
    if np.linalg.norm(translation) > 1e-15:
        xform.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in translation)))
    if rotation.magnitude() > 1e-15:
        xform.AddOrientOp().Set(_gf_quatf(rotation))


def _coerce_unit_scalar(value: Any) -> float | None:
    """Best-effort convert a glTF PBR factor to a clamped [0, 1] float.

    Returns ``None`` when the value is absent or not a finite number so that an
    unusual source material never aborts the whole export.
    """
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return float(min(1.0, max(0.0, scalar)))


def _origin_matrix(origin: Origin) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = Rotation.from_euler("xyz", origin.rpy).as_matrix()
    result[:3, 3] = np.asarray(origin.xyz, dtype=float)
    return result


def _rotation_x_to_axis(axis: tuple[float, float, float]) -> Rotation:
    target = np.asarray(axis, dtype=float)
    magnitude = float(np.linalg.norm(target))
    if magnitude <= 1e-12:
        raise ValidationError("Articulation axis must be non-zero")
    target /= magnitude
    source = np.array([1.0, 0.0, 0.0], dtype=float)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return Rotation.identity()
    if dot < -1.0 + 1e-12:
        return Rotation.from_rotvec(np.array([0.0, 0.0, math.pi]))
    cross = np.cross(source, target)
    cross /= np.linalg.norm(cross)
    return Rotation.from_rotvec(cross * math.acos(dot))


def _gf_quatf(rotation: Rotation) -> Gf.Quatf:
    x, y, z, w = rotation.as_quat()
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _base_mode(value: BaseMode | str) -> BaseMode:
    return value if isinstance(value, BaseMode) else BaseMode(str(value))


def _body_model(part: Part) -> BodyModel:
    value = part.body_model
    return value if isinstance(value, BodyModel) else BodyModel(str(value))


def _unique_filename(name: str, used: set[str]) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(name).name) or "texture.png"
    stem = Path(safe).stem
    suffix = Path(safe).suffix
    candidate = safe
    index = 2
    while candidate in used:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


__all__ = ["UsdExportResult", "export_object_to_usd"]
