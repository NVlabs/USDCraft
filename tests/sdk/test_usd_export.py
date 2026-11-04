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
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

import sdk._core._usd_common as usd_common
import sdk._core._usd_export as usd_export
import sdk._core._usd_validate as usd_validate
from sdk import (
    ArticulatedObject,
    ArticulationType,
    BaseMode,
    Box,
    CollisionApproximation,
    CollisionProperties,
    Cylinder,
    Inertia,
    Inertial,
    JointDrive,
    Mesh,
    Mimic,
    MotionLimits,
    MotionProperties,
    NewtonJointSolverTuning,
    Origin,
    Part,
    PassiveJointSpring,
    PhysxJointFriction,
    PhysxJointFrictionCoefficient,
    RigidBodyProperties,
    Sphere,
    TextureCanvas,
    ValidationError,
)
from sdk._core._usd_export import export_object_to_usd
from sdk._core.assets import AssetSession, activate_asset_session
from sdk._core.mesh import BoxGeometry, mesh_from_geometry


def _physics_model() -> ArticulatedObject:
    model = ArticulatedObject("physics_asset", base_mode=BaseMode.FIXED)
    visual_material = model.material(
        "paint",
        rgba=(0.8, 0.15, 0.05, 1.0),
        metallic=0.2,
        roughness=0.55,
        clearcoat=0.1,
    )
    model.physics_material(
        "rubber",
        static_friction=0.9,
        dynamic_friction=0.75,
        restitution=0.1,
        density=1100.0,
    )
    body = model.part(
        "main body",
        inertial=Inertial.from_geometry(Box((1.0, 0.6, 0.4)), mass=12.0),
        rigid_body=RigidBodyProperties(physics_material="rubber"),
    )
    body.visual(Box((1.0, 0.6, 0.4)), material=visual_material, name="shell")
    door = model.part(
        "door",
        inertial=Inertial.from_geometry(Box((0.08, 0.5, 0.35)), mass=1.5),
        rigid_body=RigidBodyProperties(physics_material="rubber"),
    )
    door.visual(Box((0.08, 0.5, 0.35)), material=visual_material, name="panel")
    model.articulation(
        "door hinge",
        ArticulationType.REVOLUTE,
        body,
        door,
        origin=Origin(xyz=(0.5, 0.0, 0.0)),
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(lower=0.0, upper=math.pi / 2, effort=20.0, velocity=2.0),
        motion_properties=MotionProperties(friction=0.12),
        drive=JointDrive(
            stiffness=8.0,
            damping=0.8,
            max_force=20.0,
            target_position=math.pi / 4,
        ),
    )
    return model


def _shelf_with_independent_goods(*, fixed_goods: bool = False) -> ArticulatedObject:
    model = ArticulatedObject("shelf_with_independent_goods", base_mode=BaseMode.FIXED)
    model.physics_material(
        "rack_contact",
        static_friction=0.7,
        dynamic_friction=0.5,
        restitution=0.0,
        density=7800.0,
    )
    model.physics_material(
        "product_contact",
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.05,
        density=900.0,
    )
    shelf_geometry = Box((1.0, 0.5, 0.1))
    shelf = model.part(
        "shelf",
        inertial=Inertial.from_geometry(
            shelf_geometry,
            mass=5.0,
            origin=Origin(xyz=(0.0, 0.0, 0.5)),
        ),
        rigid_body=RigidBodyProperties(physics_material="rack_contact"),
    )
    shelf.visual(
        shelf_geometry,
        origin=Origin(xyz=(0.0, 0.0, 0.5)),
        name="shelf_slab",
    )

    relation_type = ArticulationType.FIXED if fixed_goods else ArticulationType.FLOATING
    product_geometry = Box((0.1, 0.1, 0.2))
    for index, x in enumerate((-0.2, 0.2), start=1):
        product = model.part(
            f"product_{index}",
            inertial=Inertial.from_geometry(
                product_geometry,
                mass=0.25,
                origin=Origin(xyz=(0.0, 0.0, 0.1)),
            ),
            rigid_body=RigidBodyProperties(physics_material="product_contact"),
        )
        product.visual(
            product_geometry,
            origin=Origin(xyz=(0.0, 0.0, 0.1)),
            name="body",
        )
        model.articulation(
            f"place_product_{index}",
            relation_type,
            parent=shelf,
            child=product,
            origin=Origin(xyz=(x, 0.0, 0.55)),
        )
    return model


def test_usd_export_contains_pbr_and_object_physics(tmp_path: Path) -> None:
    result = export_object_to_usd(_physics_model(), tmp_path)

    assert result.usdc_path == tmp_path / "isaac" / "model.usdc"
    assert result.usdz_path == tmp_path / "model.usdz"
    assert result.usdc_path.is_file()
    assert result.usdz_path.is_file()

    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    assert stage
    assert stage.GetDefaultPrim().GetPath().pathString == "/Asset"
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    assert UsdPhysics.GetStageKilogramsPerUnit(stage) == pytest.approx(1.0)
    assert not any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse())
    root = stage.GetPrimAtPath("/Asset")
    assert root.GetCustomDataByKey("usdcraft:objectName") == "physics_asset"
    assert root.GetCustomDataByKey("usdcraft:baseMode") == "fixed"
    assert root.GetCustomDataByKey("mesh2usd:objectName") is None

    body = stage.GetPrimAtPath("/Asset/Links/main_body")
    assert body.GetCustomDataByKey("usdcraft:partName") == "main body"
    assert body.GetCustomDataByKey("mesh2usd:partName") is None
    assert body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert body.HasAPI(UsdPhysics.MassAPI)
    mass = UsdPhysics.MassAPI(body)
    assert mass.GetMassAttr().HasAuthoredValueOpinion()
    assert mass.GetMassAttr().Get() == pytest.approx(12.0)
    assert not mass.GetCenterOfMassAttr().HasAuthoredValueOpinion()
    assert not mass.GetDiagonalInertiaAttr().HasAuthoredValueOpinion()
    assert tuple(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx((0.0, 0.0, 0.0))
    assert not mass.GetPrincipalAxesAttr().HasAuthoredValueOpinion()

    visual = stage.GetPrimAtPath("/Asset/Links/main_body/Visuals/shell/Geom")
    assert visual.IsA(UsdGeom.Mesh)
    material, _ = UsdShade.MaterialBindingAPI(visual).ComputeBoundMaterial()
    assert material
    shader = UsdShade.Shader.Get(stage, "/Asset/Looks/paint/PreviewSurface")
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("metallic").Get() == pytest.approx(0.2)
    assert shader.GetInput("roughness").Get() == pytest.approx(0.55)

    collider = stage.GetPrimAtPath("/Asset/Links/main_body/Collisions/shell/Geom")
    assert collider.HasAPI(UsdPhysics.CollisionAPI)
    physics_material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
    assert physics_material.GetPath().pathString == "/Asset/PhysicsMaterials/rubber"
    physics_api = UsdPhysics.MaterialAPI(physics_material.GetPrim())
    assert physics_api.GetStaticFrictionAttr().Get() == pytest.approx(0.9)
    assert physics_api.GetDensityAttr().Get() == pytest.approx(1100.0)

    joint = UsdPhysics.RevoluteJoint.Get(stage, "/Asset/Joints/door_hinge")
    assert joint
    assert joint.GetAxisAttr().Get() == UsdPhysics.Tokens.x
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(0.0)
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(90.0)
    assert joint.GetPrim().GetCustomDataByKey("usdcraft:jointName") == "door hinge"
    assert joint.GetPrim().GetCustomDataByKey("usdcraft:effortLimit") == pytest.approx(20.0)
    assert joint.GetPrim().GetCustomDataByKey("usdcraft:velocityLimit") == pytest.approx(2.0)
    assert joint.GetPrim().GetCustomDataByKey("mesh2usd:jointName") is None
    drive = UsdPhysics.DriveAPI.Get(stage, "/Asset/Joints/door_hinge.drive:angular")
    assert drive
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(45.0)
    assert drive.GetStiffnessAttr().Get() == pytest.approx(8.0 * math.pi / 180.0)
    assert drive.GetDampingAttr().Get() == pytest.approx(0.8 * math.pi / 180.0)
    assert drive.GetMaxForceAttr().Get() == pytest.approx(20.0)
    assert joint.GetPrim().GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(
        math.degrees(2.0)
    )
    assert joint.GetPrim().GetAttribute("physxJoint:jointFriction").Get() == pytest.approx(0.12)
    api_schemas = (
        stage.GetRootLayer().GetPrimAtPath("/Asset/Joints/door_hinge").GetInfo("apiSchemas")
    )
    assert "PhysicsDriveAPI:angular" in api_schemas.prependedItems
    assert "PhysxJointAPI" in api_schemas.prependedItems
    velocity_spec = stage.GetRootLayer().GetPropertyAtPath(
        "/Asset/Joints/door_hinge.physxJoint:maxJointVelocity"
    )
    friction_spec = stage.GetRootLayer().GetPropertyAtPath(
        "/Asset/Joints/door_hinge.physxJoint:jointFriction"
    )
    assert velocity_spec is not None and velocity_spec.custom is False
    assert friction_spec is not None and friction_spec.custom is False
    usda = stage.GetRootLayer().ExportToString()
    assert "float physxJoint:maxJointVelocity" in usda
    assert "float physxJoint:jointFriction" in usda
    assert "custom float physxJoint:" not in usda
    fixed_root_joint = stage.GetPrimAtPath("/Asset/Joints/__root_fixed")
    assert fixed_root_joint.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath("/Asset").HasAPI(UsdPhysics.ArticulationRootAPI)

    packaged = Usd.Stage.Open(result.usdz_path.as_posix())
    assert packaged
    assert packaged.GetDefaultPrim().GetPath().pathString == "/Asset"


def test_usd_export_maps_effort_friction_and_passive_damping_to_newton(
    tmp_path: Path,
) -> None:
    model = _physics_model()
    model.articulations[0].motion_properties = MotionProperties(damping=0.6, friction=0.12)
    model.articulations[0].physx_joint_resistance = PhysxJointFriction(
        static_friction_effort=2.5,
        dynamic_friction_effort=2.3,
        viscous_friction_coefficient=0.42,
    )
    model.articulations[0].physx_armature = 0.03

    result = export_object_to_usd(model, tmp_path)

    physx = Usd.Stage.Open(result.usdc_path.as_posix())
    joint = physx.GetPrimAtPath("/Asset/Joints/door_hinge")
    schemas = usd_validate._authored_schema_tokens(joint)
    assert "PhysxJointAxisAPI:angular" in schemas
    assert "PhysxJointAPI" not in schemas
    namespace = "physxJointAxis:angular"
    assert joint.GetAttribute(f"{namespace}:staticFrictionEffort").Get() == pytest.approx(2.5)
    assert joint.GetAttribute(f"{namespace}:dynamicFrictionEffort").Get() == pytest.approx(2.3)
    assert joint.GetAttribute(f"{namespace}:viscousFrictionCoefficient").Get() == pytest.approx(
        0.42 * math.pi / 180.0
    )
    assert joint.GetAttribute(f"{namespace}:maxJointVelocity").Get() == pytest.approx(
        math.degrees(2.0)
    )
    assert joint.GetAttribute(f"{namespace}:armature").Get() == pytest.approx(0.03)
    assert not joint.GetAttribute("physxJoint:jointFriction")

    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    newton_joint = newton.GetPrimAtPath("/Asset/Joints/door_hinge")
    assert newton_joint.GetAttribute("mjc:frictionloss").Get() == pytest.approx(2.5)
    assert newton_joint.GetAttribute("mjc:damping").Get() == pytest.approx(1.02)
    assert not newton_joint.GetAttribute("mjc:solreffriction")
    assert not newton_joint.GetAttribute("mjc:solimpfriction")
    assert newton_joint.GetCustomDataByKey("usdcraft:newtonFrictionMapping") == {
        "source": "absolute_effort",
        "staticFrictionEffort": 2.5,
        "dynamicFrictionEffort": 2.3,
        "viscousFrictionCoefficient": 0.42,
        "frictionLoss": 2.5,
    }
    assert newton_joint.GetCustomDataByKey("usdcraft:newtonPassiveDamping") == pytest.approx(1.02)
    assert not any(name.startswith("physxJointAxis:") for name in newton_joint.GetPropertyNames())
    assert not newton_joint.GetAttribute("mjc:armature")
    assert newton_joint.GetCustomDataByKey("usdcraft:physxJointResistance") == {
        "model": "effort",
        "staticFrictionEffort": 2.5,
        "dynamicFrictionEffort": 2.3,
        "viscousFrictionCoefficient": 0.42,
    }
    assert newton_joint.GetCustomDataByKey("usdcraft:physxArmature") is None
    # Backend-mapping facts are aggregated informational notes, never per-joint warnings.
    assert not any("frictionloss" in warning for warning in result.warnings)
    (friction_note,) = [note for note in result.notes if "mjc:frictionloss" in note]
    assert friction_note.startswith("Newton/MuJoCo keeps one mjc:frictionloss threshold per joint")
    assert friction_note.endswith("as static/dynamic: door hinge (2.5/2.3)")


def test_usd_export_authors_prismatic_armature_on_linear_axis_only(tmp_path: Path) -> None:
    model = _physics_model()
    joint = model.articulations[0]
    joint.articulation_type = ArticulationType.PRISMATIC
    joint.motion_limits = MotionLimits(lower=-0.1, upper=0.1, effort=20.0, velocity=0.5)
    joint.motion_properties = MotionProperties(damping=0.2, friction=0.0)
    joint.physx_joint_resistance = PhysxJointFriction(
        static_friction_effort=4.0,
        dynamic_friction_effort=3.0,
        viscous_friction_coefficient=0.1,
    )
    joint.physx_armature = 0.015
    joint.drive = None

    result = export_object_to_usd(model, tmp_path)

    physx = Usd.Stage.Open(result.usdc_path.as_posix())
    physx_joint = physx.GetPrimAtPath("/Asset/Joints/door_hinge")
    assert "PhysxJointAxisAPI:linear" in usd_validate._authored_schema_tokens(physx_joint)
    assert "PhysxJointAPI" not in usd_validate._authored_schema_tokens(physx_joint)
    assert physx_joint.GetAttribute("physxJointAxis:linear:armature").Get() == pytest.approx(0.015)
    assert not physx_joint.GetAttribute("physxJoint:armature")

    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    newton_joint = newton.GetPrimAtPath("/Asset/Joints/door_hinge")
    assert not newton_joint.GetAttribute("mjc:armature")


def test_newton_keeps_motion_friction_as_fallback_for_load_proportional_physx_model(
    tmp_path: Path,
) -> None:
    model = _physics_model()
    model.articulations[0].physx_joint_resistance = PhysxJointFrictionCoefficient(0.04)
    model.articulations[0].physx_armature = 0.025

    result = export_object_to_usd(model, tmp_path)
    assert result.newton_usdc_path is not None

    physx = Usd.Stage.Open(result.usdc_path.as_posix())
    physx_joint = physx.GetPrimAtPath("/Asset/Joints/door_hinge")
    physx_schemas = usd_validate._authored_schema_tokens(physx_joint)
    assert "PhysxJointAPI" in physx_schemas
    assert not any(schema.startswith("PhysxJointAxisAPI:") for schema in physx_schemas)
    assert physx_joint.GetAttribute("physxJoint:jointFriction").Get() == pytest.approx(0.04)
    assert physx_joint.GetAttribute("physxJoint:armature").Get() == pytest.approx(0.025)

    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    joint = newton.GetPrimAtPath("/Asset/Joints/door_hinge")

    assert joint.GetAttribute("mjc:frictionloss").Get() == pytest.approx(0.12)
    assert not joint.GetAttribute("mjc:armature")
    assert joint.GetCustomDataByKey("usdcraft:physxArmature") is None
    assert joint.GetCustomDataByKey("usdcraft:newtonFrictionMapping") == {
        "source": "legacy_motion_properties",
        "frictionLoss": 0.12,
    }
    assert joint.GetCustomDataByKey("usdcraft:physxJointResistance") == {
        "model": "load_proportional",
        "coefficient": 0.04,
    }
    assert not any("load-proportional" in warning for warning in result.warnings)
    assert any(
        "no load-proportional joint-friction mapping" in note
        and "legacy MotionProperties.friction=0.12" in note
        for note in result.notes
    )


def test_usd_export_authors_asset_structure_and_isolated_backend_entries(
    tmp_path: Path,
) -> None:
    result = export_object_to_usd(_physics_model(), tmp_path)
    expected_layers = {
        "interface": tmp_path / "isaac" / "asset.usda",
        "physics": tmp_path / "isaac" / "physics.usdc",
        "physx": tmp_path / "isaac" / "physx.usda",
        "newton_mjc": tmp_path / "isaac" / "mujoco.usda",
    }

    assert result.newton_usdc_path == tmp_path / "isaac" / "model_newton.usdc"
    assert result.asset_structure_paths == expected_layers
    assert all(path.is_file() for path in (*expected_layers.values(), result.newton_usdc_path))

    physics = Usd.Stage.Open(expected_layers["physics"].as_posix())
    assert physics
    assert physics.GetPrimAtPath("/Asset/Joints/door_hinge")
    for prim in physics.Traverse():
        schemas = usd_validate._authored_schema_tokens(prim)
        properties = {str(prop.GetName()) for prop in prim.GetProperties()}
        assert not any(schema.startswith(("Physx", "Newton", "Mjc")) for schema in schemas)
        assert not any(name.startswith(("physx", "newton:", "mjc:")) for name in properties)

    mujoco_layer = Sdf.Layer.FindOrOpen(expected_layers["newton_mjc"].as_posix())
    assert mujoco_layer
    newton_joint_spec = mujoco_layer.GetPrimAtPath("/Asset/Joints/door_hinge")
    newton_schema_op = newton_joint_spec.GetInfo("apiSchemas")
    assert "PhysicsDriveAPI:angular" in newton_schema_op.deletedItems
    assert "MjcJointAPI" in newton_schema_op.prependedItems

    physx_layer = Sdf.Layer.FindOrOpen(expected_layers["physx"].as_posix())
    assert physx_layer
    root_path = "/Asset/Joints/__root_fixed"
    physx_root_spec = physx_layer.GetPrimAtPath(root_path)
    physx_root_schema_op = physx_root_spec.GetInfo("apiSchemas")
    assert "PhysxArticulationAPI" in physx_root_schema_op.prependedItems
    assert not any(
        str(property_spec.name).startswith("physxArticulation:")
        for property_spec in physx_root_spec.properties
    )

    physx = Usd.Stage.Open(result.usdc_path.as_posix())
    assert physx
    physx_root = physx.GetPrimAtPath(root_path)
    physx_joint = physx.GetPrimAtPath("/Asset/Joints/door_hinge")
    assert "PhysxArticulationAPI" in usd_validate._authored_schema_tokens(physx_root)
    assert not any(
        str(prop.GetName()).startswith("physxArticulation:")
        for prop in physx_root.GetAuthoredProperties()
    )
    assert physx_joint.HasAPI(UsdPhysics.DriveAPI, UsdPhysics.Tokens.angular)
    assert "PhysxJointAPI" in usd_validate._authored_schema_tokens(physx_joint)
    assert physx_joint.GetAttribute("physxJoint:jointFriction").Get() == pytest.approx(0.12)
    assert not physx_joint.GetAttribute("mjc:frictionloss")
    assert not any(name.startswith("newton:") for name in physx_joint.GetPropertyNames())
    assert not any(prim.GetTypeName() == "MjcActuator" for prim in physx.Traverse())

    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    assert newton
    newton_root = newton.GetPrimAtPath(root_path)
    newton_joint = newton.GetPrimAtPath("/Asset/Joints/door_hinge")
    assert "NewtonArticulationRootAPI" in usd_validate._authored_schema_tokens(newton_root)
    assert "PhysxArticulationAPI" not in usd_validate._authored_schema_tokens(newton_root)
    assert not any(
        str(prop.GetName()).startswith("physxArticulation:")
        for prop in newton_root.GetAuthoredProperties()
    )
    assert newton_root.GetAttribute("newton:selfCollisionEnabled").Get() is False
    assert "MjcJointAPI" in usd_validate._authored_schema_tokens(newton_joint)
    friction = newton_joint.GetAttribute("mjc:frictionloss")
    assert friction.Get() == pytest.approx(0.12)
    assert friction.GetTypeName() == Sdf.ValueTypeNames.Double
    assert friction.GetVariability() == Sdf.VariabilityUniform
    assert not newton_joint.HasAPI(UsdPhysics.DriveAPI, UsdPhysics.Tokens.angular)
    # AS3 removes only the applied DriveAPI token in the Newton overlay. The
    # source properties remain as inactive metadata for traceability.
    assert newton_joint.GetAttribute("drive:angular:physics:stiffness").Get() == pytest.approx(
        8.0 * math.pi / 180.0
    )
    assert newton_joint.GetAttribute("mjc:ref").Get() == pytest.approx(math.pi / 4)
    assert newton_joint.GetAttribute("mjc:ref").GetTypeName() == Sdf.ValueTypeNames.Double
    assert newton_joint.GetAttribute("mjc:ref").GetVariability() == Sdf.VariabilityUniform
    assert not newton_joint.GetAttribute("physxJoint:jointFriction")
    assert not newton_joint.GetAttribute("physxJoint:maxJointVelocity")

    actuator = newton.GetPrimAtPath("/Asset/Physics/door_hinge_actuator")
    assert actuator.GetTypeName() == "MjcActuator"
    assert actuator.GetRelationship("mjc:target").GetTargets() == [
        Sdf.Path("/Asset/Joints/door_hinge")
    ]
    for name, expected in (
        ("mjc:forceRange:min", -20.0),
        ("mjc:forceRange:max", 20.0),
    ):
        attribute = actuator.GetAttribute(name)
        assert attribute.Get() == pytest.approx(expected)
        assert attribute.GetTypeName() == Sdf.ValueTypeNames.Double
        assert attribute.GetVariability() == Sdf.VariabilityUniform
    gain = actuator.GetAttribute("mjc:gainPrm")
    bias = actuator.GetAttribute("mjc:biasPrm")
    assert list(gain.Get()) == pytest.approx([8.0, *([0.0] * 9)])
    assert list(bias.Get()) == pytest.approx([0.0, -8.0, -0.8, *([0.0] * 7)])
    for attribute in (gain, bias):
        assert attribute.GetTypeName() == Sdf.ValueTypeNames.DoubleArray
        assert attribute.GetVariability() == Sdf.VariabilityUniform
        assert len(attribute.Get()) == 10
    for name, expected in (("mjc:gainType", "fixed"), ("mjc:biasType", "affine")):
        attribute = actuator.GetAttribute(name)
        assert attribute.Get() == expected
        assert attribute.GetTypeName() == Sdf.ValueTypeNames.Token
        assert attribute.GetVariability() == Sdf.VariabilityUniform

    for entry in (result.usdc_path, result.newton_usdc_path):
        entry_stage = Usd.Stage.Open(entry.as_posix())
        assert entry_stage
        assert not entry_stage.GetRootLayer().subLayerPaths
        assert not any(
            prim.HasAuthoredReferences() or prim.HasAuthoredPayloads()
            for prim in entry_stage.TraverseAll()
        )
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(
            Sdf.AssetPath(entry.as_posix())
        )
        assert not unresolved

    interface = Usd.Stage.Open(expected_layers["interface"].as_posix())
    assert interface
    interface_root = interface.GetPrimAtPath("/Asset")
    variants = interface_root.GetVariantSet("Physics")
    assert set(variants.GetVariantNames()) == {"none", "physics", "physx", "newton"}
    assert variants.GetVariantSelection() == "physx"
    assert "PhysxArticulationAPI" in usd_validate._authored_schema_tokens(
        interface.GetPrimAtPath(root_path)
    )
    assert interface.GetPrimAtPath("/Asset/Joints/door_hinge").GetAttribute(
        "physxJoint:jointFriction"
    )
    assert variants.SetVariantSelection("newton")
    assert "PhysxArticulationAPI" not in usd_validate._authored_schema_tokens(
        interface.GetPrimAtPath(root_path)
    )
    interface_newton_joint = interface.GetPrimAtPath("/Asset/Joints/door_hinge")
    assert interface_newton_joint.GetAttribute("mjc:frictionloss")
    assert not interface_newton_joint.HasAPI(UsdPhysics.DriveAPI, UsdPhysics.Tokens.angular)
    assert interface.GetPrimAtPath("/Asset/Physics/door_hinge_actuator").GetTypeName() == (
        "MjcActuator"
    )
    assert variants.SetVariantSelection("physics")
    assert not interface.GetPrimAtPath("/Asset/Joints/door_hinge").GetAttribute("mjc:frictionloss")
    assert variants.SetVariantSelection("none")
    assert not interface.GetPrimAtPath("/Asset/Links")

    backend_validation = result.metrics["backend_validation"]
    assert backend_validation["physx"]["status"] == "success"
    assert backend_validation["newton"]["status"] == "success"
    assert backend_validation["physx"]["runtime_status"] == "not_run"
    assert backend_validation["newton"]["runtime_status"] == "not_run"
    assert not any("velocity limit" in warning for warning in result.warnings)
    assert any(
        "no Isaac Sim 6 MJC mapping for joint velocity limits" in note for note in result.notes
    )


def test_physx_validation_requires_articulation_schema(tmp_path: Path) -> None:
    result = export_object_to_usd(_physics_model(), tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    root = stage.GetPrimAtPath("/Asset/Joints/__root_fixed")
    assert root.RemoveAppliedSchema("PhysxArticulationAPI")
    stage.GetRootLayer().Save()

    with pytest.raises(ValidationError, match="articulation root body binding"):
        usd_validate._validate_usd_asset(result.usdc_path, backend="physx")


@pytest.mark.parametrize("base_mode", [BaseMode.FLOATING, BaseMode.KINEMATIC])
def test_unanchored_articulation_apis_are_colocated_on_root_rigid_body(
    tmp_path: Path,
    base_mode: BaseMode,
) -> None:
    model = _physics_model()
    model.base_mode = base_mode

    result = export_object_to_usd(model, tmp_path)
    root_path = "/Asset/Links/main_body"
    physx = Usd.Stage.Open(result.usdc_path.as_posix())
    assert physx
    root_body = physx.GetPrimAtPath(root_path)
    assert root_body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert root_body.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert "PhysxArticulationAPI" in usd_validate._authored_schema_tokens(root_body)
    assert not physx.GetPrimAtPath("/Asset").HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not physx.GetPrimAtPath("/Asset/Joints/__root_fixed")
    assert UsdPhysics.RigidBodyAPI(root_body).GetKinematicEnabledAttr().Get() is (
        base_mode == BaseMode.KINEMATIC
    )

    assert result.newton_usdc_path is not None
    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    assert newton
    newton_root = newton.GetPrimAtPath(root_path)
    assert newton_root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert "NewtonArticulationRootAPI" in usd_validate._authored_schema_tokens(newton_root)
    assert newton_root.GetAttribute("newton:selfCollisionEnabled").Get() is False


def test_strict_validation_rejects_old_asset_xform_root_binding(tmp_path: Path) -> None:
    model = _physics_model()
    model.base_mode = BaseMode.FLOATING
    result = export_object_to_usd(model, tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    assert stage

    root_body = stage.GetPrimAtPath("/Asset/Links/main_body")
    assert root_body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    assert root_body.RemoveAppliedSchema("PhysxArticulationAPI")
    asset = stage.GetPrimAtPath("/Asset")
    UsdPhysics.ArticulationRootAPI.Apply(asset)
    usd_common._apply_api_schema_token(asset, "PhysxArticulationAPI")
    stage.GetRootLayer().Save()

    with pytest.raises(ValidationError, match="articulation root body binding"):
        usd_validate._validate_usd_asset(result.usdc_path, backend="physx")


def test_native_components_get_independent_roots_and_loose_body_gets_none(
    tmp_path: Path,
) -> None:
    model = ArticulatedObject("split_native_components", base_mode=BaseMode.FLOATING)

    def rigid_part(name: str, mass: float) -> Part:
        part = model.part(
            name,
            inertial=Inertial.from_geometry(Box((0.2, 0.2, 0.2)), mass=mass),
        )
        part.visual(Box((0.2, 0.2, 0.2)))
        return part

    root_a = rigid_part("root_a", 3.0)
    child_a = rigid_part("child_a", 1.0)
    root_b = rigid_part("root_b", 4.0)
    child_b = rigid_part("child_b", 1.0)
    loose = rigid_part("loose", 0.5)
    model.articulation("joint_a", ArticulationType.FIXED, root_a, child_a)
    model.articulation("place_b", ArticulationType.FLOATING, root_a, root_b)
    model.articulation("joint_b", ArticulationType.FIXED, root_b, child_b)
    model.articulation("place_loose", ArticulationType.FLOATING, root_b, loose)

    result = export_object_to_usd(model, tmp_path)
    expected = {Sdf.Path("/Asset/Links/root_a"), Sdf.Path("/Asset/Links/root_b")}
    for entry, vendor_schema in (
        (result.usdc_path, "PhysxArticulationAPI"),
        (result.newton_usdc_path, "NewtonArticulationRootAPI"),
    ):
        assert entry is not None
        stage = Usd.Stage.Open(entry.as_posix())
        assert stage
        portable_paths = {
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        }
        vendor_paths = {
            prim.GetPath()
            for prim in stage.Traverse()
            if vendor_schema in usd_validate._authored_schema_tokens(prim)
        }
        assert portable_paths == expected
        assert vendor_paths == expected
        assert not stage.GetPrimAtPath("/Asset/Links/loose").HasAPI(UsdPhysics.ArticulationRootAPI)


def test_native_component_with_multiple_root_bodies_is_rejected() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Asset")
    for name in ("a", "b", "child"):
        prim = UsdGeom.Xform.Define(stage, f"/Asset/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
    for name, parent in (("from_a", "a"), ("from_b", "b")):
        joint = UsdPhysics.FixedJoint.Define(stage, f"/Asset/{name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/Asset/{parent}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/child")])

    with pytest.raises(ValidationError, match="exactly one root rigid body"):
        usd_validate._resolve_native_articulation_root_bindings(stage)


def test_usd_export_authors_newton_native_mimic_only_in_newton_entry(
    tmp_path: Path,
) -> None:
    model = ArticulatedObject("mimic_asset", base_mode=BaseMode.FIXED)
    base = model.part(
        "base",
        inertial=Inertial.from_geometry(Box((0.4, 0.4, 0.4)), mass=2.0),
    )
    left = model.part(
        "left",
        inertial=Inertial.from_geometry(Box((0.1, 0.1, 0.2)), mass=0.2),
    )
    right = model.part(
        "right",
        inertial=Inertial.from_geometry(Box((0.1, 0.1, 0.2)), mass=0.2),
    )
    base.visual(Box((0.4, 0.4, 0.4)))
    left.visual(Box((0.1, 0.1, 0.2)))
    right.visual(Box((0.1, 0.1, 0.2)))
    driver = model.articulation(
        "left hinge",
        ArticulationType.REVOLUTE,
        base,
        left,
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=-0.5, upper=0.5),
    )
    model.articulation(
        "right hinge",
        ArticulationType.REVOLUTE,
        base,
        right,
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(effort=1.0, velocity=1.0, lower=-0.5, upper=0.5),
        mimic=Mimic(joint=driver.name, multiplier=-1.0, offset=0.1),
    )

    result = export_object_to_usd(model, tmp_path)
    assert result.newton_usdc_path is not None
    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    follower = newton.GetPrimAtPath("/Asset/Joints/right_hinge")
    assert "NewtonMimicAPI" in usd_validate._authored_schema_tokens(follower)
    assert follower.GetAttribute("newton:mimicEnabled").Get() is True
    assert follower.GetAttribute("newton:mimicCoef0").Get() == pytest.approx(math.degrees(0.1))
    assert follower.GetAttribute("newton:mimicCoef1").Get() == pytest.approx(-1.0)
    assert follower.GetRelationship("newton:mimicJoint").GetTargets() == [
        Sdf.Path("/Asset/Joints/left_hinge")
    ]

    physx = Usd.Stage.Open(result.usdc_path.as_posix())
    physx_follower = physx.GetPrimAtPath("/Asset/Joints/right_hinge")
    assert "NewtonMimicAPI" not in usd_validate._authored_schema_tokens(physx_follower)
    assert not physx_follower.GetRelationship("newton:mimicJoint")
    assert physx_follower.GetCustomDataByKey("usdcraft:mimic") == {
        "joint": "left hinge",
        "multiplier": -1.0,
        "offset": 0.1,
    }


def test_newton_validation_rejects_mass_at_or_below_mujoco_minimum(
    tmp_path: Path,
) -> None:
    result = export_object_to_usd(_physics_model(), tmp_path)
    assert result.newton_usdc_path is not None
    stage = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    body = stage.GetPrimAtPath("/Asset/Links/door")
    UsdPhysics.MassAPI(body).GetMassAttr().Set(1e-16)
    stage.GetRootLayer().Save()

    with pytest.raises(ValidationError, match="mass must exceed"):
        usd_validate._validate_usd_asset(result.newton_usdc_path, backend="newton")


def test_newton_validation_rejects_incorrect_mujoco_friction(tmp_path: Path) -> None:
    model = _physics_model()
    model.articulations[0].physx_joint_resistance = PhysxJointFriction(
        static_friction_effort=2.5,
        dynamic_friction_effort=2.3,
        viscous_friction_coefficient=0.42,
    )
    result = export_object_to_usd(model, tmp_path)
    assert result.newton_usdc_path is not None
    stage = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    friction = stage.GetPrimAtPath("/Asset/Joints/door_hinge").GetAttribute("mjc:frictionloss")
    # The obsolete mapping used MotionProperties.friction instead of the
    # explicit effort-based static hold threshold.
    friction.Set(0.12)
    stage.GetRootLayer().Save()

    with pytest.raises(ValidationError, match="joint friction is invalid"):
        usd_validate._validate_usd_asset(result.newton_usdc_path, backend="newton")


@pytest.mark.parametrize(
    ("prim_path", "attribute_name", "bad_value", "message"),
    (
        pytest.param(
            "/Asset/Physics/door_hinge_actuator",
            "mjc:gainPrm",
            Vt.DoubleArray([8.0 * math.pi / 180.0, *([0.0] * 9)]),
            "actuator gains are invalid",
            id="degree-scaled-kp",
        ),
        pytest.param(
            "/Asset/Physics/door_hinge_actuator",
            "mjc:biasPrm",
            Vt.DoubleArray(
                [
                    0.0,
                    -8.0 * math.pi / 180.0,
                    -0.8 * math.pi / 180.0,
                    *([0.0] * 7),
                ]
            ),
            "actuator gains are invalid",
            id="degree-scaled-kp-kd-bias",
        ),
        pytest.param(
            "/Asset/Joints/door_hinge",
            "mjc:ref",
            45.0,
            "joint reference is invalid",
            id="degree-target-reference",
        ),
    ),
)
def test_newton_validation_rejects_old_degree_based_active_drive_mapping(
    tmp_path: Path,
    prim_path: str,
    attribute_name: str,
    bad_value: object,
    message: str,
) -> None:
    result = export_object_to_usd(_physics_model(), tmp_path)
    assert result.newton_usdc_path is not None
    stage = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    attribute = stage.GetPrimAtPath(prim_path).GetAttribute(attribute_name)
    assert attribute.Set(bad_value)
    stage.GetRootLayer().Save()

    with pytest.raises(ValidationError, match=message):
        usd_validate._validate_usd_asset(result.newton_usdc_path, backend="newton")


def _passive_newton_model() -> ArticulatedObject:
    model = ArticulatedObject("passive_newton_joint", base_mode=BaseMode.FIXED)
    base = model.part(
        "base",
        inertial=Inertial.from_geometry(Box((0.3, 0.3, 0.3)), mass=2.0),
    )
    panel = model.part(
        "panel",
        inertial=Inertial.from_geometry(Box((0.4, 0.05, 0.2)), mass=0.5),
    )
    base.visual(Box((0.3, 0.3, 0.3)))
    panel.visual(Box((0.4, 0.05, 0.2)))
    model.articulation(
        "passive hinge",
        ArticulationType.REVOLUTE,
        base,
        panel,
        motion_limits=MotionLimits(
            effort=12.0,
            velocity=1.5,
            lower=-math.pi / 2,
            upper=math.pi / 2,
        ),
        motion_properties=MotionProperties(damping=0.12, friction=0.4),
        newton_joint_solver_tuning=NewtonJointSolverTuning(
            solref_friction=(0.01, 1.0),
            solimp_friction=(0.9999, 0.9999, 0.001, 0.5, 2.0),
        ),
        passive_spring=PassiveJointSpring(stiffness=8.0, reference=0.35),
    )
    return model


def test_newton_exports_native_passive_joint_contract_without_actuator(
    tmp_path: Path,
) -> None:
    result = export_object_to_usd(_passive_newton_model(), tmp_path)

    physx = Usd.Stage.Open(result.usdc_path.as_posix())
    physx_joint = physx.GetPrimAtPath("/Asset/Joints/passive_hinge")
    physx_drive = UsdPhysics.DriveAPI(
        physx_joint,
        UsdPhysics.Tokens.angular,
    )
    assert physx_drive.GetStiffnessAttr().Get() == pytest.approx(8.0 * math.pi / 180.0)
    assert physx_drive.GetDampingAttr().Get() == pytest.approx(0.12 * math.pi / 180.0)
    assert physx_drive.GetTargetPositionAttr().Get() == pytest.approx(math.degrees(0.35))
    assert not any(name.startswith("mjc:") for name in physx_joint.GetPropertyNames())

    assert result.newton_usdc_path is not None
    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    joint = newton.GetPrimAtPath("/Asset/Joints/passive_hinge")
    assert "MjcJointAPI" in usd_validate._authored_schema_tokens(joint)
    assert not joint.HasAPI(UsdPhysics.DriveAPI, UsdPhysics.Tokens.angular)
    assert joint.GetAttribute("mjc:frictionloss").Get() == pytest.approx(0.4)
    assert joint.GetAttribute("mjc:damping").Get() == pytest.approx(0.12)
    assert list(joint.GetAttribute("mjc:solreffriction").Get()) == pytest.approx([0.01, 1.0])
    assert list(joint.GetAttribute("mjc:solimpfriction").Get()) == pytest.approx(
        [0.9999, 0.9999, 0.001, 0.5, 2.0]
    )
    assert joint.GetAttribute("mjc:stiffness").Get() == pytest.approx(8.0)
    assert joint.GetAttribute("mjc:springref").Get() == pytest.approx(0.35)
    assert not newton.GetPrimAtPath("/Asset/Physics/passive_hinge_actuator")

    for name in ("mjc:frictionloss", "mjc:damping", "mjc:stiffness", "mjc:springref"):
        attribute = joint.GetAttribute(name)
        assert attribute.GetTypeName() == Sdf.ValueTypeNames.Double
        assert attribute.GetVariability() == Sdf.VariabilityUniform
    for name in ("mjc:solreffriction", "mjc:solimpfriction"):
        attribute = joint.GetAttribute(name)
        assert attribute.GetTypeName() == Sdf.ValueTypeNames.DoubleArray
        assert attribute.GetVariability() == Sdf.VariabilityUniform


@pytest.mark.parametrize(
    ("attribute_name", "bad_value"),
    (
        pytest.param(
            "mjc:damping",
            0.12 * math.pi / 180.0,
            id="degree-scaled-passive-damping",
        ),
        pytest.param(
            "mjc:solreffriction",
            Vt.DoubleArray([0.02, 1.0]),
            id="solver-reference",
        ),
        pytest.param(
            "mjc:solimpfriction",
            Vt.DoubleArray([0.9, 0.95, 0.001, 0.5, 2.0]),
            id="solver-impedance",
        ),
        pytest.param(
            "mjc:stiffness",
            8.0 * math.pi / 180.0,
            id="degree-scaled-passive-stiffness",
        ),
        pytest.param(
            "mjc:springref",
            math.degrees(0.35),
            id="degree-passive-reference",
        ),
    ),
)
def test_newton_validation_rejects_mutated_native_passive_joint_mapping(
    tmp_path: Path,
    attribute_name: str,
    bad_value: object,
) -> None:
    result = export_object_to_usd(_passive_newton_model(), tmp_path)
    assert result.newton_usdc_path is not None
    stage = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    attribute = stage.GetPrimAtPath("/Asset/Joints/passive_hinge").GetAttribute(attribute_name)
    assert attribute.Set(bad_value)
    stage.GetRootLayer().Save()

    with pytest.raises(ValidationError, match="Newton/MuJoCo"):
        usd_validate._validate_usd_asset(result.newton_usdc_path, backend="newton")


def test_newton_validation_rejects_actuator_on_passive_joint(tmp_path: Path) -> None:
    result = export_object_to_usd(_passive_newton_model(), tmp_path)
    assert result.newton_usdc_path is not None
    stage = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    actuator = stage.DefinePrim("/Asset/Physics/passive_hinge_actuator", "MjcActuator")
    actuator.CreateRelationship("mjc:target", custom=False).SetTargets(
        [Sdf.Path("/Asset/Joints/passive_hinge")]
    )
    stage.GetRootLayer().Save()

    with pytest.raises(ValidationError, match="actuator"):
        usd_validate._validate_usd_asset(result.newton_usdc_path, backend="newton")


def test_newton_acceleration_drive_uses_direct_force_actuator_without_static_gains(
    tmp_path: Path,
) -> None:
    model = _physics_model()
    articulation = model.articulations[0]
    assert articulation.drive is not None
    articulation.drive = replace(articulation.drive, mode="acceleration")

    result = export_object_to_usd(model, tmp_path)
    assert result.newton_usdc_path is not None
    assert not any("gain/bias" in warning for warning in result.warnings)
    assert any("direct force control without static gain/bias" in note for note in result.notes)

    physx = Usd.Stage.Open(result.usdc_path.as_posix())
    physx_drive = UsdPhysics.DriveAPI.Get(physx, "/Asset/Joints/door_hinge.drive:angular")
    assert physx_drive.GetTypeAttr().Get() == UsdPhysics.Tokens.acceleration

    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    joint = newton.GetPrimAtPath("/Asset/Joints/door_hinge")
    actuator = newton.GetPrimAtPath("/Asset/Physics/door_hinge_actuator")
    assert not joint.HasAPI(UsdPhysics.DriveAPI, UsdPhysics.Tokens.angular)
    assert actuator.GetTypeName() == "MjcActuator"
    assert not actuator.GetAttribute("mjc:gainPrm").HasAuthoredValueOpinion()
    assert not actuator.GetAttribute("mjc:biasPrm").HasAuthoredValueOpinion()
    assert not actuator.GetAttribute("mjc:gainType").HasAuthoredValueOpinion()
    assert not actuator.GetAttribute("mjc:biasType").HasAuthoredValueOpinion()

    actuator.CreateAttribute(
        "mjc:gainType",
        Sdf.ValueTypeNames.Token,
        custom=False,
        variability=Sdf.VariabilityUniform,
    ).Set("fixed")
    newton.GetRootLayer().Save()
    with pytest.raises(ValidationError, match="unsupported gain parameters"):
        usd_validate._validate_usd_asset(result.newton_usdc_path, backend="newton")


def test_newton_pure_position_drive_warns_and_omits_unverified_static_gains(
    tmp_path: Path,
) -> None:
    model = _physics_model()
    articulation = model.articulations[0]
    assert articulation.drive is not None
    articulation.drive = replace(
        articulation.drive,
        damping=0.0,
        target_velocity=0.25,
    )

    result = export_object_to_usd(model, tmp_path)
    assert result.newton_usdc_path is not None
    assert not any(
        "pure-P" in warning or "target velocity" in warning for warning in result.warnings
    )
    assert any("pure-P drives" in note for note in result.notes)
    assert any("drive target velocity" in note for note in result.notes)

    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    actuator = newton.GetPrimAtPath("/Asset/Physics/door_hinge_actuator")
    assert actuator.GetTypeName() == "MjcActuator"
    assert not actuator.GetAttribute("mjc:gainPrm").HasAuthoredValueOpinion()
    assert not actuator.GetAttribute("mjc:biasPrm").HasAuthoredValueOpinion()
    assert actuator.GetAttribute("mjc:forceRange:min").Get() == pytest.approx(-20.0)
    assert actuator.GetAttribute("mjc:forceRange:max").Get() == pytest.approx(20.0)


@pytest.mark.parametrize("collider_kind", ["cylinder", "mesh"])
def test_usd_export_leaves_runtime_mass_frame_to_final_colliders(
    tmp_path: Path,
    collider_kind: str,
) -> None:
    proxy = Cylinder(radius=0.04, length=0.22)
    model = ArticulatedObject(f"auto_inertia_{collider_kind}")
    body = model.part(
        "body",
        inertial=Inertial.from_geometry(
            proxy,
            mass=0.6,
            origin=Origin(xyz=(0.0, 0.0, 0.11)),
        ),
    )
    if collider_kind == "cylinder":
        body.visual(proxy, origin=Origin(xyz=(0.0, 0.0, 0.11)), name="shape")
        expected_schema = UsdGeom.Cylinder
        asset_root = None
    else:
        mesh = mesh_from_geometry(
            BoxGeometry((0.08, 0.08, 0.22)).translate(0.0, 0.0, 0.11),
            tmp_path / "assets" / "meshes" / "body.obj",
        )
        body.visual(mesh, name="shape")
        expected_schema = UsdGeom.Mesh
        asset_root = tmp_path

    result = export_object_to_usd(
        model,
        tmp_path / "out",
        asset_root=asset_root,
    )
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    body_prim = stage.GetPrimAtPath("/Asset/Links/body")
    mass = UsdPhysics.MassAPI(body_prim)

    assert mass.GetMassAttr().HasAuthoredValueOpinion()
    assert mass.GetMassAttr().Get() == pytest.approx(0.6)
    assert not mass.GetCenterOfMassAttr().HasAuthoredValueOpinion()
    assert not mass.GetDiagonalInertiaAttr().HasAuthoredValueOpinion()
    assert tuple(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx((0.0, 0.0, 0.0))
    assert not mass.GetPrincipalAxesAttr().HasAuthoredValueOpinion()

    collision_root = stage.GetPrimAtPath("/Asset/Links/body/Collisions/shape")
    colliders = [
        prim for prim in Usd.PrimRange(collision_root) if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert len(colliders) == 1
    assert colliders[0].IsA(expected_schema)


def test_usd_export_still_rejects_non_positive_definite_sdk_inertia(tmp_path: Path) -> None:
    model = ArticulatedObject("invalid_sdk_inertia")
    body = model.part(
        "body",
        inertial=Inertial(
            mass=1.0,
            inertia=Inertia(
                ixx=-0.1,
                ixy=0.0,
                ixz=0.0,
                iyy=0.1,
                iyz=0.0,
                izz=0.1,
            ),
        ),
    )
    body.visual(Box((0.2, 0.2, 0.2)), name="shape")

    with pytest.raises(ValidationError, match="inertia tensor must be positive definite"):
        export_object_to_usd(model, tmp_path)


def test_usd_export_floating_goods_are_independent_contact_bodies(tmp_path: Path) -> None:
    result = export_object_to_usd(_shelf_with_independent_goods(), tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())

    assert not any("Floating articulation" in warning for warning in result.warnings)
    assert result.metrics["joint_count"] == 1
    assert result.metrics["authoring_relation_count"] == 2
    assert result.metrics["floating_placement_count"] == 2
    joint_paths = {
        prim.GetPath().pathString for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
    }
    assert joint_paths == {"/Asset/Joints/__root_fixed"}

    for index in (1, 2):
        product_path = f"/Asset/Links/product_{index}"
        product = stage.GetPrimAtPath(product_path)
        assert product.HasAPI(UsdPhysics.RigidBodyAPI)
        assert product.HasAPI(UsdPhysics.MassAPI)
        assert UsdPhysics.MassAPI(product).GetMassAttr().Get() == pytest.approx(0.25)

        collider = stage.GetPrimAtPath(f"{product_path}/Collisions/body/Geom")
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        physics_material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
        assert physics_material.GetPath().pathString == "/Asset/PhysicsMaterials/product_contact"


def test_usd_export_fixed_goods_emit_welding_joints(tmp_path: Path) -> None:
    result = export_object_to_usd(
        _shelf_with_independent_goods(fixed_goods=True),
        tmp_path,
    )
    stage = Usd.Stage.Open(result.usdc_path.as_posix())

    assert result.metrics["joint_count"] == 3
    assert result.metrics["authoring_relation_count"] == 2
    assert result.metrics["floating_placement_count"] == 0

    for index in (1, 2):
        joint = UsdPhysics.FixedJoint.Get(stage, f"/Asset/Joints/place_product_{index}")
        assert joint
        assert joint.GetBody0Rel().GetTargets() == [Sdf.Path("/Asset/Links/shelf")]
        assert joint.GetBody1Rel().GetTargets() == [Sdf.Path(f"/Asset/Links/product_{index}")]


def test_unregistered_api_schema_application_preserves_list_operations() -> None:
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/Joint")
    operations = Sdf.TokenListOp()
    operations.prependedItems = ["ExistingPrependAPI"]
    operations.appendedItems = ["ExistingAppendAPI"]
    operations.deletedItems = ["DeletedAPI"]
    operations.orderedItems = ["OrderedAPI"]
    prim.SetMetadata("apiSchemas", operations)

    usd_common._apply_api_schema_token(prim, "PhysxJointAPI")

    authored = stage.GetRootLayer().GetPrimAtPath("/Joint").GetInfo("apiSchemas")
    assert authored.prependedItems == ["ExistingPrependAPI", "PhysxJointAPI"]
    assert authored.appendedItems == ["ExistingAppendAPI"]
    assert authored.deletedItems == ["DeletedAPI"]
    assert authored.orderedItems == ["OrderedAPI"]

    explicit_prim = stage.DefinePrim("/ExplicitJoint")
    explicit = Sdf.TokenListOp()
    explicit.explicitItems = ["ExistingExplicitAPI"]
    explicit_prim.SetMetadata("apiSchemas", explicit)

    usd_common._apply_api_schema_token(explicit_prim, "PhysxJointAPI")

    authored_explicit = stage.GetRootLayer().GetPrimAtPath("/ExplicitJoint").GetInfo("apiSchemas")
    assert authored_explicit.explicitItems == ["ExistingExplicitAPI", "PhysxJointAPI"]


def test_usd_export_decodes_scalar_srgb_and_preserves_linear_emissive(
    tmp_path: Path,
) -> None:
    model = ArticulatedObject("color_contract")
    material = model.material(
        "finish",
        rgba=(0.5, 0.25, 0.1, 0.8),
        emissive=(0.5, 0.25, 0.1),
        clearcoat=0.4,
    )
    model.part("body").visual(Box((0.1, 0.1, 0.1)), material=material)

    stage = Usd.Stage.Open(export_object_to_usd(model, tmp_path).usdc_path.as_posix())
    shader = UsdShade.Shader.Get(stage, "/Asset/Looks/finish/PreviewSurface")

    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx(
        (0.21404114, 0.05087609, 0.01002283)
    )
    assert shader.GetInput("opacity").Get() == pytest.approx(0.8)
    assert tuple(shader.GetInput("emissiveColor").Get()) == pytest.approx((0.5, 0.25, 0.1))
    assert shader.GetInput("clearcoatRoughness").Get() == pytest.approx(0.2)


def test_usd_export_preserves_imported_linear_pbr_base_color(tmp_path: Path) -> None:
    import trimesh

    from sdk import Mesh

    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=(0.5, 0.25, 0.1, 1.0))
    )
    source = tmp_path / "colored.glb"
    mesh.export(source)

    model = ArticulatedObject("imported_color_contract")
    model.part("body").visual(Mesh(source.as_posix()), collidable=False)

    stage = Usd.Stage.Open(
        export_object_to_usd(model, tmp_path / "out", asset_root=tmp_path).usdc_path.as_posix()
    )
    shader = UsdShade.Shader.Get(stage, "/Asset/Looks/EmbeddedMaterial_001/PreviewSurface")

    # trimesh quantizes glTF factors to bytes, hence the 128/255-style values.
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx(
        (128 / 255, 64 / 255, 26 / 255)
    )


def _mesh_arrays(mesh: UsdGeom.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray([tuple(point) for point in mesh.GetPointsAttr().Get()], dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    assert np.all(counts == 3)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    normals = np.asarray([tuple(normal) for normal in mesh.GetNormalsAttr().Get()], dtype=float)
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
    assert normals.shape == (len(faces) * 3, 3)
    return points, faces, normals.reshape(-1, 3, 3)


def _triangle_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def test_usd_export_primitive_corner_normals_preserve_hard_caps_and_smooth_curves(
    tmp_path: Path,
) -> None:
    model = ArticulatedObject("normal_primitives")
    body = model.part("body")
    body.visual(Box((0.4, 0.3, 0.2)), name="box")
    body.visual(Cylinder(radius=0.2, length=0.5), name="cylinder")
    body.visual(Sphere(radius=0.25), name="sphere")

    stage = Usd.Stage.Open(export_object_to_usd(model, tmp_path).usdc_path.as_posix())

    box = UsdGeom.Mesh.Get(stage, "/Asset/Links/body/Visuals/box/Geom")
    box_points, box_faces, box_corners = _mesh_arrays(box)
    box_face_normals = _triangle_normals(box_points, box_faces)
    np.testing.assert_allclose(
        box_corners,
        np.repeat(box_face_normals[:, None, :], 3, axis=1),
        atol=1e-6,
    )

    cylinder = UsdGeom.Mesh.Get(stage, "/Asset/Links/body/Visuals/cylinder/Geom")
    cylinder_points, cylinder_faces, cylinder_corners = _mesh_arrays(cylinder)
    cylinder_triangles = cylinder_points[cylinder_faces]
    cylinder_face_normals = _triangle_normals(cylinder_points, cylinder_faces)
    caps = np.abs(cylinder_face_normals[:, 2]) > 0.99
    sides = ~caps
    assert np.max(np.abs(cylinder_corners[caps, :, :2])) < 1e-6
    assert np.min(np.abs(cylinder_corners[caps, :, 2])) > 1.0 - 1e-6
    assert np.max(np.abs(cylinder_corners[sides, :, 2])) < 1e-6
    radial = cylinder_triangles[sides, :, :2]
    radial /= np.linalg.norm(radial, axis=2)[:, :, None]
    assert np.min(np.sum(cylinder_corners[sides, :, :2] * radial, axis=2)) > 1.0 - 1e-6

    sphere = UsdGeom.Mesh.Get(stage, "/Asset/Links/body/Visuals/sphere/Geom")
    sphere_points, sphere_faces, sphere_corners = _mesh_arrays(sphere)
    radial = sphere_points[sphere_faces]
    radial /= np.linalg.norm(radial, axis=2)[:, :, None]
    assert np.min(np.sum(sphere_corners * radial, axis=2)) > 0.9999


def test_usd_export_recomputes_merged_mesh_normals_after_nonuniform_scale(
    tmp_path: Path,
) -> None:
    compound = BoxGeometry((0.5, 0.4, 0.3))
    compound.merge(BoxGeometry((0.2, 0.3, 0.4)).translate(0.7, 0.0, 0.0))
    mesh = mesh_from_geometry(compound, tmp_path / "assets" / "meshes" / "compound_visual.obj")
    mesh = replace(mesh, scale=(1.7, 0.6, 1.3))

    model = ArticulatedObject("merged_mesh_normals")
    model.part("body").visual(mesh, name="compound", collidable=False)
    result = export_object_to_usd(model, tmp_path / "out", asset_root=tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    visual_root = stage.GetPrimAtPath("/Asset/Links/body/Visuals/compound/Geom")
    visual_meshes = [prim for prim in Usd.PrimRange(visual_root) if prim.IsA(UsdGeom.Mesh)]

    assert len(visual_meshes) == 1
    points, faces, corner_normals = _mesh_arrays(UsdGeom.Mesh(visual_meshes[0]))
    face_normals = _triangle_normals(points, faces)
    np.testing.assert_allclose(
        corner_normals,
        np.repeat(face_normals[:, None, :], 3, axis=1),
        atol=1e-6,
    )


def test_mesh_collision_defaults_preserve_concave_geometry() -> None:
    assert CollisionProperties().approximation == CollisionApproximation.CONVEX_DECOMPOSITION
    assert (
        RigidBodyProperties().collision_approximation == CollisionApproximation.CONVEX_DECOMPOSITION
    )


@pytest.mark.parametrize(
    ("approximation", "expected"),
    [
        (None, CollisionApproximation.CONVEX_DECOMPOSITION),
        (CollisionApproximation.CONVEX_HULL, CollisionApproximation.CONVEX_HULL),
    ],
)
def test_usd_export_splits_disconnected_mesh_colliders(
    tmp_path: Path,
    approximation: CollisionApproximation | None,
    expected: CollisionApproximation,
) -> None:
    compound = BoxGeometry((0.5, 0.5, 0.5))
    compound.merge(BoxGeometry((0.5, 0.5, 0.5)).translate(2.0, 0.0, 0.0))
    mesh_path = tmp_path / "assets" / "meshes" / "compound.obj"
    mesh = mesh_from_geometry(compound, mesh_path)
    mesh_path.write_text(
        "mtllib compound.mtl\nusemtl painted\n" + mesh_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (mesh_path.parent / "compound.mtl").write_text(
        "newmtl painted\nKd 0.2 0.4 0.6\n",
        encoding="utf-8",
    )

    # Pin the model to the mesh root so this regression is independent of any
    # ephemeral managed-asset session created by an earlier SDK test.
    model = ArticulatedObject("compound_collider", assets=tmp_path)
    rigid_body = (
        None
        if approximation is None
        else RigidBodyProperties(collision_approximation=approximation)
    )
    body = model.part("body", rigid_body=rigid_body)
    body.visual(mesh, name="compound")

    result = export_object_to_usd(
        model,
        tmp_path / f"out_{expected.value}",
        asset_root=tmp_path,
    )
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    collision_root = stage.GetPrimAtPath("/Asset/Links/body/Collisions/compound")
    visual_root = stage.GetPrimAtPath("/Asset/Links/body/Visuals/compound")
    collision_meshes = [prim for prim in Usd.PrimRange(collision_root) if prim.IsA(UsdGeom.Mesh)]
    visual_meshes = [prim for prim in Usd.PrimRange(visual_root) if prim.IsA(UsdGeom.Mesh)]

    assert len(visual_meshes) == 1
    assert len(collision_meshes) == 2
    assert result.metrics["collision_shape_counts"]["rigid_runtime"] == 2
    embedded_materials = [
        prim
        for prim in stage.GetPrimAtPath("/Asset/Looks").GetChildren()
        if prim.GetName().startswith("EmbeddedMaterial_")
    ]
    assert len(embedded_materials) == 1
    for collider in collision_meshes:
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        mesh_api = UsdPhysics.MeshCollisionAPI(collider)
        assert mesh_api.GetApproximationAttr().Get() == expected.value
        bound_material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial()
        assert not bound_material


def test_usd_export_authors_native_spherical_joint(tmp_path: Path) -> None:
    model = ArticulatedObject("spherical_asset", base_mode=BaseMode.FIXED)
    base = model.part(
        "base",
        inertial=Inertial.from_geometry(Box((0.2, 0.2, 0.2)), mass=1.0),
    )
    base.visual(Box((0.2, 0.2, 0.2)), name="base_shell")
    payload = model.part(
        "payload",
        inertial=Inertial.from_geometry(Box((0.1, 0.1, 0.1)), mass=0.2),
    )
    payload.visual(Box((0.1, 0.1, 0.1)), name="payload_shell")
    model.articulation(
        "ball socket",
        ArticulationType.SPHERICAL,
        base,
        payload,
        origin=Origin(xyz=(0.0, 0.0, 0.1)),
        axis=(0.0, 0.0, 2.0),
        cone_deg=37.5,
    )

    result = export_object_to_usd(model, tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    joint = UsdPhysics.SphericalJoint.Get(stage, "/Asset/Joints/ball_socket")

    assert joint
    assert joint.GetAxisAttr().Get() == UsdPhysics.Tokens.x
    assert joint.GetConeAngle0LimitAttr().Get() == pytest.approx(37.5)
    assert joint.GetConeAngle1LimitAttr().Get() == pytest.approx(37.5)
    prim = joint.GetPrim()
    assert prim.GetCustomDataByKey("usdcraft:jointName") == "ball socket"
    assert prim.GetCustomDataByKey("usdcraft:articulationType") == "spherical"
    assert tuple(prim.GetCustomDataByKey("usdcraft:axis")) == pytest.approx((0.0, 0.0, 1.0))
    assert prim.GetCustomDataByKey("usdcraft:coneDeg") == pytest.approx(37.5)
    assert prim.GetCustomDataByKey("mesh2usd:articulationType") is None
    local_axis = joint.GetLocalRot0Attr().Get().Transform(Gf.Vec3f(1.0, 0.0, 0.0))
    assert tuple(local_axis) == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)


def test_usd_export_continuous_joint_uses_degree_based_physx_limits_and_gains(
    tmp_path: Path,
) -> None:
    model = ArticulatedObject("continuous_asset", base_mode=BaseMode.FIXED)
    base = model.part(
        "base",
        inertial=Inertial.from_geometry(Box((0.2, 0.2, 0.2)), mass=1.0),
    )
    base.visual(Box((0.2, 0.2, 0.2)))
    rotor = model.part(
        "rotor",
        inertial=Inertial.from_geometry(Box((0.1, 0.1, 0.1)), mass=0.2),
    )
    rotor.visual(Box((0.1, 0.1, 0.1)))
    model.articulation(
        "shaft",
        ArticulationType.CONTINUOUS,
        base,
        rotor,
        motion_limits=MotionLimits(effort=12.0, velocity=1.5),
        motion_properties=MotionProperties(damping=0.3, friction=0.08),
    )

    result = export_object_to_usd(model, tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    joint = UsdPhysics.RevoluteJoint.Get(stage, "/Asset/Joints/shaft")
    assert joint
    assert not joint.GetLowerLimitAttr().HasAuthoredValueOpinion()
    assert not joint.GetUpperLimitAttr().HasAuthoredValueOpinion()
    drive = UsdPhysics.DriveAPI.Get(stage, "/Asset/Joints/shaft.drive:angular")
    assert drive
    assert drive.GetTypeAttr().Get() == UsdPhysics.Tokens.force
    assert drive.GetStiffnessAttr().Get() == pytest.approx(0.0)
    assert drive.GetDampingAttr().Get() == pytest.approx(0.3 * math.pi / 180.0)
    assert drive.GetMaxForceAttr().Get() == pytest.approx(12.0)
    assert joint.GetPrim().GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(
        math.degrees(1.5)
    )
    assert joint.GetPrim().GetAttribute("physxJoint:jointFriction").Get() == pytest.approx(0.08)

    assert result.newton_usdc_path is not None
    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    newton_joint = newton.GetPrimAtPath("/Asset/Joints/shaft")
    assert not newton_joint.HasAPI(UsdPhysics.DriveAPI, UsdPhysics.Tokens.angular)
    assert newton_joint.GetAttribute("mjc:frictionloss").Get() == pytest.approx(0.08)
    assert newton_joint.GetAttribute("mjc:damping").Get() == pytest.approx(0.3)
    assert not newton.GetPrimAtPath("/Asset/Physics/shaft_actuator")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"damping": -0.1}, "must be >= 0"),
        ({"friction": -0.1}, "must be >= 0"),
        ({"damping": float("nan")}, "must be finite"),
        ({"friction": float("inf")}, "must be finite"),
    ],
)
def test_motion_properties_reject_invalid_engine_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        MotionProperties(**kwargs)


def test_unknown_physics_material_is_rejected() -> None:
    model = ArticulatedObject("invalid")
    part = model.part("body", rigid_body=RigidBodyProperties(physics_material="missing"))
    part.visual(Box((1.0, 1.0, 1.0)))

    with pytest.raises(ValidationError, match="unknown physics material"):
        model.validate(strict=True)


def test_usd_export_packages_pbr_texture_networks(tmp_path: Path) -> None:
    texture = tmp_path / "surface.png"
    Image.new("RGB", (2, 2), (128, 64, 255)).save(texture)
    model = ArticulatedObject("textured")
    material = model.material(
        "textured_surface",
        rgba=(1.0, 1.0, 1.0, 1.0),
        base_color_texture="surface.png",
        roughness_texture="surface.png",
        normal_texture="surface.png",
    )
    part = model.part("body")
    part.visual(Box((1.0, 1.0, 1.0)), material=material)

    result = export_object_to_usd(model, tmp_path / "out", asset_root=tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    shader = UsdShade.Shader.Get(stage, "/Asset/Looks/textured_surface/PreviewSurface")
    assert shader.GetInput("diffuseColor").HasConnectedSource()
    assert shader.GetInput("roughness").HasConnectedSource()
    assert shader.GetInput("normal").HasConnectedSource()
    assert (tmp_path / "out" / "isaac" / "textures" / "surface.png").is_file()
    assert Usd.Stage.Open(result.usdz_path.as_posix())


def test_usd_export_opacity_gating_for_base_color_textures(tmp_path: Path) -> None:
    Image.new("RGB", (2, 2), (200, 100, 50)).save(tmp_path / "opaque.png")
    Image.new("RGBA", (2, 2), (200, 100, 50, 255)).save(tmp_path / "opaque_rgba.png")
    Image.new("RGBA", (2, 2), (200, 100, 50, 128)).save(tmp_path / "cutout.png")
    model = ArticulatedObject("opacity")
    opaque = model.material(
        "opaque",
        rgba=(1.0, 1.0, 1.0, 1.0),
        base_color_texture="opaque.png",
        transmission=0.25,
    )
    cutout = model.material(
        "cutout",
        rgba=(1.0, 1.0, 1.0, 1.0),
        base_color_texture="cutout.png",
    )
    opaque_rgba = model.material(
        "opaque_rgba",
        rgba=(1.0, 1.0, 1.0, 1.0),
        base_color_texture="opaque_rgba.png",
    )
    body = model.part("body")
    body.visual(Box((1.0, 1.0, 1.0)), material=opaque, name="opaque_face")
    body.visual(Box((1.0, 1.0, 1.0)), material=opaque_rgba, name="opaque_rgba_face")
    body.visual(Box((1.0, 1.0, 1.0)), material=cutout, name="cutout_face")

    result = export_object_to_usd(model, tmp_path / "out", asset_root=tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())

    opaque_shader = UsdShade.Shader.Get(stage, "/Asset/Looks/opaque/PreviewSurface")
    # An opaque RGB base-color texture must not drive opacity; the scalar opacity
    # (here the transmission approximation 1 * (1 - 0.25)) must survive.
    assert not opaque_shader.GetInput("opacity").HasConnectedSource()
    assert opaque_shader.GetInput("opacity").Get() == pytest.approx(0.75)

    opaque_rgba_shader = UsdShade.Shader.Get(stage, "/Asset/Looks/opaque_rgba/PreviewSurface")
    assert not opaque_rgba_shader.GetInput("opacity").HasConnectedSource()

    cutout_shader = UsdShade.Shader.Get(stage, "/Asset/Looks/cutout/PreviewSurface")
    # An RGBA base-color texture with real transparency keeps the alpha->opacity wiring.
    assert cutout_shader.GetInput("opacity").HasConnectedSource()


@pytest.mark.parametrize(
    ("alpha_mode", "alpha_cutoff", "expected_connection", "expected_threshold"),
    (
        ("OPAQUE", None, False, None),
        ("MASK", None, True, 0.5),
        ("MASK", 0.37, True, 0.37),
        ("BLEND", 0.37, True, None),
    ),
)
def test_usd_export_preserves_imported_gltf_alpha_semantics(
    tmp_path: Path,
    alpha_mode: str,
    alpha_cutoff: float | None,
    expected_connection: bool,
    expected_threshold: float | None,
) -> None:
    import trimesh

    texture = Image.new("RGBA", (2, 2), (200, 100, 50, 255))
    texture.putpixel((0, 0), (200, 100, 50, 64))
    material_kwargs: dict[str, object] = {
        "baseColorTexture": texture,
        "alphaMode": alpha_mode,
    }
    if alpha_cutoff is not None:
        material_kwargs["alphaCutoff"] = alpha_cutoff
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=float),
        material=trimesh.visual.material.PBRMaterial(**material_kwargs),
    )
    source = tmp_path / f"{alpha_mode.lower()}.glb"
    mesh.export(source)

    model = ArticulatedObject(f"imported_{alpha_mode.lower()}")
    model.part("body").visual(Mesh(source.as_posix()), collidable=False)
    stage = Usd.Stage.Open(
        export_object_to_usd(model, tmp_path / "out", asset_root=tmp_path).usdc_path.as_posix()
    )
    shader = UsdShade.Shader.Get(stage, "/Asset/Looks/EmbeddedMaterial_001/PreviewSurface")

    assert shader.GetInput("opacity").HasConnectedSource() is expected_connection
    threshold = shader.GetInput("opacityThreshold")
    if expected_threshold is None:
        assert not threshold
    else:
        assert threshold.Get() == pytest.approx(expected_threshold)


def _local_pose_matrix(pos: object, quat: object) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    imaginary = quat.GetImaginary()
    matrix.SetRotateOnly(
        Gf.Quatd(
            float(quat.GetReal()),
            Gf.Vec3d(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
        )
    )
    matrix.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    return matrix


def test_usd_export_rotated_prismatic_joint_frames_are_consistent(tmp_path: Path) -> None:
    model = ArticulatedObject("rotated", base_mode=BaseMode.FIXED)
    base = model.part("base", inertial=Inertial.from_geometry(Box((0.4, 0.4, 0.4)), mass=2.0))
    base.visual(Box((0.4, 0.4, 0.4)))
    slider = model.part("slider", inertial=Inertial.from_geometry(Box((0.2, 0.2, 0.2)), mass=0.5))
    slider.visual(Box((0.2, 0.2, 0.2)))
    model.articulation(
        "slide",
        ArticulationType.PRISMATIC,
        base,
        slider,
        # Non-identity rpy and a negative axis: the regression target for the
        # body0/body1 joint-frame math.
        origin=Origin(xyz=(0.3, 0.1, 0.0), rpy=(0.0, 0.0, math.pi / 3)),
        axis=(0.0, 0.0, -1.0),
        motion_limits=MotionLimits(lower=0.0, upper=0.5, effort=10.0, velocity=1.0),
        motion_properties=MotionProperties(damping=3.0, friction=0.25),
        physx_joint_resistance=PhysxJointFriction(
            static_friction_effort=6.0,
            dynamic_friction_effort=4.5,
            viscous_friction_coefficient=5.0,
        ),
    )

    result = export_object_to_usd(model, tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    joint = UsdPhysics.PrismaticJoint.Get(stage, "/Asset/Joints/slide")
    assert joint
    assert joint.GetAxisAttr().Get() == UsdPhysics.Tokens.x
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(0.0)
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(0.5)
    drive = UsdPhysics.DriveAPI.Get(stage, "/Asset/Joints/slide.drive:linear")
    assert drive
    assert drive.GetStiffnessAttr().Get() == pytest.approx(0.0)
    assert drive.GetDampingAttr().Get() == pytest.approx(3.0)
    assert drive.GetMaxForceAttr().Get() == pytest.approx(10.0)
    namespace = "physxJointAxis:linear"
    assert joint.GetPrim().GetAttribute(f"{namespace}:maxJointVelocity").Get() == pytest.approx(1.0)
    assert joint.GetPrim().GetAttribute(f"{namespace}:staticFrictionEffort").Get() == pytest.approx(
        6.0
    )
    assert joint.GetPrim().GetAttribute(
        f"{namespace}:dynamicFrictionEffort"
    ).Get() == pytest.approx(4.5)
    assert joint.GetPrim().GetAttribute(
        f"{namespace}:viscousFrictionCoefficient"
    ).Get() == pytest.approx(5.0)

    assert result.newton_usdc_path is not None
    newton = Usd.Stage.Open(result.newton_usdc_path.as_posix())
    newton_joint = newton.GetPrimAtPath("/Asset/Joints/slide")
    assert not newton_joint.HasAPI(UsdPhysics.DriveAPI, UsdPhysics.Tokens.linear)
    assert newton_joint.GetAttribute("mjc:frictionloss").Get() == pytest.approx(6.0)
    assert newton_joint.GetAttribute("mjc:damping").Get() == pytest.approx(8.0)
    assert not newton.GetPrimAtPath("/Asset/Physics/slide_actuator")
    assert any("mjc:frictionloss" in note and note.endswith("(6/4.5)") for note in result.notes)

    cache = UsdGeom.XformCache()
    parent_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/Asset/Links/base"))
    child_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/Asset/Links/slider"))
    frame0 = _local_pose_matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
    frame1 = _local_pose_matrix(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
    world0 = frame0 * parent_world
    world1 = frame1 * child_world
    # The two body-relative joint frames must coincide in world space at the zero pose,
    # otherwise the joint is born stressed/teleported.
    for row in range(4):
        for col in range(4):
            assert world0[row][col] == pytest.approx(world1[row][col], abs=1e-5)


def test_usd_export_packages_texture_canvas_decal(tmp_path: Path) -> None:
    with activate_asset_session(AssetSession(tmp_path)):
        model = ArticulatedObject("labeled")
        body = model.part("body")
        body.visual(Box((1.0, 0.5, 1.0)))
        label = TextureCanvas(64, 64, background=(1.0, 1.0, 1.0, 1.0))
        label.rect((0.1, 0.1, 0.8, 0.8), fill=(0.8, 0.05, 0.02, 1.0))
        body.decal(label, width=0.4, height=0.4, origin=Origin(xyz=(0.0, -0.251, 0.0)))

    result = export_object_to_usd(model, tmp_path / "out", asset_root=tmp_path)

    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    reader = UsdShade.Shader.Get(stage, "/Asset/Looks/EmbeddedMaterial_001/PrimvarReader_st")
    assert reader.GetInput("varname").GetTypeName() == "string"
    assert reader.GetInput("varname").Get() == "st"

    packaged = Usd.Stage.Open(result.usdz_path.as_posix())
    texture = UsdShade.Shader.Get(packaged, "/Asset/Looks/EmbeddedMaterial_001/BaseColorTexture")
    packaged_asset = texture.GetInput("file").Get()
    assert packaged_asset.path.startswith("0/")
    with zipfile.ZipFile(result.usdz_path) as archive:
        assert packaged_asset.path in archive.namelist()


def _obj_with_sliver(path: Path, *, sliver_extent: float) -> Path:
    """A unit box plus an unstitched, zero-area triangle of the given extent far from the box."""

    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in box.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in box.faces]
    base = len(box.vertices)
    # Three collinear vertices: rank 1, zero area — the shape tessellation leaves behind.
    lines += [
        "v 1.000000 0.000000 0.000000",
        f"v {1.0 + sliver_extent / 2:.6f} 0.000000 0.000000",
        f"v {1.0 + sliver_extent:.6f} 0.000000 0.000000",
        f"f {base + 1} {base + 2} {base + 3}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_collision_split_drops_tessellation_slivers_and_keeps_real_components() -> None:
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    sliver = trimesh.Trimesh(
        vertices=[[1.0, 0.0, 0.0], [1.0005, 0.0, 0.0], [1.0009, 0.0, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    sheet = trimesh.Trimesh(
        vertices=[[2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [2.5, 0.5, 0.0], [2.0, 0.5, 0.0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    warnings: list[str] = []
    combined = trimesh.util.concatenate([box, sliver, sliver.copy().apply_translation((0, 1, 0))])
    kept = usd_export._split_collision_components(combined, label="shell", warnings=warnings)
    assert len(kept) == 1 and kept[0].is_watertight
    assert len(warnings) == 1 and "'shell'" in warnings[0]
    # A real planar sheet is not a sliver: it stays and the backend validator reports it.
    warnings.clear()
    kept = usd_export._split_collision_components(
        trimesh.util.concatenate([box, sheet]), label="plate", warnings=warnings
    )
    assert len(kept) == 2 and warnings == []
    # A mesh that is nothing but slivers is passed through unchanged for the validator to name.
    only_slivers = usd_export._split_collision_components(sliver, label="dust", warnings=warnings)
    assert len(only_slivers) == 1 and warnings == []


def test_usd_export_survives_tessellation_slivers_in_a_mesh_collider(tmp_path: Path) -> None:
    mesh_path = _obj_with_sliver(tmp_path / "assets" / "meshes" / "shell.obj", sliver_extent=0.0009)
    model = ArticulatedObject("sliver_shell", assets=tmp_path)
    body = model.part(
        "body",
        rigid_body=RigidBodyProperties(collision_approximation=CollisionApproximation.CONVEX_HULL),
    )
    body.visual(Mesh(mesh_path.as_posix()), name="shell")

    result = export_object_to_usd(model, tmp_path / "out", asset_root=tmp_path)

    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    collision_root = stage.GetPrimAtPath("/Asset/Links/body/Collisions/shell")
    colliders = [
        prim for prim in Usd.PrimRange(collision_root) if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert len(colliders) == 1
    visual_root = stage.GetPrimAtPath("/Asset/Links/body/Visuals/shell")
    visual_points = (
        UsdGeom.Mesh(next(prim for prim in Usd.PrimRange(visual_root) if prim.IsA(UsdGeom.Mesh)))
        .GetPointsAttr()
        .Get()
    )
    # The visual mesh is untouched: it still carries the sliver's three vertices.
    assert len(visual_points) == 8 + 3
    assert any("dropped 1 zero-volume sliver" in warning for warning in result.warnings)


def test_newton_validation_lists_every_degenerate_collider_with_a_repair_hint() -> None:
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Asset/Links/carcass")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(1.0)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
    for name in ("shell_component_000", "trough_component_003"):
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/Links/carcass/Collisions/{name}")
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 1.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/Asset/Links/carcass/Collisions/rib")
    cube.CreateSizeAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    with pytest.raises(ValidationError) as excinfo:
        usd_validate._validate_newton_bodies(stage, Path("model_newton.usdc"))
    message = str(excinfo.value)
    assert "shell_component_000" in message and "trough_component_003" in message
    assert "/Collisions/rib" not in message
