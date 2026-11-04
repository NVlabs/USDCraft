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

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pxr import Usd, UsdPhysics

from sdk import (
    ArticulatedObject,
    ArticulationType,
    BodyModel,
    Box,
    Inertial,
    MotionLimits,
    RigidBodyProperties,
    ValidationError,
)
from sdk._core._urdf_export import compile_object_to_urdf_xml
from sdk._core._usd_export import export_object_to_usd
from sdk._core.exact_collisions import compile_object_model_with_exact_collisions


def test_part_body_model_defaults_and_string_coercion() -> None:
    model = ArticulatedObject("body_models")

    assert model.part("rigid").body_model == BodyModel.RIGID
    flexible = model.link("hose", body_model="flexible_1d")
    assert flexible.body_model == BodyModel.FLEXIBLE_1D
    assert flexible.body_realization == "static_rest_pose"

    with pytest.raises(ValidationError, match="Unknown body model"):
        model.part("unknown", body_model="clothish")


def test_validation_rejects_nonrigid_physics_and_movable_articulations() -> None:
    model = ArticulatedObject("nonrigid_joint")
    base = model.part("base")
    hose = model.part("hose", body_model=BodyModel.FLEXIBLE_1D)
    model.articulation(
        "move_hose",
        ArticulationType.REVOLUTE,
        parent=base,
        child=hose,
        motion_limits=MotionLimits(lower=-0.2, upper=0.2),
    )

    with pytest.raises(ValidationError, match="must be fixed.*non-rigid"):
        model.validate(strict=True)

    standalone = ArticulatedObject("nonrigid_physics")
    standalone.part(
        "cloth",
        body_model=BodyModel.SURFACE_DEFORMABLE,
        inertial=Inertial.from_geometry(Box((0.2, 0.2, 0.01)), mass=0.1),
    )
    with pytest.raises(ValidationError, match="cannot define inertial"):
        standalone.validate(strict=True)

    rigid_properties = ArticulatedObject("nonrigid_rigid_properties")
    rigid_properties.part(
        "foam",
        body_model=BodyModel.VOLUME_DEFORMABLE,
        rigid_body=RigidBodyProperties(),
    )
    with pytest.raises(ValidationError, match="cannot define rigid_body"):
        rigid_properties.validate(strict=True)


def test_bounded_motion_limits_must_include_authored_zero_pose() -> None:
    model = ArticulatedObject("invalid_zero_pose")
    base = model.part("base")
    slider = model.part("slider")
    model.articulation(
        "slide",
        ArticulationType.PRISMATIC,
        parent=base,
        child=slider,
        motion_limits=MotionLimits(lower=0.1, upper=0.5),
    )

    with pytest.raises(ValidationError, match="must include q=0"):
        model.validate(strict=True)


def test_exact_collision_copy_retains_nonrigid_type_for_host_qc() -> None:
    model = ArticulatedObject("host_qc")
    hose = model.part("hose", body_model=BodyModel.FLEXIBLE_1D)
    hose.visual(Box((0.5, 0.02, 0.02)), name="hose_rest_shape")

    compiled = compile_object_model_with_exact_collisions(model)

    assert compiled.parts[0].body_model == BodyModel.FLEXIBLE_1D
    assert len(compiled.parts[0].collisions) == 1
    assert compiled.parts[0].collisions[0].name == "hose_rest_shape"


@pytest.mark.parametrize(
    "body_model",
    [
        BodyModel.FLEXIBLE_1D,
        BodyModel.SURFACE_DEFORMABLE,
        BodyModel.VOLUME_DEFORMABLE,
    ],
)
def test_urdf_records_nonrigid_rest_shape_collision_geometry(body_model: BodyModel) -> None:
    model = ArticulatedObject("typed_urdf")
    hose = model.part("hose", body_model=body_model)
    hose.visual(Box((0.5, 0.02, 0.02)), name="hose_rest_shape")

    root = ET.fromstring(compile_object_to_urdf_xml(model))
    link = root.find("link")

    assert link is not None
    assert link.attrib["usdcraft_body_model"] == body_model.value
    assert link.attrib["usdcraft_body_realization"] == "static_rest_pose"
    assert link.find("visual") is not None
    assert link.find("inertial") is None
    assert link.find("collision") is None
    rest_shape = link.find("usdcraft_collision")
    assert rest_shape is not None
    assert rest_shape.attrib == {
        "body_model": body_model.value,
        "role": "deformable_contact",
        "realization": "rest_shape",
        "name": "hose_rest_shape",
    }
    assert rest_shape.find("geometry/box").attrib == {"size": "0.5 0.02 0.02"}


def test_urdf_keeps_rigid_collisions_standard_only() -> None:
    model = ArticulatedObject("rigid_urdf")
    body = model.part("body")
    body.visual(Box((0.4, 0.3, 0.2)), name="body_shape")

    root = ET.fromstring(compile_object_to_urdf_xml(model))
    link = root.find("link")

    assert link is not None
    assert link.find("collision") is not None
    assert link.find("usdcraft_collision") is None


def test_urdf_visual_only_export_omits_deformable_rest_shape_collision() -> None:
    model = ArticulatedObject("visual_only_urdf")
    hose = model.part("hose", body_model=BodyModel.FLEXIBLE_1D)
    hose.visual(Box((0.5, 0.02, 0.02)), name="hose_rest_shape")

    root = ET.fromstring(compile_object_to_urdf_xml(model, include_physical_collisions=False))
    link = root.find("link")

    assert link is not None
    assert link.find("visual") is not None
    assert link.find("collision") is None
    assert link.find("usdcraft_collision") is None


@pytest.mark.parametrize(
    "body_model",
    [
        BodyModel.FLEXIBLE_1D,
        BodyModel.SURFACE_DEFORMABLE,
        BodyModel.VOLUME_DEFORMABLE,
    ],
)
def test_usd_uses_static_rest_presentation_for_nonrigid_attachment(
    tmp_path: Path,
    body_model: BodyModel,
) -> None:
    model = ArticulatedObject("typed_usd")
    body = model.part("body")
    body.visual(Box((0.2, 0.2, 0.4)), name="shell")
    hose = model.part("hose", body_model=body_model)
    hose.visual(Box((0.5, 0.02, 0.02)), name="hose_rest_shape")
    model.articulation(
        "body_to_hose",
        ArticulationType.FIXED,
        parent=body,
        child=hose,
    )

    result = export_object_to_usd(model, tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    assert stage
    body_prim = stage.GetPrimAtPath("/Asset/Links/body")
    hose_prim = stage.GetPrimAtPath("/Asset/Links/hose")

    assert body_prim.GetCustomDataByKey("usdcraft:bodyModel") == "rigid"
    assert body_prim.GetCustomDataByKey("usdcraft:bodyRealization") == "native"
    assert hose_prim.GetCustomDataByKey("usdcraft:bodyModel") == body_model.value
    assert hose_prim.GetCustomDataByKey("usdcraft:bodyRealization") == "static_rest_pose"
    assert not hose_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not hose_prim.HasAPI(UsdPhysics.MassAPI)
    assert all("Deformable" not in schema for schema in hose_prim.GetAppliedSchemas())
    body_collider = stage.GetPrimAtPath("/Asset/Links/body/Collisions/shell/Geom")
    hose_collision = stage.GetPrimAtPath("/Asset/Links/hose/Collisions/hose_rest_shape")
    hose_collider = stage.GetPrimAtPath("/Asset/Links/hose/Collisions/hose_rest_shape/Geom")
    assert body_collider.HasAPI(UsdPhysics.CollisionAPI)
    assert hose_collision.GetCustomDataByKey("usdcraft:collisionRole") == "deformable_contact"
    assert hose_collision.GetCustomDataByKey("usdcraft:collisionRealization") == "rest_shape"
    assert hose_collision.GetCustomDataByKey("usdcraft:bodyModel") == body_model.value
    assert hose_collision.GetCustomDataByKey("usdcraft:runtimeCollisionEnabled") is False
    assert hose_collider
    assert not hose_collider.HasAPI(UsdPhysics.CollisionAPI)
    assert not hose_collider.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert not stage.GetPrimAtPath("/Asset/Joints/body_to_hose")
    assert hose_prim.GetCustomDataByKey("usdcraft:attachment") == {
        "articulation": "body_to_hose",
        "parent": "body",
        "child": "hose",
        "type": "fixed",
        "realization": "static_rest_pose",
    }
    assert not stage.GetPrimAtPath("/Asset").HasAPI(UsdPhysics.ArticulationRootAPI)
    assert result.metrics["body_model_counts"] == {
        "rigid": 1,
        "flexible_1d": int(body_model == BodyModel.FLEXIBLE_1D),
        "surface_deformable": int(body_model == BodyModel.SURFACE_DEFORMABLE),
        "volume_deformable": int(body_model == BodyModel.VOLUME_DEFORMABLE),
    }
    assert result.metrics["collision_shape_counts"] == {
        "rigid_runtime": 1,
        "deformable_rest_shape": 1,
        "deformable_live": 0,
    }
    assert result.metrics["collision_geometry_derivation_enabled"] is True
    assert "physics_collisions_included" not in result.metrics
    packaged = Usd.Stage.Open(result.usdz_path.as_posix())
    assert packaged
    packaged_hose_collision = packaged.GetPrimAtPath("/Asset/Links/hose/Collisions/hose_rest_shape")
    packaged_hose_collider = packaged.GetPrimAtPath(
        "/Asset/Links/hose/Collisions/hose_rest_shape/Geom"
    )
    assert (
        packaged_hose_collision.GetCustomDataByKey("usdcraft:collisionRealization") == "rest_shape"
    )
    assert not packaged_hose_collider.HasAPI(UsdPhysics.CollisionAPI)
    # Static rest-pose export of a non-rigid body model is a fixed exporter capability, reported
    # as aggregated informational notes rather than per-part warnings.
    assert not result.warnings
    assert any(
        "runtime deformation is not represented" in note and f"hose ({body_model.value})" in note
        for note in result.notes
    )
    assert any(
        "no native physics joint is authored" in note and "body_to_hose" in note
        for note in result.notes
    )
