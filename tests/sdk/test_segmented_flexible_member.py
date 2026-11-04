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
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pxr import Usd, UsdPhysics

from sdk._core._urdf_export import compile_object_to_urdf_xml
from sdk._core._usd_export import export_object_to_usd
from sdk._core.articulated_object import ArticulatedObject
from sdk._core.errors import ValidationError
from sdk._core.exact_collisions import compile_object_model_with_exact_collisions
from sdk._core.segmented_flexible_member import (
    PlanarSegmentJoint,
    SegmentedFlexibleMemberSpec,
    SpatialSegmentJoint,
    add_segmented_flexible_member,
)
from sdk._core.testing import TestContext as SDKTestContext
from sdk._core.types import (
    ArticulationType,
    BaseMode,
    BodyModel,
    Box,
    Cylinder,
    Inertial,
    MotionLimits,
    MotionProperties,
    Origin,
    Part,
    RigidBodyProperties,
    Sphere,
)


def _base_model() -> tuple[ArticulatedObject, Part]:
    model = ArticulatedObject("segmented_member", base_mode=BaseMode.FIXED)
    model.material("steel", rgba=(0.25, 0.27, 0.3, 1.0))
    model.physics_material(
        "chain_contact",
        static_friction=0.55,
        dynamic_friction=0.45,
        restitution=0.05,
    )
    base = model.part(
        "support",
        inertial=Inertial.from_geometry(Box((0.3, 0.3, 0.1)), mass=5.0),
        rigid_body=RigidBodyProperties(kinematic=True),
    )
    base.visual(Box((0.3, 0.3, 0.1)), name="support_shape")
    return model, base


def test_spatial_member_builds_collidable_rigid_segments_and_native_joints() -> None:
    model, base = _base_model()
    spec = SegmentedFlexibleMemberSpec(
        link_count=3,
        segment_geometry=Cylinder(radius=0.012, length=0.12),
        segment_mass=0.04,
        rest_axis=(0.0, 0.0, -2.0),
        joint=SpatialSegmentJoint(cone_deg=50.0),
    )

    member = add_segmented_flexible_member(
        model,
        "hoist_chain",
        base,
        spec=spec,
        attachment_origin=Origin(xyz=(0.0, 0.0, 0.4)),
        material="steel",
        physics_material="chain_contact",
    )

    assert member.name == "hoist_chain"
    assert member.parts == tuple(model.parts[1:])
    assert member.joints == tuple(model.articulations)
    assert member.head.name == "hoist_chain_link_000"
    assert member.tail is member.parts[-1]
    assert spec.segment_length == pytest.approx(0.12)
    assert spec.total_mass == pytest.approx(0.12)
    assert spec.segment_pitch == pytest.approx(spec.minimum_joint_pitch)
    assert spec.segment_pitch > spec.segment_length
    assert spec.total_length == pytest.approx(3.0 * spec.segment_pitch)
    assert member.tail_attachment_origin.xyz == pytest.approx((0.0, 0.0, -spec.segment_pitch))

    for index, part in enumerate(member.parts):
        assert part.body_model == BodyModel.RIGID
        assert part.rigid_body == RigidBodyProperties(
            enabled=True,
            kinematic=False,
            physics_material="chain_contact",
        )
        assert part.inertial is not None
        assert part.inertial.mass == pytest.approx(0.04)
        assert part.inertial.origin.xyz == pytest.approx((0.0, 0.0, -spec.segment_pitch / 2.0))
        assert part.inertial.origin.rpy == pytest.approx((0.0, math.pi, 0.0))
        assert all(
            value > 0.0
            for value in (
                part.inertial.inertia.ixx,
                part.inertial.inertia.iyy,
                part.inertial.inertia.izz,
            )
        )
        assert len(part.visuals) == 1
        assert part.visuals[0].collidable is True
        assert part.visuals[0].geometry == spec.segment_geometry
        assert part.meta["usdcraft:segmentIndex"] == index
        assert part.meta["usdcraft:simulationRepresentation"] == ("articulated_rigid_segments")

    for index, joint in enumerate(member.joints):
        assert joint.articulation_type == ArticulationType.SPHERICAL
        assert joint.parent == ("support" if index == 0 else member.parts[index - 1].name)
        assert joint.child == member.parts[index].name
        assert joint.axis == pytest.approx((0.0, 0.0, -1.0))
        assert joint.cone_deg == pytest.approx(50.0)
        assert joint.motion_limits is None
        assert joint.motion_properties is None
        assert joint.origin.xyz == pytest.approx(
            (0.0, 0.0, 0.4) if index == 0 else (0.0, 0.0, -spec.segment_pitch)
        )

    model.validate(strict=True)
    compiled = compile_object_model_with_exact_collisions(model)
    for source, physical in zip(member.parts, compiled.parts[1:]):
        assert source.collisions == []
        assert len(physical.collisions) == 1
        assert physical.collisions[0].geometry == spec.segment_geometry
        assert physical.collisions[0].origin == source.visuals[0].origin


def test_planar_member_uses_damped_revolute_joints_and_exposes_tail() -> None:
    model, base = _base_model()
    spec = SegmentedFlexibleMemberSpec(
        link_count=2,
        segment_geometry=Box((0.018, 0.025, 0.15)),
        segment_mass=0.06,
        rest_axis=(0.0, -1.0, 0.0),
        joint=PlanarSegmentJoint(
            axis=(1.0, 0.0, 0.0),
            lower=-0.65,
            upper=0.8,
            effort=7.0,
            velocity=3.0,
            damping=0.18,
        ),
    )

    member = add_segmented_flexible_member(model, "belt", base, spec=spec)
    payload = model.part(
        "payload",
        inertial=Inertial.from_geometry(Box((0.08, 0.08, 0.04)), mass=0.2),
        rigid_body=RigidBodyProperties(),
    )
    payload.visual(Box((0.08, 0.08, 0.04)), name="payload_shape")
    model.articulation(
        "belt_to_payload",
        ArticulationType.FIXED,
        parent=member.tail,
        child=payload,
        origin=member.tail_attachment_origin,
    )

    for index, joint in enumerate(member.joints):
        assert joint.articulation_type == ArticulationType.REVOLUTE
        assert joint.axis == pytest.approx((1.0, 0.0, 0.0))
        assert joint.motion_limits == MotionLimits(
            lower=-0.65,
            upper=0.8,
            effort=7.0,
            velocity=3.0,
        )
        assert joint.motion_properties is not None
        assert joint.motion_properties.damping == pytest.approx(0.18)
        if index > 0:
            assert joint.origin.xyz == pytest.approx((0.0, -spec.segment_pitch, 0.0))

    assert member.parts[0].visuals[0].origin.xyz == pytest.approx(
        (0.0, -spec.segment_pitch / 2.0, 0.0)
    )
    model.validate(strict=True)


def test_spatial_usdc_is_native_and_urdf_fallback_preserves_semantics(
    tmp_path: Path,
) -> None:
    model, base = _base_model()
    member = add_segmented_flexible_member(
        model,
        "cable",
        base,
        spec=SegmentedFlexibleMemberSpec(
            link_count=2,
            segment_geometry=Cylinder(radius=0.01, length=0.1),
            segment_mass=0.025,
            joint=SpatialSegmentJoint(cone_deg=35.0),
        ),
    )

    result = export_object_to_usd(model, tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    assert stage
    assert stage.GetPrimAtPath("/Asset/Joints/__root_fixed").HasAPI(UsdPhysics.ArticulationRootAPI)

    for index, part in enumerate(member.parts):
        part_path = f"/Asset/Links/cable_link_{index:03d}"
        prim = stage.GetPrimAtPath(part_path)
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert prim.HasAPI(UsdPhysics.MassAPI)
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(0.025)
        collider = stage.GetPrimAtPath(
            f"{part_path}/Collisions/cable_link_{index:03d}_collision_shape/Geom"
        )
        assert collider.HasAPI(UsdPhysics.CollisionAPI)

        joint_path = f"/Asset/Joints/cable_joint_{index:03d}"
        joint = UsdPhysics.SphericalJoint.Get(stage, joint_path)
        assert joint
        assert joint.GetCollisionEnabledAttr().HasAuthoredValueOpinion()
        assert joint.GetCollisionEnabledAttr().Get() is False
        assert joint.GetConeAngle0LimitAttr().Get() == pytest.approx(35.0)
        assert joint.GetConeAngle1LimitAttr().Get() == pytest.approx(35.0)
        assert tuple(joint.GetLocalPos0Attr().Get()) == pytest.approx(
            (0.0, 0.0, 0.0) if index == 0 else (0.0, 0.0, -member.spec.segment_pitch)
        )
        assert not UsdPhysics.DriveAPI.Get(stage, f"{joint_path}.drive:angular")

    root = ET.fromstring(compile_object_to_urdf_xml(model))
    for index in range(2):
        joint = root.find(f"joint[@name='cable_joint_{index:03d}']")
        assert joint is not None
        assert joint.attrib == {
            "name": f"cable_joint_{index:03d}",
            "type": "floating",
            "usdcraft_type": "spherical",
            "usdcraft_axis": "0 0 -1",
            "usdcraft_cone_deg": "35",
        }
        assert joint.find("dynamics") is None
        origin = joint.find("origin")
        if index == 0:
            assert origin is None
        else:
            assert origin is not None
            assert tuple(float(value) for value in origin.attrib["xyz"].split()) == pytest.approx(
                (0.0, 0.0, -member.spec.segment_pitch),
                abs=1e-6,
            )


def test_planar_member_exports_limits_and_damping_to_usdc_and_urdf(tmp_path: Path) -> None:
    model, base = _base_model()
    member = add_segmented_flexible_member(
        model,
        "planar_rope",
        base,
        spec=SegmentedFlexibleMemberSpec(
            link_count=1,
            segment_geometry=Sphere(radius=0.03),
            segment_mass=0.02,
            joint=PlanarSegmentJoint(
                lower=-0.4,
                upper=0.55,
                effort=4.0,
                velocity=2.5,
                damping=0.14,
                friction=0.06,
            ),
        ),
    )

    result = export_object_to_usd(model, tmp_path)
    stage = Usd.Stage.Open(result.usdc_path.as_posix())
    assert stage
    joint_path = "/Asset/Joints/planar_rope_joint_000"
    joint = UsdPhysics.RevoluteJoint.Get(stage, joint_path)
    assert joint
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(math.degrees(-0.4))
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(math.degrees(0.55))
    drive = UsdPhysics.DriveAPI.Get(stage, f"{joint_path}.drive:angular")
    assert drive
    assert drive.GetStiffnessAttr().Get() == pytest.approx(0.0)
    assert drive.GetDampingAttr().Get() == pytest.approx(0.14 * math.pi / 180.0)
    assert not any("passive damping" in warning for warning in result.warnings)
    joint_prim = stage.GetPrimAtPath(joint_path)
    assert joint_prim.GetAttribute("physxJoint:jointFriction").Get() == pytest.approx(0.06)

    root = ET.fromstring(compile_object_to_urdf_xml(model))
    urdf_joint = root.find(f"joint[@name='{member.joints[0].name}']")
    assert urdf_joint is not None
    assert urdf_joint.attrib["type"] == "revolute"
    assert urdf_joint.find("limit").attrib == {
        "effort": "4",
        "velocity": "2.5",
        "lower": "-0.4",
        "upper": "0.55",
    }
    assert urdf_joint.find("dynamics").attrib == {"damping": "0.14", "friction": "0.06"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "link_count": 0,
                "segment_geometry": Cylinder(0.01, 0.1),
                "segment_mass": 0.1,
            },
            "link_count must be >= 1",
        ),
        (
            {
                "link_count": 2,
                "segment_geometry": Cylinder(0.01, 0.1),
                "segment_mass": 0.0,
            },
            "segment_mass must be > 0",
        ),
        (
            {
                "link_count": 2,
                "segment_geometry": Cylinder(-0.01, 0.1),
                "segment_mass": 0.1,
            },
            "Cylinder.radius must be > 0",
        ),
        (
            {
                "link_count": 2,
                "segment_geometry": Cylinder(0.01, 0.1),
                "segment_mass": 0.1,
                "rest_axis": (0.0, 0.0, 0.0),
            },
            "rest_axis must be non-zero",
        ),
        (
            {
                "link_count": 2,
                "segment_geometry": Cylinder(0.01, 0.1),
                "segment_mass": 0.1,
                "joint": PlanarSegmentJoint(axis=(0.0, 0.0, 1.0)),
            },
            "must be perpendicular to rest_axis",
        ),
        (
            {
                "link_count": 2,
                "segment_geometry": Cylinder(0.01, 0.1),
                "segment_mass": 0.1,
                "joint_pitch": 0.1,
            },
            "joint_pitch must be >= collision-safe minimum",
        ),
    ],
)
def test_spec_rejects_nonphysical_parameters(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SegmentedFlexibleMemberSpec(**kwargs)  # type: ignore[arg-type]


def test_spatial_joint_rejects_nonportable_damping() -> None:
    with pytest.raises(ValidationError, match="not portable for spherical joints"):
        SpatialSegmentJoint(damping=0.1)

    assert SpatialSegmentJoint(damping=0.0).damping is None


def test_joint_bend_must_stay_below_pi_for_finite_safe_pitch() -> None:
    with pytest.raises(ValidationError, match=r"cone_deg must be in \(0, 180\)"):
        SpatialSegmentJoint(cone_deg=180.0)
    with pytest.raises(ValidationError, match="maximum bend must be < pi"):
        PlanarSegmentJoint(lower=-0.1, upper=math.pi)


def test_explicit_collision_safe_joint_pitch_is_preserved() -> None:
    automatic = SegmentedFlexibleMemberSpec(
        link_count=3,
        segment_geometry=Cylinder(0.01, 0.1),
        segment_mass=0.02,
        joint=PlanarSegmentJoint(lower=-0.8, upper=0.8),
    )
    explicit_pitch = 1.2 * automatic.minimum_joint_pitch
    explicit = SegmentedFlexibleMemberSpec(
        link_count=3,
        segment_geometry=Cylinder(0.01, 0.1),
        segment_mass=0.02,
        joint=PlanarSegmentJoint(lower=-0.8, upper=0.8),
        joint_pitch=explicit_pitch,
    )

    assert explicit.joint_pitch == pytest.approx(explicit_pitch)
    assert explicit.segment_pitch == pytest.approx(explicit_pitch)
    assert explicit.total_length == pytest.approx(3.0 * explicit_pitch)


def test_planar_auto_pitch_passes_exhaustive_sampled_pose_overlap_check() -> None:
    model, base = _base_model()
    base.visual(
        Cylinder(radius=0.004, length=0.24),
        origin=Origin(xyz=(0.0, 0.0, -0.17)),
        name="planar_hanger_post",
    )
    spec = SegmentedFlexibleMemberSpec(
        link_count=4,
        segment_geometry=Cylinder(radius=0.012, length=0.1),
        segment_mass=0.03,
        joint=PlanarSegmentJoint(lower=-0.8, upper=0.8, damping=0.1),
    )
    add_segmented_flexible_member(
        model,
        "planar_chain",
        base,
        spec=spec,
        attachment_origin=Origin(xyz=(0.0, 0.0, -0.3)),
    )
    model.validate(strict=True)

    context = SDKTestContext(model, seed=7)
    passed = context.fail_if_parts_overlap_in_sampled_poses(
        max_pose_samples=128,
        overlap_tol=1e-6,
        overlap_volume_tol=0.0,
        ignore_adjacent=False,
        ignore_fixed=True,
    )
    assert passed, context.report()


def test_spatial_auto_pitch_passes_deterministic_spherical_qc_slice() -> None:
    model, base = _base_model()
    base.visual(
        Cylinder(radius=0.004, length=0.24),
        origin=Origin(xyz=(0.0, 0.0, -0.17)),
        name="spatial_hanger_post",
    )
    spec = SegmentedFlexibleMemberSpec(
        link_count=4,
        segment_geometry=Cylinder(radius=0.012, length=0.1),
        segment_mass=0.03,
        joint=SpatialSegmentJoint(cone_deg=math.degrees(0.8)),
    )
    add_segmented_flexible_member(
        model,
        "spatial_chain",
        base,
        spec=spec,
        attachment_origin=Origin(xyz=(0.0, 0.0, -0.3)),
    )
    model.validate(strict=True)

    context = SDKTestContext(model, seed=7)
    passed = context.fail_if_parts_overlap_in_sampled_poses(
        max_pose_samples=128,
        overlap_tol=1e-6,
        overlap_volume_tol=0.0,
        ignore_adjacent=False,
        ignore_fixed=True,
    )
    assert passed, context.report()


def test_zero_planar_damping_still_authors_effort_and_velocity_limits(tmp_path: Path) -> None:
    model, base = _base_model()
    member = add_segmented_flexible_member(
        model,
        "undamped_rope",
        base,
        spec=SegmentedFlexibleMemberSpec(
            link_count=1,
            segment_geometry=Cylinder(0.01, 0.1),
            segment_mass=0.02,
            joint=PlanarSegmentJoint(damping=0.0),
        ),
    )

    assert member.joints[0].motion_properties == MotionProperties(damping=0.0, friction=0.0)
    stage = Usd.Stage.Open(export_object_to_usd(model, tmp_path).usdc_path.as_posix())
    assert stage
    drive = UsdPhysics.DriveAPI.Get(
        stage,
        "/Asset/Joints/undamped_rope_joint_000.drive:angular",
    )
    assert drive
    assert drive.GetStiffnessAttr().Get() == pytest.approx(0.0)
    assert drive.GetDampingAttr().Get() == pytest.approx(0.0)
    assert drive.GetMaxForceAttr().Get() == pytest.approx(10.0)
    joint = stage.GetPrimAtPath("/Asset/Joints/undamped_rope_joint_000")
    assert joint.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(
        math.degrees(5.0)
    )


def test_duplicate_member_name_is_rejected_without_partial_mutation() -> None:
    model, base = _base_model()
    spec = SegmentedFlexibleMemberSpec(
        link_count=2,
        segment_geometry=Cylinder(0.01, 0.1),
        segment_mass=0.02,
    )
    add_segmented_flexible_member(model, "chain", base, spec=spec)
    counts = (len(model.parts), len(model.articulations))

    with pytest.raises(ValidationError, match="generated names already exist"):
        add_segmented_flexible_member(model, "chain", base, spec=spec)

    assert (len(model.parts), len(model.articulations)) == counts


def test_member_rejects_nonrigid_parent_before_mutation() -> None:
    model = ArticulatedObject("invalid_parent")
    hose = model.part("hose", body_model=BodyModel.FLEXIBLE_1D)
    spec = SegmentedFlexibleMemberSpec(
        link_count=2,
        segment_geometry=Cylinder(0.01, 0.1),
        segment_mass=0.02,
    )

    with pytest.raises(ValidationError, match="parent must be BodyModel.RIGID"):
        add_segmented_flexible_member(model, "chain", hose, spec=spec)

    assert model.parts == [hose]
    assert model.articulations == []
