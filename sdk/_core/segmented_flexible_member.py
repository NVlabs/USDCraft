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
from dataclasses import dataclass
from typing import Union, cast

from .articulated_object import ArticulatedObject
from .errors import ValidationError
from .types import (
    Articulation,
    ArticulationType,
    BodyModel,
    Box,
    Cylinder,
    Inertial,
    Material,
    MotionLimits,
    MotionProperties,
    Origin,
    Part,
    PhysxJointFrictionCoefficient,
    RigidBodyProperties,
    Sphere,
    Vec3,
)

SegmentPrimitive = Union[Box, Cylinder, Sphere]

__all__ = [
    "PlanarSegmentJoint",
    "SegmentedFlexibleMember",
    "SegmentedFlexibleMemberSpec",
    "SpatialSegmentJoint",
    "add_segmented_flexible_member",
]


_EPSILON = 1e-9
_SAFE_PITCH_MARGIN_RATIO = 0.01


def _finite(value: float, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValidationError(f"{field_name} must be finite")
    return normalized


def _positive(value: float, *, field_name: str) -> float:
    normalized = _finite(value, field_name=field_name)
    if normalized <= 0.0:
        raise ValidationError(f"{field_name} must be > 0")
    return normalized


def _nonnegative(value: float, *, field_name: str) -> float:
    normalized = _finite(value, field_name=field_name)
    if normalized < 0.0:
        raise ValidationError(f"{field_name} must be >= 0")
    return normalized


def _unit_vec3(value: Vec3, *, field_name: str) -> Vec3:
    if len(value) != 3:
        raise ValidationError(f"{field_name} must have 3 elements")
    vector = tuple(_finite(component, field_name=field_name) for component in value)
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude <= _EPSILON:
        raise ValidationError(f"{field_name} must be non-zero")
    return (
        vector[0] / magnitude,
        vector[1] / magnitude,
        vector[2] / magnitude,
    )


def _segment_length(geometry: SegmentPrimitive) -> float:
    """Return the primitive extent along its local +Z segment axis."""

    if isinstance(geometry, Box):
        if len(geometry.size) != 3:
            raise ValidationError("segment_geometry Box.size must have 3 elements")
        dimensions = tuple(
            _positive(component, field_name="segment_geometry Box.size")
            for component in geometry.size
        )
        return dimensions[2]
    if isinstance(geometry, Cylinder):
        _positive(geometry.radius, field_name="segment_geometry Cylinder.radius")
        return _positive(geometry.length, field_name="segment_geometry Cylinder.length")
    if isinstance(geometry, Sphere):
        radius = _positive(geometry.radius, field_name="segment_geometry Sphere.radius")
        return 2.0 * radius
    raise ValidationError(
        "segment_geometry must be a collidable Box, Cylinder, or Sphere primitive"
    )


def _circumscribed_radius(geometry: SegmentPrimitive) -> float:
    """Return the radius of a sphere enclosing the centered primitive."""

    if isinstance(geometry, Box):
        if len(geometry.size) != 3:
            raise ValidationError("segment_geometry Box.size must have 3 elements")
        dimensions = tuple(
            _positive(component, field_name="segment_geometry Box.size")
            for component in geometry.size
        )
        return 0.5 * math.sqrt(sum(component * component for component in dimensions))
    if isinstance(geometry, Cylinder):
        radius = _positive(geometry.radius, field_name="segment_geometry Cylinder.radius")
        length = _positive(geometry.length, field_name="segment_geometry Cylinder.length")
        return math.sqrt(radius * radius + 0.25 * length * length)
    if isinstance(geometry, Sphere):
        return _positive(geometry.radius, field_name="segment_geometry Sphere.radius")
    raise ValidationError(
        "segment_geometry must be a collidable Box, Cylinder, or Sphere primitive"
    )


def _safe_joint_pitch(geometry: SegmentPrimitive, max_bend: float) -> float:
    """Conservative pitch whose adjacent circumspheres stay separated at max bend."""

    if not 0.0 <= max_bend < math.pi:
        raise ValidationError("maximum segment bend must be in [0, pi)")
    radius = _circumscribed_radius(geometry)
    center_distance_factor = math.cos(0.5 * max_bend)
    if center_distance_factor <= _EPSILON:  # pragma: no cover - guarded by max_bend
        raise ValidationError("maximum segment bend is too close to pi for a finite joint pitch")
    tangent_pitch = (2.0 * radius) / center_distance_factor
    return tangent_pitch * (1.0 + _SAFE_PITCH_MARGIN_RATIO)


def _z_axis_rpy(axis: Vec3) -> Vec3:
    """Return XYZ Euler angles rotating primitive +Z onto ``axis``."""

    x, y, z = axis
    pitch = math.acos(max(-1.0, min(1.0, z)))
    yaw = 0.0 if abs(x) <= _EPSILON and abs(y) <= _EPSILON else math.atan2(y, x)
    return (0.0, pitch, yaw)


@dataclass(frozen=True)
class SpatialSegmentJoint:
    """A native spherical joint for spatial bending between rigid segments.

    ``cone_deg`` is the symmetric bending limit around the member's rest axis.
    OpenUSD has no portable passive damping for spherical joints, so spatial
    members accept only ``None`` or zero damping. Use ``PlanarSegmentJoint``
    when portable damping is required. URDF falls back to a floating joint while
    preserving the spherical semantics in ``usdcraft`` vendor attributes.
    """

    cone_deg: float = 60.0
    damping: float | None = None

    def __post_init__(self) -> None:
        cone_deg = _finite(self.cone_deg, field_name="spatial_joint.cone_deg")
        if not 0.0 < cone_deg < 180.0:
            raise ValidationError("spatial_joint.cone_deg must be in (0, 180)")
        object.__setattr__(self, "cone_deg", cone_deg)
        if self.damping is not None:
            damping = _nonnegative(self.damping, field_name="spatial_joint.damping")
            if damping > 0.0:
                raise ValidationError(
                    "spatial_joint.damping is not portable for spherical joints; "
                    "use PlanarSegmentJoint for damped motion"
                )
        object.__setattr__(self, "damping", None)


@dataclass(frozen=True)
class PlanarSegmentJoint:
    """A native revolute joint for a member constrained to one bending plane."""

    axis: Vec3 = (1.0, 0.0, 0.0)
    lower: float = -math.pi / 2.0
    upper: float = math.pi / 2.0
    effort: float = 10.0
    velocity: float = 5.0
    damping: float = 0.05
    friction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "axis",
            _unit_vec3(self.axis, field_name="planar_joint.axis"),
        )
        lower = _finite(self.lower, field_name="planar_joint.lower")
        upper = _finite(self.upper, field_name="planar_joint.upper")
        if lower >= upper:
            raise ValidationError("planar_joint.lower must be < planar_joint.upper")
        if not lower <= 0.0 <= upper:
            raise ValidationError("planar_joint limits must include q=0 as the rest pose")
        if max(abs(lower), abs(upper)) >= math.pi:
            raise ValidationError("planar_joint maximum bend must be < pi radians")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self,
            "effort",
            _positive(self.effort, field_name="planar_joint.effort"),
        )
        object.__setattr__(
            self,
            "velocity",
            _positive(self.velocity, field_name="planar_joint.velocity"),
        )
        object.__setattr__(
            self,
            "damping",
            _nonnegative(self.damping, field_name="planar_joint.damping"),
        )
        object.__setattr__(
            self,
            "friction",
            _nonnegative(self.friction, field_name="planar_joint.friction"),
        )


SegmentJoint = Union[SpatialSegmentJoint, PlanarSegmentJoint]


@dataclass(frozen=True)
class SegmentedFlexibleMemberSpec:
    """Direct-SDK discretization parameters for a chain, rope, or cable.

    This is an articulated rigid-body approximation, not a live deformable-body
    solver. Every link becomes a collidable ``BodyModel.RIGID`` part with authored
    mass and inertia. At q=0, each primitive is centered in one ``segment_pitch``
    cell along ``rest_axis``. The automatically derived pitch leaves clearance at
    both ends and conservatively separates adjacent circumspheres at the maximum
    joint bend, preventing collision geometry from intersecting as the chain moves.
    Leave ``joint_pitch=None`` to use that minimum automatically, or provide a
    larger explicit pitch when the surrounding assembly needs extra clearance.

    This public type remains available for explicit direct-SDK and downstream use.
    USDCraft Authors (Generation and Reconstruction) must not use it, or reproduce
    its rigid-segment joint-chain topology by hand, for a hose, rope, cable, cord,
    wire, chain, strap, belt, or flexible suspension. USDCraft represents those
    members as one semantic ``BodyModel.FLEXIBLE_1D`` static rest shape and
    currently does not claim distributed flexible dynamics.
    """

    link_count: int
    segment_geometry: SegmentPrimitive
    segment_mass: float
    rest_axis: Vec3 = (0.0, 0.0, -1.0)
    joint: SegmentJoint = SpatialSegmentJoint()
    joint_pitch: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.link_count, bool) or not isinstance(self.link_count, int):
            raise ValidationError("link_count must be an integer")
        if self.link_count < 1:
            raise ValidationError("link_count must be >= 1")
        _segment_length(self.segment_geometry)
        object.__setattr__(
            self,
            "segment_mass",
            _positive(self.segment_mass, field_name="segment_mass"),
        )
        object.__setattr__(
            self,
            "rest_axis",
            _unit_vec3(self.rest_axis, field_name="rest_axis"),
        )
        if not isinstance(self.joint, (SpatialSegmentJoint, PlanarSegmentJoint)):
            raise ValidationError("joint must be a SpatialSegmentJoint or PlanarSegmentJoint")
        if isinstance(self.joint, PlanarSegmentJoint):
            dot = sum(a * b for a, b in zip(self.rest_axis, self.joint.axis))
            if abs(dot) > 1e-6:
                raise ValidationError(
                    "planar_joint.axis must be perpendicular to rest_axis so it bends the member"
                )
        if self.joint_pitch is not None:
            joint_pitch = _positive(self.joint_pitch, field_name="joint_pitch")
            if joint_pitch + _EPSILON < self.minimum_joint_pitch:
                raise ValidationError(
                    f"joint_pitch must be >= collision-safe minimum "
                    f"{self.minimum_joint_pitch:.9g} for the segment geometry and joint limits"
                )
            object.__setattr__(self, "joint_pitch", joint_pitch)

    @property
    def segment_length(self) -> float:
        """Primitive extent along its local +Z segment axis."""

        return _segment_length(self.segment_geometry)

    @property
    def total_mass(self) -> float:
        """Combined authored mass of all rigid segments."""

        return self.link_count * self.segment_mass

    @property
    def segment_circumscribed_radius(self) -> float:
        """Radius of the sphere conservatively enclosing one segment primitive."""

        return _circumscribed_radius(self.segment_geometry)

    @property
    def maximum_bend(self) -> float:
        """Maximum adjacent-joint bend used for pitch safety, in radians."""

        if isinstance(self.joint, SpatialSegmentJoint):
            return math.radians(self.joint.cone_deg)
        return max(abs(self.joint.lower), abs(self.joint.upper))

    @property
    def minimum_joint_pitch(self) -> float:
        """Smallest accepted center-cell pitch for collision-safe adjacent motion."""

        return _safe_joint_pitch(self.segment_geometry, self.maximum_bend)

    @property
    def segment_pitch(self) -> float:
        """Resolved q=0 distance between consecutive joint origins."""

        return self.minimum_joint_pitch if self.joint_pitch is None else self.joint_pitch

    @property
    def total_length(self) -> float:
        """Joint-to-joint rest length from the attachment through the tail cell."""

        return self.link_count * self.segment_pitch


@dataclass(frozen=True)
class SegmentedFlexibleMember:
    """Parts and joints created for one discrete articulated flexible member.

    ``tail_attachment_origin`` is expressed in the tail part frame and locates
    the downstream end of its pitch cell, ready for a payload articulation.
    """

    name: str
    spec: SegmentedFlexibleMemberSpec
    parts: tuple[Part, ...]
    joints: tuple[Articulation, ...]
    tail: Part
    tail_attachment_origin: Origin

    @property
    def head(self) -> Part:
        return self.parts[0]


def add_segmented_flexible_member(
    model: ArticulatedObject,
    name: str,
    parent: str | Part,
    *,
    spec: SegmentedFlexibleMemberSpec,
    attachment_origin: Origin = Origin(),
    material: Material | str | None = None,
    physics_material: str | None = None,
) -> SegmentedFlexibleMember:
    """Attach a direct-SDK segmented chain, rope, or cable to ``parent``.

    Geometry is authored once per rigid segment and remains collidable, allowing
    the normal Host collision derivation to create one primitive collider per
    segment. The return value exposes every created part and joint plus ``tail``
    so callers can attach a payload without reconstructing generated names.

    This helper is not a USDCraft Author surface (Generation or Reconstruction).
    USDCraft uses a single ``BodyModel.FLEXIBLE_1D`` static rest shape for flexible
    members and reports genuinely required distributed flexible dynamics as
    unsupported.
    """

    member_name = str(name).strip()
    if not member_name:
        raise ValidationError("segmented flexible member name must be non-empty")
    if not isinstance(spec, SegmentedFlexibleMemberSpec):
        raise ValidationError("spec must be a SegmentedFlexibleMemberSpec")
    if not isinstance(attachment_origin, Origin):
        raise ValidationError("attachment_origin must be an Origin")

    parent_part = model.get_part(parent)
    if parent_part.body_model != BodyModel.RIGID:
        raise ValidationError(
            "segmented flexible member parent must be BodyModel.RIGID so the attachment "
            "uses a native physics joint"
        )
    if parent_part.rigid_body is not None and not parent_part.rigid_body.enabled:
        raise ValidationError("segmented flexible member parent cannot disable its rigid body")
    if isinstance(material, str) and material not in {item.name for item in model.materials}:
        raise ValidationError(f"Unknown visual material: {material!r}")
    normalized_physics_material: str | None = None
    if physics_material is not None:
        normalized_physics_material = str(physics_material).strip()
        if not normalized_physics_material:
            raise ValidationError("physics_material must be non-empty when provided")
        if normalized_physics_material not in {item.name for item in model.physics_materials}:
            raise ValidationError(f"Unknown physics material: {normalized_physics_material!r}")

    part_names = tuple(f"{member_name}_link_{index:03d}" for index in range(spec.link_count))
    joint_names = tuple(f"{member_name}_joint_{index:03d}" for index in range(spec.link_count))
    existing_part_names = {part.name for part in model.parts}
    existing_joint_names = {joint.name for joint in model.articulations}
    duplicate_parts = sorted(existing_part_names.intersection(part_names))
    duplicate_joints = sorted(existing_joint_names.intersection(joint_names))
    if duplicate_parts or duplicate_joints:
        conflicts = duplicate_parts + duplicate_joints
        raise ValidationError(
            "Segmented flexible member generated names already exist: "
            + ", ".join(repr(item) for item in conflicts)
        )

    pitch = spec.segment_pitch
    step = cast(
        tuple[float, float, float], tuple(component * pitch for component in spec.rest_axis)
    )
    center = cast(
        tuple[float, float, float],
        tuple(component * (pitch / 2.0) for component in spec.rest_axis),
    )
    segment_origin = Origin(xyz=center, rpy=_z_axis_rpy(spec.rest_axis))
    downstream_origin = Origin(xyz=step)
    rigid_body = RigidBodyProperties(
        enabled=True,
        kinematic=False,
        physics_material=normalized_physics_material,
    )
    common_meta = {
        "usdcraft:flexibleMember": member_name,
        "usdcraft:simulationRepresentation": "articulated_rigid_segments",
    }

    created_parts: list[Part] = []
    created_joints: list[Articulation] = []
    joint_parent = parent_part
    for index, (part_name, joint_name) in enumerate(zip(part_names, joint_names)):
        part = model.part(
            part_name,
            inertial=Inertial.from_geometry(
                spec.segment_geometry,
                mass=spec.segment_mass,
                origin=segment_origin,
            ),
            rigid_body=rigid_body,
            body_model=BodyModel.RIGID,
            meta={
                **common_meta,
                "usdcraft:segmentIndex": index,
                "usdcraft:segmentCount": spec.link_count,
            },
        )
        part.visual(
            spec.segment_geometry,
            origin=segment_origin,
            material=material,
            name=f"{part_name}_collision_shape",
            collidable=True,
        )

        origin = attachment_origin if index == 0 else downstream_origin
        if isinstance(spec.joint, SpatialSegmentJoint):
            joint = model.articulation(
                joint_name,
                ArticulationType.SPHERICAL,
                parent=joint_parent,
                child=part,
                origin=origin,
                axis=spec.rest_axis,
                cone_deg=spec.joint.cone_deg,
                meta={**common_meta, "usdcraft:jointMode": "spatial"},
            )
        else:
            motion_properties = MotionProperties(
                damping=spec.joint.damping,
                friction=spec.joint.friction,
            )
            joint = model.articulation(
                joint_name,
                ArticulationType.REVOLUTE,
                parent=joint_parent,
                child=part,
                origin=origin,
                axis=spec.joint.axis,
                motion_limits=MotionLimits(
                    lower=spec.joint.lower,
                    upper=spec.joint.upper,
                    effort=spec.joint.effort,
                    velocity=spec.joint.velocity,
                ),
                motion_properties=motion_properties,
                physx_joint_resistance=PhysxJointFrictionCoefficient(spec.joint.friction),
                meta={**common_meta, "usdcraft:jointMode": "planar"},
            )
        created_parts.append(part)
        created_joints.append(joint)
        joint_parent = part

    return SegmentedFlexibleMember(
        name=member_name,
        spec=spec,
        parts=tuple(created_parts),
        joints=tuple(created_joints),
        tail=created_parts[-1],
        tail_attachment_origin=downstream_origin,
    )
