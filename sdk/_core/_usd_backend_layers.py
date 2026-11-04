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

"""PhysX and Newton/MuJoCo backend layers and the Asset Structure interface layer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

from ._usd_common import _MJC_PARAMETER_COUNT, _apply_api_schema_token, _joint_type
from ._usd_validate import _resolve_native_articulation_root_bindings
from .articulated_object import ArticulatedObject
from .errors import ValidationError
from .types import ArticulationType, PhysxJointFriction, PhysxJointFrictionCoefficient

# One note per kind, aggregated over joints. Each kind is a fixed capability boundary of the Newton
# (MuJoCo) or PhysX layer that the SDK documentation already states; the affected values always
# stay available in the exported customData. Because every conforming joint triggers them, they are
# not warnings an Author or user can act on, so they are reported as informational notes and kept
# out of the compile warnings.
_BACKEND_NOTE_TEMPLATES: dict[str, str] = {
    "mimic": (
        "Mimic articulations are preserved as portable customData `usdcraft:mimic`; the Newton "
        "layer also authors NewtonMimicAPI while the PhysX layer keeps metadata-only mimic "
        "semantics: {joints}"
    ),
    "acceleration_drive": (
        "Newton/MuJoCo has no verified Isaac Sim 6 MJC equivalent for acceleration-mode drives; "
        "their actuators are emitted for direct force control without static gain/bias "
        "parameters: {joints}"
    ),
    "pure_p_drive": (
        "Newton/MuJoCo cannot use Isaac Sim 6's verified MJC actuator conversion for pure-P "
        "drives; the shared drive properties remain inactive metadata and runtime tuning is "
        "required: {joints}"
    ),
    "target_velocity": (
        "Newton/MuJoCo has no Isaac Sim 6 MJC static mapping for drive target velocity; runtime "
        "control is required: {joints}"
    ),
    "friction_loss": (
        "Newton/MuJoCo keeps one mjc:frictionloss threshold per joint (the absolute static hold "
        "effort); the separate dynamic effort stays in customData `usdcraft:newtonFrictionMapping` "
        "as static/dynamic: {joints}"
    ),
    "velocity_limit": (
        "Newton/MuJoCo has no Isaac Sim 6 MJC mapping for joint velocity limits; the values stay "
        "in usdcraft customData: {joints}"
    ),
    "transmission_opacity": (
        "UsdPreviewSurface has no transmission input; material transmission is approximated by "
        "reducing opacity: {joints}"
    ),
    "non_rigid_body": (
        "Non-rigid body models are exported as static rest-pose geometry; runtime deformation is "
        "not represented (part and body model): {joints}"
    ),
    "non_rigid_attachment": (
        "Fixed articulations involving a non-rigid part are preserved as usdcraft attachment "
        "customData; no native physics joint is authored: {joints}"
    ),
    "load_proportional_friction": (
        "Newton/MuJoCo has no load-proportional joint-friction mapping; the coefficient stays in "
        "customData and the joint uses the noted Newton fallback: {joints}"
    ),
    "spherical_static": (
        "Spherical articulations have only static Newton validation; downstream runtime behavior "
        "remains unverified: {joints}"
    ),
}


class _BackendMappingNotes:
    """Collect per-joint backend-mapping facts and render one note per kind."""

    def __init__(self) -> None:
        self._joints: dict[str, list[str]] = {}

    def add(self, kind: str, joint_name: str, detail: str | None = None) -> None:
        if kind not in _BACKEND_NOTE_TEMPLATES:
            raise KeyError(f"Unknown backend note kind {kind!r}")
        entry = f"{joint_name} ({detail})" if detail else str(joint_name)
        self._joints.setdefault(kind, []).append(entry)

    def render(self) -> list[str]:
        return [
            _BACKEND_NOTE_TEMPLATES[kind].format(joints=", ".join(joints))
            for kind, joints in self._joints.items()
        ]


def _author_physx_layer(
    object_model: ArticulatedObject,
    path: Path,
    physics_path: Path,
    joint_paths: dict[str, Sdf.Path],
) -> list[str]:
    stage = _create_backend_layer(path, physics_path, backend="physx")
    for binding in _resolve_native_articulation_root_bindings(stage, path=path):
        _apply_api_schema_token(
            stage.OverridePrim(binding.api_path),
            "PhysxArticulationAPI",
        )

    for joint in object_model.articulations:
        joint_path = joint_paths.get(joint.name)
        if joint_path is None:
            continue
        joint_type = _joint_type(joint)
        angular = joint_type in {
            ArticulationType.REVOLUTE,
            ArticulationType.CONTINUOUS,
        }
        scalar = angular or joint_type == ArticulationType.PRISMATIC
        velocity_limit = (
            joint.motion_limits.velocity if scalar and joint.motion_limits is not None else None
        )
        friction = (
            joint.motion_properties.friction
            if scalar and joint.motion_properties is not None
            else None
        )
        resistance = joint.physx_joint_resistance if scalar else None
        armature = joint.physx_armature if scalar else None
        if velocity_limit is None and friction is None and resistance is None and armature is None:
            continue

        prim = stage.OverridePrim(joint_path)
        axis_name = "angular" if angular else "linear"
        if isinstance(resistance, PhysxJointFriction):
            prim.SetCustomDataByKey(
                "usdcraft:physxJointResistance",
                {
                    "model": "effort",
                    "staticFrictionEffort": resistance.static_friction_effort,
                    "dynamicFrictionEffort": resistance.dynamic_friction_effort,
                    "viscousFrictionCoefficient": resistance.viscous_friction_coefficient,
                },
            )
            _apply_api_schema_token(prim, f"PhysxJointAxisAPI:{axis_name}")
            namespace = f"physxJointAxis:{axis_name}"
            prim.CreateAttribute(
                f"{namespace}:staticFrictionEffort",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(float(resistance.static_friction_effort))
            prim.CreateAttribute(
                f"{namespace}:dynamicFrictionEffort",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(float(resistance.dynamic_friction_effort))
            viscous = float(resistance.viscous_friction_coefficient)
            if angular:
                viscous *= math.pi / 180.0
            prim.CreateAttribute(
                f"{namespace}:viscousFrictionCoefficient",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(viscous)
            if velocity_limit is not None:
                value = math.degrees(velocity_limit) if angular else float(velocity_limit)
                prim.CreateAttribute(
                    f"{namespace}:maxJointVelocity",
                    Sdf.ValueTypeNames.Float,
                    custom=False,
                ).Set(float(value))
            if armature is not None:
                prim.CreateAttribute(
                    f"{namespace}:armature",
                    Sdf.ValueTypeNames.Float,
                    custom=False,
                ).Set(float(armature))
        else:
            if isinstance(resistance, PhysxJointFrictionCoefficient):
                prim.SetCustomDataByKey(
                    "usdcraft:physxJointResistance",
                    {
                        "model": "load_proportional",
                        "coefficient": resistance.coefficient,
                    },
                )
            _apply_api_schema_token(prim, "PhysxJointAPI")
            if velocity_limit is not None:
                value = math.degrees(velocity_limit) if angular else float(velocity_limit)
                prim.CreateAttribute(
                    "physxJoint:maxJointVelocity",
                    Sdf.ValueTypeNames.Float,
                    custom=False,
                ).Set(float(value))
            coefficient = (
                resistance.coefficient
                if isinstance(resistance, PhysxJointFrictionCoefficient)
                else friction
            )
            if coefficient is not None:
                prim.CreateAttribute(
                    "physxJoint:jointFriction",
                    Sdf.ValueTypeNames.Float,
                    custom=False,
                ).Set(float(coefficient))
            if armature is not None:
                prim.CreateAttribute(
                    "physxJoint:armature",
                    Sdf.ValueTypeNames.Float,
                    custom=False,
                ).Set(float(armature))
        if armature is not None:
            prim.SetCustomDataByKey("usdcraft:physxArmature", float(armature))
    stage.GetRootLayer().Save()
    return []


def _author_newton_mjc_layer(
    object_model: ArticulatedObject,
    path: Path,
    physics_path: Path,
    joint_paths: dict[str, Sdf.Path],
    *,
    notes: _BackendMappingNotes | None = None,
) -> list[str]:
    stage = _create_backend_layer(path, physics_path, backend="newton")
    warnings: list[str] = []
    notes = notes if notes is not None else _BackendMappingNotes()
    actuator_scope: UsdGeom.Scope | None = None
    for binding in _resolve_native_articulation_root_bindings(stage, path=path):
        root_prim = stage.OverridePrim(binding.api_path)
        _apply_api_schema_token(root_prim, "NewtonArticulationRootAPI")
        root_prim.CreateAttribute(
            "newton:selfCollisionEnabled",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        ).Set(False)

    for joint in object_model.articulations:
        joint_path = joint_paths.get(joint.name)
        if joint_path is None:
            continue
        prim = stage.OverridePrim(joint_path)
        joint_type = _joint_type(joint)
        angular = joint_type in {
            ArticulationType.REVOLUTE,
            ArticulationType.CONTINUOUS,
        }
        scalar = angular or joint_type == ArticulationType.PRISMATIC

        drive_name = UsdPhysics.Tokens.angular if angular else UsdPhysics.Tokens.linear
        has_shared_drive = scalar and prim.HasAPI(UsdPhysics.DriveAPI, drive_name)
        if has_shared_drive:
            shared_drive = UsdPhysics.DriveAPI(prim, drive_name)
            if joint.drive is not None:
                max_force_attr = shared_drive.GetMaxForceAttr()
                # SDK gains use SI/radian units. The shared UsdPhysics angular
                # DriveAPI uses per-degree coefficients, but MJC actuators use
                # radians, so construct Newton gains from the original SDK values.
                stiffness = float(joint.drive.stiffness)
                damping = float(joint.drive.damping)
                acceleration_mode = joint.drive.mode == "acceleration"

                if actuator_scope is None:
                    actuator_scope = UsdGeom.Scope.Define(stage, "/Asset/Physics")
                actuator_path = actuator_scope.GetPath().AppendChild(f"{joint_path.name}_actuator")
                actuator = stage.DefinePrim(actuator_path, "MjcActuator")
                actuator.CreateRelationship("mjc:target", custom=False).SetTargets([joint_path])

                if max_force_attr.HasAuthoredValueOpinion():
                    max_force = float(max_force_attr.Get())
                    if not math.isfinite(max_force) or max_force < 0.0:
                        raise ValidationError(
                            f"Newton actuator {joint.name!r} has an invalid force range"
                        )
                    _create_uniform_attribute(
                        actuator,
                        "mjc:forceRange:min",
                        Sdf.ValueTypeNames.Double,
                        -max_force,
                    )
                    _create_uniform_attribute(
                        actuator,
                        "mjc:forceRange:max",
                        Sdf.ValueTypeNames.Double,
                        max_force,
                    )

                gain_parameters: list[float] | None = None
                bias_parameters: list[float] | None = None
                if acceleration_mode:
                    notes.add("acceleration_drive", joint.name)
                elif stiffness > 0.0 and damping > 0.0:
                    gain_parameters = [stiffness, *([0.0] * (_MJC_PARAMETER_COUNT - 1))]
                    bias_parameters = [
                        0.0,
                        -stiffness,
                        -damping,
                        *([0.0] * (_MJC_PARAMETER_COUNT - 3)),
                    ]
                elif stiffness == 0.0 and damping > 0.0:
                    gain_parameters = [damping, *([0.0] * (_MJC_PARAMETER_COUNT - 1))]
                    bias_parameters = [
                        0.0,
                        0.0,
                        -damping,
                        *([0.0] * (_MJC_PARAMETER_COUNT - 3)),
                    ]
                elif stiffness > 0.0:
                    notes.add("pure_p_drive", joint.name)

                if gain_parameters is not None and bias_parameters is not None:
                    _create_uniform_attribute(
                        actuator,
                        "mjc:gainPrm",
                        Sdf.ValueTypeNames.DoubleArray,
                        Vt.DoubleArray(gain_parameters),
                    )
                    _create_uniform_attribute(
                        actuator,
                        "mjc:biasPrm",
                        Sdf.ValueTypeNames.DoubleArray,
                        Vt.DoubleArray(bias_parameters),
                    )
                    _create_uniform_attribute(
                        actuator,
                        "mjc:gainType",
                        Sdf.ValueTypeNames.Token,
                        "fixed",
                    )
                    _create_uniform_attribute(
                        actuator,
                        "mjc:biasType",
                        Sdf.ValueTypeNames.Token,
                        "affine",
                    )

                target_position_attr = shared_drive.GetTargetPositionAttr()
                if target_position_attr.HasAuthoredValueOpinion():
                    _apply_api_schema_token(prim, "MjcJointAPI")
                    reference = float(target_position_attr.Get())
                    if angular:
                        reference = math.radians(reference)
                    _create_uniform_attribute(
                        prim,
                        "mjc:ref",
                        Sdf.ValueTypeNames.Double,
                        reference,
                    )
                if shared_drive.GetTargetVelocityAttr().HasAuthoredValueOpinion():
                    notes.add("target_velocity", joint.name)

            # Newton uses MjcActuator only for an explicit active JointDrive.
            # Passive damping and passive springs are native joint properties.
            # Remove only the shared API token so its attributes remain inactive
            # traceability metadata in the Newton composition.
            if not prim.RemoveAPI(UsdPhysics.DriveAPI, drive_name):
                raise ValidationError(
                    f"Could not remove shared DriveAPI from Newton articulation {joint.name!r}"
                )

        if scalar:
            resistance = joint.physx_joint_resistance
            motion = joint.motion_properties
            passive_damping = float(motion.damping or 0.0) if motion is not None else 0.0
            damping_authored = motion is not None and motion.damping is not None
            friction: float | None = None
            friction_mapping: dict[str, object] | None = None
            if isinstance(resistance, PhysxJointFriction):
                friction = float(resistance.static_friction_effort)
                passive_damping += float(resistance.viscous_friction_coefficient)
                damping_authored = damping_authored or (
                    resistance.viscous_friction_coefficient != 0.0
                )
                friction_mapping = {
                    "source": "absolute_effort",
                    "staticFrictionEffort": float(resistance.static_friction_effort),
                    "dynamicFrictionEffort": float(resistance.dynamic_friction_effort),
                    "viscousFrictionCoefficient": float(resistance.viscous_friction_coefficient),
                    "frictionLoss": friction,
                }
                notes.add(
                    "friction_loss",
                    joint.name,
                    f"{resistance.static_friction_effort:g}/{resistance.dynamic_friction_effort:g}",
                )
            else:
                if motion is not None and motion.friction is not None:
                    friction = float(motion.friction)
                    friction_mapping = {
                        "source": "legacy_motion_properties",
                        "frictionLoss": friction,
                    }
                if isinstance(resistance, PhysxJointFrictionCoefficient):
                    notes.add(
                        "load_proportional_friction",
                        joint.name,
                        (
                            f"legacy MotionProperties.friction={friction:g}"
                            if friction is not None
                            else "no Newton dry-friction value"
                        ),
                    )

            tuning = joint.newton_joint_solver_tuning
            passive_spring = joint.passive_spring
            if (
                any(value is not None for value in (friction, tuning, passive_spring))
                or damping_authored
            ):
                _apply_api_schema_token(prim, "MjcJointAPI")
            if friction is not None:
                _create_uniform_attribute(
                    prim,
                    "mjc:frictionloss",
                    Sdf.ValueTypeNames.Double,
                    friction,
                )
            if friction_mapping is not None:
                prim.SetCustomDataByKey(
                    "usdcraft:newtonFrictionMapping",
                    friction_mapping,
                )
            if damping_authored:
                _create_uniform_attribute(
                    prim,
                    "mjc:damping",
                    Sdf.ValueTypeNames.Double,
                    passive_damping,
                )
                prim.SetCustomDataByKey(
                    "usdcraft:newtonPassiveDamping",
                    passive_damping,
                )
            if tuning is not None:
                if tuning.solref_friction is not None:
                    _create_uniform_attribute(
                        prim,
                        "mjc:solreffriction",
                        Sdf.ValueTypeNames.DoubleArray,
                        Vt.DoubleArray(tuning.solref_friction),
                    )
                if tuning.solimp_friction is not None:
                    _create_uniform_attribute(
                        prim,
                        "mjc:solimpfriction",
                        Sdf.ValueTypeNames.DoubleArray,
                        Vt.DoubleArray(tuning.solimp_friction),
                    )
            if passive_spring is not None:
                _create_uniform_attribute(
                    prim,
                    "mjc:stiffness",
                    Sdf.ValueTypeNames.Double,
                    float(passive_spring.stiffness),
                )
                _create_uniform_attribute(
                    prim,
                    "mjc:springref",
                    Sdf.ValueTypeNames.Double,
                    float(passive_spring.reference),
                )

        if scalar and joint.motion_limits is not None:
            notes.add("velocity_limit", joint.name)
        if joint_type == ArticulationType.SPHERICAL:
            notes.add("spherical_static", joint.name)

        if joint.mimic is not None:
            leader_path = joint_paths.get(joint.mimic.joint)
            if leader_path is None:
                raise ValidationError(
                    f"Newton mimic source {joint.mimic.joint!r} has no native USD joint"
                )
            _apply_api_schema_token(prim, "NewtonMimicAPI")
            prim.CreateAttribute(
                "newton:mimicEnabled",
                Sdf.ValueTypeNames.Bool,
                custom=False,
            ).Set(True)
            offset = math.degrees(joint.mimic.offset) if angular else float(joint.mimic.offset)
            prim.CreateAttribute(
                "newton:mimicCoef0",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(float(offset))
            prim.CreateAttribute(
                "newton:mimicCoef1",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(float(joint.mimic.multiplier))
            prim.CreateRelationship("newton:mimicJoint", custom=False).SetTargets([leader_path])
    stage.GetRootLayer().Save()
    return warnings


def _create_uniform_attribute(
    prim: Usd.Prim,
    name: str,
    value_type: Sdf.ValueTypeName,
    value: object,
) -> Usd.Attribute:
    attribute = prim.CreateAttribute(
        name,
        value_type,
        custom=False,
        variability=Sdf.VariabilityUniform,
    )
    if not attribute.Set(value):
        raise ValidationError(f"Could not author USD attribute {attribute.GetPath()}")
    return attribute


def _create_backend_layer(
    path: Path,
    physics_path: Path,
    *,
    backend: Literal["physx", "newton"],
) -> Usd.Stage:
    layer = Sdf.Layer.CreateNew(path.as_posix())
    layer.subLayerPaths = [physics_path.name]
    layer.Save()
    stage = Usd.Stage.Open(layer)
    if stage is None:
        raise ValidationError(f"Could not create {backend} USD layer: {path}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = stage.GetPrimAtPath("/Asset")
    if not root:
        raise ValidationError(f"Shared physics layer has no /Asset root: {physics_path}")
    stage.SetDefaultPrim(root)
    stage.OverridePrim("/Asset").SetCustomDataByKey("usdcraft:physicsBackend", backend)
    return stage


def _author_asset_interface(
    path: Path,
    *,
    physics_path: Path,
    physx_path: Path,
    newton_mjc_path: Path,
) -> None:
    stage = Usd.Stage.CreateNew(path.as_posix())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("usdcraft:assetStructureVersion", "3.0")
    root.GetPrim().SetCustomDataByKey(
        "usdcraft:physicsBackends",
        Vt.StringArray(["physx", "newton"]),
    )
    variants = root.GetPrim().GetVariantSets().AddVariantSet("Physics")
    payloads = (
        ("none", None),
        ("physics", physics_path.name),
        ("physx", physx_path.name),
        ("newton", newton_mjc_path.name),
    )
    for name, payload in payloads:
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            if payload is not None:
                root.GetPrim().GetPayloads().AddPayload(payload, "/Asset")
    variants.SetVariantSelection("physx")
    stage.GetRootLayer().Save()
