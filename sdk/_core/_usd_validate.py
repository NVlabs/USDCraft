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

"""Post-export validation of the authored USD Asset Structure and its backend entries."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, cast

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

from ._usd_common import (
    _MJC_MIN_VALUE,
    _MJC_PARAMETER_COUNT,
    _NEWTON_ENTRY_NAME,
    _PHYSX_ENTRY_NAME,
)
from .errors import ValidationError


@dataclass(slots=True, frozen=True)
class _NativeArticulationRootBinding:
    """Mechanical root binding for one connected native USD joint component."""

    api_path: Sdf.Path
    body_path: Sdf.Path
    joint_paths: tuple[Sdf.Path, ...]
    fixed_to_world: bool


_GENERIC_VALIDATION_CHECKS = (
    "stage_open",
    "default_prim",
    "z_up",
    "meters",
    "kilograms",
    "reusable_asset_no_scene",
    "finite_authored_values",
    "dependencies_resolved",
    "relative_asset_paths",
)
_PHYSICS_GRAPH_CHECKS = (
    "physics_graph_acyclic",
    "joint_parent_child_order",
    "articulation_root_coverage",
)
_ARTICULATION_ROOT_BINDING_CHECKS = ("articulation_root_body_binding",)
_NEWTON_BODY_CHECKS = (
    "mass_inertia_or_collision_derivation",
    "collider_dimensions",
    "collider_scale",
)


def _validate_usd_asset(
    path: Path,
    *,
    backend: Literal["generic", "physx", "newton"] = "generic",
) -> tuple[str, ...]:
    stage = Usd.Stage.Open(path.as_posix())
    if stage is None:
        raise ValidationError(f"USD stage cannot be reopened: {path}")
    composition_errors = list(stage.GetCompositionErrors())
    if composition_errors:
        raise ValidationError(
            f"USD stage has composition errors: {path}: "
            + "; ".join(str(error) for error in composition_errors)
        )
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValidationError(f"USD stage has no default prim: {path}")
    if default_prim.GetPath() != Sdf.Path("/Asset"):
        raise ValidationError(f"USD stage default prim must be /Asset: {path}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValidationError(f"USD stage must use Z-up: {path}")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValidationError(f"USD stage must use meters: {path}")
    if not math.isclose(
        UsdPhysics.GetStageKilogramsPerUnit(stage),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValidationError(f"USD stage must use kilograms: {path}")
    if any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()):
        raise ValidationError(f"Reusable USD asset must not define PhysicsScene: {path}")

    _validate_finite_usd_values(stage, path)
    _validate_dependencies_and_asset_paths(path, stage)
    _validate_vendor_namespace(stage, backend=backend, path=path)
    checks = list(_GENERIC_VALIDATION_CHECKS)
    checks.append("vendor_namespace_isolation")
    _validate_physics_graph(stage, path)
    if backend in {"physx", "newton"}:
        checks.extend(_PHYSICS_GRAPH_CHECKS)
        _validate_articulation_root_body_binding(stage, path, backend=backend)
        checks.extend(_ARTICULATION_ROOT_BINDING_CHECKS)
    if backend == "physx":
        _validate_physx_schema_consistency(stage, path)
        checks.append("physx_schema_consistency")
    elif backend == "newton":
        _validate_newton_schema_consistency(stage, path)
        _validate_newton_bodies(stage, path)
        checks.append("newton_schema_consistency")
        checks.extend(_NEWTON_BODY_CHECKS)
    return tuple(checks)


def _validate_asset_interface_variants(path: Path) -> None:
    root_layer = Sdf.Layer.FindOrOpen(path.as_posix())
    if root_layer is None:
        raise ValidationError(f"USD Asset Structure interface cannot be opened: {path}")
    session_layer = Sdf.Layer.CreateAnonymous("usdcraft_asset_interface_validation")
    stage = Usd.Stage.Open(root_layer, session_layer)
    if stage is None:
        raise ValidationError(f"USD Asset Structure interface cannot be composed: {path}")
    root = stage.GetPrimAtPath("/Asset")
    variants = root.GetVariantSet("Physics")
    expected_variants = {"none", "physics", "physx", "newton"}
    if set(variants.GetVariantNames()) != expected_variants:
        raise ValidationError(
            f"USD Asset Structure interface must define Physics variants "
            f"{sorted(expected_variants)}: {path}"
        )
    if variants.GetVariantSelection() != "physx":
        raise ValidationError(
            f"USD Asset Structure interface must default to the PhysX variant: {path}"
        )

    stage.SetEditTarget(session_layer)
    for name, backend in (
        ("none", None),
        ("physics", "generic"),
        ("physx", "physx"),
        ("newton", "newton"),
    ):
        if not variants.SetVariantSelection(name):
            raise ValidationError(
                f"USD Asset Structure interface could not select Physics={name}: {path}"
            )
        composition_errors = list(stage.GetCompositionErrors())
        if composition_errors:
            raise ValidationError(
                f"USD Asset Structure Physics={name} has composition errors: {path}: "
                + "; ".join(str(error) for error in composition_errors)
            )
        _validate_finite_usd_values(stage, path)
        if backend is None:
            if any(True for _ in root.GetChildren()):
                raise ValidationError(
                    f"USD Asset Structure Physics=none must not load asset content: {path}"
                )
            continue
        _validate_vendor_namespace(
            stage, backend=cast('Literal["generic", "physx", "newton"]', backend), path=path
        )
        _validate_physics_graph(stage, path)
        if backend == "physx":
            _validate_articulation_root_body_binding(stage, path, backend="physx")
            _validate_physx_schema_consistency(stage, path)
        elif backend == "newton":
            _validate_articulation_root_body_binding(stage, path, backend="newton")
            _validate_newton_schema_consistency(stage, path)
            _validate_newton_bodies(stage, path)


def _validate_finite_usd_values(stage: Usd.Stage, path: Path) -> None:
    """Reject NaN/Inf in every composed attribute value and customData payload."""

    for layer in stage.GetLayerStack(includeSessionLayers=False):
        _require_finite_usd_value(
            layer.customLayerData,
            location=f"layer {layer.identifier} customLayerData",
            stage_path=path,
        )

    for prim in stage.Traverse():
        prim_path = prim.GetPath()
        _require_finite_usd_value(
            prim.GetCustomData(),
            location=f"{prim_path} customData",
            stage_path=path,
        )
        for prop in prim.GetProperties():
            _require_finite_usd_value(
                prop.GetCustomData(),
                location=f"{prop.GetPath()} customData",
                stage_path=path,
            )
            if not isinstance(prop, Usd.Attribute):
                continue
            attribute = prop
            if attribute.HasAuthoredValueOpinion():
                _require_finite_usd_value(
                    attribute.Get(),
                    location=str(attribute.GetPath()),
                    stage_path=path,
                )
            for time_code in attribute.GetTimeSamples():
                _require_finite_usd_value(
                    time_code,
                    location=f"{attribute.GetPath()} time sample",
                    stage_path=path,
                )
                _require_finite_usd_value(
                    attribute.Get(time_code),
                    location=f"{attribute.GetPath()} at time {time_code}",
                    stage_path=path,
                )


def _require_finite_usd_value(
    value: object,
    *,
    location: str,
    stage_path: Path,
    _seen: set[int] | None = None,
) -> None:
    if value is None or isinstance(value, (str, bytes, bool, Integral)):
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValidationError(
                f"USD stage contains a non-finite numeric value at {location}: {stage_path}"
            )
        return
    if isinstance(value, Sdf.TimeCode):
        _require_finite_usd_value(
            value.GetValue(),
            location=f"{location}.value",
            stage_path=stage_path,
            _seen=_seen,
        )
        return

    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)

    if isinstance(value, Mapping):
        try:
            for key, item in value.items():
                _require_finite_usd_value(
                    item,
                    location=f"{location}.{key}",
                    stage_path=stage_path,
                    _seen=seen,
                )
        finally:
            seen.remove(identity)
        return

    get_real = getattr(value, "GetReal", None)
    get_imaginary = getattr(value, "GetImaginary", None)
    if callable(get_real) and callable(get_imaginary):
        try:
            _require_finite_usd_value(
                get_real(),
                location=f"{location}.real",
                stage_path=stage_path,
                _seen=seen,
            )
            _require_finite_usd_value(
                get_imaginary(),
                location=f"{location}.imaginary",
                stage_path=stage_path,
                _seen=seen,
            )
        finally:
            seen.remove(identity)
        return

    # Bulk numeric containers (Vt arrays, Gf vectors/matrices, plain float lists) are validated
    # in one vectorized pass; the per-element recursion below would take seconds for a single
    # dense mesh attribute and this validator runs over every attribute of every exported layer.
    try:
        bulk = np.asarray(value)
    except Exception:
        bulk = None
    if bulk is not None and bulk.dtype.kind in "fc":
        seen.remove(identity)
        if not np.isfinite(bulk).all():
            first_bad = np.argwhere(~np.isfinite(bulk))[0]
            suffix = "".join(f"[{int(axis_index)}]" for axis_index in first_bad)
            raise ValidationError(
                f"USD stage contains a non-finite numeric value at {location}{suffix}: {stage_path}"
            )
        return
    if bulk is not None and bulk.dtype.kind in "iub":
        seen.remove(identity)
        return

    try:
        items = iter(cast(Iterable[Any], value))
    except TypeError:
        seen.remove(identity)
        return
    try:
        for index, item in enumerate(items):
            _require_finite_usd_value(
                item,
                location=f"{location}[{index}]",
                stage_path=stage_path,
                _seen=seen,
            )
    finally:
        seen.remove(identity)


def _validate_dependencies_and_asset_paths(path: Path, stage: Usd.Stage) -> None:
    layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(path.as_posix()))
    if unresolved:
        raise ValidationError(
            f"USD stage has unresolved dependencies: {path}: "
            + ", ".join(sorted(str(value) for value in unresolved))
        )

    for layer in layers:
        try:
            dependency_groups = UsdUtils.ExtractExternalReferences(layer.identifier)
        except Exception as exc:
            raise ValidationError(
                f"Could not inspect USD dependencies in {layer.identifier}: {exc}"
            ) from exc
        for dependency_group in dependency_groups:
            for authored_path in dependency_group:
                _validate_relative_asset_path(
                    str(authored_path),
                    owner=f"layer {layer.identifier}",
                    stage_path=path,
                )

    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            try:
                value = attribute.Get()
            except Exception as exc:
                raise ValidationError(
                    f"Could not read USD attribute {attribute.GetPath()} in {path}: {exc}"
                ) from exc
            asset_paths: tuple[Sdf.AssetPath, ...]
            if isinstance(value, Sdf.AssetPath):
                asset_paths = (value,)
            elif isinstance(value, (list, tuple)) and all(
                isinstance(item, Sdf.AssetPath) for item in value
            ):
                asset_paths = tuple(value)
            else:
                continue
            for asset_path in asset_paths:
                _validate_relative_asset_path(
                    asset_path.path,
                    owner=f"attribute {attribute.GetPath()}",
                    stage_path=path,
                )

    if path.name in {_PHYSX_ENTRY_NAME, _NEWTON_ENTRY_NAME}:
        has_composition_arcs = bool(stage.GetRootLayer().subLayerPaths) or any(
            prim.HasAuthoredReferences() or prim.HasAuthoredPayloads()
            for prim in stage.TraverseAll()
        )
        if has_composition_arcs:
            raise ValidationError(f"Flattened USD entry must not retain composition arcs: {path}")


def _validate_relative_asset_path(raw: str, *, owner: str, stage_path: Path) -> None:
    value = str(raw or "").strip()
    if not value:
        return
    package_path = value.split("[", 1)[0]
    if (
        Path(package_path).is_absolute()
        or PureWindowsPath(package_path).is_absolute()
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", package_path)
    ):
        raise ValidationError(
            f"USD {owner} authors an absolute dependency in {stage_path}: {value}"
        )


def _authored_schema_tokens(prim: Usd.Prim) -> set[str]:
    tokens = {str(value) for value in prim.GetAppliedSchemas()}
    for spec in prim.GetPrimStack():
        if not spec.HasInfo("apiSchemas"):
            continue
        value = spec.GetInfo("apiSchemas")
        if isinstance(value, Sdf.TokenListOp):
            for items in (
                value.explicitItems,
                value.prependedItems,
                value.appendedItems,
                value.orderedItems,
            ):
                tokens.update(str(item) for item in items)
        elif isinstance(value, (list, tuple)):
            tokens.update(str(item) for item in value)
    return tokens


def _validate_vendor_namespace(
    stage: Usd.Stage,
    *,
    backend: Literal["generic", "physx", "newton"],
    path: Path,
) -> None:
    for prim in stage.Traverse():
        schema_tokens = _authored_schema_tokens(prim)
        property_names = {str(prop.GetName()) for prop in prim.GetProperties()}
        has_physx = any(token.startswith("Physx") for token in schema_tokens) or any(
            name.startswith("physx") for name in property_names
        )
        has_newton = any(token.startswith("Newton") for token in schema_tokens) or any(
            name.startswith("newton:") for name in property_names
        )
        has_mjc = any(token.startswith("Mjc") for token in schema_tokens) or any(
            name.startswith("mjc:") for name in property_names
        )
        forbidden = (
            has_physx or has_newton or has_mjc
            if backend == "generic"
            else (has_newton or has_mjc if backend == "physx" else has_physx)
        )
        if forbidden:
            raise ValidationError(
                f"USD {backend} stage has a conflicting vendor schema at {prim.GetPath()}: {path}"
            )


def _resolve_native_articulation_root_bindings(
    stage: Usd.Stage,
    *,
    path: Path | None = None,
) -> tuple[_NativeArticulationRootBinding, ...]:
    """Resolve one exact API site for every connected native joint component.

    This deliberately inspects only composed ``UsdPhysics.Joint`` body
    relationships. SDK-only placement edges and static non-rigid attachments
    are absent from that graph and therefore cannot accidentally merge two
    independently simulated components.
    """

    location = str(path) if path is not None else stage.GetRootLayer().identifier
    joints = sorted(
        (prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)),
        key=lambda prim: prim.GetPath().pathString,
    )
    if not joints:
        return ()

    node_paths: dict[str, Sdf.Path] = {}
    incoming: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    internal_joints: list[tuple[Sdf.Path, str, str]] = []
    world_joints: list[tuple[Sdf.Path, str]] = []

    for prim in joints:
        joint = UsdPhysics.Joint(prim)
        body0 = list(joint.GetBody0Rel().GetTargets())
        body1 = list(joint.GetBody1Rel().GetTargets())
        if len(body1) != 1 or len(body0) > 1:
            raise ValidationError(
                f"USD joint has invalid body relationships at {prim.GetPath()}: {location}"
            )

        child_path = body1[0]
        child_prim = stage.GetPrimAtPath(child_path)
        if not child_prim:
            raise ValidationError(
                f"USD joint body1 target does not exist at {prim.GetPath()}: {location}"
            )
        child_key = child_path.pathString
        node_paths[child_key] = child_path
        adjacency.setdefault(child_key, set())

        if not body0:
            if not prim.IsA(UsdPhysics.FixedJoint):
                raise ValidationError(
                    f"USD world attachment must be a fixed joint at {prim.GetPath()}: {location}"
                )
            world_joints.append((prim.GetPath(), child_key))
            continue

        parent_path = body0[0]
        parent_prim = stage.GetPrimAtPath(parent_path)
        if not parent_prim:
            raise ValidationError(
                f"USD joint body0 target does not exist at {prim.GetPath()}: {location}"
            )
        parent_key = parent_path.pathString
        node_paths[parent_key] = parent_path
        adjacency.setdefault(parent_key, set()).add(child_key)
        adjacency[child_key].add(parent_key)
        incoming.add(child_key)
        internal_joints.append((prim.GetPath(), parent_key, child_key))

    bindings: list[_NativeArticulationRootBinding] = []
    remaining = set(node_paths)
    while remaining:
        first = min(remaining)
        component: set[str] = set()
        pending = [first]
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(sorted(adjacency.get(node, ()), reverse=True))
        remaining.difference_update(component)

        root_candidates = sorted(component - incoming)
        if len(root_candidates) != 1:
            candidates = ", ".join(root_candidates) if root_candidates else "none"
            raise ValidationError(
                "USD native articulation component must have exactly one root rigid body; "
                f"found {candidates}: {location}"
            )
        root_key = root_candidates[0]
        root_path = node_paths[root_key]
        root_prim = stage.GetPrimAtPath(root_path)
        rigid_api = UsdPhysics.RigidBodyAPI(root_prim)
        if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI) or (
            rigid_api.GetRigidBodyEnabledAttr().HasAuthoredValueOpinion()
            and rigid_api.GetRigidBodyEnabledAttr().Get() is False
        ):
            raise ValidationError(
                f"USD articulation root body is not an enabled rigid body at {root_path}: "
                f"{location}"
            )

        component_world_joints = sorted(
            (
                (joint_path, body_path)
                for joint_path, body_path in world_joints
                if body_path in component
            ),
            key=lambda item: item[0].pathString,
        )
        if len(component_world_joints) > 1:
            raise ValidationError(
                f"USD native articulation component has more than one world joint: {location}"
            )
        if component_world_joints:
            api_path, world_body_key = component_world_joints[0]
            if world_body_key != root_key:
                raise ValidationError(
                    f"USD world joint {api_path} is not connected to the component root body "
                    f"{root_path}: {location}"
                )
            fixed_to_world = True
        else:
            api_path = root_path
            fixed_to_world = False

        component_joint_paths = sorted(
            (
                joint_path
                for joint_path, parent_key, child_key in internal_joints
                if parent_key in component and child_key in component
            ),
            key=lambda joint_path: joint_path.pathString,
        )
        component_joint_paths.extend(joint_path for joint_path, _ in component_world_joints)
        bindings.append(
            _NativeArticulationRootBinding(
                api_path=api_path,
                body_path=root_path,
                joint_paths=tuple(
                    sorted(component_joint_paths, key=lambda joint_path: joint_path.pathString)
                ),
                fixed_to_world=fixed_to_world,
            )
        )

    return tuple(sorted(bindings, key=lambda binding: binding.api_path.pathString))


def _schema_binding_paths(stage: Usd.Stage, schema_name: str) -> tuple[Sdf.Path, ...]:
    return tuple(
        sorted(
            (
                prim.GetPath()
                for prim in stage.Traverse()
                if schema_name in _authored_schema_tokens(prim)
            ),
            key=lambda prim_path: prim_path.pathString,
        )
    )


def _validate_articulation_root_body_binding(
    stage: Usd.Stage,
    path: Path,
    *,
    backend: Literal["generic", "physx", "newton"],
) -> None:
    bindings = _resolve_native_articulation_root_bindings(stage, path=path)
    expected_paths = tuple(binding.api_path for binding in bindings)
    portable_paths = tuple(
        sorted(
            (
                prim.GetPath()
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ),
            key=lambda prim_path: prim_path.pathString,
        )
    )
    if portable_paths != expected_paths:
        raise ValidationError(
            "USD articulation root body binding is invalid: expected "
            f"{[str(value) for value in expected_paths]}, found "
            f"{[str(value) for value in portable_paths]}: {path}"
        )

    if backend == "generic":
        return

    vendor_schema = "PhysxArticulationAPI" if backend == "physx" else "NewtonArticulationRootAPI"
    vendor_paths = _schema_binding_paths(stage, vendor_schema)
    if vendor_paths != expected_paths:
        raise ValidationError(
            f"USD {backend} articulation root body binding is invalid: expected "
            f"{[str(value) for value in expected_paths]}, found "
            f"{[str(value) for value in vendor_paths]}: {path}"
        )

    if backend == "newton":
        policy_paths = tuple(
            sorted(
                (
                    prim.GetPath()
                    for prim in stage.Traverse()
                    if _has_authored_attribute(prim, "newton:selfCollisionEnabled")
                ),
                key=lambda prim_path: prim_path.pathString,
            )
        )
        if policy_paths != expected_paths or any(
            stage.GetPrimAtPath(root_path).GetAttribute("newton:selfCollisionEnabled").Get()
            is not False
            for root_path in expected_paths
        ):
            raise ValidationError(
                "USD newton articulation root attributes are not colocated with the root API: "
                f"{path}"
            )


def _validate_physics_graph(stage: Usd.Stage, path: Path) -> None:
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    if not joints:
        _validate_articulation_root_body_binding(stage, path, backend="generic")
        return
    _validate_articulation_root_body_binding(stage, path, backend="generic")

    adjacency: dict[str, set[str]] = {}
    for prim in joints:
        joint = UsdPhysics.Joint(prim)
        body0 = list(joint.GetBody0Rel().GetTargets())
        body1 = list(joint.GetBody1Rel().GetTargets())
        parent_name = prim.GetCustomDataByKey("usdcraft:parentPart")
        child_name = prim.GetCustomDataByKey("usdcraft:childPart")
        world_parent = parent_name == "world"
        if len(body1) != 1 or (world_parent and body0) or (not world_parent and len(body0) != 1):
            raise ValidationError(
                f"USD joint has invalid body relationships at {prim.GetPath()}: {path}"
            )
        child_prim = stage.GetPrimAtPath(body1[0])
        if not child_prim:
            raise ValidationError(
                f"USD joint body1 target does not exist at {prim.GetPath()}: {path}"
            )
        if isinstance(child_name, str):
            actual_child = child_prim.GetCustomDataByKey("usdcraft:partName")
            if actual_child != child_name:
                raise ValidationError(
                    f"USD joint body1 is not the declared child at {prim.GetPath()}: {path}"
                )
        if world_parent:
            continue

        parent_prim = stage.GetPrimAtPath(body0[0])
        if not parent_prim:
            raise ValidationError(
                f"USD joint body0 target does not exist at {prim.GetPath()}: {path}"
            )
        if isinstance(parent_name, str):
            actual_parent = parent_prim.GetCustomDataByKey("usdcraft:partName")
            if actual_parent != parent_name:
                raise ValidationError(
                    f"USD joint body0 is not the declared parent at {prim.GetPath()}: {path}"
                )
        adjacency.setdefault(body0[0].pathString, set()).add(body1[0].pathString)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValidationError(f"USD articulation graph contains a cycle: {path}")
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in tuple(adjacency):
        visit(node)


def _validate_physx_schema_consistency(stage: Usd.Stage, path: Path) -> None:
    bindings = _resolve_native_articulation_root_bindings(stage, path=path)
    if _schema_binding_paths(stage, "PhysxArticulationAPI") != tuple(
        binding.api_path for binding in bindings
    ):
        raise ValidationError(f"PhysX articulation schema is missing or misplaced: {path}")

    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        drive_name = (
            UsdPhysics.Tokens.angular
            if prim.IsA(UsdPhysics.RevoluteJoint)
            else UsdPhysics.Tokens.linear
        )
        if _has_drive_metadata(prim, drive_name) and not prim.HasAPI(
            UsdPhysics.DriveAPI, drive_name
        ):
            raise ValidationError(f"PhysX scalar joint lost DriveAPI at {prim.GetPath()}: {path}")
        needs_velocity = prim.GetCustomDataByKey("usdcraft:velocityLimit") is not None
        motion = prim.GetCustomDataByKey("usdcraft:motionProperties")
        resistance = prim.GetCustomDataByKey("usdcraft:physxJointResistance")
        armature = prim.GetCustomDataByKey("usdcraft:physxArmature")
        effort_model = isinstance(resistance, dict) and resistance.get("model") == "effort"
        load_model = isinstance(resistance, dict) and resistance.get("model") == "load_proportional"
        needs_legacy_friction = load_model or (
            resistance is None and isinstance(motion, dict) and motion.get("friction") is not None
        )
        if effort_model:
            axis_name = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
            schema = f"PhysxJointAxisAPI:{axis_name}"
            namespace = f"physxJointAxis:{axis_name}"
            if schema not in _authored_schema_tokens(prim):
                raise ValidationError(
                    f"PhysX per-axis joint schema is missing at {prim.GetPath()}: {path}"
                )
            for suffix in (
                "staticFrictionEffort",
                "dynamicFrictionEffort",
                "viscousFrictionCoefficient",
            ):
                if not _has_authored_attribute(prim, f"{namespace}:{suffix}"):
                    raise ValidationError(
                        f"PhysX effort friction {suffix} is missing at {prim.GetPath()}: {path}"
                    )
            static_effort = float(prim.GetAttribute(f"{namespace}:staticFrictionEffort").Get())
            dynamic_effort = float(prim.GetAttribute(f"{namespace}:dynamicFrictionEffort").Get())
            viscous = float(prim.GetAttribute(f"{namespace}:viscousFrictionCoefficient").Get())
            if (
                not all(
                    math.isfinite(value) and value >= 0.0
                    for value in (
                        static_effort,
                        dynamic_effort,
                        viscous,
                    )
                )
                or dynamic_effort > static_effort
            ):
                raise ValidationError(
                    f"PhysX effort friction is invalid at {prim.GetPath()}: {path}"
                )
            if needs_velocity and not _has_authored_attribute(
                prim, f"{namespace}:maxJointVelocity"
            ):
                raise ValidationError(
                    f"PhysX per-axis velocity limit is missing at {prim.GetPath()}: {path}"
                )
            if armature is not None and not _has_authored_attribute(prim, f"{namespace}:armature"):
                raise ValidationError(
                    f"PhysX per-axis armature is missing at {prim.GetPath()}: {path}"
                )
            continue

        if not needs_velocity and not needs_legacy_friction and armature is None:
            continue
        if "PhysxJointAPI" not in _authored_schema_tokens(prim):
            raise ValidationError(f"PhysX joint schema is missing at {prim.GetPath()}: {path}")
        if needs_velocity and not _has_authored_attribute(prim, "physxJoint:maxJointVelocity"):
            raise ValidationError(
                f"PhysX joint velocity limit is missing at {prim.GetPath()}: {path}"
            )
        if needs_legacy_friction and not _has_authored_attribute(prim, "physxJoint:jointFriction"):
            raise ValidationError(f"PhysX joint friction is missing at {prim.GetPath()}: {path}")
        if armature is not None and not _has_authored_attribute(prim, "physxJoint:armature"):
            raise ValidationError(f"PhysX joint armature is missing at {prim.GetPath()}: {path}")


def _validate_newton_schema_consistency(stage: Usd.Stage, path: Path) -> None:
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    bindings = _resolve_native_articulation_root_bindings(stage, path=path)
    root_paths = tuple(binding.api_path for binding in bindings)
    if _schema_binding_paths(stage, "NewtonArticulationRootAPI") != root_paths:
        raise ValidationError(f"Newton articulation schema is missing or misplaced: {path}")
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not _has_authored_attribute(root, "newton:selfCollisionEnabled"):
            raise ValidationError(f"Newton self-collision policy is missing at {root_path}: {path}")

    for prim in joints:
        scalar = prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
        if scalar:
            drive_name = (
                UsdPhysics.Tokens.angular
                if prim.IsA(UsdPhysics.RevoluteJoint)
                else UsdPhysics.Tokens.linear
            )
            if prim.HasAPI(UsdPhysics.DriveAPI, drive_name):
                raise ValidationError(
                    f"Newton scalar joint retains a shared DriveAPI at {prim.GetPath()}: {path}"
                )
            drive_source = prim.GetCustomDataByKey("usdcraft:jointDrive")
            if drive_source is not None and not isinstance(drive_source, Mapping):
                raise ValidationError(
                    f"Newton/MuJoCo active-drive source is invalid at {prim.GetPath()}: {path}"
                )
            actuator_path = Sdf.Path("/Asset/Physics").AppendChild(
                f"{prim.GetPath().name}_actuator"
            )
            actuator = stage.GetPrimAtPath(actuator_path)
            if drive_source is None:
                if actuator:
                    raise ValidationError(
                        f"Newton/MuJoCo actuator has no active JointDrive at "
                        f"{actuator_path}: {path}"
                    )
            elif not actuator or actuator.GetTypeName() != "MjcActuator":
                raise ValidationError(
                    f"Newton/MuJoCo actuator is missing for {prim.GetPath()}: {path}"
                )
            else:
                targets = list(actuator.GetRelationship("mjc:target").GetTargets())
                if targets != [prim.GetPath()]:
                    raise ValidationError(
                        f"Newton/MuJoCo actuator target is invalid at {actuator_path}: {path}"
                    )
                _validate_mjc_actuator(
                    prim,
                    actuator,
                    drive_name,
                    drive_source,
                    path,
                )
            _validate_newton_passive_joint_properties(prim, drive_source, path)

        mimic = prim.GetCustomDataByKey("usdcraft:mimic")
        if not isinstance(mimic, dict):
            continue
        if "NewtonMimicAPI" not in _authored_schema_tokens(prim):
            raise ValidationError(f"Newton mimic schema is missing at {prim.GetPath()}: {path}")
        if not all(
            _has_authored_attribute(prim, name)
            for name in (
                "newton:mimicEnabled",
                "newton:mimicCoef0",
                "newton:mimicCoef1",
            )
        ):
            raise ValidationError(
                f"Newton mimic coefficients are missing at {prim.GetPath()}: {path}"
            )
        targets = list(prim.GetRelationship("newton:mimicJoint").GetTargets())
        if len(targets) != 1 or not stage.GetPrimAtPath(targets[0]):
            raise ValidationError(
                f"Newton mimic relationship is invalid at {prim.GetPath()}: {path}"
            )


def _validate_mjc_actuator(
    joint: Usd.Prim,
    actuator: Usd.Prim,
    drive_name: str,
    drive_source: Mapping[str, object],
    path: Path,
) -> None:
    prefix = f"drive:{drive_name}:physics:"
    max_force_attr = joint.GetAttribute(f"{prefix}maxForce")
    if max_force_attr.HasAuthoredValueOpinion():
        expected_force = float(max_force_attr.Get())
        if not math.isfinite(expected_force) or expected_force < 0.0:
            raise ValidationError(
                f"Newton/MuJoCo force range is invalid at {actuator.GetPath()}: {path}"
            )
        _validate_uniform_double_attribute(
            actuator,
            "mjc:forceRange:min",
            -expected_force,
            path,
            error="Newton/MuJoCo force range is invalid",
        )
        _validate_uniform_double_attribute(
            actuator,
            "mjc:forceRange:max",
            expected_force,
            path,
            error="Newton/MuJoCo force range is invalid",
        )
    elif any(
        _has_authored_attribute(actuator, name)
        for name in ("mjc:forceRange:min", "mjc:forceRange:max")
    ):
        raise ValidationError(
            f"Newton/MuJoCo force range has no source limit at {actuator.GetPath()}: {path}"
        )

    stiffness = _custom_mapping_float(
        drive_source,
        "stiffness",
        path,
        owner=f"active drive at {joint.GetPath()}",
        minimum=0.0,
    )
    damping = _custom_mapping_float(
        drive_source,
        "damping",
        path,
        owner=f"active drive at {joint.GetPath()}",
        minimum=0.0,
    )
    mode = drive_source.get("mode")
    if mode not in {"force", "acceleration"}:
        raise ValidationError(
            f"Newton/MuJoCo active-drive mode is invalid at {joint.GetPath()}: {path}"
        )
    acceleration_mode = mode == "acceleration"
    shared_type = joint.GetAttribute(f"{prefix}type")
    expected_type = UsdPhysics.Tokens.acceleration if acceleration_mode else UsdPhysics.Tokens.force
    if not shared_type.HasAuthoredValueOpinion() or shared_type.Get() != expected_type:
        raise ValidationError(
            f"Newton/MuJoCo active-drive metadata is invalid at {joint.GetPath()}: {path}"
        )
    angular_scale = math.pi / 180.0 if drive_name == UsdPhysics.Tokens.angular else 1.0
    for suffix, expected in (
        ("stiffness", stiffness * angular_scale),
        ("damping", damping * angular_scale),
    ):
        attribute = joint.GetAttribute(f"{prefix}{suffix}")
        value = attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None
        if (
            value is None
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value),
                expected,
                rel_tol=0.0,
                # Shared DriveAPI gains are schema ``float`` values. Permit the
                # unavoidable float32 round-trip while validating the unit scale.
                abs_tol=1e-6,
            )
        ):
            raise ValidationError(
                f"Newton/MuJoCo active-drive metadata is invalid at {joint.GetPath()}: {path}"
            )

    expected_gain: list[float] | None = None
    expected_bias: list[float] | None = None
    if not acceleration_mode and stiffness > 0.0 and damping > 0.0:
        expected_gain = [stiffness, *([0.0] * (_MJC_PARAMETER_COUNT - 1))]
        expected_bias = [
            0.0,
            -stiffness,
            -damping,
            *([0.0] * (_MJC_PARAMETER_COUNT - 3)),
        ]
    elif not acceleration_mode and stiffness == 0.0 and damping > 0.0:
        expected_gain = [damping, *([0.0] * (_MJC_PARAMETER_COUNT - 1))]
        expected_bias = [
            0.0,
            0.0,
            -damping,
            *([0.0] * (_MJC_PARAMETER_COUNT - 3)),
        ]

    gain = actuator.GetAttribute("mjc:gainPrm")
    bias = actuator.GetAttribute("mjc:biasPrm")
    if expected_gain is None or expected_bias is None:
        gain_type = actuator.GetAttribute("mjc:gainType")
        bias_type = actuator.GetAttribute("mjc:biasType")
        if any(
            attribute.HasAuthoredValueOpinion() for attribute in (gain, bias, gain_type, bias_type)
        ):
            raise ValidationError(
                f"Newton/MuJoCo actuator has unsupported gain parameters at "
                f"{actuator.GetPath()}: {path}"
            )
    else:
        for attribute, expected_values in ((gain, expected_gain), (bias, expected_bias)):
            _validate_uniform_double_array_attribute(
                attribute,
                expected_values,
                path,
                error=f"Newton/MuJoCo actuator gains are invalid at {actuator.GetPath()}",
            )
        for name, expected_token in (("mjc:gainType", "fixed"), ("mjc:biasType", "affine")):
            attribute = actuator.GetAttribute(name)
            if (
                attribute.GetTypeName() != Sdf.ValueTypeNames.Token
                or attribute.GetVariability() != Sdf.VariabilityUniform
                or attribute.Get() != expected_token
            ):
                raise ValidationError(
                    f"Newton/MuJoCo actuator mode is invalid at {actuator.GetPath()}: {path}"
                )

    target_position = joint.GetAttribute(f"{prefix}targetPosition")
    if target_position.HasAuthoredValueOpinion():
        if "MjcJointAPI" not in _authored_schema_tokens(joint):
            raise ValidationError(
                f"Newton/MuJoCo joint schema is missing at {joint.GetPath()}: {path}"
            )
        expected_reference = float(target_position.Get())
        if drive_name == UsdPhysics.Tokens.angular:
            expected_reference = math.radians(expected_reference)
        _validate_uniform_double_attribute(
            joint,
            "mjc:ref",
            expected_reference,
            path,
            error="Newton/MuJoCo joint reference is invalid",
        )
    elif _has_authored_attribute(joint, "mjc:ref"):
        raise ValidationError(
            f"Newton/MuJoCo joint reference has no source target at {joint.GetPath()}: {path}"
        )


def _validate_newton_passive_joint_properties(
    prim: Usd.Prim,
    drive_source: Mapping[str, object] | None,
    path: Path,
) -> None:
    motion = prim.GetCustomDataByKey("usdcraft:motionProperties")
    resistance = prim.GetCustomDataByKey("usdcraft:physxJointResistance")
    tuning = prim.GetCustomDataByKey("usdcraft:newtonJointSolverTuning")
    passive_spring = prim.GetCustomDataByKey("usdcraft:passiveSpring")
    for value, label in (
        (motion, "motion properties"),
        (resistance, "PhysX resistance"),
        (tuning, "Newton solver tuning"),
        (passive_spring, "passive spring"),
    ):
        if value is not None and not isinstance(value, Mapping):
            raise ValidationError(
                f"Newton/MuJoCo {label} source is invalid at {prim.GetPath()}: {path}"
            )

    effort_model = isinstance(resistance, Mapping) and resistance.get("model") == "effort"
    expected_friction: float | None = None
    expected_mapping_source: str | None = None
    expected_dynamic: float | None = None
    expected_viscous = 0.0
    if effort_model:
        expected_friction = _custom_mapping_float(
            resistance,
            "staticFrictionEffort",
            path,
            owner=f"PhysX effort resistance at {prim.GetPath()}",
            minimum=0.0,
        )
        expected_dynamic = _custom_mapping_float(
            resistance,
            "dynamicFrictionEffort",
            path,
            owner=f"PhysX effort resistance at {prim.GetPath()}",
            minimum=0.0,
        )
        expected_viscous = _custom_mapping_float(
            resistance,
            "viscousFrictionCoefficient",
            path,
            owner=f"PhysX effort resistance at {prim.GetPath()}",
            minimum=0.0,
        )
        if expected_dynamic > expected_friction:
            raise ValidationError(
                f"Newton/MuJoCo PhysX resistance source is invalid at {prim.GetPath()}: {path}"
            )
        expected_mapping_source = "absolute_effort"
    elif isinstance(motion, Mapping) and motion.get("friction") is not None:
        expected_friction = _custom_mapping_float(
            motion,
            "friction",
            path,
            owner=f"legacy motion friction at {prim.GetPath()}",
            minimum=0.0,
        )
        expected_mapping_source = "legacy_motion_properties"

    friction_mapping = prim.GetCustomDataByKey("usdcraft:newtonFrictionMapping")
    if expected_friction is None:
        if friction_mapping is not None or _has_authored_attribute(prim, "mjc:frictionloss"):
            raise ValidationError(
                f"Newton/MuJoCo joint friction has no source at {prim.GetPath()}: {path}"
            )
    else:
        if not isinstance(friction_mapping, Mapping):
            raise ValidationError(
                f"Newton/MuJoCo joint friction mapping is missing at {prim.GetPath()}: {path}"
            )
        if friction_mapping.get("source") != expected_mapping_source:
            raise ValidationError(
                f"Newton/MuJoCo joint friction mapping is invalid at {prim.GetPath()}: {path}"
            )
        mapped_friction = _custom_mapping_float(
            friction_mapping,
            "frictionLoss",
            path,
            owner=f"Newton friction mapping at {prim.GetPath()}",
            minimum=0.0,
        )
        if not math.isclose(mapped_friction, expected_friction, rel_tol=0.0, abs_tol=1e-12):
            raise ValidationError(
                f"Newton/MuJoCo joint friction mapping is invalid at {prim.GetPath()}: {path}"
            )
        if effort_model:
            for key, expected in (
                ("staticFrictionEffort", expected_friction),
                ("dynamicFrictionEffort", expected_dynamic),
                ("viscousFrictionCoefficient", expected_viscous),
            ):
                actual = _custom_mapping_float(
                    friction_mapping,
                    key,
                    path,
                    owner=f"Newton friction mapping at {prim.GetPath()}",
                    minimum=0.0,
                )
                if expected is None or not math.isclose(
                    actual,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValidationError(
                        f"Newton/MuJoCo joint friction mapping is invalid at "
                        f"{prim.GetPath()}: {path}"
                    )
        _require_mjc_joint_api(prim, path)
        _validate_uniform_double_attribute(
            prim,
            "mjc:frictionloss",
            expected_friction,
            path,
            error="Newton/MuJoCo joint friction is invalid",
        )

    damping_authored = isinstance(motion, Mapping) and motion.get("damping") is not None
    expected_damping = (
        _custom_mapping_float(
            motion,
            "damping",
            path,
            owner=f"passive damping at {prim.GetPath()}",
            minimum=0.0,
        )
        if damping_authored
        else 0.0
    )
    if effort_model and expected_viscous != 0.0:
        damping_authored = True
        expected_damping += expected_viscous
    damping_evidence = prim.GetCustomDataByKey("usdcraft:newtonPassiveDamping")
    if damping_authored:
        try:
            actual_evidence = float(damping_evidence)
        except (TypeError, ValueError):
            actual_evidence = math.nan
        if not math.isfinite(actual_evidence) or not math.isclose(
            actual_evidence,
            expected_damping,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValidationError(
                f"Newton/MuJoCo passive damping evidence is invalid at {prim.GetPath()}: {path}"
            )
        _require_mjc_joint_api(prim, path)
        _validate_uniform_double_attribute(
            prim,
            "mjc:damping",
            expected_damping,
            path,
            error="Newton/MuJoCo passive damping is invalid",
        )
    elif damping_evidence is not None or _has_authored_attribute(prim, "mjc:damping"):
        raise ValidationError(
            f"Newton/MuJoCo passive damping has no source at {prim.GetPath()}: {path}"
        )

    if tuning is None:
        for name in ("mjc:solreffriction", "mjc:solimpfriction"):
            if _has_authored_attribute(prim, name):
                raise ValidationError(
                    f"Newton/MuJoCo solver tuning has no source at {prim.GetPath()}: {path}"
                )
    else:
        solref = _custom_mapping_float_sequence(
            tuning,
            "solrefFriction",
            2,
            path,
            owner=f"Newton solver tuning at {prim.GetPath()}",
        )
        solimp = _custom_mapping_float_sequence(
            tuning,
            "solimpFriction",
            5,
            path,
            owner=f"Newton solver tuning at {prim.GetPath()}",
        )
        if solref is None and solimp is None:
            raise ValidationError(
                f"Newton/MuJoCo solver tuning source is empty at {prim.GetPath()}: {path}"
            )
        if solref is not None:
            if not (all(value > 0.0 for value in solref) or all(value <= 0.0 for value in solref)):
                raise ValidationError(
                    f"Newton/MuJoCo solver tuning is invalid at {prim.GetPath()}: {path}"
                )
            _require_mjc_joint_api(prim, path)
            _validate_uniform_double_array_attribute(
                prim.GetAttribute("mjc:solreffriction"),
                solref,
                path,
                error=f"Newton/MuJoCo solver tuning is invalid at {prim.GetPath()}",
            )
        elif _has_authored_attribute(prim, "mjc:solreffriction"):
            raise ValidationError(
                f"Newton/MuJoCo solref tuning has no source at {prim.GetPath()}: {path}"
            )
        if solimp is not None:
            d0, dwidth, width, midpoint, power = solimp
            if (
                not 0.0 <= d0 <= 1.0
                or not 0.0 <= dwidth <= 1.0
                or width <= 0.0
                or not 0.0 <= midpoint <= 1.0
                or power < 1.0
            ):
                raise ValidationError(
                    f"Newton/MuJoCo solver tuning is invalid at {prim.GetPath()}: {path}"
                )
            _require_mjc_joint_api(prim, path)
            _validate_uniform_double_array_attribute(
                prim.GetAttribute("mjc:solimpfriction"),
                solimp,
                path,
                error=f"Newton/MuJoCo solver tuning is invalid at {prim.GetPath()}",
            )
        elif _has_authored_attribute(prim, "mjc:solimpfriction"):
            raise ValidationError(
                f"Newton/MuJoCo solimp tuning has no source at {prim.GetPath()}: {path}"
            )

    if passive_spring is None:
        for name in ("mjc:stiffness", "mjc:springref"):
            if _has_authored_attribute(prim, name):
                raise ValidationError(
                    f"Newton/MuJoCo passive spring has no source at {prim.GetPath()}: {path}"
                )
    else:
        if drive_source is not None:
            raise ValidationError(
                f"Newton/MuJoCo passive spring conflicts with an active drive at "
                f"{prim.GetPath()}: {path}"
            )
        stiffness = _custom_mapping_float(
            passive_spring,
            "stiffness",
            path,
            owner=f"passive spring at {prim.GetPath()}",
            minimum=0.0,
        )
        if stiffness <= 0.0:
            raise ValidationError(
                f"Newton/MuJoCo passive spring stiffness is invalid at {prim.GetPath()}: {path}"
            )
        reference = _custom_mapping_float(
            passive_spring,
            "reference",
            path,
            owner=f"passive spring at {prim.GetPath()}",
        )
        _require_mjc_joint_api(prim, path)
        _validate_uniform_double_attribute(
            prim,
            "mjc:stiffness",
            stiffness,
            path,
            error="Newton/MuJoCo passive spring stiffness is invalid",
        )
        _validate_uniform_double_attribute(
            prim,
            "mjc:springref",
            reference,
            path,
            error="Newton/MuJoCo passive spring reference is invalid",
        )


def _custom_mapping_float(
    mapping: Mapping[str, Any],
    key: str,
    path: Path,
    *,
    owner: str,
    minimum: float | None = None,
) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValidationError(f"USD {owner} is invalid: {path}")
    return value


def _custom_mapping_float_sequence(
    mapping: Mapping[str, Any],
    key: str,
    length: int,
    path: Path,
    *,
    owner: str,
) -> tuple[float, ...] | None:
    raw = mapping.get(key)
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        raise ValidationError(f"USD {owner} is invalid: {path}")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        raise ValidationError(f"USD {owner} is invalid: {path}") from None
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValidationError(f"USD {owner} is invalid: {path}")
    return values


def _require_mjc_joint_api(prim: Usd.Prim, path: Path) -> None:
    if "MjcJointAPI" not in _authored_schema_tokens(prim):
        raise ValidationError(f"Newton/MuJoCo joint schema is missing at {prim.GetPath()}: {path}")


def _validate_uniform_double_attribute(
    prim: Usd.Prim,
    name: str,
    expected: float,
    path: Path,
    *,
    error: str,
) -> None:
    attribute = prim.GetAttribute(name)
    value = attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None
    if (
        value is None
        or attribute.GetTypeName() != Sdf.ValueTypeNames.Double
        or attribute.GetVariability() != Sdf.VariabilityUniform
        or not math.isfinite(float(value))
        or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValidationError(f"{error} at {prim.GetPath()}: {path}")


def _validate_uniform_double_array_attribute(
    attribute: Usd.Attribute,
    expected: tuple[float, ...] | list[float],
    path: Path,
    *,
    error: str,
) -> None:
    value = list(attribute.Get() or ()) if attribute else []
    if (
        not attribute
        or not attribute.HasAuthoredValueOpinion()
        or attribute.GetTypeName() != Sdf.ValueTypeNames.DoubleArray
        or attribute.GetVariability() != Sdf.VariabilityUniform
        or len(value) != len(expected)
        or not all(math.isfinite(float(component)) for component in value)
        or not np.allclose(value, expected, rtol=0.0, atol=1e-12)
    ):
        raise ValidationError(f"{error}: {path}")


def _has_authored_attribute(prim: Usd.Prim, name: str) -> bool:
    attribute = prim.GetAttribute(name)
    return bool(attribute and attribute.HasAuthoredValueOpinion())


def _has_drive_metadata(prim: Usd.Prim, drive_name: str) -> bool:
    prefix = f"drive:{drive_name}:physics:"
    return any(
        _has_authored_attribute(prim, f"{prefix}{name}")
        for name in (
            "type",
            "stiffness",
            "damping",
            "maxForce",
            "targetPosition",
            "targetVelocity",
        )
    )


def _validate_newton_bodies(stage: Usd.Stage, path: Path) -> None:
    xform_cache = UsdGeom.XformCache()
    collider_problems: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        if rigid_api.GetRigidBodyEnabledAttr().Get() is False:
            continue
        if rigid_api.GetKinematicEnabledAttr().Get() is True:
            continue

        collision_prims = [
            candidate
            for candidate in Usd.PrimRange(prim)
            if candidate.HasAPI(UsdPhysics.CollisionAPI)
        ]
        mass_value: float | None = None
        diagonal_inertia: tuple[float, float, float] | None = None
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attribute = mass_api.GetMassAttr()
            if mass_attribute.HasAuthoredValueOpinion():
                mass_value = float(mass_attribute.Get())
            inertia_attribute = mass_api.GetDiagonalInertiaAttr()
            if inertia_attribute.HasAuthoredValueOpinion():
                value = inertia_attribute.Get()
                diagonal_inertia = cast(
                    tuple[float, float, float], tuple(float(component) for component in value)
                )
        if mass_value is not None and (
            not math.isfinite(mass_value) or mass_value <= _MJC_MIN_VALUE
        ):
            raise ValidationError(
                f"Newton rigid body mass must exceed {_MJC_MIN_VALUE:g} at {prim.GetPath()}: {path}"
            )
        if diagonal_inertia is not None and any(
            not math.isfinite(value) or value <= _MJC_MIN_VALUE for value in diagonal_inertia
        ):
            raise ValidationError(
                f"Newton rigid body diagonal inertia must exceed {_MJC_MIN_VALUE:g} at "
                f"{prim.GetPath()}: {path}"
            )
        if not collision_prims and (mass_value is None or diagonal_inertia is None):
            raise ValidationError(
                f"Newton rigid body needs nondegenerate collision geometry or explicit "
                f"positive mass and diagonal inertia at "
                f"{prim.GetPath()}: {path}"
            )

        for collider in collision_prims:
            try:
                _validate_collider_dimensions(collider, path)
            except ValidationError as exc:
                collider_problems.append(str(exc))
                continue
            matrix = xform_cache.GetLocalToWorldTransform(collider)
            linear = np.asarray(
                [[float(matrix[row][column]) for column in range(3)] for row in range(3)],
                dtype=float,
            )
            determinant = float(np.linalg.det(linear))
            if not math.isfinite(determinant) or determinant <= _MJC_MIN_VALUE:
                collider_problems.append(
                    f"Newton collider has zero or negative scale at {collider.GetPath()}: {path}"
                )
    if collider_problems:
        raise ValidationError(_format_collider_problems(collider_problems, path))


_COLLIDER_REPAIR_HINT = (
    "Every collider is listed so one edit can fix them all. A flat, zero-thickness or sliver "
    "mesh cannot be a collider: give that part's collision geometry real thickness, or keep the "
    "detailed mesh as a visual (collidable=False) and give the part primitive collision proxies "
    "(Box/Cylinder/Sphere)."
)


def _format_collider_problems(problems: list[str], path: Path) -> str:
    if len(problems) == 1:
        return f"{problems[0]}\n{_COLLIDER_REPAIR_HINT}"
    lines = [f"Newton collider validation failed for {len(problems)} colliders: {path}"]
    lines.extend(f"  - {problem.rsplit(': ', 1)[0]}" for problem in problems)
    lines.append(_COLLIDER_REPAIR_HINT)
    return "\n".join(lines)


def _validate_collider_dimensions(prim: Usd.Prim, path: Path) -> None:
    dimensions: tuple[float, ...] = ()
    if prim.IsA(UsdGeom.Cube):
        dimensions = (float(UsdGeom.Cube(prim).GetSizeAttr().Get()),)
    elif prim.IsA(UsdGeom.Sphere):
        dimensions = (float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()),)
    elif prim.IsA(UsdGeom.Cylinder):
        cylinder = UsdGeom.Cylinder(prim)
        dimensions = (
            float(cylinder.GetRadiusAttr().Get()),
            float(cylinder.GetHeightAttr().Get()),
        )
    elif prim.IsA(UsdGeom.Capsule):
        capsule = UsdGeom.Capsule(prim)
        dimensions = (
            float(capsule.GetRadiusAttr().Get()),
            float(capsule.GetHeightAttr().Get()),
        )
    elif prim.IsA(UsdGeom.Cone):
        cone = UsdGeom.Cone(prim)
        dimensions = (
            float(cone.GetRadiusAttr().Get()),
            float(cone.GetHeightAttr().Get()),
        )
    elif prim.IsA(UsdGeom.Mesh):
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or ()
        if len(points) < 4:
            raise ValidationError(
                f"Newton mesh collider needs at least four vertices at {prim.GetPath()}: {path}"
            )
        values = np.asarray(points, dtype=float)
        if values.shape != (len(points), 3) or not np.all(np.isfinite(values)):
            raise ValidationError(
                f"Newton mesh collider has invalid vertices at {prim.GetPath()}: {path}"
            )
        dimensions = tuple(float(value) for value in np.ptp(values, axis=0))
        centered = values - np.mean(values, axis=0)
        if np.linalg.matrix_rank(centered, tol=_MJC_MIN_VALUE) < 3:
            raise ValidationError(
                f"Newton mesh collider is planar or degenerate at {prim.GetPath()}: {path}"
            )
    if any(not math.isfinite(value) or value <= _MJC_MIN_VALUE for value in dimensions):
        raise ValidationError(
            f"Newton collider has a non-positive dimension at {prim.GetPath()}: {path}"
        )
