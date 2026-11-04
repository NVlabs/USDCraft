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

"""Compiler-owned physics checks for generated assets: movable bodies, armature, gravity hold."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, cast

from sdk._core.assets import resolve_mesh_path

GENERATION_MOVABLE_BODY_PHYSICS_CHECK = "check_generation_movable_body_physics"
GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK = "check_generation_interactive_joint_physics"
GENERATION_GRAVITY_HOLD_COVERAGE_CHECK = "check_generation_gravity_hold_coverage"
GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK = "check_generation_gravity_hold_newton_impedance"
GENERATION_PHYSX_ARMATURE_STABILITY_CHECK = "check_generation_physx_armature_stability"

_GENERATION_GRAVITY_ACCELERATION = 9.81
_GENERATION_GRAVITY_POSE_SAMPLES = 65
_GENERATION_GRAVITY_HOLD_META_KEY = "gravity_hold"
_GENERATION_NEWTON_GRAVITY_HOLD_MIN_D0 = 0.999
_GENERATION_NEWTON_GRAVITY_HOLD_SOLIMP = (0.999, 0.9999, 0.001, 0.5, 2.0)
_GENERATION_PHYSX_ARMATURE_PROXY_MAX_RATIO = 100.0


def _enum_token(value: object) -> str:
    return str(getattr(value, "value", value)).strip().lower()


def _generation_movable_rigid_parts(object_model: object) -> tuple[object, ...]:
    """Find rigid Parts whose world pose can change under native rigid physics.

    A floating rigid root moves as a whole. A non-fixed rigid articulation makes
    its child movable, and rigid members connected to a movable member through
    fixed joints belong to the same moving subassembly. Fixed chains rooted to a
    fixed base are intentionally excluded because they never move in simulation.
    """

    parts = list(getattr(object_model, "parts", ()) or ())
    part_by_name = {
        str(name): part
        for part in parts
        if isinstance((name := getattr(part, "name", None)), str) and name
    }
    rigid_names = {
        name
        for name, part in part_by_name.items()
        if _enum_token(getattr(part, "body_model", "rigid")) == "rigid"
    }
    articulations = list(getattr(object_model, "articulations", ()) or ())
    rigid_articulations = [
        articulation
        for articulation in articulations
        if getattr(articulation, "parent", None) in rigid_names
        and getattr(articulation, "child", None) in rigid_names
    ]
    articulation_part_names = {
        name
        for articulation in rigid_articulations
        for name in (articulation.parent, articulation.child)
    }
    runtime_rigid_names = {
        name
        for name, part in part_by_name.items()
        if name in rigid_names
        and (
            name in articulation_part_names
            or getattr(part, "inertial", None) is not None
            or getattr(part, "rigid_body", None) is not None
        )
    }
    child_names = {
        str(child)
        for articulation in rigid_articulations
        if isinstance((child := getattr(articulation, "child", None)), str) and child
    }

    movable_names: set[str] = set()
    if _enum_token(getattr(object_model, "base_mode", "floating")) == "floating":
        movable_names.update(runtime_rigid_names - child_names)

    fixed_neighbors: dict[str, set[str]] = {name: set() for name in runtime_rigid_names}
    for articulation in rigid_articulations:
        parent = getattr(articulation, "parent", None)
        child = getattr(articulation, "child", None)
        if not isinstance(parent, str) or not isinstance(child, str):
            continue
        if _enum_token(getattr(articulation, "articulation_type", "fixed")) == "fixed":
            fixed_neighbors[parent].add(child)
            fixed_neighbors[child].add(parent)
        else:
            movable_names.add(child)

    pending = list(movable_names)
    while pending:
        name = pending.pop()
        for neighbor in fixed_neighbors.get(name, ()):
            if neighbor in movable_names:
                continue
            movable_names.add(neighbor)
            pending.append(neighbor)

    return tuple(
        part
        for part in parts
        if isinstance((name := getattr(part, "name", None)), str) and name in movable_names
    )


def _positive_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def _nonnegative_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= 0.0


def _finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _format_generation_physics_findings(introduction: str, findings: list[str]) -> str:
    if not findings:
        return ""
    preview_limit = 20
    lines = [introduction, *(f"- {finding}" for finding in findings[:preview_limit])]
    if len(findings) > preview_limit:
        lines.append(f"... ({len(findings) - preview_limit} more)")
    return "\n".join(lines)


def _vec3_dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _vec3_cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _vec3_finite(value: Any) -> tuple[float, float, float] | None:
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        return None
    return vector


def _vec3_normalize(value: object) -> tuple[float, float, float] | None:
    vector = _vec3_finite(value)
    if vector is None:
        return None
    magnitude = math.sqrt(_vec3_dot(vector, vector))
    if magnitude <= 1e-12:
        return None
    return cast("tuple[float, float, float]", tuple(component / magnitude for component in vector))


def _mat4_transform_point(
    matrix: Any,
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    return cast(
        "tuple[float, float, float]",
        tuple(
            sum(float(matrix[row][column]) * point[column] for column in range(3))
            + float(matrix[row][3])
            for row in range(3)
        ),
    )


def _mat4_transform_direction(
    matrix: Any,
    direction: tuple[float, float, float],
) -> tuple[float, float, float]:
    return cast(
        "tuple[float, float, float]",
        tuple(
            sum(float(matrix[row][column]) * direction[column] for column in range(3))
            for row in range(3)
        ),
    )


def _generation_native_articulation_components(
    object_model: object,
) -> tuple[tuple[tuple[object, ...], tuple[object, ...]], ...]:
    """Return source-model components corresponding to native USD joint graphs.

    SDK ``FLOATING`` articulations are placement edges and do not materialize as native USD
    joints. Excluding them here is what keeps independently simulated components independent in
    this authoring-only proxy, matching the native articulation-root resolver's graph boundary.
    """

    parts = list(getattr(object_model, "parts", ()) or ())
    part_by_name = {
        str(name): part
        for part in parts
        if isinstance((name := getattr(part, "name", None)), str)
        and name
        and _enum_token(getattr(part, "body_model", "rigid")) == "rigid"
    }
    native_joints = [
        articulation
        for articulation in list(getattr(object_model, "articulations", ()) or ())
        if _enum_token(getattr(articulation, "articulation_type", "")) != "floating"
        and getattr(articulation, "parent", None) in part_by_name
        and getattr(articulation, "child", None) in part_by_name
    ]

    adjacency: dict[str, set[str]] = {}
    for articulation in native_joints:
        parent = str(getattr(articulation, "parent"))
        child = str(getattr(articulation, "child"))
        adjacency.setdefault(parent, set()).add(child)
        adjacency.setdefault(child, set()).add(parent)

    components: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
    remaining = set(adjacency)
    while remaining:
        pending = [min(remaining)]
        component_names: set[str] = set()
        while pending:
            name = pending.pop()
            if name in component_names:
                continue
            component_names.add(name)
            pending.extend(sorted(adjacency.get(name, ()), reverse=True))
        remaining.difference_update(component_names)
        component_parts = tuple(part_by_name[name] for name in sorted(component_names))
        component_joints = tuple(
            sorted(
                (
                    articulation
                    for articulation in native_joints
                    if getattr(articulation, "parent", None) in component_names
                    and getattr(articulation, "child", None) in component_names
                ),
                key=lambda articulation: str(getattr(articulation, "name", "")),
            )
        )
        components.append((component_parts, component_joints))
    return tuple(components)


def _format_native_component(parts: tuple[object, ...]) -> str:
    names = sorted(str(getattr(part, "name", "")) for part in parts)
    preview = names[:8]
    suffix = "" if len(names) <= len(preview) else f", ... +{len(names) - len(preview)}"
    return f"[{', '.join(repr(name) for name in preview)}{suffix}]"


def _declared_source_inertia_principal_extrema(part: object) -> tuple[float, float] | None:
    inertial = getattr(part, "inertial", None)
    inertia = None if inertial is None else getattr(inertial, "inertia", None)
    if inertia is None:
        return None
    components: tuple[Any, ...] = tuple(
        getattr(inertia, name, None) for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    )
    if not all(_finite(value) for value in components):
        return None
    ixx, ixy, ixz, iyy, iyz, izz = (float(value) for value in components)
    try:
        numpy = importlib.import_module("numpy")
        eigenvalues = numpy.linalg.eigvalsh(
            (
                (ixx, ixy, ixz),
                (ixy, iyy, iyz),
                (ixz, iyz, izz),
            )
        )
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
    except Exception:
        return None
    if not _positive_finite(minimum) or not _positive_finite(maximum):
        return None
    return minimum, maximum


def _declared_source_inertia_axis_projection(
    part: object,
    *,
    part_world_transform: object,
    world_axis: tuple[float, float, float],
    origin_to_mat4: Any,
    mat4_mul: Any,
) -> float | None:
    """Project a declared COM inertia tensor onto one world-space direction.

    The translation in ``Inertial.origin`` is deliberately irrelevant: this is the declared
    source tensor about its own COM, not a parallel-axis or generalized-inertia calculation.
    """

    inertial = getattr(part, "inertial", None)
    inertia = None if inertial is None else getattr(inertial, "inertia", None)
    inertial_origin = None if inertial is None else getattr(inertial, "origin", None)
    if inertia is None or inertial_origin is None:
        return None
    components: tuple[Any, ...] = tuple(
        getattr(inertia, name, None) for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    )
    if not all(_finite(value) for value in components):
        return None
    ixx, ixy, ixz, iyy, iyz, izz = (float(value) for value in components)

    try:
        inertia_world_transform = mat4_mul(
            part_world_transform,
            origin_to_mat4(inertial_origin),
        )
        # world_axis = R * inertial_axis, so inertial_axis = R^T * world_axis.
        inertial_axis: tuple[float, ...] | None = tuple(
            sum(float(inertia_world_transform[row][column]) * world_axis[row] for row in range(3))
            for column in range(3)
        )
    except Exception:
        return None
    inertial_axis = _vec3_normalize(inertial_axis)
    if inertial_axis is None:
        return None
    x, y, z = inertial_axis
    projected = (
        ixx * x * x
        + 2.0 * ixy * x * y
        + 2.0 * ixz * x * z
        + iyy * y * y
        + 2.0 * iyz * y * z
        + izz * z * z
    )
    return projected if _positive_finite(projected) else None


def _check_generation_physx_armature_stability(ctx: Any, object_model: object) -> None:
    """Warn about extreme declared-source inertia ratios within native components.

    This is intentionally a provisional authoring-only risk screen. It neither changes authored
    armature nor participates in the final mechanical Host gate.
    """

    ctx.check(GENERATION_PHYSX_ARMATURE_STABILITY_CHECK, True)
    components = _generation_native_articulation_components(object_model)
    if not components:
        return

    geometry_qc = importlib.import_module("sdk._core.geometry_qc")
    compute_part_world_transforms = getattr(geometry_qc, "compute_part_world_transforms")
    origin_to_mat4 = getattr(geometry_qc, "_origin_to_mat4")
    mat4_mul = getattr(geometry_qc, "_mat4_mul")
    try:
        world_transforms = compute_part_world_transforms(object_model, {})
    except Exception as exc:
        ctx.warn(
            f"{GENERATION_PHYSX_ARMATURE_STABILITY_CHECK}: result=skipped for all native "
            "components because q=0 source transforms could not be resolved "
            f"({type(exc).__name__}). No partial inertia-ratio estimate was used."
        )
        return

    for component_parts, component_joints in components:
        component_label = _format_native_component(component_parts)
        component_by_name = {str(getattr(part, "name", "")): part for part in component_parts}
        for articulation in component_joints:
            joint_type = _enum_token(getattr(articulation, "articulation_type", ""))
            if joint_type not in {"revolute", "continuous", "prismatic"}:
                continue
            joint_name = str(getattr(articulation, "name", ""))
            child_name = str(getattr(articulation, "child", ""))
            child_part = component_by_name.get(child_name)
            if child_part is None:
                ctx.warn(
                    f"{GENERATION_PHYSX_ARMATURE_STABILITY_CHECK}: joint={joint_name!r} "
                    f"component={component_label} result=skipped because its child rigid part "
                    "could not be resolved. No partial inertia-ratio estimate was used."
                )
                continue

            if joint_type == "prismatic":
                proxy_kind = "declared source mass"
                units = "kg"
                values: dict[str, float] = {}
                unavailable: list[str] = []
                for part in component_parts:
                    part_name = str(getattr(part, "name", ""))
                    inertial = getattr(part, "inertial", None)
                    mass: Any = None if inertial is None else getattr(inertial, "mass", None)
                    if not _positive_finite(mass):
                        unavailable.append(part_name)
                    else:
                        values[part_name] = float(mass)
            else:
                proxy_kind = "declared source Inertial COM-axis projection"
                units = "kg*m^2"
                parent_name = str(getattr(articulation, "parent", ""))
                parent_transform = world_transforms.get(parent_name)
                local_axis = _vec3_normalize(getattr(articulation, "axis", None))
                if parent_transform is None or local_axis is None:
                    ctx.warn(
                        f"{GENERATION_PHYSX_ARMATURE_STABILITY_CHECK}: joint={joint_name!r} "
                        f"component={component_label} result=skipped because its q=0 joint axis "
                        "could not be resolved. No partial inertia-ratio estimate was used."
                    )
                    continue
                try:
                    joint_transform = mat4_mul(
                        parent_transform,
                        origin_to_mat4(getattr(articulation, "origin", None)),
                    )
                    world_axis = _vec3_normalize(
                        _mat4_transform_direction(joint_transform, local_axis)
                    )
                except Exception:
                    world_axis = None
                if world_axis is None:
                    ctx.warn(
                        f"{GENERATION_PHYSX_ARMATURE_STABILITY_CHECK}: joint={joint_name!r} "
                        f"component={component_label} result=skipped because its q=0 joint axis "
                        "could not be resolved. No partial inertia-ratio estimate was used."
                    )
                    continue

                values = {}
                unavailable = []
                for part in component_parts:
                    part_name = str(getattr(part, "name", ""))
                    part_transform = world_transforms.get(part_name)
                    projection = (
                        None
                        if part_transform is None
                        else _declared_source_inertia_axis_projection(
                            part,
                            part_world_transform=part_transform,
                            world_axis=world_axis,
                            origin_to_mat4=origin_to_mat4,
                            mat4_mul=mat4_mul,
                        )
                    )
                    if projection is None:
                        unavailable.append(part_name)
                    else:
                        values[part_name] = projection

            if unavailable:
                unavailable_preview = ", ".join(repr(name) for name in unavailable[:8])
                if len(unavailable) > 8:
                    unavailable_preview += f", ... +{len(unavailable) - 8}"
                ctx.warn(
                    f"{GENERATION_PHYSX_ARMATURE_STABILITY_CHECK}: joint={joint_name!r} "
                    f"type={joint_type!r} component={component_label} result=skipped because "
                    f"{proxy_kind} is unavailable or non-positive for part(s) "
                    f"{unavailable_preview}. No partial inertia-ratio estimate was used."
                )
                continue

            child_proxy = values.get(child_name)
            existing_armature = getattr(articulation, "physx_armature", None)
            existing_armature = 0.0 if existing_armature is None else existing_armature
            if child_proxy is None or not _nonnegative_finite(existing_armature):
                ctx.warn(
                    f"{GENERATION_PHYSX_ARMATURE_STABILITY_CHECK}: joint={joint_name!r} "
                    f"type={joint_type!r} component={component_label} result=skipped because "
                    "the child proxy or existing physx_armature is unavailable or invalid."
                )
                continue

            maximum_proxy = max(values.values())
            existing_armature = float(existing_armature)
            effective_child_proxy = child_proxy + existing_armature
            if not _positive_finite(effective_child_proxy):
                continue
            ratio = maximum_proxy / effective_child_proxy
            if ratio <= _GENERATION_PHYSX_ARMATURE_PROXY_MAX_RATIO + 1e-12:
                continue

            target_total_armature = max(
                0.0,
                maximum_proxy / _GENERATION_PHYSX_ARMATURE_PROXY_MAX_RATIO - child_proxy,
            )
            additional_armature = max(0.0, target_total_armature - existing_armature)
            principal_context = ""
            if joint_type != "prismatic":
                principal_extrema = {
                    str(getattr(part, "name", "")): extrema
                    for part in component_parts
                    if (extrema := _declared_source_inertia_principal_extrema(part)) is not None
                }
                child_principal = principal_extrema.get(child_name)
                if child_principal is not None and len(principal_extrema) == len(component_parts):
                    component_principal_maximum = max(
                        maximum for _minimum, maximum in principal_extrema.values()
                    )
                    principal_ratio = component_principal_maximum / child_principal[0]
                    principal_context = (
                        f" For rotation-invariant context only, child source tensor minimum "
                        f"principal inertia={child_principal[0]:.6g} {units} and component "
                        f"max-principal/child-min-principal conditioning ratio="
                        f"{principal_ratio:.6g}:1; that principal ratio does not trigger this "
                        "finding, and armature is not claimed to repair orthogonal principal "
                        "directions."
                    )

            resistance = getattr(articulation, "physx_joint_resistance", None)
            if all(
                hasattr(resistance, field_name)
                for field_name in (
                    "static_friction_effort",
                    "dynamic_friction_effort",
                    "viscous_friction_coefficient",
                )
            ):
                export_context = (
                    " With PhysxJointFriction effort resistance, public physx_armature exports "
                    "to the matching PhysxJointAxisAPI axis instance."
                )
                static_effort = getattr(resistance, "static_friction_effort", None)
                effort_hold = _positive_finite(static_effort)
            elif hasattr(resistance, "coefficient"):
                export_context = (
                    " With load-proportional PhysxJointFrictionCoefficient, public "
                    "physx_armature remains a legal joint-level physxJoint:armature; that "
                    "path does not apply the overlapping PhysxJointAxisAPI."
                )
                effort_hold = False
            else:
                export_context = (
                    " Use the public physx_armature field; its exact PhysX schema site follows "
                    "the selected resistance path."
                )
                effort_hold = False

            drive = getattr(articulation, "drive", None)
            meta = getattr(articulation, "meta", None)
            gravity_hold = (
                meta.get(_GENERATION_GRAVITY_HOLD_META_KEY) if isinstance(meta, dict) else None
            )
            if drive is not None:
                dynamics_context = (
                    " This is an active joint: prefer a physically grounded reflected rotor "
                    "inertia (J*G^2) when one exists, then validate the driven response."
                )
            elif gravity_hold is True and effort_hold:
                dynamics_context = (
                    " This passive gravity_hold joint with positive effort friction is the "
                    "authoring case most closely matched by the observed PhysX hold-instability "
                    "evidence."
                )
            elif gravity_hold is False:
                dynamics_context = (
                    " This passive joint explicitly allows gravity-driven motion, so treat the "
                    "finding only as a general articulation-stability risk and preserve that "
                    "intended motion."
                )
            else:
                dynamics_context = (
                    " This passive joint has no explicit gravity_hold=True effort-friction "
                    "evidence, so treat the finding only as a general articulation-stability "
                    "risk."
                )
            ctx.warn(
                f"{GENERATION_PHYSX_ARMATURE_STABILITY_CHECK}: joint={joint_name!r} "
                f"type={joint_type!r} child={child_name!r} component={component_label}: "
                f"provisional {proxy_kind} adjusted source-proxy screening ratio="
                f"{ratio:.6g}:1 exceeds "
                f"{_GENERATION_PHYSX_ARMATURE_PROXY_MAX_RATIO:g}:1 "
                f"(component maximum={maximum_proxy:.6g} {units}; child={child_proxy:.6g} "
                f"{units}; existing physx_armature={existing_armature:.6g} {units}; adjusted "
                "source-proxy screening denominator="
                f"{effective_child_proxy:.6g} {units}). A provisional 100:1 source "
                f"proxy target would require total physx_armature>={target_total_armature:.6g} "
                f"{units} (additional>={additional_armature:.6g} {units}). This authoring-only "
                "PhysX risk screen is empirical, not an NVIDIA threshold or runtime guarantee; "
                "review intended dynamics and runtime-test each affected asset. The proxy "
                "uses each link's declared source Inertial tensor about its COM at q=0 and "
                "excludes parallel-axis offsets, collision-derived runtime inertia, subtree or "
                "generalized inertia, and contact. It never authors armature automatically."
                f"{principal_context}{export_context}{dynamics_context}"
            )


def _primitive_volume(geometry: object) -> float | None:
    size = getattr(geometry, "size", None)
    if size is not None:
        try:
            dimensions = tuple(float(value) for value in size)
        except (TypeError, ValueError):
            return None
        if len(dimensions) != 3 or not all(
            math.isfinite(value) and value > 0.0 for value in dimensions
        ):
            return None
        return dimensions[0] * dimensions[1] * dimensions[2]

    radius = getattr(geometry, "radius", None)
    length = getattr(geometry, "length", None)
    if radius is None:
        return None
    try:
        resolved_radius = float(radius)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved_radius) or resolved_radius <= 0.0:
        return None
    if length is None:
        return (4.0 / 3.0) * math.pi * resolved_radius**3
    try:
        resolved_length = float(length)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved_length) or resolved_length <= 0.0:
        return None
    return math.pi * resolved_radius**2 * resolved_length


# Derived mesh volume/centroid measurements keyed by resolved path + mtime + scale. The gravity
# diagnostic asks for the same mesh once per part visual, and re-parsing plus re-hulling a large
# OBJ each time is pure repetition; the mtime key keeps freshly rewritten assets current.
_MESH_MEASURE_CACHE: dict[
    tuple[str, int, tuple[float, float, float] | None],
    tuple[float, tuple[float, float, float]] | None,
] = {}


def _mesh_volume_centroid(
    geometry: object,
    *,
    asset_root: Path | None,
) -> tuple[float, tuple[float, float, float]] | None:
    filename = getattr(geometry, "filename", None)
    if filename is None or asset_root is None:
        return None
    try:
        mesh_path = resolve_mesh_path(filename, assets=asset_root, ensure_dir=False)
        cache_key = (
            str(mesh_path),
            mesh_path.stat().st_mtime_ns,
            _vec3_finite(getattr(geometry, "scale", None)),
        )
    except Exception:
        cache_key = None
    if cache_key is not None and cache_key in _MESH_MEASURE_CACHE:
        return _MESH_MEASURE_CACHE[cache_key]
    result = _mesh_volume_centroid_uncached(geometry, asset_root=asset_root)
    if cache_key is not None:
        if len(_MESH_MEASURE_CACHE) > 256:
            _MESH_MEASURE_CACHE.clear()
        _MESH_MEASURE_CACHE[cache_key] = result
    return result


def _mesh_volume_centroid_uncached(
    geometry: object,
    *,
    asset_root: Path | None,
) -> tuple[float, tuple[float, float, float]] | None:
    filename = getattr(geometry, "filename", None)
    if filename is None or asset_root is None:
        return None
    try:
        import trimesh

        mesh_path = resolve_mesh_path(filename, assets=asset_root, ensure_dir=False)
        loaded = trimesh.load_mesh(mesh_path, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh) or loaded.vertices.size == 0:
            return None
        scale = _vec3_finite(getattr(geometry, "scale", None))
        if scale is not None:
            loaded = loaded.copy()
            loaded.apply_scale(scale)
        volume = abs(float(loaded.volume))
        centroid = _vec3_finite(loaded.center_mass)
        if not math.isfinite(volume) or volume <= 1e-15 or centroid is None:
            hull = loaded.convex_hull
            volume = abs(float(hull.volume))
            centroid = _vec3_finite(hull.center_mass)
        if not math.isfinite(volume) or volume <= 1e-15 or centroid is None:
            return None
        return volume, centroid
    except Exception:
        # This is an optional semantic estimate. Mesh readiness and concrete path errors are
        # reported by their dedicated authoring checks; fall back to the declared inertia proxy.
        return None


def _part_gravity_com_local(
    part: object,
    *,
    asset_root: Path | None,
) -> tuple[tuple[float, float, float], str] | None:
    """Return the coarse COM used by the authoring-only gravity diagnostic.

    USD runtimes derive mass properties from the final colliders. Generation source normally
    derives those colliders from collidable visuals, so a primitive-volume centroid is a more
    faithful estimate than an AABB center or the full visual extent. If any contributing shape is
    not a supported primitive, fall back to the declared inertial proxy rather than silently ignore
    part of the authored mass.
    """

    collisions = list(getattr(part, "collisions", ()) or ())
    if collisions:
        elements = collisions
    else:
        elements = [
            visual
            for visual in list(getattr(part, "visuals", ()) or ())
            if bool(getattr(visual, "collidable", True))
        ]

    weighted: list[tuple[float, tuple[float, float, float]]] = []
    used_mesh = False
    unsupported = False
    for element in elements:
        geometry = getattr(element, "geometry", None)
        volume = _primitive_volume(geometry)
        origin = getattr(element, "origin", None)
        if volume is not None:
            position = None if origin is None else getattr(origin, "xyz", None)
            coordinates = _vec3_finite(position)
        else:
            mesh_result = _mesh_volume_centroid(geometry, asset_root=asset_root)
            if mesh_result is None or origin is None:
                coordinates = None
            else:
                volume, mesh_centroid = mesh_result
                geometry_qc = importlib.import_module("sdk._core.geometry_qc")
                coordinates = _mat4_transform_point(
                    getattr(geometry_qc, "_origin_to_mat4")(origin), mesh_centroid
                )
                used_mesh = True
        if volume is None or coordinates is None:
            unsupported = True
            break
        weighted.append((volume, coordinates))

    if weighted and not unsupported:
        total_volume = sum(volume for volume, _ in weighted)
        if math.isfinite(total_volume) and total_volume > 0.0:
            return (
                cast(
                    "tuple[float, float, float]",
                    tuple(
                        sum(volume * position[index] for volume, position in weighted)
                        / total_volume
                        for index in range(3)
                    ),
                ),
                "primitive/mesh-collider centroid" if used_mesh else "primitive-collider centroid",
            )

    inertial = getattr(part, "inertial", None)
    origin = None if inertial is None else getattr(inertial, "origin", None)
    position = None if origin is None else getattr(origin, "xyz", None)
    coordinates = _vec3_finite(position)
    if coordinates is None:
        return None
    return coordinates, "declared inertial COM proxy"


def _gravity_downstream_parts(object_model: object, root_name: str) -> tuple[object, ...]:
    parts = list(getattr(object_model, "parts", ()) or ())
    part_by_name = {
        str(name): part
        for part in parts
        if isinstance((name := getattr(part, "name", None)), str) and name
    }
    children: dict[str, list[str]] = {}
    for articulation in list(getattr(object_model, "articulations", ()) or ()):
        if _enum_token(getattr(articulation, "articulation_type", "")) == "floating":
            # FLOATING placement edges are independent native bodies, not load-transmitting
            # descendants of the reduced-coordinate articulation above them.
            continue
        parent = getattr(articulation, "parent", None)
        child = getattr(articulation, "child", None)
        if isinstance(parent, str) and isinstance(child, str):
            children.setdefault(parent, []).append(child)

    selected: list[object] = []
    pending = [root_name]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        part = part_by_name.get(name)
        if part is not None:
            selected.append(part)
        pending.extend(children.get(name, ()))
    return tuple(selected)


def _gravity_joint_pose_samples(articulation: object, joint_type: str) -> tuple[float, ...]:
    if joint_type == "continuous":
        lower = -math.pi
        upper = math.pi
    else:
        motion_limits = getattr(articulation, "motion_limits", None)
        raw_lower = None if motion_limits is None else getattr(motion_limits, "lower", None)
        raw_upper = None if motion_limits is None else getattr(motion_limits, "upper", None)
        if raw_lower is None or raw_upper is None:
            return (0.0,)
        try:
            lower = float(raw_lower)
            upper = float(raw_upper)
        except (TypeError, ValueError):
            return (0.0,)
        if not all(math.isfinite(value) for value in (lower, upper)):
            return (0.0,)

    if upper <= lower:
        return (lower,)
    samples = [
        lower + (upper - lower) * index / (_GENERATION_GRAVITY_POSE_SAMPLES - 1)
        for index in range(_GENERATION_GRAVITY_POSE_SAMPLES)
    ]
    if lower <= 0.0 <= upper:
        samples.append(0.0)
    return tuple(sorted(set(samples)))


def _static_drive_effort(articulation: object, position: float) -> float:
    drive = getattr(articulation, "drive", None)
    if drive is None or str(getattr(drive, "mode", "force")).strip().lower() != "force":
        return 0.0
    stiffness = getattr(drive, "stiffness", 0.0)
    if not _nonnegative_finite(stiffness) or float(stiffness) == 0.0:
        return 0.0
    target = getattr(drive, "target_position", None)
    target_position = 0.0 if target is None else float(target)
    effort = float(stiffness) * (target_position - position)

    max_force: Any = getattr(drive, "max_force", None)
    if max_force is None:
        motion_limits = getattr(articulation, "motion_limits", None)
        max_force = None if motion_limits is None else getattr(motion_limits, "effort", None)
    if _nonnegative_finite(max_force):
        limit = float(max_force)
        effort = max(-limit, min(limit, effort))
    return effort


def _static_passive_spring_effort(articulation: object, position: float) -> float:
    spring = getattr(articulation, "passive_spring", None)
    if spring is None:
        return 0.0
    stiffness: Any = getattr(spring, "stiffness", None)
    reference: Any = getattr(spring, "reference", None)
    if not _positive_finite(stiffness) or not _finite(reference):
        return 0.0
    return float(stiffness) * (float(reference) - position)


def _gravity_hold_estimate(
    object_model: object,
    articulation: object,
    joint_type: str,
    *,
    asset_root: Path | None = None,
) -> tuple[float, float, float, tuple[str, ...]] | None:
    geometry_qc = importlib.import_module("sdk._core.geometry_qc")
    compute_part_world_transforms = getattr(geometry_qc, "compute_part_world_transforms")
    origin_to_mat4 = getattr(geometry_qc, "_origin_to_mat4")
    mat4_mul = getattr(geometry_qc, "_mat4_mul")

    joint_name = str(getattr(articulation, "name", ""))
    parent_name = getattr(articulation, "parent", None)
    child_name = getattr(articulation, "child", None)
    if not isinstance(parent_name, str) or not isinstance(child_name, str) or not joint_name:
        return None
    subtree_parts = _gravity_downstream_parts(object_model, child_name)
    if not subtree_parts:
        return None

    part_inputs: list[tuple[object, float, tuple[float, float, float], str]] = []
    for part in subtree_parts:
        if _enum_token(getattr(part, "body_model", "rigid")) != "rigid":
            continue
        inertial = getattr(part, "inertial", None)
        mass: Any = None if inertial is None else getattr(inertial, "mass", None)
        com_result = _part_gravity_com_local(part, asset_root=asset_root)
        if not _positive_finite(mass) or com_result is None:
            # A partial sum would silently understate the required holding
            # effort. Treat any unmeasurable rigid descendant as unknown.
            return None
        com, source = com_result
        part_inputs.append((part, float(mass), com, source))
    if not part_inputs:
        return None

    maximum_residual = -1.0
    maximum_gravity = 0.0
    maximum_position = 0.0
    sources = tuple(sorted({source for _, _, _, source in part_inputs}))
    gravity_vector_unit = (0.0, 0.0, -1.0)
    for position in _gravity_joint_pose_samples(articulation, joint_type):
        world = compute_part_world_transforms(object_model, {joint_name: position})
        parent_transform = world.get(parent_name)
        if parent_transform is None:
            return None
        joint_transform = mat4_mul(
            parent_transform,
            origin_to_mat4(getattr(articulation, "origin", None)),
        )
        joint_point = _mat4_transform_point(joint_transform, (0.0, 0.0, 0.0))
        local_axis = _vec3_normalize(getattr(articulation, "axis", None))
        if local_axis is None:
            return None
        world_axis = _vec3_normalize(_mat4_transform_direction(joint_transform, local_axis))
        if world_axis is None:
            return None

        gravity_effort = 0.0
        for part, mass, local_com, _ in part_inputs:
            part_name = getattr(part, "name", None)
            part_transform = world.get(part_name)
            if part_transform is None:
                return None
            world_com = _mat4_transform_point(part_transform, local_com)
            gravity_force = cast(
                "tuple[float, float, float]",
                tuple(
                    component * mass * _GENERATION_GRAVITY_ACCELERATION
                    for component in gravity_vector_unit
                ),
            )
            if joint_type == "prismatic":
                gravity_effort += _vec3_dot(world_axis, gravity_force)
            else:
                lever = cast(
                    "tuple[float, float, float]",
                    tuple(world_com[index] - joint_point[index] for index in range(3)),
                )
                gravity_effort += _vec3_dot(world_axis, _vec3_cross(lever, gravity_force))

        support_effort = _static_passive_spring_effort(
            articulation, position
        ) + _static_drive_effort(articulation, position)
        residual = abs(gravity_effort + support_effort)
        if residual > maximum_residual:
            maximum_residual = residual
            maximum_gravity = abs(gravity_effort)
            maximum_position = position

    if maximum_residual < 0.0:
        return None
    return maximum_residual, maximum_gravity, maximum_position, sources


def _check_generation_gravity_hold_coverage(
    ctx: Any,
    object_model: object,
    *,
    asset_root: Path | None = None,
) -> None:
    coverage_failures: list[str] = []
    coverage_warnings: list[str] = []
    newton_impedance_failures: list[str] = []
    newton_impedance_warnings: list[str] = []
    for articulation in list(getattr(object_model, "articulations", ()) or ()):
        joint_type = _enum_token(getattr(articulation, "articulation_type", ""))
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            continue
        meta = getattr(articulation, "meta", None)
        intent = meta.get(_GENERATION_GRAVITY_HOLD_META_KEY) if isinstance(meta, dict) else None
        if intent is False:
            continue
        if intent is not None and intent is not True:
            coverage_failures.append(
                f"joint={getattr(articulation, 'name', '')!r}: "
                f"meta[{_GENERATION_GRAVITY_HOLD_META_KEY!r}] must be boolean"
            )
            continue

        estimate = _gravity_hold_estimate(
            object_model,
            articulation,
            joint_type,
            asset_root=asset_root,
        )
        if estimate is None:
            finding = (
                f"joint={getattr(articulation, 'name', '')!r}: could not estimate the "
                "gravity load from positive-mass downstream-body COM data"
            )
            if intent is True:
                coverage_failures.append(finding)
            else:
                coverage_warnings.append(finding)
            continue
        maximum_residual, maximum_gravity, position, sources = estimate
        if maximum_residual <= 1e-9:
            continue

        resistance = getattr(articulation, "physx_joint_resistance", None)
        static_effort: Any = getattr(resistance, "static_friction_effort", None)
        if not _nonnegative_finite(static_effort):
            finding = (
                f"joint={getattr(articulation, 'name', '')!r} type={joint_type!r}: "
                "estimated maximum residual gravity load="
                f"{maximum_residual:.6g} at q={position:.6g} "
                f"(gravity before authored restoring effort={maximum_gravity:.6g}; "
                f"COM source={'+'.join(sources)}), but the joint has no effort-based "
                "PhysxJointFriction.static_friction_effort, so absolute hold coverage is not "
                "measurable. Use gravity_hold=True plus an effort-based resistance when it must "
                "stay where placed, or set gravity_hold=False for deliberate gravity-driven "
                "motion or modeled contact support."
            )
            if intent is True:
                coverage_failures.append(finding)
            else:
                coverage_warnings.append(finding)
            continue
        static_effort = float(static_effort)

        # MuJoCo models dry friction as a soft constraint. frictionloss sets only the absolute
        # effort threshold; a positive sustained gravity residual also needs deliberately strong
        # constraint impedance to preserve Generation's explicit "stay where placed" intent.
        # Keep this as authoring feedback and never synthesize solver tuning from the PhysX effort.
        if static_effort > 1e-9:
            tuning = getattr(articulation, "newton_joint_solver_tuning", None)
            solimp = getattr(tuning, "solimp_friction", None)
            d0 = None
            if isinstance(solimp, (tuple, list)) and solimp:
                candidate = solimp[0]
                if _nonnegative_finite(candidate):
                    d0 = float(candidate)
            if d0 is None or d0 + 1e-12 < _GENERATION_NEWTON_GRAVITY_HOLD_MIN_D0:
                authored = "absent" if d0 is None else f"{d0:.6g}"
                finding = (
                    f"joint={getattr(articulation, 'name', '')!r} type={joint_type!r}: "
                    f"estimated maximum residual gravity load={maximum_residual:.6g} at "
                    f"q={position:.6g} is assigned to positive dry-friction effort, but Newton "
                    f"friction constraint impedance d0 is {authored}. mjc:frictionloss sets the "
                    "absolute effort threshold but its default soft impedance can creep under a "
                    "sustained load. Explicitly author "
                    "NewtonJointSolverTuning(solimp_friction="
                    f"{_GENERATION_NEWTON_GRAVITY_HOLD_SOLIMP!r}) so d0 is at least "
                    f"{_GENERATION_NEWTON_GRAVITY_HOLD_MIN_D0:g}. This solver intent is "
                    "independent of friction magnitude and must not be inferred from stay_put."
                )
                if intent is True:
                    newton_impedance_failures.append(finding)
                else:
                    newton_impedance_warnings.append(finding)

        if static_effort + 1e-9 >= maximum_residual:
            continue

        coverage = static_effort / maximum_residual
        finding = (
            f"joint={getattr(articulation, 'name', '')!r} type={joint_type!r}: "
            f"static_friction_effort={static_effort:.6g} covers only {coverage:.3f}x of the "
            f"estimated maximum residual gravity load={maximum_residual:.6g} at "
            f"q={position:.6g} (gravity before authored restoring effort={maximum_gravity:.6g}; "
            f"COM source={'+'.join(sources)}). The estimate includes every mechanically connected "
            "downstream body at its q=0/default pose, joint-axis projection, joint-origin rotation, "
            "limits, an explicit passive spring, and a force-saturated force-mode drive; it "
            "excludes independent FLOATING placement branches, unmodeled payload, and contact "
            "support. "
            "If this joint must stay where placed, size max_passive_effort from the maximum "
            "residual load and add a focused authored test. If gravity-driven motion is intended, "
            f"set articulation.meta[{_GENERATION_GRAVITY_HOLD_META_KEY!r}]=False."
        )
        if intent is True:
            coverage_failures.append(finding)
        else:
            coverage_warnings.append(finding)

    ctx.check(
        GENERATION_GRAVITY_HOLD_COVERAGE_CHECK,
        not coverage_failures,
        _format_generation_physics_findings(
            "Explicit gravity_hold authoring intent is under-covered. Findings:",
            coverage_failures,
        ),
    )
    ctx.check(
        GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK,
        not newton_impedance_failures,
        _format_generation_physics_findings(
            "Explicit gravity_hold authoring intent lacks Newton dry-friction impedance. Findings:",
            newton_impedance_failures,
        ),
    )
    for finding in coverage_warnings:
        ctx.warn(f"{GENERATION_GRAVITY_HOLD_COVERAGE_CHECK}: {finding}")
    for finding in newton_impedance_warnings:
        ctx.warn(f"{GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK}: {finding}")


def _check_generation_physics_completeness(ctx: Any, object_model: object) -> None:
    body_findings: list[str] = []
    for part in _generation_movable_rigid_parts(object_model):
        part_name = str(getattr(part, "name", ""))
        missing: list[str] = []
        inertial = getattr(part, "inertial", None)
        if inertial is None:
            missing.append("inertial")
        elif not _positive_finite(getattr(inertial, "mass", None)):
            missing.append("positive finite inertial.mass")

        rigid_body = getattr(part, "rigid_body", None)
        physics_material = (
            None if rigid_body is None else getattr(rigid_body, "physics_material", None)
        )
        has_collision_geometry = bool(getattr(part, "collisions", ()) or ()) or any(
            bool(getattr(visual, "collidable", True))
            for visual in list(getattr(part, "visuals", ()) or ())
        )
        if has_collision_geometry and (
            not isinstance(physics_material, str) or not physics_material.strip()
        ):
            missing.append("rigid_body.physics_material")
        if missing:
            body_findings.append(f"part={part_name!r}: missing {', '.join(missing)}")

    ctx.check(
        GENERATION_MOVABLE_BODY_PHYSICS_CHECK,
        not body_findings,
        _format_generation_physics_findings(
            "Every simulation-movable rigid Part must author explicit inertial mass and a "
            "part-level physics-material binding when collidable. Fixed-base static chains and "
            "visual-only Parts are exempt from the material requirement. Findings:",
            body_findings,
        ),
    )

    joint_findings: list[str] = []
    scalar_types = {"revolute", "continuous", "prismatic"}
    for articulation in list(getattr(object_model, "articulations", ()) or ()):
        joint_type = _enum_token(getattr(articulation, "articulation_type", ""))
        if joint_type not in scalar_types:
            continue
        missing = []
        motion_limits = getattr(articulation, "motion_limits", None)
        effort: Any = None
        if motion_limits is None:
            missing.extend(("motion_limits.effort", "motion_limits.velocity"))
        else:
            effort = getattr(motion_limits, "effort", None)
            if not _positive_finite(effort):
                missing.append("positive finite motion_limits.effort")
            if not _positive_finite(getattr(motion_limits, "velocity", None)):
                missing.append("positive finite motion_limits.velocity")

        motion_properties = getattr(articulation, "motion_properties", None)
        passive_damping = (
            None if motion_properties is None else getattr(motion_properties, "damping", None)
        )
        drive = getattr(articulation, "drive", None)
        drive_damping = None if drive is None else getattr(drive, "damping", None)
        if drive is None:
            if not _nonnegative_finite(passive_damping):
                missing.append(
                    "passive joint requires non-negative finite motion_properties.damping"
                )
        else:
            if passive_damping is None:
                missing.append(
                    "active drive requires motion_properties.damping explicitly set to 0"
                )
            elif not _nonnegative_finite(passive_damping) or float(passive_damping) != 0.0:
                missing.append(
                    "active drive conflicts with non-zero motion_properties.damping; set "
                    "motion_properties.damping=0 and use drive.damping"
                )
            if not _nonnegative_finite(drive_damping):
                missing.append("active drive requires non-negative finite drive.damping")

            max_force = getattr(drive, "max_force", None)
            if max_force is not None:
                if not _nonnegative_finite(max_force):
                    missing.append("drive.max_force must be finite and non-negative")
                elif _positive_finite(effort) and float(max_force) > float(effort):
                    missing.append("drive.max_force must not exceed motion_limits.effort")

        friction = (
            None if motion_properties is None else getattr(motion_properties, "friction", None)
        )
        if not _nonnegative_finite(friction):
            missing.append(
                "non-negative finite motion_properties.friction for the Newton friction contract"
            )

        physx_resistance = getattr(articulation, "physx_joint_resistance", None)
        if physx_resistance is None:
            missing.append("explicit physx_joint_resistance")
        elif hasattr(physx_resistance, "coefficient"):
            if not _nonnegative_finite(getattr(physx_resistance, "coefficient", None)):
                missing.append("non-negative finite PhysX load-proportional coefficient")
        elif all(
            hasattr(physx_resistance, field_name)
            for field_name in (
                "static_friction_effort",
                "dynamic_friction_effort",
                "viscous_friction_coefficient",
            )
        ):
            static_effort: Any = getattr(physx_resistance, "static_friction_effort", None)
            dynamic_effort: Any = getattr(physx_resistance, "dynamic_friction_effort", None)
            viscous: Any = getattr(physx_resistance, "viscous_friction_coefficient", None)
            if not all(
                _nonnegative_finite(value) for value in (static_effort, dynamic_effort, viscous)
            ):
                missing.append("finite non-negative PhysX effort-friction properties")
            elif float(dynamic_effort) > float(static_effort):
                missing.append(
                    "PhysX dynamic_friction_effort no greater than static_friction_effort"
                )
            elif all(float(value) == 0.0 for value in (static_effort, dynamic_effort, viscous)):
                missing.append("at least one non-zero PhysX effort-friction property")
        else:
            missing.append("recognized PhysX joint-resistance model")

        physx_armature = getattr(articulation, "physx_armature", None)
        if physx_armature is not None and not _nonnegative_finite(physx_armature):
            missing.append("finite non-negative physx_armature")
        if missing:
            joint_name = str(getattr(articulation, "name", ""))
            joint_findings.append(
                f"joint={joint_name!r} type={joint_type!r}: missing {', '.join(missing)}"
            )

    ctx.check(
        GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK,
        not joint_findings,
        _format_generation_physics_findings(
            "Every interactive scalar joint must author effort, velocity, a deliberate PhysX "
            "joint-resistance model, the Newton friction value, and unambiguous "
            "damping. Passive joints use motion_properties.damping. Active drive joints must set "
            "motion_properties.damping=0 and use drive.damping; an explicit drive.max_force must "
            "be finite, non-negative, and no greater than motion_limits.effort. Use "
            "PhysxJointFrictionCoefficient(0.0) for an intentionally frictionless PhysX joint. "
            "Findings:",
            joint_findings,
        ),
    )
