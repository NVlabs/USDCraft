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

"""Deterministic subject-isolation evidence and frozen-decision application.

This module owns the mechanical half of Reconstruction subject isolation.  It inventories complete
surface components, offers only whole-component selection units, validates one frozen isolation
decision against exact source/frame identity, and applies that decision without changing any
retained coordinate or topology.  The clean-context Agent that makes the qualitative decision is
implemented separately in :mod:`usdcraft.reconstruction.isolation_agent`.

The inventory is surface-only.  Component connectivity is lineage, not semantic Part identity or
solid authority, and an empty band is only a deterministic selection convenience.  It never proves
free space or independently decides which side is the primary subject.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from usdcraft.reconstruction.source import (
    NormalizedSourceComponent,
    NormalizedSourceScene,
    component_data_sha256,
)

ISOLATION_INVENTORY_SCHEMA = "reconstruction_subject_isolation_inventory"
SUBJECT_ISOLATION_SCHEMA = "reconstruction_subject_isolation"
SUBJECT_ISOLATION_AGENT_CONTEXT = "clean_subject_isolation_only"
SUBJECT_ISOLATION_SKIPPED_CONTEXT = "mesh_cleanup_not_requested"
_MAX_AXIS_GAPS_PER_AXIS = 8


class SubjectIsolationError(RuntimeError):
    """A subject-isolation artifact is invalid, stale, or unsafe to apply."""


@dataclass(frozen=True, slots=True)
class IsolationSelectionUnit:
    selection_id: str
    component_ids: tuple[str, ...]
    selector_kind: str
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SubjectIsolationDecision:
    manifest_path: Path
    source_identity_sha256: str
    retained_component_ids: tuple[str, ...]
    excluded_component_ids: tuple[str, ...]
    unresolved_component_ids: tuple[str, ...]
    upstream_agent_run: bool
    upstream_agent_context: str


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_scene_identity_payload(scene: NormalizedSourceScene) -> dict[str, object]:
    """Return a path-free identity binding geometry, dependencies, and declared frame."""

    return {
        "schema": "reconstruction_normalized_scene_identity",
        "source_format": scene.source_format,
        "source_dependencies": [
            {
                "sha256": dependency.sha256,
                "byte_count": dependency.byte_count,
            }
            for dependency in scene.source_dependencies
        ],
        "frame": {
            "up_axis": scene.frame.up_axis,
            "up_axis_source": scene.frame.up_axis_source,
            "meters_per_unit": scene.frame.meters_per_unit,
            "meters_per_unit_source": scene.frame.meters_per_unit_source,
            "front_axis": scene.frame.front_axis,
            "front_axis_source": scene.frame.front_axis_source,
            "source_to_normalized": [list(row) for row in scene.frame.source_to_normalized],
        },
        "components": [
            {
                "component_id": component.component_id,
                "kind": component.kind,
                "data_sha256": component_data_sha256(component),
            }
            for component in scene.components
        ],
    }


def normalized_scene_identity_sha256(scene: NormalizedSourceScene) -> str:
    return _canonical_json_sha256(normalized_scene_identity_payload(scene))


def _vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _surface_area(component: NormalizedSourceComponent) -> float:
    if component.kind != "mesh" or component.faces is None:
        return 0.0
    triangles = component.vertices[component.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    value = float(np.linalg.norm(cross, axis=1).sum() * 0.5)
    if not math.isfinite(value) or value < 0.0:
        raise SubjectIsolationError(
            f"component {component.component_id!r} has non-finite surface area"
        )
    return value


def _display_rgba(index: int) -> tuple[int, int, int, int]:
    # Golden-ratio hue stepping keeps adjacent deterministic component ids visually separated.
    hue = (0.11 + index * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.92)
    return (
        int(round(red * 255.0)),
        int(round(green * 255.0)),
        int(round(blue * 255.0)),
        255,
    )


def _component_inventory_payloads(
    scene: NormalizedSourceScene,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    surface_areas = {
        component.component_id: _surface_area(component) for component in scene.components
    }
    total_surface_area = sum(surface_areas.values())
    total_points = sum(len(component.vertices) for component in scene.components)
    payloads: list[dict[str, object]] = []
    for index, component in enumerate(scene.components):
        minimum = component.bounds_min_xyz
        maximum = component.bounds_max_xyz
        rgba = _display_rgba(index)
        area = surface_areas[component.component_id]
        payloads.append(
            {
                "component_id": component.component_id,
                "selection_id": f"component:{component.component_id}",
                "kind": component.kind,
                "data_sha256": component_data_sha256(component),
                "bounds_min_xyz_m": _vector(minimum),
                "bounds_max_xyz_m": _vector(maximum),
                "extent_xyz_m": _vector(maximum - minimum),
                "center_xyz_m": _vector((minimum + maximum) * 0.5),
                "vertex_count": len(component.vertices),
                "triangle_count": component.topology.triangle_count,
                "surface_area_m2": area,
                "surface_area_fraction": area / total_surface_area
                if total_surface_area > 0.0
                else 0.0,
                "point_fraction": len(component.vertices) / total_points
                if total_points > 0
                else 0.0,
                "topology_classification": component.topology.classification,
                "display_rgba": list(rgba),
                "display_hex": f"#{rgba[0]:02X}{rgba[1]:02X}{rgba[2]:02X}",
                "authority": "surface_only_component_lineage",
            }
        )
    return payloads, surface_areas


def _selection_summary(
    component_ids: Sequence[str],
    *,
    by_id: Mapping[str, NormalizedSourceComponent],
    surface_areas: Mapping[str, float],
    total_surface_area: float,
    total_vertices: int,
) -> dict[str, object]:
    selected = [by_id[component_id] for component_id in component_ids]
    vertices = np.concatenate([component.vertices for component in selected], axis=0)
    area = sum(surface_areas[component_id] for component_id in component_ids)
    vertex_count = sum(len(component.vertices) for component in selected)
    return {
        "component_count": len(component_ids),
        "component_ids": list(component_ids),
        "bounds_min_xyz_m": _vector(vertices.min(axis=0)),
        "bounds_max_xyz_m": _vector(vertices.max(axis=0)),
        "surface_area_fraction": area / total_surface_area if total_surface_area > 0.0 else 0.0,
        "vertex_fraction": vertex_count / total_vertices if total_vertices > 0 else 0.0,
    }


def _strict_axis_gap_units(
    scene: NormalizedSourceScene,
    *,
    surface_areas: Mapping[str, float],
) -> tuple[IsolationSelectionUnit, ...]:
    components = tuple(scene.components)
    if len(components) < 2:
        return ()
    by_id = {component.component_id: component for component in components}
    total_surface_area = sum(surface_areas.values())
    total_vertices = sum(len(component.vertices) for component in components)
    longest_extent = float(np.max(scene.bounds_max_xyz - scene.bounds_min_xyz))
    epsilon = max(longest_extent * 1e-9, 1e-12)
    result: list[IsolationSelectionUnit] = []
    axis_names = "XYZ"
    for axis in range(3):
        ordered = sorted(
            components,
            key=lambda item: (
                float(item.bounds_min_xyz[axis]),
                float(item.bounds_max_xyz[axis]),
                item.component_id,
            ),
        )
        prefix_maximum = -math.inf
        candidates: list[tuple[int, float, float, float]] = []
        for split in range(len(ordered) - 1):
            prefix_maximum = max(prefix_maximum, float(ordered[split].bounds_max_xyz[axis]))
            next_minimum = float(ordered[split + 1].bounds_min_xyz[axis])
            width = next_minimum - prefix_maximum
            if width > epsilon:
                candidates.append((split, prefix_maximum, next_minimum, width))
        selected_candidates = sorted(
            candidates,
            key=lambda item: (-item[3], item[1], item[0]),
        )[:_MAX_AXIS_GAPS_PER_AXIS]
        selected_candidates.sort(key=lambda item: (item[1], item[2], item[0]))
        for gap_index, (split, gap_minimum, gap_maximum, width) in enumerate(selected_candidates):
            low_ids = tuple(sorted(item.component_id for item in ordered[: split + 1]))
            high_ids = tuple(sorted(item.component_id for item in ordered[split + 1 :]))
            gap_id = f"gap_{axis_names[axis].lower()}_{gap_index:03d}"
            common = {
                "axis": axis_names[axis],
                "gap_min_m": gap_minimum,
                "gap_max_m": gap_maximum,
                "gap_width_m": width,
                "evidence_semantics": (
                    "strict_component_aabb_separation_only_not_free_space_or_subject_authority"
                ),
            }
            for side, component_ids in (("low", low_ids), ("high", high_ids)):
                metadata = {
                    **common,
                    "side": side,
                    **_selection_summary(
                        component_ids,
                        by_id=by_id,
                        surface_areas=surface_areas,
                        total_surface_area=total_surface_area,
                        total_vertices=total_vertices,
                    ),
                }
                result.append(
                    IsolationSelectionUnit(
                        selection_id=f"{gap_id}:{side}",
                        component_ids=component_ids,
                        selector_kind="strict_axis_empty_band_side",
                        metadata=metadata,
                    )
                )
    return tuple(result)


def build_subject_isolation_inventory(scene: NormalizedSourceScene) -> dict[str, object]:
    """Build the exact path-free evidence inventory shown to the isolation Agent."""

    component_payloads, surface_areas = _component_inventory_payloads(scene)
    selection_units = _strict_axis_gap_units(scene, surface_areas=surface_areas)
    extent = scene.bounds_max_xyz - scene.bounds_min_xyz
    payload: dict[str, object] = {
        "schema": ISOLATION_INVENTORY_SCHEMA,
        "evidence_role": "subject_isolation_only",
        "reconstruction_agent_input": False,
        "geometry_authority": "surface_only",
        "interior_semantics": "unknown",
        "solid_authority": "none",
        "source_identity_sha256": normalized_scene_identity_sha256(scene),
        "frame": {
            "coordinate_system": "recon_world_pre_subject_normalization",
            "up_axis": "+Z",
            "meters_per_unit": 1.0,
            "semantic_front_resolved": scene.frame.front_axis is not None,
            "frame_decode_only": True,
            "recentered": False,
            "scaled": False,
        },
        "scene": {
            "bounds_min_xyz_m": _vector(scene.bounds_min_xyz),
            "bounds_max_xyz_m": _vector(scene.bounds_max_xyz),
            "extent_xyz_m": _vector(extent),
            "component_count": len(scene.components),
            "mesh_component_count": sum(item.kind == "mesh" for item in scene.components),
            "point_component_count": sum(item.kind == "points" for item in scene.components),
            "total_surface_area_m2": sum(surface_areas.values()),
            "total_vertex_count": sum(len(item.vertices) for item in scene.components),
        },
        "components": component_payloads,
        "selection_units": [
            {
                "selection_id": unit.selection_id,
                "selector_kind": unit.selector_kind,
                **dict(unit.metadata),
            }
            for unit in selection_units
        ],
        "selection_contract": {
            "individual_component_selection_id": "component:<component_id>",
            "selection_units_expand_to_complete_components": True,
            "triangle_or_point_clipping": False,
            "unspecified_components": "retain",
            "unresolved_components": "retain",
        },
    }
    payload["inventory_sha256"] = _canonical_json_sha256(payload)
    return payload


def isolation_selection_lookup(
    inventory: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    components = inventory.get("components")
    units = inventory.get("selection_units")
    if not isinstance(components, list) or not isinstance(units, list):
        raise SubjectIsolationError("isolation inventory is missing components or selection units")
    result: dict[str, tuple[str, ...]] = {}
    known_components: set[str] = set()
    for item in components:
        if not isinstance(item, Mapping):
            raise SubjectIsolationError("isolation inventory component must be an object")
        component_id = item.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise SubjectIsolationError("isolation inventory component id is invalid")
        if component_id in known_components:
            raise SubjectIsolationError(f"duplicate isolation component id {component_id!r}")
        known_components.add(component_id)
        result[f"component:{component_id}"] = (component_id,)
    for item in units:
        if not isinstance(item, Mapping):
            raise SubjectIsolationError("isolation selection unit must be an object")
        selection_id = item.get("selection_id")
        component_ids = item.get("component_ids")
        if not isinstance(selection_id, str) or not selection_id:
            raise SubjectIsolationError("isolation selection id is invalid")
        if selection_id in result:
            raise SubjectIsolationError(f"duplicate isolation selection id {selection_id!r}")
        if not isinstance(component_ids, list) or not component_ids:
            raise SubjectIsolationError(
                f"isolation selection {selection_id!r} has no component ids"
            )
        normalized = tuple(str(value) for value in component_ids)
        if len(set(normalized)) != len(normalized) or not set(normalized) <= known_components:
            raise SubjectIsolationError(
                f"isolation selection {selection_id!r} contains invalid component ids"
            )
        result[selection_id] = normalized
    return result


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise SubjectIsolationError(f"{label} must be an array of non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise SubjectIsolationError(f"{label} must not contain duplicates")
    return result


def load_subject_isolation_decision(
    manifest_path: Path,
    scene: NormalizedSourceScene,
) -> SubjectIsolationDecision:
    """Load one ready frozen decision and bind it to the exact current source scene."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise SubjectIsolationError(f"subject isolation manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubjectIsolationError(f"subject isolation manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SubjectIsolationError("subject isolation manifest must be a JSON object")
    if payload.get("schema") != SUBJECT_ISOLATION_SCHEMA:
        raise SubjectIsolationError(
            f"unsupported subject isolation schema {payload.get('schema')!r}"
        )
    if payload.get("status") != "ready":
        raise SubjectIsolationError("subject isolation manifest is not ready")
    expected_identity = normalized_scene_identity_sha256(scene)
    observed_identity = payload.get("source_identity_sha256")
    if observed_identity != expected_identity:
        raise SubjectIsolationError(
            "subject isolation source/frame identity does not match the requested source"
        )
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise SubjectIsolationError("subject isolation manifest is missing selection")
    retained = _string_tuple(
        selection.get("retained_component_ids"), label="retained_component_ids"
    )
    excluded = _string_tuple(
        selection.get("excluded_component_ids"), label="excluded_component_ids"
    )
    unresolved = _string_tuple(
        selection.get("unresolved_component_ids"), label="unresolved_component_ids"
    )
    known = tuple(component.component_id for component in scene.components)
    known_set = set(known)
    retained_set = set(retained)
    excluded_set = set(excluded)
    if not retained:
        raise SubjectIsolationError("subject isolation must retain at least one complete component")
    if retained_set & excluded_set:
        raise SubjectIsolationError("retained and excluded subject-isolation components overlap")
    if retained_set | excluded_set != known_set:
        raise SubjectIsolationError(
            "subject isolation retained/excluded components do not exactly partition the source"
        )
    if not set(unresolved) <= retained_set:
        raise SubjectIsolationError("unresolved subject-isolation components must be retained")
    if retained != tuple(component_id for component_id in known if component_id in retained_set):
        raise SubjectIsolationError("retained component ids are not in stable source order")
    if excluded != tuple(component_id for component_id in known if component_id in excluded_set):
        raise SubjectIsolationError("excluded component ids are not in stable source order")
    agent = payload.get("agent")
    if not isinstance(agent, Mapping):
        raise SubjectIsolationError("subject isolation manifest is missing its agent declaration")
    agent_run = agent.get("run")
    agent_context = agent.get("context")
    if agent_run is True and agent_context == SUBJECT_ISOLATION_AGENT_CONTEXT:
        pass
    elif agent_run is False and agent_context == SUBJECT_ISOLATION_SKIPPED_CONTEXT:
        if excluded or unresolved:
            raise SubjectIsolationError(
                "skipped mesh cleanup must retain every complete component without "
                "exclusions or unresolved retentions"
            )
    else:
        raise SubjectIsolationError(
            "subject isolation must come from one explicit clean-context isolation Agent run "
            "or one explicit mesh-cleanup-not-requested retain-all declaration"
        )
    return SubjectIsolationDecision(
        manifest_path=path,
        source_identity_sha256=expected_identity,
        retained_component_ids=retained,
        excluded_component_ids=excluded,
        unresolved_component_ids=unresolved,
        upstream_agent_run=bool(agent_run),
        upstream_agent_context=str(agent_context),
    )


def apply_subject_isolation(
    scene: NormalizedSourceScene,
    decision: SubjectIsolationDecision,
) -> NormalizedSourceScene:
    """Return the retained surface subset without changing any retained authored sample."""

    identity = normalized_scene_identity_sha256(scene)
    if identity != decision.source_identity_sha256:
        raise SubjectIsolationError("subject isolation decision is stale for this source scene")
    retained_ids = set(decision.retained_component_ids)
    retained = tuple(
        component for component in scene.components if component.component_id in retained_ids
    )
    if not retained:
        raise SubjectIsolationError("subject isolation produced an empty source scene")
    vertices = np.concatenate([component.vertices for component in retained], axis=0)
    warning = (
        "subject isolation retained complete authored surface components only; excluded surfaces "
        "remain available in the separate frozen isolation audit and prove neither empty space "
        "nor source interior"
    )
    return replace(
        scene,
        components=retained,
        bounds_min_xyz=vertices.min(axis=0),
        bounds_max_xyz=vertices.max(axis=0),
        warnings=tuple((*scene.warnings, warning)),
    )


def write_retain_all_subject_isolation(
    scene: NormalizedSourceScene,
    output_dir: Path,
) -> Path:
    """Freeze one explicit retain-all decision (mesh cleanup not requested) without starting any Agent.

    Mesh cleanup is optional: when the user does not request it, the Host records this
    deterministic declaration instead of running the Subject-Isolation Agent. Every complete
    source-surface component is retained unchanged, so downstream observation, rendering, and
    persistence consume the same frozen-decision contract in both modes.
    """

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise SubjectIsolationError(f"output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_identity = normalized_scene_identity_sha256(scene)
    inventory = build_subject_isolation_inventory(scene)
    retained = [component.component_id for component in scene.components]
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        inventory_path = staging / "component_inventory.json"
        inventory_path.write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": SUBJECT_ISOLATION_SCHEMA,
            "status": "ready",
            "scope": "reconstruction_subject_isolation_preprocessing",
            "canonical_record": False,
            "source_identity_sha256": source_identity,
            "source_identity": normalized_scene_identity_payload(scene),
            "processing_order": [
                "lossless_declared_frame_decode",
                "subject_isolation",
                "subject_rotation_recenter_or_scale_if_later_declared",
                "normalized_source_snapshot",
                "axial_and_source_overlay",
            ],
            "selection": {
                "retained_component_ids": retained,
                "excluded_component_ids": [],
                "unresolved_component_ids": [],
                "retained_component_count": len(retained),
                "excluded_component_count": 0,
                "unresolved_component_count": 0,
                "decision_units": [],
                "retained_samples_unchanged": True,
                "triangle_or_point_clipping": False,
                "repair_fill_or_inferred_surface": False,
                "excluded_surface_semantics": "not_subject_not_proven_empty_space",
            },
            "agent": {
                "run": False,
                "context": SUBJECT_ISOLATION_SKIPPED_CONTEXT,
                "reason": (
                    "user did not request mesh cleanup; every complete source-surface "
                    "component is retained without any isolation Agent"
                ),
                "attached_image_count": 0,
                "semantic_image": False,
                "axial": False,
                "source_mesh": False,
                "sdk": False,
                "later_reconstruction_context": False,
            },
            "downstream_contract": {
                "reconstruction_agent_input": False,
                "recon_observe_requires_ready_frozen_decision": True,
                "same_retained_scene_for_normalized_source_axial_renders_and_evaluation": (True),
                "render_only_or_axial_only_filtering": False,
            },
            "artifacts": {
                "inventory": {
                    "path": "component_inventory.json",
                    "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                    "byte_count": inventory_path.stat().st_size,
                    "isolation_agent_input": False,
                },
            },
        }
        (staging / "subject_isolation.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "subject_isolation.json"


__all__ = [
    "ISOLATION_INVENTORY_SCHEMA",
    "SUBJECT_ISOLATION_AGENT_CONTEXT",
    "SUBJECT_ISOLATION_SCHEMA",
    "SUBJECT_ISOLATION_SKIPPED_CONTEXT",
    "IsolationSelectionUnit",
    "SubjectIsolationDecision",
    "SubjectIsolationError",
    "apply_subject_isolation",
    "build_subject_isolation_inventory",
    "isolation_selection_lookup",
    "load_subject_isolation_decision",
    "normalized_scene_identity_payload",
    "normalized_scene_identity_sha256",
    "write_retain_all_subject_isolation",
]
