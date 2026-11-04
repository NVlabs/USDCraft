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

"""Reconstruction Author tools over frozen mesh evidence and compiled candidates.

Both execution arms use these functions. They perform no semantic interpretation and start no
agent. Mesh queries deliberately bypass the AXIAL feature detector so an Author can remeasure a
region whose automatic interpretation is uncertain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom

from usdcraft.render.browser import RenderRequest, render_generated_model_views

MAX_QUERY_POINTS = 4096
MAX_MESH_VERTICES = 2_000_000
MAX_QUERY_COMPONENTS = 4096


def _bounds(value: Sequence[float] | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError("bounds must contain six finite metres: xmin ymin zmin xmax ymax zmax")
    if np.any(array[:3] >= array[3:]):
        raise ValueError("bounds minima must be smaller than maxima")
    return array


def measure_mesh(
    source: Path,
    *,
    out_dir: Path,
    bounds: Sequence[float] | None = None,
    max_points: int = 256,
) -> dict[str, Any]:
    """Return exact selected-vertex bounds and bounded samples/projections in source world frame.

    Bounds select vertices, not a clipped solid or semantic part. Sparse and empty selections are
    observations, not proof of missing parts. Input is the staged metre/Z-up normalized surface.
    """
    region = _bounds(bounds)
    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise ValueError("max_points must be an integer")
    if not 1 <= max_points <= MAX_QUERY_POINTS:
        raise ValueError(f"max_points must be between 1 and {MAX_QUERY_POINTS}")
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError("Could not open the staged source mesh")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("measure_mesh expects the normalized Z-up, metre source")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    selected = []
    total = 0
    component_count = 0
    selected_component_count = 0
    components: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Points)):
            continue
        points = UsdGeom.PointBased(prim).GetPointsAttr().Get()
        if points is None:
            continue
        component_id = f"c{component_count:06d}"
        component_count += 1
        total += len(points)
        if total > MAX_MESH_VERTICES:
            raise ValueError(f"Source exceeds query vertex limit {MAX_MESH_VERTICES}")
        vertices = np.asarray(points, dtype=float)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float)
        vertices = vertices @ matrix[:3, :3] + matrix[3, :3]
        if not np.isfinite(vertices).all():
            raise ValueError("Source contains nonfinite vertices")
        source_count = len(vertices)
        if region is not None:
            vertices = vertices[np.all((vertices >= region[:3]) & (vertices <= region[3:]), axis=1)]
        if len(vertices):
            selected.append(vertices)
            selected_component_count += 1
            if len(components) < MAX_QUERY_COMPONENTS:
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                components.append(
                    {
                        "id": component_id,
                        "source_vertex_count": source_count,
                        "selected_vertex_count": len(vertices),
                        "min_xyz_m": lo.tolist(),
                        "max_xyz_m": hi.tolist(),
                        "size_xyz_m": (hi - lo).tolist(),
                    }
                )
    values = np.vstack(selected) if selected else np.empty((0, 3))
    payload: dict[str, Any] = {
        "ok": True,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "units": "metres",
        "frame": "normalized_source_world_q0_Z_up",
        "selection": "vertices within inclusive bounds; not semantic segmentation or solid clipping",
        "bounds": None if region is None else region.tolist(),
        "source_vertex_count": total,
        "selected_vertex_count": len(values),
        "empty_is_absence_evidence": False,
        "component_scope": "anonymous source geometry prims, not semantic parts; bounds cover selected vertices only",
        "source_component_count": component_count,
        "selected_component_count": selected_component_count,
        "components": components,
        "stored_component_count": len(components),
        "components_truncated": selected_component_count > len(components),
        "images": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(values):
        lo, hi = values.min(axis=0), values.max(axis=0)
        indices = np.linspace(0, len(values) - 1, min(max_points, len(values)), dtype=int)
        payload.update(
            min_xyz_m=lo.tolist(),
            max_xyz_m=hi.tolist(),
            size_xyz_m=(hi - lo).tolist(),
            quantiles_xyz_m=np.quantile(values, [0.01, 0.5, 0.99], axis=0).tolist(),
            points_xyz_m=values[indices].tolist(),
            sampling="deterministic vertex-index subsample; not uniform surface area",
        )
        # A larger display sample remains bounded independently of the returned coordinate array.
        display = values[np.linspace(0, len(values) - 1, min(4096, len(values)), dtype=int)]
        sheet = Image.new("RGB", (1536, 560), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for tile, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            scale = 440 / max(float(hi[a] - lo[a]), float(hi[b] - lo[b]), 1e-9)
            for point in display:
                x = tile * 512 + 35 + (point[a] - lo[a]) * scale
                y = 500 - (point[b] - lo[b]) * scale
                draw.ellipse((x, y, x + 1, y + 1), fill=(40, 70, 110))
            draw.text(
                (tile * 512 + 15, 12),
                f"{'XYZ'[a]} horizontal / {'XYZ'[b]} vertical; metres",
                fill=(0, 0, 0),
            )
            draw.text(
                (tile * 512 + 15, 530),
                f"min {lo[a]:.4f}, {lo[b]:.4f} max {hi[a]:.4f}, {hi[b]:.4f}",
                fill=(0, 0, 0),
            )
        path = out_dir / "mesh_projections.png"
        sheet.save(path)
        payload["images"] = [str(path.resolve())]
    if len(values) >= 3:
        centred = values - values.mean(axis=0)
        eigenvalues, axes = np.linalg.eigh(centred.T @ centred / len(values))
        normal = axes[:, 0]
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        payload["plane_fit"] = {
            "normal_xyz": normal.round(6).tolist(),
            "centroid_xyz_m": values.mean(axis=0).round(6).tolist(),
            "residual_p90_mm": round(float(np.percentile(np.abs(centred @ normal), 90)) * 1000, 3),
            "eigenvalues_m2": eigenvalues.tolist(),
            "inclination_to_xy_deg": round(
                float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))), 2
            ),
            "scope": "vertex-weighted least-squares plane; assess residual and eigenvalues before using orientation; mixed or linear regions are ambiguous",
        }
    detail = out_dir / "measurement.json"
    detail.write_text(json.dumps(payload, indent=2) + "\n")
    return {
        **payload,
        "points_xyz_m": payload.get("points_xyz_m", [])[:16],
        "components": components[:16],
        "returned_component_count": min(16, len(components)),
        "detail_path": str(detail.resolve()),
        "returned_sample_count": min(16, len(values)),
        "stored_sample_count": min(max_points, len(values)),
    }


def render_source(
    asset_dir: Path,
    *,
    out_dir: Path,
    azimuth: float = 45,
    elevation: float = 30,
    renderer: Callable | None = None,
) -> dict[str, Any]:
    """View the staged source carrier at an Author-selected camera, before any modeling."""
    if not np.isfinite([azimuth, elevation]).all() or not -90 <= elevation <= 90:
        raise ValueError("Camera must be finite with elevation within [-90, 90] degrees")
    urdf = asset_dir / "model.urdf"
    glb = asset_dir / "assets" / "source.glb"
    identity = {
        "source_carrier_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "source_carrier_glb_sha256": hashlib.sha256(glb.read_bytes()).hexdigest(),
    }
    result = (renderer or render_generated_model_views)(
        RenderRequest(
            asset_dir=asset_dir,
            out_dir=out_dir,
            views=(("source", float(azimuth), float(elevation)),),
            size=768,
        )
    )
    payload = {
        "ok": True,
        "images": [str(p.resolve()) for p in result.image_paths],
        "backend": result.backend,
        "fallback_reason": getattr(result, "fallback_reason", None),
        "capabilities": result.capabilities,
        "azimuth": azimuth,
        "elevation": elevation,
        "scope": "frozen source surface; qualitative shape evidence, not calibrated pixel measurements",
        "units": "metres",
        "up_axis": "Z",
        **identity,
    }
    (out_dir / "feedback.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def render_candidate(
    asset_dir: Path,
    *,
    out_dir: Path,
    azimuth: float = 45,
    elevation: float = 30,
    joint_positions: Mapping[str, float] | None = None,
    renderer: Callable | None = None,
) -> dict[str, Any]:
    """Render actual compiled visual geometry at a requested camera and joint pose."""
    from xml.etree import ElementTree

    positions = dict(joint_positions or {})
    if not np.isfinite([azimuth, elevation, *positions.values()]).all():
        raise ValueError("Camera and joint positions must be finite")
    if not -90 <= elevation <= 90:
        raise ValueError("elevation must be within [-90, 90] degrees")
    urdf = asset_dir / "model.urdf"
    names = {j.attrib["name"] for j in ElementTree.parse(urdf).getroot().findall("joint")}
    if set(positions) - names:
        raise ValueError(f"Unknown joints: {sorted(set(positions) - names)}")
    result = (renderer or render_generated_model_views)(
        RenderRequest(
            asset_dir=asset_dir,
            out_dir=out_dir,
            views=(("author", float(azimuth), float(elevation)),),
            size=768,
            joint_positions=positions,
        )
    )
    payload = {
        "ok": True,
        "images": [str(p.resolve()) for p in result.image_paths],
        "backend": result.backend,
        "fallback_reason": getattr(result, "fallback_reason", None),
        "capabilities": result.capabilities,
        "joint_positions": positions,
        "azimuth": azimuth,
        "elevation": elevation,
        "model_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "scope": "compiled visual geometry; posed kinematically, not a physics simulation",
    }
    (out_dir / "feedback.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
