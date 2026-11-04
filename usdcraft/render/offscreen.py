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

"""Headless software renderer for URDF models (pure numpy, no GL/GPU).

Renders a URDF to PNG views with a numpy z-buffer rasterizer that runs identically on macOS,
Linux, and headless CI. Generation uses these views as non-blocking authoring diagnostics.

Fidelity notes: Lambert shading with per-visual base color and alpha
(`usdcraft_pbr` transmission renders as transparency). Gloss/reflections are
not simulated; consumers must not judge specular appearance from these
renders.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from PIL import Image

from usdcraft.render.contract import STANDARD_RENDER_VIEWS

DEFAULT_VIEWS: Tuple[Tuple[str, float, float], ...] = STANDARD_RENDER_VIEWS
DEFAULT_FOV_DEG = 35.0


_LIGHT_DIRS = np.array([[0.4, -0.6, 0.7], [-0.6, -0.3, 0.4]], dtype=np.float64)
_LIGHT_WEIGHTS = (0.55, 0.30)
_AMBIENT = 0.38


@dataclass
class VisualBatch:
    vertices: np.ndarray  # (N, 3) world space
    faces: np.ndarray  # (M, 3) int
    rgba: Tuple[float, float, float, float]
    uvs: Optional[np.ndarray] = None  # (N, 2) per-vertex texture coords
    texture: Optional[np.ndarray] = None  # (H, W, 4) uint8 RGBA texture image
    link: Optional[str] = None  # owning URDF link name, when known
    element: Optional[str] = None  # owning URDF visual name, when known


# --- URDF assembly -----------------------------------------------------------


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _origin_transform(elem: Optional[ET.Element]) -> np.ndarray:
    transform = np.eye(4)
    if elem is None:
        return transform
    xyz = [float(v) for v in (elem.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (elem.get("rpy") or "0 0 0").split()]
    transform[:3, :3] = _rpy_matrix(*rpy)
    transform[:3, 3] = xyz
    return transform


def _parse_rgba(material: Optional[ET.Element]) -> Optional[Tuple[float, float, float, float]]:
    if material is None:
        return None
    color = material.find("color")
    if color is None or not color.get("rgba"):
        return None
    values = [float(v) for v in cast(str, color.get("rgba")).split()]
    if len(values) == 3:
        values.append(1.0)
    # usdcraft_pbr transmission renders as extra transparency.
    pbr = material.find("usdcraft_pbr")
    if pbr is not None and pbr.get("transmission"):
        transmission = float(cast(str, pbr.get("transmission")))
        values[3] = min(values[3], 1.0 - 0.85 * transmission)
    return (values[0], values[1], values[2], values[3])


def _primitive_mesh(geom: ET.Element) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    import trimesh

    box = geom.find("box")
    if box is not None:
        size = [float(v) for v in cast(str, box.get("size")).split()]
        mesh = trimesh.creation.box(extents=size)
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)
    cylinder = geom.find("cylinder")
    if cylinder is not None:
        mesh = trimesh.creation.cylinder(
            radius=float(cast(str, cylinder.get("radius"))),
            height=float(cast(str, cylinder.get("length"))),
            sections=24,
        )
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)
    sphere = geom.find("sphere")
    if sphere is not None:
        mesh = trimesh.creation.icosphere(
            subdivisions=2, radius=float(cast(str, sphere.get("radius")))
        )
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)
    return None


def _axis_angle_transform(axis: Sequence[float] | np.ndarray, angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.eye(4)
    x, y, z = vector / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one = 1.0 - c
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=np.float64,
    )
    return transform


def _joint_motion_transform(joint: ET.Element, value: float) -> np.ndarray:
    joint_type = str(joint.get("type") or "fixed")
    axis_element = joint.find("axis")
    axis_text = (
        str(joint.get("usdcraft_axis") or "")
        if str(joint.get("usdcraft_type") or "") == "spherical"
        else ""
    )
    if not axis_text:
        axis_text = cast(str, axis_element.get("xyz")) if axis_element is not None else "0 0 1"
    axis = [float(item) for item in axis_text.split()]
    if joint_type in {"revolute", "continuous"}:
        return _axis_angle_transform(axis, value)
    if joint_type == "prismatic":
        vector = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        transform = np.eye(4)
        if norm > 1e-12:
            transform[:3, 3] = vector / norm * float(value)
        return transform
    if str(joint.get("usdcraft_type") or "") == "spherical":
        rest_axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(rest_axis))
        if norm <= 1e-12:
            return np.eye(4)
        rest_axis /= norm
        references = (
            np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
            np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        )
        reference = min(references, key=lambda item: abs(float(np.dot(rest_axis, item))))
        swing_axis = np.cross(rest_axis, reference)
        return _axis_angle_transform(swing_axis, value)
    return np.eye(4)


def _resolve_joint_positions(
    joints: Sequence[ET.Element], requested: Mapping[str, float] | None
) -> dict[str, float]:
    values = {str(name): float(value) for name, value in (requested or {}).items()}
    by_name = {str(joint.get("name") or ""): joint for joint in joints}
    # Resolve simple URDF mimic chains after explicit source positions are known.
    for _ in range(len(joints) + 1):
        changed = False
        for name, joint in by_name.items():
            if name in values:
                continue
            mimic = joint.find("mimic")
            if mimic is None:
                continue
            source = str(mimic.get("joint") or "")
            if source not in values:
                continue
            multiplier = float(mimic.get("multiplier") or 1.0)
            offset = float(mimic.get("offset") or 0.0)
            values[name] = values[source] * multiplier + offset
            changed = True
        if not changed:
            break
    return values


def load_urdf_visuals(
    urdf_dir: Path,
    *,
    joint_positions: Mapping[str, float] | None = None,
) -> List[VisualBatch]:
    """Assemble URDF visual geometry in world space at a requested joint pose.

    ``urdf_dir`` must contain ``model.urdf``; mesh filenames are resolved
    relative to it. Unspecified joints remain at q=0.
    """
    import trimesh

    urdf_path = urdf_dir / "model.urdf"
    root = ET.fromstring(urdf_path.read_text())

    material_library: Dict[str, Tuple[float, float, float, float]] = {}
    for material in root.findall("material"):
        rgba = _parse_rgba(material)
        if material.get("name") and rgba:
            material_library[cast(str, material.get("name"))] = rgba

    # Rest-pose forward kinematics over the joint tree.
    link_transforms: Dict[str | None, np.ndarray] = {}
    joints = root.findall("joint")
    pose = _resolve_joint_positions(joints, joint_positions)
    children = {
        cast(ET.Element, j.find("child")).get("link") for j in joints if j.find("child") is not None
    }
    links = [link.get("name") for link in root.findall("link")]
    for link_name in links:
        if link_name not in children:
            link_transforms[link_name] = np.eye(4)
    pending = [j for j in joints]
    while pending:
        progressed = False
        remaining = []
        for joint in pending:
            parent = cast(ET.Element, joint.find("parent")).get("link")
            child = cast(ET.Element, joint.find("child")).get("link")
            if parent in link_transforms:
                joint_origin = _origin_transform(joint.find("origin"))
                motion = _joint_motion_transform(joint, pose.get(str(joint.get("name") or ""), 0.0))
                link_transforms[child] = link_transforms[parent] @ joint_origin @ motion
                progressed = True
            else:
                remaining.append(joint)
        pending = remaining
        if not progressed:
            for joint in pending:  # orphaned joints: place at identity
                link_transforms.setdefault(
                    cast(ET.Element, joint.find("child")).get("link"), np.eye(4)
                )
            break

    mesh_cache: Dict[
        str, Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]
    ] = {}
    batches: List[VisualBatch] = []
    for link in root.findall("link"):
        link_tf = link_transforms.get(link.get("name"), np.eye(4))
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            uvs = None
            texture = None
            mesh_el = geom.find("mesh")
            if mesh_el is not None:
                filename = mesh_el.get("filename") or ""
                if filename not in mesh_cache:
                    mesh_path = urdf_dir / filename
                    if not mesh_path.exists():
                        glb = mesh_path.with_suffix(".glb")
                        mesh_path = glb if glb.exists() else mesh_path
                    loaded = cast("trimesh.Trimesh", trimesh.load(mesh_path, force="mesh"))
                    mesh_cache[filename] = (
                        np.asarray(loaded.vertices, dtype=np.float64),
                        np.asarray(loaded.faces, dtype=np.int64),
                        _extract_uv_texture(loaded),
                    )
                vertices, faces, uv_tex = mesh_cache[filename]
                if uv_tex is not None:
                    uvs, texture = uv_tex
                scale = mesh_el.get("scale")
                if scale:
                    vertices = vertices * np.array([float(v) for v in scale.split()])
            else:
                primitive = _primitive_mesh(geom)
                if primitive is None:
                    continue
                vertices, faces = primitive

            rgba = _parse_rgba(visual.find("material"))
            if rgba is None:
                ref = visual.find("material")
                name = ref.get("name") if ref is not None else None
                rgba = material_library.get(name or "", (0.74, 0.74, 0.72, 1.0))

            transform = link_tf @ _origin_transform(visual.find("origin"))
            world = vertices @ transform[:3, :3].T + transform[:3, 3]
            batches.append(
                VisualBatch(
                    world,
                    np.asarray(faces, dtype=np.int64),
                    rgba,
                    uvs=uvs,
                    texture=texture,
                    link=link.get("name"),
                    element=visual.get("name"),
                )
            )
    return batches


def _extract_uv_texture(loaded) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Pull per-vertex UVs and the RGBA texture image out of a textured mesh."""
    visual = getattr(loaded, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is None:
        return None
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is None:
        return None
    texture = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return (np.asarray(uv, dtype=np.float64), texture)


# --- rasterizer ---------------------------------------------------------------


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def _shade(normals_cam: np.ndarray, base_rgb: np.ndarray) -> np.ndarray:
    n = normals_cam.copy()
    flip = n[:, 2] < 0
    n[flip] = -n[flip]
    intensity = np.full(n.shape[0], _AMBIENT)
    for direction, weight in zip(_LIGHT_DIRS, _LIGHT_WEIGHTS):
        d = direction / np.linalg.norm(direction)
        intensity += weight * np.clip(n @ d, 0.0, None)
    return np.clip(base_rgb[None, :] * np.clip(intensity, 0.0, 1.0)[:, None], 0.0, 1.0)


def _shade_intensity(normals_cam: np.ndarray) -> np.ndarray:
    """Per-face Lambert intensity in [0, 1] (the scalar _shade multiplies by base)."""
    n = normals_cam.copy()
    flip = n[:, 2] < 0
    n[flip] = -n[flip]
    intensity = np.full(n.shape[0], _AMBIENT)
    for direction, weight in zip(_LIGHT_DIRS, _LIGHT_WEIGHTS):
        d = direction / np.linalg.norm(direction)
        intensity += weight * np.clip(n @ d, 0.0, None)
    return np.clip(intensity, 0.0, 1.0)


def bbox_center_radius(batches: List[VisualBatch]) -> Tuple[np.ndarray, float]:
    """Bounding-sphere center + radius over every batch vertex (shared framing)."""
    all_vertices = np.concatenate([b.vertices for b in batches])
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2.0
    radius = float(np.linalg.norm(all_vertices - center, axis=1).max())
    return center, radius


def _view_and_focal(
    center: np.ndarray,
    radius: float,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    size: int,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> Tuple[np.ndarray, float]:
    """Camera (view matrix, focal) orbiting ``center`` at a fixed framing.

    Sharing this between the model and the mesh (same center/radius) is what
    makes their silhouette masks pixel-comparable for ``compare_to_mesh``.
    """
    distance = radius / math.tan(math.radians(fov_deg) / 2.0) * 1.25
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    eye = center + distance * np.array(
        [
            math.sin(azimuth) * math.cos(elevation),
            -math.cos(azimuth) * math.cos(elevation),
            math.sin(elevation),
        ]
    )
    view = _look_at(eye, center, np.array([0.0, 0.0, 1.0]))
    focal = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return view, focal


def _rasterize_view(
    batches: List[VisualBatch],
    *,
    azimuth_deg: float,
    elevation_deg: float,
    size: int,
    fov_deg: float = DEFAULT_FOV_DEG,
    camera_center: Sequence[float] | np.ndarray | None = None,
    camera_radius: float | None = None,
) -> Image.Image:
    if (camera_center is None) != (camera_radius is None):
        raise ValueError("camera_center and camera_radius must be supplied together")
    if camera_center is None:
        center, radius = bbox_center_radius(batches)
    else:
        center = np.asarray(camera_center, dtype=np.float64)
        radius = float(cast(float, camera_radius))
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("camera_center must be a finite xyz vector")
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("camera_radius must be finite and positive")
    view, focal = _view_and_focal(
        center,
        radius,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        size=size,
        fov_deg=fov_deg,
    )

    framebuffer = np.ones((size, size, 3), dtype=np.float64)
    zbuffer = np.full((size, size), -np.inf)

    # Each record: dict with tri (M,3,3 cam-space), alpha, and either flat
    # `colors` (M,3) or textured `tri_uvs` (M,3,2) + `texture` + `intensity` (M,).
    opaque: List[dict] = []
    transparent: List[dict] = []
    for batch in batches:
        cam = batch.vertices @ view[:3, :3].T + view[:3, 3]
        tri = cam[batch.faces]  # (M, 3, 3)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = lengths[:, 0] > 1e-12
        unit_normals = np.divide(normals, np.where(lengths > 1e-12, lengths, 1.0))
        if batch.uvs is not None and batch.texture is not None:
            record = {
                "tri": tri[valid],
                "alpha": 1.0,
                "tri_uvs": batch.uvs[batch.faces][valid],
                "texture": batch.texture,
                "intensity": _shade_intensity(unit_normals)[valid],
            }
            opaque.append(record)  # MASK alpha → resolved per texel
        else:
            colors = _shade(unit_normals, np.array(batch.rgba[:3]))
            record = {"tri": tri[valid], "alpha": batch.rgba[3], "colors": colors[valid]}
            (opaque if batch.rgba[3] >= 0.999 else transparent).append(record)

    def project(tri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = tri[:, :, 2]
        z = np.where(z > -1e-6, -1e-6, z)
        px = size / 2.0 + focal * (tri[:, :, 0] / -z)
        py = size / 2.0 - focal * (tri[:, :, 1] / -z)
        return np.stack([px, py], axis=-1), z

    def _sample_texture(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        th, tw = texture.shape[:2]
        # glTF UV origin is top-left; wrap into [0,1).
        tx = np.clip((np.mod(u, 1.0) * tw).astype(int), 0, tw - 1)
        ty = np.clip((np.mod(v, 1.0) * th).astype(int), 0, th - 1)
        return texture[ty, tx]  # (K, 4) uint8

    def fill(record: dict, write_z: bool) -> None:
        tri_screen, tri_z = project(record["tri"])
        alpha = record["alpha"]
        colors: Any = record.get("colors")
        texture = record.get("texture")
        intensity: Any = record.get("intensity")
        for idx in range(tri_screen.shape[0]):
            pts = tri_screen[idx]
            zs = tri_z[idx]
            min_x = max(int(np.floor(pts[:, 0].min())), 0)
            max_x = min(int(np.ceil(pts[:, 0].max())), size - 1)
            min_y = max(int(np.floor(pts[:, 1].min())), 0)
            max_y = min(int(np.ceil(pts[:, 1].max())), size - 1)
            if min_x > max_x or min_y > max_y:
                continue
            xs = np.arange(min_x, max_x + 1) + 0.5
            ys = np.arange(min_y, max_y + 1) + 0.5
            grid_x, grid_y = np.meshgrid(xs, ys)
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x2, y2 = pts[2]
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-12:
                continue
            w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
            w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            depth = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
            rows = grid_y.astype(int)[inside]
            cols = grid_x.astype(int)[inside]
            depths = depth[inside]
            visible = depths > zbuffer[rows, cols]
            if not visible.any():
                continue
            rows, cols, depths = rows[visible], cols[visible], depths[visible]

            if texture is not None:
                bw0, bw1, bw2 = w0[inside][visible], w1[inside][visible], w2[inside][visible]
                uv = record["tri_uvs"][idx]
                u = bw0 * uv[0, 0] + bw1 * uv[1, 0] + bw2 * uv[2, 0]
                v = bw0 * uv[0, 1] + bw1 * uv[1, 1] + bw2 * uv[2, 1]
                texels = _sample_texture(texture, u, v).astype(np.float64) / 255.0
                opaque_texel = texels[:, 3] >= 0.5  # MASK alpha cutoff
                if not opaque_texel.any():
                    continue
                rows, cols, depths = rows[opaque_texel], cols[opaque_texel], depths[opaque_texel]
                rgb = texels[opaque_texel, :3] * intensity[idx]
                framebuffer[rows, cols] = np.clip(rgb, 0.0, 1.0)
                if write_z:
                    zbuffer[rows, cols] = depths
                continue

            color = colors[idx]
            if alpha >= 0.999:
                framebuffer[rows, cols] = color
                if write_z:
                    zbuffer[rows, cols] = depths
            else:
                framebuffer[rows, cols] = (
                    framebuffer[rows, cols] * (1.0 - alpha) + color[None, :] * alpha
                )

    for record in opaque:
        fill(record, write_z=True)
    # Transparent pass: far-to-near, blended in front of opaque surfaces only.
    transparent_tris = []
    for record in transparent:
        mean_z = record["tri"][:, :, 2].mean(axis=1)
        for idx in np.argsort(mean_z):
            transparent_tris.append(
                (
                    {
                        "tri": record["tri"][idx : idx + 1],
                        "alpha": record["alpha"],
                        "colors": record["colors"][idx : idx + 1],
                    },
                    mean_z[idx],
                )
            )
    transparent_tris.sort(key=lambda item: item[1])
    for record, _ in transparent_tris:
        fill(record, write_z=False)

    return Image.fromarray((framebuffer * 255).astype(np.uint8))


def render_urdf_views(
    urdf_dir: Path,
    out_dir: Path,
    *,
    views: Sequence[Tuple[str, float, float]] = DEFAULT_VIEWS,
    size: int = 640,
    joint_positions: Mapping[str, float] | None = None,
    camera_center: Sequence[float] | np.ndarray | None = None,
    camera_radius: float | None = None,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> List[Path]:
    """Render ``urdf_dir/model.urdf`` to one PNG per view; returns the paths."""
    batches = load_urdf_visuals(urdf_dir, joint_positions=joint_positions)
    if not batches:
        raise ValueError(f"URDF in {urdf_dir} has no visual geometry")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for name, azimuth, elevation in views:
        image = _rasterize_view(
            batches,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            size=size,
            fov_deg=fov_deg,
            camera_center=camera_center,
            camera_radius=camera_radius,
        )
        path = out_dir / f"{name}.png"
        image.save(path)
        outputs.append(path)
    return outputs
