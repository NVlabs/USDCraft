# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

import json
import math
import struct
import xml.etree.ElementTree as ET
from contextlib import suppress
from pathlib import Path

import numpy as np

from ._mesh.normals import crease_aware_face_varying_normals
from .assets import resolve_mesh_path


def _pad4(data: bytes, *, fill: bytes) -> bytes:
    padding = (-len(data)) % 4
    if padding == 0:
        return data
    return data + (fill * padding)


def _obj_vertex_index(token: str, vertex_count: int) -> int:
    raw = token.split("/", 1)[0].strip()
    if raw == "":
        raise ValueError("OBJ face token is missing a vertex index")
    index = int(raw)
    if index > 0:
        return index - 1
    if index < 0:
        return vertex_count + index
    raise ValueError("OBJ face indices are 1-based and may not be zero")


def _obj_normal_index(token: str, normal_count: int) -> int:
    raw = token.strip()
    if raw == "":
        raise ValueError("OBJ face token is missing a normal index")
    index = int(raw)
    if index > 0:
        return index - 1
    if index < 0:
        return normal_count + index
    raise ValueError("OBJ normal indices are 1-based and may not be zero")


def load_obj_triangles(
    obj_path: Path,
    *,
    normal_crease_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    face_normal_indices: list[tuple[int, int, int]] = []
    metadata_normal_crease_deg: float | None = None

    if normal_crease_deg is not None:
        normal_crease_deg = float(normal_crease_deg)
        if not math.isfinite(normal_crease_deg) or not 0.0 <= normal_crease_deg <= 180.0:
            raise ValueError("normal_crease_deg must be finite and between 0 and 180")

    for raw_line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("# usdcraft_normal_crease_deg "):
            try:
                candidate = float(line.removeprefix("# usdcraft_normal_crease_deg ").strip())
            except ValueError:
                candidate = math.nan
            if math.isfinite(candidate) and 0.0 <= candidate <= 180.0:
                metadata_normal_crease_deg = candidate
            continue
        if not line or line.startswith("#"):
            continue

        if line.startswith("v "):
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            continue

        if line.startswith("vn "):
            parts = line.split()
            if len(parts) < 4:
                continue
            normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            continue

        if line.startswith("f "):
            tokens = line.split()[1:]
            if len(tokens) < 3:
                continue
            corner_indices: list[int] = []
            corner_normal_indices: list[int | None] = []
            for token in tokens:
                parts = token.split("/")
                corner_indices.append(_obj_vertex_index(token, len(vertices)))
                if len(parts) >= 3 and parts[2].strip() != "":
                    corner_normal_indices.append(_obj_normal_index(parts[2], len(normals)))
                else:
                    corner_normal_indices.append(None)
            for i in range(1, len(corner_indices) - 1):
                faces.append((corner_indices[0], corner_indices[i], corner_indices[i + 1]))
                normal_a = corner_normal_indices[0]
                normal_b = corner_normal_indices[i]
                normal_c = corner_normal_indices[i + 1]
                if normal_a is not None and normal_b is not None and normal_c is not None:
                    face_normal_indices.append((int(normal_a), int(normal_b), int(normal_c)))

    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {obj_path}")
    if not faces:
        raise ValueError(f"OBJ contains no faces: {obj_path}")

    vertex_array = np.asarray(vertices, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.uint32)
    normal_array = np.asarray(normals, dtype=np.float32) if normals else None
    normal_index_array = (
        np.asarray(face_normal_indices, dtype=np.uint32)
        if len(face_normal_indices) == len(faces)
        else None
    )
    effective_crease = (
        normal_crease_deg if normal_crease_deg is not None else metadata_normal_crease_deg
    )
    if effective_crease is not None:
        normal_array = crease_aware_face_varying_normals(
            vertex_array,
            face_array,
            crease_angle_degrees=effective_crease,
        ).astype(np.float32)
        normal_index_array = np.arange(len(faces) * 3, dtype=np.uint32).reshape((-1, 3))
    return vertex_array, face_array, normal_array, normal_index_array


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri_vertices = vertices[faces]
    normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > 1e-12
    normalized = np.zeros_like(normals, dtype=np.float32)
    normalized[nonzero] = (normals[nonzero] / lengths[nonzero][:, None]).astype(np.float32)
    normalized[~nonzero] = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    return normalized.astype(np.float32, copy=False)


def _compute_corner_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    crease_angle_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    face_normals = _compute_face_normals(vertices, faces)
    incident_faces: list[list[int]] = [[] for _ in range(int(vertices.shape[0]))]
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            incident_faces[int(vertex_index)].append(face_index)

    crease = float(crease_angle_deg)
    if not math.isfinite(crease) or crease < 0.0 or crease > 180.0:
        raise ValueError("crease_angle_deg must be finite and between 0 and 180")
    cosine_threshold = math.cos(math.radians(crease))
    normal_lookup: dict[tuple[float, ...], int] = {}
    normal_values: list[tuple[float, float, float]] = []
    face_normal_indices = np.zeros_like(faces, dtype=np.uint32)

    for face_index, face in enumerate(faces):
        current = face_normals[face_index]
        for corner_index, vertex_index in enumerate(face):
            accum = np.zeros(3, dtype=np.float64)
            for neighbor_face_index in incident_faces[int(vertex_index)]:
                candidate = face_normals[neighbor_face_index]
                if float(np.dot(current, candidate)) < cosine_threshold:
                    continue
                accum += candidate
            accum_len = float(np.linalg.norm(accum))
            if accum_len <= 1e-12:
                normal = current
            else:
                normal = np.asarray(accum / accum_len, dtype=np.float32)
            key = tuple(round(float(component), 6) for component in normal)
            normal_index = normal_lookup.get(key)
            if normal_index is None:
                normal_index = len(normal_values)
                normal_lookup[key] = normal_index
                normal_values.append((float(normal[0]), float(normal[1]), float(normal[2])))
            face_normal_indices[face_index, corner_index] = normal_index

    return np.asarray(normal_values, dtype=np.float32), face_normal_indices


def _expand_indexed_attributes(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    face_normal_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expanded_vertices: list[tuple[float, float, float]] = []
    expanded_normals: list[tuple[float, float, float]] = []
    expanded_indices: list[int] = []
    corner_lookup: dict[tuple[int, int], int] = {}

    for face, normal_face in zip(faces, face_normal_indices, strict=True):
        for vertex_index, normal_index in zip(face, normal_face, strict=True):
            key = (int(vertex_index), int(normal_index))
            expanded_index = corner_lookup.get(key)
            if expanded_index is None:
                expanded_index = len(expanded_vertices)
                corner_lookup[key] = expanded_index
                vx, vy, vz = vertices[int(vertex_index)]
                nx, ny, nz = normals[int(normal_index)]
                expanded_vertices.append((float(vx), float(vy), float(vz)))
                expanded_normals.append((float(nx), float(ny), float(nz)))
            expanded_indices.append(expanded_index)

    return (
        np.asarray(expanded_vertices, dtype=np.float32),
        np.asarray(expanded_normals, dtype=np.float32),
        np.asarray(expanded_indices, dtype=np.uint32),
    )


def build_glb_bytes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vertex_normals: np.ndarray | None = None,
    face_normal_indices: np.ndarray | None = None,
) -> bytes:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Faces must have shape (M, 3)")

    if (
        vertex_normals is None
        or vertex_normals.ndim != 2
        or vertex_normals.shape[1] != 3
        or face_normal_indices is None
        or face_normal_indices.shape != faces.shape
    ):
        vertex_normals, face_normal_indices = _compute_corner_normals(vertices, faces)

    vertices, normals, indices = _expand_indexed_attributes(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
        np.ascontiguousarray(vertex_normals, dtype=np.float32),
        np.ascontiguousarray(face_normal_indices, dtype=np.uint32),
    )

    positions_blob = vertices.tobytes()
    normals_blob = normals.tobytes()
    indices_blob = indices.tobytes()

    pos_offset = 0
    norm_offset = len(positions_blob)
    index_offset = norm_offset + len(normals_blob)
    binary_blob = positions_blob + normals_blob + indices_blob
    binary_blob = _pad4(binary_blob, fill=b"\x00")

    mins = vertices.min(axis=0).tolist()
    maxs = vertices.max(axis=0).tolist()
    index_min = int(indices.min())
    index_max = int(indices.max())

    gltf = {
        "asset": {"version": "2.0", "generator": "synth-urdf-agent"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "mode": 4,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(binary_blob)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": len(positions_blob),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": norm_offset,
                "byteLength": len(normals_blob),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": len(indices_blob),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(vertices.shape[0]),
                "type": "VEC3",
                "min": mins,
                "max": maxs,
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": int(normals.shape[0]),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": int(indices.shape[0]),
                "type": "SCALAR",
                "min": [index_min],
                "max": [index_max],
            },
        ],
    }

    json_blob = json.dumps(gltf, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    json_blob = _pad4(json_blob, fill=b" ")

    total_length = 12 + 8 + len(json_blob) + 8 + len(binary_blob)
    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_chunk = struct.pack("<I4s", len(json_blob), b"JSON") + json_blob
    bin_chunk = struct.pack("<I4s", len(binary_blob), b"BIN\x00") + binary_blob
    return header + json_chunk + bin_chunk


def _per_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_n)
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > 1e-12
    normals[nonzero] /= lengths[nonzero][:, None]
    normals[~nonzero] = (0.0, 0.0, 1.0)
    return normals.astype(np.float32)


def build_textured_glb_bytes(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_png: bytes,
    *,
    alpha_mode: str = "MASK",
    alpha_cutoff: float = 0.5,
    metallic: float = 0.0,
    roughness: float = 0.85,
) -> bytes:
    """Build a GLB for a textured mesh: per-vertex POSITION/NORMAL/TEXCOORD_0
    plus an embedded PNG bound as the material's baseColorTexture.

    Used for decals (clean per-vertex UVs); does not share the corner-normal
    re-indexing path of ``build_glb_bytes``.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    uvs = np.ascontiguousarray(uvs, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Faces must have shape (M, 3)")
    if uvs.shape != (vertices.shape[0], 2):
        raise ValueError("uvs must have shape (N, 2) aligned to vertices")
    if alpha_mode not in ("OPAQUE", "MASK", "BLEND"):
        raise ValueError("alpha_mode must be OPAQUE, MASK, or BLEND")

    normals = _per_vertex_normals(vertices, faces)
    indices = faces.reshape(-1).astype(np.uint32)

    blobs = [vertices.tobytes(), normals.tobytes(), uvs.tobytes(), indices.tobytes(), texture_png]
    offsets = []
    binary = b""
    for blob in blobs:
        binary = _pad4(binary, fill=b"\x00")
        offsets.append(len(binary))
        binary += blob
    binary = _pad4(binary, fill=b"\x00")
    pos_off, norm_off, uv_off, idx_off, img_off = offsets

    material = {
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": float(metallic),
            "roughnessFactor": float(roughness),
        },
        "alphaMode": alpha_mode,
        "doubleSided": True,
    }
    if alpha_mode == "MASK":
        material["alphaCutoff"] = float(alpha_cutoff)

    gltf = {
        "asset": {"version": "2.0", "generator": "synth-urdf-agent"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "materials": [material],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "samplers": [{}],
        "textures": [{"source": 0, "sampler": 0}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(blobs[0]), "target": 34962},
            {"buffer": 0, "byteOffset": norm_off, "byteLength": len(blobs[1]), "target": 34962},
            {"buffer": 0, "byteOffset": uv_off, "byteLength": len(blobs[2]), "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(blobs[3]), "target": 34963},
            {"buffer": 0, "byteOffset": img_off, "byteLength": len(blobs[4])},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(vertices.shape[0]),
                "type": "VEC3",
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": int(normals.shape[0]),
                "type": "VEC3",
            },
            {"bufferView": 2, "componentType": 5126, "count": int(uvs.shape[0]), "type": "VEC2"},
            {
                "bufferView": 3,
                "componentType": 5125,
                "count": int(indices.shape[0]),
                "type": "SCALAR",
                "min": [int(indices.min())],
                "max": [int(indices.max())],
            },
        ],
    }

    json_blob = json.dumps(gltf, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    json_blob = _pad4(json_blob, fill=b" ")
    total_length = 12 + 8 + len(json_blob) + 8 + len(binary)
    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_chunk = struct.pack("<I4s", len(json_blob), b"JSON") + json_blob
    bin_chunk = struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    return header + json_chunk + bin_chunk


def convert_obj_file_to_glb(
    obj_path: Path,
    glb_path: Path | None = None,
    *,
    normal_crease_deg: float | None = None,
) -> Path:
    obj_path = Path(obj_path)
    out_path = Path(glb_path) if glb_path is not None else obj_path.with_suffix(".glb")

    vertices, faces, normals, face_normal_indices = load_obj_triangles(
        obj_path,
        normal_crease_deg=normal_crease_deg,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(
        build_glb_bytes(
            vertices,
            faces,
            vertex_normals=normals,
            face_normal_indices=face_normal_indices,
        )
    )
    return out_path


def convert_urdf_visual_meshes_to_glb(
    urdf_xml: str,
    *,
    asset_root: Path,
) -> tuple[str, list[str]]:
    root = ET.fromstring(urdf_xml)
    warnings: list[str] = []
    for mesh_el in root.findall(".//visual/geometry/mesh"):
        filename = mesh_el.attrib.get("filename")
        if not isinstance(filename, str) or filename.strip() == "":
            continue
        if not filename.lower().endswith(".obj"):
            continue

        mesh_path = resolve_mesh_path(filename, assets=asset_root)

        raw_crease = mesh_el.attrib.get("usdcraft_normal_crease_deg")
        crease: float | None = None
        if raw_crease is not None:
            try:
                crease = float(raw_crease)
            except ValueError:
                crease = math.nan
            if not math.isfinite(crease) or not 0.0 <= crease <= 180.0:
                warnings.append(
                    f"Invalid usdcraft_normal_crease_deg on visual mesh: {raw_crease!r}"
                )
                continue

        if crease is None:
            glb_path = mesh_path.with_suffix(".glb")
            converted_filename = filename[:-4] + ".glb"
        else:
            crease_token = f"{crease:.12g}".replace(".", "_")
            variant_suffix = f".usdcraft_normals_{crease_token}.glb"
            glb_path = mesh_path.with_name(mesh_path.stem + variant_suffix)
            converted_filename = filename[:-4] + variant_suffix
        try:
            needs_refresh = (not glb_path.exists()) or (
                mesh_path.stat().st_mtime_ns > glb_path.stat().st_mtime_ns
            )
            if needs_refresh:
                convert_obj_file_to_glb(
                    mesh_path,
                    glb_path,
                    normal_crease_deg=crease,
                )
        except Exception as exc:
            warnings.append(f"Failed to convert mesh to GLB: {mesh_path} ({exc})")
            continue

        mesh_el.set("filename", converted_filename)

    with suppress(Exception):
        ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode"), warnings
