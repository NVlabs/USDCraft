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
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh

from sdk._core.mesh import BoxGeometry, mesh_from_geometry
from sdk._core.viewer_assets import convert_obj_file_to_glb, convert_urdf_visual_meshes_to_glb


def _glb_positions_and_normals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    json_length, json_kind = struct.unpack_from("<I4s", payload, 12)
    assert json_kind == b"JSON"
    json_start = 20
    gltf = json.loads(payload[json_start : json_start + json_length].decode("utf-8"))
    binary_header = json_start + json_length
    _binary_length, binary_kind = struct.unpack_from("<I4s", payload, binary_header)
    assert binary_kind == b"BIN\x00"
    binary_start = binary_header + 8

    arrays: list[np.ndarray] = []
    for accessor_index in (0, 1):
        accessor = gltf["accessors"][accessor_index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        offset = binary_start + int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        count = int(accessor["count"])
        arrays.append(
            np.frombuffer(payload, dtype="<f4", count=count * 3, offset=offset).reshape(-1, 3)
        )
    return arrays[0], arrays[1]


def test_convert_urdf_visual_meshes_to_glb_resolves_legacy_mesh_prefix(tmp_path) -> None:
    mesh_from_geometry(
        BoxGeometry((1.0, 1.0, 1.0)),
        tmp_path / "assets" / "meshes" / "part.obj",
    )

    urdf_xml = (
        "<robot name='legacy'>"
        "<link name='base'>"
        "<visual><geometry><mesh filename='meshes/part.obj'/></geometry></visual>"
        "</link>"
        "</robot>"
    )

    converted_xml, warnings = convert_urdf_visual_meshes_to_glb(
        urdf_xml,
        asset_root=tmp_path,
    )

    assert warnings == []
    assert "meshes/part.glb" in converted_xml
    assert (tmp_path / "assets" / "meshes" / "part.glb").exists()


def test_convert_obj_file_to_glb_preserves_triangle_count(tmp_path: Path) -> None:
    obj_path = tmp_path / "plate.obj"
    obj_path.write_text(
        "\n".join(
            [
                "o mesh",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
                "f 3 2 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    glb_path = convert_obj_file_to_glb(obj_path)
    glb_mesh = trimesh.load_mesh(glb_path, force="mesh")

    assert len(glb_mesh.faces) == 2


def test_urdf_obj_conversion_builds_per_crease_normal_variants(tmp_path: Path) -> None:
    obj_path = tmp_path / "assets" / "meshes" / "bent.obj"
    obj_path.parent.mkdir(parents=True)
    obj_path.write_text(
        "\n".join(
            [
                "o bent",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 2 1 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    urdf_xml = (
        "<robot name='creases'><link name='base'>"
        "<visual name='hard'><geometry><mesh filename='assets/meshes/bent.obj' "
        "usdcraft_normal_crease_deg='0'/></geometry></visual>"
        "<visual name='smooth'><geometry><mesh filename='assets/meshes/bent.obj' "
        "usdcraft_normal_crease_deg='180'/></geometry></visual>"
        "</link></robot>"
    )

    converted_xml, warnings = convert_urdf_visual_meshes_to_glb(
        urdf_xml,
        asset_root=tmp_path,
    )

    assert warnings == []
    root = ET.fromstring(converted_xml)
    filenames = [element.attrib["filename"] for element in root.findall(".//visual/geometry/mesh")]
    assert filenames == [
        "assets/meshes/bent.usdcraft_normals_0.glb",
        "assets/meshes/bent.usdcraft_normals_180.glb",
    ]

    corner_normals: list[np.ndarray] = []
    for filename in filenames:
        positions, normals = _glb_positions_and_normals(tmp_path / filename)
        center_indices = np.flatnonzero(np.linalg.norm(positions, axis=1) <= 1e-12)
        assert len(center_indices) == 2
        corner_normals.append(np.asarray(normals[center_indices], dtype=float))

    assert float(np.dot(corner_normals[0][0], corner_normals[0][1])) == pytest.approx(0.0, abs=1e-6)
    assert float(np.dot(corner_normals[1][0], corner_normals[1][1])) == pytest.approx(1.0, abs=1e-6)
