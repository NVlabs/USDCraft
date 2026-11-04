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

import io
import json
import struct

import numpy as np
import pytest

from sdk import DecalGeometry, TextureCanvas
from sdk._core.viewer_assets import build_textured_glb_bytes


def _decal_arrays():
    canvas = TextureCanvas(32, 32, background=(0.0, 0.0, 1.0, 1.0))
    decal = DecalGeometry(canvas, width=0.2, height=0.3)
    vertices = np.array(decal.vertices, dtype=np.float32)
    faces = np.array(decal.faces, dtype=np.uint32)
    uvs = np.array(decal.uvs, dtype=np.float32)
    png = io.BytesIO()
    canvas.render().save(png, format="PNG")
    return vertices, faces, uvs, png.getvalue()


def _parse_gltf_json(glb: bytes) -> dict:
    assert glb[:4] == b"glTF"
    json_len = struct.unpack("<I", glb[12:16])[0]
    return json.loads(glb[20 : 20 + json_len])


def test_textured_glb_has_uv_texture_and_material() -> None:
    vertices, faces, uvs, png = _decal_arrays()
    glb = build_textured_glb_bytes(vertices, faces, uvs, png, alpha_mode="MASK")

    doc = _parse_gltf_json(glb)
    prim = doc["meshes"][0]["primitives"][0]
    assert "TEXCOORD_0" in prim["attributes"]
    assert prim["material"] == 0
    assert doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert doc["materials"][0]["alphaMode"] == "MASK"
    assert doc["images"][0]["mimeType"] == "image/png"
    assert len(doc["textures"]) == 1


def test_textured_glb_roundtrips_through_trimesh() -> None:
    import trimesh

    vertices, faces, uvs, png = _decal_arrays()
    glb = build_textured_glb_bytes(vertices, faces, uvs, png)

    loaded = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
    assert len(loaded.vertices) >= 4
    # UV channel survives the round-trip.
    assert loaded.visual.uv is not None
    assert len(loaded.visual.uv) == len(loaded.vertices)


def test_textured_glb_validates_uv_shape() -> None:
    vertices, faces, _, png = _decal_arrays()
    with pytest.raises(ValueError, match="uvs must have shape"):
        build_textured_glb_bytes(vertices, faces, np.zeros((2, 2), dtype=np.float32), png)
