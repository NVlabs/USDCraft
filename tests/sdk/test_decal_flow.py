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

import json
import struct
import tempfile
from pathlib import Path

from sdk import (
    ArticulatedObject,
    Box,
    CylinderGeometry,
    Inertial,
    TestContext,
    TextureCanvas,
    mesh_from_geometry,
)
from sdk._core.assets import AssetSession, activate_asset_session
from sdk._core.exact_collisions import compile_object_model_with_exact_collisions


def _decal_glb_material(tmp_path: Path, canvas: TextureCanvas) -> dict[str, object]:
    with activate_asset_session(AssetSession(tmp_path)):
        model = ArticulatedObject(name="decal_alpha_mode")
        decal = model.part("body").decal(canvas, width=0.1, height=0.1)

    glb = Path(decal.geometry.materialized_path).read_bytes()
    assert glb[:4] == b"glTF"
    json_length = struct.unpack("<I", glb[12:16])[0]
    document = json.loads(glb[20 : 20 + json_length])
    return document["materials"][0]


def _labeled_bottle():
    model = ArticulatedObject(name="labeled_bottle")
    model.meta["mode"] = "static"
    body_mat = model.material("red", rgba=(0.85, 0.1, 0.1, 1.0), roughness=0.5)
    body = model.part("body")
    body.visual(
        mesh_from_geometry(CylinderGeometry(0.04, 0.18, radial_segments=32), "body"),
        material=body_mat,
    )
    body.inertial = Inertial.from_geometry(Box((0.08, 0.08, 0.18)), mass=0.4)
    label = TextureCanvas(128, 160, background=(0, 0, 0, 0))
    label.rect((0.05, 0.1, 0.9, 0.5), fill=(0, 0, 0.6, 1))
    label.text((0.5, 0.3), "FIRE", size=0.2, fill=(1, 1, 1, 1), anchor="mm", bold=True)
    decal = body.decal(label, width=0.12, height=0.12, curvature_radius=0.04)
    return model, body, decal


def test_decal_is_visual_only_and_references_glb() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with activate_asset_session(AssetSession(Path(tmp))):
            model, body, decal = _labeled_bottle()

            assert decal.collidable is False
            assert decal.geometry.filename.startswith("assets/meshes/")
            assert decal.geometry.filename.endswith(".glb")

            # The decal GLB asset was written.
            glbs = list(Path(tmp).rglob("*.glb"))
            assert len(glbs) == 1

            ctx = TestContext(model)
            ctx.check_model_valid()
            ctx.check_mesh_assets_ready()


def test_decal_excluded_from_collision_derivation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with activate_asset_session(AssetSession(Path(tmp))):
            model, _, _ = _labeled_bottle()
            compiled = compile_object_model_with_exact_collisions(model)

            body = next(p for p in compiled.parts if p.name == "body")
            assert len(body.visuals) == 2  # cylinder + decal
            assert len(body.collisions) == 1  # only the cylinder; decal excluded


def test_decal_glb_is_content_addressed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with activate_asset_session(AssetSession(Path(tmp))):
            model = ArticulatedObject(name="x")
            part = model.part("p")
            c1 = TextureCanvas(64, 64, background=(1, 0, 0, 1))
            c2 = TextureCanvas(64, 64, background=(1, 0, 0, 1))
            d1 = part.decal(c1, width=0.1, height=0.1)
            d2 = part.decal(c2, width=0.1, height=0.1)
            # Identical canvases dedupe to the same asset.
            assert d1.geometry.filename == d2.geometry.filename


def test_decal_glb_is_opaque_when_canvas_alpha_is_fully_opaque(tmp_path: Path) -> None:
    canvas = TextureCanvas(32, 32, background=(0.8, 0.2, 0.1, 1.0))

    material = _decal_glb_material(tmp_path, canvas)

    assert material["alphaMode"] == "OPAQUE"
    assert "alphaCutoff" not in material


def test_decal_glb_is_masked_when_canvas_contains_transparency(tmp_path: Path) -> None:
    canvas = TextureCanvas(32, 32, background=(0.0, 0.0, 0.0, 0.0))
    canvas.rect((0.2, 0.2, 0.6, 0.6), fill=(0.8, 0.2, 0.1, 1.0))

    material = _decal_glb_material(tmp_path, canvas)

    assert material["alphaMode"] == "MASK"
    assert material["alphaCutoff"] == 0.5
