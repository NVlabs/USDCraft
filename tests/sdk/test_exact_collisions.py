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

from pathlib import Path

import pytest

import sdk._core.exact_collisions as ec
from sdk import ArticulatedObject, AssetContext, Box, Material, Mesh, Origin, TextureInput
from sdk._core.mesh import (
    BoxGeometry,
    mesh_from_geometry,
)
from sdk._core.types import Collision


def _write_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_model_with_mesh(
    tmp_path: Path, mesh: Mesh, *, origin: Origin | None = None
) -> ArticulatedObject:
    assets = AssetContext(tmp_path)
    model = ArticulatedObject(name="exact_collision_model", assets=assets)
    part = model.part("body")
    part.visual(mesh, origin=origin or Origin(), name="body_visual")
    return model


def test_compile_exact_collisions_reuses_raw_visual_mesh(tmp_path: Path) -> None:
    mesh_path = tmp_path / "assets" / "meshes" / "raw.obj"
    _write_obj(mesh_path)
    mesh = Mesh(filename="assets/meshes/raw.obj")
    visual_origin = Origin(xyz=(0.1, 0.2, 0.3), rpy=(0.0, 0.1, 0.2))

    compiled = ec.compile_object_model_with_exact_collisions(
        _build_model_with_mesh(tmp_path, mesh, origin=visual_origin),
        asset_root=tmp_path,
    )

    collisions = compiled.parts[0].collisions
    assert len(collisions) == 1
    assert collisions[0].geometry == mesh
    assert collisions[0].origin == visual_origin
    assert collisions[0].name == "body_visual"
    assert not (tmp_path / "assets" / "meshes" / "collision" / "cache").exists()


def test_compile_exact_collisions_mirrors_primitive_backed_mesh_without_rewrite(
    tmp_path: Path,
) -> None:
    mesh = mesh_from_geometry(
        BoxGeometry((0.10, 0.20, 0.30)),
        tmp_path / "assets" / "meshes" / "box.obj",
    )

    compiled = ec.compile_object_model_with_exact_collisions(
        _build_model_with_mesh(tmp_path, mesh),
        asset_root=tmp_path,
    )

    collisions = compiled.parts[0].collisions
    assert len(collisions) == 1
    assert collisions[0].geometry == mesh
    assert collisions[0].origin == Origin()
    assert collisions[0].name == "body_visual"


def test_compile_exact_collisions_mirrors_legacy_mesh_prefix_without_rewrite(
    tmp_path: Path,
) -> None:
    mesh = mesh_from_geometry(
        BoxGeometry((0.10, 0.20, 0.30)),
        tmp_path / "assets" / "meshes" / "box.obj",
    )
    legacy_mesh = Mesh(
        filename="meshes/box.obj",
        source_geometry=mesh.source_geometry,
        source_transform=mesh.source_transform,
    )

    compiled = ec.compile_object_model_with_exact_collisions(
        _build_model_with_mesh(tmp_path, legacy_mesh),
        asset_root=tmp_path,
    )

    collisions = compiled.parts[0].collisions
    assert len(collisions) == 1
    assert collisions[0].geometry == legacy_mesh
    assert collisions[0].origin == Origin()
    assert collisions[0].name == "body_visual"


def test_compile_exact_collisions_rejects_explicit_part_collisions() -> None:
    model = ArticulatedObject(name="explicit_part_collisions")
    part = model.part("body")
    part.visual(Box((0.1, 0.1, 0.1)))
    part.collisions.append(Collision(geometry=Box((0.1, 0.1, 0.1))))

    with pytest.raises(ec.ValidationError, match="Part\\.collisions"):
        ec.compile_object_model_with_exact_collisions(model)


def test_compile_exact_collisions_cache_tracks_visual_collidable_flag() -> None:
    model = ArticulatedObject(name="collidable_cache_key")
    part = model.part("body")
    visual = part.visual(Box((0.1, 0.1, 0.1)), name="optional_surface")

    collidable = ec.compile_object_model_with_exact_collisions(model)
    assert [collision.name for collision in collidable.parts[0].collisions] == ["optional_surface"]

    visual.collidable = False
    visual_only = ec.compile_object_model_with_exact_collisions(model)

    assert visual_only is not collidable
    assert visual_only.parts[0].collisions == []


def test_compile_exact_collisions_cache_tracks_registered_material_pbr() -> None:
    model = ArticulatedObject(name="material_cache_key")
    material = model.material(
        "finish",
        rgba=(0.4, 0.3, 0.2, 1.0),
        roughness=0.3,
        base_color_texture=TextureInput("first.png", source_color_space="sRGB"),
    )
    model.part("body").visual(Box((0.1, 0.1, 0.1)), material=material)

    first = ec.compile_object_model_with_exact_collisions(model)

    material.roughness = 0.7
    second = ec.compile_object_model_with_exact_collisions(model)
    assert second is not first
    assert second.materials[0].roughness == pytest.approx(0.7)

    material.base_color_texture = TextureInput("second.png", source_color_space="sRGB")
    third = ec.compile_object_model_with_exact_collisions(model)
    assert third is not second
    assert third.materials[0].base_color_texture is not None
    assert third.materials[0].base_color_texture.file == "second.png"


def test_compile_exact_collisions_cache_tracks_inline_material_pbr() -> None:
    model = ArticulatedObject(name="inline_material_cache_key")
    material = Material("inline", rgba=(0.4, 0.3, 0.2, 1.0), clearcoat=0.2)
    model.part("body").visual(Box((0.1, 0.1, 0.1)), material=material)

    first = ec.compile_object_model_with_exact_collisions(model)
    material.clearcoat = 0.6
    second = ec.compile_object_model_with_exact_collisions(model)

    assert second is not first
    compiled_material = second.parts[0].visuals[0].material
    assert isinstance(compiled_material, Material)
    assert compiled_material.clearcoat == pytest.approx(0.6)


_NAME_ONLY_HINT = "has no filename; write the mesh with mesh_from_geometry(...) or pass filename="


def test_require_mesh_filename_returns_text_or_explains_name_only_mesh() -> None:
    from sdk._core.assets import require_mesh_filename

    assert require_mesh_filename(Mesh(filename=Path("assets/meshes/a.obj"))) == (
        "assets/meshes/a.obj"
    )
    with pytest.raises(ec.ValidationError) as excinfo:
        require_mesh_filename(Mesh(name="handle"), owner="part 'door'")
    assert str(excinfo.value) == f"Mesh 'handle' on part 'door' {_NAME_ONLY_HINT}"


def test_compile_exact_collisions_rejects_name_only_mesh_with_part_context(
    tmp_path: Path,
) -> None:
    model = _build_model_with_mesh(tmp_path, Mesh(name="handle"))

    with pytest.raises(ec.ValidationError) as excinfo:
        ec.compile_object_model_with_exact_collisions(model, asset_root=tmp_path)
    assert str(excinfo.value) == (
        f"Mesh 'handle' on part 'body' visual 'body_visual' {_NAME_ONLY_HINT}"
    )


def test_urdf_and_usd_export_reject_name_only_mesh_clearly(tmp_path: Path) -> None:
    from sdk._core._urdf_export import compile_object_to_urdf_xml
    from sdk._core._usd_export import export_object_to_usd

    model = _build_model_with_mesh(tmp_path, Mesh(name="handle"))

    expected = f"Mesh 'handle' {_NAME_ONLY_HINT}"
    with pytest.raises(ec.ValidationError) as urdf_error:
        compile_object_to_urdf_xml(model, asset_root=tmp_path, include_physical_collisions=False)
    assert str(urdf_error.value) == expected
    with pytest.raises(ec.ValidationError) as usd_error:
        export_object_to_usd(model, tmp_path / "usd", include_physical_collisions=False)
    assert str(usd_error.value) == expected
