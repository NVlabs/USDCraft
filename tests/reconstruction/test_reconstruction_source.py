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
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from pxr import Gf, Usd, UsdGeom

from usdcraft.reconstruction.source import (
    SourceInput,
    SourceMetadataError,
    inspect_source,
    load_normalized_source,
    write_normalized_source,
)


def _define_mesh(stage: Usd.Stage, path: str, geometry: trimesh.Trimesh) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    faces = np.asarray(geometry.faces, dtype=np.int64)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, vertex)) for vertex in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(value) for value in faces.reshape(-1)])
    return mesh


def _write_usd(
    path: Path,
    geometry: trimesh.Trimesh,
    *,
    up_axis: str | None = "Z",
    meters_per_unit: float | None = 1.0,
    translation: tuple[float, float, float] | None = None,
) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    if up_axis is not None:
        UsdGeom.SetStageUpAxis(stage, up_axis)
    if meters_per_unit is not None:
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    if translation is None:
        _define_mesh(stage, "/Mesh", geometry)
    else:
        root = UsdGeom.Xform.Define(stage, "/Root")
        root.AddTranslateOp().Set(Gf.Vec3d(*translation))
        _define_mesh(stage, "/Root/Mesh", geometry)
    stage.GetRootLayer().Save()


def test_usd_schema_fallback_is_not_treated_as_authored_frame_metadata(tmp_path: Path) -> None:
    source = tmp_path / "missing_metadata.usda"
    _write_usd(source, trimesh.creation.box(), up_axis=None, meters_per_unit=None)

    inspection = inspect_source(SourceInput(source))

    assert not inspection.ready
    assert inspection.embedded_up_axis is None
    assert inspection.embedded_meters_per_unit is None
    assert inspection.required_inputs == ("up_axis", "meters_per_unit")
    with pytest.raises(SourceMetadataError, match="up_axis, meters_per_unit"):
        load_normalized_source(SourceInput(source))

    explicit = inspect_source(SourceInput(source, up_axis="+Z", meters_per_unit=1.0))
    assert explicit.ready
    assert explicit.resolved_frame is not None
    assert explicit.resolved_frame.up_axis_source == "explicit"
    assert explicit.resolved_frame.meters_per_unit_source == "explicit"


def test_y_up_centimetre_usd_is_baked_to_metres_without_recentering(tmp_path: Path) -> None:
    source = tmp_path / "centimetres.usda"
    box = trimesh.creation.box(extents=(20.0, 40.0, 60.0))
    _write_usd(
        source,
        box,
        up_axis="Y",
        meters_per_unit=0.01,
        translation=(100.0, 200.0, 300.0),
    )

    scene = load_normalized_source(SourceInput(source))

    # Y-up -> Z-up is Rx(+90): (x, y, z) -> (x, -z, y), after centimetres -> metres.
    assert scene.bounds_min_xyz == pytest.approx((0.9, -3.3, 1.8))
    assert scene.bounds_max_xyz == pytest.approx((1.1, -2.7, 2.2))
    assert scene.bounds_max_xyz - scene.bounds_min_xyz == pytest.approx((0.2, 0.6, 0.4))
    assert scene.components[0].topology.surface_closed
    assert np.asarray(scene.frame.source_to_normalized) == pytest.approx(
        np.asarray(
            (
                (0.01, 0.0, 0.0, 0.0),
                (0.0, 0.0, -0.01, 0.0),
                (0.0, 0.01, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
    )


def test_signed_up_and_front_axes_build_a_right_handed_canonical_frame(tmp_path: Path) -> None:
    source = tmp_path / "frame.obj"
    source.write_text("# frame metadata inspection only\n", encoding="utf-8")

    inspection = inspect_source(
        SourceInput(source, up_axis="-Z", front_axis="+Y", meters_per_unit=0.001)
    )

    assert inspection.ready
    assert inspection.resolved_frame is not None
    matrix = inspection.resolved_frame.matrix
    assert matrix[:3, :3] @ np.asarray((0.0, 0.0, -1.0)) == pytest.approx((0.0, 0.0, 0.001))
    assert matrix[:3, :3] @ np.asarray((0.0, 1.0, 0.0)) == pytest.approx((0.0, -0.001, 0.0))
    assert np.linalg.det(matrix[:3, :3] / 0.001) == pytest.approx(1.0)


@pytest.mark.parametrize("suffix", [".obj", ".stl", ".ply", ".off", ".xyz"])
def test_ambiguous_mesh_formats_require_explicit_axis_and_units(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / f"source{suffix}"
    source.write_bytes(b"placeholder")

    inspection = inspect_source(SourceInput(source))

    assert inspection.required_inputs == ("up_axis", "meters_per_unit")
    assert inspect_source(SourceInput(source, up_axis="Z", meters_per_unit=1.0)).ready


def test_glb_uses_format_frame_contract_and_bakes_y_up_to_z_up(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    geometry = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    source.write_bytes(trimesh.Scene(geometry).export(file_type="glb"))

    inspection = inspect_source(SourceInput(source))
    scene = load_normalized_source(SourceInput(source))

    assert inspection.ready
    assert inspection.resolved_frame is not None
    assert inspection.resolved_frame.up_axis == "+Y"
    assert inspection.resolved_frame.up_axis_source == "format_contract"
    assert inspection.resolved_frame.meters_per_unit_source == "format_contract"
    assert scene.bounds_max_xyz - scene.bounds_min_xyz == pytest.approx((1.0, 3.0, 2.0))


def test_obj_dependency_closure_includes_material_libraries_and_textures(
    tmp_path: Path,
) -> None:
    materials = tmp_path / "materials"
    textures = tmp_path / "textures"
    materials.mkdir()
    textures.mkdir()
    texture = textures / "base color.png"
    Image.new("RGB", (2, 2), (90, 120, 150)).save(texture)
    material = materials / "surface.mtl"
    material.write_text(
        'newmtl surface\nKd 1 1 1\nmap_Kd -o -1 0 -clamp on "../textures/base color.png"\n',
        encoding="utf-8",
    )
    source = tmp_path / "source.obj"
    source.write_text(
        "mtllib materials/surface.mtl\n"
        "usemtl surface\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "f 1/1 2/2 3/3\n",
        encoding="utf-8",
    )

    scene = load_normalized_source(
        SourceInput(source, up_axis="+Z", front_axis="-Y", meters_per_unit=1.0)
    )

    assert {item.path for item in scene.source_dependencies} == {
        source.resolve(),
        material.resolve(),
        texture.resolve(),
    }


def test_one_mesh_is_split_into_edge_connected_closed_and_open_components(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed.usda"
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    triangle = trimesh.Trimesh(
        vertices=((2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
        faces=((0, 1, 2),),
        process=False,
    )
    _write_usd(source, trimesh.util.concatenate((box, triangle)))

    scene = load_normalized_source(SourceInput(source))

    assert len(scene.components) == 2
    by_class = {component.topology.classification: component for component in scene.components}
    assert by_class["closed_orientable_surface"].topology.surface_closed
    open_topology = by_class["open_surface"].topology
    assert not open_topology.surface_closed
    assert open_topology.boundary_edge_count == 3


def test_stl_duplicate_seam_vertices_use_exact_weld_for_topology_only(tmp_path: Path) -> None:
    source = tmp_path / "box.stl"
    trimesh.creation.box().export(source)

    scene = load_normalized_source(SourceInput(source, up_axis="+Z", meters_per_unit=1.0))

    assert len(scene.components) == 1
    topology = scene.components[0].topology
    assert topology.exact_welded_vertex_count > 0
    assert topology.boundary_edge_count == 0
    assert topology.surface_closed


@pytest.mark.parametrize("suffix", [".obj", ".off"])
def test_trimesh_common_surface_formats_load_as_closed_components(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / f"box{suffix}"
    trimesh.creation.box().export(source)

    scene = load_normalized_source(SourceInput(source, up_axis="+Z", meters_per_unit=1.0))

    assert len(scene.components) == 1
    assert scene.components[0].kind == "mesh"
    assert scene.components[0].topology.surface_closed


def test_xyz_loads_as_point_samples(tmp_path: Path) -> None:
    source = tmp_path / "points.xyz"
    source.write_text("0 0 0\n1 0 0\n0 1 0\n", encoding="utf-8")

    scene = load_normalized_source(SourceInput(source, up_axis="+Z", meters_per_unit=1.0))

    assert len(scene.components) == 1
    assert scene.components[0].kind == "points"
    assert scene.components[0].topology.classification == "point_samples"


def test_ply_point_cloud_preserves_authored_normals_and_colors(tmp_path: Path) -> None:
    source = tmp_path / "points.ply"
    source.write_text(
        """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
0 0 0 0 0 1 255 0 0 255
1 0 0 0 0 1 0 255 0 255
0 1 0 0 0 1 0 0 255 128
""",
        encoding="utf-8",
    )

    scene = load_normalized_source(SourceInput(source, up_axis="+Z", meters_per_unit=1.0))

    assert len(scene.components) == 1
    component = scene.components[0]
    assert component.kind == "points"
    assert component.topology.classification == "point_samples"
    assert not component.topology.surface_closed
    assert component.normals == pytest.approx(np.asarray(((0.0, 0.0, 1.0),) * 3))
    assert component.colors_rgba is not None
    assert component.colors_rgba[:, :3] == pytest.approx(np.eye(3))
    assert component.colors_rgba[:, 3] == pytest.approx((1.0, 1.0, 128.0 / 255.0))

    artifacts = write_normalized_source(scene, tmp_path / "normalized-points")
    stage = Usd.Stage.Open(str(artifacts.normalized_usdc))
    assert stage is not None
    normalized_points = UsdGeom.Points.Get(stage, "/Source/Points/c000000")
    assert normalized_points
    assert len(normalized_points.GetPointsAttr().Get()) == 3


def test_usd_points_and_native_instance_proxies_are_expanded(tmp_path: Path) -> None:
    source = tmp_path / "instances.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    model = UsdGeom.Xform.Define(stage, "/Model")
    _define_mesh(stage, "/Model/Mesh", trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    model.GetPrim().SetInstanceable(True)
    copied = stage.OverridePrim("/Copy")
    copied.GetReferences().AddInternalReference("/Model")
    UsdGeom.Xformable(copied).AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))
    points = UsdGeom.Points.Define(stage, "/Samples")
    points.CreatePointsAttr(
        (Gf.Vec3f(0.0, 2.0, 0.0), Gf.Vec3f(1.0, 2.0, 0.0), Gf.Vec3f(0.0, 2.0, 1.0))
    )
    stage.GetRootLayer().Save()

    scene = load_normalized_source(SourceInput(source))

    assert sum(component.kind == "mesh" for component in scene.components) == 2
    assert sum(component.kind == "points" for component in scene.components) == 1
    assert scene.bounds_min_xyz == pytest.approx((-0.5, -0.5, -0.5))
    assert scene.bounds_max_xyz == pytest.approx((2.5, 2.0, 1.0))


def test_normalized_artifacts_are_deterministic_z_up_metric_usd_and_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_usd(source, trimesh.creation.box(extents=(1.0, 2.0, 3.0)))
    scene = load_normalized_source(SourceInput(source))

    first = write_normalized_source(scene, tmp_path / "first")
    second = write_normalized_source(scene, tmp_path / "second")

    assert first.normalized_usdc.read_bytes() == second.normalized_usdc.read_bytes()
    assert first.source_manifest.read_bytes() == second.source_manifest.read_bytes()
    stage = Usd.Stage.Open(str(first.normalized_usdc))
    assert stage is not None
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    assert stage.GetDefaultPrim().GetPath() == "/Source"
    manifest = json.loads(first.source_manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == "reconstruction_normalized_source"
    assert manifest["geometry"]["authority"] == "surface_only"
    assert manifest["geometry"]["interior_semantics"] == "unknown"
    assert manifest["geometry"]["solid_authority"] == "none"
    assert (
        manifest["geometry"]["provenance_semantics"] == "component_lineage_only_not_solid_authority"
    )
    component = manifest["geometry"]["components"][0]
    assert component["topology"]["surface_closed"] is True
    assert component["topology"]["interior_semantics"] == "unknown"
    assert "closed_solid_eligible" not in component["topology"]
    assert manifest["source_to_normalized"]["recentered"] is False
    assert manifest["source_to_normalized"]["ground_snapped"] is False
    assert manifest["geometry"]["bounds_min_xyz_m"] == pytest.approx((-0.5, -1.0, -1.5))
    assert manifest["geometry"]["bounds_max_xyz_m"] == pytest.approx((0.5, 1.0, 1.5))
    assert manifest["normalized_artifact"]["sha256"] == first.normalized_usdc_sha256
    serialized = first.source_manifest.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized


def test_textured_glb_preserves_preview_uv_and_pbr_material_across_surface_split(
    tmp_path: Path,
) -> None:
    vertices = np.asarray(
        (
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    uv = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (2, 2), (192, 96, 48)),
        metallicRoughnessTexture=Image.new("RGB", (2, 2), (12, 128, 220)),
        metallicFactor=0.15,
        roughnessFactor=0.65,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    source = tmp_path / "textured.glb"
    source.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))

    scene = load_normalized_source(SourceInput(source, front_axis="+Z"))

    assert len(scene.components) == 2
    for component in scene.components:
        assert component.render_visual is not None
        assert component.render_visual.kind == "texture"
        assert np.asarray(component.render_visual.uv).shape == (3, 2)
        assert component.render_visual.material.baseColorTexture is not None
        assert component.render_visual.material.metallicRoughnessTexture is not None
        assert component.colors_rgba is None

    artifacts = write_normalized_source(scene, tmp_path / "normalized-textured")
    manifest = json.loads(artifacts.source_manifest.read_text(encoding="utf-8"))
    assert {component["render_appearance"] for component in manifest["geometry"]["components"]} == {
        "native_texture_or_material"
    }


def test_usdz_package_members_are_dependencies_through_the_package_itself(tmp_path: Path) -> None:
    import zipfile

    from pxr import Sdf, UsdShade, UsdUtils

    from usdcraft.reconstruction.source import SourceNormalizationError, _package_member_path

    layer = tmp_path / "toast.usdc"
    texture = tmp_path / "toast_tex.png"
    Image.new("RGB", (8, 8), (200, 120, 60)).save(texture)
    stage = Usd.Stage.CreateNew(str(layer))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Toast")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = _define_mesh(stage, "/Toast/Body", trimesh.creation.box(extents=(0.3, 0.2, 0.25)))
    material = UsdShade.Material.Define(stage, "/Toast/Mat")
    shader = UsdShade.Shader.Define(stage, "/Toast/Mat/Tex")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("toast_tex.png"))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    package = tmp_path / "toast.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer)), str(package))
    with zipfile.ZipFile(package) as archive:
        assert any(name.endswith("toast_tex.png") for name in archive.namelist())

    scene = load_normalized_source(SourceInput(package))

    # The package is one self-contained file: its member texture is reached only through it, so
    # the closure is the package alone and no package-relative path is treated as a file path.
    assert [dependency.path for dependency in scene.source_dependencies] == [package.resolve()]
    assert scene.components

    with pytest.raises(SourceNormalizationError, match="missing"):
        _package_member_path(f"{package}[0/does_not_exist.png]")
    assert _package_member_path(str(layer)) is None


def test_surface_split_shares_atlas_and_preserves_independent_uvs(tmp_path, monkeypatch):
    # A disconnected textured source must not copy one full atlas for every island.
    from usdcraft.reconstruction.orientation import _native_render_visual
    from usdcraft.reconstruction.source import _split_mesh_components

    mesh = trimesh.util.concatenate(
        [
            trimesh.creation.box(),
            trimesh.creation.box(transform=trimesh.transformations.translation_matrix([3, 0, 0])),
        ]
    )
    uv = np.arange(len(mesh.vertices) * 2, dtype=float).reshape(-1, 2) / 100
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGBA", (16, 16), "red")
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    def forbid_copy(*args, **kwargs):
        raise AssertionError("Source texture material must not be deep-copied")

    monkeypatch.setattr(material, "copy", forbid_copy)
    components = _split_mesh_components(
        provenance="test",
        vertices=mesh.vertices,
        faces=mesh.faces,
        normals=None,
        colors=None,
        render_visual=mesh.visual,
    )
    assert len(components) == 2
    for component in components:
        visual = component.render_visual
        assert visual.material is material
        indices = [
            np.flatnonzero(np.all(mesh.vertices == vertex, axis=1))[0]
            for vertex in component.vertices
        ]
        np.testing.assert_array_equal(visual.uv, uv[indices])
        carrier_visual = _native_render_visual(component, len(component.vertices))
        assert carrier_visual.material is material
        np.testing.assert_array_equal(carrier_visual.uv, visual.uv)
        carrier_visual.uv[0] = [0.75, 0.25]
        np.testing.assert_array_equal(visual.uv, uv[indices])
    np.testing.assert_array_equal(mesh.visual.uv, uv)
