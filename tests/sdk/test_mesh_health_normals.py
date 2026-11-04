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

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom

import sdk._core.exact_collisions as exact_collisions
from sdk._core._mesh.booleans import boolean_union, cut_opening_on_face
from sdk._core._mesh.health import mesh_health, require_mesh_health
from sdk._core._mesh.normals import crease_aware_face_varying_normals
from sdk._core._mesh.primitives import (
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    DomeGeometry,
    MeshGeometry,
    RoundedBoxGeometry,
    SphereGeometry,
    TorusGeometry,
)
from sdk._core._urdf_export import compile_object_to_urdf_xml
from sdk._core._usd_export import export_object_to_usd
from sdk._core.articulated_object import ArticulatedObject
from sdk._core.errors import ValidationError
from sdk._core.geometry_scale import scale_geometry_to_size
from sdk._core.mesh import mesh_from_geometry
from sdk._core.types import Mesh
from sdk._core.viewer_assets import load_obj_triangles


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vertices": [(0.0, 0.0, float("nan"))], "faces": []},
        {
            "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "faces": [(0, 1, 3)],
        },
        {
            "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "faces": [(0, 1, 1)],
        },
        {
            "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "faces": [(0, 1, 2.0)],
        },
        {
            "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "faces": [(0, 1, 2)],
            "uvs": [(0.0, 0.0)],
        },
    ],
)
def test_mesh_geometry_rejects_malformed_arrays(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MeshGeometry(**kwargs)  # type: ignore[arg-type]


def test_mesh_geometry_validation_catches_invalid_direct_mutation() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    geometry.vertices[0] = (float("inf"), 0.0, 0.0)

    with pytest.raises(ValidationError, match="finite"):
        geometry.validate()
    with pytest.raises(ValidationError, match="finite"):
        geometry.to_obj()

    with pytest.raises(ValidationError, match="finite"):
        from sdk._core._mesh.booleans import _manifold_from_geometry

        _manifold_from_geometry(geometry)


def test_mesh_geometry_validation_normalizes_valid_direct_mutation() -> None:
    geometry = MeshGeometry(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )
    geometry.vertices[0] = ("0", "0", "0")  # type: ignore[list-item]
    geometry.uvs[0] = ("0", "0")  # type: ignore[index,list-item]
    geometry.normal_crease_deg = "45"  # type: ignore[assignment]

    geometry.validate()

    assert geometry.vertices[0] == (0.0, 0.0, 0.0)
    assert geometry.uvs is not None and geometry.uvs[0] == (0.0, 0.0)
    assert geometry.normal_crease_deg == 45.0
    assert "# usdcraft_normal_crease_deg 45" in geometry.to_obj()


def test_mesh_health_does_not_normalize_or_replace_mutable_source_arrays() -> None:
    geometry = MeshGeometry(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
    )
    vertices = geometry.vertices
    geometry.vertices[0] = ("0", "0", "0")  # type: ignore[list-item]

    mesh_health(geometry)

    assert geometry.vertices is vertices
    assert geometry.vertices[0] == ("0", "0", "0")


@pytest.mark.parametrize(
    "geometry",
    [
        BoxGeometry((1.0, 2.0, 3.0)),
        CylinderGeometry(1.0, 2.0),
        ConeGeometry(1.0, 2.0),
        SphereGeometry(1.0),
        DomeGeometry(1.0),
        CapsuleGeometry(1.0, 2.0),
        CapsuleGeometry(1.0, 0.0),
        TorusGeometry(2.0, 0.5),
    ],
)
def test_closed_builtin_mesh_primitives_are_clean_outward_solids(
    geometry: MeshGeometry,
) -> None:
    report = mesh_health(geometry)

    assert report.clean
    assert report.healthy
    assert report.watertight
    assert report.winding_consistent
    assert report.component_count == 1
    assert report.surface_component_count == 1
    assert report.signed_volume > 0.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BoxGeometry((float("nan"), 1.0, 1.0)),
        lambda: BoxGeometry((1.0, 0.0, 1.0)),
        lambda: CylinderGeometry(0.0, 1.0),
        lambda: ConeGeometry(1.0, float("inf")),
        lambda: SphereGeometry(-1.0),
        lambda: RoundedBoxGeometry((1.0, float("nan"), 1.0), 0.1),
        lambda: RoundedBoxGeometry((1.0, 1.0, 1.0), float("nan")),
        lambda: RoundedBoxGeometry((1.0, 1.0, 1.0), -0.1),
        lambda: CapsuleGeometry(1.0, float("nan")),
        lambda: TorusGeometry(0.1, 0.1),
        lambda: TorusGeometry(0.1, 0.2),
    ],
)
def test_builtin_mesh_primitives_reject_non_finite_or_non_solid_dimensions(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_mesh_health_distinguishes_open_diagnostics_from_structural_errors() -> None:
    open_surface = MeshGeometry(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
    )
    open_report = open_surface.health()

    assert open_report.healthy
    assert not open_report.clean
    assert not open_report.watertight
    assert open_report.component_count == 0
    assert open_report.surface_component_count == 1
    assert "boundary_edges" in open_report.issue_codes

    degenerate = MeshGeometry(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        faces=[(0, 1, 2)],
    )
    degenerate_report = mesh_health(degenerate)
    assert not degenerate_report.healthy
    finding = next(
        finding for finding in degenerate_report.findings if finding.issue == "degenerate_faces"
    )
    assert finding.severity == "error"
    assert finding.count == 1
    assert finding.sample_indices == (0,)
    assert finding.bounds == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))


def test_mesh_health_and_normals_keep_vertex_bow_tie_fans_separate() -> None:
    geometry = MeshGeometry(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
        ],
        faces=[(0, 1, 2), (0, 3, 4)],
    )

    report = mesh_health(geometry)
    normals = crease_aware_face_varying_normals(
        np.asarray(geometry.vertices),
        np.asarray(geometry.faces),
        crease_angle_degrees=180.0,
    ).reshape(-1, 3, 3)

    assert not report.healthy
    assert "nonmanifold_vertices" in report.issue_codes
    assert float(np.dot(normals[0, 0], normals[1, 0])) == pytest.approx(0.0, abs=1e-6)


def test_normal_generation_refuses_pathological_vertex_valence() -> None:
    count = 4097
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    vertices = np.vstack(
        (
            np.zeros((1, 3), dtype=float),
            np.column_stack((np.cos(angles), np.sin(angles), np.zeros(count))),
        )
    )
    faces = np.asarray(
        [(0, index + 1, ((index + 1) % count) + 1) for index in range(count)],
        dtype=np.int64,
    )

    with pytest.raises(ValueError, match="normal smoothing limit"):
        crease_aware_face_varying_normals(vertices, faces)


def test_normal_generation_bounds_nonmanifold_edge_adjacency() -> None:
    count = 4097
    vertices = np.vstack(
        (
            np.zeros((1, 3), dtype=float),
            np.ones((1, 3), dtype=float),
            np.column_stack(
                (
                    np.full(count, 0.5),
                    np.arange(1, count + 1, dtype=float),
                    np.zeros(count),
                )
            ),
        )
    )
    faces = np.asarray([(0, 1, index + 2) for index in range(count)], dtype=np.int64)

    with pytest.raises(ValueError, match="normal smoothing limit"):
        crease_aware_face_varying_normals(vertices, faces)


def test_mesh_health_counts_cavity_shell_as_one_solid() -> None:
    outer = BoxGeometry((2.0, 2.0, 2.0))
    cavity = BoxGeometry((1.0, 1.0, 1.0))
    cavity.faces = [(a, c, b) for a, b, c in cavity.faces]
    outer.merge(cavity)

    report = require_mesh_health(
        outer,
        require_watertight=True,
        require_single_solid=True,
    )

    assert report.component_count == 1
    assert report.surface_component_count == 2
    assert report.signed_volume == pytest.approx(7.0)
    assert "multiple_components" not in report.issue_codes


def test_mesh_health_classifies_a_valid_micron_scale_solid() -> None:
    report = mesh_health(BoxGeometry((1e-6, 1e-6, 1e-6)))

    assert report.healthy
    assert report.watertight
    assert report.component_count == 1
    assert report.signed_volume == pytest.approx(1e-18)


def test_mesh_health_volume_is_stable_far_from_the_origin() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0)).translate(1e9, -1e9, 1e9)

    report = mesh_health(geometry)

    assert report.healthy
    assert report.component_count == 1
    assert report.signed_volume == pytest.approx(1.0)


def test_mesh_health_reports_multiple_positive_volume_solids_without_global_failure() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    geometry.merge(BoxGeometry((1.0, 1.0, 1.0)).translate(2.0, 0.0, 0.0))

    report = mesh_health(geometry)

    assert report.healthy
    assert report.watertight
    assert report.component_count == 2
    assert report.surface_component_count == 2
    assert "multiple_components" in report.issue_codes
    with pytest.raises(ValidationError, match="component_count=2"):
        require_mesh_health(geometry, require_single_solid=True)


def test_mesh_health_rejects_external_inverted_shell_as_fake_cavity() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    inverted = BoxGeometry((0.25, 0.25, 0.25)).translate(2.0, 0.0, 0.0)
    inverted.faces = [(a, c, b) for a, b, c in inverted.faces]
    geometry.merge(inverted)

    report = mesh_health(geometry)

    assert report.component_count == 1
    assert report.surface_component_count == 2
    assert not report.healthy
    assert "inward_orientation" in report.issue_codes
    with pytest.raises(ValidationError, match="inward_orientation"):
        require_mesh_health(geometry, require_single_solid=True)


def test_mesh_health_rejects_non_alternating_nested_shell_orientation() -> None:
    geometry = BoxGeometry((3.0, 3.0, 3.0))
    cavity = BoxGeometry((2.0, 2.0, 2.0))
    cavity.faces = [(a, c, b) for a, b, c in cavity.faces]
    nested_wrong = BoxGeometry((1.0, 1.0, 1.0))
    nested_wrong.faces = [(a, c, b) for a, b, c in nested_wrong.faces]
    geometry.merge(cavity).merge(nested_wrong)

    report = mesh_health(geometry)

    assert not report.healthy
    assert report.component_count == 1
    assert "inward_orientation" in report.issue_codes
    with pytest.raises(ValidationError, match="inward_orientation"):
        require_mesh_health(geometry, require_single_solid=True)


def test_normal_crease_validates_and_survives_managed_mesh_and_resize(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="between 0 and 180"):
        Mesh(filename="assets/meshes/example.obj", normal_crease_deg=181.0)
    with pytest.raises(ValidationError, match="must be numeric"):
        Mesh(filename="assets/meshes/example.obj", normal_crease_deg="hard")  # type: ignore[arg-type]

    geometry = BoxGeometry((1.0, 1.0, 1.0)).use_hard_normals()
    copied = geometry.copy()
    mesh = mesh_from_geometry(copied, tmp_path / "assets" / "meshes" / "box.obj")

    assert copied.normal_crease_deg == 0.0
    assert mesh.normal_crease_deg == 0.0
    assert "# usdcraft_normal_crease_deg 0" in Path(str(mesh.materialized_path)).read_text()

    resized = scale_geometry_to_size(
        mesh,
        (2.0, 3.0, 4.0),
        asset_root=tmp_path,
    )
    assert isinstance(resized, Mesh)
    assert resized.name == mesh.name
    assert resized.materialized_path == mesh.materialized_path
    assert resized.normal_crease_deg == 0.0


def test_merge_resets_conflicting_normal_crease_to_default() -> None:
    hard = BoxGeometry((1.0, 1.0, 1.0)).use_hard_normals()
    smooth = BoxGeometry((1.0, 1.0, 1.0)).translate(2.0, 0.0, 0.0).use_smooth_normals()

    hard.merge(smooth)

    assert hard.normal_crease_deg is None


def test_merge_and_union_canonicalize_equivalent_default_normal_policies() -> None:
    explicit = BoxGeometry((1.0, 1.0, 1.0)).set_normal_crease_deg(60.0)
    implicit = BoxGeometry((1.0, 1.0, 1.0)).translate(0.5, 0.0, 0.0)

    forward_merge = explicit.copy().merge(implicit)
    reverse_merge = implicit.copy().merge(explicit)
    forward_union = boolean_union(explicit, implicit)
    reverse_union = boolean_union(implicit, explicit)

    assert forward_merge.normal_crease_deg is None
    assert reverse_merge.normal_crease_deg is None
    assert forward_union.normal_crease_deg is None
    assert reverse_union.normal_crease_deg is None


def test_obj_materialization_preserves_aligned_uv_face_indices(tmp_path: Path) -> None:
    geometry = MeshGeometry(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )
    mesh = mesh_from_geometry(geometry, tmp_path / "assets" / "meshes" / "uv.obj")
    obj = Path(str(mesh.materialized_path)).read_text()

    assert "vt 1.000000 0.000000" in obj
    assert "f 1/1 2/2 3/3" in obj


def test_obj_serialization_increases_precision_to_preserve_small_valid_face() -> None:
    geometry = MeshGeometry(
        vertices=[(0.0, 0.0, 0.0), (4e-7, 0.0, 0.0), (0.0, 4e-7, 0.0)],
        faces=[(0, 1, 2)],
    )

    vertex_lines = [line for line in geometry.to_obj().splitlines() if line.startswith("v ")]
    serialized = np.asarray(
        [[float(value) for value in line.split()[1:]] for line in vertex_lines],
        dtype=float,
    )

    assert (
        np.linalg.norm(np.cross(serialized[1] - serialized[0], serialized[2] - serialized[0])) > 0
    )


def test_obj_serialization_rejects_coordinate_degenerate_face() -> None:
    geometry = MeshGeometry(
        vertices=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
    )

    with pytest.raises(ValidationError, match="coordinate-degenerate"):
        geometry.to_obj()


def test_merge_validates_source_without_mutating_its_public_arrays() -> None:
    receiver = BoxGeometry((1.0, 1.0, 1.0))
    source = BoxGeometry((0.5, 0.5, 0.5)).translate(2.0, 0.0, 0.0)
    source_vertices = source.vertices
    source_faces = source.faces

    receiver.merge(source)

    assert source.vertices is source_vertices
    assert source.faces is source_faces


@pytest.mark.parametrize("face", ("+x", "-x", "+y", "-y", "+z", "-z"))
def test_cut_opening_on_every_face_preserves_host_normal_policy(face: str) -> None:
    shell = BoxGeometry((1.0, 1.0, 1.0)).use_hard_normals()
    original_face_count = len(shell.faces)

    result = cut_opening_on_face(
        shell,
        face=face,
        opening_profile=[(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)],
        depth=0.1,
    )

    assert result is shell
    assert result.normal_crease_deg == 0.0
    for triangle in result.faces[original_face_count:]:
        points = np.asarray([result.vertices[index] for index in triangle], dtype=float)
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        centroid = points.mean(axis=0)
        depth_axis = {"x": 0, "y": 1, "z": 2}[face[1]]
        centerline = np.zeros(3, dtype=float)
        centerline[depth_axis] = centroid[depth_axis]
        assert float(np.dot(normal, centerline - centroid)) > 0.0


def test_normal_crease_changes_exact_collision_fingerprint_payload() -> None:
    hard = Mesh(filename="assets/meshes/example.obj", normal_crease_deg=0.0)
    smooth = Mesh(filename="assets/meshes/example.obj", normal_crease_deg=180.0)

    assert exact_collisions._serialize_geometry(hard) != exact_collisions._serialize_geometry(
        smooth
    )


def _bent_open_geometry(*, crease: float) -> MeshGeometry:
    return MeshGeometry(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        faces=[(0, 1, 2), (1, 0, 3)],
        normal_crease_deg=crease,
    )


def _shared_edge_corner_normals(
    points: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[np.ndarray] = []
    for face_index, face in enumerate(faces):
        corner = int(np.flatnonzero(face == 0)[0])
        selected.append(normals[face_index, corner])
    assert len(selected) == 2
    return selected[0], selected[1]


def test_managed_normal_crease_is_honored_by_viewer_obj_loader(tmp_path: Path) -> None:
    hard_mesh = mesh_from_geometry(
        _bent_open_geometry(crease=0.0),
        tmp_path / "assets" / "meshes" / "hard.obj",
    )
    smooth_mesh = mesh_from_geometry(
        _bent_open_geometry(crease=180.0),
        tmp_path / "assets" / "meshes" / "smooth.obj",
    )

    for mesh, expected_dot in ((hard_mesh, 0.0), (smooth_mesh, 1.0)):
        vertices, faces, normals, normal_indices = load_obj_triangles(
            Path(str(mesh.materialized_path))
        )
        assert normals is not None
        assert normal_indices is not None
        first, second = _shared_edge_corner_normals(
            vertices,
            faces,
            normals[normal_indices],
        )
        assert float(np.dot(first, second)) == pytest.approx(expected_dot, abs=1e-6)

        expected = crease_aware_face_varying_normals(
            vertices,
            faces,
            crease_angle_degrees=float(mesh.normal_crease_deg),
        ).reshape(-1, 3, 3)
        np.testing.assert_allclose(normals[normal_indices], expected, atol=1e-6)


def test_managed_normal_crease_reaches_usd_face_varying_normals(tmp_path: Path) -> None:
    hard_mesh = mesh_from_geometry(
        _bent_open_geometry(crease=0.0),
        tmp_path / "assets" / "meshes" / "hard.obj",
    )
    smooth_mesh = mesh_from_geometry(
        _bent_open_geometry(crease=180.0),
        tmp_path / "assets" / "meshes" / "smooth.obj",
    )
    model = ArticulatedObject("normal_policy", assets=tmp_path)
    body = model.part("body")
    body.visual(hard_mesh, name="hard", collidable=False)
    body.visual(smooth_mesh, name="smooth", collidable=False)

    urdf_root = ET.fromstring(compile_object_to_urdf_xml(model, include_physical_collisions=False))
    assert [
        element.attrib.get("usdcraft_normal_crease_deg")
        for element in urdf_root.findall(".//visual/geometry/mesh")
    ] == ["0", "180"]

    stage = Usd.Stage.Open(export_object_to_usd(model, tmp_path / "out").usdc_path.as_posix())
    dots: dict[str, float] = {}
    for name in ("hard", "smooth"):
        root = stage.GetPrimAtPath(f"/Asset/Links/body/Visuals/{name}/Geom")
        mesh_prims = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
        assert len(mesh_prims) == 1
        usd_mesh = UsdGeom.Mesh(mesh_prims[0])
        points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=float)
        faces = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        normals = np.asarray(usd_mesh.GetNormalsAttr().Get(), dtype=float).reshape(-1, 3, 3)
        first, second = _shared_edge_corner_normals(points, faces, normals)
        dots[name] = float(np.dot(first, second))

    assert dots["hard"] == pytest.approx(0.0, abs=1e-6)
    assert dots["smooth"] == pytest.approx(1.0, abs=1e-6)
