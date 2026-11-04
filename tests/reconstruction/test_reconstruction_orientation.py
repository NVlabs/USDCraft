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

"""Strict render session and normalized-source carrier shared by Reconstruction renders."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from usdcraft.reconstruction import orientation as orientation_module
from usdcraft.reconstruction.orientation import (
    OrientationCarrierError,
    OrientationRenderContractError,
    _ChromiumChannelLadder,
    _normalized_identity,
    _write_carrier,
)
from usdcraft.reconstruction.source import (
    MeshTopology,
    NormalizedSourceArtifacts,
    NormalizedSourceComponent,
    NormalizedSourceScene,
    ResolvedSourceFrame,
    SourceDependency,
)
from usdcraft.render.browser import DEFAULT_PROFILE, RenderRequest, RenderResult
from usdcraft.render.contract import SELF_CHECK_RENDER_SIZE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _topology(*, points: bool = False) -> MeshTopology:
    return MeshTopology(
        classification="point_samples" if points else "open_surface",
        vertex_count=4,
        triangle_count=0 if points else 2,
        exact_welded_vertex_count=4,
        boundary_edge_count=0 if points else 4,
        non_manifold_edge_count=0,
        inconsistent_winding_edge_count=0,
        degenerate_triangle_count=0,
        duplicate_triangle_count=0,
        oriented_surface_integral_m3=0.0,
        surface_closed=False,
    )


def _scene(tmp_path: Path, *, points: bool = False) -> NormalizedSourceScene:
    vertices = np.asarray(
        [
            [-0.5, -0.25, 0.0],
            [0.5, -0.25, 0.0],
            [0.5, 0.25, 1.0],
            [-0.5, 0.25, 1.0],
        ],
        dtype=np.float64,
    )
    faces = None if points else np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    component = NormalizedSourceComponent(
        component_id="source component",
        kind="points" if points else "mesh",
        provenance="/Source/Geometry",
        vertices=vertices,
        faces=faces,
        normals=None,
        colors_rgba=np.asarray(
            [
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [255, 255, 255, 255],
            ],
            dtype=np.uint8,
        ),
        topology=_topology(points=points),
    )
    frame = ResolvedSourceFrame(
        up_axis="+Z",
        up_axis_source="embedded_metadata",
        meters_per_unit=1.0,
        meters_per_unit_source="embedded_metadata",
        front_axis="-Y",
        front_axis_source="explicit",
        source_to_normalized=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    source_path = tmp_path / "private source with spaces.usdc"
    source_path.write_bytes(b"source must not appear in the carrier")
    return NormalizedSourceScene(
        source_path=source_path,
        source_format="usd",
        frame=frame,
        components=(component,),
        bounds_min_xyz=vertices.min(axis=0),
        bounds_max_xyz=vertices.max(axis=0),
        source_dependencies=(
            SourceDependency(
                path=source_path,
                sha256=_sha256(source_path),
                byte_count=source_path.stat().st_size,
            ),
        ),
        warnings=(),
    )


def _normalized_artifacts(tmp_path: Path) -> NormalizedSourceArtifacts:
    root = tmp_path / "normalized"
    root.mkdir()
    normalized = root / "normalized.usdc"
    manifest = root / "source_manifest.json"
    normalized.write_bytes(b"normalized-usdc")
    manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
    return NormalizedSourceArtifacts(
        root=root,
        normalized_usdc=normalized,
        source_manifest=manifest,
        normalized_usdc_sha256=_sha256(normalized),
        source_manifest_sha256=_sha256(manifest),
    )


def test_carrier_preserves_native_texture_material_without_urdf_override(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    component = scene.components[0]
    render_visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (2, 2), (156, 84, 42)),
            metallicRoughnessTexture=Image.new("RGB", (2, 2), (8, 128, 224)),
            metallicFactor=0.2,
            roughnessFactor=0.7,
        ),
    )
    scene = replace(
        scene,
        components=(replace(component, colors_rgba=None, render_visual=render_visual),),
    )

    carrier_urdf, carrier_glb, stats = _write_carrier(scene, tmp_path / "carrier-textured")

    assert "<material" not in carrier_urdf.read_text(encoding="utf-8")
    assert str(scene.source_path) not in carrier_urdf.read_text(encoding="utf-8")
    carrier = trimesh.load(carrier_glb, force="scene", process=False)
    visual = next(iter(carrier.geometry.values())).visual
    assert visual.kind == "texture"
    assert visual.material.baseColorTexture is not None
    assert visual.material.metallicRoughnessTexture is not None
    assert stats["native_material_components"] == 1
    assert stats["native_texture_material_components"] == 1
    assert stats["vertex_color_components"] == 0
    assert stats["neutral_fallback_components"] == 0


def test_carrier_uses_neutral_glb_material_only_when_appearance_is_missing(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    component = scene.components[0]
    scene = replace(scene, components=(replace(component, colors_rgba=None),))

    carrier_urdf, carrier_glb, stats = _write_carrier(scene, tmp_path / "carrier-neutral")

    assert "<material" not in carrier_urdf.read_text(encoding="utf-8")
    carrier = trimesh.load(carrier_glb, force="scene", process=False)
    visual = next(iter(carrier.geometry.values())).visual
    assert visual.kind == "texture"
    assert visual.material.name == "recon_source_neutral"
    assert visual.material.baseColorTexture is None
    assert stats["neutral_fallback_components"] == 1


def test_carrier_keeps_vertex_colors_and_point_only_sources(tmp_path: Path) -> None:
    _urdf, colored_glb, colored = _write_carrier(_scene(tmp_path), tmp_path / "carrier-colors")
    assert colored["vertex_color_components"] == 1
    assert trimesh.load(colored_glb, force="scene", process=False).geometry

    (tmp_path / "points").mkdir()
    _urdf, point_glb, points = _write_carrier(
        _scene(tmp_path / "points", points=True), tmp_path / "carrier-points"
    )
    carrier = trimesh.load(point_glb, force="scene", process=False)
    assert isinstance(carrier, trimesh.Scene)
    assert len(carrier.geometry) == 2
    assert points["point_count"] == 4
    assert points["point_visual_coverage_sentinels"] == 1


def test_normalized_artifact_hash_mismatch_is_a_typed_carrier_error(tmp_path: Path) -> None:
    normalized = _normalized_artifacts(tmp_path)
    identity = _normalized_identity(normalized)
    assert identity == {
        "normalized_usdc_sha256": normalized.normalized_usdc_sha256,
        "source_manifest_sha256": normalized.source_manifest_sha256,
    }
    normalized.normalized_usdc.write_bytes(b"changed-after-manifest")
    with pytest.raises(OrientationCarrierError, match="hash does not match"):
        _normalized_identity(normalized)


def _request(tmp_path: Path) -> RenderRequest:
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    return RenderRequest(
        asset_dir=asset_dir,
        out_dir=out_dir,
        views=(("mesh_render_045", 45.0, 35.0),),
        size=SELF_CHECK_RENDER_SIZE,
        profile=DEFAULT_PROFILE,
        model_hash="model-hash",
    )


def test_strict_session_ladder_tries_bundled_chromium_then_installed_chrome_without_numpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channels: list[str] = []
    closed: list[str] = []

    class FakeViewerSession:
        def __init__(self, *, dist_dir: Path, browser_channel: str = "chromium"):
            self.dist_dir = dist_dir
            self.channel = browser_channel
            channels.append(browser_channel)

        def render(
            self,
            request: RenderRequest,
            *,
            allow_numpy_fallback: bool = True,
        ) -> RenderResult | None:
            assert allow_numpy_fallback is False
            if self.channel == "chromium":
                return None
            return RenderResult(
                backend="viewer_chromium",
                capabilities={"geometry": True},
                profile=request.profile,
                images=[],
                manifest_path=None,
                diagnostics={},
            )

        def close(self) -> None:
            closed.append(self.channel)

    monkeypatch.setattr(orientation_module, "ViewerRenderSession", FakeViewerSession)
    ladder = _ChromiumChannelLadder(tmp_path / "dist")
    result = ladder.render(_request(tmp_path), allow_numpy_fallback=False)
    ladder.close()

    assert result is not None and result.backend == "viewer_chromium"
    assert channels == ["chromium", "chrome"]
    assert closed == ["chrome", "chromium"]
    # The ladder itself refuses to be asked for a software fallback.
    with pytest.raises(OrientationRenderContractError, match="fallback disabled"):
        _ChromiumChannelLadder(tmp_path / "dist").render(_request(tmp_path / "again"))


def test_ladder_returns_none_when_every_channel_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableSession:
        def __init__(self, *, dist_dir: Path, browser_channel: str = "chromium"):
            self.channel = browser_channel

        def render(self, request: RenderRequest, *, allow_numpy_fallback: bool = True) -> None:
            return None

        def close(self) -> None:
            pass

    monkeypatch.setattr(orientation_module, "ViewerRenderSession", UnavailableSession)
    ladder = _ChromiumChannelLadder(tmp_path / "dist")
    assert ladder.render(_request(tmp_path), allow_numpy_fallback=False) is None
    ladder.close()
