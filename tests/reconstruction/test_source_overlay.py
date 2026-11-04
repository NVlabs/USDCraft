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

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from usdcraft.reconstruction import orientation
from usdcraft.reconstruction.source import (
    MeshTopology,
    NormalizedSourceArtifacts,
    NormalizedSourceComponent,
    NormalizedSourceScene,
    ResolvedSourceFrame,
    SourceDependency,
)
from usdcraft.reconstruction.source_overlay import prepare_source_overlay


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_overlay_exports_without_starting_renderer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(orientation, "_default_bundle_builder", lambda *a: pytest.fail("no build"))
    monkeypatch.setattr(
        orientation, "_default_session_factory", lambda *a: pytest.fail("no browser")
    )
    result = prepare_source_overlay(_scene(tmp_path), _normalized(tmp_path), tmp_path / "overlay")
    assert result.carrier_urdf.is_file() and result.carrier_glb.is_file()
    assert not list(result.root.rglob("*.png"))
    manifest = json.loads(result.manifest.read_text())
    assert manifest["images"] == [] and manifest["render_policy"] == "author_requested"


def _scene(tmp_path: Path) -> NormalizedSourceScene:
    vertices = np.asarray(
        ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 1.0)),
        dtype=np.float64,
    )
    component = NormalizedSourceComponent(
        component_id="component_0000",
        kind="mesh",
        provenance="/Source/Triangle",
        vertices=vertices,
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        normals=None,
        colors_rgba=np.asarray(((255, 0, 0, 255),) * 3, dtype=np.uint8),
        topology=MeshTopology(
            classification="open_surface",
            vertex_count=3,
            triangle_count=1,
            exact_welded_vertex_count=3,
            boundary_edge_count=3,
            non_manifold_edge_count=0,
            inconsistent_winding_edge_count=0,
            degenerate_triangle_count=0,
            duplicate_triangle_count=0,
            oriented_surface_integral_m3=0.0,
            surface_closed=False,
        ),
    )
    source = tmp_path / "private-source.glb"
    source.write_bytes(b"private source")
    frame = ResolvedSourceFrame(
        up_axis="+Y",
        up_axis_source="format_contract",
        meters_per_unit=1.0,
        meters_per_unit_source="format_contract",
        front_axis="-Z",
        front_axis_source="explicit",
        source_to_normalized=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    return NormalizedSourceScene(
        source_path=source,
        source_format="gltf",
        frame=frame,
        components=(component,),
        bounds_min_xyz=vertices.min(axis=0),
        bounds_max_xyz=vertices.max(axis=0),
        source_dependencies=(SourceDependency(source, _sha256(source), source.stat().st_size),),
        warnings=(),
    )


def _normalized(tmp_path: Path) -> NormalizedSourceArtifacts:
    root = tmp_path / "normalized"
    root.mkdir()
    source = root / "normalized.usdc"
    manifest = root / "source_manifest.json"
    source.write_bytes(b"normalized source")
    manifest.write_text("{}\n", encoding="utf-8")
    return NormalizedSourceArtifacts(
        root=root,
        normalized_usdc=source,
        source_manifest=manifest,
        normalized_usdc_sha256=_sha256(source),
        source_manifest_sha256=_sha256(manifest),
    )
