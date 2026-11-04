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

import cadquery as cq

from sdk._core.assets import AssetContext, AssetSession, activate_asset_session
from sdk._extensions.cadquery import cadquery as cadquery_module


def test_export_cadquery_mesh_allocates_suffix_for_managed_name_conflicts(tmp_path) -> None:
    assets = AssetContext(tmp_path)
    session = AssetSession(tmp_path)
    with activate_asset_session(session):
        export_a = cadquery_module.export_cadquery_mesh(
            cq.Workplane("XY").box(1, 2, 3),
            "shared_name",
            assets=assets,
        )
        export_b = cadquery_module.export_cadquery_mesh(
            cq.Workplane("XY").box(2, 2, 3),
            "shared_name",
            assets=assets,
        )

    assert export_a.mesh.filename == "assets/meshes/shared_name.obj"
    assert export_b.mesh.filename is not None
    assert export_b.mesh.filename.startswith("assets/meshes/shared_name--")
    assert export_b.mesh.filename.endswith(".obj")
    assert export_b.mesh.materialized_path is not None
    assert Path(str(export_b.mesh.materialized_path)).exists()
