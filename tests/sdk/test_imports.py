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

import sdk
from sdk._extensions.cadquery import mesh_components_from_cadquery, mesh_from_cadquery


def test_sdk_exports() -> None:
    assert hasattr(sdk, "ArticulatedObject")
    assert hasattr(sdk, "CapsuleGeometry")
    assert hasattr(sdk, "DomeGeometry")
    assert hasattr(sdk, "MeshGeometry")
    assert hasattr(sdk, "MeshHealthReport")
    assert hasattr(sdk, "mesh_health")
    assert hasattr(sdk, "snap_to")
    assert hasattr(sdk, "weld")
    assert hasattr(sdk, "smooth_difference")
    assert hasattr(sdk, "SweepSection")
    assert hasattr(sdk, "SurfaceFrame")
    assert hasattr(sdk, "SurfaceWrapMapping")
    assert hasattr(sdk, "ValidationError")
    assert hasattr(sdk, "TestContext")
    assert hasattr(sdk, "SectionLoftSpec")
    assert hasattr(sdk, "section_loft")
    assert hasattr(sdk, "repair_loft")
    assert hasattr(sdk, "ShellPartitionRegion")
    assert hasattr(sdk, "ShellPartitionSpec")
    assert hasattr(sdk, "partition_shell")
    assert hasattr(sdk, "surface_frame")
    assert hasattr(sdk, "align_centers")
    assert hasattr(sdk, "place_on_surface")
    assert hasattr(sdk, "place_on_face")
    assert hasattr(sdk, "wrap_profile_onto_surface")
    assert hasattr(sdk, "wrap_mesh_onto_surface")
    assert hasattr(sdk, "scale_geometry_to_size")
    assert hasattr(sdk, "SegmentedFlexibleMemberSpec")
    assert hasattr(sdk, "SpatialSegmentJoint")
    assert hasattr(sdk, "PlanarSegmentJoint")
    assert hasattr(sdk, "add_segmented_flexible_member")
    assert hasattr(sdk, "CadQueryMeshExport")
    assert hasattr(sdk, "export_cadquery_components")
    assert hasattr(sdk, "mesh_components_from_cadquery")
    assert hasattr(sdk, "mesh_from_cadquery")
    assert hasattr(sdk, "SpurGear")
    assert hasattr(sdk, "RingGear")
    assert hasattr(sdk, "BevelGearPair")
    assert hasattr(sdk, "Worm")
    assert hasattr(sdk, "gear")
    assert hasattr(sdk, "addGear")
    assert callable(mesh_from_cadquery)
    assert callable(mesh_components_from_cadquery)
    assert not hasattr(sdk, "BufferGeometry")
    assert not hasattr(sdk, "place_on_top")
    assert not hasattr(sdk, "place_in_front_of")
    assert not hasattr(sdk, "align_centers_xy")
