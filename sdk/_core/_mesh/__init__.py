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

from contextlib import suppress

from .assemblies import (
    AFrameGeometry,
    BeltSeatGeometry,
    CasterForkGeometry,
    CasterGeometry,
    FlexibleHoseGeometry,
    GableRoofGeometry,
    HangingChainGeometry,
    LadderGeometry,
    MoldedSeatGeometry,
    PostBeamBracketGeometry,
    SlideChuteGeometry,
)
from .booleans import (
    _boolean_union_many,
    _geometry_from_manifold,
    _manifold_from_geometry,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    cut_opening_on_face,
    mesh_from_geometry,
)
from .brackets import ClevisBracketGeometry, PivotForkGeometry, TrunnionYokeGeometry
from .common import (
    WirePath,
    resample_side_sections,
    rounded_rect_profile,
    sample_arc_3d,
    sample_catmull_rom_spline_2d,
    sample_catmull_rom_spline_3d,
    sample_cubic_bezier_spline_2d,
    sample_cubic_bezier_spline_3d,
    split_superellipse_side_loft,
    superellipse_profile,
    superellipse_side_loft,
)
from .containers import BottleGeometry, CapGeometry, JarGeometry
from .controls import BezelGeometry, KnobGeometry
from .decals import DecalGeometry
from .fans import BlowerWheelGeometry, FanRotorGeometry
from .furniture import (
    BarHandleGeometry,
    BridgeHandleGeometry,
    CabinetCarcassGeometry,
    PanelGeometry,
    TaperedLegGeometry,
)
from .health import MeshHealthFinding, MeshHealthReport, mesh_health
from .hinges import BarrelHingeGeometry, PianoHingeGeometry
from .panels import (
    PerforatedPanelGeometry,
    SlotPatternPanelGeometry,
    VentGrilleGeometry,
)
from .primitives import (
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    DomeGeometry,
    ExtrudeGeometry,
    ExtrudeWithHolesGeometry,
    LatheGeometry,
    LoftGeometry,
    MeshGeometry,
    RoundedBoxGeometry,
    SphereGeometry,
    TorusGeometry,
)
from .sdf import (
    Sdf,
    mesh_from_sdf,
    sdf_box,
    sdf_capsule,
    sdf_cylinder,
    sdf_ellipsoid,
    sdf_sphere,
    sdf_torus,
)
from .shelving import ShelvingUnitGeometry
from .smooth_mesh import SnapRefused, smooth_difference, snap_to, weld
from .smoothing import (
    displace_geometry,
    simplify_geometry,
    smooth_geometry,
    subdivide_geometry,
    taubin_smooth_geometry,
)
from .specs import (
    BezelCutout,
    BezelEdgeFeature,
    BezelFace,
    BezelFlange,
    BezelMounts,
    BezelRecess,
    BezelVisor,
    BoltPattern,
    FanRotorBlade,
    FanRotorHub,
    FanRotorShroud,
    HingeHolePattern,
    HingePinStyle,
    KnobBore,
    KnobGrip,
    KnobIndicator,
    KnobRelief,
    KnobSkirt,
    KnobTopFeature,
    TireCarcass,
    TireGroove,
    TireShoulder,
    TireSidewall,
    TireTread,
    VentGrilleFrame,
    VentGrilleMounts,
    VentGrilleSlats,
    VentGrilleSleeve,
    WheelBore,
    WheelFace,
    WheelFlange,
    WheelHub,
    WheelRim,
    WheelSpokes,
)
from .sweeps import (
    ArcPipeGeometry,
    PipeGeometry,
    SweepGeometry,
    SweepSection,
    WirePolylineGeometry,
    beam_between,
    carve_pocket,
    sweep_profile_along_spline,
    tube_from_spline_points,
    wire_from_points,
)
from .tube_network import tube_network_from_paths
from .wheels import TireGeometry, WheelGeometry

__all__ = [
    "AFrameGeometry",
    "ArcPipeGeometry",
    "BarHandleGeometry",
    "BeltSeatGeometry",
    "BridgeHandleGeometry",
    "beam_between",
    "carve_pocket",
    "RoundedBoxGeometry",
    "BarrelHingeGeometry",
    "BezelCutout",
    "BezelEdgeFeature",
    "BezelFace",
    "BezelFlange",
    "BezelGeometry",
    "BezelMounts",
    "BezelRecess",
    "BezelVisor",
    "BlowerWheelGeometry",
    "BoltPattern",
    "BottleGeometry",
    "BoxGeometry",
    "CabinetCarcassGeometry",
    "CapGeometry",
    "CapsuleGeometry",
    "CasterForkGeometry",
    "CasterGeometry",
    "ClevisBracketGeometry",
    "ConeGeometry",
    "CylinderGeometry",
    "DecalGeometry",
    "DomeGeometry",
    "ExtrudeGeometry",
    "ExtrudeWithHolesGeometry",
    "FanRotorBlade",
    "FanRotorGeometry",
    "FanRotorHub",
    "FanRotorShroud",
    "FlexibleHoseGeometry",
    "GableRoofGeometry",
    "HingeHolePattern",
    "HingePinStyle",
    "HangingChainGeometry",
    "LadderGeometry",
    "JarGeometry",
    "KnobBore",
    "KnobGeometry",
    "KnobGrip",
    "KnobIndicator",
    "KnobRelief",
    "KnobSkirt",
    "KnobTopFeature",
    "LatheGeometry",
    "LoftGeometry",
    "MeshGeometry",
    "MeshHealthFinding",
    "MeshHealthReport",
    "MoldedSeatGeometry",
    "PanelGeometry",
    "PerforatedPanelGeometry",
    "PianoHingeGeometry",
    "PipeGeometry",
    "PostBeamBracketGeometry",
    "PivotForkGeometry",
    "Sdf",
    "ShelvingUnitGeometry",
    "SlideChuteGeometry",
    "SlotPatternPanelGeometry",
    "SphereGeometry",
    "SnapRefused",
    "SweepGeometry",
    "SweepSection",
    "TaperedLegGeometry",
    "TireCarcass",
    "TireGeometry",
    "TireGroove",
    "TireShoulder",
    "TireSidewall",
    "TireTread",
    "TorusGeometry",
    "TrunnionYokeGeometry",
    "VentGrilleFrame",
    "VentGrilleGeometry",
    "VentGrilleMounts",
    "VentGrilleSlats",
    "VentGrilleSleeve",
    "WheelBore",
    "WheelFace",
    "WheelFlange",
    "WheelGeometry",
    "WheelHub",
    "WheelRim",
    "WheelSpokes",
    "WirePath",
    "WirePolylineGeometry",
    "boolean_difference",
    "boolean_intersection",
    "boolean_union",
    "cut_opening_on_face",
    "displace_geometry",
    "mesh_from_geometry",
    "mesh_health",
    "mesh_from_sdf",
    "resample_side_sections",
    "rounded_rect_profile",
    "sample_arc_3d",
    "sdf_box",
    "sdf_capsule",
    "sdf_cylinder",
    "sdf_ellipsoid",
    "sdf_sphere",
    "sdf_torus",
    "sample_catmull_rom_spline_2d",
    "sample_catmull_rom_spline_3d",
    "sample_cubic_bezier_spline_2d",
    "sample_cubic_bezier_spline_3d",
    "simplify_geometry",
    "smooth_geometry",
    "smooth_difference",
    "snap_to",
    "split_superellipse_side_loft",
    "subdivide_geometry",
    "superellipse_profile",
    "superellipse_side_loft",
    "sweep_profile_along_spline",
    "taubin_smooth_geometry",
    "tube_from_spline_points",
    "tube_network_from_paths",
    "wire_from_points",
    "weld",
]

_COMPAT_MODULE = "sdk._core.mesh"
for _name in __all__:
    _obj = globals().get(_name)
    if getattr(_obj, "__module__", "").startswith("sdk._core._mesh"):
        with suppress(Exception):
            _obj.__module__ = _COMPAT_MODULE

del _COMPAT_MODULE, _name, _obj
