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

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class SdkProfile:
    """The canonical SDK authoring surface exposed to supported asset Author runtimes.

    Prompt selection belongs to the product runtime, not the SDK. Keeping this
    profile product- and provider-agnostic prevents SDK documentation and
    scaffolding from depending on an execution arm.
    """

    package_name: str
    scaffold_path: Path
    docs_full: tuple[Path, ...]


_COMMON_DOCS = (
    Path("sdk/_docs/common/00_quickstart.md"),
    Path("sdk/_docs/common/10_errors.md"),
    Path("sdk/_docs/common/20_core_types.md"),
    Path("sdk/_docs/common/30_articulated_object.md"),
    Path("sdk/_docs/common/40_assets.md"),
    Path("sdk/_docs/common/50_placement.md"),
    Path("sdk/_docs/common/70_probe_tooling.md"),
    Path("sdk/_docs/common/80_testing.md"),
)

_BASE_DOCS = (
    Path("sdk/_docs/base/40_mesh_geometry.md"),
    Path("sdk/_docs/base/41_panels_and_grilles.md"),
    Path("sdk/_docs/base/42_brackets_and_mounts.md"),
    Path("sdk/_docs/base/43_fans_and_rotors.md"),
    Path("sdk/_docs/base/44_knobs_and_controls.md"),
    Path("sdk/_docs/base/45_wires.md"),
    Path("sdk/_docs/base/46_section_lofts.md"),
    Path("sdk/_docs/base/47_bezels_and_frames.md"),
    Path("sdk/_docs/base/48_wheels_and_tires.md"),
    Path("sdk/_docs/base/49_hinges.md"),
    Path("sdk/_docs/base/50_organic_surfaces.md"),
    Path("sdk/_docs/base/51_containers_and_vessels.md"),
    Path("sdk/_docs/base/52_shelving_and_panel_furniture.md"),
    Path("sdk/_docs/base/53_sdf_organic_blending.md"),
    Path("sdk/_docs/base/54_decals_and_labels.md"),
    Path("sdk/_docs/base/55_common_assemblies.md"),
)

_CADQUERY_DOCS = (
    Path("sdk/_docs/cadquery/35_cadquery.md"),
    Path("sdk/_docs/cadquery/36_cadquery_primer.md"),
    Path("sdk/_docs/cadquery/37_cadquery_workplane.md"),
    Path("sdk/_docs/cadquery/38_cadquery_sketch.md"),
    Path("sdk/_docs/cadquery/39_cadquery_assembly.md"),
    Path("sdk/_docs/cadquery/39d_cadquery_gears.md"),
    Path("sdk/_docs/cadquery/39b_cadquery_free_function.md"),
    Path("sdk/_docs/cadquery/39c_cadquery_api_ref.md"),
)

SDK_PROFILES: dict[str, SdkProfile] = {
    "sdk": SdkProfile(
        package_name="sdk",
        scaffold_path=Path("scaffold.py"),
        docs_full=_COMMON_DOCS[:4] + _BASE_DOCS + _CADQUERY_DOCS + _COMMON_DOCS[4:],
    ),
}


SUPPORTED_SDK_PACKAGES = frozenset(SDK_PROFILES)


def get_sdk_profile(package_name: str) -> SdkProfile:
    try:
        return SDK_PROFILES[package_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SDK package: {package_name!r}. "
            f"Expected one of {sorted(SUPPORTED_SDK_PACKAGES)!r}."
        ) from exc
