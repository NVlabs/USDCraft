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

"""CadQuery authoring extension for the SDK."""

from .cadquery import (
    CadQueryMeshExport,
    cadquery_local_aabb,
    export_cadquery_components,
    export_cadquery_mesh,
    mesh_components_from_cadquery,
    mesh_from_cadquery,
    tessellate_cadquery,
)

__all__ = [
    "CadQueryMeshExport",
    "cadquery_local_aabb",
    "export_cadquery_components",
    "export_cadquery_mesh",
    "mesh_components_from_cadquery",
    "mesh_from_cadquery",
    "tessellate_cadquery",
]


def __getattr__(name: str):
    if name == "save_cadquery_obj":
        from .cadquery import save_cadquery_obj

        globals()[name] = save_cadquery_obj
        return save_cadquery_obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
