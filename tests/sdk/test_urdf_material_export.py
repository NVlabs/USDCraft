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

from sdk import ArticulatedObject, Box, Inertial
from sdk._core._urdf_export import compile_object_to_urdf_xml


def _single_part_model(**material_kwargs) -> ArticulatedObject:
    model = ArticulatedObject(name="material_export")
    material = model.material("finish", **material_kwargs)
    part = model.part("root")
    part.visual(Box((0.1, 0.1, 0.1)), material=material)
    part.inertial = Inertial.from_geometry(Box((0.1, 0.1, 0.1)), mass=1.0)
    return model


def test_pbr_material_exports_vendor_element() -> None:
    model = _single_part_model(
        rgba=(0.8, 0.2, 0.2, 1.0),
        metallic=0.9,
        roughness=0.15,
        emissive=(0.1, 0.0, 0.0),
        clearcoat=0.4,
    )

    root = ET.fromstring(compile_object_to_urdf_xml(model))
    pbr = root.find("material/usdcraft_pbr")

    assert pbr is not None
    assert pbr.attrib["metallic"] == "0.9"
    assert pbr.attrib["roughness"] == "0.15"
    assert pbr.attrib["emissive_rgb"] == "0.1 0 0"
    assert pbr.attrib["clearcoat"] == "0.4"
    assert pbr.attrib["clearcoat_roughness"] == "0.2"
    assert "transmission" not in pbr.attrib
    assert root.find("material/mesh2usd_pbr") is None


def test_plain_material_exports_no_pbr_element() -> None:
    model = _single_part_model(rgba=(0.8, 0.2, 0.2, 1.0))

    root = ET.fromstring(compile_object_to_urdf_xml(model))

    assert root.find("material/usdcraft_pbr") is None
    color = root.find("material/color")
    assert color is not None
