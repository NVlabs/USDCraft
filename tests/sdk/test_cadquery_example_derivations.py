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

import re
from pathlib import Path

import pytest

cq = pytest.importorskip("cadquery")

_PYTHON_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_EXAMPLE_PATHS = (
    "sdk/_examples/cadquery/bga_package.md",
    "sdk/_examples/cadquery/parametric_pin_header.md",
    "sdk/_examples/cadquery/rj45_surface_mount_jack.md",
    "sdk/_examples/cadquery/din_rail_clip.md",
    "sdk/_examples/cadquery/mecanum_wheel.md",
    "sdk/_examples/cadquery/raspberry_pi_3_model_b_assembly.md",
    "sdk/_examples/cadquery/pitray_clip.md",
    "sdk/_examples/cadquery/worm_gear.md",
    "sdk/_examples/cadquery/involute_gear.md",
    "sdk/_examples/cadquery/bevel_gears.md",
)

pytestmark = pytest.mark.slow  # every test here executes a CadQuery example document


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _extract_python_fence(path: Path) -> str:
    match = _PYTHON_FENCE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"No Python fence found in {path}")
    return match.group(1)


@pytest.mark.parametrize("relative_path", _EXAMPLE_PATHS)
def test_derived_cadquery_example_executes(relative_path: str) -> None:
    path = _repo_root() / relative_path
    namespace = {"cq": cq, "cadquery": cq}

    exec(_extract_python_fence(path), namespace)

    result = namespace.get("result")
    assert isinstance(result, (cq.Workplane, cq.Shape, cq.Assembly))
