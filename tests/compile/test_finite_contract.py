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

from pathlib import Path

import pytest

from usdcraft.compile.command import compile_model

_MODEL_TEMPLATE = """from __future__ import annotations

from sdk import (
    ArticulatedObject,
    ArticulationType,
    BaseMode,
    Box,
    Material,
    Mimic,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("finite_host_gate", base_mode=BaseMode.FLOATING)
    paint = model.material("paint", rgba=(0.1, 0.2, 0.3, 1.0))
    base = model.part("base")
    base_visual = base.visual(
        Box((0.2, 0.2, 0.2)),
        origin=Origin(),
        material=paint,
        collidable=False,
    )
    middle = model.part("middle")
    middle.visual(Box((0.1, 0.1, 0.1)))
    tip = model.part("tip")
    tip.visual(Box((0.1, 0.1, 0.1)))
    model.articulation(
        "driver",
        ArticulationType.REVOLUTE,
        base,
        middle,
        motion_limits=MotionLimits(effort=2.0, velocity=1.0, lower=-0.5, upper=0.5),
    )
    follower = model.articulation(
        "follower",
        ArticulationType.REVOLUTE,
        middle,
        tip,
        motion_limits=MotionLimits(effort=2.0, velocity=1.0, lower=-0.5, upper=0.5),
        mimic=Mimic("driver"),
    )
    {mutation}
    return model


def run_tests() -> TestReport:
    return TestContext(object_model).report()


object_model = build_object_model()
"""


@pytest.mark.parametrize(
    "mutation",
    (
        'object.__setattr__(base_visual.geometry, "size", (float("nan"), 0.2, 0.2))',
        'object.__setattr__(base_visual.origin, "xyz", (float("nan"), 0.0, 0.0))',
        'paint.rgba = (float("nan"), 0.2, 0.3, 1.0)',
        'object.__setattr__(follower.mimic, "multiplier", float("nan"))',
    ),
    ids=("box", "visual-origin", "material", "mimic"),
)
def test_final_compile_rejects_mutated_nonfinite_serialized_values(
    tmp_path: Path,
    mutation: str,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text(_MODEL_TEMPLATE.format(mutation=mutation), encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be finite"):
        compile_model(model_path, tmp_path / "build", render=False)
